"""PPO with a centralised critic (MAPPO: Yu et al., 2021), docs/en/training/training.md.

* Every own unit is an agent with its own ratio new / old probability of its order; all units of
  a side share the side's advantage (one reward per side, one value of the critic that sees
  the whole field). Units that take no orders (dead, routing, padding) are left out of the policy
  loss and the entropy: their log-probability is zero and they are masked.
* The actor's and the critic's gradients are clipped apart, each to max_grad (clipped together, a
  large critic loss's gradient froze the policy: 02.10, docs/en/training/training.md).
* The entropy bonus is on the order kind only. The full entropy counts the move point's 128 bins
  only when the unit moves, so a bonus on it pays the network for moving (and for switching
  points): in a first run the full entropy rose from 3.5 to 5 and order changes from 79 to 98
  a minute in 20 updates, before any battle had ended.
  With an entropy floor (run.py --entropy-target) the bonus's weight follows the kind's entropy
  (entropy_weight): up while it is below the target, back down to the scheduled weight above it.
* GAE over the rollout; the value of the last state bootstraps; a finished battle cuts it.
* The memory (GRU) is trained through time: a minibatch is a set of whole chunks (T decisions of
  some battles); the actor runs its memory through each chunk from the memory stored when the
  chunk began, emptied where a new battle begins (recurrent PPO, as R2D2's stored state).
* The drills' teacher (update's teach=, run.py --drill-teach; tools/nn/train/drills/teach.py): with
  labels in the batch, + weight x the cross-entropy of the policy against the drill's skilled script on
  the labelled units of that drill's battles (over all its units: only a share of its battles may be
  labelled, the adaptive teacher's, run.py --drill-teach auto); its term and the agreement are logged per
  drill.
"""
from dataclasses import dataclass

import torch

from tools.nn.model import eyes as meyes
from tools.nn.model import heads as hd
from tools.nn.train.drills import teach as drill_teach
from tools.nn.train.rollout import full_obs


# The eyes' loss weight: the four heads' squared errors summed x this. On a fresh v2 batch (vs nearest, 3-5 min into
# the battles) the eyes' gradient on the actor at weight 1 was 0.05 of the policy loss's (0.004 vs 0.08); at 5 ~0.25:
# it shapes the shared trunk without outweighing PPO (the eyes' own heads learn at Adam's pace whatever the weight)
# (docs/en/training/model.md "Eyes")
EYES_WEIGHT = 5.0


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.9997      # per decision of 0.5 s: a horizon of ~3300 x 0.5 s (~28 min); a win
    #                            10 minutes away still counts 0.7. gae() takes gamma and lam per decision:
    #                            run.py gives these per-0.5 s defaults ** (decision s / 0.5) (cadence.py;
    #                            at the game's 1 s decisions 0.99940 and 0.9025), the horizon in seconds the same
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 1
    minibatch: int = 4096      # decisions per minibatch (whole chunks: minibatch // T rows): one optimizer step
    accum: int = 1             # ... computed in this many parts whose gradients add up (gradient accumulation):
    #                            the GPU memory of minibatch / accum, the optimizer's steps those of minibatch
    lr: float = 1e-4
    target_kl: float = 0.05    # stop the epochs early when the mean per-unit KL passes this
    entropy: float = 0.01      # on the order kind's entropy (at most log 5 = 1.6)
    value: float = 0.5
    max_grad: float = 0.5
    anchor: float = 0.0        # weight of KL(policy || reference) on the order kind and target (0: off)
    eyes: float = EYES_WEIGHT  # weight of the eyes' loss (v2 with eyes, tools/nn/model/eyes.py; batch["eyes"])
    adv_norm: str = "batch"    # normalise the side's advantage over the minibatch ("batch") or over each
    #                            role's rows apart ("role": the attacker's, bigger with its idle cost, no
    #                            longer outweighs the defender's)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages and returns [T, R]. dones[t] [R]: the battle ended at step t (no bootstrap past it)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        go_on = 1.0 - dones[t].float()
        nxt = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * go_on - values[t]
        running = delta + gamma * lam * go_on * running
        adv[t] = running
    return adv, adv + values


def policy_loss(lp_new, lp_old, adv, mask, clip):
    """Clipped surrogate per unit. lp_* [S, N], adv [S] (the side's), mask [S, N] -> (loss, clip share)."""
    ratio = torch.exp(lp_new - lp_old)
    a = adv[:, None]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    m = mask.float()
    n = m.sum().clamp(min=1)
    clipped = (((ratio - 1).abs() > clip).float() * m).sum() / n
    return -(surrogate * m).sum() / n, clipped


def kind_entropy(logits):
    """[B, N] entropy of the order kind."""
    return torch.distributions.Categorical(logits=logits["kind"], validate_args=False).entropy()


def entropy_weight(weight, entropy, target, low, high, rate=1.25):
    """The next update's weight of the kind's entropy under an entropy floor: x rate while the kind's
    entropy (the update's mean) is below target, / rate at or above it, kept within [low, high].
    Iteration 1 (02.10): at the fixed 0.003 the bonus's gradient was 1e-4 to 1e-2 of the policy's and
    87-99.7 % of the units' kind choices had a probability above 0.99: no exploration."""
    w = weight * rate if entropy < target else weight / rate
    return min(high, max(low, w))


def masked_mean(x, mask):
    m = mask.float()
    return (x * m).sum() / m.sum().clamp(min=1)


def _rows(x, idx):
    """[T, R, ...] -> the rows idx, flattened to [T * len(idx), ...]."""
    y = x[:, idx]
    return y.reshape(-1, *y.shape[2:])


def categorical_kl(a, b):
    """KL(softmax a || softmax b) over the last axis, safe where b underflows to 0 (torch's
    kl_divergence gives inf there) and where both are masked (-1e9)."""
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return (la.exp() * (la - lb)).sum(-1)


def anchor_kl(logits, ref, ctrl):
    """Mean over units that take orders of KL(policy || reference) on the order kind, plus the target's
    KL weighted by the policy's probability of attacking."""
    k = categorical_kl(logits["kind"], ref["kind"])
    p_attack = torch.softmax(logits["kind"], -1)[..., hd.ATTACK]
    t = categorical_kl(logits["target"], ref["target"])
    return masked_mean(k + p_attack * t, ctrl)


def distance(actor, other, batch, minibatch):
    """The mean per-unit KL(actor || other) on the order kind and the target (anchor_kl), without
    gradients, over the first minibatch's worth of the batch's learner rows (each with its whole
    chunk; both networks start the chunk from the memory the actor began it with, as the anchor's
    reference does). E.g. the distance of the trained actor from the network the run started from."""
    T, R = batch["reward"].shape
    idx = torch.arange(min(R, max(1, minibatch // T)), device=batch["reward"].device)
    with torch.no_grad():
        obs = full_obs({k: v[:, idx] for k, v in batch["obs"].items()}, batch.get("abil_static"))
        act = chunk_action(batch["action"], idx)
        a, _ = actor.sequence(obs, batch["h0"][idx], batch["reset"][:, idx], act)
        b, _ = other.sequence(obs, batch["h0"][idx], batch["reset"][:, idx], act)
        a = {k: v.reshape(-1, *v.shape[2:]) for k, v in a.items()}
        b = {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}
        ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
        return float(anchor_kl(a, b, ctrl))


def chunk_action(action, idx):
    """The actions [T, R, ...] of the rows idx, kept [T, len(idx), ...] (the chained heads' logits are those under
    the actions taken: Actor.sequence's action)."""
    return hd.Action(**{f: getattr(action, f)[:, idx] for f in action.parts()})


def deciding(obs):
    """[.., N] the units that decide: v2's "free" (not held by a commitment, tools/nn/model/commit.py), else ctrl."""
    return obs["free"] if "free" in obs else obs["ctrl"]


def normalise(a, groups=None):
    """(a - mean) / std over all of a, or over each group apart (groups: bool of a's shape, e.g. the
    learner attacks)."""
    if groups is None:
        return (a - a.mean()) / (a.std() + 1e-8)
    out = torch.zeros_like(a)
    for m in (groups, ~groups):
        if int(m.sum()) > 1:
            out[m] = (a[m] - a[m].mean()) / (a[m].std() + 1e-8)
    return out


def explained_variance(value, ret):
    """1 - Var(return - value) / Var(return): 1 a perfect critic, 0 no better than the mean."""
    v = ret.var()
    return float(1 - (ret - value).var() / v) if v > 1e-12 else 0.0


def critic_stats(batch, adv, ret):
    """The critic's quality and the advantage's scale on a rollout, by the learner's role (batch["attacks"]
    [T, R]): explained variance of the returns, the std of the return and of the advantage, the std of
    the per-step reward and of the TD error (r + gamma V' - V at lambda 0 is mostly the critic's own step
    to step jitter when it is far above the reward's)."""
    out = {"ev": explained_variance(batch["value"], ret)}
    att = batch.get("attacks")
    if att is None:
        return out
    for role, m in (("attack", att), ("defend", ~att)):
        if not bool(m.any()):
            continue
        out[f"ev_{role}"] = explained_variance(batch["value"][m], ret[m])
        out[f"ret_std_{role}"] = float(ret[m].std())
        out[f"adv_mean_{role}"] = float(adv[m].mean())
        out[f"adv_std_{role}"] = float(adv[m].std())
        out[f"reward_std_{role}"] = float(batch["reward"][m].std())
    return out


def update(actor, critic, opt, batch, cfg=PPOConfig(), train_policy=True, reference=None, teach=None):
    """Some epochs of minibatch updates on one rollout (batch from rollout.collect). -> stats.
    train_policy False: the critic only (a warm-up for a critic that starts from nothing while the
    actor already plays: a checkpoint saved without one). reference: a frozen actor (e.g. the one the
    run started from) the policy is held near by cfg.anchor x KL, so PPO's noisy steps do not wash out
    what it started with while it looks for better. teach: {drill: weight now} of the imitation term
    (drills/teach.py) on the batch's labels (batch["teach"]); its stats (out["teach"]) whenever the
    batch has labels.

    A minibatch is a set of learner rows with their whole chunk of T decisions: the actor runs its
    memory through the chunk from the memory the chunk began with (Actor.sequence)."""
    adv, ret = gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg.gamma, cfg.lam)
    T, R = adv.shape
    # The actor's and the critic's gradients are clipped apart (each to max_grad). Clipped together
    # (until 02.10), a critic loss's gradient (then the per-unit value's, norm 200-2000; the actor's
    # 0.2-0.5) scaled the actor's by 0.0006-0.0025: per parameter ~1e-8, far below Adam's eps 1e-5, so
    # the policy took no step at all (KL 0.0002 an update, iterations it1-it3 flat; analyst probe 02.10).
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    critic_params = [p for p in critic.parameters() if p.requires_grad]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "entropy_all": 0.0, "kl": 0.0, "clip": 0.0,
             "anchor_kl": 0.0,
             "grad_norm": 0.0,          # the actor's gradient norm before clipping (max_grad)
             "grad_norm_critic": 0.0}   # the critic's
    n, stop = 0, False
    actor.train()
    critic.train()
    kinds = batch["action"].parts()
    taught = batch.get("teach")
    eyes_tg = batch.get("eyes") if cfg.eyes else None
    if eyes_tg is not None:
        for h in meyes.HEADS:
            stats[f"eyes_{h}"] = 0.0
            stats[f"eyes_ref_{h}"] = 0.0
    teach_acc = {n: [0.0] * 7 for n in taught["names"]} if taught else {}
    for _ in range(cfg.epochs):
        order = torch.randperm(R, device=adv.device)
        for idx in order.split(max(1, cfg.minibatch // T)):
            # the advantage normalised over the whole minibatch, then its rows in cfg.accum parts: the
            # parts' gradients add up to the minibatch's (each part's loss x its share of the rows)
            a_all = normalise(_rows(adv, idx), _rows(batch["attacks"], idx) if cfg.adv_norm == "role"
                              and batch.get("attacks") is not None else None).reshape(T, len(idx))
            parts = torch.arange(len(idx), device=idx.device).tensor_split(max(1, min(cfg.accum, len(idx))))
            opt.zero_grad(set_to_none=True)
            step = dict.fromkeys(("policy_loss", "value_loss", "entropy", "kl", "clip", "entropy_all", "anchor_kl")
                                 + tuple(k for k in stats if k.startswith("eyes_")), 0.0)
            for part in parts:
                share = len(part) / len(idx)
                pidx = idx[part]
                obs = full_obs({k: v[:, pidx] for k, v in batch["obs"].items()}, batch.get("abil_static"))
                cobs = {k: _rows(v, pidx) for k, v in batch["critic_obs"].items()}
                act = hd.Action(**{k: _rows(getattr(batch["action"], k), pidx) for k in kinds})
                seq_act = chunk_action(batch["action"], pidx)
                a = a_all[:, part].reshape(-1)
                # The critic's warm-up (train_policy False): the actor runs without a graph, for the stats
                # only; with one its activations took ~3.5 GB more at the peak (13.5 GB on a 16 GB card).
                with torch.set_grad_enabled(train_policy):
                    logits, _ = actor.sequence(obs, batch["h0"][pidx], batch["reset"][:, pidx], seq_act)
                    logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
                    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
                    lp, ent = hd.log_prob(logits, act, ctrl)
                    old = _rows(batch["lp"], pidx)
                    pl, clipped = policy_loss(lp, old, a, ctrl, cfg.clip)
                    free = deciding(obs).reshape(-1, ctrl.shape[-1])
                    entropy = masked_mean(kind_entropy(logits), free)
                    if reference is not None and cfg.anchor:
                        with torch.no_grad():
                            ref, _ = reference.sequence(obs, batch["h0"][pidx], batch["reset"][:, pidx], seq_act)
                            ref = {k: v.reshape(-1, *v.shape[2:]) for k, v in ref.items()}
                        anchored = anchor_kl(logits, ref, ctrl)
                    else:
                        anchored = torch.zeros((), device=adv.device)
                    imitation = torch.zeros((), device=adv.device)
                    if taught:
                        label = hd.Action(*(_rows(getattr(taught["action"], k), pidx) for k in ("kind", "point", "target", "run")))
                        picked = _rows(taught["picked"], pidx) if taught.get("picked") is not None else None
                        imitation, sums = drill_teach.terms(logits, label, _rows(taught["valid"], pidx),
                                                            _rows(taught["drill"], pidx), taught["names"], teach,
                                                            picked)
                        for name, v in sums.items():
                            teach_acc[name] = [x + y for x, y in zip(teach_acc[name], v)]
                    seen = torch.zeros((), device=adv.device)
                    if eyes_tg is not None and train_policy:
                        flat = {k: obs[k].reshape(-1, *obs[k].shape[2:]) for k in ("own", "attend", "pos")}
                        seen, parts_e = meyes.loss(actor.cfg, logits, {k: _rows(v, pidx) for k, v in eyes_tg.items()},
                                                   flat)
                        for h, (mse, ref) in parts_e.items():
                            step[f"eyes_{h}"] += float(mse.detach()) * share
                            step[f"eyes_ref_{h}"] += float(ref) * share
                vl = ((critic(cobs) - _rows(ret, pidx)) ** 2).mean()
                policy_part = pl - cfg.entropy * entropy + cfg.anchor * anchored + imitation + cfg.eyes * seen
                loss = cfg.value * vl + (policy_part if train_policy else 0.0)
                (loss * share).backward()
                with torch.no_grad():
                    kl = masked_mean(old - lp, ctrl)
                for k, x in (("policy_loss", pl), ("value_loss", vl), ("entropy", entropy), ("kl", kl),
                             ("clip", clipped), ("entropy_all", masked_mean(ent, ctrl)), ("anchor_kl", anchored)):
                    step[k] += float(x.detach()) * share
            # one norm clip for the actor's whole loss (PPO + anchor + teacher + the eyes), one for the critic's
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad)
            grad_norm_critic = torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad)
            if torch.isfinite(grad_norm) and torch.isfinite(grad_norm_critic):
                opt.step()
            else:                                    # a non-finite gradient: no step (it would poison the weights)
                stats["skipped"] = stats.get("skipped", 0.0) + 1.0
                opt.zero_grad(set_to_none=True)
            step["grad_norm"], step["grad_norm_critic"] = float(grad_norm), float(grad_norm_critic)
            kl = step["kl"]
            for k, x in step.items():
                stats[k] += x
            n += 1
            if train_policy and cfg.target_kl and float(kl) > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    actor.eval()
    critic.eval()
    out = {k: v / max(1, n) for k, v in stats.items()}
    out["minibatches"] = n
    out["skipped"] = stats.get("skipped", 0.0)
    for h in meyes.HEADS:
        # explained share over the whole update (one minibatch whose targets are all ~0 gave -270 averaged per
        # minibatch: a ratio of two tiny numbers, not an error of the eyes)
        if f"eyes_ref_{h}" in out:
            out[f"eyes_ev_{h}"] = 1.0 - out[f"eyes_{h}"] / max(out.pop(f"eyes_ref_{h}"), 1e-9)
    if taught:
        out["teach"] = drill_teach.summary(teach_acc, teach, taught.get("shares"))
    out["reward"] = float(batch["reward"].sum(0).mean())
    out["value_mean"] = float(batch["value"].mean())
    out["return_mean"] = float(ret.mean())
    out.update(critic_stats(batch, adv, ret))
    return out
