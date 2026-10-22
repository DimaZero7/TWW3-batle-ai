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
  The other v2 heads (target, sector, cell, hold, run) have floors of their own (cfg.heads, run.py --head-entropy,
  head_entropy.py): each its weight x its entropy's share of the maximum, on the rows where it is used. Every head's
  entropy is logged (head_share_<head>, head_ent_<head>) whether it has a floor or not.
* GAE over the rollout; the value of the last state bootstraps; a finished battle cuts it.
* The memory (GRU) is trained through time: a minibatch is a set of whole chunks (T decisions of
  some battles); the actor runs its memory through each chunk from the memory stored when the
  chunk began, emptied where a new battle begins (recurrent PPO, as R2D2's stored state).
* The drills' teacher (update's teach=, run.py --drill-teach; tools/nn/train/drills/teach.py): with
  labels in the batch, + weight x the cross-entropy of the policy against the drill's skilled script on
  the labelled units of that drill's battles (over all its units: only a share of its battles may be
  labelled, the adaptive teacher's, run.py --drill-teach auto); its term and the agreement are logged per
  drill.
* Learning by observation (update's observe=, run.py --observe-dir; tools/nn/observe): beside every minibatch a few
  chunks of another player's recorded battles, + weight x the imitation of his orders the network's critic found
  better than expected (observe/loss.py; the weight decays every update), and + anchor x KL(policy || the network the
  run started from) on the minibatch's own rows (its weight decays too). Its stats: out["observe"]. Once the term's
  weight is negligible (store.NEGLIGIBLE) the term and its chunks are skipped; the anchor goes on by its own weight.
* The general (S28, tools/nn/general/learn.py; batch["general"] from rollout.collect, an executor with a general): the
  executor's task input is computed again in every part from the chunk's plan table (learn.task_for, with the
  gradient: the executor's loss reaches the general's network); before the epochs the general takes its own PPO
  steps on the plans that ended (learn.update; update's general= its LearnConfig, None: off; first, so its plans'
  ratios start from the weights they were made with, not from those the executor's epochs moved). Its stats:
  general_*.
  The executor's bonus for carrying out its missions (batch["exec"] [T, R, N], run.py --general-exec) is each unit's
  own part of its advantage (signal.exec_advantage, added to the side's normalised one): the policy loss's advantage is
  then per unit; the reward, the returns and the critic are as without it.
* The stats are summed on the device in float64 (Python's float arithmetic: the same numbers) and read once at the end;
  the GPU is read once a minibatch only (whether the gradient is finite, the KL for target_kl): a float() a stat (~30 a
  part, ~3500 an update) made the CPU wait for the GPU each time.
"""
import contextlib
from dataclasses import dataclass

import torch

from tools.nn.general import signal as gsignal
from tools.nn.model import eyes as meyes
from tools.nn.model import heads as hd
from tools.nn.train import head_entropy
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
    heads: tuple = ()          # ((head, weight), ...): the per-head entropy floors' weights now (head_entropy.py)
    adv_norm: str = "batch"    # normalise the side's advantage over the minibatch ("batch") or over each
    #                            role's rows apart ("role": the attacker's, bigger with its idle cost, no
    #                            longer outweighs the defender's)
    ability: float = 0.0       # weight of the ability branches' term now (batch["abil_branch"], ability_branch.py;
    #                            run.py --ability-branch-weight, decaying; --ability-weight under the own signal): 0 off
    ability_own: bool = False  # the ability head learns from its own term only: its log-probability is out of the main
    #                            surrogate (and of the KL's main part), run.py --ability-own-signal
    ability_body_grad: float = 1.0  # the gradient through the ability head's input (the unit's thought) x this: the
    #                                 body's share of the ability term (0: none); used under ability_own
    ability_epochs: int = 1    # passes of the ability term over the update's priced rows: the main one + this - 1 of the
    #                            ability head alone on the stored thoughts (ability_epochs; run.py --ability-epochs)
    amp: str = ""              # "bf16" / "bf16h": the update's actor and critic passes under torch.autocast(bfloat16)
    #                            ("bf16h": the actor's heads in float32; _amp, run.py --amp)


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
    """Clipped surrogate per unit. lp_* [S, N], adv [S] (the side's) or [S, N] (each unit's: the side's + its own
    mission bonus, --general-exec), mask [S, N] -> (loss, clip share)."""
    ratio = torch.exp(lp_new - lp_old)
    a = adv[:, None] if adv.dim() == 1 else adv
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    m = mask.float()
    n = m.sum().clamp(min=1)
    clipped = (((ratio - 1).abs() > clip).float() * m).sum() / n
    return -(surrogate * m).sum() / n, clipped


def ability_log_prob(logits, act):
    """[S, N] the ability head's log-probability of the choice made (none: index 0)."""
    la = torch.log_softmax(torch.nan_to_num(logits["ability"], nan=hd.NEG), -1)
    return la.gather(-1, (act.ability.clamp(min=-1) + 1)[..., None])[..., 0]


def ability_term(logits, act, branch, ctrl, clip, lp=None):
    """The ability branches' term (ability_branch.py): - the clipped surrogate of the ability head's log-probability of
    the choice made under the branches' advantage, on the priced rows only, summed over them and divided by the units
    that take orders (as policy_loss: a weight of 0.1 is a tenth of PPO's per-unit push on those rows). Only the ability
    logits enter it: not the reward, not the critic, not the other heads. branch: {"adv", "mask", "lp"} [S, N] (the
    minibatch's rows); lp: ability_log_prob already made. -> (term, priced rows)."""
    if lp is None:
        lp = ability_log_prob(logits, act)
    m = branch["mask"]
    ratio = torch.exp(torch.where(m, lp - branch["lp"], torch.zeros_like(lp)))
    a = branch["adv"]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    n = ctrl.float().sum().clamp(min=1)
    return -(surrogate * m.float()).sum() / n, m.float().sum()


# The ability head's own parameters (ChainHeads / Heads): the unit's query, the slots' key (its encoder of the slot's
# state and passport) and the "none" logit. Not the heads' norm, nor the body (encoder.py AbilityEncoder / AbilityExtra
# feed the unit's token: the shared body). run.py --ability-lr-mult gives them an Adam group of their own
ABILITY_HEAD = ("heads.ability_q.", "heads.ability_k.", "heads.ability_none.")


def ability_head_params(actor):
    """[parameters] of the ability head (ABILITY_HEAD), in the actor's order."""
    return [p for n, p in actor.named_parameters() if n.startswith(ABILITY_HEAD)]


def head_ability_logits(heads, u, obs_t, ctrl):
    """[B, N, 1 + ACTIVE] the ability logits of ChainHeads.forward (tools/nn/model/chain.py: the same operations, kept
    here so the model's files - the simulator's version, version.py - stay as they are; a test holds them equal) from
    the units' thought u [B, N, d] (after the heads' norm): the ability head alone."""
    from tools.nn.model import chain
    usable = obs_t["abil_ok"] & ctrl[..., None]
    bad = u.new_zeros(u.shape[0])
    score, bad = chain._cap(heads._ability_scores(u, obs_t, usable), bad)
    slot = torch.where(usable, score, torch.full_like(score, hd.NEG))
    none, bad = chain._cap(heads.ability_none(u), bad)
    return torch.cat([none, slot], -1)


class ThoughtTap:
    """Within it (on), every run of the ability head keeps its input - the units' thought u [B, N, d], detached - in
    .u (a forward hook on heads.ability_none): the update stores the priced rows' thoughts for the extra ability epochs
    (ability_epochs). Off, or an actor without the chained heads (chain.ChainHeads): nothing."""

    def __init__(self, actor, on):
        heads = getattr(actor, "heads", None)
        self.on = bool(on) and hasattr(heads, "_ability_scores")
        self.heads, self.u, self.handle = heads, None, None

    def __enter__(self):
        if self.on:
            self.handle = self.heads.ability_none.register_forward_hook(
                lambda mod, args, out: setattr(self, "u", args[0].detach()))
        return self

    def take(self):
        u, self.u = self.u, None
        return u

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle, self.u = None, None
        return False


def ability_rows(u, obs, ctrl, act, branch, n_ctrl, share):
    """A part's priced rows kept for the extra ability epochs: the thought u [S, N, d] (ThoughtTap), the slots
    (obs abil / abil_idx / abil_ok [S, N, ..]), ctrl [S, N], the ability chosen, the branches' advantage and old
    log-probability (branch [S, N]) where branch["mask"]; n_ctrl: the part's units that take orders (ability_term's
    divisor), share: the part's share of its minibatch."""
    b, n = branch["mask"].nonzero(as_tuple=True)
    return {"u": u[b, n], "abil": obs["abil"][b, n], "abil_idx": obs["abil_idx"][b, n], "abil_ok": obs["abil_ok"][b, n],
            "ctrl": ctrl[b, n], "ability": act.ability[b, n], "adv": branch["adv"][b, n], "lp": branch["lp"][b, n],
            "n": n_ctrl, "share": share}


def kl_k3(ratio):
    """Per row, (ratio - 1) - log ratio (ratio = new / old probability of the choice made): an unbiased estimate of
    KL(old || new) that is never negative (Schulman's k3); the plain old - new log-probability can come out negative
    over the few dozen priced rows of a minibatch."""
    return (ratio - 1) - torch.log(ratio)


def ability_epochs(actor, opt, cache, cfg):
    """cfg.ability_epochs - 1 more passes of the ability term (as ability_term: the clipped surrogate of the ability
    ratio, new / the rollout's) over the update's stored priced rows (cache: per minibatch, its parts' ability_rows),
    by the ability head alone (head_ability_logits on the stored thoughts: no body, no memory), one optimizer step
    a minibatch (Adam steps only the parameters with a gradient: the head's), its gradient clipped to max_grad. -> the
    last pass's stats (each minibatch's before its step): ability_extra_kl (the head's KL(old || new) over the priced
    rows, kl_k3), ability_extra_clip (their share with |ratio - 1| > clip), ability_extra_loss (a minibatch's term),
    ability_extra_rows (priced rows a minibatch), ability_extra_steps, ability_extra_skipped (non-finite gradients)."""
    params = ability_head_params(actor)
    heads = actor.heads
    zero = torch.zeros((), dtype=torch.float64, device=params[0].device)
    kl_s, clip_s, rows, loss_s, steps, skipped = zero, zero, zero, zero, 0, 0
    full = [mb for mb in cache if any(c["u"].shape[0] for c in mb)]
    for e in range(cfg.ability_epochs - 1):
        last = e == cfg.ability_epochs - 2
        for i in torch.randperm(len(full)).tolist():
            opt.zero_grad(set_to_none=True)
            for c in full[i]:
                M = c["u"].shape[0]
                if not M:
                    continue
                obs_t = {k: c[k][:, None] for k in ("abil", "abil_idx", "abil_ok")}
                logits = head_ability_logits(heads, c["u"][:, None], obs_t, c["ctrl"][:, None])
                lp = ability_log_prob({"ability": logits}, hd.Action(None, None, None, None, c["ability"][:, None]))[:, 0]
                ratio = torch.exp(lp - c["lp"])
                a = c["adv"]
                surrogate = torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                term = -surrogate.sum() / c["n"]
                (cfg.ability * term * c["share"]).backward()
                if last:
                    with torch.no_grad():
                        kl_s = kl_s + _f(kl_k3(ratio).sum())
                        clip_s = clip_s + _f(((ratio - 1).abs() > cfg.clip).sum())
                        rows = rows + M
                        loss_s = loss_s + _f(term) * c["share"]
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
            if bool(torch.isfinite(gn)):
                opt.step()
            else:                                    # a non-finite gradient: no step
                skipped += 1
            steps += 1
    opt.zero_grad(set_to_none=True)
    r = _read({"kl": kl_s, "clip": clip_s, "rows": rows, "loss": loss_s})
    n_mb = max(1, len(full))
    return {"ability_extra_kl": r["kl"] / max(1.0, r["rows"]), "ability_extra_clip": r["clip"] / max(1.0, r["rows"]),
            "ability_extra_loss": r["loss"] / n_mb, "ability_extra_rows": r["rows"] / n_mb,
            "ability_extra_steps": float(steps), "ability_extra_skipped": float(skipped)}


def scale_grad(x, s):
    """x in the forward pass (bit for bit), its gradient x s in the backward pass (s 0: none)."""
    if s == 0:
        return x.detach()
    d = x.detach()
    return d + (x - d) * s


class AbilityInputGrad:
    """Within it, the gradient that flows out of the ability head into its input - the unit's thought, the input of
    ChainHeads.ability_q and ability_none (heads.Heads: the same names) - is x scale: the body's share of the ability
    head's own term (--ability-body-grad; module doc). Forward pre-hooks on the two layers, removed on exit; scale 1
    (or no such layers) adds none."""

    def __init__(self, actor, scale):
        self.actor, self.scale, self.handles = actor, float(scale), []

    def __enter__(self):
        heads = getattr(self.actor, "heads", None)
        if self.scale != 1.0 and heads is not None:
            s = self.scale
            for name in ("ability_q", "ability_none"):
                m = getattr(heads, name, None)
                if m is not None:
                    self.handles.append(m.register_forward_pre_hook(
                        lambda mod, args: (scale_grad(args[0], s),) + tuple(args[1:])))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


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


def _f(x):
    """A 0-d stat for the update's sums, kept on its device: float64 as Python's float() gives it (no read of the GPU)."""
    return x.detach().double()


def _read(d):
    """The dict's tensor values (0-d, one device) as Python floats, in one read of the GPU; the others as they are."""
    keys = [k for k, v in d.items() if torch.is_tensor(v)]
    if keys:
        vals = torch.stack([d[k].reshape(()) for k in keys]).tolist()
        d = {**d, **dict(zip(keys, vals))}
    return d


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


def with_task(actor, batch, obs):
    """obs (a part's [T, P, ..] observation) with the executor's task input computed from the batch's plan table
    (tools/nn/general/learn.task_for) when the actor reads one and the batch has the general's plans; else as it is."""
    table = (batch.get("general") or {}).get("table")
    if table is None or "task_plan" not in obs or getattr(actor, "gcfg", None) is None:
        return obs
    from tools.nn.general import learn as glearn
    task, on = glearn.task_for(actor, table, obs)
    return dict(obs, task=task, task_on=on)


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
        a, _ = actor.sequence(with_task(actor, batch, obs), batch["h0"][idx], batch["reset"][:, idx], act)
        b, _ = other.sequence(obs, batch["h0"][idx], batch["reset"][:, idx], act)
        a = {k: v.reshape(-1, *v.shape[2:]) for k, v in a.items()}
        b = {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}
        ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
        return float(anchor_kl(a, b, ctrl))


def _float32(x):
    """x's floating tensors (in tuples, lists and dicts too) as float32."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: _float32(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_float32(v) for v in x)
    return x


@contextlib.contextmanager
def _fp32_heads(model):
    """Within it, model.heads (if any) runs in float32 outside autocast: its inputs cast up, autocast off for its
    forward (pre / post hooks)."""
    heads = getattr(model, "heads", None)
    if heads is None:
        yield
        return
    off = []

    def pre(mod, args, kwargs):
        ctx = torch.autocast("cuda", enabled=False)
        ctx.__enter__()
        off.append(ctx)
        return _float32(args), _float32(kwargs)

    def post(mod, args, kwargs, out):
        off.pop().__exit__(None, None, None)
        return out
    h1 = heads.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = heads.register_forward_hook(post, with_kwargs=True)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()
        while off:
            off.pop().__exit__(None, None, None)


@contextlib.contextmanager
def _amp(cfg, model=None):
    """cfg.amp on CUDA: "bf16" - the passes under torch.autocast(bfloat16); "bf16h" - the same with model's heads in
    float32 (_fp32_heads: the logits' own rounding out of the log-probabilities); else nothing."""
    if cfg.amp not in ("bf16", "bf16h") or not torch.cuda.is_available():
        yield
        return
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if cfg.amp == "bf16h" and model is not None:
            with _fp32_heads(model):
                yield
        else:
            yield


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


def update(actor, critic, opt, batch, cfg=PPOConfig(), train_policy=True, reference=None, teach=None, observe=None,
           general=None):
    """Some epochs of minibatch updates on one rollout (batch from rollout.collect). -> stats (_update). Under the
    ability head's own signal (cfg.ability_own) the gradient through the ability head's input is x
    cfg.ability_body_grad (AbilityInputGrad). general: the general's LearnConfig (tools/nn/general/learn.py): its PPO
    steps on the batch's ended plans before the unit network's epochs (module doc); None: none."""
    extra = (cfg.ability_epochs > 1 and train_policy and bool(cfg.ability) and batch.get("abil_branch") is not None
             and batch["action"].ability is not None)
    ready = (batch.get("general") or {}).get("ready")
    g_out = {}
    if general is not None and train_policy and ready is not None and getattr(actor, "gcfg", None) is not None:
        from tools.nn.general import learn as glearn
        # The general's PPO first: the unit network's epochs move the general's weights too (the executor's gradient
        # through the task cards, learn.task_for: most of its tensors, each by ~lr with Adam however small the
        # gradient), so after them its plans' ratios no longer start near 1 (build/diag_clip: clip 0.3-0.5 at kl_plan
        # 0.02-0.06). It needs nothing of those epochs (the ended plans' records only). (With --general-branches the
        # chunk's mission kind prices: learn.update's branch term.)
        # (--general-imitate: the teacher's ended plans, the imitation's records)
        # (--general-kind-critic: the kind critic's estimate of the ended plans' groups)
        g_out = glearn.update(actor, opt, ready, general, branch=batch["general"].get("branch"),
                              teach=batch["general"].get("teach"), kind_q=batch["general"].get("kind_q"))
    with AbilityInputGrad(actor, cfg.ability_body_grad if cfg.ability_own else 1.0), ThoughtTap(actor, extra) as tap:
        out = _update(actor, critic, opt, batch, cfg, train_policy, reference, teach, observe, tap if tap.on else None)
    out.update(g_out)
    return out


def _update(actor, critic, opt, batch, cfg=PPOConfig(), train_policy=True, reference=None, teach=None, observe=None,
            tap=None):
    """Some epochs of minibatch updates on one rollout (batch from rollout.collect). -> stats.
    train_policy False: the critic only (a warm-up for a critic that starts from nothing while the
    actor already plays: a checkpoint saved without one). reference: a frozen actor (e.g. the one the
    run started from) the policy is held near by cfg.anchor x KL, so PPO's noisy steps do not wash out
    what it started with while it looks for better. teach: {drill: weight now} of the imitation term
    (drills/teach.py) on the batch's labels (batch["teach"]); its stats (out["teach"]) whenever the
    batch has labels. observe: the demonstrations' term (tools/nn/observe/store.py Observe; None: off).

    A minibatch is a set of learner rows with their whole chunk of T decisions: the actor runs its
    memory through the chunk from the memory the chunk began with (Actor.sequence).

    cfg.ability_own: the ability head's log-probability is left out of the main surrogate (the main advantage no longer
    trains it; its old part comes from the branches' batch["abil_branch"]["lp_all"]); the KL for target_kl stays the
    whole action's (the main part's + the ability head's, logged apart as ability_kl).

    tap (ThoughtTap; cfg.ability_epochs > 1): the priced rows' thoughts are kept, and after the epochs the ability head
    alone takes cfg.ability_epochs - 1 more passes over them (ability_epochs). Whenever the ability term is on, its KL
    and clip share on the priced rows are logged (ability_branch_kl, ability_branch_clip)."""
    adv, ret = gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg.gamma, cfg.lam)
    T, R = adv.shape
    # the general's executor's bonus for carrying out its missions (--general-exec; tools/nn/general/runtime.execute):
    # each unit's own part of its advantage [T, R, N] (a reward-to-go at gamma x lambda, in the normalised advantage's
    # units), added to the side's after its normalisation; the reward, the returns and the critic do not see it
    ex_adv = (gsignal.exec_advantage(batch["exec"].float(), batch["done"].float(), cfg.gamma * cfg.lam)
              if batch.get("exec") is not None and train_policy else None)
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
    obs_on = observe is not None and train_policy
    if obs_on:
        observe.before(actor)                # the demonstrations' stored memories again when due
        obs_acc, obs_sums, obs_n = {"loss": 0.0, "anchor_kl": 0.0}, {}, 0
        obs_w, obs_anchor = observe.weight, observe.anchor
        obs_term = observe.on                # its weight not negligible (store.NEGLIGIBLE): the term and its chunks
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
    branch = batch.get("abil_branch") if cfg.ability and train_policy else None
    if branch is not None and batch["action"].ability is None:
        branch = None
    own = bool(cfg.ability_own) and batch["action"].ability is not None
    if own:
        ab_all = batch.get("abil_branch")
        if ab_all is None or "lp_all" not in ab_all:
            raise ValueError("ability_own needs the branches' lp_all (run.py --ability-own-signal with --ability-branch)")
        stats["ability_kl"] = 0.0
    if branch is not None:
        stats["ability_branch_loss"] = 0.0      # the term (before its weight), a minibatch's mean
        stats["ability_branch_rows"] = 0.0      # priced rows a minibatch
        stats["ability_branch_kl"] = 0.0        # the ability head's KL(old || new) over the priced rows (kl_k3)
        stats["ability_branch_clip"] = 0.0      # their share with the ability ratio outside the clip
    cache = [] if tap is not None and branch is not None else None   # per minibatch: its parts' priced rows
    head_w = dict(cfg.heads)
    head_acc = {}                                # head -> [share x rows, nats x rows, rows] (head_entropy.measure)
    need_kl = bool(train_policy and cfg.target_kl)
    for _ in range(cfg.epochs):
        order = torch.randperm(R, device=adv.device)
        for idx in order.split(max(1, cfg.minibatch // T)):
            # the advantage normalised over the whole minibatch, then its rows in cfg.accum parts: the
            # parts' gradients add up to the minibatch's (each part's loss x its share of the rows)
            a_all = normalise(_rows(adv, idx), _rows(batch["attacks"], idx) if cfg.adv_norm == "role"
                              and batch.get("attacks") is not None else None).reshape(T, len(idx))
            parts = torch.arange(len(idx), device=idx.device).tensor_split(max(1, min(cfg.accum, len(idx))))
            opt.zero_grad(set_to_none=True)
            kept = []
            step = dict.fromkeys(("policy_loss", "value_loss", "entropy", "kl", "clip", "entropy_all", "anchor_kl")
                                 + tuple(k for k in stats if k.startswith("eyes_") or k.startswith("ability_branch_")
                                         or k == "ability_kl"), 0.0)
            for part in parts:
                share = len(part) / len(idx)
                pidx = idx[part]
                obs = full_obs({k: v[:, pidx] for k, v in batch["obs"].items()}, batch.get("abil_static"))
                cobs = full_obs({k: v[:, pidx] for k, v in batch["critic_obs"].items()}, batch.get("abil_static"))
                cobs = {k: v.reshape(-1, *v.shape[2:]) for k, v in cobs.items()}
                act = hd.Action(**{k: _rows(getattr(batch["action"], k), pidx) for k in kinds})
                seq_act = chunk_action(batch["action"], pidx)
                a = a_all[:, part].reshape(-1)
                if ex_adv is not None:
                    a = a[:, None] + _rows(ex_adv, pidx)                  # [S, N]: each unit's mission bonus too
                # The critic first, its backward pass at once: the actor and the critic share no weights, so the
                # gradients are those of one backward pass of the sum, and the critic's activations (~2.2 MB a
                # decision) are freed before the actor's are made (the part's peak memory)
                with _amp(cfg):
                    v_new = critic(cobs)
                vl = ((v_new.float() - _rows(ret, pidx)) ** 2).mean()
                (cfg.value * vl * share).backward()
                # The critic's warm-up (train_policy False): the actor runs without a graph, for the stats
                # only; with one its activations took ~3.5 GB more at the peak (13.5 GB on a 16 GB card).
                with torch.set_grad_enabled(train_policy):
                    obs = with_task(actor, batch, obs)       # the general's cards again, with the gradient
                    with _amp(cfg, actor):
                        logits, _ = actor.sequence(obs, batch["h0"][pidx], batch["reset"][:, pidx], seq_act)
                    logits = {k: v.float() for k, v in logits.items()}
                    logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
                    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
                    lp_ab = old_ab = None
                    if own:
                        # the main surrogate without the ability head: its log-probability (and entropy) left out
                        lp, ent = hd.log_prob(logits, hd.Action(**{k: getattr(act, k) for k in kinds
                                                                   if k != "ability"}), ctrl)
                        lp_ab = torch.where(ctrl, ability_log_prob(logits, act), torch.zeros_like(lp))
                        old_ab = _rows(ab_all["lp_all"], pidx)
                    else:
                        lp, ent = hd.log_prob(logits, act, ctrl)
                    # (the full entropy is for the log only: its graph - softmaxes over the 256 sectors - let go now)
                    ent_all = masked_mean(ent.detach(), ctrl)
                    del ent
                    old = _rows(batch["lp"], pidx) - (old_ab if own else 0.0)
                    pl, clipped = policy_loss(lp, old, a, ctrl, cfg.clip)
                    free = deciding(obs).reshape(-1, ctrl.shape[-1])
                    entropy = masked_mean(kind_entropy(logits), free)
                    spread = None
                    if "sector" in logits:                       # v2: every head's spread (head_entropy.py)
                        measured = head_entropy.measure(logits, act, free, ctrl,
                                                        grad=tuple(h for h, w in head_w.items() if w))
                        spread = head_entropy.bonus(measured, head_w)
                        for h, (sh, nats, rows) in measured.items():
                            acc = head_acc.setdefault(h, [0.0, 0.0, 0.0])
                            r = _f(rows)
                            acc[0] += _f(sh) * r
                            acc[1] += _f(nats) * r
                            acc[2] += r
                    if reference is not None and cfg.anchor:
                        with torch.no_grad():
                            ref, _ = reference.sequence(obs, batch["h0"][pidx], batch["reset"][:, pidx], seq_act)
                            ref = {k: v.reshape(-1, *v.shape[2:]) for k, v in ref.items()}
                        anchored = anchor_kl(logits, ref, ctrl)
                    else:
                        anchored = torch.zeros((), device=adv.device)
                    observed = torch.zeros((), device=adv.device)
                    if obs_on and obs_anchor > 0:
                        with torch.no_grad():
                            oref, _ = observe.anchor_net.sequence(obs, batch["h0"][pidx], batch["reset"][:, pidx], seq_act)
                            oref = {k: v.reshape(-1, *v.shape[2:]) for k, v in oref.items()}
                        observed = anchor_kl(logits, oref, ctrl)
                        obs_acc["anchor_kl"] += _f(observed) * share
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
                            step[f"eyes_{h}"] += _f(mse) * share
                            step[f"eyes_ref_{h}"] += _f(ref) * share
                    branched = torch.zeros((), device=adv.device)
                    if branch is not None and "ability" in logits:
                        br = {k: _rows(v, pidx) for k, v in branch.items() if k in ("adv", "mask", "lp")}
                        lp_b = lp_ab if lp_ab is not None else ability_log_prob(logits, act)
                        branched, priced = ability_term(logits, act, br, ctrl, cfg.clip, lp=lp_b)
                        step["ability_branch_loss"] += _f(branched) * share
                        step["ability_branch_rows"] += _f(priced)
                        with torch.no_grad():
                            ratio_b = torch.exp(torch.where(br["mask"], lp_b - br["lp"], torch.zeros_like(lp_b)))
                            step["ability_branch_kl"] += _f(masked_mean(kl_k3(ratio_b), br["mask"])) * share
                            step["ability_branch_clip"] += _f(masked_mean(((ratio_b - 1).abs() > cfg.clip).float(),
                                                                          br["mask"])) * share
                        if cache is not None:
                            flat = {k: obs[k].reshape(-1, *obs[k].shape[2:]) for k in ("abil", "abil_idx", "abil_ok")}
                            kept.append(ability_rows(tap.take(), flat, ctrl, act, br, ctrl.float().sum().clamp(min=1),
                                                     share))
                policy_part = (pl - cfg.entropy * entropy + cfg.anchor * anchored + imitation + cfg.eyes * seen
                               + (obs_anchor * observed if obs_on else 0.0) - (spread if spread is not None else 0.0)
                               + cfg.ability * branched)
                if train_policy:
                    (policy_part * share).backward()
                with torch.no_grad():
                    kl = masked_mean(old - lp, ctrl)
                    if own:                          # the whole action's KL: the main part's + the ability head's
                        kl_ab = masked_mean(old_ab - lp_ab, ctrl)
                        step["ability_kl"] += _f(kl_ab) * share
                        kl = kl + kl_ab
                for k, x in (("policy_loss", pl), ("value_loss", vl), ("entropy", entropy), ("kl", kl),
                             ("clip", clipped), ("entropy_all", ent_all), ("anchor_kl", anchored)):
                    step[k] += _f(x) * share
            if obs_on and obs_term:
                # the demonstrations: their own chunks, one backward pass beside the minibatch's
                lo, so = observe.term(actor)
                if tap is not None:
                    tap.take()                       # (the demonstrations' thoughts: not kept)
                (obs_w * lo).backward()
                obs_acc["loss"] += float(lo.detach())
                obs_n += 1
                for k, x in so.items():
                    obs_sums[k] = obs_sums.get(k, 0.0) + x
            # one norm clip for the actor's whole loss (PPO + anchor + teacher + the eyes + the observation), one for
            # the critic's
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad)
            grad_norm_critic = torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad)
            # the minibatch's one read of the GPU: is the gradient finite, and the KL (target_kl's early stop)
            finite = (torch.isfinite(grad_norm) & torch.isfinite(grad_norm_critic)).double()
            read = (torch.stack([finite, torch.as_tensor(step["kl"], dtype=torch.float64, device=finite.device)])
                    if need_kl else finite[None]).tolist()
            if read[0]:
                opt.step()
            else:                                    # a non-finite gradient: no step (it would poison the weights)
                stats["skipped"] = stats.get("skipped", 0.0) + 1.0
                opt.zero_grad(set_to_none=True)
            if cache is not None:
                cache.append(kept)
            step["grad_norm"], step["grad_norm_critic"] = _f(grad_norm), _f(grad_norm_critic)
            for k, x in step.items():
                stats[k] += x
            n += 1
            if need_kl and read[1] > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    extra_stats = ability_epochs(actor, opt, cache, cfg) if cache else {}
    del cache
    actor.eval()
    critic.eval()
    stats = _read(stats)
    out = {k: v / max(1, n) for k, v in stats.items()}
    out["minibatches"] = n
    out.update(extra_stats)
    out["skipped"] = stats.get("skipped", 0.0)
    for h in meyes.HEADS:
        # explained share over the whole update (one minibatch whose targets are all ~0 gave -270 averaged per
        # minibatch: a ratio of two tiny numbers, not an error of the eyes)
        if f"eyes_ref_{h}" in out:
            out[f"eyes_ev_{h}"] = 1.0 - out[f"eyes_{h}"] / max(out.pop(f"eyes_ref_{h}"), 1e-9)
    for h, acc in head_acc.items():
        s_sum, n_sum, r = _read(dict(enumerate(acc))).values()
        if r:
            out[f"head_share_{h}"] = s_sum / r
            out[f"head_ent_{h}"] = n_sum / r
    if taught:
        out["teach"] = drill_teach.summary(teach_acc, teach, taught.get("shares"))
    if obs_on:
        from tools.nn.observe import loss as obs_loss
        obs_acc = _read(obs_acc)
        out["observe"] = {"anchor_kl": obs_acc["anchor_kl"] / max(1, n), "weight_now": obs_w, "anchor_now": obs_anchor}
        if obs_term:                          # (the term's stats while it counts)
            out["observe"].update(loss=obs_acc["loss"] / max(1, obs_n), **obs_loss.summary(obs_sums))
        observe.after()
    out["reward"] = float(batch["reward"].sum(0).mean())
    if ex_adv is not None:
        # the mission bonus's part of the units' advantage (the side's is normalised: std 1), over the units paid
        paid = batch["exec"] != 0
        if bool(paid.any()):
            out["general_exec_adv_mean"] = float(ex_adv[paid].mean())
            out["general_exec_adv_std"] = float(ex_adv[paid].std()) if int(paid.sum()) > 1 else 0.0
            out["general_exec_paid"] = float(paid.float().mean())
    out["value_mean"] = float(batch["value"].mean())
    out["return_mean"] = float(ret.mean())
    out.update(critic_stats(batch, adv, ret))
    return out
