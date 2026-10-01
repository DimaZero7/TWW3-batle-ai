"""PPO with a centralised critic (MAPPO: Yu et al., 2021), docs/en/training/training.md.

* Every own unit is an agent with its own ratio new / old probability of its order; all units of
  a side share the side's advantage (one reward per side, one value of the critic that sees
  the whole field). Units that take no orders (dead, routing, padding) are left out of the policy
  loss and the entropy: their log-probability is zero and they are masked.
* The entropy bonus is on the order kind only. The full entropy counts the move point's 128 bins
  only when the unit moves, so a bonus on it pays the network for moving (and for switching
  points): in a first run the full entropy rose from 3.5 to 5 and order changes from 79 to 98
  a minute in 20 updates, before any battle had ended.
* GAE over the rollout; the value of the last state bootstraps; a finished battle cuts it.
* The memory (GRU) is trained through time: a minibatch is a set of whole chunks (T decisions of
  some battles); the actor runs its memory through each chunk from the memory stored when the
  chunk began, emptied where a new battle begins (recurrent PPO, as R2D2's stored state).
"""
from dataclasses import dataclass

import torch

from tools.nn.model import heads as hd


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.9997      # per decision (0.5 s): a horizon of ~3300 decisions (~28 min); a win
    #                            10 minutes away still counts 0.7
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 1
    minibatch: int = 4096      # decisions per minibatch (whole chunks: minibatch // T rows); bounds GPU memory
    lr: float = 1e-4
    target_kl: float = 0.05    # stop the epochs early when the mean per-unit KL passes this
    entropy: float = 0.01      # on the order kind's entropy (at most log 5 = 1.6)
    value: float = 0.5
    max_grad: float = 0.5
    anchor: float = 0.0        # weight of KL(policy || reference) on the order kind and target (0: off)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages and returns [T, R]. dones[t]: the battle ended at step t (no bootstrap past it)."""
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
    return torch.distributions.Categorical(logits=logits["kind"]).entropy()


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


def update(actor, critic, opt, batch, cfg=PPOConfig(), train_policy=True, reference=None):
    """Some epochs of minibatch updates on one rollout (batch from rollout.collect). -> stats.
    train_policy False: the critic only (a warm-up for a critic that starts from nothing while the
    actor already plays, e.g. after behaviour cloning). reference: a frozen actor (e.g. the cloned
    one) the policy is held near by cfg.anchor x KL, so PPO's noisy steps do not wash out what it
    started with while it looks for better.

    A minibatch is a set of learner rows with their whole chunk of T decisions: the actor runs its
    memory through the chunk from the memory the chunk began with (Actor.sequence)."""
    adv, ret = gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg.gamma, cfg.lam)
    T, R = adv.shape
    params = [p for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "entropy_all": 0.0, "kl": 0.0, "clip": 0.0,
             "anchor_kl": 0.0}
    n, stop = 0, False
    actor.train()
    critic.train()
    kinds = ("kind", "point", "target", "run")
    for _ in range(cfg.epochs):
        order = torch.randperm(R, device=adv.device)
        for idx in order.split(max(1, cfg.minibatch // T)):
            obs = {k: v[:, idx] for k, v in batch["obs"].items()}
            cobs = {k: _rows(v, idx) for k, v in batch["critic_obs"].items()}
            act = hd.Action(*(_rows(getattr(batch["action"], k), idx) for k in kinds))
            a = _rows(adv, idx)
            a = (a - a.mean()) / (a.std() + 1e-8)
            logits, _ = actor.sequence(obs, batch["h0"][idx], batch["reset"][:, idx])
            logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
            ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
            lp, ent = hd.log_prob(logits, act, ctrl)
            old = _rows(batch["lp"], idx)
            pl, clipped = policy_loss(lp, old, a, ctrl, cfg.clip)
            entropy = masked_mean(kind_entropy(logits), ctrl)
            if reference is not None and cfg.anchor:
                with torch.no_grad():
                    ref, _ = reference.sequence(obs, batch["h0"][idx], batch["reset"][:, idx])
                    ref = {k: v.reshape(-1, *v.shape[2:]) for k, v in ref.items()}
                anchored = anchor_kl(logits, ref, ctrl)
            else:
                anchored = torch.zeros((), device=adv.device)
            v = critic(cobs)
            vl = ((v - _rows(ret, idx)) ** 2).mean()
            policy_part = pl - cfg.entropy * entropy + cfg.anchor * anchored
            loss = cfg.value * vl + (policy_part if train_policy else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
            opt.step()
            with torch.no_grad():
                kl = masked_mean(old - lp, ctrl)
            for k, x in (("policy_loss", pl), ("value_loss", vl), ("entropy", entropy), ("kl", kl), ("clip", clipped),
                         ("entropy_all", masked_mean(ent, ctrl)), ("anchor_kl", anchored)):
                stats[k] += float(x.detach())
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
    out["reward"] = float(batch["reward"].sum(0).mean())
    out["value_mean"] = float(batch["value"].mean())
    out["return_mean"] = float(ret.mean())
    return out
