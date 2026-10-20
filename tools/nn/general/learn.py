"""The general's training (PPO on two levels, docs/en/training/general.md "Training"; design: "Обучение").

* The executor (the unit network) learns as before (tools/nn/train/ppo.py): its task input is computed again in the
  update from the chunk's plan table (task_for: the general's cards with gradients, under the plans' locked membership
  and choices), so its loss reaches the group council, the group collection, the councils and the readers.
* The general's choices (mission kind, target 1, target 2, deployment place: discrete, sampled) learn by PPO from the
  plans that ended (update): each own group is an agent with its ratio of the plan's choices, all groups of a plan share
  its advantage = the plan's return - the plan value's prediction at the plan (normalised over the update's plans);
  + an entropy bonus on the choices (entropy); the plan value ("Оценка замысла") regresses the return. No reward for
  carrying a mission out: the return is the battle's reward (v2).
* Simplification 4: the mission branches (counterfactual replays) and the bonus for missions that tell apart are not
  built: LearnConfig.branches and .distinct must be 0.
"""
from dataclasses import dataclass

import torch

from tools.nn.general.config import NotBuilt


@dataclass(frozen=True)
class LearnConfig:
    clip: float = 0.2
    entropy: float = 0.01      # the entropy bonus on the general's choices (simplification 4: on)
    value: float = 0.5         # the plan value's loss weight
    epochs: int = 2
    minibatch: int = 512       # plans a minibatch
    max_grad: float = 0.5
    target_kl: float = 0.03    # the epochs stop when a minibatch's mean KL per group passes this (0: never)
    branches: float = 0.0      # simplification 4: mission branches - not built (> 0 refused)
    distinct: float = 0.0      # simplification 4: the bonus for missions that tell apart - not built (> 0 refused)

    def __post_init__(self):
        if self.branches:
            raise NotBuilt("general mission branches (--general-branches): not built yet, keep 0")
        if self.distinct:
            raise NotBuilt("general bonus for distinct missions (--general-distinct): not built yet, keep 0")


def task_for(actor, table, obs):
    """(task [T, P, N, task_dim], task_on [T, P, N]) of the update's rows (obs: their [T, P, ..] observation with
    task_plan, task_geo, task_on): the plans' cards computed again from the table's records (with gradients when they
    are on), the geometry as stored."""
    plan = obs["task_plan"]
    T, P = plan.shape
    flat = plan.reshape(-1)
    ok = flat >= 0
    uniq, inv = torch.unique(flat.clamp(min=0), return_inverse=True)
    rec = {k: v[uniq] for k, v in table.items()}
    gen = actor.general
    c = gen.cards(rec)
    mix, kc = gen.unit_cards(c, rec["kind"], rec["main"])
    cards = torch.cat([mix, kc], -1)[inv]                                      # [T * P, N, 2D]
    N = cards.shape[1]
    cards = cards.reshape(T, P, N, -1)
    on = obs["task_on"] & ok.reshape(T, P, 1)
    geo = obs["task_geo"].to(cards.dtype)
    return torch.cat([cards, geo], -1) * on[..., None].to(cards.dtype), on


def _ev(value, ret):
    v = ret.var()
    return float(1 - (ret - value).var() / v) if v > 1e-12 else 0.0


def update(actor, opt, ready, cfg=LearnConfig()):
    """PPO of the general on the ended plans `ready` (runtime.end_chunk's: records [M, ..] with "target"). -> stats
    (general_*): policy loss, entropy, KL, clip share, the plan value's loss and explained variance (before the step),
    plans, groups a plan."""
    gen = actor.general
    params = list(gen.parameters())
    M = ready["lp"].shape[0]
    G = gen.cfg.places
    target = ready["target"].float()
    adv = target - ready["value"].float()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if M > 1 else adv * 0
    sums = {"pl": 0.0, "ent": 0.0, "vl": 0.0, "kl": 0.0, "clip": 0.0}
    n, stop = 0, False
    actor.train()
    for _ in range(cfg.epochs):
        if stop:
            break
        for mb in torch.randperm(M, device=target.device).split(cfg.minibatch):
            rec = {k: v[mb] for k, v in ready.items()}
            c, d = gen(rec, act={k: rec[k] for k in ("kind", "t1", "t2", "place")})
            mine = rec["active"][:, :G]
            m = mine.float()
            cnt = m.sum().clamp(min=1)
            ratio = torch.exp((d["lp"] - rec["lp"]) * m)
            a = adv[mb][:, None]
            surr = torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
            pl = -(surr * m).sum() / cnt
            ent = (d["ent"] * m).sum() / cnt
            vl = ((d["value"].float() - target[mb]) ** 2).mean()
            loss = pl - cfg.entropy * ent + cfg.value * vl
            with torch.no_grad():
                kl = float(((rec["lp"] - d["lp"]) * m).sum() / cnt)
            if cfg.target_kl and kl > cfg.target_kl and n:
                stop = True                          # (a minibatch past the KL: no step on it, no more epochs)
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
            if bool(torch.isfinite(gn)):
                opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                sums["pl"] += float(pl)
                sums["ent"] += float(ent)
                sums["vl"] += float(vl)
                sums["kl"] += kl
                sums["clip"] += float((((ratio - 1).abs() > cfg.clip).float() * m).sum() / cnt)
            n += 1
    actor.eval()
    out = {f"general_{k}": v / max(1, n) for k, v in sums.items()}
    out["general_value_ev"] = _ev(ready["value"].float(), target)
    out["general_value_mse"] = float(((ready["value"].float() - target) ** 2).mean())
    out["general_plans"] = float(M)
    out["general_steps"] = float(n)
    out["general_return_mean"] = float(target.mean())
    out["general_groups"] = float(ready["active"][:, :G].float().sum(1).mean())
    return out
