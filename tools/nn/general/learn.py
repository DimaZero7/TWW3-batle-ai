"""The general's training (PPO on two levels, docs/en/training/general.md "Training"; design: "Обучение").

* The executor (the unit network) learns as before (tools/nn/train/ppo.py): its task input is computed again in the
  update from the chunk's plan table (task_for: the general's cards with gradients, under the plans' locked membership
  and choices), so its loss reaches the group council, the group collection, the councils and the readers.
* The general's choices (mission kind, target, deployment place: discrete, sampled) learn by PPO from the plans that
  ended (update): each own group is an agent with its ratio of the plan's choices - the probability now / the one the
  plan was made with (kept with the plan: a plan made before the last update is priced honestly) -, all groups of a
  plan share its advantage = the plan's return - the plan value's prediction at the plan (normalised over the update's
  plans); + an entropy bonus on each head apart (entropy_kind, entropy_target, entropy_place: the kind's ~ln 10 and the
  target's ~ln 36 are not one sum); the plan value ("Оценка замысла") regresses the return. The epochs' KL stop is
  against the policy at the update's start (the plans' own log-probabilities are older: their gap is logged apart,
  kl_plan). No reward for carrying a mission out: the return is the battle's reward (v2).
  With run.py --general-return critic the records carry "base" (the unit network's critic's value where the plan began)
  and the return is the critic's target (runtime.critic_target): the advantage is target - base (the critic explains
  ~0.995 of the return's spread); the plan value then only learns the same target beside it (its explained share
  logged).
* The bonus for missions that tell apart (LearnConfig.distinct, run.py --general-distinct W) is the executor's, paid in
  the rollout (runtime.py, distinct.py): not a term here.
* The mission branches (LearnConfig.branches = the share of the plans replayed, run.py --general-branches P; branch.py):
  the replayed plans' records with one group's kind price (batch["general"]["branch"]); every minibatch step adds
  branch_weight x the clipped surrogate of that group's kind log-probability under that advantage (branch_term; the
  same steps, the same KL stop). P = 0: no branches, nothing changes (test).
"""
from dataclasses import dataclass

import torch

from tools.nn.general.net import CHOICES


@dataclass(frozen=True)
class LearnConfig:
    clip: float = 0.2
    entropy_kind: float = 0.001    # the entropy bonus on the mission kind's choice (simplification 4: on) ...
    entropy_target: float = 0.001  # ... on the target's ...
    entropy_place: float = 0.001   # ... and on the deployment place's
    value: float = 0.5         # the plan value's loss weight
    epochs: int = 2
    minibatch: int = 512       # plans a minibatch
    max_grad: float = 0.5
    target_kl: float = 0.03    # the epochs stop when a minibatch's mean KL per group (against the policy at the
    #                            update's start) passes this (0: never)
    branches: float = 0.0      # the mission branches: the share of the plans replayed with another kind (0: off)
    branch_weight: float = 1.0  # ... their term's weight in the general's loss
    distinct: float = 0.0      # the executor's bonus for missions that tell apart: its weight W (0: off; distinct.py)
    distinct_decay: int = 0    # ... down to 0 linearly over this many updates of the chain (0: no decay)

    def __post_init__(self):
        if not 0.0 <= self.branches <= 1.0 or self.branch_weight < 0:
            raise ValueError("--general-branches is a share in [0, 1], --general-branch-weight >= 0")
        if self.distinct < 0 or self.distinct_decay < 0:
            raise ValueError("--general-distinct and --general-distinct-decay are >= 0")


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


def _rows(d, idx):
    return {k: v[idx] for k, v in d.items()}


def _ev(value, ret):
    v = ret.var()
    return float(1 - (ret - value).var() / v) if v > 1e-12 else 0.0


def pick(branch, idx):
    """The rows idx of the branches' dict (its "rec" too)."""
    return {k: ({n: x[idx] for n, x in v.items()} if isinstance(v, dict) else v[idx]) for k, v in branch.items()}


def branch_term(gen, branch, clip):
    """The mission branches' term (module doc) on branch (branch.Brancher.finish's: "rec" [E, ..], "group", "kind",
    "lp", "adv" [E]) -> (loss: minus the mean clipped surrogate of the group's kind log-probability, KL, clip share)."""
    rec = branch["rec"]
    _, d = gen(rec, act={k: rec[k] for k in CHOICES})
    g = branch["group"]
    lp = d["lp_kind"].gather(1, g[:, None])[:, 0]
    ratio = torch.exp(lp - branch["lp"])
    a = branch["adv"]
    surr = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    with torch.no_grad():
        kl = float((branch["lp"] - lp).mean())
        cl = float(((ratio - 1).abs() > clip).float().mean())
    return -surr.mean(), kl, cl


def update(actor, opt, ready, cfg=LearnConfig(), branch=None):
    """PPO of the general on the ended plans `ready` (runtime.end_chunk's: records [M, ..] with "ret", their return); branch: the
    mission branches' prices of the chunk (or None: no term). -> stats (general_*): policy loss, entropy (all and per
    head: ent_kind, ent_target, ent_place), KL (against the update's start; kl_plan: against the plans' own), clip
    share, the plan value's loss and explained variance (before the step), plans, groups a plan; with branches their
    term's loss, KL, clip share and rows."""
    gen = actor.general
    params = list(gen.parameters())
    M = ready["lp"].shape[0]
    G = gen.cfg.places
    target = ready["ret"].float()
    base = ready["base"].float() if "base" in ready else ready["value"].float()   # the critic's (--general-return critic)
    raw = target - base
    adv = (raw - raw.mean()) / (raw.std() + 1e-8) if M > 1 else raw * 0
    act_of = (lambda rec: {k: rec[k] for k in CHOICES})
    # the policy at the update's start (the KL stop's reference) and its gap to the plans' own log-probabilities
    with torch.no_grad():
        lp0 = torch.cat([gen(_rows(ready, mb), act=act_of(_rows(ready, mb)))[1]["lp"]
                         for mb in torch.arange(M, device=target.device).split(cfg.minibatch)])
        mine0 = ready["active"][:, :G].float()
        kl_plan = float(((ready["lp"] - lp0) * mine0).sum() / mine0.sum().clamp(min=1))
    sums = {"pl": 0.0, "ent": 0.0, "ent_kind": 0.0, "ent_target": 0.0, "ent_place": 0.0, "vl": 0.0, "kl": 0.0,
            "clip": 0.0}
    use_b = branch is not None and cfg.branch_weight > 0 and len(branch["adv"]) > 0
    if use_b:
        sums.update(branch_loss=0.0, branch_kl=0.0, branch_clip=0.0)
    n, stop = 0, False
    actor.train()
    for _ in range(cfg.epochs):
        if stop:
            break
        for mb in torch.randperm(M, device=target.device).split(cfg.minibatch):
            rec = _rows(ready, mb)
            c, d = gen(rec, act=act_of(rec))
            mine = rec["active"][:, :G]
            m = mine.float()
            cnt = m.sum().clamp(min=1)
            ratio = torch.exp((d["lp"] - rec["lp"]) * m)
            a = adv[mb][:, None]
            surr = torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
            pl = -(surr * m).sum() / cnt
            ent = (d["ent"] * m).sum() / cnt
            parts = (d["ent_parts"] * m[..., None]).sum((0, 1)) / cnt                # [3] kind, target, place
            vl = ((d["value"].float() - target[mb]) ** 2).mean()
            loss = (pl - cfg.entropy_kind * parts[0] - cfg.entropy_target * parts[1] - cfg.entropy_place * parts[2]
                    + cfg.value * vl)
            with torch.no_grad():
                kl = float(((lp0[mb] - d["lp"]) * m).sum() / cnt)
            if cfg.target_kl and kl > cfg.target_kl and n:
                stop = True                          # (a minibatch past the KL: no step on it, no more epochs)
                break
            if use_b:
                E = len(branch["adv"])
                part = branch if E <= cfg.minibatch else pick(branch, torch.randperm(E, device=target.device)
                                                                       [:cfg.minibatch])
                bl, bkl, bcl = branch_term(gen, part, cfg.clip)
                loss = loss + cfg.branch_weight * bl
                sums["branch_loss"] += float(bl)
                sums["branch_kl"] += bkl
                sums["branch_clip"] += bcl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
            if bool(torch.isfinite(gn)):
                opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                sums["pl"] += float(pl)
                sums["ent"] += float(ent)
                for j, k in enumerate(("ent_kind", "ent_target", "ent_place")):
                    sums[k] += float(parts[j])
                sums["vl"] += float(vl)
                sums["kl"] += kl
                sums["clip"] += float((((ratio - 1).abs() > cfg.clip).float() * m).sum() / cnt)
            n += 1
    actor.eval()
    out = {f"general_{k}": v / max(1, n) for k, v in sums.items()}
    out["general_kl_plan"] = kl_plan
    out["general_value_ev"] = _ev(ready["value"].float(), target)
    out["general_value_mse"] = float(((ready["value"].float() - target) ** 2).mean())
    out["general_plans"] = float(M)
    out["general_steps"] = float(n)
    out["general_return_mean"] = float(target.mean())
    out["general_return_std"] = float(target.std()) if M > 1 else 0.0
    out["general_adv_mean"] = float(raw.mean())                    # the advantage before its normalisation
    out["general_adv_std"] = float(raw.std()) if M > 1 else 0.0
    if "base" in ready:
        out["general_critic_ev"] = _ev(base, target)               # the unit critic's value at the plan vs the target
    out["general_groups"] = float(ready["active"][:, :G].float().sum(1).mean())
    if use_b:
        out["general_branch_rows"] = float(len(branch["adv"]))
    out.update(task_in_norms(actor))
    return out


def task_in_norms(actor):
    """The executor's task input's weight norms (stats general_task_in_*): all, the group card's columns, the mission
    kind card's, the targets' geometry's - how much the executor listens to its task."""
    w = actor.task_in.weight.detach().float()
    D = actor.gcfg.card
    return {"general_task_in_w": float(w.norm()), "general_task_in_group": float(w[:, :D].norm()),
            "general_task_in_kind": float(w[:, D:2 * D].norm()), "general_task_in_geo": float(w[:, 2 * D:].norm())}
