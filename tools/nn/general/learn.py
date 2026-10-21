"""The general's training (PPO on two levels, docs/en/training/general.md "Training"; design: "Обучение").

* The executor (the unit network) learns as before (tools/nn/train/ppo.py): its task input is computed again in the
  update from the chunk's plan table (task_for: the general's cards with gradients, under the plans' locked membership
  and choices), so its loss reaches the group council, the group collection, the councils and the readers.
* The general's choices (mission kind, target, deployment place: discrete, sampled) learn by PPO from the plans that
  ended (update): each own group is an agent with its ratio of the plan's choices (the joint log-probability of its
  kind, target and place) - the probability now / a reference -, all groups of a plan share its advantage = the plan's
  return - the plan value's prediction at the plan (normalised over the update's plans). The reference (stale plans):
  a plan begun in this chunk was made with the weights the update starts from - its own log-probability; an older one
  (begun before the last update: ~40 % of the ended plans with 30 s plans and 64 s chunks) - the policy at the update's
  start (lp0), so its ratio starts at 1 and PPO's clip does not throw it away, and its lag behind the policy that made
  it is a separate truncated importance weight c = min(1, exp(lp0 - lp_plan)) on its group's advantage (V-trace's
  rho bar = 1: the data kept, the bias bounded). c = 1 for the fresh plans: the plain PPO bit for bit (records without
  "new": all fresh). + an entropy bonus on each head apart (entropy_kind, entropy_target, entropy_place: the kind's ~ln 10 and the
  target's ~ln 36 are not one sum); the plan value ("Оценка замысла") regresses the return. The epochs' KL stop is
  against the policy at the update's start (the plans' own log-probabilities are older: their gap is logged apart,
  kl_plan). No reward for carrying a mission out in the general's return: it is the battle's reward (v2) - the
  executor's bonus for it (LearnConfig.exec, run.py --general-exec W) goes into the units' own advantage only
  (ppo.update), so the general gains nothing by picking easy missions. The teacher's plans (LearnConfig.teacher) are
  not among the ended plans: their PPO is the general's own choices only.
* The teacher's imitation (LearnConfig.imitate W, run.py --general-imitate; down to 0 over --general-imitate-decay
  updates): the teacher's ended plans (runtime: batch["general"]["teach"], their inputs, the network's groups and the
  rule's kind and target where it gave one, "use") add W x the mean negative log-probability of the teacher's kind
  and target (the target's under the teacher's kind) over those groups to every minibatch step (imitate_term) - only
  the general's weights (its network alone; the executor is not in it). Logged before the steps: imitate_nll,
  imitate_acc_kind (the network's most likely kind is the teacher's), imitate_acc_target (its most likely target
  under the teacher's kind), imitate_rows / imitate_groups, imitate_w.
* The general's small execution bonus (LearnConfig.exec_general S, run.py --general-exec-general): each own group's
  advantage of a plan + S x its centred measure over the plan (runtime "exec_c": the mean of measure - its kind's
  running mean over the plan's decisions and the group's units, the executor's own centring: an easy kind has no gain,
  only "can this be done here") - in the normalised advantage's units (adv sd = 1). S = 0: the plain PPO bit for bit.
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
    exec: float = 0.0          # the executor's bonus for carrying out its missions: its weight W (0: measured and
    #                            logged, not paid; mission.py, runtime.execute)
    exec_decay: int = 0        # ... down to 0 linearly over this many updates of the chain (0: no decay)
    teacher: float = 0.0       # the share of the learner's battles whose missions the teacher gives (teacher.py)
    imitate: float = 0.0       # the teacher's imitation: its term's weight W now (run.py passes W after the decay; 0: off)
    imitate_decay: int = 0     # ... W down to 0 linearly over this many updates of the chain (run.py; 0: no decay)
    exec_general: float = 0.0  # the general's bonus for its groups' centred measure: S (in adv sd units; 0: off)

    def __post_init__(self):
        if not 0.0 <= self.branches <= 1.0 or self.branch_weight < 0:
            raise ValueError("--general-branches is a share in [0, 1], --general-branch-weight >= 0")
        if self.distinct < 0 or self.distinct_decay < 0:
            raise ValueError("--general-distinct and --general-distinct-decay are >= 0")
        if self.exec < 0 or self.exec_decay < 0:
            raise ValueError("--general-exec and --general-exec-decay are >= 0")
        if not 0.0 <= self.teacher <= 1.0:
            raise ValueError("--general-teacher is a share in [0, 1]")
        if self.imitate < 0 or self.imitate_decay < 0 or self.exec_general < 0:
            raise ValueError("--general-imitate, --general-imitate-decay and --general-exec-general are >= 0")


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


def kind_lp(d, group):
    """The kind's log-probability [E] of each row's group (d: the general's output)."""
    return d["lp_kind"].gather(1, group[:, None])[:, 0]


def stale(lp_plan, lp0, new):
    """(ref, w) of the ratio (module doc, "stale plans"): the plans begun in this chunk (new [M]) keep their own
    log-probability lp_plan and the weight 1; the older ones take the update start's lp0 and c = min(1,
    exp(lp0 - lp_plan)). lp_plan, lp0: [M] or [M, G]."""
    fresh = new.reshape(new.shape + (1,) * (lp_plan.dim() - 1))
    ref = torch.where(fresh, lp_plan, lp0)
    w = torch.where(fresh, torch.ones_like(lp0), torch.exp((lp0 - lp_plan).clamp(max=0.0)))
    return ref, w


def branch_term(gen, branch, clip):
    """The mission branches' term (module doc) on branch (branch.Brancher.finish's: "rec" [E, ..], "group", "kind",
    "lp", "adv" [E]; with "ref" and "w" [E] (update's stale plans): the ratio's reference and the importance weight,
    else the plan's own log-probability and 1) -> (loss: minus the mean clipped surrogate of the group's kind
    log-probability, KL against the reference, clip share)."""
    rec = branch["rec"]
    _, d = gen(rec, act={k: rec[k] for k in CHOICES})
    lp = kind_lp(d, branch["group"])
    ref = branch.get("ref", branch["lp"])
    ratio = torch.exp(lp - ref)
    a = branch["adv"] * branch["w"] if "w" in branch else branch["adv"]
    surr = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    with torch.no_grad():
        kl = float((ref - lp).mean())
        cl = float(((ratio - 1).abs() > clip).float().mean())
    return -surr.mean(), kl, cl


def imitate_term(gen, teach):
    """The teacher's imitation (module doc) on its plans' records teach ([E, ..] with "use" [E, G]) -> (loss: the mean
    negative log-probability of the teacher's kind + target over the groups it gave a mission that the network could
    choose, stats {nll, acc_kind, acc_target, groups}), or (None, {}) without such a group."""
    _, d = gen(teach, act={k: teach[k] for k in CHOICES})
    G = gen.cfg.places
    kind, target = teach["kind"], teach["target"]
    k_ok = d["kind_ok"].gather(2, kind[..., None])[..., 0]
    t_ok = d["target_ok"].gather(2, target[..., None])[..., 0]
    use = teach["use"] & teach["active"][:, :G] & k_ok & t_ok          # (a pair the network may not choose: none)
    n = use.sum()
    if not bool(n):
        return None, {}
    nll = -(d["lp_parts"][..., 0] + d["lp_parts"][..., 1])
    loss = (nll * use.float()).sum() / n
    with torch.no_grad():
        acc_k = ((d["kind_logp"].argmax(-1) == kind) & use).sum() / n
        acc_t = ((d["target_logits"].argmax(-1) == target) & use).sum() / n
    return loss, {"nll": float(loss.detach()), "acc_kind": float(acc_k), "acc_target": float(acc_t),
                  "groups": float(n)}


def update(actor, opt, ready, cfg=LearnConfig(), branch=None, teach=None):
    """PPO of the general on the ended plans `ready` (runtime.end_chunk's: records [M, ..] with "ret", their return); branch: the
    mission branches' prices of the chunk (or None: no term). -> stats (general_*): policy loss, entropy (all and per
    head: ent_kind, ent_target, ent_place), KL (against the update's start; kl_plan: against the plans' own), clip
    share (and apart: clip_fresh of the plans begun in the chunk, clip_old of the older ones; fresh_share; stale_w: the
    older ones' mean importance weight c, branch_stale_w: the branches'), the plan
    value's loss and explained variance (before the step), plans, groups a plan; with branches their
    term's loss, KL, clip share and rows; teach: the teacher's ended plans (the imitation's records, module doc; None:
    no term) - imitate_* (nll and accuracies before the steps, the term's mean over the steps); with exec_general S > 0
    exec_general_term (the mean |S x centred measure| of the groups measured, in adv sd) and exec_general_mean (signed)."""
    gen = actor.general
    params = list(gen.parameters())
    M = ready["lp"].shape[0]
    G = gen.cfg.places
    target = ready["ret"].float()
    base = ready["base"].float() if "base" in ready else ready["value"].float()   # the critic's (--general-return critic)
    raw = target - base
    adv = (raw - raw.mean()) / (raw.std() + 1e-8) if M > 1 else raw * 0
    # the general's execution bonus a group (module doc), in the normalised advantage's units: + S x its centred measure
    x_term = None
    if cfg.exec_general > 0 and "exec_c" in ready:
        x_ok = ready["exec_ok"] & ready["active"][:, :G]
        x_term = cfg.exec_general * torch.nan_to_num(ready["exec_c"].float()) * x_ok.float()      # [M, G]
    act_of = (lambda rec: {k: rec[k] for k in CHOICES})
    # the policy at the update's start (the KL stop's reference) and its gap to the plans' own log-probabilities
    with torch.no_grad():
        lp0 = torch.cat([gen(_rows(ready, mb), act=act_of(_rows(ready, mb)))[1]["lp"]
                         for mb in torch.arange(M, device=target.device).split(cfg.minibatch)])
        mine0 = ready["active"][:, :G].float()
        kl_plan = float(((ready["lp"] - lp0) * mine0).sum() / mine0.sum().clamp(min=1))
        new = ready["new"] if "new" in ready else torch.ones(M, dtype=torch.bool, device=target.device)
        ref, w_is = stale(ready["lp"], lp0, new)          # the ratio's reference, the stale plans' weight [M, G]
        old0 = mine0 * (~new)[:, None].float()
        stale_w = float((w_is * old0).sum() / old0.sum()) if float(old0.sum()) else None
    sums = {"pl": 0.0, "ent": 0.0, "ent_kind": 0.0, "ent_target": 0.0, "ent_place": 0.0, "vl": 0.0, "kl": 0.0,
            "clip": 0.0}
    use_b = branch is not None and cfg.branch_weight > 0 and len(branch["adv"]) > 0
    # the teacher's imitation: its numbers before the steps (logged whatever its weight), its term at weight imitate
    imit = {}
    if teach is not None and teach["lp"].shape[0] > 0:
        with torch.no_grad():
            _, imit = imitate_term(gen, teach)
    use_i = bool(imit) and cfg.imitate > 0
    if use_i:
        sums["imitate_loss"] = 0.0
    b_stale_w = None
    if use_b:
        sums.update(branch_loss=0.0, branch_kl=0.0, branch_clip=0.0)
        if "new" in branch:
            # the branches' plans the same way: one begun before this chunk (a deployment at the last step of the
            # chunk before, captured in this one) takes the update start's kind log-probability and its weight
            with torch.no_grad():
                rb = branch["rec"]
                lp0_b = torch.cat([kind_lp(gen(_rows(rb, mb), act=act_of(_rows(rb, mb)))[1], branch["group"][mb])
                                   for mb in torch.arange(len(branch["adv"]), device=target.device)
                                   .split(cfg.minibatch)])
            b_ref, b_w = stale(branch["lp"], lp0_b, branch["new"])
            branch = dict(branch, ref=b_ref, w=b_w)
            b_old = ~branch["new"]
            b_stale_w = float(b_w[b_old].mean()) if bool(b_old.any()) else None
    n, stop = 0, False
    cs = {"fresh": [0.0, 0.0], "old": [0.0, 0.0]}            # clipped groups, groups of the fresh and the old plans
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
            ratio = torch.exp((d["lp"] - ref[mb]) * m)
            if x_term is None:
                a = adv[mb][:, None] * w_is[mb]               # (x 1 for the fresh plans: the plain PPO bit for bit)
            else:
                a = (adv[mb][:, None] + x_term[mb]) * w_is[mb]
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
                sums["branch_loss"] += float(bl.detach())
                sums["branch_kl"] += bkl
                sums["branch_clip"] += bcl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if use_i:
                # the imitation's gradient added after the minibatch's backward (its graph is freed first): the two
                # graphs of the general never stand in memory together (g12: +1.4 GB peak, the card overflowed)
                E = teach["lp"].shape[0]
                part = teach if E <= cfg.minibatch else _rows(teach, torch.randperm(E, device=target.device)
                                                              [:cfg.minibatch])
                il, _ = imitate_term(gen, part)
                if il is not None:
                    (cfg.imitate * il).backward()
                    sums["imitate_loss"] += float(il.detach())
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
                out_c = ((ratio - 1).abs() > cfg.clip).float() * m
                sums["clip"] += float(out_c.sum() / cnt)
                if "new" in rec:
                    # the plans begun in this chunk (made with the weights the update starts from) and the older ones
                    for name, sel in (("fresh", rec["new"]), ("old", ~rec["new"])):
                        mm = m * sel[:, None].float()
                        cs[name][0] += float(out_c[sel].sum())
                        cs[name][1] += float(mm.sum())
            n += 1
    actor.eval()
    out = {f"general_{k}": v / max(1, n) for k, v in sums.items()}
    out["general_kl_plan"] = kl_plan
    for name, (c_n, g_n) in cs.items():
        if g_n:
            out[f"general_clip_{name}"] = c_n / g_n           # over all the steps' groups of those plans
    if "new" in ready:
        out["general_fresh_share"] = float(ready["new"].float().mean())
    if stale_w is not None:
        out["general_stale_w"] = stale_w                 # the stale plans' mean importance weight (their groups)
    if b_stale_w is not None:
        out["general_branch_stale_w"] = b_stale_w
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
    if imit:
        out.update({f"general_imitate_{k}": v for k, v in imit.items()})
        out["general_imitate_rows"] = float(teach["lp"].shape[0])
    if teach is not None:
        out["general_imitate_w"] = float(cfg.imitate)
    if x_term is not None:
        x_ok = ready["exec_ok"] & ready["active"][:, :G]
        k = float(x_ok.sum())
        out["general_exec_general_term"] = float(x_term.abs().sum() / k) if k else 0.0
        out["general_exec_general_mean"] = float(x_term.sum() / k) if k else 0.0
    out.update(task_in_norms(actor))
    return out


def task_in_norms(actor):
    """The executor's task input's weight norms (stats general_task_in_*): all, the group card's columns, the mission
    kind card's, the targets' geometry's - how much the executor listens to its task."""
    w = actor.task_in.weight.detach().float()
    D = actor.gcfg.card
    return {"general_task_in_w": float(w.norm()), "general_task_in_group": float(w[:, :D].norm()),
            "general_task_in_kind": float(w[:, D:2 * D].norm()), "general_task_in_geo": float(w[:, 2 * D:].norm())}
