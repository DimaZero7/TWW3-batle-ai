"""The mission branches (S28; docs/en/training/general.md "Mission branches"; run.py --general-branches P, off at 0).

Why: the general's mission kind learns from the plan's advantage (the reward over the plan against the critic): all
groups of a plan share one number, and one group's kind is a small part of it. A branch prices one group's kind
directly: "the same plan, with another kind for this group".

Capture (rollout.Battles.step -> Brancher.capture, after the decision's networks, before its simulator steps): the
learner's plans made for this decision (Runtime marks them "fresh": a re-plan of this step, a deployment at the end of
the step before) are taken with the chance `share`, within `cap` an update spread over the chunk's decisions and over
the battle kinds' slots of the branch batch (the ability branches' way: ability_branch.composition). Of a taken plan
one of its own groups is drawn (uniformly), and for every copy c another kind b_c, `alt`: "uniform" over the
candidates, or "policy": by the general's probabilities among them. The candidates (Config.kinds): "all" - every kind
the general could give the group at the plan (its kind_logp not masked), each with its own target: the group's target
when the kind takes it, else the natural one of its types (alt_targets: attack / flank / shoot the nearest enemy group,
hold the coarse tile the group stands on, screen our nearest other group, withdraw to our nearest other group or else
the tile nearest the point WITHDRAW_M behind it); "target" - only the kinds that take the group's target as it is
(config.MISSION_KINDS - an enemy group: attack, flank, shoot; our group: hold, screen, withdraw; a tile: hold,
withdraw): the atlas of g17r found it biased - withdraw was compared only with hold and screen, never with attack. The snapshot is ability_branch.snapshot
(the battle before the decision's simulator steps, the networks' memories after it, the decision's orders, the
general's plans in force); the plan's record (Runtime.record) and the kind's log-probability go with it.

Play (Brancher.finish, the chunk's end, before the update): every snapshot battle in `copies` copies, two batches -
"chosen" (the plan as it is) and "other" (the group's units' mission kind card = kinds[b_c]; the rest of the plan -
its groups, targets, places - the same) - `horizon_s` on. Open loop for the general: the plans are frozen in the
copies (Runtime.branch: no re-plans); the executors (and every other network, the scripts) play as in the batch, so a
kind changes the battle only through what the executor does with it. The first decision is the snapshot's own (its
orders, made under the chosen kind) in both: the swapped kind acts from the second decision. Common random numbers:
copy c of both sits at the same place of two batches of one shape, every decision of both starts from the same seed,
the copy's battle id is the same (ability_branch.play_pair's way): a swap to the same kind prices exactly 0 (test).
greedy: the networks take their most likely action (the copies then differ by the simulator's draws and the orders'
latency only: less of the battle's chaos in the price).

The price (budget share, from the planning side), "chosen" - "other": "trade" - the gold exchange at the horizon,
(the enemy's gold lost - its own) / budget (ability_branch.trade, the v2 reward's gold term); "reward" - the side's
reward summed over the horizon. Mean over the copies. The kind choice's advantage = signal.kind_advantage (x 1, or
x (1 - pi(a)) with "policy"; / the chunk's root mean square, within +-ADV_CLIP). learn.update adds
LearnConfig.branch_weight x the clipped surrogate of that group's kind log-probability (the plan's record computed
again: the gradient reaches the cards as the general's own PPO does).

Cost: 2 x horizon_s / decision_s sequential decisions a branch batch (a GPU step is launches, not work: a batch of
2 x copies x slots battles costs about one batch's step); with cap <= batch no more than one batch an update.
"""
import math
import time
from dataclasses import dataclass, field

import torch

from tools.nn.general import signal as gsignal
from tools.nn.sim import state as S
from tools.nn.train import ability_branch as ab
from tools.nn.train import reward

ALTS = ("uniform", "policy")
KINDS = ("all", "target")
WITHDRAW_M = 300.0          # a withdraw without another group of ours: to the tile nearest this far behind the group
PRICES = ("trade", "reward")
ZERO = ab.ZERO              # a price within this share of the budget either way is "even" (the log's better / worse)
ADV_FLOOR = ab.ADV_FLOOR    # the weighted prices' root mean square at least this (budget share)
ADV_CLIP = ab.ADV_CLIP      # the advantage within +-this


@dataclass(frozen=True)
class Config:
    share: float = 0.2          # the chance a plan of the learner is replayed (run.py --general-branches)
    copies: int = 4             # copies of each branch (common random numbers in pairs), each with its own other kind
    horizon_s: float = 60.0     # seconds of battle each branch plays
    cap: int = 64               # snapshots an update at most (0: no cap); spread over the chunk's decisions
    batch: int = 64             # snapshots a branch batch (x copies battles in each of the two branches)
    alt: str = "uniform"        # the other kind: "uniform" over the K - 1 others, "policy" by the general without a
    price: str = "trade"        # "trade": the gold exchange at the horizon; "reward": the side's reward over it
    greedy: bool = False        # the branches' networks take their most likely action, not a sample
    kinds: str = "all"          # the other kinds: "all" (each with its own target) or "target" (those taking the target)
    seed: int = 0

    def __post_init__(self):
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"general branches: share {self.share} not in (0, 1]")
        if self.alt not in ALTS:
            raise ValueError(f"general branches: alt {self.alt!r} not one of {ALTS}")
        if self.kinds not in KINDS:
            raise ValueError(f"general branches: kinds {self.kinds!r} not one of {KINDS}")
        if self.price not in PRICES:
            raise ValueError(f"general branches: price {self.price!r} not one of {PRICES}")
        if self.copies < 1 or self.horizon_s <= 0 or self.batch < 1 or self.cap < 0:
            raise ValueError("general branches: copies >= 1, horizon > 0, batch >= 1, cap >= 0")


@dataclass
class _Chunk:
    T: int = 0
    t: int = 0
    snaps: list = field(default_factory=list)
    recs: list = field(default_factory=list)          # the taken plans' records (Runtime.record)
    entries: list = field(default_factory=list)       # per capture {"t", "side", "group", "kind", "lp", "alt", "w", "at",
    #                                                   "new": the plan begun in this chunk (learn.update's stale plans)}
    n_battles: int = 0
    taken: int = 0
    room: dict = None                                 # {battle kind: slots left} with a cap
    plans: object = None                              # [] on the device: the learner's new plans seen this chunk


def side_reward(env, r, side):
    """[b] the reward r [R] (env.rows_learn's) of each battle's side (side [b], 1 / 2; 0 where it is not a learner
    row)."""
    n = len(side)
    pos = torch.full((2 * n,), -1, dtype=torch.long, device=r.device)
    pos[env.rows_learn] = torch.arange(len(env.rows_learn), device=r.device)
    p = pos[(side - 1) * n + torch.arange(n, device=r.device)]
    return torch.where(p >= 0, r[p.clamp(min=0)], torch.zeros_like(p, dtype=r.dtype))


def swap_kind(env, actor, side, group, alt, target=None, tgt=None):
    """A branch env's plans with one group's mission kind changed: battle j's side side[j], its group group[j] (our
    place index; -1: none) gets kind alt[j] (-1: none) - the kind card of the group's units = the executor's table's
    row alt (General.exec_table) - and, given, the target target[j] (-1: kept) with its world point tgt[j] (NaN for a
    group)."""
    rt = env.general_rt
    n = len(side)
    row = (side - 1) * n + torch.arange(n, device=side.device)
    ok = (group >= 0) & (alt >= 0)
    main = rt.r["main"][row]
    hit = (main == group[:, None]) & ok[:, None]                                    # [n, N]
    card = actor.general.exec_table().detach()[alt.clamp(min=0)].to(rt.r["kc"].dtype)     # [n, D]
    rt.r["kc"][row] = torch.where(hit[..., None], card[:, None, :], rt.r["kc"][row])
    # (the kind itself too: the task's masks follow it, taskmask.py)
    G = rt.r["kind"].shape[1]
    gk = (torch.arange(G, device=side.device) == group[:, None]) & ok[:, None]      # [n, G]
    rt.r["kind"][row] = torch.where(gk, alt[:, None].clamp(min=0), rt.r["kind"][row])
    if target is not None:
        gt = gk & (target >= 0)[:, None]
        rt.r["target"][row] = torch.where(gt, target[:, None].clamp(min=0), rt.r["target"][row])
        rt.r["tgt"][row] = torch.where(gt[..., None], tgt[:, None, :].to(rt.r["tgt"].dtype), rt.r["tgt"][row])


def alt_targets(rt, rows, group, alt):
    """(target [k, C], tgt [k, C, 2]) of the other kinds alt [k, C] of the plans of rows [k] and their group [k]
    (Config.kinds "all", module doc): the group's own target where the kind takes it, else the natural one of the
    kind's types; tgt the world point of a coarse tile (NaN for a group)."""
    r, G = rt.r, rt.G
    gen = rt.learner.general
    C = gen.C
    dev = rows.device
    k = len(rows)
    ar = torch.arange(k, device=dev)
    main = r["main"][rows]
    pos = r["in/pos"][rows].float()
    known = r["in/on"][rows] & (r["in/own"][rows] | (pos[..., -1] > 0.5))
    xy = pos[..., :2] * 500.0                                                       # net.POS_M: the side's frame, m
    mem = ((main[..., None] == torch.arange(2 * G, device=dev)) & known[..., None]).float()
    n = mem.sum(1)
    cent = torch.einsum("kng,knc->kgc", mem, xy) / n.clamp(min=1)[..., None]       # [k, 2G, 2]
    active = r["active"][rows] & (n > 0)
    from tools.nn.general import tiles as gtiles
    tc = gtiles.coarse_centres(rt.mcfg, gen.cfg.target_grid, dev)                   # [C, 2] frame, m
    cidx = gtiles.coarse_index(rt.mcfg.sectors, gen.cfg.target_grid, dev)
    c_ok = torch.zeros(k, C, device=dev).index_add_(1, cidx, r["in/tile_ok"][rows].float()) > 0
    me = cent[ar, group]                                                            # [k, 2]
    not_me = torch.ones(k, 2 * G, dtype=torch.bool, device=dev)
    not_me[ar, group] = False
    inf = float("inf")
    d_g = (cent - me[:, None]).norm(dim=-1).masked_fill(~(active & not_me), inf)  # [k, 2G]
    d_own, d_foe = d_g[:, :G], d_g[:, G:]
    near_own, near_foe = d_own.argmin(-1), G + d_foe.argmin(-1)
    has_own, has_foe = torch.isfinite(d_own.min(-1).values), torch.isfinite(d_foe.min(-1).values)
    d_t = (tc[None] - me[:, None]).norm(dim=-1).masked_fill(~c_ok, inf)           # [k, C]
    here = 2 * G + d_t.argmin(-1)
    back = me.clone()
    back[:, 0] = back[:, 0] - WITHDRAW_M
    behind = 2 * G + (tc[None] - back[:, None]).norm(dim=-1).masked_fill(~c_ok, inf).argmin(-1)
    from tools.nn.general.config import MISSION_KINDS
    by = {m.name: m.index for m in MISSION_KINDS}
    nat = torch.full((k, gen.cfg.kinds), -1, dtype=torch.long, device=dev)          # each kind's natural target
    for name in ("attack", "flank", "shoot"):
        if by[name] < gen.cfg.kinds:
            nat[:, by[name]] = torch.where(has_foe, near_foe, torch.full_like(near_foe, -1))
    if by["hold"] < gen.cfg.kinds:
        nat[:, by["hold"]] = here
    if by["screen"] < gen.cfg.kinds:
        nat[:, by["screen"]] = torch.where(has_own, near_own, torch.full_like(near_own, -1))
    if by["withdraw"] < gen.cfg.kinds:
        nat[:, by["withdraw"]] = torch.where(has_own, near_own, behind)
    cur = r["target"][rows].gather(1, group[:, None])[:, 0]                         # [k]
    pair = gen.pair_ok.to(dev)                                                      # [K, 2G + C]
    keep = pair[alt.clamp(min=0), cur[:, None].expand_as(alt)]                       # [k, C]
    target = torch.where(keep, cur[:, None].expand_as(alt), nat.gather(1, alt.clamp(min=0)))
    ti = target - 2 * G
    is_t = (ti >= 0) & (ti < C)
    tw = r["tile_w"][rows]                                                          # [k, C', 2] world
    w = tw.gather(1, ti.clamp(0, C - 1)[..., None].expand(-1, -1, 2))
    tgt = torch.where(is_t[..., None], w, torch.full_like(w, float("nan")))
    return target, tgt


def play_kinds(template, actor, snap, side, group, alt, copies, decisions, seed, price="trade", greedy=False,
               alt_t=None, alt_w=None):
    """The two branches of each snapshot battle (snap: b battles; side, group [b]: the plan's side and group, alt
    [b, copies]: copy c's other kind; alt_t [b, copies], alt_w [b, copies, 2]: its target and the target's world point,
    None: the target kept), `decisions` decisions (the snapshot's own first). -> (diff [b, copies]: price
    "chosen" - "other" (module doc), budget [b, copies] gold)."""
    b = len(snap["keys"])
    dev = template.device
    rep = torch.arange(b, device=dev).repeat_interleave(copies)
    big = ab.select(snap, rep)
    copy = torch.arange(copies, device=dev).repeat(b)
    big["u/" + S.BATTLE_ID] = big["u/" + S.BATTLE_ID] + ab.ID_STEP * (copy + 1)[:, None]
    s_rep = side[rep]
    envs = {}
    for name in ("chosen", "other"):
        env, parts, scripted = ab.branch_env(template, big)
        if name == "other":
            swap_kind(env, actor, s_rep, group[rep], alt.reshape(-1),
                      None if alt_t is None else alt_t.reshape(-1), None if alt_w is None else alt_w.reshape(-1, 2))
        if greedy:
            env._act = ab._greedy_act(env)
        envs[name] = (env, parts, scripted)
    total = {name: torch.zeros(len(rep), device=dev) for name in envs}
    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        for i in range(decisions):
            for name in ("chosen", "other"):
                env, parts, scripted = envs[name]
                torch.manual_seed(seed * 7919 + i)
                if i == 0:
                    r = env.replay(parts, scripted)
                else:
                    r = env.step(actor, greedy=greedy)["reward"]
                if price == "reward":
                    total[name] += side_reward(env, r.float(), s_rep)
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        for env, _, _ in envs.values():
            ab._unhook(env)
    if price == "trade":
        end = {name: ab.trade(envs[name][0].st.u, s_rep, template.weights.rout_share) for name in envs}
    else:
        end = total
    budget = reward.budget(envs["other"][0].st.u)
    return (end["chosen"] - end["other"]).reshape(b, copies), budget.reshape(b, copies)


def _cat(parts):
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


class Brancher:
    """Captures the learner's new plans during a chunk (capture) and prices their kind at its end (finish)."""

    def __init__(self, cfg=Config(), device="cpu"):
        self.cfg = cfg
        self.gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 7907)
        self.chunk = None
        self.stats = {}
        self.calls = 0
        self.copies = None                       # the last chunk's prices per copy [E, copies]
        self.comp = None                         # {battle kind: slots} of a branch batch (ability_branch.composition)
        self.kind_b = None                       # [B] each main battle's kind, on the host

    def slots(self):
        cfg = self.cfg
        return max(1, min(cfg.batch, cfg.cap) if cfg.cap else cfg.batch)

    def begin(self, env, T):
        if self.kind_b is None or len(self.kind_b) != env.B:
            self.comp = ab.composition(env.ctrl, self.slots())
            self.kind_b = ab.kinds_of_ctrl(env.ctrl).tolist()
        room = None
        if self.cfg.cap:
            rounds = max(1, math.ceil(self.cfg.cap / sum(self.comp.values())))
            room = {k: v * rounds for k, v in self.comp.items()}
        self.chunk = _Chunk(T=T, room=room, plans=torch.zeros((), device=env.device))

    def alternatives(self, logp, kind, valid=None):
        """The copies' other kinds [k, copies] and the prices' weights [k] of plans whose group's kind log-probabilities
        are logp [k, K] and chosen kind `kind` [k] (Config.alt; signal.kind_advantage). valid [k, K]: the kinds that
        take the group's target (its type: config.MISSION_KINDS; None: all) - the other kind is one of them (each row
        has one besides the chosen: capture's has_alt)."""
        k, K = logp.shape
        C = self.cfg.copies
        dev = logp.device
        if valid is None and self.cfg.alt == "uniform":
            draw = torch.randint(0, K - 1, (k, C), generator=self.gen, device=dev)
            return gsignal.other_kind(draw, kind[:, None]), torch.ones(k, device=dev)
        others = torch.ones(k, K, dtype=torch.bool, device=dev) if valid is None else valid.clone()
        others[torch.arange(k, device=dev), kind] = False
        flat = others.float() / others.float().sum(1, keepdim=True).clamp(min=1)
        if self.cfg.alt == "uniform":
            return torch.multinomial(flat, C, replacement=True, generator=self.gen), torch.ones(k, device=dev)
        p = logp.exp()
        pa = p.gather(1, kind[:, None])[:, 0]
        q = p * others.float()
        s = q.sum(1, keepdim=True)
        q = torch.where(s > 1e-8, q / s.clamp(min=1e-12), flat)
        # (the weight 1 - pi(chosen) is exact when the other kinds are all the rest; with the target's types only, it
        # is 1 - pi(chosen) of the kinds that take that target, the same formula over them)
        pv = pa / (pa + s[:, 0]).clamp(min=1e-12) if valid is not None else pa
        return torch.multinomial(q, C, replacement=True, generator=self.gen), gsignal.policy_weight(pv)

    def capture(self, env, info):
        rt = env.general_rt
        r = rt.r
        c = self.chunk
        B, G = rt.B, rt.G
        rows = torch.arange(2 * B, device=env.device)
        live = r["fresh"] & r["has"] & (r["owner"] == 0) & ~env.st.done[rows % B] & r["active"][:, :G].any(-1)
        r["fresh"].zero_()
        if c is None:                              # (a step outside collect: not captured)
            return
        t = c.t
        c.t += 1
        c.plans += live.sum()
        draw = torch.rand(2 * B, generator=self.gen, device=env.device)
        nz = (live & (draw < self.cfg.share)).nonzero().squeeze(1)              # reads the GPU
        if not len(nz):
            return
        if self.cfg.cap:
            left = max(0, self.cfg.cap - c.taken)
            m = min(left, max(1, math.ceil(self.cfg.cap / max(1, c.T))))
            order = torch.randperm(len(nz), generator=self.gen, device=self.gen.device).tolist()
            kb = [self.kind_b[i] for i in (nz % B).tolist()]
            keep = []
            for i in order:
                if len(keep) >= m:
                    break
                if c.room.get(kb[i], 0) > 0:
                    c.room[kb[i]] -= 1
                    keep.append(i)
            if not keep:
                return
            nz = nz[torch.tensor(keep, dtype=torch.long, device=nz.device)]
        K = r["kind_logp"].shape[-1]
        if self.cfg.kinds == "all":
            # every kind the general could give the group at the plan (not masked), each with its own target
            valid = r["kind_logp"][nz] > -1e6                                           # [k, G, K]
        else:
            # the kinds that take each group's target (its type; config.MISSION_KINDS): the swap keeps the target
            pair = rt.learner.general.pair_ok.to(env.device)                            # [K, 2G + C]
            valid = pair[:, r["target"][nz]].permute(1, 2, 0)                            # [k, G, K]
        chosen = torch.nn.functional.one_hot(r["kind"][nz], K).bool()
        can = r["active"][nz, :G] & (valid & ~chosen).any(-1)                            # a group with another kind
        keep = can.any(-1)
        if not bool(keep.all()):
            nz, can = nz[keep], can[keep]
            if not len(nz):
                return
            valid = valid[keep]
        g = torch.multinomial(can.float(), 1, generator=self.gen)[:, 0]                 # one own group a plan
        kind = r["kind"][nz].gather(1, g[:, None])[:, 0]
        logp = r["kind_logp"][nz].gather(1, g[:, None, None].expand(-1, 1, K))[:, 0]       # [k, K]
        lp = logp.gather(1, kind[:, None])[:, 0]
        alt, w = self.alternatives(logp, kind, valid.gather(1, g[:, None, None].expand(-1, 1, K))[:, 0])
        if self.cfg.kinds == "all":
            alt_t, alt_w = alt_targets(rt, nz, g, alt)
        else:
            alt_t = torch.full_like(alt, -1)
            alt_w = torch.full(alt.shape + (2,), float("nan"), device=alt.device)
        b = nz % B
        ub, inv = torch.unique(b, return_inverse=True)
        c.snaps.append(ab.snapshot(env, ub, info))
        c.recs.append(rt.record(nz))
        c.entries.append({"t": torch.full_like(nz, t), "side": nz // B + 1, "group": g, "kind": kind, "lp": lp,
                          "alt": alt, "w": w.float(), "at": c.n_battles + inv, "new": r["new"][nz].clone(),
                          "alt_t": alt_t, "alt_w": alt_w.float()})
        c.n_battles += len(ub)
        c.taken += len(nz)

    @torch.no_grad()
    def finish(self, env, actor):
        """Plays the chunk's branches -> {"rec": the plans' records [E, ..], "group", "kind", "lp" (the kind's
        log-probability at the plan), "adv" (the kind choice's advantage), "price" [E]} or None (nothing taken); the
        log's numbers in self.stats."""
        c, cfg = self.chunk, self.cfg
        self.chunk = None
        if c is None:
            return None
        t0 = time.time()
        plans = int(c.plans) if c.plans is not None else 0
        E = sum(len(x["side"]) for x in c.entries)
        st = {"plans": plans, "branches": E, "taken_share": round(E / max(1, plans), 4)}
        if not E:
            self.stats = st
            return None
        e = _cat(c.entries)
        snap = ab.concat(c.snaps)
        diff, budget, played, decisions = self._play(env, actor, snap, e)
        K = diff.shape[1]
        price = diff.mean(1)
        se = diff.std(1) / math.sqrt(K) if K > 1 else torch.zeros_like(price)
        adv = gsignal.kind_advantage(price, e["w"], ADV_FLOOR, ADV_CLIP)
        gold = price * budget.mean(1)
        st.update(battles=played, battle_s=round(played * decisions * env.decision_s),
                  price_mean=round(float(price.mean()), 5),
                  price_std=round(float(price.std()), 5) if E > 1 else 0.0,
                  price_rms=round(float(price.pow(2).mean().sqrt()), 5), price_se=round(float(se.mean()), 5),
                  gold_mean=round(float(gold.mean()), 1),
                  better=round(float((price > ZERO).float().mean()), 3),
                  worse=round(float((price < -ZERO).float().mean()), 3),
                  weight_mean=round(float(e["w"].mean()), 4))
        rel = ab.reliability(diff)
        if rel is not None:
            st["price_rel"] = round(rel, 3)
        st["seconds"] = round(time.time() - t0, 2)
        self.stats = st
        self.copies = diff
        return {"rec": _cat(c.recs), "group": e["group"], "kind": e["kind"], "lp": e["lp"], "adv": adv,
                "price": price, "new": e["new"]}

    def _play(self, env, actor, snap, e):
        """The entries' branches in batches of the fixed composition (ability_branch.Brancher._play's way) ->
        (diff [E, copies], budget [E, copies], battles played, decisions a branch)."""
        cfg = self.cfg
        dev = env.device
        at = e["at"]
        E, K = len(at), cfg.copies
        kinds = ab.kind_of(snap)[at].tolist()
        decisions = max(1, int(round(cfg.horizon_s / env.decision_s)))
        diff = torch.zeros(E, K, device=dev)
        budget = torch.ones(E, K, device=dev)
        comp = self.comp or ab.composition(env.ctrl, self.slots())
        todo = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in comp}
        assert sum(map(len, todo.values())) == E, "a snapshot of a battle kind the main batch does not have"
        played = 0
        ab._release(dev)
        while any(todo.values()):
            part, fill = [], []
            for k in sorted(comp):
                part += todo[k][:comp[k]]
                fill += [k] * (comp[k] - len(todo[k][:comp[k]]))
                todo[k] = todo[k][comp[k]:]
            ix = torch.tensor(part, dtype=torch.long, device=dev)
            sub = ab.select(snap, at[ix])
            side, group, alt = e["side"][ix], e["group"][ix], e["alt"][ix]
            alt_t, alt_w = e["alt_t"][ix], e["alt_w"][ix]
            if fill:
                ps, _, pside, _ = ab.pad_snap(env, sub, 0, fill)
                sub = ab.concat([sub, ps])
                side = torch.cat([side, pside])
                group = torch.cat([group, torch.full_like(pside, -1)])
                alt = torch.cat([alt, torch.full((len(fill), K), -1, dtype=alt.dtype, device=dev)])
                alt_t = torch.cat([alt_t, torch.full((len(fill), K), -1, dtype=alt_t.dtype, device=dev)])
                alt_w = torch.cat([alt_w, torch.full((len(fill), K, 2), float("nan"), device=dev)])
            self.calls += 1
            d, bg = play_kinds(env, actor, sub, side, group, alt, K, decisions, cfg.seed + self.calls, cfg.price,
                               cfg.greedy, alt_t, alt_w)
            diff[ix] = d[:len(part)]
            budget[ix] = bg[:len(part)]
            played += 2 * K * (len(part) + len(fill))
        ab._release(dev)
        return diff, budget, played, decisions


def text(st):
    """One log line of the branches' numbers (Brancher.stats)."""
    if not st:
        return ""
    out = f"mission branches {st.get('branches', 0)} of {st.get('plans', 0)} plans"
    if "price_mean" in st:
        out += (f", price {st['price_mean']:+.4f} sd {st['price_std']:.4f} (se {st['price_se']:.4f}, real share "
                f"{st.get('price_rel', float('nan')):.2f}), chosen better {st['better']:.2f} worse {st['worse']:.2f}, "
                f"{st['battles']} battles {st['seconds']:.1f} s")
    return out
