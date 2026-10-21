"""The mission branches (S28; docs/en/training/general.md "Mission branches"; run.py --general-branches P, off at 0).

Why: the general's mission kind learns from the plan's advantage (the reward over the plan against the critic): all
groups of a plan share one number, and one group's kind is a small part of it. A branch prices one group's kind
directly: "the same plan, with another kind for this group".

Capture (rollout.Battles.step -> Brancher.capture, after the decision's networks, before its simulator steps): the
learner's plans made for this decision (Runtime marks them "fresh": a re-plan of this step, a deployment at the end of
the step before) are taken with the chance `share`, within `cap` an update spread over the chunk's decisions and over
the battle kinds' slots of the branch batch (the ability branches' way: ability_branch.composition). Of a taken plan
one of its own groups is drawn (uniformly), and for every copy c another kind b_c (`alt`: "uniform" over the K - 1
others, or "policy": by the general's probabilities without the chosen kind). The snapshot is ability_branch.snapshot
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
    seed: int = 0

    def __post_init__(self):
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"general branches: share {self.share} not in (0, 1]")
        if self.alt not in ALTS:
            raise ValueError(f"general branches: alt {self.alt!r} not one of {ALTS}")
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
    entries: list = field(default_factory=list)       # per capture {"t", "side", "group", "kind", "lp", "alt", "w", "at"}
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


def swap_kind(env, actor, side, group, alt):
    """A branch env's plans with one group's mission kind changed: battle j's side side[j], its group group[j] (our
    place index; -1: none) gets kind alt[j] (-1: none) - the kind card of the group's units = the general's
    kinds[alt]."""
    rt = env.general_rt
    n = len(side)
    row = (side - 1) * n + torch.arange(n, device=side.device)
    ok = (group >= 0) & (alt >= 0)
    main = rt.r["main"][row]
    hit = (main == group[:, None]) & ok[:, None]                                    # [n, N]
    card = actor.general.kinds.detach()[alt.clamp(min=0)].to(rt.r["kc"].dtype)      # [n, D]
    rt.r["kc"][row] = torch.where(hit[..., None], card[:, None, :], rt.r["kc"][row])


def play_kinds(template, actor, snap, side, group, alt, copies, decisions, seed, price="trade", greedy=False):
    """The two branches of each snapshot battle (snap: b battles; side, group [b]: the plan's side and group, alt
    [b, copies]: copy c's other kind), `decisions` decisions (the snapshot's own first). -> (diff [b, copies]: price
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
            swap_kind(env, actor, s_rep, group[rep], alt.reshape(-1))
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

    def alternatives(self, logp, kind):
        """The copies' other kinds [k, copies] and the prices' weights [k] of plans whose group's kind log-probabilities
        are logp [k, K] and chosen kind `kind` [k] (Config.alt; signal.kind_advantage)."""
        k, K = logp.shape
        C = self.cfg.copies
        dev = logp.device
        if self.cfg.alt == "uniform":
            draw = torch.randint(0, K - 1, (k, C), generator=self.gen, device=dev)
            return gsignal.other_kind(draw, kind[:, None]), torch.ones(k, device=dev)
        p = logp.exp()
        pa = p.gather(1, kind[:, None])[:, 0]
        q = p.scatter(1, kind[:, None], 0.0)
        s = q.sum(1, keepdim=True)
        flat = torch.ones_like(q).scatter(1, kind[:, None], 0.0) / (K - 1)
        q = torch.where(s > 1e-8, q / s.clamp(min=1e-12), flat)
        return torch.multinomial(q, C, replacement=True, generator=self.gen), gsignal.policy_weight(pa)

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
        g = torch.multinomial(r["active"][nz, :G].float(), 1, generator=self.gen)[:, 0]    # one own group a plan
        kind = r["kind"][nz].gather(1, g[:, None])[:, 0]
        logp = r["kind_logp"][nz].gather(1, g[:, None, None].expand(-1, 1, K))[:, 0]       # [k, K]
        lp = logp.gather(1, kind[:, None])[:, 0]
        alt, w = self.alternatives(logp, kind)
        b = nz % B
        ub, inv = torch.unique(b, return_inverse=True)
        c.snaps.append(ab.snapshot(env, ub, info))
        c.recs.append(rt.record(nz))
        c.entries.append({"t": torch.full_like(nz, t), "side": nz // B + 1, "group": g, "kind": kind, "lp": lp,
                          "alt": alt, "w": w.float(), "at": c.n_battles + inv})
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
                "price": price}

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
            if fill:
                ps, _, pside, _ = ab.pad_snap(env, sub, 0, fill)
                sub = ab.concat([sub, ps])
                side = torch.cat([side, pside])
                group = torch.cat([group, torch.full_like(pside, -1)])
                alt = torch.cat([alt, torch.full((len(fill), K), -1, dtype=alt.dtype, device=dev)])
            self.calls += 1
            d, bg = play_kinds(env, actor, sub, side, group, alt, K, decisions, cfg.seed + self.calls, cfg.price,
                               cfg.greedy)
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
