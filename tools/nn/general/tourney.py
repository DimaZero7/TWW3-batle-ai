"""The tourney of the learned group collection (GeneralConfig.groups "learned", assign.py; docs/ru/training/general.md
"Сбор групп: турнир"): how good a membership is, measured by the battle it leads to, the same start replayed.

A round: S starts (generated armies, the training's scheme, random training seeds; both roles and both sides: the
learner's side alternates every two starts, the bank alternates the attacker), each played V = len(ks) x copies
times - every K of the set `ks` (the number of groups our collection must make, Runtime b["K"]) `copies` times -
against a fixed opponent (the past slot: S27 by default), to the end (or limit_s). The copies of a start begin the
same: the start's randomisation (randomise.apply) is drawn once and copied to all its battles (share_start), and
they share the battle id (the state's "row": the simulator's hash draws are common). Every decision begins from the
same seed of torch's generator (torch.manual_seed(seed x 7919 + decision), the caller's state kept: fork_rng).

Two ways to place the variants:
  "batch" (default): one batch of S x V battles - one decision of all of them costs about one of S (the GPU's step is
          launches, not work); the networks' draws differ by place (the shared battle id keeps the simulator's common);
  "lanes": V batches of S battles stepped in lockstep (ability_branch.play_pair's way): variant v of start s sits at
          the same place of its batch, so it shares every draw with the other variants (the networks', the latency's)
          but the membership's: each copy has its own generator for the membership (General.assign_gen, seeded by the
          copy's number) - two lanes of one K and one copy are the same battle to the bit (the test). V times slower.

-> per battle: its start s, K, copy, the learner's side, its trade at the end (ability_branch.trade: (the enemy's gold
lost - our own) / budget), won, length; and the learner's plans' records (Runtime.record + the membership's
log-probability "lp_assign", "k_used", the battle's number, the plan's time, whether it is the deployment's).

The advantages (advantages()): a battle's membership advantage is its trade less the mean of the battles of the same
start and K (with 2 copies: half their difference); the K head's of a start is each K's mean trade less the mean over
the K played.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from tools.nn.train import league, randomise, reward, rollout, scenes

ROUND_SEEDS = 7_000_003        # a round's starts: seeds of generate.TRAIN_SEEDS drawn from the round's number


@dataclass
class Config:
    states: int = 32                         # S starts a round
    ks: tuple = (2, 3, 5, 8)                 # the K set every start is played with
    copies: int = 2                          # copies of each (start, K): the membership's baseline
    limit_s: float = 900.0                   # a battle's limit, s (not ended: its trade there)
    armies: str = "v3"                       # the generator's scheme (the training's: run.py --armies)
    max_units: int = 19
    spread: bool = True                      # the training's randomisation of the start (once a start, shared)
    fighters: bool = False                   # the fighters lab (run.py --fighters)
    mode: str = "batch"                      # "batch" or "lanes" (module doc)
    seed: int = 1
    check_every: int = 8                     # decisions between the checks whether every battle ended
    ks_random: int = 0                       # > 0: each round this many K drawn from 1..ks_max (instead of ks)
    ks_max: int = 10
    compile: object = None                   # rollout.Battles' compile (None: on CUDA)
    extra: dict = field(default_factory=dict)

    @property
    def V(self):
        return len(self.ks) * self.copies


def round_ks(cfg, rnd):
    """The K set of round rnd: cfg.ks, or cfg.ks_random distinct K of 1..ks_max drawn from the round's number."""
    if cfg.ks_random <= 0:
        return tuple(int(k) for k in cfg.ks)
    rng = np.random.default_rng(cfg.seed * 1_000_003 + rnd)
    return tuple(sorted(int(k) for k in rng.choice(np.arange(1, cfg.ks_max + 1), cfg.ks_random, replace=False)))


def round_seeds(cfg, rnd):
    """The S training seeds of round rnd (generate.TRAIN_SEEDS)."""
    rng = np.random.default_rng(ROUND_SEEDS + cfg.seed * 7919 + rnd)
    return [int(x) for x in rng.integers(0, 1_000_000_000, cfg.states)]


class _Rows:
    """A battle source that starts battle b as bank row idx[b] (rollout.Battles' source)."""

    def __init__(self, source, idx):
        self.bank = source.bank
        self.idx = idx

    def pick(self, want=None):
        return self.idx.to(self.bank.state.device)


def share_start(env, src):
    """Battle b starts as battle src[b] (in place, before the first decision): the state (the randomisation, the
    battle id), then the men, memories and health afresh - as rollout.Battles.start does."""
    st = env.st
    for k in list(st.u):
        st.u[k] = st.u[k][src].clone()
    for k in ("t", "attacker", "done", "winner", "lord_dead_s"):
        setattr(st, k, getattr(st, k)[src].clone())
    st.keys = [list(st.keys[int(i)]) for i in src.tolist()]
    if env.men is not None:
        env.men.open(st, env.params)
    from tools.nn.model import observation as ob
    state = st.observation()
    env.mem = {s: ob.start(state, env.setup, s) for s in (1, 2)}
    env.cmem = {s: ob.start(state, env.setup, s) for s in (1, 2)}
    env.health = reward.measure(st, env.weights.rout_share, env.weights.lord_rout)
    env.cur = None


def _battles(actor, enemy, cfg, source, params, device, state_of, k_of, side_of, cadence=None, men=None):
    """One rollout.Battles of the starts state_of [B] (bank rows of source), K k_of [B], the learner's side side_of
    [B] (numpy), the general's runtime made with b["K"] set."""
    from tools.nn.general.runtime import Runtime
    B = len(state_of)
    lay = league.Layout(np.zeros(B, dtype=int), np.asarray(side_of), np.full(B, league.CODE["past"]))
    idx = torch.as_tensor(np.asarray(state_of), dtype=torch.long)
    env = rollout.Battles(lay, (), device=device, params=params,
                          spread=randomise.Spread() if cfg.spread else randomise.NONE, seed=cfg.seed,
                          auto_reset=False, compile=cfg.compile, source=_Rows(source, idx), cadence=cadence,
                          men=men() if men is not None else None)
    first = {}
    src = torch.as_tensor([first.setdefault(int(s), b) for b, s in enumerate(state_of)], dtype=torch.long,
                          device=env.device)
    share_start(env, src)
    env.set_past(enemy)
    rt = Runtime(env, actor.cfg, actor.gcfg)
    env.general_rt = rt
    rt.b["K"][:] = torch.as_tensor(np.asarray(k_of), dtype=torch.long, device=env.device)
    return env, rt


def _collector(env, recs, lane):
    def on_plan(rt, rows, owner, deploy):
        sel = owner == 0
        if not bool(sel.any()):
            return
        rr = rows[sel]
        rec = rt.record(rr)
        rec["lp_assign"] = rt.r["lp_assign"][rr].clone()
        rec["k_used"] = rt.r["k_used"][rr].clone()
        rec["battle"] = rr % rt.B
        rec["lane"] = torch.full_like(rr, lane)
        rec["t_plan"] = env.st.t[rr % rt.B].float().clone()
        rec["deploy"] = torch.full_like(rr, int(bool(deploy)), dtype=torch.bool)
        recs.append(rec)
    return on_plan


def play(actor, enemy, cfg, rnd=0, device="cpu", cadence=None, ks=None, seeds=None, men=None):
    """One round (module doc) of the learner `actor` (an executor with groups "learned") against `enemy` (an actor).
    -> {"state", "K", "copy", "side", "trade", "won", "t" [B'] numpy (B' = S x V, in the order start, K, copy),
    "records" (the plans' records, concatenated; "battle" the B' index), "ks", "seeds", "decisions"}."""
    ks = tuple(ks or round_ks(cfg, rnd))
    seeds = list(seeds or round_seeds(cfg, rnd))
    S, C = len(seeds), cfg.copies
    V = len(ks) * C
    s_of = np.repeat(np.arange(S), V)
    k_of = np.tile(np.repeat(np.asarray(ks), C), S)
    c_of = np.tile(np.arange(C), S * len(ks))
    side_s = 1 + (np.arange(S) // 2) % 2
    side_of = side_s[s_of]
    dev = torch.device(device)
    cuda = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    recs = []
    gen = actor.general
    old_gen = getattr(gen, "assign_gen", None)
    lanes = []
    params = rollout.params_with_limit(cfg.limit_s)
    source = scenes.Generated(seeds, cfg.max_units, params, device, sequential=True, armies=cfg.armies)
    with torch.random.fork_rng(devices=cuda):
        if cfg.mode == "batch":
            env, rt = _battles(actor, enemy, cfg, source, params, device, s_of, k_of, side_of, cadence, men)
            rt.on_plan = _collector(env, recs, 0)
            lanes.append((env, None, np.arange(len(s_of))))
        elif cfg.mode == "lanes":
            for v in range(V):
                sel = np.nonzero(np.arange(len(s_of)) % V == v)[0]
                env, rt = _battles(actor, enemy, cfg, source, params, device, s_of[sel], k_of[sel], side_of[sel],
                                   cadence, men)
                rt.on_plan = _collector(env, recs, v)
                g = torch.Generator(device=dev).manual_seed(cfg.seed * 1009 + int(c_of[sel[0]]))
                lanes.append((env, g, sel))
        else:
            raise ValueError(f"tourney mode {cfg.mode!r}: 'batch' or 'lanes'")
        n_steps = int(cfg.limit_s / lanes[0][0].decision_s) + 2
        i = 0
        try:
            while i < n_steps:
                if i % cfg.check_every == 0 and all(bool(e.st.done.all()) for e, _, _ in lanes):
                    break
                for env, g, _ in lanes:
                    torch.manual_seed(cfg.seed * 7919 + rnd * 104729 + i)
                    gen.assign_gen = g
                    env.step(actor, None, False)
                i += 1
        finally:
            gen.assign_gen = old_gen
    from tools.nn.train.ability_branch import trade as side_trade
    out = {k: np.zeros(len(s_of)) for k in ("trade", "won", "t")}
    lane_of = np.zeros(len(s_of), dtype=int)
    pos_of = np.zeros(len(s_of), dtype=int)
    for li, (env, _, sel) in enumerate(lanes):
        side = torch.as_tensor(side_of[sel], device=env.device)
        out["trade"][sel] = side_trade(env.st.u, side, env.weights.rout_share).cpu().numpy()
        out["won"][sel] = (env.st.winner == side).float().cpu().numpy()
        out["t"][sel] = env.st.t.float().cpu().numpy()
        lane_of[sel], pos_of[sel] = li, np.arange(len(sel))
    rec = {k: torch.cat([r[k] for r in recs]) for k in recs[0]} if recs else {}
    if rec:
        # each record's battle as an index of the round's order (start, K, copy)
        back = {(int(lane_of[j]), int(pos_of[j])): j for j in range(len(s_of))}
        bl = [back[(int(a), int(b))] for a, b in zip(rec["lane"].tolist(), rec["battle"].tolist())]
        rec["battle"] = torch.as_tensor(bl, dtype=torch.long, device=rec["lane"].device)
    out.update(state=s_of, K=k_of, copy=c_of, side=side_of, records=rec, ks=ks, seeds=seeds, decisions=i)
    return out


def advantages(trade, state, K, ks):
    """(a [B] each battle's trade less the mean of the battles of its start and K, ak [S, len(ks)] each K's mean trade
    of a start less the mean over the K played, mean [S, len(ks)]) of the round's per-battle trade, start and K."""
    trade, state, K = np.asarray(trade, float), np.asarray(state), np.asarray(K)
    S = int(state.max()) + 1
    mean = np.zeros((S, len(ks)))
    a = np.zeros_like(trade)
    for s in range(S):
        for j, k in enumerate(ks):
            sel = (state == s) & (K == k)
            if sel.any():
                mean[s, j] = trade[sel].mean()
                a[sel] = trade[sel] - mean[s, j]
    return a, mean - mean.mean(1, keepdims=True), mean
