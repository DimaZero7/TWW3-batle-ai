"""Kiting in self-play, without a teacher (docs/en/training/training.md "Kiting in self-play").

A share of the ordinary self-play battles (league "self" and "past": the network on both sides, no script, no
teacher) is drawn uneven on purpose: one side (the SHOOTERS) has a lord and only fast missile units, the other
(the INFANTRY) a lord, infantry slower than every shooter by the margin below and a few missile units of its own -
equal gold, the normal map, deployment and distances (tools/nn/armies). Nothing forces a behaviour: the situation
itself pays the shooters for shoot - run back - shoot and the infantry for chasing and cutting off. Such a battle
need not be winnable for both sides (a campaign's "cannot win, take the most and live"): the gold-trade reward
teaches that already.

    army(rng) -> army description           one battle (scenario format; "kite_side": the shooters' side)
    armies(seeds) -> [(description, name)]
    Source(inner, kite, share, rows, ...)    a training source (scenes.Generated / drills.source.Mixed) with the kite
                                             battles beside its bank: a battle row of `rows` (self / past) that
                                             restarts takes a kite battle with probability `share`
    KiteBattles(...)                         rollout.Battles with the kiting meter (Meter: the log's "kite")
    check(...)                               the frame's check: on the same battles a kiting script for the shooters
                                             must trade clearly better than a naive one (stands and shoots)

    python -m tools.nn.train.kitesp --battles 128 --device cpu          # the check
    python -m tools.nn.train.kitesp --show 5 --battles 0                # rosters of 5 battles

Speed rule (SPEED_MARGIN, CHARGE_RULE): a shooter's run speed is at least (1 + SPEED_MARGIN) x every infantry
unit's run speed of the other side and above its charge speed (config/nn/units.json "speed"). SPEED_MARGIN 0.30:
a shooter that keeps its distance from a chaser stands (and shoots) a share 1 - v_chaser / v_shooter of the time
(it runs back what the chaser gained meanwhile) - 23 % at 1.3, enough for a volley every few seconds; below it the
chaser's charge burst (3.8 m/s against run 3.0 for Empire infantry, x 1.27) catches a shooter on the run, which the
charge rule also excludes. What it gives with our pools: Skaven Night Runners (5.4) and slave slingers (4.2) against
Empire infantry (2.8-3.0; flagellants 3.6 only against Night Runners alone), Night Runners against stormvermin
(3.8, charge 4.5); not Night Runners against clanrats (4.2, charge 4.8: x 1.29), not Empire militia (3.6) against
greatswords (2.8: x 1.29). The infantry side's own missile units are slower than the shooters' slowest (they do
not out-kite them).
"""
import argparse
import dataclasses
import json
import math
import sys
import time

import numpy as np
import torch

from tools import config as project

SPEED_MARGIN = 0.30          # a shooter runs at least (1 + this) x every enemy infantry unit's run speed
CHARGE_RULE = True           # ... and faster than its charge speed
EXTRA_MISSILE = 0.25         # the infantry side's own missile units: at most this share of its units (lord counted,
#                              at least one allowed: pools.Pool.cap), their weight a uniform draw below it
STYLE_ALPHA = 0.7            # Dirichlet concentration of a side's preferences among its allowed units (as generate.py)
MIN_BUDGET = 1500.0          # the battle's budget at least this: the shooters get a few units (a lord and one unit
#                              is not a kiting battle)
TRIES = 50                   # draws of a battle until both sides can spend the budget with at least one unit each
KITE_STREAM = 104729         # the kite battles' random stream: seed s gives another battle than generate.battle(s)
BANK_DEFAULT = 256           # kite battles in a training bank (run.py --kite-bank)
KITE_OPPONENTS = ("self", "past")   # the league opponents whose battles take kite battles (the network on both sides)


# --- the armies --------------------------------------------------------------------------------------

_SPEED = {}


def speed(key):
    """(run, charge) m/s of a unit (config/nn/units.json "speed")."""
    if not _SPEED:
        doc = json.loads((project.CONFIG_DIR / "nn" / "units.json").read_text(encoding="utf-8"))["units"]
        for k, p in doc.items():
            sp = p.get("speed") or {}
            _SPEED[k] = (float(sp.get("run", 0.0)), float(sp.get("charge", sp.get("run", 0.0))))
    return _SPEED[key]


def outruns(shooter_run, key, margin=SPEED_MARGIN, charge_rule=CHARGE_RULE):
    """Whether a shooter of run speed `shooter_run` outruns the infantry unit `key` by the rule."""
    run, charge = speed(key)
    return shooter_run >= (1.0 + margin) * run - 1e-9 and (not charge_rule or shooter_run > charge + 1e-9)


def pairs(pools, margin=SPEED_MARGIN, charge_rule=CHARGE_RULE):
    """[(shooters' faction, infantry's faction, threshold run speed, shooters [Unit], infantry [Unit], extra [Unit])]:
    every (faction pair, speed class) that makes a kite battle. A speed class: the shooters are the missile units of
    the first faction with run speed >= the threshold (one of their own speeds), the infantry the melee units of the
    second it outruns, extra the second's missile units slower than the threshold. A class is kept when both lists
    are non-empty."""
    out = []
    for fa, pa in pools.items():
        missile = [u for u in pa.units if u.missile]
        for thr in sorted({speed(u.key)[0] for u in missile}):
            shooters = [u for u in missile if speed(u.key)[0] >= thr - 1e-9]
            for fb, pb in pools.items():
                infantry = [u for u in pb.units if not u.missile and outruns(thr, u.key, margin, charge_rule)]
                extra = [u for u in pb.units if u.missile and speed(u.key)[0] < thr - 1e-9]
                if shooters and infantry:
                    out.append((fa, fb, thr, shooters, infantry, extra))
    return out


def _gen():
    from tools.nn.armies import generate
    return generate.default()


_MARKETS = {}


def _market(pool, units, caps):
    """A generate.Market of the pool restricted to `units`, with `caps` (cached)."""
    from tools.nn.armies import generate
    key = (pool.faction, tuple(u.key for u in units), tuple(sorted(caps.items())))
    if key not in _MARKETS:
        sub = dataclasses.replace(pool, units=tuple(units), caps=dict(caps), templates=())
        _MARKETS[key] = generate.Market(sub, generate.MAX_UNITS)
    return _MARKETS[key]


def army(rng, gen=None, classes=None):
    """A kite battle: an army description (tools/nn/sim/scenario.py format) with "kite_side" (the shooters' side,
    1 or 2) and "kite" (its draw: factions, threshold speed, budget, units)."""
    from tools.nn.armies import place as placement
    from tools.nn.sim import scenario
    gen = gen or _gen()
    classes = classes or pairs(gen.pools)
    assert classes, "no faction pair makes a kite battle (SPEED_MARGIN)"
    for _ in range(TRIES):
        fa, fb, thr, shooters, infantry, extra = classes[int(rng.integers(len(classes)))]
        pa, pb = gen.pools[fa], gen.pools[fb]
        m_a = _market(pa, shooters, {})
        m_b = _market(pb, infantry + extra, {"inf_ranged": EXTRA_MISSILE} if extra else {})
        lo, hi = gen.budget_bounds((fa, fb))
        share = gen.shares((fa, fb))
        lo = max(lo, MIN_BUDGET)
        if lo > hi:
            continue
        budget = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        b_a, b_b = budget * share[fa], budget * share[fb]
        if not (m_a.can_spend(b_a) and m_b.can_spend(b_b)):
            continue
        w_a = rng.dirichlet([STYLE_ALPHA] * len(shooters))
        m = float(rng.uniform(0.0, EXTRA_MISSILE)) if extra else 0.0
        w_b = np.concatenate([rng.dirichlet([STYLE_ALPHA] * len(infantry)) * (1.0 - m),
                              np.full(len(extra), m / max(1, len(extra)))])
        got_a = [shooters[i] for i in m_a.template_army(rng, b_a, w_a)]
        got_b = [(infantry + extra)[i] for i in m_b.template_army(rng, b_b, w_b)]
        if not got_a or not any(not u.missile for u in got_b):
            continue
        shooter_tag = "own" if rng.random() < 0.5 else "enemy"
        role = "attack" if rng.random() < 0.5 else "defend"
        arena = {k: v for k, v in gen.base.items() if k not in ("faction", "units", "description")}
        arena.update(name="kite", budget=round(budget), sides={},
                     note=f"kite battle: {fa} shooters against {fb} infantry, budget {round(budget)}")
        for tag in ("own", "enemy"):
            mine = tag == shooter_tag
            pool, units = (pa, got_a) if mine else (pb, got_b)
            arena["sides"][tag] = {"faction": pool.faction, "army": "kite_shooters" if mine else "kite_infantry",
                                   "budget": round(b_a if mine else b_b),
                                   "cost": pool.lord.cost + sum(u.cost for u in units),
                                   "units": placement.place(pool.lord, units, arena["deployment_m"])}
        desc = scenario.from_arena(arena, role)
        desc["kite_side"] = 1 if shooter_tag == "own" else 2
        desc["kite"] = {"shooters": fa, "infantry": fb, "threshold_ms": thr, "budget": round(budget), "role": role,
                        "units": {"shooters": [u.key for u in got_a], "infantry": [u.key for u in got_b]}}
        return desc
    raise ValueError(f"no kite battle in {TRIES} draws")


def armies(seeds):
    """[(army description, name)] of the kite battles of seeds (seed s: rng [s, KITE_STREAM])."""
    gen = _gen()
    classes = pairs(gen.pools)
    out = []
    for s in seeds:
        d = army(np.random.default_rng([int(s), KITE_STREAM]), gen, classes)
        out.append((d, f"kite_{int(s)}-side{d['kite_side']}-{d['kite']['role']}"))
    return out


# --- training: the source and the meter --------------------------------------------------------------

def rows_of(layout):
    """[B] bool: the battles of a league.Layout whose opponent takes kite battles (KITE_OPPONENTS)."""
    from tools.nn.train import league
    return np.isin(layout.opponent, [league.CODE[n] for n in KITE_OPPONENTS])


class _Bank:
    """Two banks (scenes.Bank) as one: the first's rows, then the second's (their tensors joined)."""

    def __init__(self, a, b):
        from tools.nn.sim import state as S
        from tools.nn.train import scenes
        assert a.N == b.N, f"banks of {a.N} and {b.N} slots"
        sa, sb = a.state, b.state
        self.state = S.State({k: torch.cat([v, sb.u[k]]) for k, v in sa.u.items()}, torch.cat([sa.t, sb.t]),
                             torch.cat([sa.attacker, sb.attacker]), torch.cat([sa.done, sb.done]),
                             torch.cat([sa.winner, sb.winner]), torch.cat([sa.lord_dead_s, sb.lord_dead_s]), sa.bounds,
                             [list(k) for k in sa.keys] + [list(k) for k in sb.keys])
        from tools.nn.model import observation as ob
        arrays = ob._Arrays(**{k: torch.cat([getattr(a.setup.arrays, k), getattr(b.setup.arrays, k)])
                               for k in scenes.LiveSetup.FIELDS})
        self.setup = scenes.LiveSetup(arrays, {s: torch.cat([a.setup.char[s], b.setup.char[s]]) for s in (1, 2)},
                                      list(a.setup.factions) + list(b.setup.factions))
        self.attacker = self.state.attacker
        self.armies = list(a.armies) + list(b.armies)
        self.names = list(a.names) + list(b.names)

    @property
    def M(self):
        return self.state.B

    @property
    def N(self):
        return self.state.N


class Source:
    """A training source with kite battles: `inner` (scenes.Generated or drills.source.Mixed) picks as before; a
    battle of `rows` ([B] bool: self / past, rows_of) that restarts takes instead, with probability `share`, a random
    kite battle (`kite`: [(description, name)], armies()). bank: the inner bank's rows, then the kite battles';
    kite_side [M]: the shooters' side of a bank row (0: not a kite battle)."""

    def __init__(self, inner, kite, share, rows, params, device, seed=0):
        from tools.nn.train import scenes
        self.inner, self.share = inner, float(share)
        H = inner.bank.N // 2
        extra = scenes.Bank([d for d, _ in kite], params, device, H, names=[n for _, n in kite])
        self.bank = _Bank(inner.bank, extra)
        dev = self.bank.state.device
        self.M_base, self.M_kite = inner.bank.M, extra.M
        self.kite_side = torch.tensor([0] * inner.bank.M + [int(d["kite_side"]) for d, _ in kite], device=dev)
        self.rows = torch.as_tensor(np.asarray(rows, dtype=bool), device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(int(seed))

    def pick(self, want=None):
        """[B] bank rows: the inner source's, a kite battle for a share of the rows that take them."""
        idx = self.inner.pick(want)
        if len(idx) != len(self.rows) or not self.M_kite or self.share <= 0:
            return idx
        dev = idx.device
        u = torch.rand(len(idx), generator=self.gen, device=dev)
        k = self.M_base + torch.randint(self.M_kite, (len(idx),), generator=self.gen, device=dev)
        take = self.rows & (u < self.share)
        if want is not None:
            take = take & (want == 0)
        return torch.where(take, k, idx)


END_M = 100.0                # a kiting episode ends when no slower melee enemy is within this (centres)


class Meter:
    """The kiting of the shooters' side in kite battles (the learner's units only), on the GPU without read-backs.

    An episode of a unit: from the moment it is in the kiting situation (drills/kiting.transfer: a standing missile
    unit, not a lord, with ammunition, a slower melee enemy within 60 m coming at it or in melee with it) until no
    slower melee enemy is within END_M, the unit falls (routs, dies) or the battle ends. A KITED episode: the unit
    ran back (transfer's applied: out of melee, moving away from the nearest such enemy at 1 m/s or more) and shot
    after that (its ammunition fell). Also the episodes it ran in, those in which it was caught in melee, the
    unit-decisions in the situation and those applied; per kite battle that ended, the shooters' gold trade
    ((the infantry's gold lost - the shooters') / budget, reward.gold_sides) and win."""

    COLS = ("episodes", "kited", "ran", "caught", "sit", "applied", "battles", "trade", "won")

    def __init__(self, B, N, device):
        z = (lambda: torch.zeros((B, N), dtype=torch.bool, device=device))
        self.ep, self.ran, self.shot, self.caught = z(), z(), z(), z()
        self.ammo = torch.zeros((B, N), device=device)
        self.acc = torch.zeros(len(self.COLS), device=device)

    def update(self, st, mine, finished):
        """After a decision's simulator steps (before finished battles restart): mine [B, N] the learner's units of
        the shooters' side of kite battles; finished [B] the battles that ended."""
        from tools.nn.train import drills as D
        from tools.nn.train.drills import kiting
        u = st.u
        sit, applied, caught = kiting.transfer(st)
        v = D.View(st)
        slower = u["run"][:, None, :] <= u["run"][:, :, None] - kiting.MIN_GAP_MS
        near = (v.foe & (u["range"] <= 0)[:, None, :] & slower & (v.d < END_M)).any(2)
        shot = u["a"] < self.ammo - 1e-6
        start = mine & sit & ~self.ep
        self.ran = torch.where(start, torch.zeros_like(self.ran), self.ran)
        self.shot = torch.where(start, torch.zeros_like(self.shot), self.shot)
        self.caught = torch.where(start, torch.zeros_like(self.caught), self.caught)
        self.ep = self.ep | start
        on = self.ep & mine
        self.shot = self.shot | (on & self.ran & shot)          # a volley after it ran (this decision's run: next one)
        self.ran = self.ran | (on & applied)
        self.caught = self.caught | (on & caught)
        end = self.ep & (~near | ~v.standing | ~mine | finished[:, None])
        f = (lambda m: m.float().sum())
        self.acc[:6] += torch.stack([f(end), f(end & self.ran & self.shot), f(end & self.ran), f(end & self.caught),
                                     f(mine & sit), f(mine & applied)])
        self.ep = self.ep & ~end
        self.ammo = u["a"].clone()

    def battles(self, st, kite_side, finished):
        """The kite battles that ended (finished [B], kite_side [B]: 0 not a kite battle): the shooters' trade, win."""
        from tools.nn.train import reward
        ks = kite_side.clamp(min=1)
        gold = reward.gold_sides(st.u)
        own = gold.gather(1, (ks - 1)[:, None]).squeeze(1)
        foe = gold.gather(1, (2 - ks)[:, None]).squeeze(1)
        trade = (foe - own) / reward.budget(st.u)
        m = (finished & (kite_side > 0)).float()
        self.acc[6:] += torch.stack([m.sum(), (m * trade).sum(), (m * (st.winner == kite_side).float()).sum()])

    def restarted(self, st, finished):
        """Finished battles restarted (their units are another battle's now)."""
        f = finished[:, None]
        for k in ("ep", "ran", "shot", "caught"):
            setattr(self, k, getattr(self, k) & ~f)
        self.ammo = torch.where(f, st.u["a"], self.ammo)

    def stats(self, reset=True):
        """{"episodes", "kited" (share of episodes), "ran", "caught" (shares), "applied" (share of the situation's
        unit-decisions), "battles", "trade", "won" (the shooters' mean)} since the last call."""
        a = dict(zip(self.COLS, self.acc.cpu().tolist()))
        if reset:
            self.acc.zero_()
        e, b = max(1.0, a["episodes"]), max(1.0, a["battles"])
        return {"episodes": int(a["episodes"]), "kited": round(a["kited"] / e, 3), "ran": round(a["ran"] / e, 3),
                "caught": round(a["caught"] / e, 3), "applied": round(a["applied"] / max(1.0, a["sit"]), 3),
                "battles": int(a["battles"]), "trade": round(a["trade"] / b, 3), "won": round(a["won"] / b, 3)}


def _battles_class():
    from tools.nn.train import rollout

    class KiteBattles(rollout.Battles):
        """rollout.Battles with the kiting meter (Meter) of a kite Source: kite_stats() for the log."""

        def start(self):
            super().start()
            self.kite_meter = Meter(self.B, self.N, self.device)
            self.kite_side = self._kite_side_of(self.bank_row)
            self.kite_meter.restarted(self.st, torch.ones(self.B, dtype=torch.bool, device=self.device))

        def _kite_side_of(self, rows):
            ks = getattr(self.source, "kite_side", None)
            return torch.zeros_like(rows) if ks is None else ks[rows.clamp(max=len(ks) - 1)]

        def _reset(self, finished):
            mine = rollout.learner_units(self.st.u, self.ctrl) & (self.st.u["side"] == self.kite_side[:, None]) & \
                (self.kite_side > 0)[:, None]
            self.kite_meter.update(self.st, mine, finished)
            self.kite_meter.battles(self.st, self.kite_side, finished)
            super()._reset(finished)
            if self.auto_reset:
                self.kite_side = torch.where(finished, self._kite_side_of(self.bank_row), self.kite_side)
                self.kite_meter.restarted(self.st, finished)

        def kite_stats(self, reset=True):
            return self.kite_meter.stats(reset)

    return KiteBattles


def __getattr__(name):
    if name == "KiteBattles":            # (rollout imports torch.compile machinery: loaded on first use)
        return _battles_class()
    raise AttributeError(name)


def text(s):
    """One log line of Meter.stats."""
    return (f"kite: {s['battles']} battles, shooters' trade {s['trade']:+.3f} won {s['won']:.2f}; {s['episodes']} "
            f"episodes, kited {s['kited']:.2f} ran {s['ran']:.2f} caught {s['caught']:.2f}; applied {s['applied']:.2f} "
            f"of the situation's unit-decisions")


# --- the frame's check ---------------------------------------------------------------------------------

def naive(st):
    """The shooters without the skill: every missile unit attacks the nearest enemy (closes to range, stands and
    shoots), the lord and melee units guard (drills/kiting.guard: hold until an enemy is within 60 m, then attack)."""
    from tools.nn.train import drills as D
    from tools.nn.train.drills import kiting
    v = D.View(st)
    i, d = v.nearest()
    o = kiting.guard(st, D.hold(st))
    return D.attack(o, v.standing & v.missile & (d < 1e9), i)


def kiter(st):
    """naive, and a missile unit with an enemy within kiting.RUN_M runs back until it is kiting.STOP_M away
    (kiting.kite's run-back: away from the enemies near, bent inward at the map's edge)."""
    from tools.nn.sim import orders as O
    k = _kite_orders(st)
    return O.merge(naive(st), k, k.kind == O.MOVE)


def _kite_orders(st):
    from tools.nn.train.drills import kiting
    return kiting.kite(st)


ENEMIES = ("ai_like", "chase")


def enemy_script(name):
    """The infantry side's script in the check: ai_like (the game-like line) or chase (drills/kiting.chase: every
    unit runs at the nearest missile unit)."""
    from tools.nn.train import opponents
    from tools.nn.train.drills import kiting
    return {"ai_like": opponents.ai_like, "chase": kiting.chase}[name]


EVAL_SEEDS = range(1_000_000_000, 1_001_000_000)


def play(script, enemy, descs, seed=0, device="cpu"):
    """The battles descs with `script` on the shooters' side and `enemy` on the other -> {"won", "trade", "seconds",
    "timeout", "ours"} per battle (as drills/verify.play: the training's randomised numbers, paired by seed)."""
    from tools.nn.sim import battle, scenario
    from tools.nn.sim.params import load
    from tools.nn.train import randomise, reward
    from tools.nn.train.drills import verify
    st = scenario.build(descs, device=device)
    ours = torch.tensor([d["kite_side"] for d in descs], device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    randomise.apply(st, torch.ones(st.B, dtype=torch.bool, device=device), randomise.Spread(), gen)
    params = load()
    verify.run(st, ours, script, enemy, params, battle.step)
    gold = reward.gold_sides(st.u).cpu().numpy()
    bud = reward.budget(st.u).cpu().numpy()
    o = ours.cpu().numpy()
    b = np.arange(st.B)
    t = st.t.cpu().numpy()
    return {"won": st.winner.cpu().numpy() == o, "trade": (gold[b, 2 - o] - gold[b, o - 1]) / np.maximum(bud, 1e-9),
            "seconds": t, "timeout": t >= params.limit_s - 1e-6, "ours": o}


PASS_TRADE = 0.05            # kiter - naive: the shooters' paired gold trade at least this, its 95 % interval above 0


def check(n=128, seed=0, device="cpu", enemies=ENEMIES, show=0):
    """{enemy: {"naive", "kiter": summaries, "paired": kiter - naive, "pass"}} on n kite battles of the eval seeds."""
    from tools.nn.train.drills import verify
    descs = [d for d, _ in armies(range(EVAL_SEEDS.start + seed, EVAL_SEEDS.start + seed + n))]
    for i, d in enumerate(descs[:show]):
        k = d["kite"]
        print(f"#{i} shooters side {d['kite_side']} ({k['shooters']}, budget {k['budget']}, {k['role']}): "
              f"{','.join(x.split('_', 3)[-1] for x in k['units']['shooters'])} | infantry ({k['infantry']}): "
              f"{','.join(x.split('_', 3)[-1] for x in k['units']['infantry'])}", flush=True)
    out = {}
    for e in enemies:
        res = {}
        for name, script in (("naive", naive), ("kiter", kiter)):
            t0 = time.time()
            res[name] = play(script, enemy_script(e), descs, seed, device)
            s = verify.summary(res[name])
            s["wall_s"] = round(time.time() - t0, 1)
            out.setdefault(e, {})[name] = s
            print(f"{e} {name}: " + json.dumps(s), flush=True)
        p = verify.paired(res["naive"], res["kiter"])
        att = np.array([d["attacker"] == d["kite_side"] for d in descs])
        p["shooters_attack"] = verify.paired(res["naive"], res["kiter"], att) if att.any() else None
        p["shooters_defend"] = verify.paired(res["naive"], res["kiter"], ~att) if (~att).any() else None
        out[e]["paired"] = p
        out[e]["pass"] = bool(p["trade"] >= PASS_TRADE and p["trade"] - p["trade_ci95"] > 0)
        print(f"{e} kiter - naive (paired, {p['battles']} battles): gold trade {p['trade']:+.4f} ± {p['trade_ci95']:.4f}, "
              f"win {p['win']:+.4f}; better in {p['better']:.2f}, worse in {p['worse']:.2f} -> "
              f"{'PASS' if out[e]['pass'] else 'FAIL'}", flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--battles", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enemy", default=",".join(ENEMIES), help="the infantry side's scripts, comma-separated")
    ap.add_argument("--show", type=int, default=0, help="print the first K battles' rosters")
    ap.add_argument("--out", help="write the results as json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.battles <= 0:
        for i, (d, name) in enumerate(armies(range(EVAL_SEEDS.start + args.seed, EVAL_SEEDS.start + args.seed
                                                    + args.show))):
            print(name, json.dumps(d["kite"]), flush=True)
        return
    res = check(args.battles, args.seed, args.device, tuple(args.enemy.split(",")), args.show)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
