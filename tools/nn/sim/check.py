"""Checks of the simulator against the game (docs/en/training/simulator.md).

Every fair recorded run of the measurement arenas and of the mirror arena is replayed in the
simulator (tools/nn/sim/replay.py: recorded orders synchronised to contact, from the recorded start), the
simulated battle is written down per second like a recording (a gamedata.Battle) and measured
by the same code as the game (tools/nn/measure.py). Then the speed: many battles at once.

Each recorded battle is replayed COPIES times from starts moved by up to JITTER_M: the copies are the simulator's
forecast of the game, scored per family and quantity by tools/nn/simskill.py (the share of game values inside the
copies' 90 % interval, the CRPS skill score, 95 % bootstrap CIs over battles). The older counts (mechanics rows
within 20 %, same winner by the copies' majority) are printed too.

    python -m tools.nn.sim.check                    # all checks, CPU
    bash tools/nn/dock.sh tools.nn.sim.check        # in the container: CPU and GPU
    python -m tools.nn.sim.check --only mechanics   # a comma list of mechanics, battles, gates, speed

Prints a table; writes build/nn-sim/check.json (not in Git).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn import gamedata, measure, simskill
from tools.nn import scenario as arena_scenario
from tools.nn.sim import battle, replay, scenario
from tools.nn.sim.params import load as load_params

OUT = project.BUILD / "nn-sim" / "check.json"
TOLERANCE = 0.2
FIGHT_NEAREST = True             # replay: a unit in melee without a recorded target attacks the nearest enemy
PLANNER = ("attack", "defend")   # battles of CA's planner against the game's AI (not the network's own runs)
NET = "net"          # the network's battles against the game's AI on generated armies (the gate): reported apart
COPIES = 19          # replays of each recorded battle: the range of 19 copies is a 90 % prediction interval
CURVE_S = (60, 120, 180)   # share of HP lost this long after the first contact
JITTER_M = 2.0       # ... from starts moved by up to this much
CHUNK = 160          # simulated battles per batch, recordings of similar length together (CPU: a done battle
                     # still costs its steps until the batch's longest is over; the per-second recording grows
                     # with the batch)
GATES = project.BUILD / "nn-gate"   # gap cards (tools/ops/gapcard.py): <gate>/gapcard.json
BOOLS = gamedata.BOOL_FIELDS
FLOATS = gamedata.FLOAT_FIELDS


class Recorder:
    """Writes the batch down once a simulated second, like the game's recordings."""

    def __init__(self, st, every_s=1.0, fields=None):
        self.every = every_s
        self.next = 0.0
        self.t = []
        self.rows = []
        self.fields = tuple(fields) if fields else FLOATS + BOOLS + ("target", "fat")
        self.done_at = [None] * st.B
        self.take(st)

    def take(self, st):
        u = st.u
        snap = {k: u[k].detach().cpu().numpy().copy() for k in self.fields}
        self.rows.append(snap)
        self.t.append(st.t.detach().cpu().numpy().copy())

    def __call__(self, st):
        t = float(st.t.max())
        if t + 1e-6 >= self.next + self.every:
            self.next += self.every
            self.take(st)
        done = st.done.cpu().numpy()
        for b in np.flatnonzero(done):
            if self.done_at[b] is None:
                self.done_at[b] = len(self.rows)

    def battle(self, b, names, keys, side, slot_of, winner, own_ai, arena):
        """The simulated battle b as a gamedata.Battle in the recording's unit order."""
        end = self.done_at[b] or len(self.rows)
        rows = self.rows[:end + 1]
        idx = np.array(slot_of)
        f = {k: np.stack([r[k][b, idx] for r in rows]).astype(bool if k in BOOLS else float)
             for k in FLOATS + BOOLS if k in self.fields}
        if "fat" in self.fields:
            f["fat"] = np.stack([r["fat"][b, idx] for r in rows]).astype(float)
        if "target" in self.fields:
            tg = np.stack([r["target"][b, idx] for r in rows]).astype(np.int64)
            lut = np.full(max(max(slot_of), int(tg.max(initial=0))) + 2, -1, dtype=np.int64)   # slot -> index
            lut[np.array(slot_of)] = np.arange(len(slot_of))
            target = lut[tg]
        else:
            target = np.full((len(rows), len(slot_of)), -1, dtype=np.int64)
        t = np.array([r[b] for r in self.t[:end + 1]])
        return gamedata.Battle(run=f"sim-{b}", own_ai=own_ai, enemy_role="?", result={"winner": int(winner),
                               "status": "completed"}, t=t, f=f, target=target, arena=arena, names=names,
                               keys=keys, side=side)


_LOADED = {}


def load(run_dir):
    """gamedata.load, parsed once per process (the checks read the same recordings several times; a changed
    events.jsonl is parsed again)."""
    f = Path(run_dir) / "events.jsonl"
    st = f.stat()
    key = (str(Path(run_dir).resolve()), st.st_mtime_ns, st.st_size)
    if key not in _LOADED:
        _LOADED[key] = gamedata.load(run_dir)
    return _LOADED[key]


def config(run_dir):
    """The run's manifest config: arena and own_ai without parsing the recording."""
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["config"]


def factions_of(run_dir, arena):
    cfg = config(run_dir)
    fac = cfg.get("factions") or {}
    if not fac:
        base = arena_scenario.load_arena(arena)
        fac = {"own": base.get("faction"), "enemy": base.get("faction")}
    return fac


def ahead(st):
    """[B] the side with more of its starting health left in standing units (1 or 2)."""
    u = st.u
    stand = (u["men"] > 0) & ~u["gone"] & ~u["r"]
    share = [((u["hp_abs"] * (stand & (u["side"] == s))).sum(1) / (u["hp0"] * (u["side"] == s)).sum(1).clamp(min=1))
             for s in (1, 2)]
    return torch.where(share[0] >= share[1], 1, 2)


def jitter(seed, run, copy, n, jitter_m):
    """[2, n] start offsets (x, z) of the n recorded units of one copy of one recorded battle, up to jitter_m: drawn
    from (seed, run name, copy) alone, so a battle's copies do not depend on its batch (what else is in it, its
    slots per side)."""
    h = int.from_bytes(run.encode("utf-8"), "little") % (2 ** 31)
    gen = torch.Generator(device="cpu").manual_seed((seed * 1000003 + h * 101 + copy) % (2 ** 63))
    return (torch.rand((2, n), generator=gen) * 2 - 1) * jitter_m


LEAN = ("x", "z", "men", "hp", "mp", "ms", "a", "r", "s", "w", "m", "mv")   # what measure.py and this file read
WORKERS = 8          # processes (CPU): torch runs an operation on fewer than ~32 000 numbers on one thread, and
                     # most of a step's are that small, so one batch per process on one thread each


def units_a_side(run_dir):
    """The most units either side of a recorded run has (its manifest; the mirror arena's 7 when not listed)."""
    listed = config(run_dir).get("units") or {}
    return max(len(listed.get("own", [])), len(listed.get("enemy", []))) or len(gamedata.SLOTS)


def simulate(run_dirs, params=None, device="cpu", copies=1, jitter_m=0.0, seed=0, end_at_recording=False,
             chunk=None, zero_first=False, measure_fn=None, workers=None):
    """Replay recorded runs, `copies` times each (start places moved by up to jitter_m, so the copies differ;
    zero_first: copy 0 from the recorded places); returns [(game Battle, [sim Battle per copy], factions)], or with
    measure_fn [measure_fn(game, sims, factions)] (computed where the batch ran: the copies are not kept).
    end_at_recording: a simulated battle stops when its recording ends; if it is not over by
    then, the side with more standing health is taken as the winner (Battle.result "cut": True).
    Otherwise replay continues through its contact phases and holds the last recorded orders.
    chunk: simulated battles per batch, recordings alike in units a side and length together (a batch steps until
    its longest battle is over; the step's cost grows with the slots). workers: batches at once in processes of one
    thread (CPU; default WORKERS; 1 or a GPU: here, one after another). Each copy's start offsets are its own, and a
    battle in a batch does not see the others; but the batching changes the order of float sums (threads, slots a
    side), and a long battle can grow such a rounding into another course (as CPU against GPU): a single copy may
    differ, the copies' spread is what holds (7 battles x 3 copies in one batch against batches of one: 9 of 70
    numbers differed, most by less than 0.1 %)."""
    run_dirs = [Path(d) for d in run_dirs]
    hs = [units_a_side(d) for d in run_dirs]
    size = [(d / "events.jsonl").stat().st_size / max(hs[k], 1) for k, d in enumerate(run_dirs)]   # ~ length
    order = sorted(range(len(run_dirs)), key=lambda k: (hs[k], size[k]))
    workers = WORKERS if workers is None else workers
    pooled = workers > 1 and device == "cpu"
    per = max(1, int(chunk or CHUNK) // max(copies, 1))
    # A batch's cost ~ its longest recording x (slots a side + 2)^2 x its battles; in processes, batches of about
    # equal cost, three a worker (the costliest go first), each at most `per` recordings.
    cost = [(hs[k] + 2) ** 2 * size[k] * copies for k in range(len(run_dirs))]
    target = sum(cost) / (3 * workers) if pooled else float("inf")
    parts, part = [], []
    for k in order:
        if part and (len(part) >= per or (hs[k] + 2) ** 2 * size[k] * copies * (len(part) + 1) > target):
            parts.append(part)
            part = []
        part.append(k)
    if part:
        parts.append(part)
    jobs = []
    for part in parts:
        jobs.append((part, [str(run_dirs[k]) for k in part], copies, jitter_m, seed, end_at_recording,
                     max(hs[k] for k in part), zero_first, measure_fn, device, params))
    jobs.sort(key=lambda j: -j[6] ** 2 * max(size[k] for k in j[0]))         # the costliest first
    pooled = pooled and len(jobs) > 1
    out = {}
    t0 = time.perf_counter()
    if pooled:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(min(workers, len(jobs)))
        results = pool.imap_unordered(_work, [j[:-1] + (None,) for j in jobs])  # params: each loads its own
    else:
        results = map(_work, jobs)
    for n, got in enumerate(results, 1):
        out.update(got)
        if n % max(1, len(jobs) // 8) == 0 or n == len(jobs):
            print(f"  [{n} of {len(jobs)} batches (up to {per} recordings x {copies}), "
                  f"{time.perf_counter() - t0:.0f} s]", flush=True)
    if pooled:
        pool.close()
        pool.join()
    return [out[k] for k in range(len(run_dirs))]


def _work(job):
    """One batch: simulate, then measure_fn per recording (or the Battles) -> {index: result}."""
    idx, run_dirs, copies, jitter_m, seed, end_at_recording, H, zero_first, measure_fn, device, params = job
    if params is None:                  # a worker process: one thread
        torch.set_num_threads(1)
    params = params or load_params()
    games = [load(d) for d in run_dirs]
    armies, facs = [], []
    for d, g in zip(run_dirs, games):
        fac = factions_of(d, g.arena)
        army = scenario.from_recording(g, d)
        army["sides"][1]["faction"] = fac.get("own")
        army["sides"][2]["faction"] = fac.get("enemy")
        armies.append(army)
        facs.append(fac)
    sims = _simulate_batch(games, armies, params, device, copies, jitter_m, seed, end_at_recording, H, zero_first,
                           fields=LEAN if measure_fn else None)
    return {k: (measure_fn(g, s, f) if measure_fn else (g, s, f)) for k, g, s, f in zip(idx, games, sims, facs)}


def _simulate_batch(games, armies, params, device, copies, jitter_m, seed, end_at_recording, H, zero_first,
                    fields=None):
    """simulate() for one batch: [[sim Battle per copy] per game]."""
    batch = [a for a in armies for _ in range(copies)]
    st = scenario.build(batch, params, device=device, per_side=H)
    slot_maps = []
    for a, g in zip(armies, games):
        m = scenario.slots(a, H)
        slot_maps.append([m[n] for n in g.names])
    if jitter_m:
        noise = torch.zeros((st.B, 2, st.N))
        for k, (g, slot_of) in enumerate(zip(games, slot_maps)):
            for c in range(copies):
                if not (zero_first and c == 0):
                    noise[k * copies + c][:, slot_of] = jitter(seed, g.run, c, len(slot_of), jitter_m)
        st.u["x"] = st.u["x"] + noise[:, 0].to(device)
        st.u["z"] = st.u["z"] + noise[:, 1].to(device)
    rows = []
    for a, g, slot_of in zip(armies, games, slot_maps):
        width = {x["name"]: x.get("width") for side in (1, 2) for x in a["sides"][side]["units"]}
        # who leaves melee on a far recorded point: missile units, and the network's own units (side 1
        # of its runs: their point is the network's move order; replay.recorded_orders)
        missile = [bool((params.units.get(k) or {}).get("missile")) for k in g.keys] if g.keys else [False] * len(g.names)
        leavers = [bool(m) or (g.own_ai == NET and int(s) == 1) for m, s in zip(missile, g.side)]
        # the network's units in melee without a recorded target are under its HOLD (replay.recorded_orders)
        nearest = [FIGHT_NEAREST and not (g.own_ai == NET and int(s) == 1) for s in g.side]
        spacing = [params.spacing_of(k) for k in g.keys] if g.keys else params.sim["formation"]["spacing_m"]
        orders = replay.recorded_orders(g, slot_of, 2 * H, [width.get(n) for n in g.names], spacing, nearest,
                                        params.sim["contact"].get("leave_m", 0.0), leavers)
        rows.extend([orders] * copies)
    rec = Recorder(st, fields=fields)
    ends = torch.tensor([float(g.t[-1]) for g in games for _ in range(copies)], device=device)
    cut = torch.zeros(st.B, dtype=torch.bool, device=device)

    def record(s):
        if end_at_recording:
            over = ~s.done & (s.t >= ends - 1e-6)
            if bool(over.any()):
                s.winner = torch.where(over, ahead(s), s.winner)
                s.done = s.done | over
                cut.copy_(cut | over)
        rec(s)
    policy = replay.Replay(rows, device=device, grace_s=0.0 if end_at_recording else 1e9)
    battle.run(st, policy, params, record=record)
    winners = st.winner.cpu().numpy()
    cuts = cut.cpu().numpy()
    out = []
    for k, g in enumerate(games):
        sims = [rec.battle(k * copies + c, g.names, g.keys, g.side, slot_maps[k], winners[k * copies + c], g.own_ai,
                           g.arena) for c in range(copies)]
        for c, sim in enumerate(sims):
            sim.result["cut"] = bool(cuts[k * copies + c])
        out.append(sims)
    return out


def rel_err(sim, game):
    if sim is None or game is None or not np.isfinite(sim) or not np.isfinite(game) or game == 0:
        return None
    return (sim - game) / abs(game)


def mean(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


DUPLICATE_ROWS = ("implied_reload_s",)   # 1 / shots_per_man_per_s: the same number twice in the old count


def won(winner, side=1):
    """1.0 if `side` won, 0.0 if the other did, None for no winner."""
    return {side: 1.0, 3 - side: 0.0}.get(int(winner or 0))


def pair_quantities(m):
    """What a melee pair is scored on (tools/nn/measure.melee_pair; None: not there). The loser's rout time is the
    fight's length (the fight ends at the first rout): only fight_s; who routed or wavered at all is who lost (in
    every recorded pair): only own_won."""
    if not m.get("contact"):
        return {}
    q = {"fight_s": m.get("fight_s"), "own_won": won(m.get("winner_side"))}
    for u in m["units"]:
        tag = "own" if u["side"] == 1 else "enemy"
        q[f"{tag}.steady_hp_per_s"] = u.get("steady_hp_per_s")
        q[f"{tag}.charge_hp_lost"] = u.get("charge_hp_lost")
        q[f"{tag}.waver_after_s"] = u.get("waver_after_s")
    return q


def pooled_hit_rate(m):
    """The implied hit rate over every counted shot of a missile run (all distance bins: game and simulator need not
    have the same last bin)."""
    bins = m.get("by_distance") or []
    shots = sum(b["shots"] for b in bins)
    return sum(b["implied_hit_rate"] * b["shots"] for b in bins) / shots if shots else None


def missile_quantities(m):
    """What a shooting run is scored on (tools/nn/measure.missile_run)."""
    if not m.get("shooting"):
        return {}
    q = {k: m.get(k) for k in ("first_shot_after_stop_s", "shots_per_man_per_s", "hp_per_s_while_shooting")}
    q["hit_rate"] = pooled_hit_rate(m)
    for ev in ("waver", "rout"):
        t = m.get(f"target_{ev}_after_first_shot_s")
        q[f"target.{ev}ed" if ev == "waver" else "target.routed"] = float(t is not None)
        q[f"target.{ev}_after_first_shot_s"] = t
    return q


def cases_of(run, game_q, sims_q):
    """simskill cases of one recorded battle: every quantity the game has, against the copies that have it."""
    return [simskill.case(run, k, v, [s.get(k) for s in sims_q]) for k, v in game_q.items()]


_PASSPORTS = {}


def passports():
    if "p" not in _PASSPORTS:
        _PASSPORTS["p"] = measure.passports()
    return _PASSPORTS["p"]


def measure_mechanics(g, sims, fac):
    """simulate()'s measure_fn of a pair or shooting run: the game's and copy 0's measure dicts, the cases of copies
    1.., and the evidence rows (first strike, distance bins)."""
    p = passports()
    k = measure.kind(g.arena)
    f, qf = (measure.melee_pair, pair_quantities) if k == "pair" else (measure.missile_run, missile_quantities)
    gm, sms = f(g, p), [f(s, p) for s in sims]
    out = {"arena": g.arena, "game": gm, "copy0": sms[0], "cases": cases_of(g.run, qf(gm), [qf(m) for m in sms[1:]]),
           "first_strike": [], "bins": []}
    if k == "pair" and gm.get("contact") and sms[0].get("contact"):
        for ug, us in zip(gm["units"], sms[0]["units"]):
            out["first_strike"].append({"run": g.run, "arena": g.arena, "unit": ug["slot"],
                                        "game": ug.get("contact_s_hp_lost"), "sim": us.get("contact_s_hp_lost")})
    if k == "missile":
        last = lambda m: ((m.get("by_distance") or [{}])[-1].get("distance_m"))
        out["bins"].append({"run": g.run, "arena": g.arena, "game_last_bin": last(gm), "sim_last_bin": last(sms[0]),
                            "game_bins": [x["distance_m"] for x in gm.get("by_distance") or []],
                            "sim_bins": [x["distance_m"] for x in sms[0].get("by_distance") or []]})
    return out


def mechanics(params=None, device="cpu", copies=COPIES, jitter_m=JITTER_M):
    """The melee pairs and the shooting runs: game against simulator, per measured number. Copy 0 replays the
    recorded start as it is (the old per-arena rows, means of the runs); copies 1..copies from starts moved by up
    to jitter_m are the forecast (cases for simskill). -> {"rows", "cases", "hit_bins"}."""
    from tools.nn.sim import check as here          # measure_fn by its importable name (worker processes)
    params = params or load_params()
    runs = [d for d in gamedata.runs() if measure.kind(config(d).get("arena", "arena")) in ("pair", "missile")]
    measured = simulate(runs, params, device, copies + 1, jitter_m, end_at_recording=False, zero_first=True,
                        measure_fn=here.measure_mechanics)
    by_arena, cases, bins, first_strike = {}, [], [], []
    for m in measured:
        by_arena.setdefault(m["arena"], []).append((m["game"], m["copy0"]))
        cases += m["cases"]
        first_strike += m["first_strike"]
        bins += m["bins"]
    rows = []
    for arena, items in sorted(by_arena.items()):
        k = measure.kind(arena)
        if k == "pair":
            def pick(m, i, key):
                if not m.get("contact"):
                    return None
                return m["units"][i].get(key)
            metrics = [("fight_s", lambda m: m.get("fight_s"))]
            for i in range(2):
                name = items[0][0]["units"][i]["slot"] if items[0][0].get("contact") else str(i)
                for key in ("steady_hp_per_s", "charge_hp_lost", "waver_after_s", "rout_after_s"):
                    metrics.append((f"{name}.{key}", lambda m, i=i, key=key: pick(m, i, key)))
            metrics.append(("winner_side", lambda m: m.get("winner_side")))
        else:
            metrics = [(key, lambda m, key=key: m.get(key)) for key in (
                "first_shot_after_stop_s", "shots_per_man_per_s", "implied_reload_s", "hp_per_s_while_shooting",
                "target_waver_after_first_shot_s", "target_rout_after_first_shot_s")]
            metrics.append(("hit_rate", lambda m: (m.get("by_distance") or [{}])[-1].get("implied_hit_rate")))
        for name, get in metrics:
            game = [get(gm) for gm, _ in items]
            sim = [get(sm) for _, sm in items]
            # a duplicate: implied_reload_s = 1 / shots_per_man_per_s; a pair's loser routs when the fight ends
            dup = name in DUPLICATE_ROWS or (name.endswith(".rout_after_s") and all(
                x is not None and gm.get("fight_s") is not None and abs(x - gm["fight_s"]) < 1e-6
                for x, (gm, _) in zip(game, items)))
            if name == "winner_side":
                agree = sum(a == b for a, b in zip(game, sim))
                rows.append({"arena": arena, "metric": name, "game": game, "sim": sim, "match": f"{agree}/{len(game)}",
                             "ok": agree == len(game), "duplicate": False})
                continue
            gmean, smean = mean(game), mean(sim)
            err = rel_err(smean, gmean)
            if gmean is None:       # the game never saw it (the winner never wavered): nor should the simulator
                rows.append({"arena": arena, "metric": name, "game": None, "game_range": [None, None], "sim": smean,
                             "rel_err": None, "ok": smean is None, "duplicate": dup})
                continue
            rows.append({"arena": arena, "metric": name, "game": gmean,
                         "game_range": [min((x for x in game if x is not None), default=None),
                                        max((x for x in game if x is not None), default=None)],
                         "sim": smean, "rel_err": err, "ok": err is not None and abs(err) <= TOLERANCE,
                         "duplicate": dup})
    return {"rows": rows, "cases": cases, "hit_bins": bins, "first_strike": first_strike}


def routs(b):
    """Routs, rallies and how long each rally took (s) over a battle's units. A rout starts when the flag comes on
    after the first record; a rally is its end with men left and the unit not shattered (a unit routing from the
    first record has no start: its end is not a rally)."""
    n_rout, n_rally, spans = 0, 0, []
    r_all = np.asarray(b.f["r"], dtype=bool)
    for i in range(len(b.names)):
        r = r_all[:, i]
        on = np.flatnonzero(r[1:] & ~r[:-1]) + 1
        off = np.flatnonzero(r[:-1] & ~r[1:]) + 1
        n_rout += len(on)
        if r[0]:
            off = off[1:]
        off = off[:len(on)]
        ok = (b.f["men"][off, i] > 0) & ~np.asarray(b.f["s"][off, i], dtype=bool)
        n_rally += int(ok.sum())
        spans += [float(b.t[e] - b.t[a]) for a, e in zip(on[:len(off)][ok], off[ok])]
    return n_rout, n_rally, spans


def whole_quantities(b, p):
    """What a whole battle is scored on, the same for the game and a simulated copy (cut at the game's end):
    own_won, the first contact (s), per side the HP lost at the end and 60/120/180 s after the first contact and the
    share of units that routed at least once, routs and rallies per unit."""
    q = {"own_won": won(b.winner)}
    c = measure.first(b.f["m"].any(axis=1))
    q["first_contact_s"] = None if c is None else float(b.t[c])
    for side, tag in ((1, "own"), (2, "enemy")):
        idx = np.nonzero(b.side == side)[0]
        hp = sum(measure.hp_abs(b, i, p) for i in idx)
        q[f"{tag}.hp_lost_end"] = float(1 - hp[-1] / hp[0])
        for after in CURVE_S:
            k = None if c is None else measure.first(b.t >= b.t[c] + after)
            k = len(b.t) - 1 if k is None else k
            q[f"{tag}.hp_lost_{after}s"] = float(1 - hp[k] / hp[0])
        q[f"{tag}.routed_share"] = float(np.mean([bool(b.f["r"][:, i].any()) for i in idx]))
    n_rout, n_rally, _ = routs(b)
    q["routs_per_unit"] = n_rout / len(b.names)
    q["rallies_per_unit"] = n_rally / len(b.names)
    return q


def majority(winners):
    """The copies' winner (1 or 2) by majority, 0 on a tie (an even split is no prediction)."""
    n1, n2 = sum(int(w) == 1 for w in winners), sum(int(w) == 2 for w in winners)
    return 1 if n1 > n2 else 2 if n2 > n1 else 0


def battle_runs(net=False):
    """The recorded whole battles of a family: CA's planner against the game's AI (the Empire-Skaven runs and the
    mirror arena), or (net) the network's battles against the game's AI on generated armies. From the manifests,
    without parsing the recordings. The network's few runs on the fixed arenas belong to neither (left out)."""
    out = []
    for d in gamedata.runs():
        cfg = config(d)
        arena, own = cfg.get("arena", "arena"), cfg.get("own_ai")
        if net and own == NET and arena.startswith("random"):
            out.append(d)
        elif not net and own in PLANNER and (arena.startswith("whole") or arena == "arena"):
            out.append(d)
    return out


def battles(params=None, device="cpu", copies=COPIES, jitter_m=JITTER_M, net=False):
    """Whole battles: the 10 Empire-Skaven runs and the fair mirror-arena runs, each replayed
    `copies` times from slightly moved starts; the simulator's winner is the majority's (a tie counts half).
    net: instead the network's battles against the game's AI on generated armies (both sides'
    recorded orders replayed). Row numbers "sim_*" are means over the copies. -> rows; each row's "cases" are the
    simskill cases of the battle (popped by main)."""
    from tools.nn.sim import check as here          # measure_fn by its importable name (worker processes)
    params = params or load_params()
    return simulate(battle_runs(net), params, device, copies, jitter_m, end_at_recording=True,
                    measure_fn=here.measure_battle)


def measure_battle(g, sims, fac):
    """simulate()'s measure_fn of a whole battle: battles()'s row, with its simskill "cases"."""
    p = passports()
    names = {1: fac.get("own", "?"), 2: fac.get("enemy", "?")}
    votes = [int(x.winner) for x in sims]
    sw = majority(votes)
    gw = g.winner
    row = {"run": g.run, "arena": g.arena, "own_ai": g.own_ai, "game_winner": int(gw), "sim_winner": int(sw),
           "sim_winner_share": round(max(votes.count(1), votes.count(2)) / len(votes), 2),
           "sim_own_won_share": round(votes.count(1) / len(votes), 3),
           "sim_cut": round(sum(x.result.get("cut", False) for x in sims) / len(sims), 2),
           "game_s": float(g.t[-1]), "sim_s": float(np.median([x.t[-1] for x in sims])),
           "match": (0.5 if sw == 0 else float(gw == sw)) if gw else None}
    for side in (1, 2):
        idx = np.nonzero(g.side == side)[0]
        for tag, bs in (("game", [g]), ("sim", sims)):
            vals = []
            for b in bs:
                hp = sum(measure.hp_abs(b, i, p) for i in idx)
                vals.append((float(1 - hp[-1] / hp[0]), float(np.nansum(b.f["men"][-1, idx])),
                             sum(bool(b.f["r"][:, i].any()) for i in idx)))
            m = np.mean(vals, axis=0)
            row[f"{tag}_hp_lost_{side}"] = round(float(m[0]), 3)
            row[f"{tag}_men_end_{side}"] = float(m[1])
            row[f"{tag}_routed_{side}"] = round(float(m[2]), 2)
    gq = whole_quantities(g, p)
    sq = [whole_quantities(s, p) for s in sims]
    for tag, qs in (("game", [gq]), ("sim", sq)):
        for side, st in ((1, "own"), (2, "enemy")):
            row[f"{tag}_hp_lost_after_contact_{side}"] = [round(float(np.mean([q[f"{st}.hp_lost_{a}s"] for q in qs])), 3)
                                                          for a in CURVE_S]
    for tag, bs in (("game", [g]), ("sim", sims)):
        rr = [routs(b) for b in bs]
        row[f"{tag}_routs"] = float(np.mean([r[0] for r in rr]))
        row[f"{tag}_rallies"] = float(np.mean([r[1] for r in rr]))
        row[f"{tag}_rally_s"] = [x for r in rr for x in r[2]]
    row["factions"] = names
    row["cases"] = cases_of(g.run, gq, sq)
    return row


def gate_cases(root=GATES):
    """simskill cases of every gap card (tools/ops/gapcard.py: the gate's network against ai_like in the simulator,
    copies of each gate battle, measured by tools/nn/battle_metrics): -> (cases, gates)."""
    cases, gates = [], []
    for path in sorted(Path(root).glob("*/gapcard.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        gates.append(path.parent.name)
        for b in doc.get("battles", []):
            game, sims = b.get("game") or {}, b.get("sims") or []
            for k, v in game.items():
                if isinstance(v, bool):
                    v = float(v)
                if not isinstance(v, (int, float)):
                    continue
                xs = [float(s.get(k)) if isinstance(s.get(k), (int, float)) else None for s in sims]
                cases.append(simskill.case(f"{path.parent.name}/{b.get('battle')}", k, v, xs))
    return cases, gates


def forecast(cases, boot=simskill.BOOT):
    """simskill.summary of a family's cases (scored here)."""
    return simskill.summary(simskill.score([c for c in cases if c is not None]), boot=boot)


def speed(device="cpu", batch=1024, params=None, seconds=None):
    """Battles a second: `batch` copies of the Empire-Skaven battle (11 slots a side), every unit
    attacking the nearest enemy, run to the end."""
    params = params or load_params()
    army = scenario.from_arena("whole_emp_v_skv", "attack")

    def fresh():
        st = scenario.build([army] * batch, params, device=device, per_side=20)
        # Different battles: move the starts by up to 10 m.
        g = torch.Generator(device="cpu").manual_seed(0)
        st.u["x"] = st.u["x"] + (torch.rand(st.u["x"].shape, generator=g) * 20 - 10).to(device)
        st.u["z"] = st.u["z"] + (torch.rand(st.u["z"].shape, generator=g) * 20 - 10).to(device)
        return st
    # Warm up (on CUDA the first steps compile the step), then time a fresh batch.
    battle.run(fresh(), replay.nearest_attack, params, until_s=2.0)
    st = fresh()
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0

    def count(_):
        nonlocal steps
        steps += 1
    battle.run(st, replay.nearest_attack, params, until_s=seconds, every_s=1.0, record=count)
    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"device": device, "batch": batch, "slots": st.N, "steps": steps, "wall_s": round(wall, 2),
            "sim_s": round(float(st.t.max()), 1), "battles_per_s": round(batch / wall, 1),
            "steps_per_s": round(steps / wall, 1), "done": int(st.done.sum()),
            "winner_side2": int((st.winner == 2).sum())}


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(argv=None):
    global CHUNK, WORKERS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="a comma list of mechanics, battles, gates, speed (default: all)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: both where cuda exists, for speed)")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--copies", type=int, default=COPIES, help="replays of each recorded battle (19: the "
                        "copies' range is a 90 %% interval)")
    parser.add_argument("--chunk", type=int, default=None, help=f"simulated battles per batch (default {CHUNK}; "
                        "on a GPU 4096: one big batch)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="batches at once, in processes of one thread "
                        "(CPU)")
    parser.add_argument("--boot", type=int, default=simskill.BOOT, help="bootstrap draws of the CIs")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    only = set((args.only or "mechanics,battles,gates,speed").split(","))
    bad = only - {"mechanics", "battles", "gates", "speed"}
    if bad:
        parser.error(f"--only: unknown {', '.join(sorted(bad))}")
    sys.stdout.reconfigure(encoding="utf-8")
    dev = args.device or "cpu"
    CHUNK = args.chunk or (CHUNK if dev == "cpu" else 4096)
    WORKERS = args.workers
    params = load_params()
    report = {"copies": args.copies, "forecast": {}, "cases": {}}
    forecast_lines = []
    t0 = time.perf_counter()

    def family(key, title, cases):
        cases = [c for c in cases if c is not None]
        report["cases"][key] = [{"battle": c["battle"], "q": c["q"], "game": c["game"],
                                 "sims": [round(float(x), 5) for x in c["sims"]]} for c in cases]
        s = forecast(cases, args.boot)
        report["forecast"][key] = dict(s or {}, title=title)
        forecast_lines.extend(simskill.lines(title, s, worst=10))
    if "mechanics" in only:
        mech = mechanics(params, dev, args.copies)
        rows = mech["rows"]
        report["mechanics"] = rows
        report["missile_bins"] = mech["hit_bins"]
        family("mechanics", "mechanics (melee pairs and shooting runs)", mech["cases"])
        print("== mechanics: game (mean of 3 runs) against the simulator (same recorded orders)")
        print(f"{'arena':26} {'metric':34} {'game':>10} {'range':>17} {'sim':>10} {'err':>7}  ok")
        for r in rows:
            if r["metric"] == "winner_side":
                print(f"{r['arena']:26} {r['metric']:34} {str(r['game']):>10} {'':>17} {str(r['sim']):>10} "
                      f"{r['match']:>7}  {'yes' if r['ok'] else 'NO'}")
                continue
            rng = r["game_range"]
            print(f"{r['arena']:26} {r['metric']:34} {fmt(r['game']):>10} {fmt(rng[0]) + '-' + fmt(rng[1]):>17} "
                  f"{fmt(r['sim']):>10} {fmt(r['rel_err'] and 100 * r['rel_err'], 0) + '%':>7}  "
                  f"{'yes' if r['ok'] else 'NO'}")
        ok = [r["ok"] for r in rows]
        uniq = [r["ok"] for r in rows if not r.get("duplicate")]
        print(f"within {int(TOLERANCE * 100)} %: {sum(ok)} of {len(ok)} (without the rows that repeat another "
              f"number - implied_reload_s, a pair loser's rout time: {sum(uniq)} of {len(uniq)})")
        for x in mech["hit_bins"]:
            if x["game_last_bin"] != x["sim_last_bin"]:
                print(f"  hit_rate row of {x['run']} ({x['arena']}): game's last distance bin {x['game_last_bin']}, "
                      f"simulator's {x['sim_last_bin']} - the old row compares different distances")
        report["first_strike"] = mech["first_strike"]
        by = {}
        for x in mech["first_strike"]:
            by.setdefault((x["arena"], x["unit"]), []).append(x)
        print("HP lost in the contact second (the record before the first melee flag to the first), mean of the runs, "
              "game / simulator: " + "; ".join(f"{a} {u} {mean([x['game'] for x in v]):.0f} / "
                                               f"{mean([x['sim'] for x in v]):.0f}" for (a, u), v in sorted(by.items())))
        print(f"[mechanics: {time.perf_counter() - t0:.0f} s]")
    if "battles" in only:
        rows = battles(params, dev, args.copies)
        family("game_ai", "battles of CA's planner against the game's AI", [c for r in rows for c in r.pop("cases")])
        report["battles"] = rows
        print("== whole battles: recorded orders replayed with contact-phase synchronisation")
        print(f"{'run':16} {'arena':16} {'own_ai':6} {'winner game/sim (share)':>24} {'s game/sim':>12} "
              f"{'HP lost 1 game/sim':>19} {'HP lost 2 game/sim':>19}")
        for r in rows:
            print(f"{r['run']:16} {r['arena']:16} {r['own_ai']:6} {r['game_winner']:>9}/{r['sim_winner']} "
                  f"({r['sim_winner_share']:.2f})       "
                  f"{r['game_s']:>6.0f}/{r['sim_s']:<5.0f} {r['game_hp_lost_1']:>9.2f}/{r['sim_hp_lost_1']:<9.2f} "
                  f"{r['game_hp_lost_2']:>9.2f}/{r['sim_hp_lost_2']:<9.2f}")
        summary = {}
        for group in ("whole", "arena"):
            rs = [r for r in rows if (r["arena"].startswith("whole") if group == "whole" else r["arena"] == "arena")
                  and r["match"] is not None]
            if rs:
                summary[group] = {"decided": len(rs), "same_winner": sum(r["match"] for r in rs),
                                  "share": round(sum(r["match"] for r in rs) / len(rs), 3)}
        allr = [r for r in rows if r["match"] is not None]
        summary["all"] = {"decided": len(allr), "same_winner": sum(r["match"] for r in allr),
                          "share": round(sum(r["match"] for r in allr) / max(len(allr), 1), 3)}
        for tag in ("game", "sim"):
            spans = [x for r in rows for x in r[f"{tag}_rally_s"]]
            summary[f"{tag}_routs_per_battle"] = round(float(np.mean([r[f"{tag}_routs"] for r in rows])), 1)
            summary[f"{tag}_rallies_per_battle"] = round(float(np.mean([r[f"{tag}_rallies"] for r in rows])), 1)
            summary[f"{tag}_rally_median_s"] = float(np.median(spans)) if spans else None
            summary[f"{tag}_duration_mean_s"] = round(float(np.mean([r[f"{tag}_s"] for r in rows])), 0)
        curves = {}
        for r in rows:
            if not r["arena"].startswith("whole"):
                continue
            for side in (1, 2):
                name = r["factions"][side]
                for tag in ("game", "sim"):
                    curves.setdefault(name, {}).setdefault(tag, []).append(r[f"{tag}_hp_lost_after_contact_{side}"])
        summary["hp_lost_after_contact"] = {
            name: {tag: [round(float(v), 3) for v in np.mean(c, axis=0)] for tag, c in d.items()}
            for name, d in curves.items()}
        for name, d in summary["hp_lost_after_contact"].items():
            print(f"{name}: HP lost {', '.join(str(x) for x in CURVE_S)} s after the first contact: "
                  f"game {d['game']}, sim {d['sim']}")
        print(f"routs / rallies a battle: game {summary['game_routs_per_battle']} / {summary['game_rallies_per_battle']}, "
              f"sim {summary['sim_routs_per_battle']} / {summary['sim_rallies_per_battle']}; rally takes (median): "
              f"game {summary['game_rally_median_s']} s, sim {summary['sim_rally_median_s']} s; battle (mean): "
              f"game {summary['game_duration_mean_s']} s, sim {summary['sim_duration_mean_s']} s")
        summary["sim_not_over_at_recording_end"] = round(float(np.mean([r["sim_cut"] for r in rows])), 2)
        print(f"simulated battles not over when the recording ended (winner = side with more standing health): "
              f"{100 * summary['sim_not_over_at_recording_end']:.0f} %")
        report["battles_summary"] = summary
        for k in ("whole", "arena", "all"):
            if k in summary:
                v = summary[k]
                print(f"same winner, {k}: {v['same_winner']:g} of {v['decided']} ({100 * v['share']:.0f} %; "
                      f"majority of {args.copies} copies, a tie counts half)")
        print(f"[battles of the game's AI: {time.perf_counter() - t0:.0f} s]")
        # The network's gate battles (generated armies), apart: both sides' recorded orders replayed.
        net_rows = battles(params, dev, args.copies, net=True)
        family("network", "the network's battles against the game's AI (recorded orders)",
               [c for r in net_rows for c in r.pop("cases")])
        report["battles_net"] = net_rows
        print("== the network's battles against the game's AI (generated armies), contact-phase replay")
        for r in net_rows:
            print(f"{r['run']:16} {r['arena']:20} winner game/sim {r['game_winner']}/{r['sim_winner']} "
                  f"({r['sim_winner_share']:.2f})  s {r['game_s']:.0f}/{r['sim_s']:.0f}  HP lost 1 "
                  f"{r['game_hp_lost_1']:.2f}/{r['sim_hp_lost_1']:.2f}  2 {r['game_hp_lost_2']:.2f}/{r['sim_hp_lost_2']:.2f}")
        decided = [r for r in net_rows if r["match"] is not None]
        report["battles_net_summary"] = {"decided": len(decided), "same_winner": sum(r["match"] for r in decided)}
        print(f"same winner, the network's battles: {sum(r['match'] for r in decided):g} of {len(decided)} "
              f"(majority of {args.copies} copies, a tie counts half)")
        print(f"[battles of the network: {time.perf_counter() - t0:.0f} s]")
    if "gates" in only:
        cases, gates = gate_cases()
        family("gates", f"gate battles: the gate's network against ai_like ({len(gates)} gap cards)", cases)
    if forecast_lines:
        print("== the simulator as a forecast of the game (tools/nn/simskill.py): per family, the 10 worst-covered "
              "quantities")
        print(chr(10).join(forecast_lines))
    if "speed" in only:
        devices = [args.device] if args.device else (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        report["speed"] = []
        for d in devices:
            r = speed(d, args.batch, params)
            report["speed"].append(r)
            print(f"== speed {d}: {r['batch']} battles x {r['slots']} slots, {r['sim_s']} s simulated, "
                  f"{r['wall_s']} s wall -> {r['battles_per_s']} battles/s, {r['steps_per_s']} steps/s "
                  f"({r['done']} finished)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report["wall_s"] = round(time.perf_counter() - t0)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
