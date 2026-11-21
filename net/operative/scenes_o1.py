"""The scenes of drill O0-1 'advance on the holder' (tmp/op1/design_opus.md 5, Astra's twin rule
tmp/astra/op1_design/out/DESIGN.md; the axes net/config/operative.json o1.scenes.axes). A scene is a start: both sides
in formation facing each other at the drawn front gap, nothing moving, both lords lord_back_m behind their side; my
group's zone snapped on my places (tactic/scenes.snap - the first frames where the units stand); the enemy's behaviour
in spec['o1'] (net/operative/sally.Enemy). The level decides first at o1.start_s (the decision tick, t_dec).

    scene(seed, held=None, twin=None)   one scene; the cause 'our mission is attack and the enemy holds': att 1, the
                 enemy a Sally holder (o1.sally_m drawn). Its twins differ ONLY by the cause (the same draws):
                 twin 'come' - the enemy moves: each unit back-placed along its line of march by its own run (passport
                 speed.run x t_dec) and kept on schedule by sally.WalkIn (a run while behind, else a walk) so that it
                 passes the cause scene's pose at t_dec, moving, then attacks (o1.comer: nearest or human); twin 'role' - att 0, the same holder
    Axes (o1.scenes.axes; train / held): map, size (my infantry; the enemy's the same count, matched gold), factions,
                 mirror, gap_m (front to front at t_dec: the centres' gap less 2 x half_depth_m), lat_m (the enemy's
                 offset across), our_form (row | rows2: 5 + 3 | square), enemy_form (row | rows2 | shift100: a row 100 m
                 across), heading_deg, sally_m, comer; held - the one axis at its held-out value
    taught_set(n), held_set(n), general_set(n, hold=False)   the exam's sets (o1.scenes): taught - n cause + n twins
                 (even i come, odd i role); held - for each held axis n cause + n twins; G - stage 0's general battles
                 (stage0/specs.drill_generals; my rows in my zone, battle.unit_battles) with att 1 against the Sally
                 holder (hold: against sim/scripts hold - it never comes)
    binds(sp)    the scene's held axis binds - its held-out value is what the scene has (None: no held axis); the gap's
                 binding in play is read by the exam off the level's first-tick gap
A spec: tactic/scenes' keys (fac, factions, mine, theirs, map, places_mine, places_theirs, lords, worn, zone, limit,
enemy 'o1') + o1 {att, enemy 'sally' | 'walkin' | 'hold', sally_m, comer, t_dec, goals [(x, z)] (walkin), cause, twin,
held, pose [(x, z)] (the cause scene's enemy places)}, axes {name: value}, seed, held.
"""
import copy
import math
import random

from net.operative import cfg
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP
from net.tactic import scenes as TS

SUBSTITUTE = 1_000_000                 # (a registered seed's k-th substitute: seed + k x this)
HELD_AXES = ("map", "size", "gap_m", "our_form", "enemy_form", "sally_m")


def c():
    return cfg()["o1"]


def cs():
    return c()["scenes"]


def draw(rnd, name, held):
    """A value of axis name (o1.scenes.axes): its held-out value when held == name, else a training one (ranges:
    uniform weighted by length; lists: a choice)."""
    v = cs()["axes"][name].get("held" if held == name else "train") or []
    assert v, f"axis {name} has no {'held' if held == name else 'train'} value"
    if isinstance(v[0], list) and len(v[0]) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         for x in v[0]) and name != "factions":
        lo, hi = rnd.choices(v, weights=[max(b - a, 1e-6) for a, b in v])[0]
        return rnd.uniform(lo, hi)
    return copy.deepcopy(rnd.choice(v))


def row_counts(kind, n):
    """[units a row] of _form(kind, n), front row first."""
    out = {}
    for _, f in _form(kind, n, 1.0, 1.0):
        out[f] = out.get(f, 0) + 1
    return [out[f] for f in sorted(out, reverse=True)]


def _form(kind, n, row_m, rank_m):
    """[(s, f)] n places in the zone's axes: the front row at f = 0, rows behind (f < 0)."""
    if kind == "row":
        rows = [n]
    elif kind == "rows2":
        back = max(1, int(round(3 * n / 8))) if n >= 2 else 0
        rows = [n - back, back] if back else [n]
    elif kind == "square":
        cols = int(math.ceil(math.sqrt(n) - 1e-9))
        rows = [cols] * (n // cols) + ([n % cols] if n % cols else [])
    else:
        raise ValueError(kind)
    out = []
    for r, k in enumerate(rows):
        out += [((i - (k - 1) / 2) * row_m, -r * rank_m) for i in range(k)]
    return out


def _world(o, h, s, f):
    return SP.world({"x": o[0], "z": o[1], "b": h}, s, f)


def _try(rnd, held, mp):
    S, C = cs(), c()
    fac = draw(rnd, "factions", held)
    n = int(draw(rnd, "size", held))
    if held == "our_form":                             # (the held form only where its rows differ from every taught form's)
        n = rnd.choice([k for k in cs()["axes"]["size"]["train"] if row_counts("square", k) not in
                        [row_counts(f, k) for f in cs()["axes"]["our_form"]["train"]]])
    mirror = bool(draw(rnd, "mirror", held))
    gap = float(draw(rnd, "gap_m", held))
    lat = float(draw(rnd, "lat_m", held))
    our_form = draw(rnd, "our_form", held)
    en_form = draw(rnd, "enemy_form", held)
    h = float(draw(rnd, "heading_deg", held)) % 360.0
    sally = float(draw(rnd, "sally_m", held))
    comer = draw(rnd, "comer", held)
    mine = SP.army(rnd, fac[0], n)
    theirs = SP.matched(rnd, fac[1], n, SP.gold(mine))
    r = float(S["place_r_m"])
    mid = (rnd.uniform(-r, r), rnd.uniform(-r, r))
    if mp not in (None, "flat"):                       # (a captured map: the middle on ground a unit can stand on)
        r2 = r + 150.0
        for _ in range(50):
            mid = (rnd.uniform(-r2, r2), rnd.uniform(-r2, r2))
            if TS.can_stand(mp, mid[0], mid[1], h):
                break
    row_m, rank_m, hd = float(S["row_m"]), float(S["rank_m"]), float(S["half_depth_m"])
    sg = -1.0 if mirror else 1.0
    fwd = (math.sin(math.radians(h)), math.cos(math.radians(h)))
    o = (mid[0] - (gap / 2) * fwd[0], mid[1] - (gap / 2) * fwd[1])        # (my front row's middle)
    ours = [(sg * s, f) for s, f in _form(our_form, n, row_m, rank_m)]
    f_e = gap + 2 * hd
    shift = (100.0 if en_form == "shift100" else 0.0) + lat
    ekind = "row" if en_form == "shift100" else en_form
    theirs_sf = [(sg * (s + shift), f_e - f) for s, f in _form(ekind, n, row_m, rank_m)]
    got = TS.snap([_world(o, h, s, f) for s, f in ours], h)
    if got is None:
        return None
    zone, snapped = got
    if not all(TS.can_stand(mp, x, z, h) for x, z in snapped):
        return None
    hb = (h + 180.0) % 360.0
    pt = [(*_world(o, h, s, f), hb) for s, f in theirs_sf]
    if not all(TS.can_stand(mp, x, z, hb) for x, z, _ in pt):
        return None
    # each of my units' straight walk forward to the enemy's front row: on ground that can be walked (points 8 m apart)
    if mp not in (None, "flat"):
        for x, z in snapped:
            for k in range(int(gap // 8.0) + 1):
                if not TS.can_stand(mp, x + 8.0 * k * fwd[0], z + 8.0 * k * fwd[1], h, box=(4.0, 4.0)):
                    return None
    # the come twin's starts must stand too (the twin shares this scene)
    t_dec = float(C["start_s"])
    starts = []
    for k, (x, z, _) in zip(theirs, pt):
        back = float(TS._passports()[k]["speed"]["run"]) * t_dec       # (sally.WalkIn keeps it on schedule)
        starts.append((x + back * fwd[0], z + back * fwd[1]))
    if not all(TS.can_stand(mp, x, z, hb) for x, z in starts):
        return None
    lb = float(S["lord_back_m"])
    mx, mz = sum(p[0] for p in snapped) / n, sum(p[1] for p in snapped) / n
    ex, ez = sum(p[0] for p in pt) / n, sum(p[1] for p in pt) / n
    lords = {1: (mx - lb * fwd[0], mz - lb * fwd[1], h), 2: (ex + lb * fwd[0], ez + lb * fwd[1], hb)}
    if not all(TS.can_stand(mp, *lords[s]) for s in (1, 2)):
        return None
    goals = [(x - 300.0 * fwd[0], z - 300.0 * fwd[1]) for x, z, _ in pt]
    sc = scfg()
    return {"kind": "o1", "fac": {1: fac[0], 2: fac[1]},
            "factions": {1: sc["factions"][fac[0]], 2: sc["factions"][fac[1]]}, "mine": mine, "theirs": theirs,
            "map": mp, "places_mine": [(x, z, h) for x, z in snapped], "places_theirs": pt, "lords": lords,
            "worn": {"mine": [None] * n, "theirs": [None] * n}, "zone": zone, "reserve": [], "enemy": "o1",
            "limit": int(C["limit_s"]),
            "o1": {"att": 1, "enemy": "sally", "sally_m": sally, "comer": comer, "t_dec": t_dec, "cause": True,
                   "twin": None, "pose": [(x, z) for x, z, _ in pt], "starts": starts, "goals": goals},
            "axes": {"factions": fac, "map": mp, "size": n, "mirror": mirror, "gap_m": gap, "lat_m": lat,
                     "our_form": our_form, "our_rows": row_counts(our_form, n), "enemy_form": en_form, "heading_deg": h, "sally_m": sally, "comer": comer}}


def scene(seed, held=None, twin=None):
    """(module doc)."""
    assert twin in (None, "come", "role"), twin
    sp, used = None, None
    for k in range(int(cs()["substitutes"]) + 1):     # (a registered seed with no layout: the next substitute seed,
        used = int(seed) + SUBSTITUTE * k              # written into the spec - deterministic, the twins the same)
        rnd = random.Random(used)
        mp = draw(rnd, "map", held)                    # (drawn once: a hard map must not be redrawn into the flat one)
        for _ in range(int(cs()["redraw"])):
            sp = _try(rnd, held, mp)
            if sp is not None:
                break
        if sp is not None:
            break
    if sp is None:
        raise RuntimeError(f"O0-1 scene seed {seed} held {held}: no layout in {cs()['redraw']} draws of "
                           f"{int(cs()['substitutes']) + 1} seeds")
    sp.update(seed=int(seed), held=held, seed_used=used)
    o = sp["o1"]
    o["held"] = held
    if twin == "come":
        o.update(enemy="walkin", cause=False, twin="come", sally_m=None)
        sp["places_theirs"] = [(x, z, b) for (x, z), (_, _, b) in zip(o["starts"], sp["places_theirs"])]
    elif twin == "role":
        o.update(att=0, cause=False, twin="role")
    return sp


def taught_set(n=None):
    n = int(cs()["n"] if n is None else n)
    s0 = int(cs()["seed"])
    return [scene(s0 + i) for i in range(n)] + [scene(s0 + i, twin="come" if i % 2 == 0 else "role") for i in range(n)]


def held_set(n=None):
    n = int(cs()["n_held"] if n is None else n)
    s0 = int(cs()["held_seed"])
    out = []
    for j, ax in enumerate(HELD_AXES):
        out += [scene(s0 + 10000 * j + i, held=ax) for i in range(n)]
        out += [scene(s0 + 10000 * j + i, held=ax, twin="come" if i % 2 == 0 else "role") for i in range(n)]
    return out


def general_set(n=None, hold=False):
    """(module doc) Set G (hold: G_hold) - stage 0's general battles under the drill's enemy, the timer o1.limit_s."""
    S = cs()
    n = int((S["general_hold"] if hold else S["n_general"]) if n is None else n)
    seed = int(S["general_seed"]) + (500 if hold else 0)
    sp = SP.drill_generals(n, SP.seeded(seed), limit=int(c()["limit_s"]))
    rnd = random.Random(seed + 1)
    for x in sp:
        x["enemy"] = "o1"
        x["o1"] = {"att": 1, "enemy": "hold" if hold else "sally", "sally_m": None if hold else draw(rnd, "sally_m", None),
                   "comer": None, "t_dec": float(c()["start_s"]), "cause": True, "twin": None, "held": None}
    return sp


def binds(sp):
    """(module doc)."""
    h = sp.get("held")
    if h is None:
        return None
    ax = sp["axes"]
    v = cs()["axes"][h]["held"]
    if h == "size":
        return len(sp["mine"]) in v
    if h == "our_form":                                # (by geometry: its rows differ from every taught form's at this size)
        n = len(sp["mine"])
        return ax["our_rows"] not in [row_counts(f, n) for f in cs()["axes"]["our_form"]["train"]]
    if h in ("map", "enemy_form"):
        return ax[h] in v
    lo, hi = v[0]
    return lo - 1e-6 <= float(ax[h]) <= hi + 1e-6


def binds_count(specs):
    out = {}
    for s in specs:
        b = binds(s)
        if b is None:
            continue
        r = out.setdefault(s["held"], {"scenes": 0, "binds": 0})
        r["scenes"] += 1
        r["binds"] += int(bool(b))
    return out
