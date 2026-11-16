"""The scenes of drill T0-2 'strike the pinned one' (tmp/tactic_slice/design_attack_v23.md section 4; review_attack_v23
B, E, G; config stage0.json drills.attack). A scene is a start - places of both sides at second 0, a plan for every
enemy unit, worn starts; nothing in melee, every order of mine the default. The level and the unit script play it; a
label is the teacher's answer on a state reached by play (tactic/tags.py, exam_pin scenes); nothing is painted but men
and health of a worn unit, nothing is selected after play. This slice: the recipe A1 and its twin N1.

    a1(seed, held=None)   A1 'lane': the anvil u and one free unit aside or aside-and-behind (aside_m, behind_deg;
                          never straight behind: its offset across the anvil's front >= orders.through_tag_m on the
                          snapped places); one enemy `at u` from start_m at the approach phi_deg (its sign drawn)
    n1(seed, held=None)   N1 'pass-by', the A1 scene of the same seed with one fact changed: the enemy's plan becomes
                          `point` (it runs past >= point_clear_m from every unit of mine and stands point_behind_m behind
                          my row, point_aside_m beyond its end) or, where no such way exists or by point_share, `halt`
                          (it stands halt_m short of u); and a bystander pair - a unit of mine and an enemy `at` it -
                          fights bystander_m from the nearer of the anvil and the free unit (both may be free here)
    make(recipe, seed)    any recipe (config recipes, own, sets): A2 wide line, A3 reserve, A4 freed late, A5 uneven
                          line, A6 second coming (i queue / ii late wing / iii E2 at the free unit), the late twins A4L /
                          A5L (every other enemy loose), the twins of a cause scene drawn from the seed - N1 (twin_n1:
                          every enemy passes by or halts, a bystander pair), N2 (twin_own: an own enemy for each free
                          unit, arriving after the anvil's; the scene ends own.end_after_s later), E1 (all halt), E3
                          (an own enemy arriving first) - and E2 far; each through build(): the slots shuffled, snapped,
                          the enemies put relative to the snapped places, at most 8 a side
    set_specs(name), held_specs()   the fixed sets S, V, a round's draw; H - each held-out axis at its value
    Layout (review G.1, critic 2 points 1-2): slots shuffled first; the zone (zfit.fit, the closed form) on the places in
    slot order; one orders.frame_walk; each unit put on its frame's centre facing the course - so the first frames are
    where the units stand; the recipe's geometry re-checked on the snapped places, the place redrawn on a miss or on
    ground that cannot be stood on (the map's passable.npz), never dropped.
    Axes (config axes): each numeric one held out at an END of its range (review E); held - the one axis drawn at its
    held-out value (None: all at training values); the spec carries every drawn value and asserts it lies in its band.
    The map axis has no held-out value yet ('TODO': the river map is 90 % 'cannot stand' in the east zone).
A spec: stage0/specs' keys (kind 'scene', fac, factions, mine, theirs, map, east, mine_zone, enemy_zone, limit) +
places_mine / places_theirs [(x, z, b)], lords {1, 2: (x, z, b)}, plan_theirs [{kind at | nearest | point | halt, k
(index into mine), x, z, gap, from_s, run}], worn {mine, theirs: [share | None]}, zone, perm, family, recipe, seed,
held, axes {name: value}, roles {anvil, free, bystander: index into mine}.
"""
import copy
import json
import math
import random
from functools import lru_cache

import numpy as np

from net import ROOT
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP

UNIT_BOX = (40.0, 24.0)            # (a unit's box on the ground for the 'cannot stand' check, m: critic 2's 40 x 24)


def c():
    return scfg()["drills"]["attack"]


@lru_cache(maxsize=None)
def _blocked(mp):
    """(blocked [rows, cols] uint8, x0, z0, step) of a captured map; None for the flat field."""
    if mp in (None, "flat"):
        return None
    p = np.load(ROOT / "sim" / "maps" / mp / "passable.npz")
    return p["blocked"], float(p["x0"]), float(p["z0"]), float(p["step"])


def can_stand(mp, x, z, b, box=UNIT_BOX):
    """A box front x depth at (x, z) facing b touches no 'cannot stand' cell of the map (5 x 3 points), and is on it."""
    g = _blocked(mp)
    if g is None:
        return True
    blk, x0, z0, st = g
    h = math.radians(b)
    for a in np.linspace(-box[0] / 2, box[0] / 2, 5):
        for d in np.linspace(-box[1] / 2, box[1] / 2, 3):
            px, pz = x + a * math.cos(h) + d * math.sin(h), z - a * math.sin(h) + d * math.cos(h)
            i, j = int(round((px - x0) / st)), int(round((pz - z0) / st))
            if not (0 <= j < blk.shape[0] and 0 <= i < blk.shape[1]) or blk[j, i]:
                return False
    return True


@lru_cache(maxsize=None)
def _men0(key):
    return int(json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"][key]["men"])


# ------------------------------------------------------------------ axes and the worn law
def bands(name, which):
    return c()["axes"][name].get(which) or []


def draw(rnd, name, held):
    """A value of axis name: its held-out value when held == name, else a training one (ranges: uniform, weighted by
    length; lists: a choice)."""
    which = "held" if held == name else "train"
    v = bands(name, which)
    assert v, f"axis {name} has no {which} value"
    if isinstance(v[0], list) and all(isinstance(x, (int, float)) for x in v[0]) and len(v[0]) == 2 and name not in (
            "factions",):
        w = [hi - lo for lo, hi in v]
        lo, hi = rnd.choices(v, weights=w)[0]
        return rnd.uniform(lo, hi)
    return copy.deepcopy(rnd.choice(v))


def in_band(name, which, x):
    v = bands(name, which)
    if isinstance(x, (int, float)) and v and isinstance(v[0], list):
        return any(lo - 1e-9 <= x <= hi + 1e-9 for lo, hi in v)
    return x in v


def worn_share(rnd, held):
    """The one law of worn starts (config worn): None (fresh) with p fresh, else a share."""
    w = c()["worn"]
    if held != "worn_share" and rnd.random() < float(w["fresh"]):
        return None
    v = w["held"] if held == "worn_share" else w["train"]
    lo, hi = rnd.choices(v, weights=[b - a for a, b in v])[0]
    return rnd.uniform(lo, hi)


def _roster(fac, held):
    keys = list(scfg()["roster"][fac])
    out = [k for k in keys if k not in bands("unit", "held")]
    if held == "unit":
        hk = [k for k in bands("unit", "held") if k in keys]
        return hk or out
    return out


# ------------------------------------------------------------------ the layout
def snap(places, course):
    """(zone {cx, cz, heading, line, rows, cols, worst}, [(x, z)] the frames' centres) of my units at rough places
    [(x, z)] in slot order (module doc): zfit.fit, one frame walk with every unit forced; None when no zone holds them."""
    import torch
    from net.tactic import orders as OR
    from net.tactic import raw as TR
    from net.tactic import zfit
    from net.tactic import zone as ZN
    from net.unit import zmap as Z
    z = zfit.fit(places, course)
    if z is None:
        return None
    G = len(places)
    t = lambda k: torch.tensor([float(z[k])])                                     # noqa: E731
    tiles = ZN.Tiles(t("cx"), t("cz"), t("heading"), torch.tensor([z["rows"]]), torch.tensor([z["cols"]]))
    mr = torch.zeros((1, G, len(TR.RAW)))
    mr[0, :, TR.I["x"]] = torch.tensor([p[0] for p in places])
    mr[0, :, TR.I["z"]] = torch.tensor([p[1] for p in places])
    able = torch.ones((1, G), dtype=torch.bool)
    try:
        fr = OR.frame_walk(mr, able, tiles, None, able.clone())["frames"]
    except Exception:                       # (the level's 'no room' error: no frame for some unit)
        return None
    x, zz = Z.world(fr["s"], fr["f"], tiles.cx[:, None], tiles.cz[:, None], tiles.heading[:, None])
    out = [(float(x[0, i]), float(zz[0, i])) for i in range(G)]
    worst = float(zfit.first_moves(out, [z])[0])
    return {**z, "worst": worst}, out


def _bearing(fx, fz, tx, tz):
    return math.degrees(math.atan2(tx - fx, tz - fz)) % 360.0


def _seg_dist(p, a, b):
    ax, az = b[0] - a[0], b[1] - a[1]
    L2 = ax * ax + az * az
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * az) / L2))
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * az)


# ------------------------------------------------------------------ the recipes
def _base(rnd, held):
    fac = draw(rnd, "factions", held)
    zone = draw(rnd, "zone", held)
    mp = draw(rnd, "map", held)
    assert mp != "TODO", "the map axis has no held-out map yet (module doc)"
    east = zone == "east"
    mz, ez = SP.zone(1 if east else 0), SP.zone(0 if east else 1)
    return {"kind": "scene", "fac": {1: fac[0], 2: fac[1]},
            "factions": {1: scfg()["factions"][fac[0]], 2: scfg()["factions"][fac[1]]}, "map": mp, "east": east,
            "mine_zone": mz, "enemy_zone": ez, "enemy": "scene", "places_mine": None, "reserve": [],
            "limit": int(c()["limit_s"]),
            "axes": {"factions": fac, "zone": zone, "map": mp}}


def _lords(sp, mine_xy, theirs_xy, away):
    """{1, 2: (x, z, b)} (config lord_back_m): mine behind my units' middle along my zone's facing; theirs beyond its
    units' middle along away (the unit vector from the scene to the enemy)."""
    back = float(c()["lord_back_m"])
    mz = sp["mine_zone"]
    mx, mzz = np.mean([p[0] for p in mine_xy]), np.mean([p[1] for p in mine_xy])
    a = math.radians(mz["b"])
    l1 = (float(mx - back * math.sin(a)), float(mzz - back * math.cos(a)), float(mz["b"]))
    ex, ez = np.mean([p[0] for p in theirs_xy]), np.mean([p[1] for p in theirs_xy])
    l2 = (float(ex + back * away[0]), float(ez + back * away[1]), _bearing(0, 0, -away[0], -away[1]))
    return {1: l1, 2: l2}


def _keys(rnd, fac, n, held):
    r = _roster(fac, held)
    out = [rnd.choice(r) for _ in range(n)]
    return out


def a1(seed, held=None):
    """The A1 scene of a seed (module doc)."""
    rnd = random.Random(seed)
    for _ in range(int(c()["redraw"])):
        sp = _a1_try(rnd, held)
        if sp is not None:
            sp.update(seed=int(seed), held=held, family="A1", recipe="A1")
            return sp
    raise RuntimeError(f"A1 seed {seed}: no layout in {c()['redraw']} draws")


def _a1_try(rnd, held):
    from net.tactic import cfg as tcfg
    sp = _base(rnd, held)
    mz = sp["mine_zone"]
    f_row = mz["depth"] / 2 - 30.0
    shift = draw(rnd, "shift_m", held) * rnd.choice((-1.0, 1.0))
    d = draw(rnd, "aside_m", held)
    th = draw(rnd, "behind_deg", held)
    side = rnd.choice((-1.0, 1.0))
    anvil = (shift, f_row)
    free = (shift + side * d * math.cos(math.radians(th)), f_row - d * math.sin(math.radians(th)))
    order = [0, 1]
    rnd.shuffle(order)                                              # (which slot is the anvil: drawn, before the zone)
    rough = {0: anvil, 1: free}
    places_sf = [rough[r] for r in order]
    keys = _keys(rnd, sp["fac"][1], 2, held)
    worn_m = [worn_share(rnd, held) for _ in range(2)]
    phi = draw(rnd, "phi_deg", held)
    psign = rnd.choice((-1.0, 1.0))
    start = draw(rnd, "start_m", held)
    e_key = _keys(rnd, sp["fac"][2], 1, held)
    worn_t = [worn_share(rnd, held)]
    pace = draw(rnd, "pace", held)
    world = [SP.world(mz, s, f) for s, f in places_sf]
    got = snap(world, float(mz["b"]))
    if got is None:
        return None
    zone, snapped = got
    ia, ifr = order.index(0), order.index(1)
    loc = [SP.local(mz, x, z) for x, z in snapped]
    d_snap = math.hypot(loc[ifr][0] - loc[ia][0], loc[ifr][1] - loc[ia][1])
    lateral = abs(loc[ifr][0] - loc[ia][0])
    if not in_band("aside_m", "held" if held == "aside_m" else "train", d_snap):
        return None
    if lateral < float(tcfg()["orders"]["through_tag_m"]) or loc[ifr][1] > loc[ia][1] + 8.0:
        return None                                                 # (straight behind, or ahead of the anvil)
    # the enemy: from the snapped anvil along the approach (0 = straight in front of my row)
    ps = math.radians(phi) * psign
    es, ef = loc[ia][0] + start * math.sin(ps), loc[ia][1] + start * math.cos(ps)
    ex, ez = SP.world(mz, es, ef)
    ax, az = snapped[ia]
    eb = _bearing(ex, ez, ax, az)
    if not all(can_stand(sp["map"], x, z, mz["b"]) for x, z in snapped) or not can_stand(sp["map"], ex, ez, eb):
        return None
    away = ((ex - ax) / start, (ez - az) / start)
    lords = _lords(sp, snapped, [(ex, ez)], away)
    if not all(can_stand(sp["map"], *lords[s]) for s in (1, 2)):
        return None
    sp.update(mine=keys, theirs=e_key, places_mine=[(x, z, float(mz["b"])) for x, z in snapped],
              places_theirs=[(ex, ez, eb)], lords=lords, zone=zone, perm=order,
              plan_theirs=[{"kind": "at", "k": ia, "run": pace == "run"}],
              worn={"mine": worn_m, "theirs": worn_t}, roles={"anvil": ia, "free": ifr, "bystander": -1})
    sp["axes"].update(aside_m=d_snap, behind_deg=th, phi_deg=phi, start_m=start, shift_m=abs(shift), pace=pace,
                      worn_share=[w for w in worn_m + worn_t if w is not None])
    return sp


def n1(seed, held=None):
    """The N1 twin of the A1 scene of the same seed (module doc)."""
    base = a1(seed, held)
    rnd = random.Random(seed * 1000003 + 1)
    n = c()["n1"]
    for _ in range(int(c()["redraw"])):
        sp = _n1_try(rnd, copy.deepcopy(base), n)
        if sp is not None:
            sp.update(recipe="N1", family="N1", twin_of=("A1", int(seed)))
            return sp
    raise RuntimeError(f"N1 seed {seed}: no layout in {c()['redraw']} draws")


def _n1_try(rnd, sp, n):
    mz = sp["mine_zone"]
    ia, ifr = sp["roles"]["anvil"], sp["roles"]["free"]
    held = sp["held"]
    loc = [SP.local(mz, x, z) for x, z, _ in sp["places_mine"]]
    # the bystander pair: at least bystander_m from every unit of mine (in N1 the anvil is free too), the nearer at
    # most bystander_m + bystander_spread_m; on the anvil's side away from the free unit, along my row or turned back by
    # up to 60 deg
    bm, band = float(n["bystander_m"]), float(n["bystander_spread_m"])
    db = rnd.uniform(bm, bm + band)
    away = -1.0 if loc[ifr][0] > loc[ia][0] else 1.0
    tb = math.radians(rnd.uniform(0.0, 60.0))
    by = (loc[ia][0] + away * db * math.cos(tb), loc[ia][1] - db * math.sin(tb))
    slot = rnd.randrange(3)                                             # (the bystander's slot: drawn)
    rough = [SP.world(mz, *p) for p in loc]
    rough.insert(slot, SP.world(mz, *by))
    got = snap(rough, float(mz["b"]))
    if got is None:
        return None
    zone, snapped = got
    ia2, ifr2 = [i + (1 if i >= slot else 0) for i in (ia, ifr)]
    L = [SP.local(mz, x, z) for x, z in snapped]
    dd = [math.hypot(L[slot][0] - L[j][0], L[slot][1] - L[j][1]) for j in (ia2, ifr2)]
    if not (min(dd) >= bm and min(dd) <= bm + band):
        return None
    if not all(can_stand(sp["map"], x, z, mz["b"]) for x, z in snapped):
        return None
    # the bystander's enemy: straight in front of it, the scene's start distance
    start = sp["axes"]["start_m"]
    bx, bz = SP.world(mz, L[slot][0], L[slot][1] + start)
    if not can_stand(sp["map"], bx, bz, (mz["b"] + 180.0) % 360.0):
        return None
    # the one fact changed: the enemy's plan
    ex, ez, eb = sp["places_theirs"][0]
    d0 = math.hypot(ex - snapped[ia2][0], ez - snapped[ia2][1])
    sp["lords"] = _lords(sp, snapped, [(ex, ez)], ((ex - snapped[ia2][0]) / d0, (ez - snapped[ia2][1]) / d0))
    if not can_stand(sp["map"], *sp["lords"][1]):
        return None
    plan = None
    if rnd.random() < float(n["point_share"]):
        # the run round my row: a waypoint beyond an end of my row (the end nearer the enemy first), point_round_m
        # beyond it at the row's front, then the point point_behind_m behind the row and point_aside_m beyond that end;
        # both legs >= point_clear_m from every unit of mine (edges: centres + 24 m, half a unit's diagonal), the point
        # >= point_clear_m from my lord
        clear = float(n["point_clear_m"]) + 24.0
        s_lo, s_hi = min(p[0] for p in L), max(p[0] for p in L)
        f_hi = max(p[1] for p in L)
        es, _ = SP.local(mz, ex, ez)
        lx, lz = sp["lords"][1][:2]
        for tau in sorted((-1.0, 1.0), key=lambda t: -t * (es - (s_lo + s_hi) / 2)):
            end = s_hi if tau > 0 else s_lo
            wx, wz = SP.world(mz, end + tau * rnd.uniform(*n["point_round_m"]), f_hi + 30.0)
            px, pz = SP.world(mz, end + tau * rnd.uniform(*n["point_aside_m"]),
                              L[ia2][1] - rnd.uniform(*n["point_behind_m"]))
            if (all(_seg_dist(q, (ex, ez), (wx, wz)) >= clear and _seg_dist(q, (wx, wz), (px, pz)) >= clear
                    for q in snapped) and math.hypot(px - lx, pz - lz) >= clear
                    and can_stand(sp["map"], px, pz, eb) and can_stand(sp["map"], wx, wz, eb)):
                plan = {"kind": "point", "x": px, "z": pz, "wx": wx, "wz": wz, "run": sp["plan_theirs"][0]["run"]}
                break
    if plan is None:
        plan = {"kind": "halt", "k": ia2, "gap": rnd.uniform(*n["halt_m"]), "run": sp["plan_theirs"][0]["run"]}
    keys = list(sp["mine"])
    keys.insert(slot, _keys(rnd, sp["fac"][1], 1, held)[0])
    worn_m = list(sp["worn"]["mine"])
    worn_m.insert(slot, worn_share(rnd, held))
    theirs = sp["theirs"] + _keys(rnd, sp["fac"][2], 1, held)
    worn_t = sp["worn"]["theirs"] + [worn_share(rnd, held)]
    perm = list(sp["perm"])
    perm.insert(slot, 2)
    sp.update(mine=keys, theirs=theirs, places_mine=[(x, z, float(mz["b"])) for x, z in snapped],
              places_theirs=[(ex, ez, eb), (bx, bz, (mz["b"] + 180.0) % 360.0)], zone=zone, perm=perm,
              plan_theirs=[plan, {"kind": "at", "k": slot, "run": sp["plan_theirs"][0]["run"]}],
              worn={"mine": worn_m, "theirs": worn_t}, roles={"anvil": ia2, "free": ifr2, "bystander": slot})
    sp["axes"]["worn_share"] = [w for w in worn_m + worn_t if w is not None]
    sp["n1_plan"] = plan["kind"]
    return sp


RECIPES = {"A1": a1, "N1": n1}


def scenes(recipe, n, seed0, held=None):
    return [RECIPES[recipe](seed0 + i, held) for i in range(n)]


def check_bands(specs):
    """The held-out check (design section 4, check 10): every drawn value of a scene lies in its band (held for the
    scene's held axis, train for the others), and the training and held-out bands of an axis do not overlap. Returns
    the count of values checked; asserts."""
    n = 0
    for name, ax in c()["axes"].items():
        tr, hd = ax.get("train") or [], ax.get("held") or []
        if tr and hd and isinstance(tr[0], list) and name != "factions":
            assert all(b[1] <= a[0] + 1e-9 or a[1] <= b[0] + 1e-9 for a in tr for b in hd), f"{name}: bands overlap"
        else:
            assert not any(x in hd for x in tr), f"{name}: a held-out value is a training one"
    w = c()["worn"]
    assert all(b[1] <= a[0] + 1e-9 or a[1] <= b[0] + 1e-9 for a in w["train"] for b in w["held"]), "worn bands overlap"
    for sp in specs:
        h = sp.get("held")
        for name, v in sp["axes"].items():
            if name == "worn_share":
                for x in v:
                    band = w["held"] if h == name else w["train"]
                    assert any(lo - 1e-9 <= x <= hi + 1e-9 for lo, hi in band), (sp["seed"], name, x)
                    n += 1
                continue
            which = "held" if h == name else "train"
            assert in_band(name, which, v), (sp["recipe"], sp["seed"], name, v, which)
            n += 1
        ukeys = bands("unit", "held")
        if h != "unit":
            assert not any(k in ukeys for k in sp["mine"] + sp["theirs"]), (sp["seed"], "a held-out unit drawn")
    return n


# ------------------------------------------------------------------ stage 2c proper: the generic layout and the recipes
# A recipe draws my units' rough places in my zone's axes (s right, f forward) by role, and its enemies relative to my
# snapped places; build() shuffles the slots, snaps, checks the ground and puts the enemies and the lords.
def _row(n, s0, f0, gap):
    return [(s0 + (i - (n - 1) / 2) * gap, f0) for i in range(n)]


def _ax(rnd, name, held, lo_hi=None):
    """A drawn value of an axis, or of a plain range [lo, hi] (a recipe's own setting, no held-out value)."""
    if lo_hi is not None:
        return rnd.uniform(*lo_hi)
    return draw(rnd, name, held)


def _cost_of(key):
    return float(_passports()[key]["multiplayer_cost"])


@lru_cache(maxsize=1)
def _passports():
    return json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"]


def _uneven(rnd, f1, f2, held, ratio):
    """(my key, the enemy's key) with costs apart by ratio or more, the stronger side drawn."""
    r1, r2 = _roster(f1, held), _roster(f2, held)
    strong_mine = rnd.random() < 0.5
    for _ in range(200):
        a, b = rnd.choice(r1), rnd.choice(r2)
        ca, cb = _cost_of(a), _cost_of(b)
        if (ca / cb if strong_mine else cb / ca) >= ratio:
            return a, b
    return a, b


def _matched(rnd, fac, key, held, tol):
    """An enemy key of fac whose cost is within tol of key's (the nearest when none)."""
    c0 = _cost_of(key)
    r = _roster(fac, held)
    ok = [k for k in r if abs(_cost_of(k) - c0) / c0 <= tol]
    return rnd.choice(ok) if ok else min(r, key=lambda k: abs(_cost_of(k) - c0))


def build(rnd, sp, mine, enemies, held):
    """The common part of every recipe (module doc). mine [{sf: (s, f), role, key, worn}] in role order; enemies
    [{place: f(loc) -> (s, f) start, face: f(loc) -> (s, f) the point it faces, plan: {kind, k: a role index or None,
    ...}, key, worn}] (loc: my snapped places by role, zone axes). Returns sp or None (redraw)."""
    mz = sp["mine_zone"]
    n = len(mine)
    order = list(range(n))
    rnd.shuffle(order)                                              # (slot j holds role order[j]: drawn first)
    got = snap([SP.world(mz, *mine[r]["sf"]) for r in order], float(mz["b"]))
    if got is None:
        return None
    zone, snapped = got
    if not all(can_stand(sp["map"], x, z, mz["b"]) for x, z in snapped):
        return None
    slot = {r: j for j, r in enumerate(order)}
    loc = [SP.local(mz, *snapped[slot[r]]) for r in range(n)]
    pt, plans = [], []
    for e in enemies:
        s, f = e["place"](loc)
        x, z = SP.world(mz, s, f)
        tx, tz = SP.world(mz, *e["face"](loc))
        b = _bearing(x, z, tx, tz)
        if not can_stand(sp["map"], x, z, b):
            return None
        pt.append((x, z, b))
        p = dict(e["plan"])
        if p.get("k") is not None:
            p["k"] = slot[p["k"]]
        if p.get("wp") is not None:
            p["wx"], p["wz"] = SP.world(mz, *p.pop("wp"))
        if p.get("pt") is not None:
            p["x"], p["z"] = SP.world(mz, *p.pop("pt"))
        plans.append(p)
    ex, ez = np.mean([p[0] for p in pt]), np.mean([p[1] for p in pt])
    mx, mzz = np.mean([p[0] for p in snapped]), np.mean([p[1] for p in snapped])
    dd = max(math.hypot(ex - mx, ez - mzz), 1e-6)
    lords = _lords(sp, snapped, [(p[0], p[1]) for p in pt], ((ex - mx) / dd, (ez - mzz) / dd))
    if not all(can_stand(sp["map"], *lords[s]) for s in (1, 2)):
        return None
    roles = {}
    for r, m in enumerate(mine):
        roles.setdefault(m["role"], []).append(slot[r])
    sp.update(mine=[mine[r]["key"] for r in order], theirs=[e["key"] for e in enemies],
              places_mine=[(x, z, float(mz["b"])) for x, z in snapped], places_theirs=pt, lords=lords, zone=zone,
              perm=order, plan_theirs=plans, worn={"mine": [mine[r]["worn"] for r in order],
                                                   "theirs": [e["worn"] for e in enemies]},
              roles={k: sorted(v) for k, v in roles.items()}, loc_roles=loc)
    sp["axes"]["worn_share"] = [w for w in sp["worn"]["mine"] + sp["worn"]["theirs"] if w is not None]
    assert len(sp["mine"]) <= 8 and len(sp["theirs"]) <= 8, "at most 8 units a side (review G.3)"
    return sp


def _head_on(rnd, held, start, phi_max=None):
    """(the signed approach angle, deg) of an enemy at its target: phi_deg's band, or a recipe's own (lines: head-on)."""
    phi = draw(rnd, "phi_deg", held) if phi_max is None else rnd.uniform(0.0, phi_max)
    return phi * rnd.choice((-1.0, 1.0))


def _at(k, d, phi, aside=0.0, back=0.0):
    """An enemy start d m from my unit k (role index) at the approach phi (deg; 0 = straight in front), aside m across
    and back m further along the approach; it faces k."""
    a = math.radians(phi)

    def place(loc):
        s, f = loc[k]
        return (s + (d + back) * math.sin(a) + aside * math.cos(a), f + (d + back) * math.cos(a) - aside * math.sin(a))
    return place, (lambda loc: loc[k])


def _enemy(rnd, sp, k, d, phi, held, pace, key=None, aside=0.0, back=0.0, plan=None):
    place, face = _at(k, d, phi, aside, back)
    return {"place": place, "face": face, "plan": plan or {"kind": "at", "k": k, "run": pace == "run"},
            "key": key or _keys(rnd, sp["fac"][2], 1, held)[0], "worn": worn_share(rnd, held)}


def _mine(rnd, sp, sf, role, held, key=None):
    return {"sf": sf, "role": role, "key": key or _keys(rnd, sp["fac"][1], 1, held)[0], "worn": worn_share(rnd, held)}


def _limit(sp, plans):
    if any(p.get("kind") == "script" and p.get("name") == "hold_shoot" for p in plans):
        sp["limit"] = int(c()["limit_long_s"])


def _recipe(name, fn):
    def make(seed, held=None):
        rnd = random.Random(seed)
        for _ in range(int(c()["redraw"])):
            sp = fn(rnd, held)
            if sp is not None:
                sp.update(seed=int(seed), held=held, recipe=name, family=name.split(".")[0])
                return sp
        raise RuntimeError(f"{name} seed {seed}: no layout in {c()['redraw']} draws")
    return make


def _common(rnd, held):
    sp = _base(rnd, held)
    mz = sp["mine_zone"]
    r = c()["recipes"]
    return sp, mz["depth"] / 2 - 30.0, draw(rnd, "shift_m", held) * rnd.choice((-1.0, 1.0)), r


def _a2(rnd, held):
    """A2 wide line: mine in a row (or stage 0's two rows), the enemy fewer by the free count - `at` a drawn subset
    of opposites, or a battle script."""
    sp, f_row, shift, r = _common(rnd, held)
    n = int(draw(rnd, "size", held))
    free_n = int(draw(rnd, "free_n", held))
    if n - free_n < 1:
        return None
    gap = rnd.uniform(*r["row_m"])
    two = rnd.random() < float(r["a2"]["two_rows"]) and n > 5
    sf = (_row(5, shift, f_row, gap) + _row(n - 5, shift, f_row - float(r["a2"]["rank_m"]), gap) if two
          else _row(n, shift, f_row, gap))
    mine = [_mine(rnd, sp, p, "row", held) for p in sf]
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    front = list(range(5 if two else n))
    tg = sorted(rnd.sample(front if len(front) >= n - free_n else list(range(n)), n - free_n))
    for k in tg:
        mine[k]["role"] = "anvil"
    for k in range(n):
        if k not in tg:
            mine[k]["role"] = "free"
    if rnd.random() < float(r["a2"]["script_share"]):
        name = _pick_script(rnd)
        en = [_enemy(rnd, sp, k, start, 0.0, held, pace, plan={"kind": "script", "name": name, "run": pace == "run"})
              for k in tg]
        sp["a2_mode"] = "script:" + name
    else:
        phi = _head_on(rnd, held, start, float(r["line_phi_max"]))
        en = [_enemy(rnd, sp, k, start, phi, held, pace) for k in tg]
        sp["a2_mode"] = "at"
    out = build(rnd, sp, mine, en, held)
    if out is not None:
        _limit(out, out["plan_theirs"])
        out["axes"].update(size=n, free_n=free_n, start_m=start, pace=pace, shift_m=abs(shift))
    return out


def _pick_script(rnd):
    g = scfg()["general"]["enemies"]
    w = {k: v for k, v in g.items() if k != "chain"}
    return rnd.choices(list(w), weights=list(w.values()))[0]


def _a3(rnd, held):
    """A3 reserve: a row of 2-6 and 1-2 units 60-100 m behind a gap or beyond a wing; as many enemies as the row, `at`
    their opposites. Never straight behind: a reserve's offset across each row unit's front >= orders.through_tag_m."""
    from net.tactic import cfg as tcfg
    sp, f_row, shift, r = _common(rnd, held)
    nr, nres = rnd.randint(*r["a3"]["row"]), rnd.randint(*r["a3"]["reserve"])
    if nr + nres > 8:
        return None
    gap = rnd.uniform(*r["row_m"])
    row = _row(nr, shift, f_row, gap)
    res = []
    for _ in range(nres):
        back = rnd.uniform(*r["a3"]["back_m"])
        if nr >= 2 and rnd.random() < 0.5:
            i = rnd.randrange(nr - 1)
            s = (row[i][0] + row[i + 1][0]) / 2
        else:
            end = rnd.choice((-1.0, 1.0))
            s = (max(p[0] for p in row) if end > 0 else min(p[0] for p in row)) + end * rnd.uniform(0.5, 1.0) * gap
        res.append((s, f_row - back))
    if len({round(p[0]) for p in res}) < len(res):
        return None
    mine = [_mine(rnd, sp, p, "anvil", held) for p in row] + [_mine(rnd, sp, p, "free", held) for p in res]
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    phi = _head_on(rnd, held, start, float(r["line_phi_max"]))
    en = [_enemy(rnd, sp, k, start, phi, held, pace) for k in range(nr)]
    out = build(rnd, sp, mine, en, held)
    if out is None:
        return None
    loc = out["loc_roles"]
    tt = float(tcfg()["orders"]["through_tag_m"])
    for j in range(nr, nr + nres):
        if any(abs(loc[j][0] - loc[i][0]) < tt and loc[j][1] < loc[i][1] for i in range(nr)):
            return None
    out["axes"].update(start_m=start, pace=pace, shift_m=abs(shift))
    return out


def _beyond(rnd, which, key="beyond_m"):
    """(a drawn centre distance, its least) beyond the reach and its band (review I.3 b, N): the band's top (config
    bands.reach_m, around tactic.json orders.reach_m) + recipes.<which>.<key> drawn; the least (the check on the snapped
    places) the band's top + the low end of recipes.<which>.beyond_m - as N1's bystanders."""
    top = float(c()["bands"]["reach_m"][1])
    r = c()["recipes"][which]
    lo, hi = r[key]
    return top + rnd.uniform(float(lo), float(hi)), top + float(r["beyond_m"][0])


def _apart(out, least):
    """The recipe's geometry re-checked on the snapped places: every two units of mine >= least apart (centres)."""
    L = out["loc_roles"]
    return all(math.hypot(L[i][0] - L[j][0], L[i][1] - L[j][1]) >= least
               for i in range(len(L)) for j in range(i + 1, len(L)))


def _a4(rnd, held, late=False):
    """A4 freed late: a row of 2-4 against as many `at` their opposites, the pairs uneven (cost ratio >= a4.ratio, the
    stronger side drawn) so that the fights end apart. late (A4L, review B 'the neighbour's fight is over or loose when
    the unit frees'), the one fact changed: the pairs stand beyond the reach and its band apart (_beyond 'late') - so a
    unit that frees has no pinned enemy within reach."""
    sp, f_row, shift, r = _common(rnd, held)
    n = rnd.randint(*r["a4"]["row"])
    gap, least = _beyond(rnd, "late") if late else (rnd.uniform(*r["row_m"]), 0.0)
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    phi = _head_on(rnd, held, start, float(r["line_phi_max"]))
    mine, en = [], []
    for k, p in enumerate(_row(n, shift, f_row, gap)):
        a, b = _uneven(rnd, sp["fac"][1], sp["fac"][2], held, float(r["a4"]["ratio"]))
        mine.append(_mine(rnd, sp, p, "anvil", held, key=a))
        en.append(_enemy(rnd, sp, k, start, phi, held, pace, key=b))
    out = build(rnd, sp, mine, en, held)
    if out is not None and late and not _apart(out, least):
        return None
    if out is not None:
        out["limit"] = int(c()["limit_long_s"])
        out["axes"].update(start_m=start, pace=pace, shift_m=abs(shift))
    return out


def _a5(rnd, held, late=False):
    """A5 uneven line: 4-8 against as many under `human` / `nearest`, mixed strengths (keys drawn freely). late (A5L):
    as A4L - late.a5_n units beyond the reach and its band apart (_beyond 'late' a5_beyond_m: 4 pairs fill the zone's
    12 tiles), each against its own enemy `at` it."""
    sp, f_row, shift, r = _common(rnd, held)
    n = int(r["late"]["a5_n"]) if late else int(draw(rnd, "size_line", held))
    gap, least = _beyond(rnd, "late", "a5_beyond_m") if late else (rnd.uniform(*r["a5"]["row_m"]), 0.0)
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    name = rnd.choice(r["a5"]["scripts"])
    mine = [_mine(rnd, sp, p, "anvil", held) for p in _row(n, shift, f_row, gap)]
    en = []
    for k in range(n):
        plan = ({"kind": "at", "k": k, "run": pace == "run"} if late
                else {"kind": "script", "name": name, "run": pace == "run"})
        en.append(_enemy(rnd, sp, k, start, 0.0, held, pace, plan=plan))
    out = build(rnd, sp, mine, en, held)
    if out is not None and late and not _apart(out, least):
        return None
    if out is not None:
        out["limit"] = int(c()["limit_long_s"])
        out["axes"].update(start_m=start, pace=pace, shift_m=abs(shift), **({} if late else {"size_line": n}))
        out["a5_script"] = name
    return out


def _a6(rnd, held, variant=None):
    """A6 second coming (the game's `tight` layout): (i) queue - u and 1-2 free neighbours 36-130 m aside, E1 `at u`, E2
    `at u` lag_m behind E1 and 0-40 m aside; (ii) late wing - a row of 3-5, as many `at` opposites and one more `at`
    the flank unit from 10-40 s, a spare 48-100 m beyond that flank; (iii) E2 `at` the free unit itself lag_m behind E1
    (its label follows the game's verdict own: within come_reach_m the teacher keeps)."""
    sp, f_row, shift, r = _common(rnd, held)
    a6 = r["a6"]
    v = variant or rnd.choices(("i", "ii", "iii"), weights=a6["variants"])[0]
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    if v == "ii":
        nr = rnd.randint(*a6["wing_row"])
        gap = rnd.uniform(*r["row_m"])
        row = _row(nr, shift, f_row, gap)
        side = rnd.choice((-1.0, 1.0))
        flank = nr - 1 if side > 0 else 0
        spare = (row[flank][0] + side * rnd.uniform(*a6["spare_m"]), f_row)
        mine = [_mine(rnd, sp, p, "anvil", held) for p in row] + [_mine(rnd, sp, spare, "free", held)]
        phi = _head_on(rnd, held, start, float(r["line_phi_max"]))
        en = [_enemy(rnd, sp, k, start, phi, held, pace) for k in range(nr)]
        en.append(_enemy(rnd, sp, flank, start, phi, held, pace, plan={
            "kind": "at", "k": flank, "from_s": rnd.uniform(*a6["from_s"]), "run": pace == "run"}))
        lag = None
    else:
        nf = rnd.randint(*a6["free"]) if v == "i" else 1
        sides = [rnd.choice((-1.0, 1.0))] if nf == 1 else [-1.0, 1.0]
        mine = [_mine(rnd, sp, (shift, f_row), "anvil", held)]
        for sd in sides:
            mine.append(_mine(rnd, sp, (shift + sd * rnd.uniform(*a6["aside_m"]), f_row), "free", held))
        phi = _head_on(rnd, held, start, float(r["line_phi_max"]))
        lag = draw(rnd, "lag_m", held)
        tgt2 = 0 if v == "i" else 1
        en = [_enemy(rnd, sp, 0, start, phi, held, pace),
              _enemy(rnd, sp, 0, start, phi, held, pace, aside=rnd.uniform(*a6["aside2_m"]) * rnd.choice((-1, 1)),
                     back=lag, plan={"kind": "at", "k": tgt2, "run": pace == "run"})]
    out = build(rnd, sp, mine, en, held)
    if out is not None:
        out["axes"].update(start_m=start, pace=pace, shift_m=abs(shift), **({"lag_m": lag} if lag is not None else {}))
        out["a6"] = v
    return out


def _e2(rnd, held):
    """E2 far: A1 with the free unit beyond the reach and its band from the anvil (_beyond 'e2': one threshold)."""
    sp, f_row, shift, r = _common(rnd, held)
    d, least = _beyond(rnd, "e2")
    side = rnd.choice((-1.0, 1.0))
    mine = [_mine(rnd, sp, (shift, f_row), "anvil", held), _mine(rnd, sp, (shift + side * d, f_row), "free", held)]
    start, pace = draw(rnd, "start_m", held), draw(rnd, "pace", held)
    en = [_enemy(rnd, sp, 0, start, _head_on(rnd, held, start), held, pace)]
    out = build(rnd, sp, mine, en, held)
    if out is not None and not _apart(out, least):
        return None
    if out is not None:
        out["axes"].update(start_m=start, pace=pace, shift_m=abs(shift))
    return out


# ------------------------------------------------------------------ the twins of a cause scene
def _clear_path(pts, a, b, clear):
    return all(_seg_dist(q, a, b) >= clear for q in pts)


def _point_plan(rnd, sp, k_slot, e_xy, n):
    """A pass-by plan for the enemy at e_xy (module doc of n1): round my row through a waypoint, then a point behind
    it; None when neither end gives a clear way."""
    mz = sp["mine_zone"]
    L = [SP.local(mz, x, z) for x, z, _ in sp["places_mine"]]
    mine_xy = [(x, z) for x, z, _ in sp["places_mine"]]
    clear = float(n["point_clear_m"]) + 24.0
    s_lo, s_hi = min(p[0] for p in L), max(p[0] for p in L)
    f_hi, f_lo = max(p[1] for p in L), min(p[1] for p in L)
    es, _ = SP.local(mz, *e_xy)
    lx, lz = sp["lords"][1][:2]
    for tau in sorted((-1.0, 1.0), key=lambda t: -t * (es - (s_lo + s_hi) / 2)):
        end = s_hi if tau > 0 else s_lo
        w = SP.world(mz, end + tau * rnd.uniform(*n["point_round_m"]), f_hi + 30.0)
        p = SP.world(mz, end + tau * rnd.uniform(*n["point_aside_m"]), f_lo - rnd.uniform(*n["point_behind_m"]))
        if (_clear_path(mine_xy, e_xy, w, clear) and _clear_path(mine_xy, w, p, clear)
                and math.hypot(p[0] - lx, p[1] - lz) >= clear and can_stand(sp["map"], *p, 0.0)
                and can_stand(sp["map"], *w, 0.0)):
            return {"kind": "point", "x": p[0], "z": p[1], "wx": w[0], "wz": w[1]}
    return None


def twin_n1(base, rnd):
    """N1 of any cause scene (module doc of n1, generalised): every enemy's plan becomes a pass-by (point, its way
    clear of my units) or, where none, a halt; a bystander pair >= bystander_m from every unit of mine. None: redraw."""
    sp = copy.deepcopy(base)
    n = c()["n1"]
    mz = sp["mine_zone"]
    L = [SP.local(mz, x, z) for x, z, _ in sp["places_mine"]]
    bm, band = float(n["bystander_m"]), float(n["bystander_spread_m"])
    if len(sp["mine"]) >= 8 or len(sp["theirs"]) >= 8:
        return None
    side = rnd.choice((-1.0, 1.0))
    s_end = max(p[0] for p in L) if side > 0 else min(p[0] for p in L)
    by = (s_end + side * rnd.uniform(bm, bm + band), max(p[1] for p in L) - rnd.uniform(0.0, 60.0))
    slot = rnd.randrange(len(L) + 1)
    rough = [SP.world(mz, *p) for p in L]
    rough.insert(slot, SP.world(mz, *by))
    got = snap(rough, float(mz["b"]))
    if got is None:
        return None
    zone, snapped = got
    Ls = [SP.local(mz, x, z) for x, z in snapped]
    dd = [math.hypot(Ls[slot][0] - q[0], Ls[slot][1] - q[1]) for j, q in enumerate(Ls) if j != slot]
    if not (min(dd) >= bm and min(dd) <= bm + band) or not all(can_stand(sp["map"], x, z, mz["b"]) for x, z in snapped):
        return None
    shiftk = lambda k: k + (1 if k >= slot else 0)                                 # noqa: E731
    start = sp["axes"].get("start_m", 150.0)
    bx, bz = SP.world(mz, Ls[slot][0], Ls[slot][1] + start)
    if not can_stand(sp["map"], bx, bz, (mz["b"] + 180.0) % 360.0):
        return None
    sp["places_mine"] = [(x, z, float(mz["b"])) for x, z in snapped]
    plans = []
    for (ex, ez, _), p in zip(sp["places_theirs"], sp["plan_theirs"]):
        q = _point_plan(rnd, sp, None, (ex, ez), n)
        run = p.get("run", True)
        if q is None:
            q = {"kind": "halt", "k": shiftk(p["k"]) if p.get("k") is not None else shiftk(0),
                 "gap": rnd.uniform(*n["halt_m"])}
        q["run"] = run
        plans.append(q)
    plans.append({"kind": "at", "k": slot, "stay": True, "run": sp["plan_theirs"][0].get("run", True)})
    sp["mine"].insert(slot, _keys(rnd, sp["fac"][1], 1, sp["held"])[0])
    sp["worn"]["mine"].insert(slot, worn_share(rnd, sp["held"]))
    sp["theirs"].append(_keys(rnd, sp["fac"][2], 1, sp["held"])[0])
    sp["worn"]["theirs"].append(worn_share(rnd, sp["held"]))
    sp["places_theirs"] = list(sp["places_theirs"]) + [(bx, bz, (mz["b"] + 180.0) % 360.0)]
    sp["roles"] = {k: [shiftk(j) for j in v] for k, v in sp["roles"].items()}
    sp["roles"]["bystander"] = [slot]
    sp.update(zone=zone, plan_theirs=plans, n1_plan=",".join(sorted({q["kind"] for q in plans[:-1]})),
              limit=int(c()["limit_s"]))
    sp["axes"]["worn_share"] = [w for w in sp["worn"]["mine"] + sp["worn"]["theirs"] if w is not None]
    sp.pop("end_s", None)
    return sp


def twin_own(base, rnd, early=False):
    """N2 'own enemy' (review B: arriving own_lag_s after the anvil's - it starts lag x its pace farther, coming all the
    way, so its counter's reach and come_own see it; matched cost +-own_cost_tol; the scene
    ends own_end_s after the anvil's expected contact) or, early, E3 'busy' (it starts at busy_share of the start
    distance and arrives before the pin): one more enemy `at` each free unit, as many as 8 a side allow."""
    sp = copy.deepcopy(base)
    o = c()["own"]
    mz = sp["mine_zone"]
    free = sp["roles"].get("free", [])
    if not free:
        return None
    start = sp["axes"].get("start_m", 150.0)
    run = sp["plan_theirs"][0].get("run", True)
    v = float(o["run_ms"] if run else o["walk_ms"])
    lag = rnd.uniform(*o["lag_s"])
    added = 0
    for g in free:
        if len(sp["theirs"]) >= 8:
            break
        gs, gf = SP.local(mz, *sp["places_mine"][g][:2])
        d = start * float(o["busy_share"]) if early else start + lag * v      # (coming all the way: lag x its pace)
        x, z = SP.world(mz, gs, gf + d)
        if not can_stand(sp["map"], x, z, (mz["b"] + 180.0) % 360.0):
            return None
        key = _matched(rnd, sp["fac"][2], sp["mine"][g], sp["held"], float(o["cost_tol"]))
        sp["theirs"].append(key)
        sp["worn"]["theirs"].append(worn_share(rnd, sp["held"]))
        sp["places_theirs"] = list(sp["places_theirs"]) + [(x, z, (mz["b"] + 180.0) % 360.0)]
        p = {"kind": "at", "k": g, "run": run}
        sp["plan_theirs"] = list(sp["plan_theirs"]) + [p]
        added += 1
    if not added:
        return None
    if not early:
        sp["end_s"] = min(float(sp["limit"]), start / v + lag + float(o["end_after_s"]))
        sp["own_lag_s"] = lag
    sp["axes"]["worn_share"] = [w for w in sp["worn"]["mine"] + sp["worn"]["theirs"] if w is not None]
    return sp


def twin_halt(base, rnd):
    """E1 'halt': every enemy halts short (n1.halt_m from the nearest unit of mine) and stands; no fight anywhere."""
    sp = copy.deepcopy(base)
    sp["plan_theirs"] = [{"kind": "halt", "k": p.get("k") if p.get("k") is not None else 0,
                          "gap": rnd.uniform(*c()["n1"]["halt_m"]), "run": p.get("run", True)} for p in sp["plan_theirs"]]
    return sp


CAUSE = {"A1": None, "A2": _a2, "A3": _a3, "A4": _a4, "A5": _a5, "A6": _a6}
TWINS = {"N1": (twin_n1, ("A1", "A2", "A3", "A6")), "N2": (lambda b, r: twin_own(b, r), ("A1",)),
         "E1": (twin_halt, ("A1", "A2", "A3", "A6")), "E3": (lambda b, r: twin_own(b, r, early=True), ("A1", "A3"))}


def make(recipe, seed, held=None, bases=None):
    """One scene of any recipe: a cause recipe (A1-A6), a late twin (A4L, A5L), a twin of a cause scene of the same
    seed (N1, N2, E1, E3 - their base recipe drawn from the seed among TWINS' bases) or E2 far."""
    if recipe == "A1":
        sp = a1(seed, held)
        sp["roles"] = {k: ([v] if v >= 0 else []) if isinstance(v, int) else v for k, v in sp["roles"].items()}
        return sp
    if recipe in ("A2", "A3", "A4", "A5", "A6"):
        return _recipe(recipe, CAUSE[recipe])(seed, held)
    if recipe == "A4L":
        return _recipe("A4L", lambda rnd, h: _a4(rnd, h, late=True))(seed, held)
    if recipe == "A5L":
        return _recipe("A5L", lambda rnd, h: _a5(rnd, h, late=True))(seed, held)
    if recipe == "E2":
        return _recipe("E2", _e2)(seed, held)
    fn, bases0 = TWINS[recipe]
    bases = [b for b in bases0 if bases is None or b in bases] or list(bases0)
    rnd = random.Random(seed * 7919 + 17)
    for t in range(int(c()["redraw"])):
        b = bases[(seed + t) % len(bases)]
        base = make(b, seed + t * 100003, held)
        sp = fn(base, rnd)
        if sp is not None:
            sp.update(recipe=recipe, family=recipe, twin_of=(b, int(base["seed"])), seed=int(seed), held=held)
            return sp
    raise RuntimeError(f"{recipe} seed {seed}: no twin in {c()['redraw']} draws")


def set_specs(name, seed0=None, counts=None):
    """A fixed set by seed (config sets): {recipe: count}, the seeds seed0 + a recipe's offset + i."""
    st = c()["sets"][name]
    seed0 = int(st["seed"] if seed0 is None else seed0)
    out = []
    for j, (rec, n) in enumerate((counts or st["counts"]).items()):
        out += [make(rec, seed0 + 10000 * j + i) for i in range(int(n))]
    return out


def held_specs(seed0=None):
    """Set H: each held-out axis x (h_cause scenes of the recipes that carry the axis + h_twin N1 / N2 scenes), exactly
    that one axis at its held-out value."""
    st = c()["sets"]["H"]
    seed0 = int(st["seed"] if seed0 is None else seed0)
    out = []
    for j, (axis, recs) in enumerate(st["axes"].items()):
        for i in range(int(st["cause"])):
            out.append(make(recs[i % len(recs)], seed0 + 10000 * j + i, held=axis))
        tw = [t for t in ("N1", "N2") if any(b in recs for b in TWINS[t][1])] or ["N1"]
        for i in range(int(st["twin"])):
            out.append(make(tw[i % len(tw)], seed0 + 10000 * j + 5000 + i, held=axis, bases=recs))
    return out
