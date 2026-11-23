"""S0-1's generator (tmp/drill_s1/design_merged.md section 4; design_opus section 5): the scenes, their twins, the held
axes, the general battles - plain python (numpy for the finder; no torch).

    scene(seed, split, held=None, twin=None)   one battle's spec (kind 's1'): a CAUSE scene (we hold, the enemy comes,
                 the zone holds a crest the teacher takes; its Delta band bin drawn first, BAND_BINS) or its twin of the
                 same seed, armies and zone differing only by the cause: 'flat' (the flat field - flat is flat), 'shift'
                 (the ridge 200 m forward, s1_ground), 'low' (a low ridge: the pinned candidate's Delta band in LOW_BAND),
                 'role' (our mission attack: the teacher keeps). The shift and low twins keep the cause scene's candidate
                 geometry (s1.geom: offsets, count, order; crest.at) - they differ by the ground's profile only, the role
                 twin by the mission only (Astra s1_step2_review 3). Every draw's layout stands (KEEP's line and the
                 enemy's rows off blocked ground on the cause's map and its twin's - crest.layout_ok; the teacher's line,
                 the candidates and the foot carry it in their validity); rejected draws are counted (s1.rejects).
                 held - one held axis (HELD) at its held value, the others taught.
    twin_kind(seed)  flat 0.2 / shift 0.2 / role 0.4 / low 0.2 (design_merged 4: flat or shifted 40 %, role 40 %, low
                 20 %); on a captured map shift and low fall back to flat (no ridge to move or lower) - recorded
    general(seed, split)  a stage-0 general battle (att 0; flat or plains, Kislev's zones; the teacher reads its
                 ground); outside the exam's splits no held value (eight units, the teacher at a held distance or band)
    found(spec) / label(spec)  the finder's tokens of the spec's ground and zone, the teacher's choice (0 KEEP, k crest)
    line(spec, arm)  the line an arm stands on: DEFAULT the centre, TEACH the teacher's, DUMB the foot (the legal line of
                 the lowest band), PLACE (flat twins only) the cause scene's teacher line on the flat ground
    places(spec, arm) [(x, z, b)] my units; binds(spec) -> {axis: bool} the held value at tick 0
    identity(spec) / family(spec)   (gen, twin, seed) / (gen, seed), gen s1.scene | s1.general: a twin is its cause
                 scene's family
Every number of the axes: AXES (design_opus 5's table); seeds: one Python stream per scene (no substitute seeds: a
rejected draw is drawn again inside the same stream, at most TRIES times).
"""
import math
import random
from functools import lru_cache

from net.strategist import crest as CR
from net.strategist import s1_ground as SG

KIND = "s1"
LIMIT_S = 600
TRIES = 400
TRIES_HELD = 4000
_TRIES_WHY = ("a held axis draws its cause 10x longer in the same stream (held distance seed 96130053 found none in 400, "
              "22.11): a seed that found one within 400 draws is unchanged; the captured draw keeps 400 (its failure falls back "
              "to a synthetic ridge - more tries there would change scenes already played)")
HELD = ("ground", "size", "band", "distance", "enemy", "shift", "zone")
AXES = {
    "ground": {"taught": {"captured": 0.3, "synthetic": 0.7}, "held": "kho_wastes_01"},
    "size": {"taught": (2, 7), "held": 8},
    "band": {"taught": (0.05, 0.30), "held": (0.30, 0.42)},
    "distance": {"taught": (0.0, 110.0), "held": (110.0, 150.0)},
    "enemy": {"taught": {"human": 0.6, "nearest": 0.4}, "held": "ai_game"},
    "shift": {"taught": (0.0, 60.0), "held": (60.0, 100.0)},
    "zone": {"taught": (416.0, 176.0, 368.0), "held": (200.0, 120.0, 240.0)},
}
_AXES_WHY = ("design_opus 5 'Axes' (taught / held): ground plains, river, synthetic / kho_wastes_01; size 2-7 / 8; crest "
             "Delta band 0.08-0.30 / 0.30-0.42; crest from the zone's centre <= 110 / 110-150 m; enemy human 0.6, nearest "
             "0.4 / ai_game; the enemy's shift along the front <= 60 (net/config/stage0.json general enemy_dz_m) / 60-100 "
             "m; zone 416 x 176 (Kislev's east.xml, centres 368 m apart) / 200 x 120 (the capture's, 240 m apart: "
             "sim/maps meta deployment_zones). Captured / synthetic 0.3 / 0.7: design_merged 0 - the drill learns mostly "
             "on synthetic ridges (the captured maps hold 15 sites at 3 units, 7 at 4, 3 at 5+)")
CAPTURED = {"ksl_plains_01": 0.8, "ksl_river_01": 0.2}
_CAPTURED_WHY = "the plains hold 6 of the 8 captured firing zones at 3 units, the river 1 (tmp/drill_s1/c/counts.txt)"
TWINS = {"flat": 0.2, "shift": 0.2, "role": 0.4, "low": 0.2}
ZONE_OK = {"blocked": 0.2, "water": 0.1}
_ZONE_OK_WHY = "design_opus 5: the zone pair moved where it fits - <= 20 % cannot stand, <= 10 % water in each zone"
MAP_HALF = 512.0
MARGIN_M = 20.0
ENEMY_ROWS_DZ = 0.0
GENERAL = {"maps": {"flat": 0.5, "ksl_plains_01": 0.5}, "enemies": {"human": 0.4 / 0.7, "nearest": 0.2 / 0.7,
                                                                       "hold_shoot": 0.1 / 0.7}}
_GENERAL_WHY = ("net/config/stage0.json general: maps flat / plains 50 / 50, Kislev's zones either side, the enemies of "
                "the stage without 'chain' (renormalised: human 0.57, nearest 0.29, hold_shoot 0.14), sizes 2-4 60 %, "
                "5-8 40 %")


# ------------------------------------------------------------------ zones
def zone_pair(cx, cz, b, front, depth, apart):
    a = math.radians(b)
    mine = {"x": cx, "z": cz, "b": b % 360.0, "front": front, "depth": depth}
    ez = {"x": cx + apart * math.sin(a), "z": cz + apart * math.cos(a), "b": (b + 180.0) % 360.0, "front": front,
          "depth": depth}
    return mine, ez


def _corners_in(z):
    for s in (-1, 1):
        for f in (-1, 1):
            x, zz = CR.world(z, s * z["front"] / 2, f * z["depth"] / 2)
            if abs(x) > MAP_HALF - MARGIN_M or abs(zz) > MAP_HALF - MARGIN_M:
                return False
    return True


def _zone_ok(G, z):
    import numpy as np
    s = np.linspace(-z["front"] / 2, z["front"] / 2, 27)
    f = np.linspace(-z["depth"] / 2, z["depth"] / 2, 12)
    S, F = np.meshgrid(s, f)
    a = math.radians(z["b"])
    X = z["x"] + S * math.cos(a) + F * math.sin(a)
    Z = z["z"] - S * math.sin(a) + F * math.cos(a)
    blk = float((CR._node(G, "blk", X, Z) > 0).mean())
    water = 0.0
    if G["g"] is not None and G["water"]:
        water = float(np.isin(CR._node(G, "g", X, Z), G["water"]).mean())
    return blk <= ZONE_OK["blocked"] and water <= ZONE_OK["water"]


# ------------------------------------------------------------------ the finder on a spec
@lru_cache(maxsize=4096)
def _found(mp, zkey, n, geom=None):
    z = dict(zip(("x", "z", "b", "front", "depth"), zkey))
    if geom is not None:
        return CR.at(SG.ground(mp), z, n, geom)
    return CR.find(SG.ground(mp), z, n)


def _zkey(z):
    return tuple(round(float(z[k]), 4) for k in ("x", "z", "b", "front", "depth"))


def _geom(fd):
    return tuple((round(float(t["s"]), 4), round(float(t["f"]), 4), round(float(t["turn"]), 4)) for t in fd["cands"])


def found(spec):
    g = (spec.get("s1") or {}).get("geom")
    return _found(spec["map"], _zkey(spec["mine_zone"]), len(spec["mine"]), None if g is None else tuple(map(tuple, g)))


def label(spec):
    """The teacher (crest.teach): 0 KEEP, k >= 1 the k-th candidate."""
    return CR.teach(found(spec), attack=bool(spec["att"]))


BAND_BINS = ((0.05, 0.08, 0.2), (0.08, 0.12, 0.25), (0.12, 0.2, 0.3), (0.2, 0.30, 0.25))
_BAND_BINS_WHY = ("Astra s1_step2_review 7e: the exam's bins judge the teacher's thresholds (it fires from Delta band 0.05) "
                  "- a taught cause scene draws its Delta band bin first (weights 0.2 / 0.25 / 0.3 / 0.25 over 0.05-0.08, "
                  "0.08-0.12, 0.12-0.2, 0.2-0.30), so each firing bin holds >= 30 of 256 taught causes")


def _band_range(rnd, held):
    if held == "band":
        return tuple(AXES["band"]["held"])
    if held is not None:                                     # (another held axis: the whole taught range, no bin)
        return BAND_BINS[0][0], BAND_BINS[-1][1]
    lo, hi, _ = rnd.choices(BAND_BINS, weights=[w for _, _, w in BAND_BINS])[0]
    return lo, hi


def _cause_ok(fd, band_range, held):
    k = CR.teach(fd, attack=False)
    if not k:
        return None
    c = fd["cands"][k - 1]
    db = c["band"] - fd["centre"]["band"]
    dist = math.hypot(c["s"], c["f"])
    lo, hi = band_range
    if not lo <= db < hi + (1e-9 if held == "band" else 0.0):
        return None
    lo, hi = AXES["distance"]["held" if held == "distance" else "taught"]
    if not lo <= dist <= hi:
        return None
    return {"k": k, "delta_band": round(db, 4), "dist": round(dist, 1), "band": round(c["band"], 4)}


LOW_BAND = (0.0, 0.05)
_LOW_BAND_WHY = ("the low twin's pinned candidate keeps Delta band in [0, 0.05): below the teacher's delta (0.05) and not "
                 "a dip (s1_ground RIDGE low_band 0.02-0.045 is the recipe's aim; the measured band is checked here)")


def _layout_ok(mp, mz, n, ez, n_theirs, dz):
    """KEEP's line (the zone's centre) and the enemy's rows off blocked ground (crest.layout_ok; Astra 4b). The
    teacher's, the candidates' and the foot's layouts are part of their validity (crest.lines / at)."""
    from net.stage0 import specs as SP
    fd = _found(mp, _zkey(mz), n)
    return bool(fd["centre"]["layout"]) and CR.layout_ok(SG.ground(mp), SP.rows(ez, n_theirs, dz))


def _twin_ok(mp_twin, mz, n, geom, kind, k):
    """The twin's own checks on its map: the teacher silent on the pinned geometry; the low twin's band interval on
    the cause's candidate k (Y2: not only KEEP)."""
    fd = _found(mp_twin, _zkey(mz), n, geom)
    if CR.teach(fd, attack=False):
        return False
    if kind == "low":
        c = fd["cands"][k - 1]
        if not LOW_BAND[0] <= c["band"] - fd["centre"]["band"] < LOW_BAND[1]:
            return False
    return True


# ------------------------------------------------------------------ the scenes
def twin_kind(seed):
    rnd = random.Random(seed * 7 + 3)
    return rnd.choices(list(TWINS), weights=list(TWINS.values()))[0]


def _pool_of(split, held):
    if split in ("taught", "held"):
        return "held_band" if held == "band" else ("held" if split == "held" else "taught")
    return split


DISTANCE_N = (2, 4)
_DISTANCE_N_WHY = ("a line 110-150 m from the zone's middle fits a 416 x 176 zone only for a front <= 179 m (4 units: "
                   "|s| <= 208 - 15 - 89.5, |f| <= 73) - the held 'distance' scenes draw 2-4 units")


ZONE_N = (2, 3)
_ZONE_N_WHY = "a 200 m zone holds a line of front <= 170 m (15 m margins): 3 units (131 m) at most - the held 'zone' draws 2-3"


def _n_of(rnd, held):
    lo, hi = DISTANCE_N if held == "distance" else (ZONE_N if held == "zone" else AXES["size"]["taught"])
    return AXES["size"]["held"] if held == "size" else rnd.randint(lo, hi)


def _zone_dims(held):
    return AXES["zone"]["held" if held == "zone" else "taught"]


def _draw_captured(rnd, mp, n, held, band, nt, dz, rej):
    G = SG.ground(mp)
    front, depth, apart = _zone_dims(held)
    for _ in range(TRIES):
        b = 15.0 * rnd.randrange(24)
        cx, cz = rnd.uniform(-MAP_HALF, MAP_HALF), rnd.uniform(-MAP_HALF, MAP_HALF)
        mz, ez = zone_pair(cx, cz, b, front, depth, apart)
        if not (_corners_in(mz) and _corners_in(ez) and _zone_ok(G, mz) and _zone_ok(G, ez)):
            continue
        ok = _cause_ok(_found(mp, _zkey(mz), n), band, held)
        if not ok:
            continue
        if not _layout_ok(mp, mz, n, ez, nt, dz):            # (a captured map's twin: the same map or the flat field)
            rej["layout"] += 1
            continue
        return mz, ez, ok
    return None


def _draw_synth(rnd, pool, n, held, band, nt, dz, kind, rej):
    front, depth, apart = _zone_dims(held)
    npool = SG.POOLS[pool][1]
    for _ in range(TRIES if held is None else TRIES_HELD):
        rej["draws_synth"] += 1
        k = rnd.randrange(npool)
        e = SG.entry(pool, k)
        mp = SG.name(pool, k, "ridge")
        b = e["phi"] + rnd.uniform(-20.0, 20.0)
        a = math.radians(b)
        # (the zone's middle: the crest -f_off ahead of it, s_off along the crest)
        f_off = rnd.uniform(-depth / 2 + 20.0, depth / 2 - 20.0)
        if held == "distance":                    # (the crest's middle 100-145 m to a side: its best line 110-150 m out)
            s_off = rnd.choice((-1.0, 1.0)) * rnd.uniform(100.0, min(145.0, front / 2 - 40.0))
            f_off = rnd.uniform(-30.0, 30.0)
        else:
            s_off = rnd.uniform(-min(e["length"] / 2, front / 2 - 40.0), min(e["length"] / 2, front / 2 - 40.0))
        cx = e["cx"] - f_off * math.sin(a) - s_off * math.cos(a)
        cz = e["cz"] - f_off * math.cos(a) + s_off * math.sin(a)
        mz, ez = zone_pair(cx, cz, b, front, depth, apart)
        if not (_corners_in(mz) and _corners_in(ez)):
            continue
        G = SG.ground(mp)
        if not (_zone_ok(G, mz) and _zone_ok(G, ez)):
            continue
        fd = _found(mp, _zkey(mz), n)
        ok = _cause_ok(fd, band, held)
        if not ok:
            continue
        # (the base without the ridge must not itself hold a crest the teacher takes: the twins differ by the cause)
        if CR.teach(_found(SG.name(pool, k, "none"), _zkey(mz), n), attack=False):
            rej["base_fires"] += 1
            continue
        tw = {"flat": "flat", "shift": SG.name(pool, k, "shift"), "low": SG.name(pool, k, "low"), "role": mp}[kind]
        if not (_layout_ok(mp, mz, n, ez, nt, dz) and _layout_ok(tw, mz, n, ez, nt, dz)):
            rej["layout"] += 1
            continue
        if kind in ("shift", "low") and not _twin_ok(tw, mz, n, _geom(fd), kind, ok["k"]):
            rej["twin_" + kind] += 1
            continue
        return mp, mz, ez, ok, k
    return None


def scene(seed, split, held=None, twin=None):
    """(module doc)"""
    from collections import Counter
    from net.stage0 import specs as SP
    from net.stage0 import cfg as scfg
    assert held is None or held in HELD, held
    assert twin in (None,) + tuple(TWINS), twin
    rnd = random.Random(int(seed))
    kind = twin_kind(seed)                                   # (the seed's twin: its checks are made for the cause too)
    n = _n_of(rnd, held)
    f1, f2, mine, theirs = SP._sides(rnd, n)
    enemy = AXES["enemy"]["held"] if held == "enemy" else rnd.choices(
        list(AXES["enemy"]["taught"]), weights=list(AXES["enemy"]["taught"].values()))[0]
    lo, hi = AXES["shift"]["held" if held == "shift" else "taught"]
    dz = rnd.choice((-1.0, 1.0)) * rnd.uniform(lo, hi)
    band = _band_range(rnd, held)
    ground_kind = "captured" if held == "ground" else rnd.choices(
        list(AXES["ground"]["taught"]), weights=list(AXES["ground"]["taught"].values()))[0]
    pool = _pool_of(split, held)
    synth_k = None
    rej = Counter()
    if ground_kind == "captured":
        mp = AXES["ground"]["held"] if held == "ground" else rnd.choices(list(CAPTURED), weights=list(CAPTURED.values()))[0]
        got = _draw_captured(rnd, mp, n, held, band, len(theirs), dz, rej)
        if got is None:                                      # (no firing placement in TRIES: a synthetic ridge instead)
            assert held != "ground", f"seed {seed}: no held-ground cause in {TRIES} draws"
            ground_kind = "synthetic"
        else:
            mz, ez, ok = got
    if ground_kind == "synthetic":
        got = _draw_synth(rnd, pool, n, held, band, len(theirs), dz, kind, rej)
        assert got is not None, f"seed {seed}: no synthetic cause in {TRIES if held is None else TRIES_HELD} draws ({split}, held {held}, band {band})"
        mp, mz, ez, ok, synth_k = got
    fd = _found(mp, _zkey(mz), n)
    cause_line = {k: fd["cands"][ok["k"] - 1][k] for k in ("x", "z", "b", "front", "band", "s", "f", "turn")}
    sp = {"kind": KIND, "seed": int(seed), "split": split, "held": held, "twin": twin, "map": mp, "att": 0,
          "fac": {1: f1, 2: f2}, "factions": {1: scfg()["factions"][f1], 2: scfg()["factions"][f2]},
          "mine": mine, "theirs": theirs, "mine_zone": mz, "enemy_zone": ez, "enemy": enemy, "dz": dz,
          "places_theirs": SP.rows(ez, len(theirs), dz), "reserve": [], "limit": LIMIT_S,
          "s1": {"cause": twin is None, "ground": ground_kind, "pool": pool if synth_k is not None else None,
                 "synth_k": synth_k, "axes": {"n": n, "delta_band": ok["delta_band"], "dist": ok["dist"],
                                              "enemy": enemy, "dz": round(dz, 1), "zone": [mz["front"], mz["depth"]],
                                              "map": mp, "band_bin": list(band)},
                 "cause_line": cause_line, "rejects": dict(rej)}}
    if twin is not None:
        sp["s1"]["twin_asked"] = twin
        if twin == "role":
            sp["att"] = 1
            sp["s1"]["twin_made"] = "role"
        elif twin == "flat" or synth_k is None:              # (flat: the flat field - flat must be flat, Astra 1; a
            sp["map"], sp["s1"]["twin_made"] = "flat", "flat"   # captured map has no ridge to move or lower)
        else:
            sp["map"] = SG.name(pool, synth_k, twin)
            sp["s1"]["twin_made"] = twin
            sp["s1"]["geom"] = [list(g) for g in _geom(fd)]  # (the cause's candidate geometry, re-measured: Astra 3)
            assert not label(sp), f"seed {seed}: the {twin} twin fires"
    return sp


EXAM_SPLITS = ("G", "taught", "held")


def _held_value(sp):
    """The held values a general battle may show (Y2): eight units, or the teacher firing at a held distance or band."""
    if len(sp["mine"]) == AXES["size"]["held"]:
        return "size"
    fd = found(sp)
    k = label(sp)
    if k:
        c = fd["cands"][k - 1]
        if AXES["distance"]["held"][0] <= math.hypot(c["s"], c["f"]) <= AXES["distance"]["held"][1]:
            return "distance"
        if AXES["band"]["held"][0] <= c["band"] - fd["centre"]["band"] <= AXES["band"]["held"][1]:
            return "band"
    return None


def general(seed, split="G"):
    """A stage-0 general battle (module doc): att 0, the teacher on its own ground. A draw whose layout cannot stand
    (KEEP's line or the enemy's rows on blocked ground) or - outside the exam's splits - that shows a held value
    (_held_value) is drawn again in the same stream; the rejects are counted in s1.rejects."""
    from collections import Counter
    from net.stage0 import specs as SP
    from net.stage0 import cfg as scfg
    rnd = random.Random(int(seed))
    sizes = scfg()["general"]["sizes"]
    rej = Counter()
    for _ in range(TRIES):
        lo, hi, _ = rnd.choices(sizes, weights=[s[2] for s in sizes])[0]
        n = rnd.randint(int(lo), int(hi))
        f1, f2, mine, theirs = SP._sides(rnd, n)
        mp = rnd.choices(list(GENERAL["maps"]), weights=list(GENERAL["maps"].values()))[0]
        east = rnd.random() < 0.5
        mz, ez = SP.zone(1 if east else 0), SP.zone(0 if east else 1)
        enemy = rnd.choices(list(GENERAL["enemies"]), weights=list(GENERAL["enemies"].values()))[0]
        dz = rnd.uniform(-60.0, 60.0)
        sp = {"kind": KIND, "seed": int(seed), "split": split, "held": None, "twin": None, "map": mp, "att": 0,
              "fac": {1: f1, 2: f2}, "factions": {1: scfg()["factions"][f1], 2: scfg()["factions"][f2]},
              "mine": mine, "theirs": theirs, "mine_zone": mz, "enemy_zone": ez, "enemy": enemy, "dz": dz,
              "places_theirs": SP.rows(ez, len(theirs), dz), "reserve": [], "limit": LIMIT_S,
              "s1": {"cause": None, "general": True, "ground": "general", "axes": {"n": n, "map": mp}}}
        if not _layout_ok(mp, mz, n, ez, len(theirs), dz):
            rej["layout"] += 1
            continue
        hv = None if split in EXAM_SPLITS else _held_value(sp)
        if hv:
            rej["held_" + hv] += 1
            continue
        sp["s1"]["rejects"] = dict(rej)
        return sp
    raise AssertionError(f"general seed {seed}: no draw in {TRIES}")


def scenes_at(s0, n, split, held=None):
    """n cause scenes from seed s0 + their n twins (twin_kind of each seed)."""
    cause = [scene(s0 + i, split, held) for i in range(n)]
    twins = [scene(s0 + i, split, held, twin=twin_kind(s0 + i)) for i in range(n)]
    return cause + twins


# ------------------------------------------------------------------ arms
ARMS = ("DEFAULT", "TEACH", "DUMB", "PLACE")


def line(spec, arm, k_net=None):
    """The line {x, z, b, ...} the arm stands on (module doc); k_net - the network's choice (0 KEEP, k a candidate)."""
    fd = found(spec)
    if arm == "DEFAULT":
        return fd["centre"]
    if arm == "TEACH":
        k = label(spec)
        return fd["cands"][k - 1] if k else fd["centre"]
    if arm == "DUMB":
        return fd["foot"] or fd["centre"]
    if arm == "PLACE":
        assert spec["s1"].get("twin_made") == "flat" or spec["s1"].get("twin_made") == "none" or \
            spec["twin"] == "flat", "PLACE is played on flat twins only"
        return dict(spec["s1"]["cause_line"])
    if arm == "NET":
        return fd["cands"][k_net - 1] if k_net else fd["centre"]
    raise KeyError(arm)


def places(spec, arm, k_net=None):
    return CR.places(line(spec, arm, k_net), len(spec["mine"]))


def binds(spec):
    """{axis: the held value holds at tick 0} of a held scene (design_opus 5 'Binding')."""
    h = spec.get("held")
    if h is None:
        return {}
    a = spec["s1"]["axes"]
    ok = {"ground": spec["map"] == AXES["ground"]["held"] if spec["twin"] in (None, "role") else True,
          "size": a["n"] == AXES["size"]["held"],
          "band": AXES["band"]["held"][0] <= a["delta_band"] <= AXES["band"]["held"][1],
          "distance": AXES["distance"]["held"][0] <= a["dist"] <= AXES["distance"]["held"][1],
          "enemy": a["enemy"] == AXES["enemy"]["held"],
          "shift": AXES["shift"]["held"][0] <= abs(a["dz"]) <= AXES["shift"]["held"][1],
          "zone": list(a["zone"]) == list(AXES["zone"]["held"][:2])}
    return {h: bool(ok[h])}


def identity(spec):
    return ("s1.scene" if not spec["s1"].get("general") else "s1.general", spec.get("twin"), int(spec["seed"]))


def family(spec):
    return ("s1.scene" if not spec["s1"].get("general") else "s1.general", int(spec["seed"]))
