"""Stage 0's battles as plain data (no torch): the roster, armies equal in gold, the general battles and the drills.

A spec is a dict:
    kind        'general' | 'c01' | 't02' | 'o03'
    fac         {1: 'emp' | 'skv', 2: ...}; factions {1: the simulator's faction, 2: ...}
    mine        [unit key] my units (no lord: battle.py adds both lords lord_back_m behind their side)
    theirs      [unit key] the enemy's (no lord)
    map         'flat' | a captured map's name; east - my zone the east one (the game's alliance 1)
    mine_zone, enemy_zone   {x, z, b, front, depth} the game's deployment zones of Kislev (the flat field the same)
    enemy       a script of tools/nn/train/opponents ('ai_game', 'nearest', 'hold_shoot') or 'chain' (our formula
                chain under 'destroy')
    places_mine None - the strategist's formula deploys my side; else [(x, z, b)] per unit of mine
    places_theirs [(x, z, b)] per unit of theirs
    reserve     [index into mine] the units of the reserve (o03; a behaviour may name them in a general battle)
    limit       the battle's length, s
"""
import math
import random

from net.stage0 import cfg

FACS = ("emp", "skv")


def _passports():
    import json
    from net import ROOT
    return json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"]


_COST = {}


def cost(key):
    if not _COST:
        pp = _passports()
        for f in FACS:
            for k in list(cfg()["roster"][f]) + [cfg()["lords"][f]]:
                _COST[k] = float(pp[k]["multiplayer_cost"])
    return _COST[key]


def gold(keys):
    return sum(cost(k) for k in keys)


def army(rnd, fac, n):
    return [rnd.choice(cfg()["roster"][fac]) for _ in range(n)]


def matched(rnd, fac, n, target, tol=None, tries=600):
    """n units of fac whose gold is within tol of target (config general gold_tol); none in tries - the nearest."""
    tol = float(cfg()["general"]["gold_tol"]) if tol is None else tol
    best, bd = None, float("inf")
    for _ in range(tries):
        a = army(rnd, fac, n)
        d = abs(gold(a) - target) / target
        if d <= tol:
            return a
        if d < bd:
            best, bd = a, d
    return best


def zone(alliance):
    """{x, z, b, front, depth} of the game's Kislev deployment zone of the alliance (0 west, 1 east)."""
    from net.matchup import armies as AR
    z = AR.zones()[alliance]
    return {"x": float(z["x"]), "z": float(z["z"]), "b": float(z["bearing_deg"]), "front": float(z["width"]),
            "depth": float(z["depth"])}


def world(zn, s, f):
    """A point of the zone's axes (s right, f forward) in the world (unit/zmap.world)."""
    h = math.radians(zn["b"])
    c, sn = math.cos(h), math.sin(h)
    return zn["x"] + s * c + f * sn, zn["z"] - s * sn + f * c


def local(zn, x, z):
    h = math.radians(zn["b"])
    dx, dz = x - zn["x"], z - zn["z"]
    c, s = math.cos(h), math.sin(h)
    return dx * c - dz * s, dx * s + dz * c


def rows(zn, n, dz=0.0, f0=0.0, row_m=None, per=None, rank_m=None):
    """[(x, z, b)] n units in rows of per across the zone, row_m apart, the rows rank_m apart backwards from f0, the
    whole shifted dz along the front (config general enemy_*)."""
    g = cfg()["general"]
    row_m = float(g["enemy_row_m"]) if row_m is None else row_m
    per = int(g["enemy_rows_of"]) if per is None else per
    rank_m = float(g["enemy_rank_m"]) if rank_m is None else rank_m
    out = []
    for i in range(n):
        r, c = divmod(i, per)
        k = min(per, n - r * per)
        s = (c - (k - 1) / 2) * row_m + dz
        x, z = world(zn, s, f0 - r * rank_m)
        out.append((x, z, zn["b"]))
    return out


def _pick(rnd, weights):
    names, w = zip(*weights.items())
    return rnd.choices(names, weights=w)[0]


def _size(rnd):
    sizes = cfg()["general"]["sizes"]
    lo, hi, _ = rnd.choices(sizes, weights=[s[2] for s in sizes])[0]
    return rnd.randint(int(lo), int(hi))


def _sides(rnd, n_mine, n_theirs=None, fac=None):
    f1 = fac[0] if fac else rnd.choice(FACS)
    f2 = fac[1] if fac else rnd.choice(FACS)
    tol = float(cfg()["general"]["gold_tol"])
    best, bd = None, float("inf")
    for _ in range(30):                         # (no enemy of that gold - e.g. 2 greatswords against slaves: my army anew)
        mine = army(rnd, f1, n_mine)
        theirs = matched(rnd, f2, n_theirs or n_mine, gold(mine), tries=200)
        d = abs(gold(theirs) - gold(mine)) / gold(mine)
        if d <= tol:
            return f1, f2, mine, theirs
        if d < bd:
            best, bd = (mine, theirs), d
    return f1, f2, best[0], best[1]


def _base(rnd, kind, f1, f2, mine, theirs, mp, east, enemy, limit):
    g = cfg()["general"]
    mz, ez = zone(1 if east else 0), zone(0 if east else 1)
    dz = rnd.uniform(-float(g["enemy_dz_m"]), float(g["enemy_dz_m"]))
    return {"kind": kind, "fac": {1: f1, 2: f2}, "factions": {1: cfg()["factions"][f1], 2: cfg()["factions"][f2]},
            "mine": mine, "theirs": theirs, "map": mp, "east": east, "mine_zone": mz, "enemy_zone": ez,
            "enemy": enemy, "dz": dz, "places_mine": None, "places_theirs": rows(ez, len(theirs), dz),
            "reserve": [], "limit": int(limit), "gold": (gold(mine), gold(theirs))}


def general(rnd, enemy=None, mp=None, n=None):
    """One general battle of the stage (config general; plan.md §1)."""
    g = cfg()["general"]
    n = n or _size(rnd)
    f1, f2, mine, theirs = _sides(rnd, n)
    mp = mp or _pick(rnd, g["maps"])
    east = rnd.random() < float(g["zone_east"])
    enemy = enemy or _pick(rnd, g["enemies"])
    return _base(rnd, "general", f1, f2, mine, theirs, mp, east, enemy, g["limit_s"])


def generals(n, rnd, **kw):
    return [general(rnd, **kw) for _ in range(n)]


def c01(rnd, enemy=None):
    """S0-1 'the line on the height' (config drills c01): my 3-6 infantry deployed by the strategist on Kislev, the
    enemy of the same gold coming."""
    d = cfg()["drills"]["c01"]
    n = rnd.randint(*d["units"])
    f1, f2, mine, theirs = _sides(rnd, n)
    east = rnd.random() < float(cfg()["general"]["zone_east"])
    return _base(rnd, "c01", f1, f2, mine, theirs, d["map"], east, enemy or _pick(rnd, d["enemies"]), d["limit_s"])


def t02(rnd, enemy=None):
    """T0-2 'pin and flank' (config drills t02): my 2-3 in a row in the middle of my zone's front, the enemy's 2 of
    the same gold gap_m ahead (shifted along the front), coming."""
    d = cfg()["drills"]["t02"]
    n = rnd.randint(*d["own"])
    f1, f2, mine, theirs = _sides(rnd, n, int(d["enemy"]))
    east = rnd.random() < float(cfg()["general"]["zone_east"])
    sp = _base(rnd, "t02", f1, f2, mine, theirs, _pick(rnd, d["maps"]), east, enemy or _pick(rnd, d["enemies"]),
               d["limit_s"])
    mz = sp["mine_zone"]
    f_me = mz["depth"] / 2 - 30.0
    s_me = rnd.uniform(-80.0, 80.0)
    sp["places_mine"] = [(x, z, b) for x, z, b in (
        (*world(mz, s_me + (i - (n - 1) / 2) * 48.0, f_me), mz["b"]) for i in range(n))]
    gap = rnd.uniform(*d["gap_m"])
    eb = (mz["b"] + 180.0) % 360.0
    sp["places_theirs"] = [(*world(mz, s_me + sp["dz"] * 0.5 + (i - 0.5) * 48.0, f_me + gap), eb)
                           for i in range(len(theirs))]
    return sp


def o03(rnd, enemy=None):
    """O0-3 'the reserve closes a breach' (config drills o03): my 6-8 - all but 2 in a row across my zone's front,
    the 2 of the reserve 60-100 m behind the row - the enemy's 6-8 of the same gold coming."""
    d = cfg()["drills"]["o03"]
    n = rnd.randint(*d["units"])
    f1, f2, mine, theirs = _sides(rnd, n)
    east = rnd.random() < float(cfg()["general"]["zone_east"])
    sp = _base(rnd, "o03", f1, f2, mine, theirs, _pick(rnd, d["maps"]), east, enemy or _pick(rnd, d["enemies"]),
               d["limit_s"])
    mz = sp["mine_zone"]
    k = int(d["reserve"])
    f_line = mz["depth"] / 2 - 30.0
    s0 = rnd.uniform(-40.0, 40.0)
    line = [(*world(mz, s0 + (i - (n - k - 1) / 2) * 48.0, f_line), mz["b"]) for i in range(n - k)]
    back = rnd.uniform(*d["reserve_back_m"])
    res = [(*world(mz, s0 + (i - (k - 1) / 2) * 60.0, f_line - back), mz["b"]) for i in range(k)]
    order = list(range(n))
    rnd.shuffle(order)                        # (which units stand in the reserve: drawn)
    places = [None] * n
    for j, i in enumerate(order):
        places[i] = (line + res)[j]
    sp["places_mine"] = places
    sp["reserve"] = sorted(order[n - k:])
    return sp


DRILLS = {"c01": c01, "t02": t02, "o03": o03}


def drills(kind, n, rnd, **kw):
    return [DRILLS[kind](rnd, **kw) for _ in range(n)]


def seeded(seed):
    return random.Random(seed)
