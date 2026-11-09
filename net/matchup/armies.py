"""The matchups' armies and places (plain python, no torch).

    base(name)        {1: [unit], 2: [unit]} - unit {key, slot, name, x, z, b, general, width, men} in the game's map
                      frame: frozen in net/matchup/<name>.json (config matchups.<name>.frozen; _base's doc says from
                      what) - the arena's places, and where the human's own run had them the places both sides really
                      stood at t = 0 (the person's deployment and the game AI's)
    augment(base, rnd)  a copy: places +- jitter_m and turned +- turn_deg, the whole battle mirrored across the front
                      (z -> -z) with the share mirror, 0-2 units (not the lord) out on each side
    description(units, attacker)   the simulator's army description (sim/scenario.build)
"""
import copy
import json
import math
from functools import lru_cache

from net import ROOT
from net.matchup import cfg

FACTIONS = {}


@lru_cache(maxsize=8)
def _base(name):
    """The matchup's frozen armies and places (config matchups.<name>.frozen: net/matchup/<name>.json). Frozen on 09.11
    from the old arena() (army_from_run of the human's run with its replacements, or armies.generate of a seed, on the
    map by the game's zones) and the places of the human's own run at t = 0 (his deployment and the game AI's) - the
    runs themselves (69 MB) stay in tmp/old-build/human/runs."""
    m = cfg()["matchups"][name]
    d = json.loads((ROOT / m["frozen"]).read_text(encoding="utf-8"))
    for s in (1, 2):
        FACTIONS[(name, s)] = d["factions"][str(s)]
    return {s: [dict(u) for u in d["units"][str(s)]] for s in (1, 2)}


def base(name):
    return copy.deepcopy(_base(name))


def faction(name, side):
    _base(name)
    return FACTIONS[(name, side)]


def augment(units, rnd, aug=None):
    """units {1: [...], 2: [...]} -> a changed copy (module doc); rnd: random.Random."""
    a = aug or cfg()["augment"]
    out = copy.deepcopy(units)
    mirror = rnd.random() < float(a["mirror"])
    j, t = float(a["jitter_m"]), float(a["turn_deg"])
    lo, hi = (int(v) for v in a["drop"])
    for s in (1, 2):
        rows = out[s]
        k = rnd.randint(lo, hi)
        cand = [i for i, u in enumerate(rows) if not u["general"]]
        for i in sorted(rnd.sample(cand, min(k, len(cand))), reverse=True):
            del rows[i]
        for u in rows:
            u["x"] += rnd.uniform(-j, j)
            u["z"] += rnd.uniform(-j, j)
            u["b"] = (u["b"] + rnd.uniform(-t, t)) % 360
            if mirror:
                u["z"] = -u["z"]
                u["b"] = (180.0 - u["b"]) % 360
    return out


def description(name, units, attacker=None):
    """The simulator's army description of one battle."""
    att = int(cfg()["attacker"] if attacker is None else attacker)
    sides = {}
    for s in (1, 2):
        sides[s] = {"faction": (units.get("factions") or {}).get(s) or faction(name, s),
                    "units": [{"key": u["key"], "x": u["x"], "z": u["z"], "b": u["b"], "general": u["general"],
                               "width": u["width"]} for u in units[s]]}
    return {"attacker": att, "sides": sides}


def zones():
    """The game's deployment zones of the map ({alliance: {x, z, width, depth, bearing_deg}})."""
    from sim import maps as arena_map
    zs = arena_map.meta(cfg()["map"])["game_deployment_zones"]
    return {int(z["alliance"]): z for z in zs}
