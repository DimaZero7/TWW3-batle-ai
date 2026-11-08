"""The matchups' armies and places (plain python, no torch).

    arena(name)       the arena of the matchup as tools.build makes it: army_from_run (a run's armies, with the
                      replacements) or armies.generate (a seed), put on the map by arena_map.on_map (the game's zones)
    base(name)        {1: [unit], 2: [unit]} - unit {key, slot, x, z, b, general, width}: the arena's places moved by its
                      origin (the game's map frame); the human's own run (config matchups.<name>.run) gives the places
                      both sides really stood at t = 0 (the person's deployment and the game AI's) where it has them
    augment(base, rnd)  a copy: places +- jitter_m and turned +- turn_deg, the whole battle mirrored across the front
                      (z -> -z) with the share mirror, 0-2 units (not the lord) out on each side
    description(units, attacker)   the simulator's army description (sim/scenario.build)
"""
import copy
import json
import math
from functools import lru_cache
from pathlib import Path

from tools import config as project
from tools.nn.matchup import cfg

FACTIONS = {}


@lru_cache(maxsize=8)
def arena(name):
    from tools.nn import arena_map
    m = cfg()["matchups"][name]
    if "run" in m:
        from tools.nn import army_from_run
        a, _ = army_from_run.load(project.ROOT / m["run"], swap=False, replace=dict(m.get("replace") or {}))
    else:
        from tools.nn.armies import generate
        a = generate.battle(int(m["seed"]), armies=m.get("armies", "v3"))
        if m.get("swap"):
            a = dict(a, sides={"own": a["sides"]["enemy"], "enemy": a["sides"]["own"]})
    return arena_map.on_map(a, cfg()["map"])


def _run_places(run):
    """{script name: (x, z, b)} of the run's first nn_sample (t = 0: after the deployment)."""
    f = project.ROOT / run / "events.jsonl"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if '"nn_sample"' not in line:
                continue
            d = json.loads(line)
            if d.get("event") == "nn_sample":
                return {u["n"]: (float(u["x"]), float(u["z"]), float(u["b"])) for u in d.get("units") or []}
    return {}


@lru_cache(maxsize=8)
def _base(name):
    from tools.nn import scenario as nn_scenario
    a = arena(name)
    pl = nn_scenario.placements(a)
    o = a.get("origin") or {"x": 0.0, "z": 0.0}
    m = cfg()["matchups"][name]
    real = _run_places(m["places"]) if "places" in m else {}
    out = {}
    for s, side in ((1, "own"), (2, "enemy")):
        FACTIONS[(name, s)] = a["sides"][side]["faction"]
        rows = []
        # the run's places by script name; a unit renamed there (a replaced key: ironsides -> long_rifles) takes the
        # run's unmatched unit of its side in order
        names = [u["script_name"] for u in pl[side]]
        free = [n for n in real if n.startswith(side + "_") and n not in names]
        for u in pl[side]:
            x, z, b = u["x"] + o["x"], u["z"] + o["z"], float(u["bearing_deg"])
            if u["script_name"] in real:
                x, z, b = real[u["script_name"]]
            elif free:
                x, z, b = real[free.pop(0)]
            rows.append({"key": u["key"], "slot": u["slot"], "name": u["script_name"], "x": x, "z": z, "b": b,
                         "general": bool(u["general"]), "width": float(u["width"]), "men": int(u["men"])})
        out[s] = rows
    return out


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
        sides[s] = {"faction": faction(name, s),
                    "units": [{"key": u["key"], "x": u["x"], "z": u["z"], "b": u["b"], "general": u["general"],
                               "width": u["width"]} for u in units[s]]}
    return {"attacker": att, "sides": sides}


def zones():
    """The game's deployment zones of the map ({alliance: {x, z, width, depth, bearing_deg}})."""
    from tools.nn import arena_map
    zs = arena_map.meta(cfg()["map"])["game_deployment_zones"]
    return {int(z["alliance"]): z for z in zs}
