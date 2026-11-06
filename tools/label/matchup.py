"""The matchup (no torch): what a battle starts from, one json so a matchup can be labelled again and again.

    {"name": "...", "map": "flat" | a build/maps folder, "limit_s": 600,
     "command": {"verb": "hold", "pace": "any"},      the order from above (the operative's input; until a strategist)
     "enemy": "attack" | "attack_run" | "hold",        the enemy's script (config enemy_modes)
     "sides": {"1": {"faction": ..., "units": [{"key", "x", "z", "b"}]},   my side (the person's): always 1
               "2": {...}}}

check(m)      the list of errors (empty: correct): the factions and units of the pool, 1-20 units a side, at most one
              general a side, the places inside the side's deployment strip, the command's words, the enemy's mode
army(m)       the simulator's army description (sim/scenario.build): the pool's widths, the general flag
save(m), load(name), names()   build/label/matchups/<name>.json
"""
import json
import re

from tools import config as project
from tools.label import cfg
from tools.label import pool as PL

VERBS = ("destroy", "hold", "tie", "avoid")          # tactic/decide.VERBS (no torch here)
PACES = ("any", "slow", "fast")
MAX_PER_SIDE = 20                                    # sim/scenario.MAX_PER_SIDE


def folder():
    return project.ROOT / cfg()["out"] / "matchups"


def safe(name):
    s = re.sub(r"[^\w\-]+", "_", str(name or "").strip(), flags=re.UNICODE).strip("_")
    return s[:60] or "matchup"


def _side(m, s):
    sides = m.get("sides") or {}
    return sides.get(str(s)) or sides.get(s) or {}


def inside(r, x, z):
    return r["min_x"] - 1e-6 <= x <= r["max_x"] + 1e-6 and r["min_z"] - 1e-6 <= z <= r["max_z"] + 1e-6


def check(m):
    errs = []
    try:
        fl = PL.field(m.get("map", "flat"))
    except ValueError as e:
        return [str(e)]
    cmd = m.get("command") or {}
    if cmd.get("verb", "hold") not in VERBS:
        errs.append(f"command verb {cmd.get('verb')!r} not one of {VERBS}")
    if cmd.get("pace", "any") not in PACES:
        errs.append(f"command pace {cmd.get('pace')!r} not one of {PACES}")
    if m.get("enemy", "attack") not in cfg()["enemy_modes"]:
        errs.append(f"enemy {m.get('enemy')!r} not one of {cfg()['enemy_modes']}")
    lim = m.get("limit_s", cfg()["limit_s"])
    if not isinstance(lim, (int, float)) or not 30 <= lim <= 3600:
        errs.append(f"limit_s {lim!r}: 30-3600 s")
    for s, strip in ((1, "own"), (2, "enemy")):
        sd = _side(m, s)
        fac = sd.get("faction")
        units = sd.get("units") or []
        if fac not in PL.factions():
            errs.append(f"side {s}: faction {fac!r} not in the pool")
            continue
        if not 1 <= len(units) <= MAX_PER_SIDE:
            errs.append(f"side {s}: {len(units)} units (1-{MAX_PER_SIDE})")
        gens = 0
        for i, u in enumerate(units):
            e = PL.unit(fac, u.get("key"))
            if e is None:
                errs.append(f"side {s} unit {i}: {u.get('key')!r} not in the pool of {fac}")
                continue
            gens += bool(e["general"])
            try:
                x, z = float(u["x"]), float(u["z"])
                float(u.get("b", 0.0))
            except (KeyError, TypeError, ValueError):
                errs.append(f"side {s} unit {i}: no place (x, z, b)")
                continue
            if not inside(fl[strip], x, z):
                errs.append(f"side {s} unit {i} ({u['key']}): ({x:.0f}, {z:.0f}) outside the deployment strip")
        if gens > 1:
            errs.append(f"side {s}: {gens} generals (at most 1)")
    return errs


def army(m):
    """The army description of sim/scenario.build (side 1 the person's; the attacker: the side whose script attacks)."""
    out = {"attacker": 1 if m.get("enemy", "attack") == "hold" else 2, "sides": {}}
    for s in (1, 2):
        sd = _side(m, s)
        fac = sd["faction"]
        units = []
        for u in sd["units"]:
            e = PL.unit(fac, u["key"])
            d = {"key": u["key"], "x": float(u["x"]), "z": float(u["z"]), "b": float(u.get("b", 0.0)) % 360.0,
                 "general": bool(e and e["general"])}
            if u.get("width"):
                d["width"] = float(u["width"])           # (the strategist's placing sets each unit's width)
            elif e and e.get("width"):
                d["width"] = float(e["width"])
            units.append(d)
        out["sides"][s] = {"faction": fac, "units": units}
    return out


def order(m):
    """The slots' order of the battle: side 1 slots 0.., side 2 slots H.. (scenario.build puts a side's general first).
    Returns ([(side, index in the matchup's list)] per slot, H)."""
    H = max(len(_side(m, s)["units"]) for s in (1, 2))
    slots = [None] * (2 * H)
    for s in (1, 2):
        units = _side(m, s)["units"]
        fac = _side(m, s)["faction"]
        idx = sorted(range(len(units)), key=lambda i: not bool((PL.unit(fac, units[i]["key"]) or {}).get("general")))
        for k, i in enumerate(idx):
            slots[(s - 1) * H + k] = (s, i)
    return slots, H


def save(m):
    errs = check(m)
    if errs:
        raise ValueError("; ".join(errs))
    name = safe(m.get("name"))
    m = dict(m, name=name)
    folder().mkdir(parents=True, exist_ok=True)
    (folder() / f"{name}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return name


def load(name):
    return json.loads((folder() / f"{safe(name)}.json").read_text(encoding="utf-8"))


def names():
    f = folder()
    return sorted(p.stem for p in f.glob("*.json")) if f.exists() else []
