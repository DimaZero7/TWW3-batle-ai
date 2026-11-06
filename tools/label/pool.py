"""The units the matchup may take and the maps (no torch).

    factions()      {faction: {"units": [{key, slot, width, general, category, men, cost, range}], "lord": key}} - the
                    pool of config/nn/pools.json (the lord, units, units_v2, units_v3: every unit the simulator and
                    the unit network know), the passport's numbers from config/nn/units.json
    maps()          [{name, frame {min_x, max_x, min_z, max_z}, preview}] - "flat" (the simulator's flat field) and each
                    captured map folder build/maps/<name>/ (meta.json frame; the picture: mapimg.py)
    field(name)     {min_x, max_x, min_z, max_z} the field drawn and the deployment strips {own, enemy} of a map
"""
import json
from functools import lru_cache

from tools import config as project
from tools.label import cfg

POOLS = project.ROOT / "config" / "nn" / "pools.json"
UNITS = project.ROOT / "config" / "nn" / "units.json"
MAPS = project.ROOT / "build" / "maps"


@lru_cache(maxsize=1)
def passports():
    return json.loads(UNITS.read_text(encoding="utf-8"))["units"]


def _entry(u, general=False):
    p = passports().get(u["key"], {})
    miss = p.get("missile") or {}
    return {"key": u["key"], "slot": u.get("slot", "lord" if general else ""), "width": u.get("width"),
            "general": general, "category": p.get("category", "?"), "men": p.get("men"),
            "cost": p.get("multiplayer_cost"), "range": miss.get("range_m") if isinstance(miss, dict) else None}


@lru_cache(maxsize=1)
def factions():
    d = json.loads(POOLS.read_text(encoding="utf-8"))["factions"]
    out = {}
    for name, f in d.items():
        units = [_entry(f["lord"], True)]
        seen = {f["lord"]["key"]}
        for grp in ("units", "units_v2", "units_v3"):
            for u in f.get(grp, []):
                if u["key"] not in seen:
                    seen.add(u["key"])
                    units.append(_entry(u))
        out[name] = {"units": units, "lord": f["lord"]["key"]}
    return out


def unit(faction, key):
    """The pool's entry of a unit of a faction (None: not in the pool)."""
    return next((u for u in factions().get(faction, {}).get("units", []) if u["key"] == key), None)


def maps():
    half = float(cfg()["deploy"]["flat"]["half_m"])
    out = [{"name": "flat", "frame": {"min_x": -half, "max_x": half, "min_z": -half, "max_z": half}, "preview": None}]
    if MAPS.exists():
        for d in sorted(MAPS.iterdir()):
            meta = d / "meta.json"
            if not meta.exists() or not (d / "heights.npz").exists():
                continue
            m = json.loads(meta.read_text(encoding="utf-8"))
            fr = m.get("frame") or {}
            out.append({"name": d.name, "frame": {k: float(fr[k]) for k in ("min_x", "max_x", "min_z", "max_z")},
                        "preview": f"/mapimg/{d.name}.png"})
    return out


def field(name):
    """{frame, own, enemy}: the drawn field and the deployment strips (rectangles min_x, max_x, min_z, max_z)."""
    m = next((m for m in maps() if m["name"] == name), None)
    if m is None:
        raise ValueError(f"no map {name!r} (flat or a folder of build/maps with heights.npz and meta.json)")
    fr = m["frame"]
    dep = cfg()["deploy"]["flat" if name == "flat" else "map"]
    cx = (fr["min_x"] + fr["max_x"]) / 2
    cz = (fr["min_z"] + fr["max_z"]) / 2
    w = float(dep["width_m"]) / 2

    def strip(z):
        return {"min_x": cx - w, "max_x": cx + w, "min_z": cz + float(z[0]), "max_z": cz + float(z[1])}
    return {"frame": fr, "own": strip(dep["own"]), "enemy": strip(dep["enemy"]), "preview": m["preview"]}
