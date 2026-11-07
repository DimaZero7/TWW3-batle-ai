"""A generated / replayed battle put on a captured map (tools.build ... --map <name>; build/maps/<name>/meta.json): the
map's battle file and catchment, the deployment zones of the map's capture (their size, the gap between their fronts,
their centres) instead of the flat arena's squares. The network sees the ground (the companion reads build/maps by the
map's path).

The arena's frame (tools/nn/scenario.py frame: the sides along x, facing each other across the map's centre) is kept;
the map's zones sit elsewhere on the map (ksl_plains_01: both at z = -190, centres 240 m apart along x), so the battle
file, the run config's places and the zones are moved by the zones' middle afterwards (shift_xml, shift_config) - the
arena's own geometry and the simulator's version do not change. The units keep their places in their side's frame; their
lateral spread is narrowed to fit the zone's front (the person redeploys anyway).

    arena = on_map(arena, "ksl_plains_01")        # map, catchment, deploy_zone, gap_m, origin, narrowed laterals
    shift_xml(path, arena["origin"])              # the written battle file
    shift_config(run_config, arena["origin"])     # units' x / z, deploy_zones
Plain python.
"""
import json
import re
from pathlib import Path

from tools import config as project

MAPS = project.ROOT / "build" / "maps"
EDGE_M = 15.0          # a unit's centre this far inside the zone's side at least
CATCHMENT = "catchment_04"   # the capture battle's (scenarios/map_capture.xml)


def meta(name, maps=MAPS):
    return json.loads((Path(maps) / name / "meta.json").read_text(encoding="utf-8"))


def on_map(arena, name, maps=MAPS):
    m = meta(name, maps)
    zs = m.get("deployment_zones") or []
    if len(zs) != 2:
        raise ValueError(f"{name}: two deployment zones expected, {len(zs)} in meta.json")
    a, b = zs
    if abs(a["z"] - b["z"]) > 1.0:
        raise ValueError(f"{name}: the zones are not side by side along x (the arena's frame): {a['z']} / {b['z']}")
    if abs(a["width"] - b["width"]) > 1 or abs(a["depth"] - b["depth"]) > 1:
        raise ValueError(f"{name}: zones of different sizes")
    front, depth = float(a["width"]), float(a["depth"])
    gap = abs(a["x"] - b["x"]) - depth
    origin = {"x": (a["x"] + b["x"]) / 2.0, "z": (a["z"] + b["z"]) / 2.0}
    sides = {}
    for side, s in arena["sides"].items():
        units = [dict(u) for u in s["units"]]
        wide = max([abs(float(u.get("lateral") or 0.0)) for u in units] + [1.0])
        k = min(1.0, (front / 2 - EDGE_M) / wide)
        for u in units:
            u["lateral"] = round(float(u.get("lateral") or 0.0) * k, 1)
            u["forward"] = round(max(-(depth - EDGE_M), min(0.0, float(u.get("forward") or 0.0))), 1)
        sides[side] = dict(s, units=units)
    # (the catchment as the capture battle had it: scenarios/map_capture.xml catchment_04; meta.json keeps none)
    return dict(arena, map=m["map"], catchment=m.get("catchment") or CATCHMENT, gap_m=gap,
                deploy_zone={"front": front, "depth": depth, "gap": gap}, origin=origin, map_name=name,
                name=f"{arena.get('name', 'arena')}_{name}", sides=sides)


def _num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") if not float(v).is_integer() else str(int(v))


def shift_xml_text(text, origin):
    """The battle file's places moved by origin: unit positions and the zones' centres (x, y = the map's z), the
    cameras (x, z)."""
    dx, dz = float(origin["x"]), float(origin["z"])

    def two(mo):
        return f'{mo.group(1)} x="{_num(float(mo.group(2)) + dx)}" y="{_num(float(mo.group(3)) + dz)}"'
    text = re.sub(r'(<(?:position|centre)) x="(-?[\d.]+)" y="(-?[\d.]+)"', two, text)

    def cam(mo):
        return (f'{mo.group(1)} x="{_num(float(mo.group(2)) + dx)}" y="{mo.group(3)}" '
                f'z="{_num(float(mo.group(4)) + dz)}"')
    return re.sub(r'(<camera_(?:start|target)_position) x="(-?[\d.]+)" y="(-?[\d.]+)" z="(-?[\d.]+)"', cam, text)


def shift_xml(path, origin):
    p = Path(path)
    p.write_text(shift_xml_text(p.read_text(encoding="utf-8"), origin), encoding="utf-8")


def shift_config(run_config, origin):
    dx, dz = float(origin["x"]), float(origin["z"])
    for side in (run_config.get("units") or {}).values():
        for u in side:
            u["x"], u["z"] = round(float(u["x"]) + dx, 1), round(float(u["z"]) + dz, 1)
    for z in (run_config.get("deploy_zones") or {}).values():
        z["x"], z["z"] = round(float(z["x"]) + dx, 1), round(float(z["z"]) + dz, 1)
    run_config["origin"] = {"x": dx, "z": dz}
    return run_config
