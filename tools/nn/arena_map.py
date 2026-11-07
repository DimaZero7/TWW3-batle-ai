"""A generated / replayed battle put on a captured map (tools.build ... --map <name>; build/maps/<name>/meta.json): the
map's battle file and catchment and its deployment zones - by default the game's own deployment of a land battle on that
map (meta "game_deployment_zones": terrain/deployment/east.xml, see GAME_DIRECTIONS below; written by
`py -3.14 -m tools.nn.arena_map --game-zones <names>`), else the capture battle's ("deployment_zones", kept beside) - their
size, the gap between their fronts, their middle - instead of the flat arena's squares. The network sees the ground (the
companion reads build/maps by the map's path).

The arena's frame (tools/nn/scenario.py frame: the sides along x, facing each other across the map's centre) is kept;
when the zones sit off the map's centre (the capture's on ksl_plains_01: both at z = -190) the battle file, the run
config's places and the zones are moved by the zones' middle afterwards (shift_xml, shift_config) - the arena's own
geometry and the simulator's version do not change. The units keep their places in their side's frame; their lateral
spread is narrowed only when it does not fit the zone's front (the person redeploys anyway).

    arena = on_map(arena, "ksl_plains_01")        # map, catchment, deploy_zone, gap_m, origin, the laterals if needed
    shift_xml(path, arena["origin"])              # the written battle file
    shift_config(run_config, arena["origin"])     # units' x / z, deploy_zones
Plain python (the game's packs: py -3.14 for --game-zones only).
"""
import json
import math
import re
from pathlib import Path

from tools import config as project

MAPS = project.ROOT / "build" / "maps"
EDGE_M = 15.0          # a unit's centre this far inside the zone's side at least
CATCHMENT = "catchment_04"   # the capture battle's (scenarios/map_capture.xml)


def meta(name, maps=MAPS):
    return json.loads((Path(maps) / name / "meta.json").read_text(encoding="utf-8"))


# The game's own deployment areas of a land battle on a map without its own (the macro maps: ksl_plains_01,
# ksl_river_01 - no deployment in their folder): terrain/deployment/{north,south,east,west}.xml in terrain.pack, the
# files Warhammer3.exe names (its strings 0x1439dd450-0x1439dd4b0), in the map's 0..1024 frame (the world's -512..512):
# east - alliance 0 x 240..416, alliance 1 x 608..784, both z 304..720 (corners rounded): 416 m across, 176 m deep, the
# fronts 192 m apart, the centres 368 m apart, symmetric on the map's centre (build/diag_chain/deploy_zones/extract.py).
GAME_DIRECTIONS = ("east", "west", "north", "south")


def game_zones(xml_text):
    """[{alliance, x, z, width, depth, bearing_deg, corners}] of a terrain/deployment/<direction>.xml (world metres:
    the file's 0..1024 less 512; width across, depth along the line between the zones; bearing - towards the other)."""
    out = []
    for aid, body in re.findall(r"<ALLIANCE id='(\d+)'>(.*?)</ALLIANCE>", xml_text, re.S):
        pts = [(float(x) - 512.0, float(y) - 512.0) for x, y in re.findall(r"x='(-?[\d.]+)' y='(-?[\d.]+)'", body)]
        xs, zs = [p[0] for p in pts], [p[1] for p in pts]
        out.append({"alliance": int(aid), "x": (min(xs) + max(xs)) / 2, "z": (min(zs) + max(zs)) / 2,
                    "ex": max(xs) - min(xs), "ez": max(zs) - min(zs), "corners": [[round(a, 2), round(b, 2)] for a, b in pts]})
    a, b = out
    along_x = abs(a["x"] - b["x"]) > abs(a["z"] - b["z"])
    for z, o in ((a, b), (b, a)):
        z["width"], z["depth"] = (z["ez"], z["ex"]) if along_x else (z["ex"], z["ez"])
        z["bearing_deg"] = round(math.degrees(math.atan2(o["x"] - z["x"], o["z"] - z["z"])) % 360.0, 2)
        del z["ex"], z["ez"]
    return out


def write_game_zones(names, maps=MAPS, direction="east"):
    """Adds "game_deployment_zones" (the game's own, game_zones of terrain/deployment/<direction>.xml; direction east: the
    zones along x, as the arena's frame) to each map's meta.json next to its capture's "deployment_zones" (kept).
    py -3.14 (the game's packs are zstd): python -m tools.nn.arena_map --game-zones ksl_plains_01 ksl_river_01"""
    from tools.nn import gamedb
    path = r"terrain\deployment" + "\\" + f"{direction}.xml"
    data = gamedb.read_entries(Path(project.load()["game_dir"]) / "data" / "terrain.pack", {path})[path]
    zones = game_zones(data.decode("utf-8"))
    for name in names:
        f = Path(maps) / name / "meta.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        m["game_deployment_zones"] = zones
        m["game_deployment_source"] = (f"terrain.pack {path}: the game's deployment of a land battle on a map without "
                                       f"its own (tools/nn/arena_map.py)")
        f.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return zones


def on_map(arena, name, maps=MAPS, zones="game"):
    """zones: 'game' - the game's own deployment of the map (meta game_deployment_zones; the default), 'capture' - the
    capture battle's (meta deployment_zones)."""
    m = meta(name, maps)
    zs = (m.get("game_deployment_zones") if zones == "game" else None) or m.get("deployment_zones") or []
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


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--game-zones"]:
        print(json.dumps(write_game_zones(sys.argv[2:]), indent=1))
