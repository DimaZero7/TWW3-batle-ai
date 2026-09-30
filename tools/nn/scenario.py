"""The arena's battle file for recording training data (config/nn/arena.json -> scenarios/nn_arena.xml).

Side 1 (alliance 0, ours) stands west of the map's centre facing east, side 2
(alliance 1, the game's AI) east facing west; their fronts (forward 0) are gap_m
apart. Which side defends is set by the alliance that wins on timeout (the game's
AI then defends; game, 28.09.2026).

The default arena (config/nn/arena.json) puts the same army on both sides. A named
arena from config/nn/arenas.json takes the map from it and gives each side its own
army: "sides": {"own": {"faction", "units"}, "enemy": {...}} (measurements, 30.09.2026).

    python -m tools.nn.scenario            # write scenarios/nn_arena.xml
    python -m tools.nn.scenario pair_spear_v_slave
"""
import json
import math
from xml.sax.saxutils import escape

from tools import config as project

ARENA = project.CONFIG_DIR / "nn" / "arena.json"
ARENAS = project.CONFIG_DIR / "nn" / "arenas.json"
SCENARIO = project.SCENARIOS / "nn_arena.xml"
SIDES = ("own", "enemy")
DEFAULT = "arena"
# What a named arena may not set: the map and its zones stay those of config/nn/arena.json.
BASE_ONLY = ("map", "catchment", "environment", "deployment_m")


def load_arena(name=None, path=None, arenas_path=None):
    """The arena by name: 'arena' (default) is config/nn/arena.json, any other a named arena
    from config/nn/arenas.json over it (its own gap, defend radius and armies)."""
    base = json.loads((path or ARENA).read_text(encoding="utf-8"))
    name = name or DEFAULT
    if name == DEFAULT:
        return dict(base, name=DEFAULT)
    named = json.loads((arenas_path or ARENAS).read_text(encoding="utf-8"))["arenas"]
    if name not in named:
        raise KeyError(f"no arena {name!r} in {(arenas_path or ARENAS).name}: {', '.join(sorted(named))}")
    own = named[name]
    assert not set(own) & set(BASE_ONLY), f"{name}: the map is set in arena.json"
    assert "sides" in own, f"{name}: a named arena sets its sides"
    merged = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    merged.update(own, name=name)
    return merged


def armies(arena):
    """{side: {"faction", "units"}}: the arena's own sides, or the same army on both."""
    if "sides" in arena:
        out = {side: arena["sides"][side] for side in SIDES}
    else:
        out = {side: {"faction": arena["faction"], "units": arena["units"]} for side in SIDES}
    for side, army in out.items():
        slots = [u["slot"] for u in army["units"]]
        assert army["units"] and len(slots) == len(set(slots)), f"{side}: slots must be unique and present"
        assert sum(bool(u.get("general")) for u in army["units"]) <= 1, f"{side}: one general at most"
    return out


def frame(side, gap_m):
    """(anchor x, anchor z, forward unit vector, right unit vector, orientation radians) of a side."""
    if side == "own":
        return -gap_m / 2, 0.0, (1.0, 0.0), (0.0, -1.0), math.pi / 2
    return gap_m / 2, 0.0, (-1.0, 0.0), (0.0, 1.0), 3 * math.pi / 2


def placements(arena):
    """{side: [{script_name, slot, key, men, general, x, z, bearing_deg, width}]}."""
    out = {}
    sides = armies(arena)
    for side in SIDES:
        ax, az, fwd, right, orient = frame(side, arena["gap_m"])
        rows = []
        for u in sides[side]["units"]:
            x = ax + fwd[0] * u["forward"] + right[0] * u["lateral"]
            z = az + fwd[1] * u["forward"] + right[1] * u["lateral"]
            rows.append({"script_name": f"{side}_{u['slot']}", "slot": u["slot"], "key": u["key"], "men": u["men"],
                         "general": bool(u.get("general")), "x": round(x, 1), "z": round(z, 1),
                         "bearing_deg": round(math.degrees(orient)) % 360, "width": u["width"]})
        out[side] = rows
    return out


UNIT = """      <unit num_soldiers="{men}" script_name="{script_name}">
        <unit_type type="{key}"/>
        <position x="{x}" y="{z}"/><orientation radians="{orient:.4f}"/><width metres="{width}"/>
        <unit_experience level="0"/>
{general}      </unit>
"""

ALLIANCE = """  <alliance id="{index}">
    <army>
      <faction>{faction}</faction>
      <deployment_area>
        <centre x="{cx}" y="{cz}"/><width metres="{size}"/><height metres="{size}"/><orientation radians="{orient:.4f}"/>
      </deployment_area>
      <camera_start_position x="{cam_x}" y="700" z="-420"/>
      <camera_target_position x="0" y="577" z="0"/>
{units}    </army>
    <victory_condition><kill_or_rout_enemy_ignore_reinforcements/></victory_condition>
    <rout_position x="{rout_x}" y="0"/>
  </alliance>
"""

TAIL = """  <battle_description>
    <time_of_day>day</time_of_day>
    <season>Spring</season>
    <precipitation_type>dry</precipitation_type>
    <battle_script prepare_for_fade_in="false">script/battle/tww3_bai/scenario.lua</battle_script>
    <type>classic</type>
    <timeout_winning_alliance_index>{defender}</timeout_winning_alliance_index>
    <duration>{duration}</duration>
    <los_visibility_enabled>true</los_visibility_enabled>
  </battle_description>
  <weather>
    <prevailing_wind x="1" y="0"/>
    <environment_key>{environment}</environment_key>
  </weather>
  <sea_surface_name>wind_level_0</sea_surface_name>
  <battle_map_definition>
    <name>{map}</name>
    <catchment_area>{catchment}</catchment_area>
  </battle_map_definition>
</battle>
"""


def scenario_xml(arena, defender, duration_s=3600):
    """defender: 'own' or 'enemy' — the side that wins on timeout, so the game sees it defending."""
    assert defender in SIDES
    places = placements(arena)
    sides = armies(arena)
    parts = ['<?xml version="1.0" encoding="utf-8"?>\n',
             "<!-- Generated by tools/nn/scenario.py from config/nn/arena.json. Edit that file, not this one. -->\n",
             "<battle>\n"]
    for index, side in enumerate(SIDES):
        ax, az, fwd, _, orient = frame(side, arena["gap_m"])
        back = arena["deployment_m"] / 2 - 40  # the zone reaches 40 m in front of the spearmen
        units = "".join(UNIT.format(men=u["men"], script_name=u["script_name"], key=escape(u["key"]), x=u["x"], z=u["z"],
                                    orient=orient, width=u["width"],
                                    general=(f"        <general><name>{side}</name><star_rating level=\"1\"/></general>\n"
                                             if u["general"] else ""))
                        for u in places[side])
        parts.append(ALLIANCE.format(index=index, faction=escape(sides[side]["faction"]), size=arena["deployment_m"],
                                     cx=round(ax - fwd[0] * back, 1), cz=round(az - fwd[1] * back, 1), orient=orient,
                                     cam_x=round(ax * 1.2), units=units, rout_x=round(-fwd[0] * 900)))
    parts.append(TAIL.format(defender=SIDES.index(defender), duration=duration_s, environment=arena["environment"],
                             map=arena["map"], catchment=arena["catchment"]))
    return "".join(parts)


def run_config(arena):
    """The entry's view of the arena: slots and script names of both sides."""
    places = placements(arena)
    return {"arena": arena.get("name", DEFAULT), "gap_m": arena["gap_m"], "defend_radius_m": arena["defend_radius_m"],
            "factions": {side: army["faction"] for side, army in armies(arena).items()},
            "units": {side: [{"script_name": u["script_name"], "slot": u["slot"], "key": u["key"],
                              "x": u["x"], "z": u["z"], "bearing_deg": u["bearing_deg"], "width": u["width"]}
                             for u in places[side]] for side in SIDES}}


def write_scenario(defender, arena=None):
    arena = arena or load_arena()
    SCENARIO.write_text(scenario_xml(arena, defender), encoding="utf-8")
    return arena


if __name__ == "__main__":
    import sys
    write_scenario("enemy", load_arena(sys.argv[1] if len(sys.argv) > 1 else None))
    print(SCENARIO)
