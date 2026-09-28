"""The enemy as the game's AI stands in defence, for the simulation only.

Measured in the player's test battle (config/armies/test_halves.json: the
game's AI defends the east half of the map, 400 x 260 m; four runs
20260928-113436, -114048, -114818, -115218; lord, 8 spearmen, 6 archers):

* every block about 41 m wide (ordered width 41.2-41.3 m), 45.4 m apart
  centre to centre in a line (about 4 m between blocks);
* the first line: 6 spearmen; the second line: the other 2 spearmen and the
  6 archers; the second line's middles 28-30 m behind the first line's
  (their archers' middle about 32 m behind their front rank);
* the lord on a flank, about 25 m behind the first line;
* facing the middle of one of the engine's 64 sectors like every unit
  (272.8 deg, docs/ru/game/units/commands.md): parallel to our line.

With other deployment areas the game's AI stands otherwise (enemy-layout,
27.09.2026: 13 units in 12-19 m wide deep blocks, archers 51-56 m behind the
front), so this is the model of the test battle, not of every battle.
"""
import math

NATIVE_DEFENDER = {"width_m": 40, "step_m": 45.4, "front_line_max": 6, "line_step_m": 28.5, "lord_back_m": 25}


def _shape(unit, width):
    shapes = unit["shapes"]
    return min(shapes, key=lambda s: abs(s["ordered_m"] - width))


def engine_facing(bearing):
    """The facing the engine gives a unit: the middle of its 64-sector (apps.orders.facing)."""
    sector = 360 / 64
    return ((math.floor((bearing % 360) / sector) + 0.5) * sector) % 360


def native_defender(anchor, bearing, units, model=None):
    """Placements like apps.formation's (front-rank centre, bearing, front, depth, role, along, back)
    for units = roster units (tools.sim.formation.roster_units)."""
    m = dict(NATIVE_DEFENDER, **(model or {}))
    bearing = engine_facing(bearing)
    b = math.radians(bearing)
    fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
    lords = [u for u in units if u["commanding"] and u["men"] == 1]
    shooters = [u for u in units if u not in lords and (u.get("range_m") or 0) > 0]
    infantry = [u for u in units if u not in lords and u not in shooters]
    lines = [infantry[:m["front_line_max"]], infantry[m["front_line_max"]:] + shooters]
    placements = []

    def add(u, role, along, middle_back, width):
        s = _shape(u, width) if width else {"ordered_m": None, "front_m": 1.0, "depth_m": 1.0}
        back = middle_back - s["depth_m"] / 2          # front rank of the block
        placements.append({"id": u["id"], "role": role, "row": 0, "bearing": bearing, "width": s["ordered_m"],
                           "front_m": s["front_m"], "depth_m": s["depth_m"], "along_m": along, "back_m": back,
                           "x": anchor[0] + rx * along - fx * back, "z": anchor[1] + rz * along - fz * back})
    # The first line's middles on the anchor's line; the front rank half a depth ahead.
    widest = 0
    for n, line in enumerate(lines):
        for i, u in enumerate(line):
            along = (i - (len(line) - 1) / 2) * m["step_m"]
            widest = max(widest, abs(along))
            add(u, "arc" if u in shooters else "wall", along, n * m["line_step_m"], m["width_m"])
    for i, u in enumerate(lords):
        add(u, "lord", -(widest + m["step_m"] / 2 + i * 5), m["lord_back_m"], None)
    return placements
