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


# How the game's AI in defence turns to our army (4 battles window_game, 28.09.2026,
# docs/ru/architecture/tasks/align-chase.md): right after deployment it may stand
# crooked (facing 238 deg with us at 278 deg); within the first minute it turns its
# line to face our army, whatever the distance (at the approach its front line is
# square to its facing, 0-4 deg, and faces us within 4 deg). Later, while our centres are
# 180-270 m apart, it turns its whole line about its centre towards us in bursts of
# 10-36 s, 6-15 deg each (about 0.5-1 deg/s), whenever it faces off us; its centre
# moves 0.1-3.4 m per burst. Without our moves it did not turn at all in 250 s.
NATIVE_REACTION = {"turn_dps": 0.7, "react_m": 270.0, "min_off_deg": 3.0}


def _wrap(deg):
    return ((deg + 540.0) % 360.0) - 180.0


def block_centre(p):
    b = math.radians(p["bearing"])
    return p["x"] - math.sin(b) * p["depth_m"] / 2, p["z"] - math.cos(b) * p["depth_m"] / 2


def army_centre(placements):
    """Mean block centre without the lord (the main body)."""
    body = [p for p in placements if p.get("role") != "lord"] or placements
    pts = [block_centre(p) for p in body]
    return sum(x for x, _ in pts) / len(pts), sum(z for _, z in pts) / len(pts)


class NativeReaction:
    """The game's AI in defence turning to face our army, in time (the simulation only)."""

    def __init__(self, placements, facing, model=None):
        self.m = dict(NATIVE_REACTION, **(model or {}))
        self.placements = [dict(p) for p in placements]
        self.facing = float(facing) % 360
        self.settled = False     # the first turn after deployment is in place, at any distance
        self.t = 0.0
        self.timeline = [{"t_s": 0.0, "placements": [dict(p) for p in self.placements]}]

    def advance(self, seconds, own_placements):
        """`seconds` pass with our army standing in own_placements; returns True when it turned."""
        cx, cz = army_centre(self.placements)
        ox, oz = army_centre(own_placements)
        to_us = math.degrees(math.atan2(ox - cx, oz - cz)) % 360
        off = _wrap(to_us - self.facing)
        dist = math.hypot(ox - cx, oz - cz)
        turned = False
        if abs(off) > self.m["min_off_deg"] and (not self.settled or dist <= self.m["react_m"]):
            lim = self.m["turn_dps"] * seconds
            turn = max(-lim, min(lim, off))
            self.facing = (self.facing + turn) % 360
            t = math.radians(turn)
            for p in self.placements:
                # The whole line swings about its centre (the game: its front line stays square
                # to its facing, 0-4 deg, also after the first turn; 2 battles, 28.09.2026).
                bx, bz = block_centre(p)
                dx, dz = bx - cx, bz - cz
                bx, bz = cx + dx * math.cos(t) + dz * math.sin(t), cz - dx * math.sin(t) + dz * math.cos(t)
                # Each block turns about its own centre (placements keep the front-rank centre).
                p["bearing"] = engine_facing(self.facing)
                b = math.radians(p["bearing"])
                p["x"], p["z"] = bx + math.sin(b) * p["depth_m"] / 2, bz + math.cos(b) * p["depth_m"] / 2
            turned = True
            if not self.settled and abs(off - turn) <= self.m["min_off_deg"]:
                self.settled = True
        elif not self.settled:
            self.settled = True
        self.t += seconds
        if turned:
            self.timeline.append({"t_s": round(self.t, 2), "placements": [dict(p) for p in self.placements]})
        return turned
