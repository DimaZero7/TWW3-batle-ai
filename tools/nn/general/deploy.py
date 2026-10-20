"""The units' layout inside their groups' places ("Раскладка отрядов", an algorithm, no weights) and the deployment it
gives ("Расстановка": a point and a bearing for every unit of the side before the battle).

Each group stands on the tile its place points at (tiles.py: a tile of the zone). On a tile its groups stand one
behind the other in the groups' order; a group's units stand in a line by their slot (the design: "по номеру слота"),
each its frontage wide and LATERAL_GAP_M apart (place.py's numbers), the line across the direction from the tile to
the enemy army's centre (the units face it); a line wider than the zone's width goes on in the next row behind,
ROW_GAP_M behind the deepest unit of the row. Every unit's centre is then kept inside the zone (its half frontage
and half depth from the edges): the layout stays in the zone (test_general_deploy.py). Plain numpy: the deployment is
rare (a battle's start), a loop over the side's units.
"""
import json
import math
from dataclasses import dataclass

import numpy as np

from tools import config as project
from tools.nn.armies import place as placement

LATERAL_GAP_M = placement.LATERAL_GAP_M
ROW_GAP_M = placement.ROW_GAP_M
MARGIN_M = placement.MARGIN_M


def depth_of(men, width, sp_h, sp_v):
    """A formation's depth, m: its ranks (men / files, files = frontage / a man's place across) x a rank's depth."""
    files = max(1.0, math.floor(max(width, sp_h) / max(sp_h, 1e-3)))
    return max(1.0, math.ceil(max(men, 1.0) / files)) * max(sp_v, 0.5)


def _sign(side):
    """+1 for side 1 (facing +x), -1 for side 2 (facing -x): numbers or arrays (numpy, torch) alike."""
    return 3 - 2 * side


def to_zone(x, z, side, gap):
    """World (x, z) -> (forward, lateral) in the side's arena frame (tools/nn/scenario.py frame(): side 1 at x = -gap / 2
    facing +x with its right -z, side 2 at +gap / 2 facing -x with its right +z)."""
    k = _sign(side)
    return k * x + gap / 2, -k * z


def from_zone(f, lat, side, gap):
    k = _sign(side)
    return k * (f - gap / 2), -k * lat


@dataclass(frozen=True)
class ZoneRule:
    """The deployment zone of place.py in the arena's frame (simplification 5): forward from front_m - deployment_m to
    front_m (place.ZONE_FRONT_M), lateral within +-deployment_m / 2. Its methods take numbers, numpy arrays or torch
    tensors."""
    gap_m: float = 350.0
    deployment_m: float = 300.0
    front_m: float = placement.ZONE_FRONT_M
    slack_m: float = 2.0          # a unit this far outside still counts as in its zone (rounding of the placement)

    @classmethod
    def of_arena(cls):
        a = json.loads((project.CONFIG_DIR / "nn" / "arena.json").read_text(encoding="utf-8"))
        return cls(gap_m=float(a["gap_m"]), deployment_m=float(a["deployment_m"]))

    @property
    def back_m(self):
        return self.front_m - self.deployment_m

    @property
    def half_m(self):
        return self.deployment_m / 2

    def to_zone(self, x, z, side):
        return to_zone(x, z, side, self.gap_m)

    def from_zone(self, f, lat, side):
        return from_zone(f, lat, side, self.gap_m)

    def inside(self, x, z, side, slack=0.0):
        f, lat = self.to_zone(x, z, side)
        return (f >= self.back_m - slack) & (f <= self.front_m + slack) & (abs(lat) <= self.half_m + slack)

    def tiles(self, cfg, frame, side, cells_in=None):
        """[R, S] the tiles whose centre is in the side's zone (and, with cells_in, inside the map); torch (tiles.py)."""
        from tools.nn.general import tiles
        w = tiles.centres_world(cfg, frame)
        s = side.reshape(-1, 1).expand(-1, w.shape[1])
        ok = self.inside(w[..., 0], w[..., 1], s)
        if cells_in is not None:
            ok = ok & cells_in.any(-1)
        return ok

    def valid(self, u, side):
        """[R] every present unit of the side stands in its zone (else the battle keeps its scripted placement); torch."""
        present = u["side"] > 0
        own = present & (u["side"] == side[:, None])
        s = side.reshape(-1, 1).expand_as(u["x"])
        ins = self.inside(u["x"].float(), u["z"].float(), s, self.slack_m)
        return (ins | ~own).all(-1) & own.any(-1)


def layout(units, groups, side, enemy_c, rule):
    """The side's deployment.

    units: [{"slot", "group" (our place index, -1: not laid out), "width", "depth"}] (the side's units in any order);
    groups: {place: (x, z) world point of its tile}; side 1 or 2; enemy_c (x, z) the enemy army's centre;
    rule: tiles.ZoneRule. -> {slot: (x, z, bearing_deg)} of the units laid out."""
    gap, back, front, half = rule.gap_m, rule.back_m, rule.front_m, rule.half_m
    room = rule.deployment_m - 2 * MARGIN_M
    by_tile = {}
    for g, xz in sorted(groups.items()):
        by_tile.setdefault((round(float(xz[0]), 3), round(float(xz[1]), 3)), []).append(g)
    out = {}
    for (tx, tz), gs in by_tile.items():
        fx, fz = enemy_c[0] - tx, enemy_c[1] - tz
        n = math.hypot(fx, fz)
        fx, fz = (fx / n, fz / n) if n > 1e-6 else ((1.0, 0.0) if side == 1 else (-1.0, 0.0))
        rx, rz = fz, -fx                                            # right of the facing
        bearing = math.degrees(math.atan2(fx, fz)) % 360.0
        behind = 0.0                                                # how far behind the tile's centre the next row is
        for g in gs:
            mine = sorted((u for u in units if u["group"] == g), key=lambda u: u["slot"])
            rows, row, w = [], [], 0.0
            for u in mine:
                need = u["width"] + (LATERAL_GAP_M if row else 0.0)
                if row and w + need > room:
                    rows.append(row)
                    row, w, need = [], 0.0, u["width"]
                row.append(u)
                w += need
            if row:
                rows.append(row)
            for row in rows:
                total = sum(u["width"] for u in row) + LATERAL_GAP_M * (len(row) - 1)
                deep = max(u["depth"] for u in row)
                left = -total / 2
                for u in row:
                    lat = left + u["width"] / 2
                    left += u["width"] + LATERAL_GAP_M
                    back_off = behind + u["depth"] / 2
                    x = tx + rx * lat - fx * back_off
                    z = tz + rz * lat - fz * back_off
                    # kept in the zone: the centre half a frontage / depth inside its edges
                    f, l = to_zone(x, z, side, gap)
                    hw, hd = u["width"] / 2, u["depth"] / 2
                    f = min(max(f, back + hd), front - hd) if front - back > 2 * hd else (front + back) / 2
                    l = min(max(l, -half + hw), half - hw) if 2 * half > 2 * hw else 0.0
                    x, z = from_zone(f, l, side, gap)
                    out[u["slot"]] = (x, z, bearing)
                behind += deep + ROW_GAP_M
    return out


def inside(points, units, side, rule, tol=1e-3):
    """True when every laid-out unit's footprint (centre +- half frontage, half depth, in the zone's axes) is inside the
    zone (the check of test_general_deploy.py)."""
    ok = True
    by = {u["slot"]: u for u in units}
    for slot, (x, z, _) in points.items():
        u = by[slot]
        f, l = to_zone(x, z, side, rule.gap_m)
        ok &= (f - u["depth"] / 2 >= rule.back_m - tol) and (f + u["depth"] / 2 <= rule.front_m + tol)
        ok &= abs(l) + u["width"] / 2 <= rule.half_m + tol
    return bool(ok)


def as_arrays(points, N):
    """{slot: (x, z, b)} -> (x [N], z [N], b [N], laid [N] bool) numpy."""
    x, z, b = np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
    laid = np.zeros(N, bool)
    for s, (px, pz, pb) in points.items():
        x[s], z[s], b[s], laid[s] = px, pz, pb, True
    return x, z, b, laid
