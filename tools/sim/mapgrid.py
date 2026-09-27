"""A captured map as the simulation's "engine" for the stand mask.

data/maps/<map>/grid-3m.npz: the 3 m grid of a map-capture run (clear =
is_area_clear per cell, ground code, height, inside radar), derived from
research/evidence and kept small in the repository for the simulation and tests.

The 3 m capture has no per-cell reachability; reachable is taken equal to
clear (at the hamlet both checks agreed cell for cell, research/analysis/hamlet).
"""
from pathlib import Path

import numpy as np

from tools import config as project


class MapGrid:
    def __init__(self, name):
        data = np.load(project.ROOT / "data" / "maps" / name / "grid-3m.npz")
        self.clear = data["clear"] == 1
        self.inside = data["inside"] == 1
        self.min_x, self.min_z, self.step = float(data["min_x"]), float(data["min_z"]), float(data["step"])

    def blocked_near(self, placements, margin=60):
        """Centres of blocked 3 m cells around the given placements (for pictures)."""
        xs = [p["x"] for p in placements]
        zs = [p["z"] for p in placements]
        x0, x1, z0, z1 = min(xs) - margin, max(xs) + margin, min(zs) - margin, max(zs) + margin
        i0, i1 = int((z0 - self.min_z) // self.step), int((z1 - self.min_z) // self.step)
        j0, j1 = int((x0 - self.min_x) // self.step), int((x1 - self.min_x) // self.step)
        out = []
        for i in range(max(0, i0), min(self.clear.shape[0], i1 + 1)):
            for j in range(max(0, j0), min(self.clear.shape[1], j1 + 1)):
                if not self.clear[i, j]:
                    out.append((self.min_x + (j + 0.5) * self.step, self.min_z + (i + 0.5) * self.step))
        return out

    def reader(self, x, z):
        """(clear, reachable) at a world point; outside the map: not standable."""
        i, j = int((z - self.min_z) // self.step), int((x - self.min_x) // self.step)
        if not (0 <= i < self.clear.shape[0] and 0 <= j < self.clear.shape[1]) or not self.inside[i, j]:
            return False, False
        c = bool(self.clear[i, j])
        return c, c


def exists(name):
    return (Path(project.ROOT) / "data" / "maps" / name / "grid-3m.npz").exists()
