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

    def reader(self, x, z):
        """(clear, reachable) at a world point; outside the map: not standable."""
        i, j = int((z - self.min_z) // self.step), int((x - self.min_x) // self.step)
        if not (0 <= i < self.clear.shape[0] and 0 <= j < self.clear.shape[1]) or not self.inside[i, j]:
            return False, False
        c = bool(self.clear[i, j])
        return c, c


def exists(name):
    return (Path(project.ROOT) / "data" / "maps" / name / "grid-3m.npz").exists()
