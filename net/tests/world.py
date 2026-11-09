"""Small battles for the tests (the old test_unit_view_net_torch's helpers): one battle from rows, the units' cards."""
import numpy as np
import torch

from net.cards import passport
from sim import scenario
from sim.params import load

P = load()


def battle(own, enemies):
    """own / enemies: [(key, x, z, b)]; the first own unit is the viewer."""
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b} for k, x, z, b in rows]
    return scenario.build([{"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(own)},
                                                     2: {"faction": "wh2_main_skv_skaven", "units": units(enemies)}}}], P)


def card(st):
    return torch.from_numpy(np.stack([passport.table(row) for row in st.keys]))
