"""Golden runs of the simulator (tools/sim/golden.py): every army's battle as before.

A failure means some module changed a whole battle. If the change is meant,
update the goldens (python -m tools.sim.golden --update) and say why in the commit.
"""
import json

import pytest

from tools.sim import formation as sim
from tools.sim import golden


@pytest.fixture(scope="module")
def planner():
    return sim.Planner()


@pytest.mark.parametrize("army", golden.armies())
def test_battle_as_the_golden(planner, army):
    path = golden.path_of(army)
    assert path.exists(), f"no golden for {army}: python -m tools.sim.golden --update {army}"
    diff = golden.differences(json.loads(path.read_text(encoding="utf-8")), golden.summary(army, planner))
    assert not diff, "\n".join(diff[:30])
