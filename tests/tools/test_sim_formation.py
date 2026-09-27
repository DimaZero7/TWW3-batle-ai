"""tools.sim.formation: the first army example plans with the real roster."""
from tools.sim import formation


def test_first_attack_plans_both_sides():
    army, _ = formation.load_army("first_attack")
    sides, units = formation.simulate(army)
    assert sides["own"]["status"] == "ok" and not sides["own"]["overlaps"]
    assert sides["own"]["roles"] == {"wall": 1, "arc": 4, "lord": 1}
    assert len(units) == 6 and sides["enemy"]["roles"] == {"wall": 2, "arc": 2, "lord": 1}
