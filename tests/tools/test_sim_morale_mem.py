"""Morale rules from the game's program checked on the game's memory (build/diag_mem/morale; config/nn/sim.json morale:
strong_enemy rule "code", routing_net.whole, fear.reach "centre_box"). Needs torch (the container)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry, morale  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
R = P.morale


def with_morale(params, **kw):
    m = copy.deepcopy(params.sim["morale"])
    m.update(kw)
    return params.with_cal("morale", **m)


def test_round_away_is_the_programs_half_away_from_zero():
    x = torch.tensor([2.5, 7.5, 12.5, -2.5, -3.375, -3.5, -13.5, 0.0, 0.4])
    assert morale.round_away(x).tolist() == [3.0, 8.0, 13.0, -3.0, -3.0, -4.0, -14.0, 0.0, 0.0]


def test_strong_points_whole_rounds_and_holds_the_cap():
    power = torch.tensor([3.9, 4.0, 4.7, 18.0, 31.9, 32.0, 50.0])
    got = morale.strong_points(power, R, whole=True).tolist()
    # -3 + (p - 4) / 28 x -21: 4.7 -> -3.525 -> -4; 18 -> -13.5 -> -14; 31.9 -> -23.925 -> -24; from 32 the database's -24
    assert got == [0.0, -3.0, -4.0, -14.0, -24.0, -24.0, -24.0]
    assert morale.strong_points(power, R).tolist()[3] == pytest.approx(-13.5)


def unit_pair(own, foe, run=(3.8, 3.0)):
    return {"cp_fixed": torch.tensor([[own, foe]]), "cp_missile": torch.zeros(1, 2), "a": torch.zeros(1, 2),
            "ammo0": torch.zeros(1, 2), "hp": torch.ones(1, 2), "run": torch.tensor([list(run)])}


def test_code_rule_weights_by_the_rectangle_gap_and_counts_only_fast_enough_enemies():
    params = with_morale(P, strong_enemy=dict(P.sim["morale"]["strong_enemy"], rule="code"))
    foes = torch.tensor([[[False, True], [True, False]]])
    friends = torch.zeros_like(foes)
    d = torch.zeros(1, 2, 2)
    u = unit_pair(100.0, 400.0)
    for gap, w in ((0.0, 1.52), (20.0, 1.52), (36.0, 1.52), (78.0, 2 - 1.6 * 78 / 120), (120.0, 0.4)):
        g = torch.tensor([[[0.0, gap], [gap, 0.0]]])
        assert float(morale.strong_power(u, d, foes, friends, params, gap=g)[0, 0]) == pytest.approx(4 * w, rel=1e-5)
    far = torch.tensor([[[0.0, 121.0], [121.0, 0.0]]])
    assert float(morale.strong_power(u, d, foes, friends, params, gap=far)[0, 0]) == 0.0
    # an enemy at 0.75 of the unit's run speed or slower is not counted (the program: ratio above 0.75)
    slow = unit_pair(100.0, 400.0, run=(4.0, 3.0))
    g = torch.zeros(1, 2, 2)
    assert float(morale.strong_power(slow, d, foes, friends, params, gap=g)[0, 0]) == 0.0
    # a friend near adds its weighted strength to the unit's own
    u3 = {k: torch.cat([v, v[:, :1]], 1) for k, v in u.items()}
    foes3 = torch.tensor([[[False, True, False], [True, False, True], [False, True, False]]])
    friends3 = torch.tensor([[[False, False, True], [False, False, False], [True, False, False]]])
    g3 = torch.zeros(1, 3, 3)
    p = float(morale.strong_power(u3, torch.zeros(1, 3, 3), foes3, friends3, params, gap=g3)[0, 0])
    assert p == pytest.approx(400 * 1.52 / (100 + 100 * 1.52), rel=1e-5)


def test_centre_box_dist_is_the_distance_to_the_other_rectangle():
    u = {"x": torch.tensor([[0.0, 0.0, 30.0]]), "z": torch.tensor([[0.0, 40.0, 0.0]]), "b": torch.tensor([[0.0, 180.0, 90.0]])}
    front = torch.tensor([[30.0, 30.0, 30.0]])
    depth = torch.tensor([[10.0, 26.0, 10.0]])
    d = geometry.centre_box_dist(u, front, depth)
    assert float(d[0, 0, 1]) == pytest.approx(40 - 13)             # to the front edge of the unit ahead (half depth 13)
    assert float(d[0, 0, 2]) == pytest.approx(30 - 5)              # unit 2 faces +x: its half depth 5 along x
    assert float(d[0, 1, 1]) == 0.0
