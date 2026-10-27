"""Drill U1 'hold the line' (tools/nn/unit/drill_hold.py): the roster, the area on the unit's grid, the enemy's start,
the end rule. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]


def test_the_roster_is_melee_infantry_of_both_factions_only():
    R = DH.roster()
    pp = DH._passports()
    assert len(R["emp"]) == 6 and len(R["skv"]) == 10
    for ks in R.values():
        for k in ks:
            assert pp[k]["caste"] == "melee_infantry" and not pp[k].get("missile") and "_mon_" not in k
    cells, m, key = DH.cells_needed()
    assert cells == 6 and 40 < m <= 48 and "stormvermin" in key


def test_the_area_holds_the_unit_on_its_grid_and_the_enemy_starts_far_enough():
    sp = DH.specs(64, torch.Generator().manual_seed(3))
    st, own, area = DH.build(sp, P, "cpu")
    a0 = {k: v[:, 0] for k, v in area.items()}
    H = st.N // 2
    for b, s in enumerate(sp):
        assert DH.min_cells() <= s["w"] <= DH.MAX_CELLS and DH.min_cells() <= s["h"] <= DH.MAX_CELLS
        assert float(area["front"][b, 0]) == s["w"] * DH.CELL
        # the unit's centre inside, on a cells' corner of its own grid, far enough from the edges to turn where it can
        d = float(Dc.signed_dist(st.u["x"][b, 0], st.u["z"][b, 0], {k: v[b] for k, v in a0.items()}))
        r = math.hypot(s["front"], s["depth"]) / 2
        assert d >= min(r, min(s["w"], s["h"]) * DH.CELL / 2) - 1e-3
        gap = math.hypot(float(st.u["x"][b, H] - st.u["x"][b, 0]), float(st.u["z"][b, H] - st.u["z"][b, 0]))
        assert DH.GAP_M[0] - 1e-3 <= gap <= DH.GAP_M[1] + 1e-3
        assert abs(float(area["b"][b, 0]) - float(st.u["b"][b, 0])) < 1e-4


def test_the_exam_is_fixed_and_balanced():
    a, b = DH.exam_specs(), DH.exam_specs()
    assert a == b and len(a) == 6 * 10 * 2 * 3
    assert {s["start"] for s in a} == set(DH.STARTS)


def test_the_end_our_rout_ends_the_enemys_rout_goes_on_while_it_is_in_the_zone():
    sp = DH.specs(3, torch.Generator().manual_seed(5))
    st, own, area = DH.build(sp, P, "cpu")
    H = st.N // 2
    end = DH.Ender(st.B, "cpu")
    st.u["r"][0, 0] = True                                   # battle 0: we rout
    st.u["r"][1, H] = True                                   # battle 1: the enemy routs, still near
    st.u["r"][2, H] = True                                   # battle 2: the enemy routs, far away
    st.u["x"][2, H], st.u["z"][2, H] = 900.0, 900.0
    battle.step(st, O.hold(st.B, st.N), P)
    st.u["r"][0, 0] = True
    st.u["r"][1, H] = True
    st.u["r"][2, H] = True
    end.after_step(st, SP)
    assert st.done.tolist() == [True, False, True]
    assert end.why.tolist() == [1, 0, 2] and st.winner.tolist()[0] == 2 and st.winner.tolist()[2] == 1
