"""The wrap of a wider line in melee (melee.wrap, tools/nn/sim/melee.py strikes / front_overlap; config/nn/sim.json
melee.wrap_why; build/diag_g49/wrap/mechanic.md): the contact's length = the fronts' overlap + at each end min(the
excess, the narrow unit's depth, max_m), grown over grow_s of the contact; the wrap's blows on the flank; the narrow unit
answers with its flank men; equal fronts unchanged; off - the overlap only. Needs torch (the container)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry, melee, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SWORD, CLANRATS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1"
FF = P.sim["melee"]["fighting_files"]
WR = P.sim["melee"]["wrap"]


def off_params(**battle):
    p = copy.deepcopy(P)
    p.sim["melee"]["wrap"] = dict(WR, on=0)
    p.battle.update(battle)
    return p


def pair(w_a, w_t, t_s=1e4, dz=0.0):
    """Swordsmen w_a m wide facing +x against clanrats w_t m wide facing -x, front to front (dz: the clanrats moved
    along the line); both long in melee (t_s)."""
    desc = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": -20.0, "z": 0.0, "b": 90.0, "width": w_a}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRATS, "x": 20.0, "z": dz, "b": 270.0,
                                                          "width": w_t}]}}}
    st = scenario.build([desc], P)
    front, depth = geometry.dims(st.u, SP)
    H = st.N // 2
    st.u["x"][0, 0], st.u["x"][0, H] = -float(depth[0, 0]) / 2, float(depth[0, H]) / 2
    st.u["contact_s"] = torch.full_like(st.u["men"], float(t_s))
    st.u["melee_s"] = torch.full_like(st.u["men"], float(t_s))
    return st, H, front, depth


def strike(st, params):
    pw = geometry.pairwise(st.u, SP)
    contact = pw["enemy"] & (pw["gap"] <= 1.0)
    return melee.strikes(st.u, pw, contact, params, torch.zeros_like(st.u["men"]))


def test_the_overlap_and_the_excess_at_each_end():
    st, H, front, _ = pair(70, 20)
    pw = geometry.pairwise(st.u, SP)
    tf = torch.ones_like(pw["dist"], dtype=torch.bool)
    ov, lo, hi = melee.front_overlap(st.u, pw, tf, pw["front"][:, :, None].expand_as(pw["dist"]))
    fa, ft = float(front[0, 0]), float(front[0, H])
    assert float(ov[0, 0, H]) == pytest.approx(ft, abs=1e-3)
    assert float(lo[0, 0, H]) == pytest.approx((fa - ft) / 2, abs=1e-3) == pytest.approx(float(hi[0, 0, H]), abs=1e-3)
    assert float(lo[0, H, 0]) == 0 and float(hi[0, H, 0]) == 0              # the narrow one runs past nothing
    st, H, front, _ = pair(70, 20, dz=10.0)                                  # the clanrats 10 m along the line
    pw = geometry.pairwise(st.u, SP)
    ov, lo, hi = melee.front_overlap(st.u, pw, tf, pw["front"][:, :, None].expand_as(pw["dist"]))
    assert float(ov[0, 0, H]) == pytest.approx(ft, abs=1e-3)
    assert sorted([float(lo[0, 0, H]), float(hi[0, 0, H])]) == pytest.approx(
        [(fa - ft) / 2 - 10, (fa - ft) / 2 + 10], abs=1e-3)


def test_a_wide_line_wraps_a_narrow_one_and_the_narrow_one_answers_with_its_flanks():
    st, H, front, depth = pair(70, 20)
    fa, ft, dt = float(front[0, 0]), float(front[0, H]), float(depth[0, H])
    excess = (fa - ft) / 2
    W = 2 * min(excess, dt, WR["max_m"])
    assert W == pytest.approx(2 * WR["max_m"])                               # 25 m past each end, 24 m deep: 9 m
    h_a, v_t, h_t = P.spacing_of(SWORD)[0], P.spacing_of(CLANRATS)[1], P.spacing_of(CLANRATS)[0]
    _, _, sector, F = strike(st, P)
    assert float(F[0, 0, H]) == pytest.approx(FF * (ft + W) / h_a, rel=1e-3)
    assert float(F[0, H, 0]) == pytest.approx(FF * (ft / h_t + W / v_t), rel=1e-3)
    _, _, _, F0 = strike(st, off_params())
    assert float(F0[0, 0, H]) == pytest.approx(FF * ft / h_a, rel=1e-3)      # off: the overlap only
    assert float(F0[0, H, 0]) == pytest.approx(FF * ft / h_t, rel=1e-3)
    assert int(sector[0, 0, H]) == 0                                          # front to front


def test_the_wraps_blows_land_on_the_flank():
    st, H, _, _ = pair(70, 20)
    rate = strike(st, P)[0]
    # the same with the flank as strong as the front: the wide side's blows weaker, the narrow side's the same
    even = copy.deepcopy(P)
    even.battle["melee_defence_direction_penalty_coefficient_flank"] = 1.0
    rate1 = strike(st, even)[0]
    assert float(rate[0, 0, H]) > float(rate1[0, 0, H]) * 1.001
    assert float(rate[0, H, 0]) == pytest.approx(float(rate1[0, H, 0]), rel=1e-6)


def test_the_wrap_grows_over_its_time_and_equal_fronts_are_unchanged():
    st, H, front, _ = pair(70, 20, t_s=WR["grow_s"] / 2)
    fa, ft = float(front[0, 0]), float(front[0, H])
    F = strike(st, P)[3]
    F0 = strike(st, off_params())[3]
    assert float(F[0, 0, H]) / float(F0[0, 0, H]) == pytest.approx((ft + WR["max_m"]) / ft, rel=1e-3)
    st, H, _, _ = pair(70, 20, t_s=0.0)
    assert torch.allclose(strike(st, P)[3], strike(st, off_params())[3])     # the contact's start: no wrap yet
    st, H, _, _ = pair(40, 40)
    # equal fronts (39.96 / 40.0 m: whole files) - as before within the 2 cm of excess
    assert torch.allclose(strike(st, P)[0], strike(st, off_params())[0], rtol=2e-3)


def test_the_wrap_is_no_deeper_than_the_narrow_unit():
    st, H, front, depth = pair(80, 70)
    fa, ft, dt = float(front[0, 0]), float(front[0, H]), float(depth[0, H])
    excess = (fa - ft) / 2
    assert excess < WR["max_m"] and dt < WR["max_m"]
    h_a = P.spacing_of(SWORD)[0]
    F = strike(st, P)[3]
    assert float(F[0, 0, H]) == pytest.approx(FF * (ft + 2 * min(excess, dt)) / h_a, rel=1e-3)
