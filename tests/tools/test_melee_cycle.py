"""melee.cycle (tools/nn/sim/melee.py cycle_rate, files_share; config/nn/sim.json melee.cycle_why): the blow cycle by the
game's program behind its switch. Needs torch: skipped in the project's .venv, run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry, melee, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"
CYC = {"on": 1, "reach_files": 1.1, "tick_s": 0.4, "retarget_s": 0.4, "walk_m": 0.2, "walk_mps": 1.5}


def t(x):
    return torch.tensor([[[float(x)]]])


def rate(p=0.4, T=4.0, hit=20.0, hp=60.0, single=False, recv=0.0, **cyc):
    c = dict(CYC, **cyc)
    return float(melee.cycle_rate(t(p), t(T), t(hit), t(hp), torch.tensor([[[single]]]), t(recv), c))


class TestCycleRate:
    def test_a_swing_every_interval_whatever_the_outcome(self):
        # no hits taken, no time to a new target: p / interval exactly (a miss costs nothing)
        assert rate(p=0.4, T=4.0, recv=0.0, retarget_s=0.0, walk_m=0.0) == pytest.approx(0.1, rel=1e-6)
        assert rate(p=0.1, T=4.0, recv=0.0, retarget_s=0.0, walk_m=0.0) == pytest.approx(0.025, rel=1e-6)

    def test_hits_taken_break_swings_before_their_blow(self):
        lam, w = 0.5, 0.4
        got = rate(recv=lam, tick_s=w, retarget_s=0.0, walk_m=0.0)
        assert got == pytest.approx(0.1 * math.exp(-lam * w), rel=1e-6)

    def test_a_kill_costs_the_search_for_a_new_target(self):
        # busy x (1 + a busy) = 1, a = p / T x unbroken x kills a hit x search
        search = 0.4 + 0.2 / 1.5
        a = 0.1 * (20.0 / 60.0) * search
        busy = 2 / (1 + math.sqrt(1 + 4 * a))
        assert busy * (1 + a * busy) == pytest.approx(1.0, rel=1e-9)
        assert rate(recv=0.0) == pytest.approx(0.1 * busy, rel=1e-6)
        assert rate(recv=0.0, hit=60.0) < rate(recv=0.0, hit=20.0)          # a kill every hit: more searching

    def test_a_lone_target_is_one_pool_no_search(self):
        assert rate(recv=0.0, single=True) == pytest.approx(0.1, rel=1e-6)


def face_off(cyc):
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SPEAR, "x": 0, "z": 0, "b": 0, "general": False}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": SLAVE, "x": 0, "z": 30, "b": 180, "general": False}]}}}],
        P)
    H = st.N // 2
    _, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    st.u["z"][0, H] = depth[0, 0] / 2 + depth[0, H] / 2 + min(0.0, P.sim["contact"]["reach_m"]) / 2
    st.u["contact_s"][:] = 300.0                                 # past the opening wave
    pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
    contact = pw["enemy"] & (pw["gap"] <= 1.0)
    params = P.with_cal("melee", cycle=dict(P.sim["melee"].get("cycle") or {}, **cyc))
    return st, pw, contact, params, H


class TestStrikes:
    def test_off_keeps_the_fitted_rule(self):
        st, pw, contact, params, H = face_off({"on": 0})
        assert melee.files_share(params.sim["melee"]) == params.sim["melee"]["fighting_files"]
        z = torch.zeros_like(st.u["men"])
        rate, hit, _, F = melee.strikes(st.u, pw, contact, params, z)
        base = melee.strikes(st.u, pw, contact, P.with_cal("melee", cycle={"on": 0}), z)[0]
        assert torch.allclose(rate, base)

    def test_on_strikes_with_the_men_within_reach_at_the_interval_from_the_swing(self):
        st, pw, contact, params, H = face_off(dict(CYC, retarget_s=0.0, walk_m=0.0, tick_s=0.0))
        z = torch.zeros_like(st.u["men"])
        _, _, _, F_off = melee.strikes(st.u, pw, contact, P.with_cal("melee", cycle={"on": 0}), z)
        rate, hit, _, F = melee.strikes(st.u, pw, contact, params, z)
        ff = P.sim["melee"]["fighting_files"]
        # the men striking scale from fighting_files to reach_files of the files in contact
        assert float(F[0, 0, H]) == pytest.approx(float(F_off[0, 0, H]) * 1.1 / ff, rel=1e-4)
        # with nothing taken off: F x p / interval x the hit
        p = (float(rate[0, 0, H]) / float(F[0, 0, H]) / float(hit[0, 0, H])) * float(st.u["interval"][0, 0])
        assert 0.05 < p < 0.95
        rate2, hit2, _, F2 = melee.strikes(st.u, pw, contact, params.with_cal(
            "melee", cycle=dict(params.sim["melee"]["cycle"], tick_s=0.4)), z)
        assert float(rate2[0, 0, H]) < float(rate[0, 0, H])        # hits taken break some swings
