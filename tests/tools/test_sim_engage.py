"""The formed combat distance as the melee's hold (geometry.engage_gap / contact_limit, config/nn/sim.json
contact.engage): two formations either of which fights already stay in contact while their outer men are within the
database's entity_action_attack_formed_combat_distance (+ the larger man's radius) of each other; the start of a contact
and a lone man's rule are unchanged; the companion's mirror (exchange.gaps) agrees. Needs torch."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SWORD, CLANRAT, GENERAL = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1", "wh_main_emp_cha_general_0"
COMBAT = P.battle["entity_action_attack_formed_combat_distance"]


def on(params=P, engage=1):
    p = copy.deepcopy(params)
    p.sim["contact"]["engage"] = engage
    return p


def off(params=P):
    return on(params, 0)


def units(rows, m=(False, False)):
    """Two units: Empire infantry spacing 1.48 x 1.6, clanrats' 1.6 x 1.7, radius 0.65 (or a lone man)."""
    t = lambda k: torch.tensor([[float(r[k]) for r in rows]])     # noqa: E731
    u = {k: t(k) for k in ("x", "z", "b", "men", "width", "sp_h", "sp_v", "radius")}
    u["men0"] = u["men"].clone()
    u["side"] = torch.tensor([[1, 2]])
    u["m"] = torch.tensor([list(m)])
    return u


EMP = dict(x=0, z=0, b=0, men=120, width=30, sp_h=1.48, sp_v=1.6, radius=0.65)


def rat(**kw):
    return dict(dict(x=0, z=20, b=180, men=160, width=30, sp_h=1.6, sp_v=1.7, radius=0.65), **kw)


class TestEngageGap:
    def test_head_on_the_two_half_ranks_off_the_combat_distance(self):
        u = units([EMP, rat()])
        pw = geometry.pairwise(u, 1.5)
        eng = geometry.engage_gap(u, pw, COMBAT)
        assert float(eng[0, 0, 1]) == pytest.approx(COMBAT + 0.65 - 0.8 - 0.85)
        assert float(eng[0, 1, 0]) == pytest.approx(float(eng[0, 0, 1]))

    def test_through_a_flank_the_file_spacing(self):
        # the clanrats 25 m to the swordsmen's right facing them: i is met through its flank (h), j through its front
        u = units([EMP, rat(x=25, z=0, b=270)])
        pw = geometry.pairwise(u, 1.5)
        assert float(geometry.engage_gap(u, pw, COMBAT)[0, 0, 1]) == pytest.approx(COMBAT + 0.65 - 0.74 - 0.85)

    def test_the_larger_radius_counts(self):
        u = units([EMP, rat(radius=0.9)])
        pw = geometry.pairwise(u, 1.5)
        assert float(geometry.engage_gap(u, pw, COMBAT)[0, 0, 1]) == pytest.approx(COMBAT + 0.9 - 0.8 - 0.85)


class TestContactLimit:
    def lim(self, params, rows, m):
        u = units(rows, m)
        return float(geometry.contact_limit(u, geometry.pairwise(u, 1.5), params.sim["contact"], params.battle)[0, 0, 1])

    def test_off_is_reach_plus_hold(self):
        c = P.sim["contact"]
        assert self.lim(off(), [EMP, rat()], (False, False)) == pytest.approx(c["reach_m"])
        assert self.lim(off(), [EMP, rat()], (True, False)) == pytest.approx(c["reach_m"] + c["hold_m"])

    def test_on_holds_to_the_combat_distance_in_melee_only(self):
        c = P.sim["contact"]
        assert self.lim(on(), [EMP, rat()], (False, False)) == pytest.approx(c["reach_m"])     # the start: unchanged
        assert self.lim(on(), [EMP, rat()], (False, True)) == pytest.approx(COMBAT + 0.65 - 0.8 - 0.85)

    def test_a_lone_man_keeps_his_rule(self):
        c = P.sim["contact"]
        lord = dict(x=0, z=0, b=0, men=1, width=1.4, sp_h=1.48, sp_v=1.6, radius=0.7)
        assert self.lim(on(), [lord, rat()], (True, True)) == pytest.approx(c["lord_reach_m"] + c["hold_m"])


def fight(params, gap, order=None):
    """Swordsmen and clanrats both in melee, holding, their boxes `gap` m apart head-on: in contact after a step?
    order: the swordsmen's instead of the hold - "withdraw" (west), "other" (attack a second clanrat unit 5 m beyond
    their right flank: nearer than contact.leave_m, not a leave)."""
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": -50, "z": 0, "b": 90}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRAT, "x": 50, "z": 0, "b": 270},
                                                         {"key": CLANRAT, "x": 50, "z": 80, "b": 270}]}}}], params)
    front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
    H = st.N // 2
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + gap / 2)
    st.u["x"][0, H] = st.u["x"][0, H + 1] = depth[0, H] / 2 + gap / 2
    st.u["z"][0, H + 1] = front[0, 0] / 2 + 5.0 + front[0, H + 1] / 2
    st.u["m"][0, 0] = True
    st.u["m"][0, H] = True
    o = O.hold(st.B, st.N)
    if order == "withdraw":
        o.kind[0, 0], o.x[0, 0], o.z[0, 0] = O.WITHDRAW, -300.0, 0.0
    elif order == "other":
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, H + 1
    battle.step(st, o, params)
    return bool(st.u["m"][0, 0]) and bool(st.u["m"][0, H])


class TestBattle:
    def test_fronts_at_the_games_fighting_distance_stay_in_contact_only_with_the_switch(self):
        # the reform probe: in a long fight the game's fronts stand ~1.3 m (closest men) apart, the boxes ~+0.5..+1.5 m
        assert not fight(off(), 1.0)
        assert fight(on(), 1.0)
        assert not fight(on(), 2.5)            # beyond the combat distance: apart

    def test_a_leaver_or_an_attacker_of_another_unit_is_not_held(self):
        # contact.engage_release: the combat distance holds units that stay fighting; one leaving (a withdraw) or told
        # to attack another unit touches its foe by the old hold only (the fatleave / fresh probes)
        rel = copy.deepcopy(on())
        rel.sim["contact"]["engage_release"] = 1
        held = copy.deepcopy(on())
        held.sim["contact"]["engage_release"] = 0
        for order in ("withdraw", "other"):
            assert not fight(rel, 1.0, order)
            assert fight(held, 1.0, order)
        assert fight(rel, 1.0)                 # a held pair: still in contact
        assert fight(rel, -1.0, "other")       # overlapping boxes: the old hold holds

    def test_overlapping_boxes_touch_either_way(self):
        assert fight(off(), -1.0) and fight(on(), -1.0)


def test_the_companion_mirror_agrees():
    from tools.nn.companion import exchange
    p = on()
    c = p.sim["contact"]
    shape = {"sp_h": np.array([1.48, 1.6]), "sp_v": np.array([1.6, 1.7]), "width": np.array([30.0, 30.0]),
             "men0": np.array([120.0, 160.0]), "radius": np.array([0.65, 0.65]), "spacing": 1.5,
             "reach_m": c["reach_m"], "lord_reach_m": c["lord_reach_m"], "hold_m": c["hold_m"], "engage_m": COMBAT}
    state = {"x": np.array([[0.0, 0.0]]), "z": np.array([[0.0, 20.0]]), "b": np.array([[0.0, 180.0]]),
             "men": np.array([[120.0, 160.0]]), "m": np.array([[True, False]])}
    gap, at = exchange.gaps(state, shape)
    u = units([EMP, rat()], (True, False))
    assert at[0, 1] == pytest.approx(float(geometry.contact_limit(u, geometry.pairwise(u, 1.5), c, p.battle)[0, 0, 1]))
    state["m"] = np.array([[False, False]])
    assert exchange.gaps(state, shape)[1][0, 1] == pytest.approx(c["reach_m"])
    shape["engage_m"] = None
    state["m"] = np.array([[True, False]])
    assert exchange.gaps(state, shape)[1][0, 1] == pytest.approx(c["reach_m"] + c["hold_m"])


def test_the_switch_is_on():
    # on since S26 (config/nn/sim.json contact.engage_why)
    assert P.sim["contact"].get("engage") == 1
