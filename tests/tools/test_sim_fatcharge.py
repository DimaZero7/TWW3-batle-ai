"""Fatigue and bracing by the fatrun and brace probes (docs/en/game/mechanics/fatigue.md, melee.md): a ranked unit's
fatigue points a tick (calibration.rank), the held unit's first strike within the bracing angle
(contact.stand_first_angle), the charge reflection from any side (melee.reflect_any_angle), no 'ready' from a standing
enemy's distance alone (fatigue.calibration.ready_enemy_m 0). Needs torch (the container)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, fatigue, geometry, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tests.tools.test_sim_open_rules import (CLANRATS, P, SLAVE, SPEAR, SWORD, army, face_off,  # noqa: E402
                                             orders)


def cal(**values):
    p = copy.deepcopy(P)
    p.sim["fatigue"]["calibration"].update(values)
    return p


def test_a_ranked_unit_tires_less_and_rests_more():
    # the database's unit_stats_land_experience_bonuses: rank 0 / 5 / 9 add 0 / -2 / -4 points a tick
    u = {"fatigue": torch.tensor([5000.0, 5000.0, 5000.0]), "fat": torch.zeros(3), "rank": torch.tensor([0, 5, 9])}
    run = {k: torch.zeros(3, dtype=torch.bool) for k in ("idle", "walking", "melee", "charging", "shooting")}
    run["running"] = torch.ones(3, dtype=torch.bool)
    fatigue.step(u, run, cal(), 1.0)
    assert u["fatigue"].tolist() == pytest.approx([5040.0, 5020.0, 5000.0])
    rest = {k: torch.zeros(3, dtype=torch.bool) for k in ("walking", "running", "melee", "charging", "shooting")}
    rest["idle"] = torch.ones(3, dtype=torch.bool)
    fatigue.step(u, rest, cal(), 1.0)
    assert u["fatigue"].tolist() == pytest.approx([5040.0 - 180, 5020.0 - 200, 5000.0 - 220])
    # no rank on the unit (the simulator's units today): nothing changes
    v = {"fatigue": torch.tensor([5000.0]), "fat": torch.zeros(1)}
    fatigue.step(v, {k: x[:1] for k, x in run.items()}, cal(), 1.0)
    assert float(v["fatigue"][0]) == pytest.approx(5040.0)


@pytest.mark.parametrize("near_m,ready", [(80, True), (0, False)])
def test_a_standing_enemy_near_makes_ready_only_with_the_distance_rule(near_m, ready, monkeypatch):
    st = face_off(SPEAR, SLAVE, gap=10)
    seen = {}
    monkeypatch.setattr(fatigue, "step", lambda u, activity, params, dt: seen.update(activity))
    battle.step(st, O.hold(st.B, st.N), cal(ready_enemy_m=near_m))
    assert ("enemy_near" in seen and bool(seen["enemy_near"][0, 0])) is ready


def charge_into(b_target, params):
    """Clanrats charge (14 m run-up, from +x) into held swordsmen facing b_target; HP the charger loses in the contact
    step (b 90: facing it; 30: the charger 60 deg off their facing)."""
    st = face_off(SWORD, CLANRATS, gap=14.0, b1=b_target)
    H = st.N // 2
    for _ in range(20):
        hp0 = float(st.u["hp_abs"][0, H])
        m0 = bool(st.u["m"][0, 0])
        battle.step(st, orders(st, **{str(H): (O.ATTACK, 0, True)}), params)
        if bool(st.u["m"][0, 0]) and not m0:
            return hp0 - float(st.u["hp_abs"][0, H])
    return None


def test_the_held_unit_strikes_at_once_within_the_bracing_angle():
    old = P.with_cal("contact", stand_first_angle=None)
    new = P.with_cal("contact", stand_first_angle="bracing")
    front = charge_into(90.0, new)
    at60_new, at60_old = charge_into(30.0, new), charge_into(30.0, old)
    at120 = charge_into(330.0, new)          # the charger 120 deg off the facing: beyond 80
    assert front is not None and at60_new is not None and at60_old is not None and at120 is not None
    assert at60_new > 0.6 * front and at60_new > 1.5 * at60_old
    assert at120 < 0.25 * front


@pytest.mark.parametrize("any_angle", [False, True])
def test_the_reflection_from_behind_only_with_any_angle(any_angle):
    params = P.with_cal("melee", reflect_any_angle=any_angle)

    def rate(charge):
        st = scenario.build([army([(SPEAR, 0, 0, 270)], [(CLANRATS, 30, 0, 270)])], params)   # spearmen face away
        H = st.N // 2
        front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
        st.u["x"][0, H] = (float(depth[0, 0]) + float(depth[0, H])) / 2
        pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
        ch = torch.zeros_like(st.u["men"])
        ch[0, H] = charge
        return float(melee.strikes(st.u, pw, pw["enemy"], params, ch)[1][0, 0, H])
    if any_angle:
        assert rate(1.0) > 1.5 * rate(0.0)
    else:
        assert rate(1.0) == pytest.approx(rate(0.0), rel=1e-5)
