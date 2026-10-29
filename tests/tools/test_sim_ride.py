"""The cavalry's third step (tools/nn/sim/impact.py, the ride-in; build/diag_cav/mechanic.md M25-M29): the brace is each
man's own - a charger that knocks the first rank down rides in among the men, who get up and walk out of the brace (no
charge defence, no reflection, no bracing of a new landing from the mean getting-up time on); a held unit strikes a
charger that rode in in full; the front rank's first blow lands as the men it knocked down get up, the ranks behind
strike at once. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry, impact, melee  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402

from tests.tools.test_sim_impact import CLANRATS, CSPEAR, KNIGHTS, P, SP, SVHALB, landing, touching  # noqa: E402

UP = (P.battle["knockdown_getting_up_delay_min"] + P.battle["knockdown_getting_up_delay_max"]) / 2     # 1.85 s


class TestRideIn:
    @pytest.mark.parametrize("target, ride", [(CSPEAR, 1.0), (SVHALB, 0.0)])
    def test_the_first_rank_down_is_a_ride_in(self, target, ride):
        # braced clanrat spearmen go down (dv 8.6 > 7), braced stormvermin stay up (6.4: m B 1200 > the horse's 1000)
        _, st, H = landing(KNIGHTS, target)
        assert float(st.u["ride"][0, H]) == ride

    def test_lying_the_brace_holds_up_it_is_gone(self):
        _, st, H = landing(KNIGHTS, CSPEAR)
        zero = torch.zeros_like(st.u["men"])
        assert not bool(impact.ridden(st.u, P)[0, H])
        assert float(impact.bracing(st.u, P, zero, zero)[0, H]) > 6.0          # a reaction freezes the brace (M26)
        st.u["kd_t"][0, H] = UP
        assert bool(impact.ridden(st.u, P)[0, H])
        assert float(impact.bracing(st.u, P, zero, zero)[0, H]) == 1.0          # up and walking: B 1 (M25)

    def test_no_larger_enemy_touching_the_men_re_form(self):
        _, st, H = landing(KNIGHTS, CSPEAR)
        pw = geometry.pairwise(st.u, SP)
        none = torch.zeros_like(pw["enemy"])
        speed = torch.zeros_like(st.u["men"])
        impact.land(st.u, pw, none, torch.zeros_like(st.u["r"]), P, P.dt, speed, speed)
        assert float(st.u["ride"][0, H]) == 0.0

    def test_off_the_switch_no_ride(self):
        p = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], ride_in=0))
        _, st, H = landing(KNIGHTS, CSPEAR, params=p)
        st.u["kd_t"][0, H] = 5.0
        assert float(st.u["ride"][0, H]) == 0.0 and not bool(impact.ridden(st.u, p)[0, H])


def strikes(st, H, charge, ride, kd_t, hold=False):
    st.u["ride"][0, H], st.u["kd_t"][0, H] = ride, kd_t
    if hold:
        st.u["order_kind"][0, H] = O.HOLD
    st.u["vx"].zero_()
    pw = geometry.pairwise(st.u, SP)
    ch = torch.zeros_like(st.u["men"])
    ch[0, 0] = charge
    rate, hit, _, F = melee.strikes(st.u, pw, pw["enemy"], P, ch)
    return float(rate[0, 0, H]), float(hit[0, 0, H]), float(rate[0, H, 0])


class TestChargeDefenceByMan:
    def test_the_walking_men_take_the_charge_bonus(self):
        st, H = touching(KNIGHTS, CSPEAR)
        _, hit_braced, _ = strikes(st, H, 1.0, 0.0, 5.0)
        _, hit_lying, _ = strikes(st, H, 1.0, 1.0, 1.0)
        _, hit_up, _ = strikes(st, H, 1.0, 1.0, UP)
        _, hit_free, _ = strikes(st, H, 0.0, 0.0, 5.0)
        assert hit_braced == pytest.approx(hit_free) and hit_lying == pytest.approx(hit_free)    # defended: no bonus
        assert hit_up > 1.5 * hit_free                                            # up and walking: the bonus 48 lands

    def test_a_held_unit_strikes_a_charger_that_rode_in_in_full(self):
        st, H = touching(KNIGHTS, CLANRATS)
        _, _, held = strikes(st, H, 0.0, 0.0, 5.0, hold=True)
        _, _, rode = strikes(st, H, 0.0, 1.0, 5.0, hold=True)
        assert rode == pytest.approx(held / P.sim["contact"]["hold_rate"], rel=1e-5)


class TestFirstBlowLate:
    def test_the_front_blow_lands_as_the_men_get_up(self):
        _, st, H = landing(KNIGHTS, CSPEAR)
        pw = geometry.pairwise(st.u, SP)
        total, dt = 0.0, P.dt
        for k in range(1, 8):
            st.u["kd_t"][0, H] = k * dt
            pair, share = impact.deferred(st.u, P, pw["enemy"], dt)
            assert bool(pair[0, 0, H]) and not bool(pair[0, H, 0])
            if k * dt <= P.battle["knockdown_getting_up_delay_min"]:
                assert float(share[0, 0, H]) == 0.0
            total += float(share[0, 0, H])
        assert total == pytest.approx(1.0)
