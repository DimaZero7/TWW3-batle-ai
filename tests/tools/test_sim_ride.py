"""The cavalry's third step (tools/nn/sim/impact.py, the ride-in; build/diag_cav/mechanic.md M25-M29): the brace is each
man's own - a charger that knocks the first rank down rides in among the men, who get up and walk out of the brace (no
charge defence, no reflection, no bracing of a new landing from the mean getting-up time on); a held unit strikes a
charger that rode in in full; the front rank's first blow lands as the men it knocked down get up, the ranks behind
strike at once. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, impact, melee  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402

from tests.tools.test_sim_impact import CLANRATS, CSPEAR, KNIGHTS, P, SP, SVHALB, keep, landing, touching  # noqa: E402

UP = ((P.battle["knockdown_getting_up_delay_min"] + P.battle["knockdown_getting_up_delay_max"]) / 2
      + impact.rise(P))     # 1.85 s lying + 1.1 s getting up (contact.impact wave)


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
        for k in range(1, 10):
            st.u["kd_t"][0, H] = k * dt
            pair, share = impact.deferred(st.u, P, pw["enemy"], dt)
            assert bool(pair[0, 0, H]) and not bool(pair[0, H, 0])
            if k * dt <= P.battle["knockdown_getting_up_delay_min"] + impact.rise(P):
                assert float(share[0, 0, H]) == 0.0                     # lying or getting up: no target (M31)
            total += float(share[0, 0, H])
        assert total == pytest.approx(1.0)

    @pytest.mark.parametrize("wave_on", [1, 0])
    def test_the_ranks_behind_wait_too(self, wave_on):
        # (contact.impact wave; build/diag_cav/mechanic.md M31-M34) 1.5 s after riding in among knocked clanrats no knight
        # strikes them: the men in reach lie, the men behind are beyond them; without the wave the ranks behind strike
        p = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], wave={"on": wave_on, "rise_s": 1.1}))
        _, st, H = landing(KNIGHTS, CLANRATS, params=p)
        assert float(st.u["ride"][0, H]) == 1.0
        rates = []
        for t in (1.5, 4.0 + 5.1):
            st.u["ran_in"][0, 0] = True
            st.u["contact_s"][0, 0] = st.u["melee_s"][0, 0] = t + p.dt
            st.u["kd_t"][0, H] = t
            pw = geometry.pairwise(st.u, SP)
            rate, _, _, _ = melee.strikes(st.u, pw, pw["enemy"], p, torch.zeros_like(st.u["men"]))
            rates.append(float(rate[0, 0, H]))
        assert rates[1] > 0                                    # long after: the steady rate
        assert (rates[0] == 0.0) == bool(wave_on)


# ---- the front rank's lance and the timer (contact.impact collide_down / strike_first; build/diag_cav/mechanic.md M35-M39)
NEW = P
FIRST_OFF = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], strike_first=0))


class TestCollisionOnTheKnockedOnly:
    def test_a_man_lying_takes_no_collision_damage(self):
        # (M35) a second landing 0.5 s after the first: the first rank lies (reaction 0xf) - no reaction, no damage
        hp1, st, H = landing(KNIGHTS, CLANRATS)
        assert hp1 > 0 and float(impact.lying(st.u, P)[0, H]) == 1.0
        pw = geometry.pairwise(st.u, SP)
        fire = torch.zeros_like(st.u["r"])
        fire[0, 0] = True
        speed = torch.sqrt(st.u["vx"] ** 2 + st.u["vz"] ** 2)
        hp2 = impact.land(st.u, pw, pw["enemy"], fire, P, P.dt, speed, torch.zeros_like(st.u["men"]))
        assert float(hp2[0, 0, H]) == 0.0

    def test_the_lances_kill_before_the_bodies_touch(self):
        # (M36) the men the front rank's lances kill are no bodies to meet: hp less min(kills, men met) x HP a man met
        st, H = touching(KNIGHTS, CLANRATS)
        st.u["vx"][0, 0] = 10.8
        pw = geometry.pairwise(st.u, SP)
        fire = torch.zeros_like(st.u["r"])
        fire[0, 0] = True
        speed = torch.sqrt(st.u["vx"] ** 2 + st.u["vz"] ** 2)
        hp, men0, per0 = impact.land(st.u, pw, pw["enemy"], fire, P, P.dt, speed, torch.zeros_like(st.u["men"]), parts=True)
        n0, d0 = float(men0[0, 0, H]), float(per0[0, 0, H])
        assert n0 > 5 and d0 > 0
        hp_man = float(st.u["hp_man"][0, H])
        for kills in (3.0, 1e3):
            swing = torch.zeros_like(hp)
            swing[0, 0, H] = kills * hp_man
            left = impact.lance_first(st.u, hp, men0, per0, swing)
            assert float(left[0, 0, H]) == pytest.approx(float(hp[0, 0, H]) - min(kills, n0) * d0, rel=1e-5, abs=1e-3)


class TestFrontRankStrikesFirst:
    def test_the_switch_needs_the_wave_and_the_ride_in(self):
        assert impact.strike_first(P) and not impact.strike_first(FIRST_OFF)
        no_wave = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], wave={"on": 0, "rise_s": 1.1}))
        assert not impact.strike_first(no_wave)

    def deferred_swing(self, params):
        """The knights' swing on clanrats they rode in among, as the men get up (late), and F, ff, the front rank."""
        _, st, H = landing(KNIGHTS, CLANRATS, params=params)
        t = 3.0
        st.u["ran_in"][0, 0] = True
        st.u["contact_s"][0, 0] = st.u["melee_s"][0, 0] = t + params.dt
        st.u["kd_t"][0, H] = t
        st.u["vx"].zero_()
        pw = geometry.pairwise(st.u, SP)
        late = torch.zeros_like(st.u["r"])
        late[0, 0] = True
        ran = torch.zeros_like(st.u["r"])
        rate, _, _, F, swing = melee.strikes(st.u, pw, pw["enemy"], params, torch.zeros_like(st.u["men"]), first=True,
                                             ran=ran, late=late)
        h, _ = melee.spacing_of(st.u, SP)
        length = min(float(pw["face_i"][0, 0, H]), float(pw["face_j"][0, 0, H]))
        return float(swing[0, 0, H]), float(F[0, 0, H]), length / float(h[0, 0])

    def test_the_ranks_behind_swing_as_the_men_get_up(self):
        # (M37, M39 4) the front rank swung at the landing; as the men get up every knight within reach (F / fighting_files,
        # the first strike's count) but the front rank (the template's files on the contact) swings
        on, F, front = self.deferred_swing(NEW)
        off, F_off, _ = self.deferred_swing(FIRST_OFF)
        assert F == pytest.approx(F_off)
        ff = melee.files_share(P.sim["melee"])
        assert on == pytest.approx(off * max(F / ff - front, 0.0) / F, rel=1e-4)

    @pytest.mark.parametrize("t, front_back", [(1.5, False), (4.0, False), (5.6, True)])
    def test_the_front_rank_strikes_again_an_interval_after_the_landing(self, t, front_back):
        # (M37) one timer a man from his lance blow: the front rank is back 5.1 s after the landing (not after the men got
        # up); before the men get up no knight has a target
        def rate(params):
            _, st, H = landing(KNIGHTS, CLANRATS, params=params)
            st.u["ran_in"][0, 0] = True
            st.u["contact_s"][0, 0] = st.u["melee_s"][0, 0] = t + params.dt
            st.u["kd_t"][0, H] = t
            st.u["vx"].zero_()
            pw = geometry.pairwise(st.u, SP)
            assert float(st.u["interval"][0, 0]) == pytest.approx(5.1, abs=0.05)      # the lance's interval
            r, _, _, _ = melee.strikes(st.u, pw, pw["enemy"], params, torch.zeros_like(st.u["men"]))
            return float(r[0, 0, H])
        on, off = rate(NEW), rate(FIRST_OFF)
        if t < UP:
            assert on == 0.0 and off == 0.0
        elif front_back:
            assert on > off > 0
        else:
            assert on == pytest.approx(off, rel=1e-5)        # both: the misses back since the men got up

    def test_the_lances_land_at_the_landing(self):
        # (M36, M39 1) a charge of knights into clanrats: the step it lands the clanrats lose the front rank's lance blow
        # on top of the collision damage; without strike_first the blow waits for the men to get up
        def first_loss(params):
            st, H = touching(KNIGHTS, CLANRATS, params=params, gap=3.0)
            st.u["vx"][0, 0] = float(st.u["run"][0, 0])
            st.u["runup"][0, 0] = 100.0
            st.u["leadership"] = st.u["leadership"] + 1e4
            hp0 = float(st.u["hp_abs"][0, H])
            for _ in range(6):
                battle.step(st, keep(st), params, params.dt)
                if float(st.u["ride"][0, H]) >= 0.5:
                    return hp0 - float(st.u["hp_abs"][0, H]), st, H
            raise AssertionError("no landing")
        on, st, H = first_loss(NEW)
        off, _, _ = first_loss(FIRST_OFF)
        assert on > off + 150.0                         # the lances of ~9 files at full charge on 60 HP clanrats
