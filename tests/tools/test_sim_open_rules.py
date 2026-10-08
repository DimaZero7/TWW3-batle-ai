"""The open melee and whole-battle rules by the game (build/open_melee/spec.md R1-R4, R6, R7; build/open_battle/spec.md
1a, 2, 4, 5, 6; docs/en/training/simulator.md): the struck side's first strike, braced reflection and a move order in
contact against hold_rate, the chase out of melee and its 24 s window, the run-up towards the enemy, the wounded pool,
shattering by the database's thresholds, the rally blocked by any enemy and its clock, the charge into a free target
only, the flank and rear sectors. Needs torch (the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, melee, morale, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
SWORD, CLANRATS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1"
TOUCH = min(0.0, P.sim["contact"]["reach_m"])


def army(side1, side2, attacker=1):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                            2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def face_off(key1, key2, gap=0.0, b1=90.0):
    st = scenario.build([army([(key1, -50, 0, b1)], [(key2, 50, 0, 270)])], P)
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    gap = gap + TOUCH
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + gap / 2)
    H = st.N // 2
    st.u["x"][0, H] = depth[0, H] / 2 + gap / 2
    return st


def orders(st, **units):
    """units: slot -> (kind, target or (x, z), run)."""
    o = replay.hold(st)
    for s, (kind, arg, run) in units.items():
        s = int(s)
        o.kind[0, s] = kind
        if kind == O.ATTACK:
            o.target[0, s] = arg
        elif arg is not None:
            o.x[0, s], o.z[0, s] = arg
        o.run[0, s] = run
    return o


# --- R1: a standing unit that is reached strikes at once ---

class TestStandFirstStrike:
    def charge_into(self, b_target):
        """Clanrats charge (14 m run-up) into held swordsmen (no charge reflection) facing them (b 90: the charger
        comes from +x) or facing away (270); HP the charger loses in the contact step."""
        st = face_off(SWORD, CLANRATS, gap=14.0, b1=b_target)
        H = st.N // 2
        lost = None
        for _ in range(20):
            hp0 = float(st.u["hp_abs"][0, H])
            m0 = bool(st.u["m"][0, 0])
            battle.step(st, orders(st, **{str(H): (O.ATTACK, 0, True)}), P)
            if bool(st.u["m"][0, 0]) and not m0:
                lost = hp0 - float(st.u["hp_abs"][0, H])
                break
        return lost

    def test_a_held_unit_charged_from_the_front_strikes_at_once_from_behind_not(self):
        front = self.charge_into(90.0)        # facing the charger (it comes from +x)
        back = self.charge_into(270.0)        # facing away
        assert front is not None and back is not None
        assert front > 30.0 and back < 0.25 * front

    def test_the_first_swing_is_not_cut_by_hold_rate(self):
        on = self.charge_into(90.0)
        old = P.with_cal("contact", first_strike_full=0)
        st = face_off(SWORD, CLANRATS, gap=14.0, b1=90.0)
        H = st.N // 2
        cut = None
        for _ in range(20):
            hp0 = float(st.u["hp_abs"][0, H])
            m0 = bool(st.u["m"][0, 0])
            battle.step(st, orders(st, **{str(H): (O.ATTACK, 0, True)}), old)
            if bool(st.u["m"][0, 0]) and not m0:
                cut = hp0 - float(st.u["hp_abs"][0, H])
                break
        assert on > 1.5 * cut


# --- R2, R3: hold_rate ---

class TestHoldRate:
    def rate_of(self, kind_0, params=P, reflect=False, charge=0.0):
        st = face_off(SPEAR, CLANRAT)
        H = st.N // 2
        st.u["order_kind"][0, 0] = kind_0
        st.u["order_kind"][0, H] = O.ATTACK
        st.u["order_target"][0, H] = 0
        if reflect:
            st.u["reflect"][0, 0] = True
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= P.sim["contact"]["reach_m"] + 0.01)
        ch = torch.zeros_like(st.u["men"])
        ch[0, H] = charge
        rate, _, _, _ = melee.strikes(st.u, pw, contact, params, ch)
        return float(rate[0, 0, H])

    def test_a_move_order_in_contact_strikes_like_a_held_unit(self):
        attack, held, move = (self.rate_of(k) for k in (O.ATTACK, O.HOLD, O.MOVE))
        hr = P.sim["contact"]["hold_rate"]
        assert held == pytest.approx(hr * attack, rel=1e-4) and move == pytest.approx(held, rel=1e-4)
        off = P.with_cal("contact", hold_move=0)
        assert self.rate_of(O.MOVE, off) == pytest.approx(attack, rel=1e-4)

    def test_braced_reflection_is_not_cut_by_hold_rate(self):
        thr = P.battle["charge_reflect_min_charge_factor_threshold"]
        full = self.rate_of(O.ATTACK, reflect=True, charge=max(thr, 0.9))
        held = self.rate_of(O.HOLD, reflect=True, charge=max(thr, 0.9))
        assert held == pytest.approx(full, rel=1e-4)
        cut = self.rate_of(O.HOLD, P.with_cal("contact", hold_reflect_full=0), reflect=True, charge=max(thr, 0.9))
        assert cut == pytest.approx(P.sim["contact"]["hold_rate"] * full, rel=1e-4)


class TestMoveThrough:
    """contact.leave_away_only (build/probes7 P3): a move order to a far point THROUGH the enemy it touches keeps the
    unit fighting (CA's planner's far move points); a far point away from it is a leave (no strikes)."""
    def dealt(self, point_x, params=P):
        st = face_off(SPEAR, CLANRAT)          # spearmen west facing east, clanrat spearmen east
        H = st.N // 2
        o = orders(st, **{str(H): (O.ATTACK, 0, False)})
        for _ in range(6):
            battle.step(st, o, params)
        hp = float(st.u["hp_abs"][0, H])
        o = orders(st, **{"0": (O.MOVE, (point_x, 0.0), False), str(H): (O.ATTACK, 0, False)})
        for _ in range(10):
            battle.step(st, o, params)
        return hp - float(st.u["hp_abs"][0, H]), bool(st.u["m"][0, 0])

    def test_a_far_point_through_the_enemy_fights_on_one_away_leaves(self):
        through, m_through = self.dealt(60.0)
        away, _ = self.dealt(-200.0)
        assert through > 0 and m_through and away == pytest.approx(0.0, abs=1e-3)
        off, _ = self.dealt(60.0, P.with_cal("contact", leave_away_only=0))
        assert off == pytest.approx(0.0, abs=1e-3)

    def test_a_far_point_through_the_enemy_strikes_in_full_a_near_one_held(self):
        far, _ = self.dealt(60.0, P.with_cal("contact", move_far_full=1))       # (off in sim.json: one lane)
        held, _ = self.dealt(60.0)
        assert far > held > 0


# --- an attack order on an enemy it does not touch, given in melee, is a leave (contact.attack_leave) ---

class TestAttackLeave:
    """Spearmen fighting clanrat spearmen are told to attack a second clanrat unit 150 m off (the game: such units
    strike 0.01-0.03 men/s for 10 s, moving 0.90-0.98 of the time; build/v2gap/game_far_sh.py)."""
    def dealt(self, target, params=P, key=SPEAR):
        st = scenario.build([army([(key, -50, 0, 90)], [(CLANRAT, 50, 0, 270), (CLANRAT, 0, 150, 270)])], P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["x"][0, 0] = -(depth[0, 0] / 2 + TOUCH / 2)
        st.u["x"][0, H] = depth[0, H] / 2 + TOUCH / 2
        o = orders(st, **{"0": (O.ATTACK, H, False), str(H): (O.ATTACK, 0, False)})
        for _ in range(6):
            battle.step(st, o, params)
        assert bool(st.u["m"][0, 0])
        hp = float(st.u["hp_abs"][0, H])
        o = orders(st, **{"0": (O.ATTACK, H if target == "near" else H + 1, False), str(H): (O.HOLD, None, False)})
        for _ in range(int(1.5 / params.dt)):                       # within contact.pin_melee_s: still in contact
            battle.step(st, o, params)
        return hp - float(st.u["hp_abs"][0, H])

    def test_attacking_a_far_enemy_from_melee_strikes_nobody_the_touched_one_is_fought(self):
        assert P.sim["contact"]["attack_leave"] == 1
        assert self.dealt("far") == pytest.approx(0.0, abs=1e-3)
        assert self.dealt("near") > 0
        # off (and the old full strikes on a touched non-target): fights on
        assert self.dealt("far", P.with_cal("contact", attack_leave=0, fresh_incidental=None)) > 0

    def test_a_missile_unit_is_not_concerned(self):
        archers = "wh_main_emp_inf_crossbowmen"
        assert self.dealt("far", key=archers) == pytest.approx(self.dealt("far", P.with_cal("contact", attack_leave=0),
                                                                               key=archers))


# --- the fresh-order probe (build/charge-probe/runs/20261008-145559, -145641): a new order in melee ---

class TestFreshOrders:
    """Swordsmen and clanrats attack each other from 3 m, a second clanrat unit 4 m beside the clanrats (held); 10 s
    after contact the swordsmen get a new order (tools/nn/charge_probe.py plan fresh)."""
    def lane(self, params=P):
        st = scenario.build([army([(SWORD, 0, 20, 180)], [(CLANRATS, 0, -10, 0), (CLANRATS, 34, -10, 0)])], P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["z"][0, H] = st.u["z"][0, H + 1] = -depth[0, H] / 2
        st.u["x"][0, H + 1] = 34.0
        st.u["z"][0, 0] = 3.0 + depth[0, 0] / 2
        st.u["unbreakable"][:] = True
        for _ in range(int(12 / params.dt)):
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, False), str(H): (O.ATTACK, 0, False)}), params)
        assert bool(st.u["m"][0, 0])
        return st, H

    def after(self, order, seconds, params=P):
        """HP the first clanrats and the second lose in `seconds` after the order; the swordsmen's x moved."""
        st, H = self.lane(params)
        hp0, hp1, x0 = float(st.u["hp_abs"][0, H]), float(st.u["hp_abs"][0, H + 1]), float(st.u["x"][0, 0])
        kind, arg = order
        if kind == O.MOVE:
            arg = (float(st.u["x"][0, 0]), float(st.u["z"][0, 0]) + arg)
        elif kind == O.ATTACK:
            arg = H + arg
        for k in range(int(seconds / params.dt)):
            o = orders(st, **{"0": (kind, arg, kind == O.ATTACK), str(H): (O.ATTACK, 0, False)})
            if k > 0:
                o.kind[0, 0] = O.KEEP
            battle.step(st, o, params)
        return hp0 - float(st.u["hp_abs"][0, H]), hp1 - float(st.u["hp_abs"][0, H + 1]), float(st.u["x"][0, 0]) - x0

    def test_attacking_the_near_second_unit_spares_the_first_and_walks_to_the_second(self):
        # the game: the first clanrats lost 0 HP for 30 s, the swordsmen walked 8 m to the second unit in 6-8 s
        first, second, dx = self.after((O.ATTACK, 1), 10.0)
        assert first == pytest.approx(0.0, abs=1e-3) and dx > 3.0
        old = P.with_cal("contact", fresh_incidental=None, retarget_walk=0)
        first_old, _, dx_old = self.after((O.ATTACK, 1), 10.0, old)
        assert first_old > 100.0 and abs(dx_old) < 1.0                      # the old rule: fights on where it stands

    def test_a_walk_5_m_back_is_a_leave_until_the_window(self):
        # the game: swordsmen told to walk 5 m back struck nothing for 21-25 s (chased by the clanrats attacking them)
        first, _, _ = self.after((O.MOVE, 5.0), 15.0)
        assert first == pytest.approx(0.0, abs=1e-3)
        held, _, _ = self.after((O.MOVE, 5.0), 15.0, P.with_cal("contact", leave_away_m=10.0))
        assert held > 50.0                                                  # within leave_m: fought on at hold_rate
        unlatched, _, _ = self.after((O.MOVE, 5.0), 15.0, P.with_cal("contact", leave_latch=0))
        assert unlatched > 0.0                                              # arrived, it would fight again at once


# --- C2 / R4: leaving melee, the chase and its 24 s window ---

class TestMeleeExit:
    def leave(self, chased, seconds, params=None):
        """Swordsmen fight clanrats 10 s, then withdraw west; the clanrats attack them (chased) or hold."""
        P = params or globals()["P"]
        st = face_off(SWORD, CLANRATS)
        H = st.N // 2
        for _ in range(20):
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, False), str(H): (O.ATTACK, 0, False)}), P)
        assert bool(st.u["m"][0, 0])
        hp0, hpH = float(st.u["hp_abs"][0, 0]), float(st.u["hp_abs"][0, H])
        out_at, struck = None, False
        for k in range(int(seconds / P.dt)):
            o = orders(st, **{"0": (O.WITHDRAW, (-300.0, 0.0), True)})
            if k > 0:                                                # the withdraw once, then no new orders
                o.kind[:] = O.KEEP
            if chased:
                o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, True
            hpH_before = float(st.u["hp_abs"][0, H])
            battle.step(st, o, P)
            if out_at is None and not bool(st.u["m"][0, 0]):
                out_at = (k + 1) * P.dt
            if float(st.u["hp_abs"][0, H]) < hpH_before - 1e-6:
                struck = struck or (k + 1) * P.dt
        return st, out_at, hp0 - float(st.u["hp_abs"][0, 0]), struck

    def test_an_unchased_unit_is_held_2_s_then_walks_out_within_5_s_as_in_the_fatleave_probe(self):
        # the game (fatleave, build/probes7/leavefat.py): out of the enemy's melee at 5.5-6 s, health lost until
        # 3.0-5.5 s; held contact.pin_melee_s 2 s, then the about-face and the walk out
        assert P.sim["contact"]["pin_melee_s"] == 2.0
        st, out_at, lost, _ = self.leave(chased=False, seconds=10)
        assert out_at is not None and 3.0 <= out_at <= 5.0 and lost < 90.0
        st0, out0, _, _ = self.leave(chased=False, seconds=10, params=P.with_cal("contact", pin_melee_s=0.0))
        assert out0 is not None and out0 < out_at                   # without the hold: out sooner

    def test_a_chased_unit_stays_in_contact_until_the_window_then_fights(self):
        st, out_at, lost, struck = self.leave(chased=True, seconds=30)
        brk = P.battle["melee_breakoff_secs"]
        assert out_at is None or out_at >= brk                      # the chasers keep it in contact
        assert lost > 100.0                                          # struck from behind the whole time
        assert struck and struck >= brk - P.dt                       # it strikes back only after the window


# --- R6: the run-up only towards the enemy ---

def test_running_away_from_the_enemy_is_no_run_up():
    st = face_off(SWORD, CLANRATS, gap=40.0)
    H = st.N // 2
    for _ in range(12):                                              # run 6 s west, away from the clanrats
        battle.step(st, orders(st, **{"0": (O.MOVE, (-300.0, 0.0), True)}), P)
    assert float(st.u["runup"][0, 0]) == pytest.approx(0.0, abs=1e-6)
    st2 = face_off(SWORD, CLANRATS, gap=60.0)
    for _ in range(12):                                              # run towards them
        battle.step(st2, orders(st2, **{"0": (O.ATTACK, H, True)}), P)
    assert float(st2.u["runup"][0, 0]) > 10.0


# --- R7: the wounded pool ---

def test_the_wounded_pool_turns_melee_hp_into_whole_men_beyond_its_cap():
    st = face_off(SWORD, SLAVE)
    H = st.N // 2
    hm = float(st.u["hp_man"][0, H])
    # many wounded already (W = 30 men's worth of HP missing among the living): new blows kill whole men
    st.u["hp_abs"][0, H] = float(st.u["hp_abs"][0, H]) - 30 * hm * 0.6
    men0, hp0 = float(st.u["men"][0, H]), float(st.u["hp_abs"][0, H])
    battle.step(st, orders(st, **{"0": (O.ATTACK, H, False)}), P)
    lost_hp = hp0 - float(st.u["hp_abs"][0, H])
    lost_men = men0 - float(st.u["men"][0, H])
    assert lost_hp > 0 and lost_men == pytest.approx(lost_hp / hm, rel=0.02)
    # with the exponent rule (pool off) fewer men die for the same HP
    st2 = face_off(SWORD, SLAVE)
    st2.u["hp_abs"][0, H] = float(st2.u["hp_abs"][0, H]) - 30 * hm * 0.6
    battle.step(st2, orders(st2, **{"0": (O.ATTACK, H, False)}), P.with_cal("kills", wound_pool=0))
    assert men0 - float(st2.u["men"][0, H]) < lost_men


# --- O1: projectile kills - hits spread evenly over the living men, each with his own health ---

def _poisson_alive(K, tau):
    return sum(math.exp(-tau) * tau ** k / math.factorial(k) for k in range(K))


def test_shot_kills_follow_even_hits_on_men_with_their_own_health():
    # 180 men of 50 HP, hits of 50/3 HP (3 to kill): volleys of 60 hits; the unit's own bookkeeping (HP lost = hits x
    # HP a hit, men by the rule) keeps the Poisson living share 180 Q(3, hits / man so far)
    men, hp_man, hit = torch.tensor([180.0]), torch.tensor([50.0]), torch.tensor([50.0 / 3])
    hp, total, tau = torch.tensor([9000.0]), 0.0, 0.0
    for v in range(6):
        k = melee.shot_kills(torch.tensor([60.0]), men, hp_man, hp, hit)
        tau += 60.0 / float(men)
        men, hp, total = men - k, hp - 60.0 * hit, total + 60
        assert float(men) == pytest.approx(180 * _poisson_alive(3, tau), rel=0.06, abs=1.5)
    assert float(men) < 0.75 * 180                   # more men die than the old HP-share rule kept late


def test_a_first_volley_on_fresh_men_kills_only_the_men_struck_k_times():
    k = melee.shot_kills(torch.tensor([64.0]), torch.tensor([180.0]), torch.tensor([50.0]), torch.tensor([9000.0]),
                         torch.tensor([50.0 / 3]))
    assert float(k) == pytest.approx(180 * (1 - _poisson_alive(3, 64 / 180)), rel=0.05)   # ~1 man (the game: 0-3)
    one = melee.shot_kills(torch.tensor([10.0]), torch.tensor([90.0]), torch.tensor([10.0]), torch.tensor([900.0]),
                           torch.tensor([10.0]))
    assert float(one) == pytest.approx(10.0)          # a hit that kills is a man


def test_a_shot_unit_keeps_fewer_men_for_its_health_than_the_exponent_rule():
    st = face_off(SPEAR, SLAVE, gap=80.0)
    H = st.N // 2
    hm = float(st.u["hp_man"][0, H])
    men0 = float(st.u["men"][0, H])
    # half its men's worth of wounds among the living already: the next hits kill more often
    st.u["hp_abs"][0, H] = men0 * hm * 0.5
    hits = torch.tensor([30.0])
    k_new = float(melee.shot_kills(hits, st.u["men"][0, H:H + 1], st.u["hp_man"][0, H:H + 1],
                                   st.u["hp_abs"][0, H:H + 1], torch.tensor([hm / 3])))
    k_old = float(30 * (hm / 3) / hm * melee.kill_share(torch.tensor(hm), torch.tensor(hm / 3), 0.5))
    assert k_new > 1.5 * k_old


# --- C1: shattering by the database's thresholds ---

class TestShatter:
    def ctx(self, st):
        z = torch.zeros_like(st.u["men"], dtype=torch.bool)
        return {"aura": torch.zeros_like(st.u["men"]), "lord_dead_points": torch.zeros_like(st.u["men"]), "secure": z,
                "in_melee": z, "in_combat": z, "foe_lost": torch.zeros_like(st.u["men"]), "collapse": z,
                "routing_friends": torch.zeros_like(st.u["men"]), "routing_enemies": torch.zeros_like(st.u["men"]),
                "under_fire": z, "strong_enemy": z, "enemy_near": z | True}

    @pytest.mark.parametrize("routs, hp, shattered", [(1, 0.04, True), (1, 0.06, False), (2, 0.08, True),
                                                     (2, 0.12, False)])
    def test_a_routing_unit_shatters_below_the_database_health(self, routs, hp, shattered):
        st = face_off(SPEAR, SLAVE, gap=200.0)
        st.u["r"][0, 0] = True
        st.u["rout_count"][0, 0] = float(routs)
        st.u["rout_s"][0, 0] = 5.0
        st.u["hp"][0, 0] = hp
        st.u["morale"][0, 0] = -5.0
        morale.step(st.u, self.ctx(st), P, P.dt)
        assert bool(st.u["s"][0, 0]) == shattered

    def test_morale_at_the_floor_shatters_without_army_collapse_unless_unbreakable(self, monkeypatch):
        monkeypatch.setattr(morale, "target_points", lambda u, ctx, params, parts=None: torch.full_like(u["morale"], -90.0))
        for unbreakable, routing in ((False, True), (False, False), (True, False)):
            st = face_off(SPEAR, SLAVE, gap=200.0)
            st.u["morale"][0, 0] = -49.0
            st.u["r"][0, 0] = routing
            st.u["rout_count"][0, 0] = float(routing)
            st.u["hp"][0, 0] = 0.8
            st.u["unbreakable"][0, 0] = unbreakable
            morale.step(st.u, self.ctx(st), P, P.dt)
            assert bool(st.u["s"][0, 0]) == (not unbreakable), (unbreakable, routing)


# --- C5: the rally ---

class TestRally:
    def rally_after(self, enemy_x, enemy_routing, seconds=40):
        # (a second unit of ours far off keeps the army from collapsing)
        st = scenario.build([army([(SPEAR, 0, 0, 90), (SPEAR, -600, 0, 90)],
                                  [(SLAVE, enemy_x, 0, 270), (SLAVE, 700, 400, 270)])], P)   # (the battle goes on)
        H = st.N // 2
        st.u["r"][0, 0] = True
        st.u["rout_count"][0, 0] = 1.0
        st.u["rout_s"][0, 0] = 30.0
        st.u["morale"][0, 0] = 20.0
        if enemy_routing:
            st.u["r"][0, H] = True
        for k in range(int(seconds / P.dt)):
            st.u["x"][0, 0], st.u["z"][0, 0] = 0.0, 0.0                 # kept in place
            st.u["x"][0, H], st.u["z"][0, H] = float(enemy_x), 0.0
            st.u["morale"][0, H] = -5.0 if enemy_routing else st.u["morale"][0, H]
            battle.step(st, replay.hold(st), P)
            if not bool(st.u["r"][0, 0]):
                return (k + 1) * P.dt
        return None

    def test_a_routing_enemy_near_blocks_the_rally(self):
        assert self.rally_after(80.0, enemy_routing=True) is None
        assert self.rally_after(120.0, enemy_routing=True) is not None

    def test_the_rally_comes_after_the_wait(self):
        t = self.rally_after(300.0, enemy_routing=False)
        assert t is not None and t >= P.sim["morale"]["rally_wait_s"] - P.dt - 1e-6


# --- C3: no charge into a target already fighting our units ---

def test_no_sprint_and_no_charge_into_a_target_already_in_melee_with_ours():
    def approach(busy):
        rows = [(SWORD, -40, 0, 90)] + ([(SPEAR, 30, 0, 270)] if busy else [])
        st = scenario.build([army(rows, [(CLANRAT, 0, 0, 270)])], P)
        H = st.N // 2
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        if busy:   # our spearmen already in contact with the clanrats' rear (the swordsmen come from the front)
            st.u["x"][0, 1] = float(st.u["x"][0, H]) + (float(depth[0, 1]) + float(depth[0, H])) / 2 + TOUCH - 0.5
        speeds, charged = [], False
        for _ in range(40):
            o = orders(st, **{"0": (O.ATTACK, H, True)})
            if busy:
                o.kind[0, 1], o.target[0, 1] = O.ATTACK, H
            battle.step(st, o, P)
            speeds.append(float(torch.hypot(st.u["vx"][0, 0], st.u["vz"][0, 0])))
            charged = charged or float(st.u["charge"][0, 0]) > 0.5
            if bool(st.u["m"][0, 0]):
                break
        return max(speeds), charged
    free_v, free_c = approach(False)
    busy_v, busy_c = approach(True)
    run = P.units[SWORD]["speed"]["run"]
    assert free_v > busy_v + 0.3 and free_c and not busy_c and busy_v <= run + 0.05


# --- C4: the flank and rear sectors ---

def test_the_sectors_are_45_and_135_degrees():
    sector = geometry.sector(torch.tensor([40.0, 50.0, 130.0, 140.0]) * geometry.DEG, P.sim["contact"]["front_deg"],
                             P.sim["contact"]["rear_deg"])
    assert sector.tolist() == [0, 1, 1, 2]
