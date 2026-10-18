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
ARCHER = "wh2_dlc13_emp_inf_archers_0"
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
        # a move's point near (R3's 5 m lanes: held; a far one through the enemy strikes in full, contact.move_far_full)
        st.u["ox"][0, 0], st.u["oz"][0, 0] = st.u["x"][0, 0] + 3.0, st.u["z"][0, 0]
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
    def dealt(self, point_x, params=P, in_melee=False, key=SPEAR):
        """The spearmen's move order given before the contact (P3: the planner's push from the go), or in melee."""
        st = face_off(key, CLANRAT)            # spearmen west facing east, clanrat spearmen east
        H = st.N // 2
        move = orders(st, **{"0": (O.MOVE, (point_x, 0.0), False), str(H): (O.ATTACK, 0, False)})
        o = orders(st, **{str(H): (O.ATTACK, 0, False)}) if in_melee else move
        for _ in range(6):
            battle.step(st, o, params)
        hp = float(st.u["hp_abs"][0, H])
        for _ in range(10):
            battle.step(st, move, params)
        return hp - float(st.u["hp_abs"][0, H]), bool(st.u["m"][0, 0])

    def test_a_move_through_the_enemy_given_in_melee_is_a_leave(self):
        # the meleeorders probe: run 5 / 15 / 40 m ahead through the clanrats from melee - 0 HP dealt at 5-20 s
        through, _ = self.dealt(60.0, in_melee=True)
        assert through == pytest.approx(0.0, abs=1e-3)
        old, _ = self.dealt(60.0, P.with_cal("contact", move_melee_leave=0), in_melee=True)
        assert old > 0

    def test_a_far_point_through_the_enemy_fights_on_one_away_leaves(self):
        through, m_through = self.dealt(60.0)
        away, _ = self.dealt(-200.0)
        assert through > 0 and m_through and away == pytest.approx(0.0, abs=1e-3)
        off, _ = self.dealt(60.0, P.with_cal("contact", leave_away_only=0))
        assert off == pytest.approx(0.0, abs=1e-3)

    def test_a_shooters_far_point_through_the_enemy_is_a_leave(self):
        # contact.missile_leave_any (the 72 gate battles: the game AI's shooters in melee under a run move through the
        # nearest enemy moved 0.92 of the seconds at 2.3 m/s): a formation with a missile weapon leaves whatever the
        # direction; the spearmen of P3 still fight on
        assert P.sim["contact"]["missile_leave_any"]
        through, _ = self.dealt(60.0, key=ARCHER)
        assert through == pytest.approx(0.0, abs=1e-3)
        off, _ = self.dealt(60.0, P.with_cal("contact", missile_leave_any=0), key=ARCHER)
        spear, _ = self.dealt(60.0)
        assert off > 0 and spear > 0

    def test_a_far_point_through_the_enemy_strikes_in_full_a_near_one_held(self):
        # on in sim.json (P3 and the walkpass probe's 'through' lanes: the walker pushing in under its move order strikes
        # the held clanrats at 0.6 of the attacking rate; held: 0.33)
        assert P.sim["contact"]["move_far_full"] == 1
        far, _ = self.dealt(60.0)
        held, _ = self.dealt(60.0, P.with_cal("contact", move_far_full=0))
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

    def test_a_lord_attacking_a_far_enemy_from_melee_strikes_nobody(self):
        # contact.attack_leave_lord (build/gaps/lord.md: the network's lord in melee told to attack a target 15+ m away
        # kills 0.02-0.09 a second in the game, moving 51-79 % of the time; the twin before 0.21-0.28)
        general = "wh_main_emp_cha_general_0"
        assert P.sim["contact"]["attack_leave_lord"] == 1
        assert self.dealt("far", key=general) == pytest.approx(0.0, abs=1e-3)
        assert self.dealt("near", key=general) > 0
        assert self.dealt("far", P.with_cal("contact", attack_leave_lord=0), key=general) > 0

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
        # the second clanrats (held, out of the fight) stand where the probe's did at the order: their near edge
        # 4.4-5.2 m behind the swordsmen's front line (both runs at 10-12.5 s; the start line above put it level with
        # that front, 3 m beside the swordsmen's flank - a corner contact after a 2 m step aside with rect_gap)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        st.u["z"][0, H + 1] = st.u["z"][0, 0] - depth[0, 0] / 2 - 4.5 - depth[0, H + 1] / 2
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


class TestHoldBreakoff:
    """The holdmelee probe (build/charge-probe/runs/20261011-090746, -090858): 10 s after contact the swordsmen get a
    run 15 m back or an attack on the clanrats aside, then a halt 3 s later; or a halt only. contact.hold_breakoff."""
    lane = TestFreshOrders.lane
    def halted(self, first, params=P, halt_at=3.0, until=30.0):
        """Both clanrat units' HP lost a second by the swordsmen at (halt, first order + 24 s) and after it to `until`."""
        st, H = self.lane(params)
        win = float(params.battle["melee_breakoff_secs"])
        lost, k_end = [], int(until / params.dt)
        for k in range(k_end):
            t = k * params.dt
            if first is None or t >= halt_at:
                o = orders(st, **{"0": (O.HOLD, None, False), str(H): (O.ATTACK, 0, False)})
                if k > 0 and (first is None or t > halt_at):
                    o.kind[0, 0] = O.KEEP
            else:
                kind, arg = first
                arg = (float(st.u["x"][0, 0]), float(st.u["z"][0, 0]) + arg) if kind == O.MOVE else H + arg
                o = orders(st, **{"0": (kind, arg, True), str(H): (O.ATTACK, 0, False)})
                if k > 0:
                    o.kind[0, 0] = O.KEEP
            before = float(st.u["hp_abs"][0, H] + st.u["hp_abs"][0, H + 1])
            battle.step(st, o, params)
            lost.append((t, before - float(st.u["hp_abs"][0, H] + st.u["hp_abs"][0, H + 1])))
        quiet = sum(v for t, v in lost if halt_at + params.dt <= t < win - params.dt)
        back = sum(v for t, v in lost if t >= win + params.dt)
        return quiet / (win - halt_at - 2 * params.dt), back / (until - win - params.dt)

    def test_a_halt_after_a_leave_or_an_attack_aside_strikes_nobody_to_the_window(self):
        assert P.sim["contact"]["hold_breakoff"] == 1
        for first in ((O.MOVE, 15.0), (O.ATTACK, 1)):
            quiet, back = self.halted(first)
            assert quiet == pytest.approx(0.0, abs=1e-3)                  # the game: 0 HP for 10-25 s, 6 lanes of 6
            assert back > 5.0                                              # fights again from the window on
            off_quiet, _ = self.halted(first, P.with_cal("contact", hold_breakoff=0))
            assert off_quiet > 5.0                                         # off: the hold fights at once

    def test_a_plain_halt_fights_on(self):
        quiet, _ = self.halted(None)
        assert quiet > 5.0
        assert quiet == pytest.approx(self.halted(None, P.with_cal("contact", hold_breakoff=0))[0])


class TestHoldBreakoffLord:
    """The holdlord probe (build/charge-probe/runs/20261011-143705, -143803; build/hold/holdlord.md): the General fights
    clanrats 10 s, then an attack on a second clanrat unit 25 m aside and a halt 3 s later - he strikes the clanrats he
    touches nothing for 15 s after the halt, back 22.5 s / over 40 s after the attack order. contact.hold_breakoff_lord."""
    GENERAL = "wh_main_emp_cha_general_0"

    def lane(self, params=P):
        st = scenario.build([army([(self.GENERAL, 0, 20, 180, True)], [(CLANRATS, 0, -10, 0), (CLANRATS, 60, -10, 0)])],
                            P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["z"][0, H] = st.u["z"][0, H + 1] = -depth[0, H] / 2
        st.u["x"][0, H + 1] = 60.0
        st.u["z"][0, 0] = 0.5
        st.u["unbreakable"][:] = True
        for _ in range(int(10 / params.dt)):
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, False), str(H): (O.ATTACK, 0, False)}), params)
        assert bool(st.u["m"][0, 0])
        return st, H

    halted = TestHoldBreakoff.halted

    def test_a_halt_after_an_attack_aside_keeps_the_lord_out_to_the_window(self):
        assert P.sim["contact"]["hold_breakoff_lord"] == 1
        quiet, back = self.halted((O.ATTACK, 1))
        assert quiet == pytest.approx(0.0, abs=1e-3)                       # the game: 0 HP for 15 s after the halt
        assert back > 5.0                                                   # fights again from the window on
        off_quiet, _ = self.halted((O.ATTACK, 1), P.with_cal("contact", hold_breakoff_lord=0))
        assert off_quiet > 5.0                                              # the old rule: the halt fought at once

    def test_a_plain_halt_of_the_lord_is_not_concerned(self):
        quiet, _ = self.halted(None)
        assert quiet > 5.0
        assert quiet == pytest.approx(self.halted(None, P.with_cal("contact", hold_breakoff_lord=0))[0])


# --- the recontact probe: the opening (wave, first strike) comes at a formation's first contact only ---

class TestOpeningOnce:
    """Swordsmen charge held clanrats from 30 m, fight 20 s, run 40 m straight back, wait, charge again (the recontact
    probe, build/charge-probe/runs/20261011-143904, -144020, -144146: the game's first 3 s after a second contact 0.24-1.35
    of its 10-20 s rate after 2-80 s out, the first contact's 2.87). contact.opening_once."""
    def run(self, params=P, wait_s=20.0):
        """(HP the clanrats lose a second in 0-3 s after the first / the second contact, the same in 10-20 s after
        the second; the swordsmen's loss 0-3 s after the second)."""
        st = face_off(SWORD, CLANRATS, gap=30.0)
        H = st.N // 2
        st.u["unbreakable"][:] = True
        dt = params.dt
        log, phase, t_phase = [], "in1", 0.0
        x_back = None
        for k in range(int(200 / dt)):
            if phase in ("in1", "in2"):
                o = orders(st, **{"0": (O.ATTACK, H, True)})
            elif phase == "fight":
                o = orders(st, **{"0": (O.ATTACK, H, True)})
                if t_phase >= 20.0:
                    phase, t_phase = "back", 0.0
                    x_back = float(st.u["x"][0, 0]) - 40.0
            if phase == "back":
                o = orders(st, **{"0": (O.MOVE, (x_back, float(st.u["z"][0, 0])), True)})
                if not bool(st.u["m"][0, 0]) and abs(float(st.u["x"][0, 0]) - x_back) < 1.0:
                    phase, t_phase = "wait", 0.0
            if phase == "wait":
                o = orders(st, **{"0": (O.HOLD, None, False)})
                if t_phase >= wait_s:
                    phase, t_phase = "in2", 0.0
            hp_tg, hp_a, m0 = float(st.u["hp_abs"][0, H]), float(st.u["hp_abs"][0, 0]), bool(st.u["m"][0, 0])
            battle.step(st, o, params)
            m1 = bool(st.u["m"][0, 0])
            if phase in ("in1", "in2") and m1 and not m0:
                log.append(("c1" if phase == "in1" else "c2", k * dt))
                phase = "fight" if phase == "in1" else "after"
                t_phase = 0.0
            log.append((k * dt, hp_tg - float(st.u["hp_abs"][0, H]), hp_a - float(st.u["hp_abs"][0, 0])))
            t_phase += dt
            if phase == "after" and t_phase > 21.0:
                break
        marks = {r[0]: r[1] for r in log if isinstance(r[0], str)}
        rows = [r for r in log if not isinstance(r[0], str)]

        def rate(c, lo, hi, i=1):
            return sum(r[i] for r in rows if marks[c] + lo - 1e-6 <= r[0] < marks[c] + hi - 1e-6) / (hi - lo)
        return rate("c1", 0, 3), rate("c2", 0, 3), rate("c2", 10, 20), rate("c2", 0, 3, 2), rate("c2", 10, 20, 2)

    def test_a_second_contact_after_a_time_out_has_no_new_opening(self):
        assert P.sim["contact"]["opening_once"] == 1
        first, re_0_3, re_10_20, a_0_3, a_10_20 = self.run()
        old = self.run(P.with_cal("contact", opening_once=0))
        assert old[1] > 2.5 * old[2]                       # the old rule: the second contact bursts like the first
        assert re_0_3 < 0.5 * old[1]
        assert re_0_3 < 2.0 * re_10_20                     # what is left is the charge bonus (fading over 13 s)
        assert first > 2.0 * re_0_3                        # the first contact keeps its opening
        assert a_0_3 < 1.3 * a_10_20                       # the held clanrats strike no new first blow either
        assert old[3] > 1.5 * a_0_3


# --- the first strike swings the steady front, not the opening wave's men ---

def test_the_first_strike_counts_the_steady_front_not_the_wave():
    """contact.first_strike_steady: the charger's contact step with the wave at its peak (x2.48) - the swing is the
    same as with no wave at all, the rate part keeps the wave."""
    def contact_step(params):
        st = face_off(SWORD, CLANRATS, gap=14.0)
        H = st.N // 2
        for _ in range(40):
            hp0 = float(st.u["hp_abs"][0, H])
            m0 = bool(st.u["m"][0, 0])
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, True)}), params)
            if bool(st.u["m"][0, 0]) and not m0:
                return hp0 - float(st.u["hp_abs"][0, H])
        return None
    # (the charge's front rank and the pause after the swing off: this rule alone - contact.first_strike_rank and
    # melee.swing_interval have their own tests in test_sim_melee_core.py)
    PB = P.with_cal("contact", first_strike_rank=0).with_cal("melee", swing_interval=0)
    assert PB.sim["contact"]["first_strike_steady"] == 1
    W = dict(PB.sim["melee"]["wave"])
    on = contact_step(PB)
    off = contact_step(PB.with_cal("contact", first_strike_steady=0))
    no_wave = contact_step(PB.with_cal("melee", wave=dict(W, amp=0.0)))
    no_wave_off = contact_step(PB.with_cal("melee", wave=dict(W, amp=0.0)).with_cal("contact", first_strike_steady=0))
    assert no_wave == pytest.approx(no_wave_off, rel=1e-4)    # without a wave the flag changes nothing
    assert on < 0.7 * off                                      # the swing loses the wave's x2.48
    assert on > no_wave                                        # the step's rate keeps the wave


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
        # (lost: the game's leavers 17-82 HP over the 10 s, fatleave L3-L5; here 63 HP before melee.wave, 92 with it -
        # the exit starts 10 s into the fight, still inside the opening wave)
        assert out_at is not None and 3.0 <= out_at <= 5.0 and lost < 100.0
        st0, out0, _, _ = self.leave(chased=False, seconds=10, params=P.with_cal("contact", pin_melee_s=0.0))
        assert out0 is not None and out0 < out_at                   # without the hold: out sooner

    def test_the_engage_hold_lets_a_leaver_go(self):
        # contact.engage_release: the formed combat distance holds only units that stay fighting - a leaver touches by
        # the old hold, so it walks out as in fatleave (with the engage hold on it: 120 HP lost, out at 5 s)
        assert P.sim["contact"]["engage"] and P.sim["contact"]["engage_release"]
        _, out_on, lost_on, _ = self.leave(chased=False, seconds=10)
        _, out_off, lost_off, _ = self.leave(chased=False, seconds=10, params=P.with_cal("contact", engage_release=0))
        assert out_on < out_off and lost_on < lost_off - 15.0

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


def test_front_kills_with_all_hits_on_the_front_rank_kill_sooner_and_lose_the_hits_beyond_a_man():
    # kills.missile_front (melee.front_kills): 120 men of 69 HP, 20 in the front rank, hits of 7.5 HP (~9 to kill), 40 hits
    # a step all on the front rank for 5 steps (200 hits: 10 a front man): the front dies (~ half of it), the even
    # spread kills nobody; the HP lost beyond the dead men's health comes back as lost hits
    men, hp_man, ph = torch.tensor([120.0]), torch.tensor([69.0]), torch.tensor([7.5])
    hp, fw = torch.tensor([8280.0]), torch.tensor([0.0])
    hp_u, men_u = hp.clone(), men.clone()
    dead_f = dead_u = lost = 0.0
    for _ in range(5):
        k, w, fw = melee.front_kills(torch.tensor([40.0]), torch.tensor([0.0]), men, torch.tensor([20.0]), hp_man, hp, ph,
                                     fw)
        hp = hp - (40.0 - w) * ph
        men, dead_f, lost = men - k, dead_f + float(k), lost + float(w)
        ku = melee.shot_kills(torch.tensor([40.0]), men_u, hp_man, hp_u, ph)
        hp_u, men_u, dead_u = hp_u - 40.0 * ph, men_u - ku, dead_u + float(ku)
    assert dead_u < 0.5 and 6 < dead_f < 16
    assert 0 < lost < 0.2 * 200
    # the front's own wounds stay within the unit's
    assert 0 <= float(fw) <= float(men * hp_man - hp) + 0.5
    # all of a step's hits even (none on the front) is shot_kills
    k, w, _ = melee.front_kills(torch.tensor([0.0]), torch.tensor([60.0]), torch.tensor([180.0]), torch.tensor([20.0]),
                                torch.tensor([50.0]), torch.tensor([9000.0]), torch.tensor([50.0 / 3]), torch.tensor([0.0]))
    assert float(k) == pytest.approx(float(melee.shot_kills(torch.tensor([60.0]), torch.tensor([180.0]),
                                                            torch.tensor([50.0]), torch.tensor([9000.0]),
                                                            torch.tensor([50.0 / 3]))), rel=0.05, abs=0.05)


def test_the_front_rank_takes_most_flat_bullets_and_few_arrows():
    # missile.front_share: a Ratling Gun's flat bullets (a 37 m low path at 100 m) on swordsmen mostly strike the front
    # rank; archers' arcing arrows spread over the depth
    from tools.nn.sim import missile
    gun = face_off(SWORD, "wh2_dlc12_skv_inf_ratling_gun_0", gap=100.0)
    bows = face_off(ARCHER, SLAVE, gap=100.0)
    H = gun.N // 2
    sp = P.sim["formation"]["spacing_m"]
    f_gun = float(missile.front_share(gun.u, geometry.pairwise(gun.u, sp), P)[0, H, 0])
    f_bow = float(missile.front_share(bows.u, geometry.pairwise(bows.u, sp), P)[0, 0, H])
    assert 0.6 <= f_gun <= 1.0 and 0.0 <= f_bow <= 0.45 and f_bow < f_gun, (f_gun, f_bow)


def test_a_narrow_column_catches_only_the_shots_across_its_span():
    # missile.accuracy.width (missile.width_capture): the Ratling Gun 100 m off a swordsmen column 6 m wide misses
    # across more often than a 30 m front (the same men deeper); side on, the 6 m column is a wide target again;
    # a span of one man is the lone man's erf
    from tools.nn.sim import missile
    st = face_off(SWORD, "wh2_dlc12_skv_inf_ratling_gun_0", gap=100.0)
    H = st.N // 2
    sp = P.sim["formation"]["spacing_m"]
    u = st.u
    pw = geometry.pairwise(u, sp)
    sig = torch.full_like(pw["dist"], 1.6)
    rx = u["radius"][:, None, :].expand_as(sig)
    wide = float(missile.width_capture(u, pw, sig, rx, P)[0, H, 0])
    pw6 = dict(pw, front=torch.full_like(pw["front"], 6.0))
    narrow = float(missile.width_capture(u, pw6, sig, rx, P)[0, H, 0])
    side = float(missile.width_capture(u, dict(pw6, depth=torch.full_like(pw["depth"], 48.0),
                                               rel_j=torch.full_like(pw["rel_j"], math.pi / 2)), sig, rx, P)[0, H, 0])
    one = float(missile.width_capture(u, dict(pw, front=u["sp_h"].clone()), sig, rx, P)[0, H, 0])
    assert 0.9 < wide <= 1.0 and narrow < 0.9 * wide and side > 0.95
    assert one == pytest.approx(math.erf(0.65 / (math.sqrt(2) * 1.6)), rel=0.02)
    assert missile.width_capture(u, pw, sig, rx, P.with_cal("missile", accuracy={**P.sim["missile"]["accuracy"],
                                                                                    "width": False})) is None


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
    def rally_after(self, enemy_x, enemy_routing, seconds=40, params=P):
        # (a second unit of ours far off keeps the army from collapsing)
        P = params
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

    @staticmethod
    def near_shut():
        """P with the first bin of morale.rally_hazard (any enemy within 95 m: the game's 0.3 % a second) at 0 - to
        see that a near enemy puts the router in that bin."""
        hz = dict(P.sim["morale"]["rally_hazard"])
        hz["p_per_s"] = [0.0] + list(hz["p_per_s"][1:])
        return P.with_cal("morale", rally_hazard=hz)

    def test_a_routing_enemy_near_holds_the_rally_to_the_first_bin(self):
        assert self.rally_after(80.0, enemy_routing=True, params=self.near_shut()) is None
        # (beyond 95 m a routing enemy does not hold it; the nearest standing one is 800 m off: 150 m and beyond)
        assert self.rally_after(120.0, enemy_routing=True, seconds=900) is not None

    def test_the_rally_comes_after_the_wait(self):
        t = self.rally_after(130.0, enemy_routing=False, seconds=300)      # 125-150 m
        assert t is not None and t >= P.sim["morale"]["rally_wait_s"] - P.dt - 1e-6
        assert self.rally_after(80.0, enemy_routing=False, params=self.near_shut()) is None   # within 95 m: first bin


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
    # (charge.blocked_side off: our unit on any side of the target blocks the charge)
    if not P.sim["charge"].get("blocked_side"):
        assert free_v > busy_v + 0.3 and free_c and not busy_c and busy_v <= run + 0.05


def test_a_busy_target_blocks_the_charge_only_from_the_side_ours_fight_it_on():
    """charge.blocked_side (the busycharge probe): swordsmen charge the front of clanrats that our spearmen already fight -
    with the spearmen at the clanrats' back the charge is whole (the sprint, the bonus), with them at their front (between
    the two) there is none."""
    params = P.with_cal("charge", blocked_side=1, free_target_only=1)

    def approach(where):
        rows = [(SWORD, -40, 0, 90)] + [(SPEAR, 30 if where == "rear" else -30, 0, 270 if where == "rear" else 90)]
        st = scenario.build([army(rows, [(CLANRAT, 0, 0, 270)])], params)
        H = st.N // 2
        front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
        off = (float(depth[0, 1]) + float(depth[0, H])) / 2 + TOUCH - 0.5
        st.u["x"][0, 1] = float(st.u["x"][0, H]) + (off if where == "rear" else -off)
        speeds, charged = [], False
        for _ in range(40):
            o = orders(st, **{"0": (O.ATTACK, H, True)})
            o.kind[0, 1], o.target[0, 1] = O.ATTACK, H
            battle.step(st, o, params)
            st.u["b"][0, H] = 270.0                    # the clanrats keep facing the swordsmen
            speeds.append(float(torch.hypot(st.u["vx"][0, 0], st.u["vz"][0, 0])))
            charged = charged or float(st.u["charge"][0, 0]) > 0.5
            if bool(st.u["m"][0, 0]):
                break
        return max(speeds), charged
    rear_v, rear_c = approach("rear")
    front_v, front_c = approach("front")
    run = params.units[SWORD]["speed"]["run"]
    assert rear_c and rear_v > run + 0.3
    assert not front_c and front_v <= run + 0.05


# --- C4: the flank and rear sectors ---

def test_the_sectors_are_45_and_135_degrees():
    sector = geometry.sector(torch.tensor([40.0, 50.0, 130.0, 140.0]) * geometry.DEG, P.sim["contact"]["front_deg"],
                             P.sim["contact"]["rear_deg"])
    assert sector.tolist() == [0, 1, 1, 2]


# --- noise.blows: the copies roll their blows ---

def test_rolled_blows_keep_the_mean_and_spread_as_the_probes_identical_lanes():
    """noise.blows (config/nn/sim.json noise.blows_why): off by default; on, a pair's HP a step is a Poisson number of
    blows of the mean HP a blow - over 48 copies of swordsmen <-> clanrats the HP lost at 30 s keeps the mean (within
    3 %) and spreads by a CV near the probes' identical lanes (0.10 at 30 s: between 0.05 and 0.16)."""
    assert P.sim["noise"]["blows"] is False
    copies = 48

    def lost(params):
        st = scenario.build([army([(SWORD, -50, 0, 90.0)], [(CLANRATS, 50, 0, 270)])] * copies, params)
        front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["x"][:, 0] = -(depth[:, 0] / 2 + TOUCH / 2)
        st.u["x"][:, H] = depth[:, H] / 2 + TOUCH / 2
        hp0 = st.u["hp_abs"][:, H].clone()
        o = replay.hold(st)
        o.kind[:, 0], o.target[:, 0] = O.ATTACK, H
        o.kind[:, H], o.target[:, H] = O.ATTACK, 0
        for k in range(int(30 / params.dt)):
            battle.step(st, o, params)
            if k == 0:
                o.kind[:] = O.KEEP
        return (hp0 - st.u["hp_abs"][:, H]).numpy()
    torch.manual_seed(7)
    mean_det = float(lost(P).mean())
    rolled = lost(P.with_cal("noise", blows=True))
    assert abs(rolled.mean() / mean_det - 1) < 0.03
    cv = rolled.std() / rolled.mean()
    assert 0.05 < cv < 0.16, cv
