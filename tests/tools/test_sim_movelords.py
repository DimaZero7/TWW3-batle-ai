"""The movement, lords, army-destruction, fatigue and melee-exit batch (06.10.2026, build/movelords): each rule of the
game the simulator follows, on one unit or a pair. Needs torch: run in the container (docs/en/training/simulator.md)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, effects, fatigue, morale, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"
ARCHER = "wh2_dlc13_emp_inf_archers_0"
GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
FOE_SEEKER = "wh_main_character_abilities_foe_seeker"


def army(side1, side2, attacker=1):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                            2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def lone(key, faction_side=1, b=0.0):
    """One unit (the far enemy a skavenslave unit 600 m away, never ordered)."""
    if faction_side == 1:
        return scenario.build([army([(key, 0, 0, b, key in (GENERAL,))], [(SLAVE, 600, 600, 0)])], P)
    return scenario.build([army([(SPEAR, 600, 600, 0)], [(key, 0, 0, b, True)])], P)


def move_to(st, i, x, z, run=True):
    o = replay.hold(st)
    o.kind[0, i], o.x[0, i], o.z[0, i], o.run[0, i] = O.MOVE, x, z, run
    return o


def keep(st):
    o = O.hold(st.B, st.N)
    o.kind[:] = O.KEEP
    return o


class TestTurning:
    """turn.move_turn: a unit on the move turns at its men's turn rate (the database's battle_entities turn_rate) and
    runs only the way it faces (speed x cos of the angle off, none beyond 90 deg)."""

    @pytest.mark.parametrize("key,side,rate", [(GENERAL, 1, 120.0), (WARLORD, 2, 180.0), (SPEAR, 1, 120.0)])
    def test_a_unit_ordered_behind_turns_at_its_database_rate_before_it_runs(self, key, side, rate):
        st = lone(key, side, b=0.0)
        i = 0 if side == 1 else st.N // 2
        assert float(st.u["turn"][0, i]) == rate
        battle.step(st, move_to(st, i, 0.0, -150.0), P)          # the point right behind it
        turned = abs((float(st.u["b"][0, i]) + 180) % 360 - 180)
        assert turned == pytest.approx(min(180.0, rate * P.dt), abs=0.01)
        speed = math.hypot(float(st.u["vx"][0, i]), float(st.u["vz"][0, i]))
        off = 180.0 - turned
        assert speed <= max(0.0, math.cos(math.radians(off))) * float(st.u["run"][0, i]) + 1e-6
        if off > 90:
            assert speed == 0.0
        for _ in range(8):
            battle.step(st, keep(st), P)
        assert abs((float(st.u["b"][0, i]) - 180 + 180) % 360 - 180) < 1.0     # faces the way now
        assert math.hypot(float(st.u["vx"][0, i]), float(st.u["vz"][0, i])) > 0.9 * float(st.u["run"][0, i])

    def test_a_unit_facing_its_point_starts_as_before(self):
        st = lone(SPEAR, 1, b=180.0)
        battle.step(st, move_to(st, 0, 0.0, -150.0), P)
        speed = math.hypot(float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0]))
        assert speed == pytest.approx(min(float(st.u["run"][0, 0]), float(st.u["accel"][0, 0]) * P.dt), rel=1e-4)

    def test_off_the_unit_faces_the_way_it_walks_at_once(self):
        p = P.with_cal("turn", move_turn=0)
        st = lone(SPEAR, 1, b=0.0)
        battle.step(st, move_to(st, 0, 0.0, -150.0), p)
        assert float(st.u["b"][0, 0]) == pytest.approx(180.0, abs=0.01)

    def test_a_formation_steps_back_a_few_metres_without_turning(self):
        st = lone(SPEAR, 1, b=0.0)
        battle.step(st, move_to(st, 0, 0.0, -5.0, run=False), P)     # within contact.step_m
        assert float(st.u["b"][0, 0]) == pytest.approx(0.0, abs=0.01)


class TestArmyStrength:
    """morale.collapse strength "cp": (the database's melee_cp + abilities' cp + missile_cp x the ammunition curve)
    x health; routers at routing_weight in the side's sum."""

    def test_combat_potential_of_lords_and_shooters(self):
        st = scenario.build([army([(GENERAL, 0, 0, 0, True), (ARCHER, 50, 0, 0)], [(WARLORD, 600, 0, 0, True)])], P)
        H = st.N // 2
        assert float(st.u["cp_fixed"][0, 0]) == 950.0 and float(st.u["cp_fixed"][0, H]) == 900.0
        assert float(st.u["cp_fixed"][0, 1]) == 100.0 and float(st.u["cp_missile"][0, 1]) == 250.0
        cal = P.sim["morale"]["collapse"]
        st.u["a"][0, 1] = 0
        share = morale.ammo_share(st.u, float(cal["ammo_slope"]), float(cal["ammo_midpoint"]))
        assert float(share[0, 1]) == pytest.approx(1 / (1 + math.exp(14 * 0.25)), rel=1e-5)   # 2.9 % of missile_cp

    def test_the_thresholds_on_the_combat_potential(self):
        # General (950) + archers (350) against the Warlord (900): the Empire at 0.2 health, the Warlord full
        st = scenario.build([army([(GENERAL, 0, 0, 0, True), (ARCHER, 50, 0, 0)], [(WARLORD, 600, 0, 0, True)])], P)
        u = st.u
        H = st.N // 2
        u["hp"][0, 0] = u["hp"][0, 1] = 0.2              # own 0.2 x 1300 = 260 <= 0.22 x 1300; 900 >= 2.6 x 260
        assert bool(morale.army_collapse(u, P)[0, 0]) and not bool(morale.army_collapse(u, P)[0, H])
        u["hp"][0, 0] = u["hp"][0, 1] = 0.3              # 390: above 0.22 x 1300 = 286
        assert not bool(morale.army_collapse(u, P)[0, 0])
        u["hp"][0, 0] = u["hp"][0, 1] = 0.2
        u["hp"][0, H] = 0.5                              # 450 < 2.6 x 260
        assert not bool(morale.army_collapse(u, P)[0, 0])


class TestLordFatigueAndWounds:
    def _melee_activity(self, n, single):
        z = torch.zeros(n, dtype=torch.bool)
        return {"melee": torch.ones(n, dtype=torch.bool), "charging": z, "shooting": z, "running": z, "walking": z,
                "idle": z, "active": torch.ones(n, dtype=torch.bool), "attack": torch.ones(n, dtype=torch.bool),
                "single": torch.tensor(single), "run_order": z, "routing": z}

    def test_a_lord_in_melee_gains_the_database_19_a_tick_and_foe_seeker_takes_1_percent_a_second(self):
        u = {"fatigue": torch.full((2,), 10000.), "fat": torch.zeros(2)}
        act = self._melee_activity(2, [True, True])
        act["vigour"] = torch.tensor([0.0, -0.01])
        fatigue.step(u, act, P, 1.0)
        assert u["fatigue"].tolist() == pytest.approx([10190.0, 10190.0 - 300.0])

    def test_foe_seeker_vigour_while_active(self):
        st = scenario.build([army([(GENERAL, 0, 0, 0, True)], [(SLAVE, 600, 0, 0)])], P)
        assert P.abilities[FOE_SEEKER]["vigour_per_s"] == -0.01
        slots = abilities.slot_keys(GENERAL, P.units, P.abilities)
        k = slots.index(FOE_SEEKER)
        standing = torch.ones_like(st.u["r"])
        assert float(abilities.vigour(st.u, P, standing)[0, 0]) == 0.0
        st.u[f"ab{k}_on"][0, 0] = 10.0
        assert float(abilities.vigour(st.u, P, standing)[0, 0]) == pytest.approx(-0.01)

    def test_wounds_come_on_5_s_after_a_lord_falls_below_a_quarter_and_stay(self):
        assert effects.quarter_delay(P) == 5.0
        st = scenario.build([army([(GENERAL, 0, 0, 0, True)], [(SLAVE, 600, 0, 0)])], P)
        u = st.u
        dmg0, run0 = float(u["damage"][0, 0]), float(u["run"][0, 0])
        u["hp_abs"][0, 0] = 0.2 * float(u["hp0"][0, 0])
        u["hp"][0, 0] = 0.2
        seen = []
        for _ in range(14):
            orig = {k: u[k].clone() for k in ("damage", "run")}
            innate = effects.apply(u, P, P.dt, torch.ones_like(u["r"]), torch.zeros_like(u["r"]), torch.zeros(1, st.N, st.N),
                                   torch.ones(1, st.N, st.N, dtype=torch.bool))
            seen.append((float(u["low_s"][0, 0]), float(u["damage"][0, 0]) / dmg0, float(u["run"][0, 0]) / run0))
            effects.restore(u, innate)
            u["low_s"][0, 0] += P.dt                  # battle.py: the time since it fell below a quarter
            u["hp"][0, 0] = 0.3 if len(seen) > 12 else 0.2   # healed above: it stays (Update 2.0)
        for low, d, r in seen:
            on = low >= 5.0
            assert d == pytest.approx(0.8 if on else 1.0) and r == pytest.approx(0.9 if on else 1.0)
        assert seen[-1][1] == pytest.approx(0.8)


class TestMeleeExit:
    """contact.breakoff: a unit still touching an enemy melee_breakoff_secs (24 s) after it began to leave drops its
    order and fights again."""

    def test_a_chased_unit_fights_again_after_the_breakoff_window(self):
        from tests.tools.test_sim import face_off
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        st.u["morale"][:] = 1e6
        st.u["leadership"][:] = 1e6
        brk = P.rules["battle"]["melee_breakoff_secs"]
        assert brk == 24.0
        kinds, kills = [], []                                   # (kills: the target's HP lost)
        d0 = float(st.u["x"][0, H] - st.u["x"][0, 0])
        hp_h0 = float(st.u["hp_abs"][0, H])
        for n in range(int(40 / P.dt)):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            if n == 0:
                o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.WITHDRAW, -300.0, 0.0, True
            else:
                o.kind[0, 0] = O.KEEP
            # the enemy keeps up: the leaver stays in contact (here: put back against it)
            st.u["x"][0, 0] = st.u["x"][0, H] - d0
            battle.step(st, o, P)
            kinds.append(int(st.u["order_kind"][0, 0]))
            kills.append(hp_h0 - float(st.u["hp_abs"][0, H]))
        drop = kinds.index(O.HOLD)
        assert (drop + 1) * P.dt == pytest.approx(brk + P.dt, abs=P.dt)
        assert kills[drop - 1] - kills[4] == pytest.approx(0.0, abs=1e-6)      # no blows while leaving
        assert kills[-1] > kills[drop]                                           # it fights again

    def test_the_window_runs_from_leaving_through_gaps_out_of_contact(self):
        from tests.tools.test_sim import face_off
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        st.u["morale"][:] = 1e6
        d0 = float(st.u["x"][0, H] - st.u["x"][0, 0])
        kinds = []
        for n in range(int(40 / P.dt)):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.kind[0, 0] = O.WITHDRAW if n == 0 else O.KEEP
            o.x[0, 0], o.run[0, 0] = -300.0, True
            # in contact, except a gap from 10 to 15 s
            gap = 10 <= n * P.dt < 15
            st.u["x"][0, 0] = st.u["x"][0, H] - d0 - (40.0 if gap else 0.0)
            battle.step(st, o, P)
            kinds.append(int(st.u["order_kind"][0, 0]))
        assert kinds.index(O.HOLD) * P.dt == pytest.approx(P.rules["battle"]["melee_breakoff_secs"], abs=1.0)

    @pytest.mark.parametrize("on", [1, 0])
    def test_a_chased_lord_fights_again_after_the_window_too(self, on):
        # contact.breakoff_lord (the leave probe: the General leaving chasing clanrats fought again 24-25 s after the
        # order); off: the window counts for formations only and the lord's withdraw never ends
        from tests.tools.test_sim import face_off
        p = P.with_cal("contact", breakoff_lord=on)
        st = face_off(GENERAL, SLAVE)
        H = st.N // 2
        assert float(st.u["men0"][0, 0]) <= 1
        st.u["morale"][:] = 1e6
        st.u["leadership"][:] = 1e6
        d0 = float(st.u["x"][0, H] - st.u["x"][0, 0])
        hp_h0 = float(st.u["hp_abs"][0, H])
        kinds, dealt = [], []
        for n in range(int(32 / P.dt)):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.kind[0, 0] = O.WITHDRAW if n == 0 else O.KEEP
            o.x[0, 0], o.run[0, 0] = -300.0, True
            st.u["x"][0, 0] = st.u["x"][0, H] - d0              # the chasers keep up: he stays in contact
            battle.step(st, o, p)
            kinds.append(int(st.u["order_kind"][0, 0]))
            dealt.append(hp_h0 - float(st.u["hp_abs"][0, H]))
        if on:
            drop = kinds.index(O.HOLD)
            assert (drop + 1) * P.dt == pytest.approx(P.rules["battle"]["melee_breakoff_secs"] + P.dt, abs=P.dt)
            assert dealt[drop - 1] - dealt[2] == pytest.approx(0.0, abs=1e-6)      # no blows while leaving
            assert dealt[-1] > dealt[drop]                                           # he fights again
        else:
            assert O.HOLD not in kinds and dealt[-1] - dealt[2] == pytest.approx(0.0, abs=1e-6)

    def _shooter_chased(self, p, order_at, seconds=40.0, point_m=300.0, arrive_at=None):
        """Archers (slot 0) against skavenslaves attacking them, kept in contact; the archers get a move point_m straight
        back at step order_at (0: before the fight - the first step, out of melee; 4: in melee). arrive_at: the step the
        archers are put on their point (the slaves against them). Returns the order kinds, leave_s and HP the slaves
        lost, a step each."""
        from tests.tools.test_sim import face_off
        st = face_off(ARCHER, SLAVE)
        H = st.N // 2
        st.u["morale"][:] = 1e6
        st.u["leadership"][:] = 1e6
        d0 = float(st.u["x"][0, H] - st.u["x"][0, 0])
        hp_h0 = float(st.u["hp_abs"][0, H])
        kinds, leave, dealt = [], [], []
        for n in range(int(seconds / P.dt)):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            if n == order_at:
                o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, float(st.u["x"][0, 0]) - point_m, 0.0, False
            elif n > order_at:
                o.kind[0, 0] = O.KEEP
            if arrive_at is not None and n >= arrive_at:
                st.u["x"][0, 0] = st.u["ox"][0, 0]                 # at its point, the slaves against it
                st.u["x"][0, H] = st.u["x"][0, 0] + d0
            else:
                st.u["x"][0, 0] = st.u["x"][0, H] - d0              # the chasers keep up: it stays in contact
            battle.step(st, o, p)
            kinds.append(int(st.u["order_kind"][0, 0]))
            leave.append(float(st.u["leave_s"][0, 0]))
            dealt.append(hp_h0 - float(st.u["hp_abs"][0, H]))
        return kinds, leave, dealt

    def test_a_shooters_move_given_before_the_contact_is_neither_held_nor_dropped(self):
        # contact.pre_order_free (the shootcontact probe: archers and crossbowmen moving away when the clanrats caught
        # them kept moving all 75 s): no pin_s, no 24 s window; it still strikes nobody while it leaves
        assert P.sim["contact"]["pre_order_free"]
        kinds, leave, dealt = self._shooter_chased(P, order_at=0)
        assert O.HOLD not in kinds and set(kinds) == {O.MOVE}
        assert max(leave) == 0.0
        assert dealt[-1] - dealt[0] == pytest.approx(0.0, abs=1e-6)

    def test_a_shooters_move_given_in_melee_is_held_and_dropped_at_the_window(self):
        # the wavemiss2 probe: a shooter told to move in melee strikes back 25-35 s after the order
        kinds, leave, dealt = self._shooter_chased(P, order_at=4)
        assert max(leave) >= P.sim["contact"]["pin_s"] - P.dt
        drop = kinds[4:].index(O.HOLD)                 # (steps 0-3: the archers hold before the order)
        assert (drop + 1) * P.dt == pytest.approx(P.rules["battle"]["melee_breakoff_secs"] + P.dt, abs=P.dt)

    def test_off_a_move_given_before_the_contact_is_held_and_dropped_too(self):
        kinds, leave, _ = self._shooter_chased(P.with_cal("contact", pre_order_free=0), order_at=0)
        assert max(leave) >= P.sim["contact"]["pin_s"] - P.dt and O.HOLD in kinds

    def test_a_move_given_before_the_contact_ends_at_its_point_and_the_shooter_fights(self):
        # no leave_latch for it: at its point the move is over and the shooter strikes (a move given in melee stays a
        # leave there until the window drops it)
        _, _, pre = self._shooter_chased(P, order_at=0, seconds=12.0, point_m=15.0, arrive_at=10)
        assert pre[-1] - pre[10] > 0
        _, _, melee_ord = self._shooter_chased(P, order_at=4, seconds=12.0, point_m=15.0, arrive_at=10)
        assert melee_ord[-1] - melee_ord[10] == pytest.approx(0.0, abs=1e-6)

    def test_off_a_leaver_never_strikes(self):
        p = P.with_cal("contact", breakoff=0)
        from tests.tools.test_sim import face_off
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        st.u["morale"][:] = 1e6
        d0 = float(st.u["x"][0, H] - st.u["x"][0, 0])
        for n in range(int(30 / P.dt)):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.kind[0, 0] = O.WITHDRAW if n == 0 else O.KEEP
            o.x[0, 0], o.run[0, 0] = -300.0, True
            st.u["x"][0, 0] = st.u["x"][0, H] - d0
            battle.step(st, o, p)
        assert int(st.u["order_kind"][0, 0]) == O.WITHDRAW
