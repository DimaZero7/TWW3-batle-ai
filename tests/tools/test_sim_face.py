"""The facing order (tools/nn/sim/orders.py Orders.face, battle.py; config/nn/sim.json turn.face_why): a standing unit at
its point re-forms for turn.face_delay_s with its facing unchanged, then turns to the facing - the middle of one of the
game's 64 sectors (src/apps/orders/facing.lua) - at the standing rate. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
ARCHER = "wh2_dlc13_emp_inf_archers_0"
GENERAL = "wh_main_emp_cha_general_0"
TURN = P.sim["turn"]


def army(side1, side2):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def lone(key=SPEAR, b=0.0):
    """One unit at the origin (the enemy a skavenslave unit 600 m away, never ordered)."""
    return scenario.build([army([(key, 0, 0, b, key == GENERAL)], [(SLAVE, 600, 600, 0)])], P)


def order(st, kind=O.HOLD, face=-1.0, x=0.0, z=0.0, target=-1, run=False, i=0):
    o = replay.hold(st)
    o.kind[0, i], o.face[0, i], o.x[0, i], o.z[0, i], o.target[0, i], o.run[0, i] = kind, face, x, z, target, run
    return o


def keep(st):
    o = O.hold(st.B, st.N)
    o.kind[:] = O.KEEP
    return o


def off(b, goal):
    return abs((b - goal + 180) % 360 - 180)


def bearings(st, first, then, seconds, i=0):
    """[(t, bearing of slot i)] after the first order and then(st) orders each step."""
    out = []
    for k in range(int(round(seconds / P.dt))):
        battle.step(st, first if k == 0 else then(st), P)
        out.append(((k + 1) * P.dt, float(st.u["b"][0, i])))
    return out


def first_within(series, goal, deg=10.0):
    return next(t for t, b in series if off(b, goal) <= deg)


class TestSnap:
    def test_the_facing_is_the_middle_of_its_64_sector_as_facing_lua(self):
        face = torch.tensor([[0.0, 5.62, 84.38, 89.99, 90.0, 95.62, 359.9, 180.0, -1.0, -5.0]])
        want = [2.8125, 2.8125, 87.1875, 87.1875, 92.8125, 92.8125, 357.1875, 182.8125, -1.0, -1.0]
        assert O.snap_face(face)[0].tolist() == pytest.approx(want)
        assert O.FACE_SECTOR_DEG == pytest.approx(5.625)

    def test_orders_without_a_facing_face_nowhere_and_a_bad_one_is_refused(self):
        o = O.Orders(kind=torch.zeros((1, 2), dtype=torch.int64), x=torch.zeros((1, 2)), z=torch.zeros((1, 2)),
                     target=torch.full((1, 2), -1), run=torch.zeros((1, 2), dtype=torch.bool))
        assert o.face.tolist() == [[-1.0, -1.0]]
        assert O.hold(1, 2).face.tolist() == [[-1.0, -1.0]]
        O.check(o, 2)
        o.face[0, 0] = 360.0
        with pytest.raises(ValueError):
            O.check(o, 2)
        o.face[0, 0] = float("nan")
        with pytest.raises(ValueError):
            O.check(o, 2)


class TestNoFacingIsTheOldStep:
    def test_a_battle_without_facings_runs_bit_for_bit_as_without_the_rule(self):
        """The rule is guarded by the state's oface: a state without it is the old simulator; with it and no facing
        ordered every field the two share is the same, step after step (moves, attacks, a melee, shooters)."""
        def build():
            return scenario.build([army([(SPEAR, -40, 0, 90), (ARCHER, -60, 30, 90), (SPEAR, -40, -40, 90)],
                                        [(CLANRAT, 40, 0, 270), (SLAVE, 60, 40, 270)])], P)
        a, b = build(), build()
        for k in ("oface", "oface_s"):
            del b.u[k]
        H = a.N // 2
        for k in range(60):
            for st in (a, b):
                o = replay.hold(st)
                o.kind[0, 0], o.target[0, 0], o.run[0, 0] = O.ATTACK, H, True
                o.kind[0, 2], o.x[0, 2], o.z[0, 2] = O.MOVE, -20.0 + k, -60.0
                o.kind[0, H + 1], o.target[0, H + 1] = O.ATTACK, 0
                battle.step(st, o, P)
        for k, v in b.u.items():
            assert torch.equal(a.u[k], v), k
        assert torch.equal(a.t, b.t) and torch.equal(a.winner, b.winner)


class TestTurnInPlace:
    def test_a_formation_keeps_its_facing_while_it_reforms_then_turns_at_the_standing_rate(self):
        st = lone(SPEAR, b=0.0)
        series = bearings(st, order(st, face=90.0), keep, 10.0)
        goal = 92.8125                                         # the sector's middle
        delay, rate = TURN["face_delay_s"], TURN["formation_deg_s"]
        assert all(b == 0.0 for t, b in series if t < delay - 1e-6)
        start = math.ceil(delay / P.dt - 1e-6) * P.dt          # the first step whose clock reaches the delay
        assert first_within(series, goal, 0.01) == pytest.approx(start + math.ceil(goal / (rate * P.dt)) * P.dt - P.dt)
        # the probe's spearmen: within 10 deg of a ~90 deg facing at 5.1 / 6.0 s (turns_game.json)
        assert 4.0 <= first_within(series, goal) <= 6.0
        assert series[-1][1] == pytest.approx(goal) and float(st.u["oface"][0, 0]) == pytest.approx(goal)

    def test_an_about_face_takes_the_same_reform_then_twice_the_turn(self):
        st = lone(SPEAR, b=0.0)
        series = bearings(st, order(st, face=180.0), keep, 12.0)
        assert all(b == 0.0 for t, b in series if t < TURN["face_delay_s"] - 1e-6)
        assert 4.5 <= first_within(series, 182.8125) <= 6.5           # the probe: 4.8 s
        assert series[-1][1] == pytest.approx(182.8125)

    def test_a_lone_lord_turns_sooner_at_his_own_rate(self):
        st = lone(GENERAL, b=0.0)
        rate = float(st.u["turn"][0, 0])
        series = bearings(st, order(st, face=180.0), keep, 6.0)
        assert all(b == 0.0 for t, b in series if t < TURN["face_delay_single_s"] - 1e-6)
        start = math.ceil(TURN["face_delay_single_s"] / P.dt - 1e-6) * P.dt
        assert first_within(series, 182.8125, 0.01) == pytest.approx(
            start + math.ceil(off(0.0, 182.8125) / (rate * P.dt)) * P.dt - P.dt)     # the short way round

    def test_the_same_facing_reissued_every_decision_does_not_restart_the_reform(self):
        a, b = lone(SPEAR), lone(SPEAR)
        kept = bearings(a, order(a, face=90.0), keep, 8.0)
        again = bearings(b, order(b, face=90.0), lambda st: order(st, face=91.0), 8.0)   # 91: the same sector
        assert kept == again

    def test_keep_holds_the_facing_and_a_new_order_without_one_drops_it(self):
        st = lone(SPEAR)
        bearings(st, order(st, face=90.0), keep, 2.0)
        assert float(st.u["oface"][0, 0]) == pytest.approx(92.8125)
        series = bearings(st, order(st, kind=O.HOLD), keep, 8.0)      # a new HOLD, no facing
        assert float(st.u["oface"][0, 0]) == -1.0
        assert all(b == 0.0 for _, b in series)

    def test_another_facing_restarts_the_reform(self):
        st = lone(SPEAR)
        bearings(st, order(st, face=90.0), keep, 3.0)                  # 3 s of the re-form to 90
        series = bearings(st, order(st, face=270.0), keep, 3.0)
        assert all(b == 0.0 for _, b in series)                        # a new clock: still re-forming


class TestAfterAMove:
    def test_a_move_faces_its_way_then_takes_the_facing_at_the_point(self):
        st = lone(SPEAR, b=0.0)
        series = bearings(st, order(st, kind=O.MOVE, x=0.0, z=40.0, face=270.0), keep, 40.0)
        moving = [b for t, b in series if t <= 5.0]
        assert all(off(b, 0.0) < 1.0 for b in moving)                 # it walks the way it faces
        assert math.hypot(float(st.u["x"][0, 0]), float(st.u["z"][0, 0]) - 40.0) <= 1.0
        assert series[-1][1] == pytest.approx(272.8125)               # 270 -> the sector's middle


class TestWhereTheFacingDoesNothing:
    def test_in_melee_the_facing_waits(self):
        def fight():
            st = scenario.build([army([(SPEAR, -50, 0, 90)], [(CLANRAT, 50, 0, 270)])], P)
            front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
            H = st.N // 2
            touch = min(0.0, P.sim["contact"]["reach_m"])
            st.u["x"][0, 0] = -(depth[0, 0] / 2 + touch / 2)
            st.u["x"][0, H] = depth[0, H] / 2 + touch / 2
            return st, H
        (a, H), (b, _) = fight(), fight()

        def attack(st, face):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.face[0, 0] = face
            return o
        # (in melee the facing waits turn.face_melee_delay_s, test_sim_face_melee.py; until then nothing changes the
        # bearing of the unit told to face, the control's drifts at contact.melee_turn_deg_s at most)
        delay = P.sim["turn"]["face_melee_delay_s"]
        b0 = float(b.u["b"][0, 0])
        for k in range(int(delay / P.dt) - 1):
            battle.step(a, attack(a, -1.0), P)
            battle.step(b, attack(b, 0.0), P)                 # spearmen in melee told to face north
            assert bool(b.u["m"][0, 0])
            assert float(b.u["b"][0, 0]) == b0
            assert abs(float(a.u["b"][0, 0]) - b0) <= P.sim["contact"]["melee_turn_deg_s"] * P.dt * (k + 1) + 1e-4

    def test_an_attack_ignores_the_facing(self):
        a, b = lone(SPEAR), lone(SPEAR)
        H = a.N // 2
        sa = bearings(a, order(a, kind=O.ATTACK, target=H), keep, 6.0)
        sb = bearings(b, order(b, kind=O.ATTACK, target=H, face=200.0), keep, 6.0)
        assert sa == sb

    def test_a_shooter_with_a_target_in_range_keeps_facing_it(self):
        def build():
            return scenario.build([army([(ARCHER, 0, 0, 0)], [(SLAVE, 0, 120, 180)])], P)
        a, b = build(), build()
        sa = bearings(a, order(a), keep, 10.0)
        sb = bearings(b, order(b, face=180.0), keep, 10.0)    # told to face away from the slaves it shoots
        assert sa == sb
        assert float(b.u["a"][0, 0]) < float(b.u["ammo0"][0, 0])     # it shot
