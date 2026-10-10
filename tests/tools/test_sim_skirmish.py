"""The game's skirmish mode in the simulator (tools/nn/sim/skirmish.py, config/nn/sim.json skirmish): a shooter with the
mode on runs straight away from an enemy attacking it that comes within its start gap, over its order and without shooting, until
the enemy attacking it is at its stop gap; who has the mode on (the game's AI and the scripts on, a network off). Needs
torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, geometry, scenario  # noqa: E402
from tools.nn.sim import skirmish as SK  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
ARCHER, CLANRAT = "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_clanrats_0"
SWORD, STARS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_night_runners_0"
SLINGERS = "wh2_main_skv_inf_skavenslave_slingers_0"
SLING_RUNNERS = "wh2_main_skv_inf_night_runners_1"         # Night Runners with slings: run 5.4 m/s


def pair(shooter, enemy, gap, skirmish=True, shooter_side=1, params=P):
    """The shooter facing east at x 0, the enemy east of it facing west, the formations' edges `gap` m apart; the
    shooter's side sets the mode (skirmish), the enemy's side holds without the mode."""
    rows = {shooter_side: [{"key": shooter, "x": 0.0, "z": 0.0, "b": 90.0, "name": "sh"}],
            3 - shooter_side: [{"key": enemy, "x": 200.0, "z": 0.0, "b": 270.0, "name": "en"}]}
    fac = {ARCHER: "wh_main_emp_empire", SWORD: "wh_main_emp_empire"}
    sides = {s: {"faction": fac.get(rows[s][0]["key"], "wh2_main_skv_skaven"), "units": rows[s],
                 "skirmish": skirmish if s == shooter_side else False} for s in (1, 2)}
    st = scenario.build([{"attacker": 3 - shooter_side, "sides": sides}], params)
    H = st.N // 2
    i, j = (0, H) if shooter_side == 1 else (H, 0)
    sg = SK.soldier_gap(st.u, params.sim["formation"]["spacing_m"])
    st.u["x"][0, j] = st.u["x"][0, j] + (gap - float(sg[0, i, j]))
    st.u["morale"][:] = 1e6
    st.u["leadership"][:] = 1e6
    return st, i, j


def edge_gap(st, i, j):
    """The skirmish rule's distance: nearest man to nearest man (skirmish.soldier_gap)."""
    return float(SK.soldier_gap(st.u, P.sim["formation"]["spacing_m"])[0, i, j])


def hold(st):
    return O.hold(st.B, st.N)


def chase(st, i, j, run=True):
    """The shooter i holds; the enemy j attacks it."""
    o = hold(st)
    o.kind[0, j], o.target[0, j], o.run[0, j] = O.ATTACK, i, run
    return o


class TestWho:
    def test_the_gaps_come_from_sim_json_by_unit_and_only_for_units_that_can_skirmish(self):
        cal = P.sim["skirmish"]
        assert P.static(ARCHER)["sk_m"] == pytest.approx(cal["start_gap_m"]["default"])
        assert P.static(SLINGERS)["sk_m"] == pytest.approx(cal["start_gap_m"][SLINGERS])
        assert P.static(ARCHER)["sk_stop_m"] == pytest.approx(cal["stop_gap_m"]["default"])
        assert P.static(CLANRAT)["sk_m"] == 0.0                      # no can_skirmish (and no missile)
        off = P.with_cal("skirmish", on=0)
        assert off.static(ARCHER)["sk_m"] == 0.0

    def test_on_where_the_game_ai_plays_unless_the_side_says(self):
        def build(spec2):
            a = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": [{"key": ARCHER, "x": 0, "z": 0}]},
                                          2: {"faction": "wh2_main_skv_skaven", "units": [{"key": SLINGERS, "x": 300, "z": 0}],
                                              **spec2}}}
            return scenario.build([a], P)
        st = build({})
        H = st.N // 2
        assert not bool(st.u["sk"][0, 0]) and bool(st.u["sk"][0, H])     # side 2: the game's AI by default
        st = build({"skirmish": False})
        assert not bool(st.u["sk"][0, H])

    def test_set_rule_switches_it_with_the_ai_rule(self):
        st, i, j = pair(ARCHER, CLANRAT, 100.0, skirmish=False)
        abilities.set_rule(st.u, torch.tensor([[True, False]]))           # a script plays side 1
        assert bool(st.u["sk"][0, i])
        abilities.set_rule(st.u, torch.tensor([[False, True]]))           # a network plays side 1
        assert not bool(st.u["sk"][0, i])
        assert not bool(st.u["sk"][0, j])                                  # clanrats: never


class TestRun:
    def test_an_enemy_attacking_it_within_the_start_gap_sends_a_held_shooter_straight_away_without_shooting(self):
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        a0 = float(st.u["a"][0, i])
        x0 = float(st.u["x"][0, i])
        st.u["vx"][0, j] = -float(st.u["run"][0, j])                   # the clanrats already running at it
        for _ in range(int(12 / P.dt)):
            battle.step(st, chase(st, i, j), P)
            assert bool(st.u["skr"][0, i])
            assert not bool(st.u["fire"][0, i])
        assert float(st.u["a"][0, i]) == a0                               # no shot on the run
        assert int(st.u["order_kind"][0, i]) == O.HOLD                     # over the order: it stays HOLD
        assert float(st.u["x"][0, i]) < x0 - 20                            # straight away (west) ...
        assert abs(float(st.u["z"][0, i])) < 1.0
        assert math.hypot(float(st.u["vx"][0, i]), float(st.u["vz"][0, i])) == pytest.approx(float(st.u["run"][0, i]),
                                                                                            rel=0.02)  # ... at a run

    def test_it_stops_at_the_stop_gap_turns_back_and_shoots_again(self):
        st, i, j = pair(SLING_RUNNERS, SWORD, 30.0, shooter_side=2)
        stop = float(st.u["sk_stop_m"][0, i])
        st.u["vx"][0, j] = -float(st.u["walk"][0, j])                  # the swordsmen already walking at it
        for _ in range(int(60 / P.dt)):
            battle.step(st, chase(st, i, j, run=False), P)
            if not bool(st.u["skr"][0, i]):
                break
        assert not bool(st.u["skr"][0, i])
        assert edge_gap(st, i, j) == pytest.approx(stop, abs=3.0)
        a0 = float(st.u["a"][0, i])
        for _ in range(int(15 / P.dt)):
            battle.step(st, chase(st, i, j, run=False), P)
        assert float(st.u["a"][0, i]) < a0                                  # it shoots again from where it stopped
        assert not bool(st.u["skr"][0, i])                                  # (the walker beyond the start gap)

    def test_only_an_enemy_moving_at_it_starts_a_run(self):
        # skirmish.cone_deg: a standing enemy within the gap, or one running across (90 deg off the line), starts none
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        x0 = float(st.u["x"][0, i])
        for _ in range(int(3 / P.dt)):
            battle.step(st, hold(st), P)                                    # the clanrats hold
        assert not bool(st.u["skr"][0, i]) and float(st.u["x"][0, i]) == pytest.approx(x0, abs=0.5)
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        o = hold(st)
        o.kind[0, j], o.x[0, j], o.z[0, j], o.run[0, j] = O.MOVE, float(st.u["x"][0, j]), 400.0, True   # across
        for _ in range(int(4 / P.dt)):
            battle.step(st, o, P)
            o = hold(st)
            o.kind[0, j] = O.KEEP
        assert not bool(st.u["skr"][0, i])

    def test_a_run_from_an_enemy_that_attacks_another_ends_by_the_hazard_and_does_not_start_again_inside(self):
        # skirmish.other_end_per_s: the clanrats move straight at it (a move order through it, not an attack)
        def run(p_end, steps):
            p = P.with_cal("skirmish", other_end_per_s=p_end)
            st, i, j = pair(ARCHER, CLANRAT, 35.0, params=p)
            st.u["vx"][0, j] = -float(st.u["run"][0, j])
            o = hold(st)
            o.kind[0, j], o.x[0, j], o.z[0, j], o.run[0, j] = O.MOVE, -400.0, 0.0, True
            seen = []
            for _ in range(steps):
                battle.step(st, o, p)
                o = hold(st)
                o.kind[0, j] = O.KEEP
                seen.append(bool(st.u["skr"][0, i]))
            return seen
        assert all(run(0.0, 8))                                         # no hazard: it runs on
        seen = run(1.0, 8)                                              # a sure end each second
        assert seen[0] and not any(seen[1:])                            # (no new start while the enemy stays inside)

    def test_the_far_point_is_shown_and_the_order_point_comes_back(self):
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        st.u["vx"][0, j] = -float(st.u["run"][0, j])
        battle.step(st, chase(st, i, j), P)
        assert bool(st.u["skr"][0, i])
        assert math.hypot(float(st.u["ox"][0, i] - st.u["x"][0, i]), float(st.u["oz"][0, i] - st.u["z"][0, i])) > 1000
        p = P.with_cal("skirmish", show_far_point=0)
        st, i, j = pair(ARCHER, CLANRAT, 35.0, params=p)
        st.u["vx"][0, j] = -float(st.u["run"][0, j])
        battle.step(st, chase(st, i, j), p)
        assert math.hypot(float(st.u["ox"][0, i] - st.u["x"][0, i]), float(st.u["oz"][0, i] - st.u["z"][0, i])) < 1.0

    def test_the_soldier_gap_is_nearest_man_to_nearest_man(self):
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        h, v = float(st.u["sp_v"][0, i]), float(st.u["sp_v"][0, j])
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        # facing each other: the rectangles' edge gap plus half a rank of each (the men stand inside the rectangle)
        assert edge_gap(st, i, j) == pytest.approx(float(pw["gap"][0, i, j]) + h / 2 + v / 2, abs=0.05)

    def test_the_run_ends_when_the_chaser_lets_it_go(self):
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        for _ in range(int(4 / P.dt)):
            battle.step(st, chase(st, i, j), P)
        assert bool(st.u["skr"][0, i])
        o = hold(st)
        o.kind[0, j], o.x[0, j], o.z[0, j], o.run[0, j] = O.MOVE, 300.0, 300.0, True   # the clanrats go elsewhere
        for _ in range(int(3 / P.dt)):                                   # (once they have turned away)
            battle.step(st, o, P)
            o = hold(st)
            o.kind[0, j] = O.KEEP
        assert not bool(st.u["skr"][0, i])

    def test_beyond_the_start_gap_and_with_the_mode_off_it_holds_and_shoots(self):
        for gap, sk in ((70.0, True), (35.0, False)):
            st, i, j = pair(ARCHER, CLANRAT, gap, skirmish=sk)
            a0, x0 = float(st.u["a"][0, i]), float(st.u["x"][0, i])
            for _ in range(int(5 / P.dt)):
                battle.step(st, chase(st, i, j, run=False), P)
            assert not bool(st.u["skr"][0, i])
            assert float(st.u["x"][0, i]) == pytest.approx(x0, abs=0.5)
            assert float(st.u["a"][0, i]) < a0

    def test_off_in_sim_json_nobody_runs(self):
        p = P.with_cal("skirmish", on=0)
        st, i, j = pair(ARCHER, CLANRAT, 35.0, params=p)
        x0 = float(st.u["x"][0, i])
        for _ in range(int(5 / P.dt)):
            battle.step(st, chase(st, i, j), p)
        assert float(st.u["x"][0, i]) == pytest.approx(x0, abs=0.5)

    def test_it_runs_over_a_move_order_towards_the_enemy(self):
        st, i, j = pair(ARCHER, CLANRAT, 35.0)
        x0 = float(st.u["x"][0, i])
        for _ in range(int(6 / P.dt)):
            o = chase(st, i, j)
            o.kind[0, i], o.x[0, i], o.z[0, i], o.run[0, i] = O.MOVE, 300.0, 0.0, True
            battle.step(st, o, P)
        assert float(st.u["x"][0, i]) < x0 - 5
        assert int(st.u["order_kind"][0, i]) == O.MOVE

    def test_a_new_order_ends_the_run_and_the_unit_carries_it_out(self):
        # skirmish.order_cuts: another order (here a move 20 m aside) ends a run under way; with the chaser beyond the
        # start gap no new run starts and the unit walks to its point; order_cuts 0: the run goes on over it
        for cuts in (1, 0):
            p = P.with_cal("skirmish", order_cuts=cuts)
            st, i, j = pair(SLING_RUNNERS, SWORD, 30.0, shooter_side=2, params=p)
            st.u["vx"][0, j] = -float(st.u["walk"][0, j])              # the swordsmen walking at it
            for _ in range(int(30 / P.dt)):
                battle.step(st, chase(st, i, j, run=False), p)
                if edge_gap(st, i, j) > 66.0:          # (the turn to the new point shortens the gap ~20 m)
                    break
            assert bool(st.u["skr"][0, i])
            z0 = float(st.u["z"][0, i])
            o = chase(st, i, j, run=False)
            o.kind[0, i], o.x[0, i], o.z[0, i], o.run[0, i] = O.MOVE, float(st.u["x"][0, i]), z0 + 20.0, True
            battle.step(st, o, p)
            assert bool(st.u["skr"][0, i]) is (cuts == 0)
            for _ in range(int(4 / P.dt)):
                o = chase(st, i, j, run=False)
                o.kind[0, i] = O.KEEP
                battle.step(st, o, p)
                assert bool(st.u["skr"][0, i]) is (cuts == 0)
            if cuts:
                assert float(st.u["z"][0, i]) > z0 + 10.0                  # on its way to the ordered point
                assert edge_gap(st, i, j) > float(st.u["sk_m"][0, i])

    def test_a_shooter_that_fires_on_the_move_keeps_shooting_on_the_run(self):
        # the Night Runners' throwing stars (fire_move, fire arc 360): they shoot behind on the run, as on any move
        st, i, j = pair(STARS, SWORD, 25.0, shooter_side=2)
        assert bool(st.u["fire_move"][0, i])
        a0 = float(st.u["a"][0, i])
        for _ in range(int(8 / P.dt)):
            battle.step(st, chase(st, i, j), P)
        assert bool(st.u["skr"][0, i])
        assert float(st.u["a"][0, i]) < a0

    def test_no_new_run_in_melee_but_a_caught_runner_runs_on(self):
        for melee_run, runs in ((1, True), (0, False)):
            p = P.with_cal("skirmish", melee_run=melee_run)
            st, i, j = pair(ARCHER, CLANRAT, -3.0, params=p)         # overlapping: in contact (contact.reach_m -2.5)
            st.u["vx"][0, j] = -3.0                                  # the clanrats running at it (west)
            battle.step(st, chase(st, i, j), p)
            assert bool(st.u["m"][0, i]) and not bool(st.u["skr"][0, i])    # caught standing: it fights
            st, i, j = pair(ARCHER, CLANRAT, -3.0, params=p)
            st.u["skr"][0, i] = True                                 # caught on the run
            st.u["vx"][0, j] = -3.0
            hp_j = float(st.u["hp_abs"][0, j])
            battle.step(st, chase(st, i, j), p)
            assert bool(st.u["skr"][0, i]) is runs
            if runs:                                                 # a leave: it strikes nobody
                assert float(st.u["hp_abs"][0, j]) == pytest.approx(hp_j)


class TestReplay:
    def test_the_engines_far_point_keeps_the_order_before_it(self):
        # replay.engine_far_rows: the seconds the game records its skirmish run (a point >= 1000 m away, not routing)
        # stand for the last second before them; a router's far point and the first seconds stay their own
        import types
        import numpy as np
        from tools.nn.sim import replay
        ox = np.array([[5000.0], [10.0], [3000.0], [3000.0], [12.0], [4000.0]])
        f = {"x": np.zeros((6, 1)), "z": np.zeros((6, 1)), "ox": ox, "oz": np.zeros((6, 1)),
             "r": np.array([[False], [False], [False], [False], [False], [True]])}
        b = types.SimpleNamespace(f=f)
        assert replay.engine_far_rows(b, 0, 1000.0).tolist() == [-1, 1, 1, 1, 4, 5]
