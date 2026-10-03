"""tools/nn/train/drills/kiting.py: the frame's battles are a kiting situation (our missile units
faster than every chaser, the gap in range, we attack), the enemy chases our missile units, the
skilled script runs back from a near chaser and stays on the map, and the scripts run in the
simulator."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import drills as D
from tools.nn.train.drills import kiting as K

SEEDS = range(64)
UNITS = json.loads((Path(__file__).resolve().parents[2] / "config/nn/units.json").read_text(encoding="utf-8"))["units"]


def _centre(units):
    return sum(u["x"] for u in units) / len(units), sum(u["z"] for u in units) / len(units)


class TestFrame:
    def test_our_missile_units_outrun_every_chaser_and_the_gap_is_in_range(self):
        for desc, ours in D.battles(K.DRILL, SEEDS):
            mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
            assert desc["attacker"] == ours
            assert 1 <= len(mine) <= 2 and 1 <= len(theirs) <= 3 and len(theirs) >= len(mine)
            ours_run = [UNITS[u["key"]]["speed"]["run"] for u in mine]
            assert all((UNITS[u["key"]].get("missile") or {}).get("ammo", 0) > 0 for u in mine)
            for e in theirs:
                p = UNITS[e["key"]]
                assert p["missile"] is None and "unbreakable" not in p["attributes"]
                assert p["speed"]["run"] <= min(ours_run) - K.MIN_GAP_MS
            (ax, az), (bx, bz) = _centre(mine), _centre(theirs)
            assert K.GAP_M[0] - 1 <= math.hypot(ax - bx, az - bz) <= K.GAP_M[1] + 1
            assert all(abs(u["x"]) <= D.MAP_HALF_M and abs(u["z"]) <= D.MAP_HALF_M for u in mine + theirs)

    def test_both_sides_are_played_and_the_rosters_vary(self):
        pairs = D.battles(K.DRILL, SEEDS)
        assert {o for _, o in pairs} == {1, 2}
        sizes = {(len(d["sides"][o]["units"]), len(d["sides"][3 - o]["units"])) for d, o in pairs}
        assert {(1, 1), (2, 2), (2, 3)} <= sizes


def _state(pairs):
    descs = [p[0] for p in pairs]
    st = scenario.build(descs)
    return st, torch.tensor([p[1] for p in pairs])


class TestScripts:
    def test_the_enemy_chases_our_nearest_missile_unit(self):
        pairs = D.battles(K.DRILL, range(8))
        st, ours = _state(pairs)
        o = K.enemy(st)
        u = st.u
        for b, (_, side) in enumerate(pairs):
            for i in range(st.N):
                if int(u["side"][b, i]) != 3 - side:
                    continue
                assert int(o.kind[b, i]) == O.ATTACK and bool(o.run[b, i])
                t = int(o.target[b, i])
                assert int(u["side"][b, t]) == side and float(u["range"][b, t]) > 0

    def test_the_skilled_script_runs_back_from_a_near_chaser(self):
        pairs = D.battles(K.DRILL, range(8), both_sides=False)
        st, _ = _state(pairs)
        u = st.u
        # move every chaser of battle 0 to 30 m in front of our first unit
        i0 = 0
        H = st.N // 2
        u["x"][0, i0], u["z"][0, i0] = 0.0, 0.0                       # at the map's centre
        for j in range(H, st.N):
            if int(u["side"][0, j]) == 2:
                u["x"][0, j] = u["x"][0, i0] + 30.0
                u["z"][0, j] = u["z"][0, i0]
        o = K.skilled(st)
        assert int(o.kind[0, i0]) == O.MOVE and bool(o.run[0, i0])
        assert float(o.x[0, i0]) < float(u["x"][0, i0]) - 10          # away from the chasers
        # a battle whose chasers are far: hold and shoot
        far = [b for b in range(1, st.B)]
        assert all(int(o.kind[b, 0]) == O.HOLD for b in far)

    def test_the_retreat_bends_inward_at_the_map_edge(self):
        pairs = D.battles(K.DRILL, range(1), both_sides=False)
        st, _ = _state(pairs)
        u = st.u
        H = st.N // 2
        u["x"][0, 0], u["z"][0, 0] = 790.0, 0.0                        # ours near the east edge
        for j in range(H, st.N):
            if int(u["side"][0, j]) == 2:
                u["x"][0, j], u["z"][0, j] = 760.0, 0.0                # chasers between it and the centre
        o = K.skilled(st)
        assert int(o.kind[0, 0]) == O.MOVE
        assert float(o.x[0, 0]) < 790.0 + 1.0 and abs(float(o.z[0, 0])) > 20.0

    def test_scripts_run_a_few_battles(self):
        pairs = D.battles(K.DRILL, range(4))
        descs = [p[0] for p in pairs]
        params = load()
        for script in (K.naive, K.skilled):
            st, ours = _state(pairs)
            for _ in range(40):
                orders = D.merged(st, ours, script(st), K.enemy(st))
                O.check(orders, st.N)
                battle.step(st, orders, params, params.dt)
            assert bool(torch.isfinite(st.u["x"]).all())

    def test_the_enemy_script_works_on_an_ordinary_battle(self):
        st = scenario.build([{"attacker": 1, "sides": {
            1: {"faction": K.EMPIRE, "units": [D.unit("wh_main_emp_cha_general_0", 0, -100, 0, general=True),
                                                D.unit("wh2_dlc13_emp_inf_archers_0", 30, -100, 0, 40)]},
            2: {"faction": K.SKAVEN, "units": [D.unit("wh2_main_skv_cha_warlord_0", 0, 100, 180, general=True),
                                                D.unit("wh2_main_skv_inf_clanrats_1", 30, 100, 180, 30)]}}}],
            per_side=20)
        assert st.N == 40
        o = K.enemy(st)
        O.check(o, st.N)
        assert int(o.target[0, 20]) == 1 and int(o.target[0, 0]) == 20   # both sides chase a shooter / the nearest
        assert int(o.kind[0, 5]) == O.HOLD                                # an empty slot holds


class TestVolley:
    def test_the_first_shot_after_halting_is_a_volley_then_the_steady_rate(self):
        """tools/nn/sim/missile.py: the men reload all the time, every loaded man shoots when the unit
        can (the game: ~80 of 90 archers shoot within 1 s of the first shot, build/archer-range)."""
        st = scenario.build([{"attacker": 1, "sides": {
            1: {"faction": K.SKAVEN, "units": [D.unit(K.OURS[0], 0, 0, 0, K.OURS[2])]},
            2: {"faction": K.EMPIRE, "units": [D.unit(K.SWORDS[0], 0, 120, 180, K.WIDTH_M)]}}}])
        params = load()
        dt = params.dt
        a0 = float(st.u["a"][0, 0])
        men = float(st.u["men"][0, 0])
        fired = []
        for _ in range(int(20 / dt)):
            a = float(st.u["a"][0, 0])
            battle.step(st, O.hold(st.B, st.N), params, dt)
            fired.append(a - float(st.u["a"][0, 0]))
        first = next(i for i, f in enumerate(fired) if f > 0)
        assert fired[first] == pytest.approx(men, rel=1e-3)               # the whole unit at once
        steady = fired[first + 1:]
        rate = sum(steady) / (len(steady) * dt)
        assert rate == pytest.approx(men / float(st.u["reload"][0, 0]), rel=0.05)
        assert a0 - float(st.u["a"][0, 0]) == pytest.approx(sum(fired))
