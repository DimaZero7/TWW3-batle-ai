"""tools/nn/train/drills/hold_fire.py: the frame's battles are a "fire into the melee" situation (our
infantry in contact with the enemy lord, our arcing shooters behind it, free enemy missile
units nearer to our shooters than to the melee, we attack), the scripts pick the right targets, and they
run in the simulator."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.sim.replay import half_depth
from tools.nn.train import drills as D
from tools.nn.train.drills import hold_fire as H

SEEDS = range(64)


def _d(a, b):
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


class TestFrame:
    def test_the_melee_shooters_and_free_enemies_are_placed_as_the_frame_says(self):
        for desc, ours in D.battles(H.DRILL, SEEDS, broad=0.0):
            mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
            assert desc["attacker"] == ours
            fo, fe = desc["sides"][ours]["faction"], desc["sides"][3 - ours]["faction"]
            m, shooters = mine[0], mine[1:]
            e, free = theirs[0], theirs[1:]
            assert m["key"] in H.OURS_MELEE[fo]
            assert e["key"] == H.LORD[fe] and e["general"]
            combo = [c for c in H.SHOOT if c[1] == shooters[0]["key"]]
            assert combo and combo[0][0] == fo and all(s["key"] == combo[0][1] for s in shooters)
            assert H.K_SHOOT[0] <= len(shooters) <= H.K_SHOOT[1] and combo[0][2][0] <= len(free) <= combo[0][2][1]
            assert all(f["key"] in H.FREE[fe] for f in free)
            # in contact: the formations' depths apart (+0.5 m)
            he = 1.0 if e["general"] else float(half_depth(H.MEN[e["key"]], 30.0))
            assert abs(_d(m, e) - (float(half_depth(H.MEN[m["key"]], 30.0)) + he + 0.5)) < 1e-3
            for s in shooters:
                assert H.DIST_M[0] - 1 <= _d(s, m) <= math.hypot(H.DIST_M[1], 60.0) + 1
            # a free enemy: nearer to our nearest shooter than to the melee, farther from it than the melee
            for f in free:
                near = min(_d(f, s) for s in shooters)
                assert near < min(_d(f, m), _d(f, e))
                assert near > min(_d(s, e) for s in shooters) - 30
            assert all(abs(u["x"]) <= D.MAP_HALF_M and abs(u["z"]) <= D.MAP_HALF_M for u in mine + theirs)

    def test_both_sides_and_both_factions_occur(self):
        pairs = D.battles(H.DRILL, SEEDS, broad=0.0)
        assert {o for _, o in pairs} == {1, 2}
        assert {d["engaged"] for d, _ in pairs} == {"lord"}
        assert {d["sides"][o]["faction"] for d, o in pairs} == {H.EMPIRE, H.SKAVEN}
        assert {len(d["sides"][3 - o]["units"]) for d, o in pairs} == {2, 3}


def _state(pairs):
    descs = [p[0] for p in pairs]
    st = scenario.build(descs)
    return st, torch.tensor([p[1] for p in pairs])


def _settle(st, pairs, steps=6):
    """A few steps with every unit holding but our melee unit and the engaged enemy attacking each other:
    the pair touches (u["m"])."""
    params = load()
    for _ in range(steps):
        o = O.hold(st.B, st.N)
        for b, (desc, ours) in enumerate(pairs):
            Hs = st.N // 2
            mo, me = (0, Hs) if ours == 1 else (Hs, 0)
            o.kind[b, mo], o.target[b, mo] = O.ATTACK, me
            o.kind[b, me], o.target[b, me] = O.ATTACK, mo
        battle.step(st, o, params, params.dt)


class TestScripts:
    def test_naive_shoots_into_the_melee_skilled_at_a_free_enemy(self):
        pairs = D.battles(H.DRILL, range(8), broad=0.0)
        st, _ = _state(pairs)
        _settle(st, pairs)
        u = st.u
        assert bool(u["m"][:, 0].all()) and bool(u["m"][:, st.N // 2].all())
        nv, sk = H.naive(st), H.skilled(st)
        for b, (_, ours) in enumerate(pairs):
            for i in range(st.N):
                if int(u["side"][b, i]) != ours or float(u["range"][b, i]) <= 0 or float(u["men"][b, i]) <= 0:
                    continue
                assert int(nv.kind[b, i]) == O.ATTACK and bool(u["m"][b, int(nv.target[b, i])])
                assert int(sk.kind[b, i]) == O.ATTACK
                t = int(sk.target[b, i])
                assert int(u["side"][b, t]) == 3 - ours and not bool(u["m"][b, t]) and float(u["range"][b, t]) > 0

    def test_skilled_with_no_free_enemy_steps_out_of_range_of_the_melee(self):
        pairs = D.battles(H.DRILL, range(4), both_sides=False, broad=0.0)
        st, _ = _state(pairs)
        _settle(st, pairs)
        u = st.u
        Hs = st.N // 2
        u["men"][:, Hs + 1:] = 0.0                        # the free enemies are gone
        o = H.skilled(st)
        for b in range(st.B):
            for i in range(1, Hs):
                if float(u["men"][b, i]) <= 0:
                    continue
                assert int(o.kind[b, i]) == O.MOVE
                dx, dz = float(o.x[b, i] - u["x"][b, Hs]), float(o.z[b, i] - u["z"][b, Hs])
                assert math.hypot(dx, dz) > float(u["range"][b, i]) + H.HOLD_OUT_M
        assert bool((o.kind[:, 0] == O.ATTACK).all())  # our infantry fights on

    def test_scripts_run_a_few_battles(self):
        pairs = D.battles(H.DRILL, range(4), broad=0.0)
        params = load()
        for script in (H.naive, H.skilled):
            st, ours = _state(pairs)
            for _ in range(40):
                orders = D.merged(st, ours, script(st), H.enemy(st))
                O.check(orders, st.N)
                battle.step(st, orders, params, params.dt)
            assert bool(torch.isfinite(st.u["x"]).all())

    def test_the_enemy_script_works_on_an_ordinary_battle(self):
        st = scenario.build([{"attacker": 1, "sides": {
            1: {"faction": H.EMPIRE, "units": [D.unit("wh_main_emp_cha_general_0", 0, -100, 0, general=True),
                                                D.unit("wh2_dlc13_emp_inf_archers_0", 30, -100, 0, 40)]},
            2: {"faction": H.SKAVEN, "units": [D.unit("wh2_main_skv_cha_warlord_0", 0, 100, 180, general=True),
                                                D.unit("wh2_main_skv_inf_skavenslave_slingers_0", 30, 100, 180, 35)]}}}],
            per_side=20)
        o = H.enemy(st)
        O.check(o, st.N)
        assert int(o.kind[0, 0]) == O.ATTACK and int(o.kind[0, 20]) == O.ATTACK   # the lords charge
        assert int(o.kind[0, 1]) == O.HOLD and int(o.kind[0, 21]) == O.HOLD       # shooters hold
