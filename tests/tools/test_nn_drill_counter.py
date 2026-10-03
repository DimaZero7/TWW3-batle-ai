"""tools/nn/train/drills/counter.py: the frame's battles put our unit X opposite its counter C (its
nearest enemy) and our Z, which beats C, opposite Y; the passport matchup agrees; the scripts give
orders and run in the simulator."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import drills as D
from tools.nn.train.drills import counter as C

SEEDS = range(64)


def _nearest(u, others):
    return min(others, key=lambda e: math.hypot(e["x"] - u["x"], e["z"] - u["z"]))


class TestFrame:
    def test_each_of_our_units_faces_a_pairing_of_the_combos(self):
        combos = set(C.COMBOS)
        for desc, ours in D.battles(C.DRILL, SEEDS, broad=0.0):
            mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
            assert len(mine) == len(theirs) == 2 and desc["attacker"] == ours
            near = [_nearest(m, theirs) for m in mine]
            assert near[0] is not near[1]
            got = {(mine[i]["key"], mine[1 - i]["key"], near[i]["key"], near[1 - i]["key"]) for i in (0, 1)}
            assert got & combos

    def test_the_matchup_says_x_loses_to_its_counter_and_z_beats_it(self):
        descs = [d for d, _ in D.battles(C.DRILL, SEEDS, both_sides=False, broad=0.0)]
        st = scenario.build(descs)
        adv = C.rates(st.u, C.CHARGE_W) / C.rates(st.u).transpose(1, 2).clamp(min=1e-9)
        H = st.N // 2
        for b, desc in enumerate(descs):
            mine, theirs = desc["sides"][1]["units"], desc["sides"][2]["units"]
            for x in (0, 1):
                z = 1 - x
                c = theirs.index(_nearest(mine[x], theirs))
                if (mine[x]["key"], mine[z]["key"], theirs[c]["key"], theirs[1 - c]["key"]) in C.COMBOS:
                    assert float(adv[b, x, H + c]) < 1.0              # X loses to its nearest
                    assert float(adv[b, z, H + c]) >= C.BEATS         # Z beats it


class TestScripts:
    def test_skilled_takes_the_matchups_choice(self):
        pairs = D.battles(C.DRILL, range(16), broad=0.0)
        descs = [p[0] for p in pairs]
        st = scenario.build(descs)
        o = C.skilled(st)
        correct, bad = C.roles(st)
        for b, (desc, ours) in enumerate(pairs):
            mine = [i for i in range(st.N) if int(st.u["side"][b, i]) == ours]
            assert all(int(o.kind[b, i]) == O.ATTACK for i in mine)
            # each takes its matchup's choice; the unit that beats something never takes its nearest
            # enemy when that one beats it
            assert all(bool(correct[b, i, int(o.target[b, i])]) for i in mine)
            theirs = desc["sides"][3 - ours]["units"]
            for k, i in enumerate(mine):
                c = theirs.index(_nearest(desc["sides"][ours]["units"][k], theirs))
                c_slot = (2 - ours) * (st.N // 2) + c
                if bool(bad[b, i, c_slot]) and bool((correct[b, i] & ~bad[b, i]).any()):
                    assert int(o.target[b, i]) != c_slot

    def test_the_scripts_run_in_the_simulator(self):
        pairs = D.battles(C.DRILL, range(4), broad=0.0)
        descs = [p[0] for p in pairs]
        ours = torch.tensor([p[1] for p in pairs])
        params = load()
        for script in (C.naive, C.skilled):
            st = scenario.build(descs)
            for _ in range(20):
                battle.step(st, D.merged(st, ours, script(st), C.enemy(st)), params, params.dt)
            assert torch.isfinite(st.u["x"]).all() and not bool(st.done.any())

    def test_the_scripts_run_on_an_ordinary_battle(self):
        desc = D.army([D.unit("wh_main_emp_cha_general_0", 0, -80, 0, general=True),
                       D.unit("wh2_dlc13_emp_inf_archers_0", 40, -80, 0, 40), D.unit(C.GS, -40, -80, 0, 30)],
                      [D.unit("wh2_main_skv_cha_warlord_0", 0, 80, 180, general=True),
                       D.unit("wh2_main_skv_inf_skavenslave_slingers_0", 30, 80, 180, 35)], 1,
                      C.EMPIRE, "wh2_main_skv_skaven")
        st = scenario.build([desc], per_side=20)
        for f in (C.enemy, C.naive, C.skilled):
            o = f(st)
            assert o.kind.shape == (1, 40)
            assert bool((o.target < 40).all())
