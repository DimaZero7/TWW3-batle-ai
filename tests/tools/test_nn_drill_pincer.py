"""tools/nn/train/drills/pincer.py: the frame's battles are a pincer situation (two of ours per enemy
unit, in a column opposite it, we attack, the enemy holds and defends), the skilled script sends
one unit of each pair round to the enemy's flank, and the scripts run in the simulator."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train.drills import pincer as P  # noqa: E402

SEEDS = range(48)


def test_two_of_ours_per_enemy_unit_in_a_column_and_we_attack():
    rosters = set()
    for desc, ours in D.battles(P.DRILL, SEEDS):
        mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
        assert desc["attacker"] == ours
        assert P.K[0] <= len(theirs) <= P.K[1] and len(mine) == 2 * len(theirs)
        assert {u["key"] for u in theirs} == {P.ENEMY[0]} and len({u["key"] for u in mine}) == 1
        rosters.add(mine[0]["key"])
        for e in theirs:
            near = sorted(mine, key=lambda u: math.dist((u["x"], u["z"]), (e["x"], e["z"])))[:2]
            d = [math.dist((u["x"], u["z"]), (e["x"], e["z"])) for u in near]
            assert P.GAP_M[0] - 1 <= d[0] <= P.GAP_M[1] + P.JITTER_M + 1
            assert d[1] - d[0] >= P.COLUMN_M[0] - 2 * P.JITTER_M          # the second stands behind the first
    assert rosters == {k for k, _, _ in P.OURS}
    assert {o for _, o in D.battles(P.DRILL, SEEDS)} == {1, 2}


def test_the_skilled_script_pins_one_and_sends_the_other_to_the_flank():
    desc, ours = D.battles(P.DRILL, [0], both_sides=False)[0]
    st = scenario.build([desc])
    o = P.skilled(st)
    O.check(o, st.N)
    mine = (st.u["side"][0] == ours).nonzero().flatten().tolist()
    kinds = sorted(int(o.kind[0, i]) for i in mine)
    assert kinds.count(O.MOVE) == len(mine)                               # pinners walk up, flankers run round
    assert sum(bool(o.run[0, i]) for i in mine) == len(mine) // 2         # the flankers run
    assert all(int(k) == O.HOLD for k in P.enemy(st).kind[0])


def test_scripts_run_a_few_battles():
    pairs = D.battles(P.DRILL, range(4))
    st = scenario.build([d for d, _ in pairs])
    ours = torch.tensor([o for _, o in pairs])
    p = load()
    for _ in range(40):
        battle.step(st, D.merged(st, ours, P.skilled(st), P.enemy(st)), p, p.dt)
    assert torch.isfinite(st.u["x"]).all()
