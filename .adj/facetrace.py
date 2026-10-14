"""Who turns the walker (walkpass2 twin): log every facing call that moves the walker's bearing (debug)."""
import sys
import traceback
from pathlib import Path

from tools.nn import walkpass_probe as wp, charge_probe as cp
from tools.nn.sim import movement
from tools.nn.sim.params import load

R = "build/charge-probe/runs/"
runs = [Path(R + r) for r in ("20261013-033936", "20261013-034116", "20261013-034309", "20261013-034447")]
if sys.argv[1] == "side":
    runs = [Path("build/adjmelee/runs/20261014-111840")]
    pick = [ln for ln in cp.load_run(runs[0]) if ln["spec"].get("target2") and ln["spec"].get("a_dx") == float(sys.argv[2])][:1]
else:
    lanes = wp.game_lanes(runs)
    kind, edge, run = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "run"
    pick = [ln for ln in lanes if ln["spec"].get("kind") == kind and ln["spec"].get("edge_m") == edge
            and bool(ln["spec"].get("at_run")) == run][:1]
p = load()
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    d = p.sim
    ks = k.split(".")
    for kk in ks[:-1]:
        d = d[kk]
    d[ks[-1]] = eval(v)
of, ol = movement.face, movement.limit_turn
cnt = [0]


def wrap(fn, name):
    def f(u, *a, **k):
        b0 = u["b"][0].clone()
        out = fn(u, *a, **k)
        dbs = (u["b"][0] - b0).abs()
        for i in range(len(dbs)):
            if float(dbs[i]) > 0.5 and int(u["side"][0, i]) == 1 and float(u["men"][0, i]) > 1 and cnt[0] < 60:
                cnt[0] += 1
                line = traceback.extract_stack()[-2].lineno
                print(f"{name} from battle.py:{line} unit {i} b {float(b0[i]):6.1f} -> {float(u['b'][0, i]):6.1f}")
        return out
    return f


movement.face = wrap(of, "face")
movement.limit_turn = wrap(ol, "limit")
cp.sim_lanes(pick, params=p, copies=1, jitter_m=0.0)
