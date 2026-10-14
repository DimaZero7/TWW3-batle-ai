"""walkpass2 twin, one case, rect off/on: per-2 s walker speed, clanrat bearing, melee flags (debug)."""
import sys
from pathlib import Path
import numpy as np
from tools.nn import walkpass_probe as wp, charge_probe as cp
from tools.nn.sim.params import load
R = "build/charge-probe/runs/"
runs = [Path(R + r) for r in ("20261013-033936", "20261013-034116", "20261013-034309", "20261013-034447")]
lanes = wp.game_lanes(runs)
kind, edge, run = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "run"
sets = sys.argv[4:]
pick = [l for l in lanes if l["spec"].get("kind") == kind and l["spec"].get("edge_m") == edge and bool(l["spec"].get("at_run")) == run][:1]
def show(ln, label):
    s = ln["samples"]
    t = np.array([x[0] for x in s]); ax = np.array([x[1]["x"] for x in s]); az = np.array([x[1]["z"] for x in s])
    tb = np.array([x[2]["b"] for x in s]); am = [bool(x[1].get("m")) for x in s]
    sp = np.hypot(np.diff(ax), np.diff(az)) / np.maximum(np.diff(t), 1e-6)
    print(label)
    for k in range(0, len(t) - 1, 4):
        print(f"  t {t[k]:5.1f} a ({ax[k]:7.1f},{az[k]:6.1f}) v {sp[k]:4.2f} m {int(am[k])} tg_b {tb[k]:6.1f} a_hp {x_hp(s[k][1]):.3f} t_hp {x_hp(s[k][2]):.3f}")
def x_hp(r):
    return float(r.get("hpu") if r.get("hpu") is not None else r.get("hp", 0))
show(pick[0], "GAME")
for rect in (False, True):
    p = load()
    p.sim["contact"]["rect_gap"] = rect
    for kv in sets:
        k, v = kv.split("=")
        d = p.sim
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d[kk]
        d[ks[-1]] = eval(v)
    sim = cp.sim_lanes(pick, params=p, copies=1)
    show(sim[0], f"SIM rect={rect}")
