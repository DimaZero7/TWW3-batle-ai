"""Trace one walkpass2 lane in the sim: walker speed, cover (blocked_share), gap to the clanrats, their bearing (debug).
    python .adj/wptrace.py idle -5 walk [k=v ...]"""
import sys
from pathlib import Path

from tools.nn import walkpass_probe as wp, charge_probe as cp
from tools.nn.sim import geometry
from tools.nn.sim.params import load

R = "build/charge-probe/runs/"
runs = [Path(R + r) for r in ("20261013-033936", "20261013-034116", "20261013-034309", "20261013-034447")]
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
orig = geometry.blocked_share
n = [0]


def hook(u, pw, mask, gx, gz):
    out = orig(u, pw, mask, gx, gz)
    n[0] += 1
    if n[0] % 10 == 0:
        men = u["men"][0]
        A = 0 if u["side"][0, 0] == 1 and men[0] > 1 else int((u["side"][0] == 1).nonzero()[1])
        C = [i for i in range(len(men)) if int(u["side"][0, i]) == 2 and float(men[i]) > 1][0]
        sp = float((u["vx"][0, A] ** 2 + u["vz"][0, A] ** 2).sqrt())
        print(f"t~{n[0] * 0.1:5.1f} walker z {float(u['z'][0, A]):6.1f} v {sp:4.2f} cover {float(out[0, A]):4.2f} "
              f"gap {float(pw['gap'][0, A, C]):6.2f} touch {int(mask[0, A, C])} clan b {float(u['b'][0, C]):5.1f} "
              f"walker b {float(u['b'][0, A]):5.1f} dx {float(u['x'][0, A] - u['x'][0, C]):5.1f}")
    return out


geometry.blocked_share = hook
cp.sim_lanes(pick, params=p, copies=1, jitter_m=0.0)
