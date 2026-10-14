#!/bin/bash
# walkpass2 twin, rect off and on: compact rows (case, push speed) -> /work/wp_<tag>.txt
cd /repo || exit 1
R=build/charge-probe/runs
runs="$R/20261013-033936 $R/20261013-034116 $R/20261013-034309 $R/20261013-034447 $R/20261013-014012 $R/20261013-014115"
python - "$@" <<'EOF'
import sys, io, contextlib, json
sys.argv = ["x"]
from tools.nn import walkpass_probe as wp, charge_probe as cp
from tools.nn.sim.params import load
from pathlib import Path
R = "build/charge-probe/runs/"
runs = [Path(R + r) for r in ("20261013-033936", "20261013-034116", "20261013-034309", "20261013-034447", "20261013-014012", "20261013-014115")]
lanes = wp.game_lanes(runs)
import numpy as np
from collections import defaultdict
def summ(rows, label):
    g = defaultdict(list)
    for r in rows:
        if r and r.get("kind") != "inmelee":
            g[(r["run"], r["kind"], r["edge"])].append(r)
    out = {}
    for k in sorted(g, key=lambda k: (not k[0], str(k[1]), k[2] or 0)):
        v = np.array([x.get("push_v", np.nan) for x in g[k]], float)
        out[k] = np.nanmean(v) if np.isfinite(v).any() else np.nan
    return out
G = summ([wp.measure(x) for x in lanes], "game")
res = {}
for rect, tw in ((False, 0), (True, 0), (True, 1)):
    p = load()
    p.sim["contact"]["rect_gap"] = rect; p.sim["contact"]["push_keep_facing"] = tw
    res[(rect, tw)] = summ([wp.measure(x) for x in cp.sim_lanes(lanes, params=p, copies=4)], str(rect))
print("case (run, kind, edge): push m/s  game | ray | rect")
for k in G:
    print(f"  {'run' if k[0] else 'walk'} {k[1]:6s} {k[2]!s:>4}: {G[k]:5.2f} | {res[(False, 0)].get(k, np.nan):5.2f} | {res[(True, 0)].get(k, np.nan):5.2f} | {res[(True, 1)].get(k, np.nan):5.2f}")
EOF
