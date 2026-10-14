"""Trace one sidecontact lane in the sim: per ~2 s the A-B / A-C gap, contact, men striking F and the rate (debug)."""
import sys
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn.sim import melee, geometry
from tools.nn.sim.params import load

runs = [Path("build/adjmelee/runs/20261014-111840"), Path("build/adjmelee/runs/20261014-111937")]
a_dx = float(sys.argv[1])
rect = sys.argv[2] == "1"
sets = sys.argv[3:]
lanes = [ln for r in runs for ln in cp.load_run(r) if ln["spec"].get("target2") and ln["spec"].get("a_dx") == a_dx][:1]
p = load()
p.sim["contact"]["rect_gap"] = rect
for kv in sets:
    k, v = kv.split("=", 1)
    d = p.sim
    ks = k.split(".")
    for kk in ks[:-1]:
        d = d[kk]
    d[ks[-1]] = eval(v)
LOG = []
orig = melee.strikes
clock = [0]


def hook(u, pw, contact, params, charge_now, first=False):
    out = orig(u, pw, contact, params, charge_now, first)
    clock[0] += 1
    if clock[0] % 20 == 0:
        # slots: side 1 a = 0 (after lord?), find by men
        LOG.append((u["x"][0].cpu().numpy().copy(), u["z"][0].cpu().numpy().copy(), u["b"][0].cpu().numpy().copy(),
                    pw["gap"][0].cpu().numpy().copy(), contact[0].cpu().numpy().copy(), out[3][0].cpu().numpy().copy(),
                    out[0][0].cpu().numpy().copy(), u["side"][0].cpu().numpy().copy(), u["men"][0].cpu().numpy().copy()))
    return out


melee.strikes = hook
from tools.nn.sim import battle  # noqa
battle.melee.strikes = hook
cp.sim_lanes(lanes, params=p, copies=1, jitter_m=0.0)
x, z, b, gap, con, F, rate, side, men = LOG[0]
print("units:", [(i, int(side[i]), round(float(x[i]), 1), round(float(z[i]), 1), int(men[i])) for i in range(len(x))])
alive = [i for i in range(len(x)) if men[i] > 1]
A = [i for i in alive if side[i] == 1][0]
tgs = [i for i in alive if side[i] == 2]
B, C = sorted(tgs, key=lambda i: x[i])[:2]
print("A B C", A, B, C)
for k, (x, z, b, gap, con, F, rate, side, men) in enumerate(LOG):
    print(f"t~{(k + 1) * 20 * 0.1:5.1f} A x {x[A] - x[B]:5.1f} z {z[A] - z[B]:5.1f} b {b[A]:5.0f} | gapAB {gap[A, B]:5.2f} AC {gap[A, C]:5.2f}"
          f" con {int(con[A, B])}{int(con[A, C])} | F A->B {F[A, B]:5.1f} A->C {F[A, C]:5.1f} B->A {F[B, A]:5.1f} C->A {F[C, A]:5.1f}"
          f" | rate A->B {rate[A, B]:5.1f} A->C {rate[A, C]:5.1f} C->A {rate[C, A]:5.1f}")
