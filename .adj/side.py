"""sidecontact probe table: per lane (attacker offset a_dx, bystander or not) the HP each unit lost a second, the
attacker's centre across the lane off its target's (does it slide onto its target?). Game runs; --sim adds the
simulator twin (torch), rect off / on; --set=k.k=v overrides.
    python .adj/side.py RUN... [--sim] [--set=melee.contact_overlap=True]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

WINDOWS = ((0, 10), (10, 30), (30, 60))


def rows(lane):
    sp = lane["spec"]
    s = lane["samples"]
    t = np.array([x[0] for x in s])
    am = np.array([bool(x[1].get("m")) for x in s])
    c0 = min(lane["contacts"].values()) if lane.get("contacts") else (t[np.argmax(am)] if am.any() else t[0])
    t2 = {round(k, 2): v for k, v in (lane.get("t2") or [])}

    def hp(r):
        return float(r["hp"]) if r and r.get("hp") is not None else np.nan
    A = np.array([hp(x[1]) for x in s])
    B = np.array([hp(x[2]) for x in s])
    C = np.array([hp(t2.get(round(x[0], 2))) for x in s])
    ax = np.array([x[1]["x"] for x in s])
    tx = np.array([x[2]["x"] for x in s])
    out = {"a_dx": sp.get("a_dx", 0), "by": bool(sp.get("target2"))}
    for lo, hi in WINDOWS:
        k0, k1 = np.searchsorted(t, c0 + lo), min(np.searchsorted(t, c0 + hi), len(t) - 1)
        dt = max(t[k1] - t[k0], 1e-6)
        out[f"A{lo}"] = (A[k0] - A[k1]) / dt
        out[f"B{lo}"] = (B[k0] - B[k1]) / dt
        out[f"C{lo}"] = (C[k0] - C[k1]) / dt if sp.get("target2") else np.nan
        out[f"off{lo}"] = float(np.mean(ax[k0:k1 + 1] - tx[k0:k1 + 1]))
    return out


def table(rs, label):
    g = defaultdict(list)
    for r in rs:
        g[(r["by"], r["a_dx"])].append(r)
    print(label)
    print("  lane          n | A's centre off B's, m  0-10 10-30 30-60 | HP/s lost: target B, bystander C, attacker A"
          "  (0-10 s | 10-30 | 30-60 after contact)")
    for k in sorted(g):
        v = g[k]

        def m(key):
            return np.nanmean([x[key] for x in v])
        print(f"  {'by' if k[0] else 'solo'} a_dx {k[1]:4.0f} {len(v)} | {m('off0'):5.1f} {m('off10'):5.1f} {m('off30'):5.1f} | "
              + " | ".join(f"B {m('B' + str(w[0])):5.1f} C {m('C' + str(w[0])):5.1f} A {m('A' + str(w[0])):5.1f}"
                           for w in WINDOWS))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    lanes = [ln for r in runs for ln in cp.load_run(r)]
    table([rows(ln) for ln in lanes], "GAME")
    if "--sim" in sys.argv:
        from tools.nn.sim.params import load
        sets = [a[6:] for a in sys.argv if a.startswith("--set=")]
        for rect in (False, True):
            p = load()
            p.sim["contact"]["rect_gap"] = rect
            for kv in sets:
                k, v = kv.split("=", 1)
                d = p.sim
                ks = k.split(".")
                for kk in ks[:-1]:
                    d = d[kk]
                d[ks[-1]] = eval(v)
            table([rows(ln) for ln in cp.sim_lanes(lanes, params=p, copies=4)], f"SIM rect={rect} {' '.join(sets)}")


if __name__ == "__main__":
    main()
