"""The walkpass probe's table (tools/nn/charge_probe.py plan walkpass): does a unit walking past an enemy get caught?

Per lane kind (idle / busy enemy, edge gap 0 / 3 / 6 / 10 m, straight through; at a run / a walk): the walker's seconds in
melee, its seconds near the clanrats (centres within NEAR_M), its mean speed there and its seconds standing (< 0.5 m/s)
there, whether it reached its point, the HP it lost and the clanrats lost (shares of their start), and (game only) the
share of the near seconds the clanrats' engine target is the walker. 'inmelee': after the move order given in melee, the
seconds until the walker is out of melee, the metres it made in the next 10 s and the HP both lost in those 10 s.

    python -m tools.nn.walkpass_probe [runs...]             # the game's runs (default: every walkpass run)
    python -m tools.nn.walkpass_probe [runs...] --sim       # + the simulator on the same lanes (torch)
    python -m tools.nn.walkpass_probe --layout              # the simulator alone on the plan's places (torch)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

NEAR_M = 40.0       # centre to centre (30 m wide formations side by side at a 0-10 m edge gap: 30-40 m)
STILL_MPS = 0.5


def _hp(rows, start):
    """HP share of the start: the game's hpu (unary), else hp over the first row's."""
    if rows and rows[0] and rows[0].get("hpu") is not None:
        return np.array([np.nan if r is None or r.get("hpu") is None else float(r["hpu"]) for r in rows])
    h0 = float(start.get("hp") or 0) or 1.0
    return np.array([np.nan if r is None or r.get("hp") is None else float(r["hp"]) / h0 for r in rows])


def measure(lane):
    sp = lane["spec"]
    s = [x for x in lane["samples"] if x[1] and x[2]]
    if len(s) < 3:
        return None
    t = np.array([x[0] for x in s])
    a, tg = [x[1] for x in s], [x[2] for x in s]
    ax, az = np.array([r["x"] for r in a], float), np.array([r["z"] for r in a], float)
    tx, tz = np.array([r["x"] for r in tg], float), np.array([r["z"] for r in tg], float)
    am = np.array([bool(r.get("m")) for r in a])
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    spd = np.hypot(np.diff(ax, append=ax[-1]), np.diff(az, append=az[-1])) / np.maximum(dt, 1e-6)
    ha, ht = _hp(a, a[0]), _hp(tg, tg[0])
    out = {"kind": sp.get("kind"), "edge": sp.get("edge_m"), "run": bool(sp.get("at_run")), "lane": sp.get("name")}
    if sp.get("kind") == "inmelee":
        ph = [p for p in lane.get("phases") or [] if str(p.get("phase", "")).startswith("after:")]
        if not ph:
            return None
        t1 = ph[0]["t"] / 1000
        k = t >= t1
        if not k.any():
            return None
        i1 = int(np.argmax(k))
        free = np.nonzero(k & ~am)[0]
        i10 = int(np.argmin(np.abs(t - (t1 + 10))))
        out.update(out_s=float(t[free[0]] - t1) if len(free) else np.nan,
                   moved10=float(np.hypot(ax[i10] - ax[i1], az[i10] - az[i1])),
                   a_lost10=float(ha[i1] - ha[i10]), t_lost10=float(ht[i1] - ht[i10]),
                   melee_after=float(dt[k & am].sum()))
        return out
    near = np.hypot(ax - tx, az - tz) <= NEAR_M
    end_z = sp["z"] - sp.get("pass_beyond_m", 100) + 5
    reached = np.nonzero(az <= end_z)[0]
    walker = sp.get("attacker")
    ct = np.array([(r.get("ct") == walker) for r in tg]) if any("ct" in r for r in tg) else None
    out.update(melee_s=float(dt[am].sum()), near_s=float(dt[near].sum()),
               near_v=float((spd * dt)[near].sum() / max(dt[near].sum(), 1e-6)),
               still_s=float(dt[near & (spd < STILL_MPS)].sum()),
               reach_s=float(t[reached[0]] - t[0]) if len(reached) else np.nan,
               a_lost=float(ha[0] - np.nanmin(ha)), t_lost=float(ht[0] - np.nanmin(ht)),
               ct_walker=float(dt[near & ct].sum() / max(dt[near].sum(), 1e-6)) if ct is not None else np.nan)
    return out


def table(rows, label):
    groups = defaultdict(list)
    for r in rows:
        if r:
            groups[(r["run"], r["kind"], r["edge"])].append(r)
    print(f"## {label}")
    for key in sorted(groups, key=lambda k: (not k[0], str(k[1]), k[2] or 0)):
        g = groups[key]
        run, kind, edge = key

        def m(name):
            v = np.array([x.get(name, np.nan) for x in g], float)
            return np.nanmean(v) if np.isfinite(v).any() else np.nan
        head = f"  {'run ' if run else 'walk'} {kind:8s} edge {edge if edge is not None else '-':>2} m (n {len(g)}):"
        if kind == "inmelee":
            print(f"{head} out of melee {m('out_s'):5.1f} s after the move, moved {m('moved10'):5.1f} m in 10 s, "
                  f"HP lost in those 10 s walker {m('a_lost10'):.3f} enemy {m('t_lost10'):.3f}")
        else:
            print(f"{head} melee {m('melee_s'):5.1f} s, near {m('near_s'):5.1f} s at {m('near_v'):.2f} m/s, standing "
                  f"{m('still_s'):5.1f} s, reached its point at {m('reach_s'):5.1f} s, HP lost walker {m('a_lost'):.3f} "
                  f"enemy {m('t_lost'):.3f}, enemy targets it {m('ct_walker'):.2f}")


def game_lanes(run_dirs):
    out = []
    for d in run_dirs:
        try:
            cfg = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["config"]
        except (OSError, KeyError, ValueError):
            continue
        if cfg.get("plan") == "walkpass":
            out += cp.load_run(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.layout:
        table([measure(x) for x in cp.sim_lanes(cp.layout_lanes("walkpass"), device=a.device, copies=a.copies)],
              "the simulator on the plan's places")
        return
    lanes = game_lanes(a.runs or cp.runs())
    table([measure(x) for x in lanes], f"the game ({len(lanes)} lanes)")
    if a.sim and lanes:
        table([measure(x) for x in cp.sim_lanes(lanes, device=a.device, copies=a.copies)], "the simulator, same lanes")


if __name__ == "__main__":
    main()
