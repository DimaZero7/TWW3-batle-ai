"""The walkpass probe's table (tools/nn/charge_probe.py plan walkpass): does a unit walking past an enemy get caught?

Per lane kind (idle / busy enemy, edge gap 0 / 3 / 6 / 10 m, straight through; at a run / a walk): the walker's seconds in
melee, its seconds near the clanrats (centres within NEAR_M), its mean speed there and its seconds standing (< 0.5 m/s)
there, whether it reached its point, the HP it lost and the clanrats lost (shares of their start), and (game only) the
share of the near seconds the clanrats' engine target is the walker. 'inmelee': after the move order given in melee, the
seconds until the walker is out of melee, the metres it made in the next 10 s and the HP both lost in those 10 s.
The path (walkpass2: the walker's straight line clipping the clanrats' edge): its bend - the walker centre's largest
step aside from its straight line, away from the clanrats (+) and towards them - the seconds it stood (< STILL_MPS)
on the way (after its first START_S, before its point), its slowest speed on the way, and the edge gap it actually
passed at (walker centre to clanrats' centre across the line, less the half widths, when its centre passes theirs) and
its push - the way it made towards its point a second from its first melee second until it got PAST_M beyond the
clanrats' centre (the controls stop 21-22 m beyond it: their point) or the lane ended (net: the game's formation centre jitters by metres in a scrum), at least PUSH_MIN_S long.

    python -m tools.nn.walkpass_probe [runs...]             # the game's runs (default: every walkpass run)
    python -m tools.nn.walkpass_probe [runs...] --sim       # + the simulator on the same lanes (torch)
    python -m tools.nn.walkpass_probe --layout [--plan walkpass2]  # the simulator alone on the plan's places (torch)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

NEAR_M = 40.0       # centre to centre (30 m wide formations side by side at a 0-10 m edge gap: 30-40 m)
STILL_MPS = 0.5
PAST_M = 16.0       # the push ends this far beyond the clanrats' start centre (along the walk)
PUSH_MIN_S = 5.0    # a push shorter than this (the walker got there within 5 s of its first melee second): not counted
START_S = 3.0       # the walker's start from a halt (the game's units take ~2 s to get going)


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
    # the path: lateral steps from the start's x (+: away from the clanrats), the stops on the way, the edge passed at
    side = 1.0 if ax[0] >= tx[0] else -1.0
    dev = (ax - ax[0]) * side
    # (a 2 s window: the game's positions come in steps, a 0.5 s sample can read 0 at a run)
    j0, j1 = np.clip(np.arange(len(t)) - 2, 0, len(t) - 1), np.clip(np.arange(len(t)) + 2, 0, len(t) - 1)
    v2 = np.hypot(ax[j1] - ax[j0], az[j1] - az[j0]) / np.maximum(t[j1] - t[j0], 1e-6)
    way = (t - t[0] >= START_S) & ((t < t[reached[0]]) if len(reached) else np.ones_like(t, bool))
    passed = np.nonzero(az <= tz)[0]
    half = (float(sp.get("a_width") or cp.WIDTH_M) + float(sp.get("t_width") or cp.WIDTH_M)) / 2
    out.update(bend_away=float(dev[way].max()) if way.any() else np.nan,
               bend_to=float(-dev[way].min()) if way.any() else np.nan,
               stop_s=float(dt[way & (v2 < STILL_MPS)].sum()),
               min_v=float(v2[way].min()) if way.any() else np.nan,
               edge_at=float(abs(ax[passed[0]] - tx[passed[0]]) - half) if len(passed) else np.nan)
    first_m = np.nonzero(am)[0]
    if len(first_m):
        i0 = first_m[0]
        past = np.nonzero(az - tz[0] <= -PAST_M)[0]
        i1 = (past[past > i0][0] if (past > i0).any() else len(t) - 1)
        if t[i1] - t[i0] >= PUSH_MIN_S:
            out.update(push_v=float((az[i0] - az[i1]) / (t[i1] - t[i0])), push_s=float(t[i1] - t[i0]))
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
            print(f"{'':30s} path: bend away {m('bend_away'):5.1f} m, towards {m('bend_to'):5.1f} m, stood on the way "
                  f"{m('stop_s'):5.1f} s, slowest {m('min_v'):.2f} m/s, passed at an edge gap of {m('edge_at'):6.1f} m, "
                  f"push {m('push_v'):.2f} m/s over {m('push_s'):4.1f} s")


def game_lanes(run_dirs):
    out = []
    for d in run_dirs:
        try:
            cfg = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["config"]
        except (OSError, KeyError, ValueError):
            continue
        if cfg.get("plan") in ("walkpass", "walkpass2"):
            out += cp.load_run(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--plan", default="walkpass", choices=("walkpass", "walkpass2"), help="--layout: the plan")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.layout:
        table([measure(x) for x in cp.sim_lanes(cp.layout_lanes(a.plan), device=a.device, copies=a.copies)],
              "the simulator on the plan's places")
        return
    lanes = game_lanes(a.runs or cp.runs())
    table([measure(x) for x in lanes], f"the game ({len(lanes)} lanes)")
    if a.sim and lanes:
        table([measure(x) for x in cp.sim_lanes(lanes, device=a.device, copies=a.copies)], "the simulator, same lanes")


if __name__ == "__main__":
    main()
