"""The approach in battle (formation-probe with config approach): paths, stalls, crowding.

    python -m tools.analysis.approach_game <run dir> --map moorlands-route --out DIR

For every manoeuvre: how long it took; for every unit: path length, when it
stopped, the longest stretch it was 'moving' but moved less than 2 m in 10 s
(stuck, as in the obstacle task), and the closest soldiers of two different
units at the end (crowding). Draws the unit paths over the map's blocked cells.
Output: summary.json, paths.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.analysis.formation_probe import closest  # noqa: E402
from tools.analysis.move_probe import soldiers  # noqa: E402
from tools.sim.mapgrid import MapGrid  # noqa: E402

STUCK_WINDOW_MS, STUCK_MOVE_M = 10000, 2.0


def stuck_ms(track):
    """Longest window while moving with < 2 m progress over 10 s. track: [(t_ms, x, z, moving)]."""
    worst = 0
    for i, (t, x, z, moving) in enumerate(track):
        j = i
        while j > 0 and track[j - 1][0] > t - STUCK_WINDOW_MS:
            j -= 1
        if j > 0:
            j -= 1
        t0, x0, z0, _ = track[j]
        if t - t0 >= STUCK_WINDOW_MS and all(m for _, _, _, m in track[j:i + 1]) and math.hypot(x - x0, z - z0) < STUCK_MOVE_M:
            worst = max(worst, t - t0)
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    decisions = [r for r in rows if r["event"] == "approach_decision"]
    manoeuvres = [r for r in rows if r["event"] == "approach_manoeuvre"]
    samples = [r for r in rows if r["event"] == "approach_sample"]
    # Group samples by manoeuvre (t_ms restarts at each manoeuvre).
    runs, current = [], []
    for s in samples:
        if current and s["t_ms"] < current[-1]["t_ms"]:
            runs.append(current)
            current = []
        current.append(s)
    if current:
        runs.append(current)
    report = []
    for n, (m, run) in enumerate(zip(manoeuvres, runs)):
        units = {}
        for s in run:
            for u in s["units"]:
                mo = u["motion"]
                units.setdefault(u["script_name"], []).append((s["t_ms"], mo["x"], mo["z"], mo.get("is_moving") is True))
        per_unit = {}
        for name, track in units.items():
            path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(track, track[1:]))
            last_move = max((t for t, _, _, moving in track if moving), default=0)
            per_unit[name] = {"path_m": round(path, 1), "stopped_s": round(last_move / 1000, 1),
                              "stuck_ms": stuck_ms(track)}
        report.append({"kind": m["kind"], "took_s": m["t_ms"] / 1000, "ended": m["reason"],
                       "closest_at_end": closest(m["units"]), "units": per_unit,
                       "max_stuck_ms": max(v["stuck_ms"] for v in per_unit.values()) if per_unit else 0,
                       "slowest_unit_s": max(v["stopped_s"] for v in per_unit.values()) if per_unit else 0})
    summary = {"decisions": [{k: d[k] for k in ("decision", "reason", "gap_m", "stop_gap_m")} |
                             {"advance_m": (d.get("path") or {}).get("advance_m"),
                              "detour": (d.get("path") or {}).get("detour")} for d in decisions],
               "manoeuvres": report}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    grid = MapGrid(args.map)
    fig, ax = plt.subplots(figsize=(9, 11))
    all_pts = [(u["motion"]["x"], u["motion"]["z"]) for s in samples for u in s["units"]]
    xs, zs = zip(*all_pts)
    blocked = grid.blocked_near([{"x": x, "z": z} for x, z in ((min(xs), min(zs)), (max(xs), max(zs)))], margin=40)
    if blocked:
        ax.scatter(*zip(*blocked), s=8, marker="s", c="#8c564b", alpha=0.4, lw=0)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    names = sorted({u["script_name"] for s in samples for u in s["units"]})
    for k, name in enumerate(names):
        track = [(u["motion"]["x"], u["motion"]["z"]) for s in samples for u in s["units"] if u["script_name"] == name]
        tx, tz = zip(*track)
        ax.plot(tx, tz, lw=0.9, c=colors[k % 20])
    for m in manoeuvres:
        for u in m["units"]:
            if u.get("soldiers_dm"):
                p = soldiers(u)
                ax.scatter(p[:, 0], p[:, 1], s=0.3, c="k", alpha=0.4)
    worst = max((r["max_stuck_ms"] for r in report), default=0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set(xlabel="X, m", ylabel="Z, m")
    ax.set_title("Сближение в игре: линии — пути отрядов, точки — бойцы в конце каждого манёвра, коричневое — скала\n"
                 + "; ".join(f'{r["kind"]} {r["took_s"]:.0f} с' for r in report)
                 + f"; дольше всего без продвижения: {worst / 1000:.0f} с", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "paths.png", dpi=100)
    plt.close(fig)
    print(json.dumps({"manoeuvres": [{k: v for k, v in r.items() if k != "units"} for r in report]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
