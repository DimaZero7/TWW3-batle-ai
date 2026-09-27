"""How the game AI stands (enemy-layout runs): distances that make one "group".

    python -m tools.analysis.enemy_layout <run dir> [<run dir> ...] --out DIR

Each run dir holds events.jsonl of one layout. For the final snapshot
(soldier positions, full view — research data) it measures:
  centre links   distance between unit middles (middle of soldiers)
  gap links      shortest distance between soldiers of two units (edge to edge)
  group link      the longest link of the minimum spanning tree: the smallest
                 threshold at which chaining neighbours joins the whole army
                 into one group; computed with and without the lord
  extent         front and depth of the army in its own frame
  seen           share of enemy units our side saw, over the hold
Output: summary.json, layouts.png.
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

from tools.analysis.move_probe import soldiers  # noqa: E402

KIND_COLORS = {"general": "#d62728", "spearmen": "#1f77b4", "archers": "#2ca02c"}


def kind(key):
    return "general" if "_cha_general" in key else "archers" if "archers" in key else "spearmen"


def spanning_tree(n, dist):
    """Prim's minimum spanning tree: list of (i, j, length)."""
    if n < 2:
        return []
    inside, edges = {0}, []
    while len(inside) < n:
        best = None
        for i in inside:
            for j in range(n):
                if j not in inside and (best is None or dist[i][j] < best[2]):
                    best = (i, j, dist[i][j])
        edges.append(best)
        inside.add(best[1])
    return edges


def links(points_list):
    """Centre and gap distance matrices for units given as soldier point arrays."""
    n = len(points_list)
    middles = [p.mean(0) for p in points_list]
    centre = [[float(np.hypot(*(middles[i] - middles[j]))) for j in range(n)] for i in range(n)]
    gap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = points_list[i], points_list[j]
            d = float(np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])))
            gap[i][j] = gap[j][i] = d
    return middles, centre, gap


def group(units, with_lord=True):
    chosen = [u for u in units if with_lord or kind(u["key"]) != "general"]
    pts = [soldiers(u) for u in chosen]
    middles, centre, gap = links(pts)
    tree_c, tree_g = spanning_tree(len(chosen), centre), spanning_tree(len(chosen), gap)
    return {"units": len(chosen),
            "group_link_centre_m": round(max((e[2] for e in tree_c), default=0), 1),
            "group_link_gap_m": round(max((e[2] for e in tree_g), default=0), 1),
            "nearest_centre_m": [round(min(c for j, c in enumerate(row) if j != i), 1) for i, row in enumerate(centre)]
            if len(chosen) > 1 else [],
            "tree_gap": [(chosen[i]["name"], chosen[j]["name"], round(d, 1)) for i, j, d in tree_g]}, middles


def extent(units, bearing):
    b = math.radians(bearing)
    fwd, right = np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])
    pts = np.vstack([soldiers(u) for u in units])
    c = pts.mean(0)
    return {"front_m": round(float(np.ptp((pts - c) @ right)), 1), "depth_m": round(float(np.ptp((pts - c) @ fwd)), 1)}


def analyse(run_dir):
    rows = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    layout = next(r["layout"] for r in rows if "layout" in r)
    end = next(r for r in rows if r["event"] == "enemy_snapshot" and r["stage"] == "end")
    units = [u for u in end["units"] if u.get("soldiers_dm")]
    mode = next((r for r in rows if r["event"] == "enemy_mode"), {})
    samples = [r for r in rows if r["event"] == "enemy_sample"]
    seen = [sum(u["seen"] is True for u in s["units"]) / len(s["units"]) for s in samples]
    moving_last = max((s["t_ms"] for s in samples if any(u["motion"].get("is_moving") for u in s["units"])), default=0)
    bearing = float(np.median([u["motion"]["bearing"] for u in units if kind(u["key"]) != "general"]))
    whole, middles = group(units, True)
    no_lord, _ = group(units, False)
    lord = [i for i, u in enumerate(units) if kind(u["key"]) == "general"]
    lord_gap = None
    if lord:
        others = [soldiers(u) for i, u in enumerate(units) if i not in lord]
        lp = soldiers(units[lord[0]])
        lord_gap = round(min(float(np.min(np.hypot(lp[:, None, 0] - o[None, :, 0], lp[:, None, 1] - o[None, :, 1])))
                             for o in others), 1) if others else None
    return {"layout": layout, "units": len(units), "defend": mode.get("used"),
            "with_lord": whole, "without_lord": no_lord, "lord_gap_to_army_m": lord_gap,
            "extent": extent(units, bearing), "facing_deg": round(bearing, 1),
            "seen_share_min": round(min(seen), 2) if seen else None, "last_movement_s": moving_last / 1000,
            "_units": units, "_middles": [m.tolist() for m in middles]}


def draw(results, path):
    n = len(results)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        for u in r["_units"]:
            p = soldiers(u)
            ax.scatter(p[:, 0], p[:, 1], s=1, c=KIND_COLORS[kind(u["key"])])
        names = [u["name"] for u in r["_units"]]
        for a, b, d in r["with_lord"]["tree_gap"]:
            pa, pb = r["_middles"][names.index(a)], r["_middles"][names.index(b)]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], c="k", lw=0.6)
            ax.annotate(f"{d:.0f}", ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2), fontsize=7)
        ax.set_title(f'{r["layout"]}: группа склеивается при зазоре {r["with_lord"]["group_link_gap_m"]} м '
                     f'(без лорда {r["without_lord"]["group_link_gap_m"]} м)\n'
                     f'фронт × глубина {r["extent"]["front_m"]} × {r["extent"]["depth_m"]} м', fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Штатный ИИ в обороне. Синие — копейщики, зелёные — лучники, красные — лорд; "
                 "линии — самые короткие связи, числа — зазор между краями строя, м")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    results = [analyse(r) for r in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)
    draw(results, args.out / "layouts.png")
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    summary = {"layouts": clean,
               "group_link_gap_max_m": max(r["with_lord"]["group_link_gap_m"] for r in clean),
               "group_link_centre_max_m": max(r["with_lord"]["group_link_centre_m"] for r in clean)}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in clean:
        print(f'{r["layout"]:18s} units {r["units"]:2d}  gap link {r["with_lord"]["group_link_gap_m"]:5} m '
              f'(no lord {r["without_lord"]["group_link_gap_m"]:5})  centre link {r["with_lord"]["group_link_centre_m"]:5} m  '
              f'lord gap {r["lord_gap_to_army_m"]}  extent {r["extent"]}  seen min {r["seen_share_min"]}  '
              f'moved until {r["last_movement_s"]} s  defend {r["defend"]}')
    print("max gap link", summary["group_link_gap_max_m"], "max centre link", summary["group_link_centre_max_m"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
