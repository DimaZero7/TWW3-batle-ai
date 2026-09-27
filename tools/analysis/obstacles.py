"""Offline obstacle analysis of a captured map; never starts the game.

    python -m tools.analysis.obstacles <capture dir> <output dir>

Input: a capture directory with clear.npy, inside.npy, height.npy, forest.npy
(arrays [iz, ix], south-to-north rows) and optionally objects.json. Grid
origin and step come from summary.json when present.

Output:
  summary.json   obstacle clusters, slopes, gaps between obstacles, test candidates
  gaps.csv       every gap between two obstacle clusters up to --max-gap metres
  obstacles.png  obstacles, forest, objects and gaps on the height map
  clearance.png  free space around each cell (distance to the nearest obstacle)

"Obstacle" here = a cell inside the radar frame where is_area_clear() was
false. That is NOT verified passability (see docs/ru/game/map); the report
states it. Gap width is the free distance between two clusters' cell edges.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GAP_BINS = [(3, 10, "narrow"), (10, 20, "medium"), (20, 31, "wide")]
# Thin line objects that is_area_clear() may not see at the grid step.
LINE_OBJECT_PREFIXES = ("glb_fence",)


def line_objects(objects, link_m=9.0, max_gap_m=20.0):
    """Groups fence-like object origins into lines and finds gaps between line ends.

    Returns (lines, gaps). Gap width = origin distance minus one median section
    length (origins are section centres), i.e. an estimate of the free width.
    """
    pts = [o for o in objects if o["name"].startswith(LINE_OBJECT_PREFIXES)]
    if len(pts) < 3:
        return [], []
    xy = np.array([[o["x"], o["z"]] for o in pts])
    tree = cKDTree(xy)
    section = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    lab = np.arange(len(xy))
    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for i, j in tree.query_pairs(link_m):
        lab[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(xy))])
    lines = []
    for r in np.unique(roots):
        q = xy[roots == r]
        if len(q) >= 3:
            lines.append({"sections": int(len(q)), "center": q.mean(0).round(1).tolist(),
                          "length_m": round(float(np.ptp(q, 0).max()) + section, 1)})
    raw = []
    for i, j in tree.query_pairs(max_gap_m):
        if roots[i] != roots[j]:
            d = float(np.hypot(*(xy[i] - xy[j])))
            raw.append((d - section, ((xy[i] + xy[j]) / 2).round(1).tolist()))
    raw.sort()
    gaps = []
    for width, mid in raw:
        if width >= 1 and all(math.dist(mid, g["mid"]) > 15 for g in gaps):
            gaps.append({"width_m": round(width, 1), "mid": mid})
    lines.sort(key=lambda l: -l["sections"])
    return {"section_m": round(section, 1), "count": int(len(pts)), "lines": lines}, gaps


def load(capture):
    arrays = {name: np.load(capture / f"{name}.npy") for name in ("clear", "inside", "height", "forest")}
    origin, step = -512.0, 3.0
    summary = capture / "summary.json"
    if summary.exists():
        grid = json.loads(summary.read_text(encoding="utf-8")).get("grid", {})
        origin, step = float(grid.get("min_x", origin)), float(grid.get("step", step))
    objects = []
    if (capture / "objects.json").exists():
        objects = json.loads((capture / "objects.json").read_text(encoding="utf-8"))
    return arrays, origin, step, objects


def world(ix, iz, origin, step):
    return origin + (ix + 0.5) * step, origin + (iz + 0.5) * step


def analyse(arrays, origin, step, objects, max_gap):
    inside = arrays["inside"].astype(bool)
    blocked = inside & (arrays["clear"] == 0)
    labels, count = ndimage.label(blocked, structure=np.ones((3, 3)))
    iz_all, ix_all = np.nonzero(blocked)
    xs, zs = world(ix_all, iz_all, origin, step)
    cell_label = labels[iz_all, ix_all]
    tree = cKDTree(np.column_stack([xs, zs]))

    obj_xy = np.array([[o["x"], o["z"]] for o in objects]) if objects else np.zeros((0, 2))
    obj_tree = cKDTree(obj_xy) if len(obj_xy) else None

    clusters = []
    for lab in range(1, count + 1):
        sel = cell_label == lab
        cx, cz = xs[sel], zs[sel]
        cells = int(sel.sum())
        area = cells * step * step
        span = float(max(cx.max() - cx.min(), cz.max() - cz.min()) + step)
        hull_area = area
        if cells >= 3:
            try:
                corners = np.concatenate([np.column_stack([cx + dx, cz + dz])
                                          for dx in (-step / 2, step / 2) for dz in (-step / 2, step / 2)])
                hull_area = float(ConvexHull(corners).volume)
            except Exception:
                hull_area = area
        near = []
        if obj_tree is not None:
            idx = set()
            for p in np.column_stack([cx, cz]):
                idx.update(obj_tree.query_ball_point(p, step * 2))
            near = sorted({objects[i]["name"] for i in idx})
        clusters.append({"id": lab, "cells": cells, "area_m2": area, "span_m": span,
                         "center": [float(cx.mean()), float(cz.mean())],
                         "concavity": round(1 - area / hull_area, 3) if hull_area > 0 else 0.0,
                         "objects": near[:8]})

    # Gaps: closest pair of cells between two clusters within max_gap.
    gaps = {}
    for i, p in enumerate(np.column_stack([xs, zs])):
        for j in tree.query_ball_point(p, max_gap + step):
            a, b = cell_label[i], cell_label[j]
            if a >= b:
                continue
            d = math.dist(p, (xs[j], zs[j]))
            if (a, b) not in gaps or d < gaps[(a, b)][0]:
                gaps[(a, b)] = (d, i, j)
    free = ~blocked
    clearance = ndimage.distance_transform_edt(free) * step  # centre-to-centre metres
    by_id = {c["id"]: c for c in clusters}
    gap_rows = []
    for (a, b), (d, i, j) in gaps.items():
        width = d - step  # free distance between cell edges
        if width <= 0 or width > max_gap:
            continue
        mx, mz = (xs[i] + xs[j]) / 2, (zs[i] + zs[j]) / 2
        # A real passage: the straight segment between the two clusters is free.
        n = max(2, int(d / (step / 2)))
        ok = True
        for t in np.linspace(0, 1, n)[1:-1]:
            px, pz = xs[i] + (xs[j] - xs[i]) * t, zs[i] + (zs[j] - zs[i]) * t
            gx, gz = int((px - origin) // step), int((pz - origin) // step)
            if blocked[gz, gx] and labels[gz, gx] not in (a, b):
                ok = False
                break
        if not ok:
            continue
        gap_rows.append({"a": int(a), "b": int(b), "width_m": round(width, 1),
                         "mid_x": round(float(mx), 1), "mid_z": round(float(mz), 1),
                         "a_cells": by_id[a]["cells"], "b_cells": by_id[b]["cells"],
                         "a_objects": ",".join(by_id[a]["objects"][:3]),
                         "b_objects": ",".join(by_id[b]["objects"][:3])})
    gap_rows.sort(key=lambda g: g["width_m"])

    gy, gx = np.gradient(np.where(inside, arrays["height"], np.nan), step)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope_stats = {f"over_{t}_deg": int(np.nansum(slope > t)) for t in (15, 25, 35, 45)}
    steep_free = int(np.nansum((slope > 35) & ~blocked & inside))

    candidates = {}
    for low, high, name in GAP_BINS:
        pool = [g for g in gap_rows if low <= g["width_m"] < high and min(g["a_cells"], g["b_cells"]) >= 4]
        pool.sort(key=lambda g: -(g["a_cells"] + g["b_cells"]))
        candidates[f"gap_{name}"] = pool[:3]
    corners = sorted((c for c in clusters if c["cells"] >= 20), key=lambda c: -c["concavity"])[:3]
    fences, fence_gaps = line_objects(objects)
    candidates["fence_gap"] = [{"mid_x": g["mid"][0], "mid_z": g["mid"][1], "width_m": g["width_m"]}
                               for g in fence_gaps[:3]]
    candidates["corner"] = [{"cluster": c["id"], "center": [round(v, 1) for v in c["center"]],
                             "span_m": c["span_m"], "concavity": c["concavity"], "objects": c["objects"][:4]}
                            for c in corners]

    sizes = np.array([c["cells"] for c in clusters]) if clusters else np.zeros(0)
    summary = {
        "grid": {"origin": origin, "step_m": step, "shape": list(blocked.shape),
                 "inside_cells": int(inside.sum())},
        "obstacle_definition": "inside the radar frame and is_area_clear() == false; NOT verified passability",
        "obstacles": {"cells": int(blocked.sum()), "area_m2": float(blocked.sum() * step * step),
                      "clusters": count,
                      "single_cell_clusters": int((sizes == 1).sum()),
                      "clusters_ge_10_cells": int((sizes >= 10).sum()),
                      "largest_cells": int(sizes.max()) if len(sizes) else 0,
                      "with_objects_nearby": sum(1 for c in clusters if c["objects"]),
                      "without_objects_nearby": sum(1 for c in clusters if not c["objects"])},
        "forest_cells_inside": int(((arrays["forest"] == 1) & inside).sum()),
        "slopes": dict(slope_stats, steep_over_35_but_clear=steep_free,
                       note="is_area_clear does not mark steep slopes; passability on slopes unknown"),
        "gaps": {"total": len(gap_rows),
                 **{name: sum(1 for g in gap_rows if low <= g["width_m"] < high) for low, high, name in GAP_BINS}},
        "clearance_m": {"free_cells_under_5m": int(((clearance < 5) & free & inside).sum()),
                        "free_cells_under_10m": int(((clearance < 10) & free & inside).sum())},
        "fence_lines": dict(fences or {}, gaps=fence_gaps[:20],
                            note="is_area_clear barely marks fences; whether they block units is unverified"),
        "candidates": candidates,
        "largest_clusters": sorted(clusters, key=lambda c: -c["cells"])[:10],
    }
    return summary, gap_rows, blocked, clearance, slope, inside


def draw(out, arrays, origin, step, objects, blocked, clearance, inside, gap_rows, candidates):
    extent = (origin, origin + blocked.shape[1] * step, origin, origin + blocked.shape[0] * step)
    height = np.where(inside, arrays["height"], np.nan)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(height, origin="lower", extent=extent, cmap="Greys", alpha=0.55)
    forest = np.where((arrays["forest"] == 1) & inside, 1.0, np.nan)
    ax.imshow(forest, origin="lower", extent=extent, cmap="Greens", vmin=0, vmax=1.6, alpha=0.45)
    ax.imshow(np.where(blocked, 1.0, np.nan), origin="lower", extent=extent, cmap="autumn", vmin=0, vmax=1)
    if objects:
        ax.scatter([o["x"] for o in objects], [o["z"] for o in objects], s=3, c="k", alpha=0.5, label="object origins")
    colours = {"narrow": "#d62728", "medium": "#ff7f0e", "wide": "#1f77b4"}
    fence_gaps = candidates.get("_fence_gaps_all", [])
    if fence_gaps:
        ax.scatter([g["mid"][0] for g in fence_gaps], [g["mid"][1] for g in fence_gaps], s=40, marker="x",
                   c="#9467bd", label=f"gap between fence lines ({len(fence_gaps)})")
    for low, high, name in GAP_BINS:
        sel = [g for g in gap_rows if low <= g["width_m"] < high]
        ax.scatter([g["mid_x"] for g in sel], [g["mid_z"] for g in sel], s=14, c=colours[name],
                   label=f"gap {low}-{high - 1} m ({len(sel)})")
    for key, items in candidates.items():
        if key.startswith("_"):
            continue
        for k, c in enumerate(items):
            x, z = (c["mid_x"], c["mid_z"]) if "mid_x" in c else c["center"]
            ax.annotate(f"{key}#{k + 1}", (x, z), xytext=(6, 6), textcoords="offset points", fontsize=8,
                        weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set(xlabel="X, m", ylabel="Z, m", title="Obstacles (is_area_clear = false), forest, objects, gaps")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "obstacles.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    shown = np.where(inside & ~blocked, np.minimum(clearance, 40), np.nan)
    im = ax.imshow(shown, origin="lower", extent=extent, cmap="viridis")
    ax.contour(np.where(inside, clearance, 99), levels=[5, 10, 15], origin="lower", extent=extent,
               colors=["#d62728", "#ff7f0e", "#ffffff"], linewidths=0.6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="free space to nearest obstacle, m (capped at 40)")
    ax.set(xlabel="X, m", ylabel="Z, m", title="Clearance: contours at 5 / 10 / 15 m")
    fig.tight_layout()
    fig.savefig(out / "clearance.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-gap", type=float, default=30.0)
    args = parser.parse_args(argv)
    arrays, origin, step, objects = load(args.capture)
    summary, gap_rows, blocked, clearance, slope, inside = analyse(arrays, origin, step, objects, args.max_gap)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "gaps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_rows[0]) if gap_rows else ["a"])
        writer.writeheader()
        writer.writerows(gap_rows)
    marks = dict(summary["candidates"], _fence_gaps_all=summary["fence_lines"]["gaps"])
    draw(args.output, arrays, origin, step, objects, blocked, clearance, inside, gap_rows, marks)
    print(json.dumps({k: summary[k] for k in ("obstacles", "slopes", "gaps", "clearance_m")}, indent=2))
    print("fence gaps:", summary["fence_lines"]["gaps"][:6])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
