"""Passability of a captured window: is_area_clear vs can_reach_position, lanes.

    python -m tools.analysis.passability <grid.csv> <output dir> [--objects objects.json]
        [--focus MIN_X MAX_X MIN_Z MAX_Z] [--arrays PATH]

Input: tww3_bai_map_capture_grid.csv from a map-capture run with --features
(columns reach_side_1/2 = can_reach_position of each side's first unit).

Output:
  arrays.npz      clear, reach (side 1), height, ground codes, x/z of cell centres
                  (raw data: --arrays puts it in the local evidence archive)
  summary.json    agreement between "not clear" and "not reachable", lanes
  passability.png both masks, objects, lane cells coloured by free width
  lanes.png       zoom on the focus area with free width per reachable cell

"Free width" of a reachable cell = 2 x distance to the nearest unreachable cell
(a corridor this wide is centred on the cell), from a Euclidean distance transform.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

GROUNDS = ["forest", "grass", "mud", "sharp_stones", "shallow_water", "deep_water"]


def load_grid(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ix = np.array([int(r["ix"]) for r in rows])
    iz = np.array([int(r["iz"]) for r in rows])
    shape = (iz.max() + 1, ix.max() + 1)
    grid = {k: np.full(shape, -1, dtype=np.int8) for k in ("clear", "reach", "inside", "ground")}
    grid["height"] = np.full(shape, np.nan)
    grid["x"] = np.full(shape, np.nan)
    grid["z"] = np.full(shape, np.nan)
    for r, a, b in zip(rows, iz, ix):
        grid["clear"][a, b] = int(r["clear"])
        grid["reach"][a, b] = int(r.get("reach_side_1", -1))
        grid["inside"][a, b] = int(r["inside_radar"])
        g = r["ground"]
        grid["ground"][a, b] = GROUNDS.index(g) if g in GROUNDS else 99
        grid["height"][a, b] = float(r["height"])
        grid["x"][a, b] = float(r["x"])
        grid["z"][a, b] = float(r["z"])
    step = float(grid["x"][0, 1] - grid["x"][0, 0])
    return grid, step


def analyse(grid, step, focus=None):
    clear = grid["clear"] == 1
    reach = grid["reach"] == 1
    known = (grid["clear"] >= 0) & (grid["reach"] >= 0)
    blocked = ~reach & known
    free_width = 2 * ndimage.distance_transform_edt(reach) * step  # centre-to-edge doubled
    agreement = {
        "cells": int(known.sum()),
        "clear_and_reachable": int((clear & reach).sum()),
        "not_clear_and_not_reachable": int((~clear & ~reach & known).sum()),
        "clear_but_not_reachable": int((clear & ~reach & known).sum()),
        "not_clear_but_reachable": int((~clear & reach & known).sum()),
    }
    labels, count = ndimage.label(blocked, structure=np.ones((3, 3)))
    sizes = ndimage.sum(blocked, labels, range(1, count + 1)) if count else []
    summary = {"step_m": step, "agreement": agreement,
               "unreachable_clusters": int(count),
               "unreachable_cluster_sizes_m2": sorted([round(float(s) * step * step, 1) for s in sizes], reverse=True)[:15]}
    if focus:
        x0, x1, z0, z1 = focus
        in_focus = (grid["x"] >= x0) & (grid["x"] <= x1) & (grid["z"] >= z0) & (grid["z"] <= z1)
        lane = reach & in_focus & (free_width < 20)
        widths = free_width[lane]
        bins = [0, 3, 5, 8, 12, 20]
        hist = {f"{bins[i]}-{bins[i + 1]} m": int(((widths >= bins[i]) & (widths < bins[i + 1])).sum())
                for i in range(len(bins) - 1)}
        # Enclosed reachable pockets: reachable cells in focus whose free width is small AND
        # which lie inside the bounding box of unreachable cells (the built-up area).
        bz, bx = np.nonzero(blocked & in_focus)
        built = np.zeros_like(blocked)
        if len(bz):
            built[bz.min():bz.max() + 1, bx.min():bx.max() + 1] = True
        inner = reach & built
        summary["focus"] = {"window": focus,
                            "unreachable_cells": int((blocked & in_focus).sum()),
                            "built_up_bbox": ([round(float(grid["x"][0, bx.min()]), 1), round(float(grid["x"][0, bx.max()]), 1),
                                               round(float(grid["z"][bz.min(), 0]), 1), round(float(grid["z"][bz.max(), 0]), 1)]
                                              if len(bz) else None),
                            "reachable_cells_inside_built_up_bbox": int(inner.sum()),
                            "lane_cells_by_free_width": hist,
                            "narrowest_free_width_inside_built_up_m": (round(float(free_width[inner].min()), 1)
                                                                        if inner.any() else None)}
    return summary, blocked, free_width


def draw(out, grid, step, blocked, free_width, objects, focus):
    x0, x1 = float(np.nanmin(grid["x"])) - step / 2, float(np.nanmax(grid["x"])) + step / 2
    z0, z1 = float(np.nanmin(grid["z"])) - step / 2, float(np.nanmax(grid["z"])) + step / 2
    extent = (x0, x1, z0, z1)
    clear = grid["clear"] == 1
    reach = grid["reach"] == 1
    # 0 clear+reach, 1 not clear+not reach, 2 clear but unreachable, 3 not clear but reachable
    cls = np.zeros(blocked.shape)
    cls[~clear & ~reach] = 1
    cls[clear & ~reach] = 2
    cls[~clear & reach] = 3
    cmap = ListedColormap(["#f7f7f7", "#404040", "#d62728", "#1f77b4"])
    for name, zoom in (("passability.png", None), ("lanes.png", focus)):
        fig, ax = plt.subplots(figsize=(11, 12))
        ax.imshow(cls, origin="lower", extent=extent, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        lane = np.where(reach & (free_width < 20), free_width, np.nan)
        im = ax.imshow(lane, origin="lower", extent=extent, cmap="plasma_r", vmin=0, vmax=20,
                       alpha=0.8 if zoom else 0.5, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.7, label="free width of reachable cells under 20 m, m")
        for o in objects:
            if x0 <= o["x"] <= x1 and z0 <= o["z"] <= z1:
                house = o.get("category") == "house" and not o["name"].startswith(("wh3_glb_rubble", "glb_fence"))
                ax.scatter(o["x"], o["z"], s=18 if house else 6, c="#2ca02c" if house else "#7f7f7f",
                           marker="s" if house else "o", zorder=3)
                if house and zoom:
                    ax.annotate(o["name"].replace("emp_rural_", "house ").replace("wh3_glb_", ""),
                                (o["x"], o["z"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        if zoom:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
        ax.set(xlabel="X, m", ylabel="Z, m",
               title="dark: not clear & unreachable · red: clear but unreachable · blue: not clear but reachable\n"
                     "colour: free width of reachable cells (<20 m) · green squares: houses/rocks · grey: small objects")
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--arrays", type=Path, help="where to write arrays.npz (default: output dir)")
    parser.add_argument("--focus", type=float, nargs=4, metavar=("MIN_X", "MAX_X", "MIN_Z", "MAX_Z"))
    args = parser.parse_args(argv)
    grid, step = load_grid(args.grid)
    objects = json.loads(args.objects.read_text(encoding="utf-8")) if args.objects else []
    summary, blocked, free_width = analyse(grid, step, args.focus)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays or args.output / "arrays.npz", **{k: v for k, v in grid.items()}, free_width=free_width)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(args.output, grid, step, blocked, free_width, objects, args.focus)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
