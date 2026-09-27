"""Report of a move-probe run (src/entries/move_probe.lua).

    python -m tools.analysis.move_probe <events.jsonl> <output dir> [--grid arrays.npz]

--grid: arrays.npz from tools.analysis.passability (1 m reach mask); adds the
distance of every soldier to the nearest unreachable cell.

Output:
  summary.json   per leg: end reason, time, path, detour, front/centre offsets,
                 formation width and depth, closest soldier to the block, pauses
  shapes.png     final formation of every shape leg in the unit's own frame
  traverses.png  centre tracks and soldier snapshots over the reach mask

Frames: bearing 0 = +Z, 90 = +X. forward = (sin b, cos b), right = (cos b, -sin b).
"front" = the leading rank = centroid + max forward offset of the soldiers.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(r for r in rows if r["event"] == "start")
    legs = {leg["name"]: dict(leg, samples=[], ordered=None, end=None) for leg in start["plan"]["legs"]}
    for r in rows:
        leg = legs.get(r.get("leg"))
        if leg is None:
            continue
        if r["event"] == "move_sample":
            leg["samples"].append(r)
        elif r["event"] == "leg_ordered":
            leg["ordered"] = r
        elif r["event"] == "leg_end":
            leg["end"] = r
    ready = next(r for r in rows if r["event"] == "ready")
    return start, ready, [legs[leg["name"]] for leg in start["plan"]["legs"]]


def soldiers(row):
    return np.asarray(row["soldiers_dm"], dtype=float).reshape(-1, 2) / 10


def frame(bearing_deg):
    b = math.radians(bearing_deg)
    return np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])


def geometry(row, bearing=None):
    """Width/depth of the soldiers in the unit's own frame (or a given bearing)."""
    p = soldiers(row)
    fwd, right = frame(row["motion"]["bearing"] if bearing is None else bearing)
    centroid = p.mean(0)
    f, r = (p - centroid) @ fwd, (p - centroid) @ right
    return {"centroid": centroid, "front": centroid + f.max() * fwd,
            "width_m": float(r.max() - r.min()), "width_p95_m": float(np.percentile(r, 97.5) - np.percentile(r, 2.5)),
            "depth_m": float(f.max() - f.min()), "centre_to_centroid_m": float(np.hypot(*(centroid - [row["motion"]["x"], row["motion"]["z"]])))}


class Block:
    """Distance from a point to the nearest unreachable 1 m cell (NaN outside the grid)."""

    def __init__(self, path):
        from scipy import ndimage
        a = np.load(path)
        self.reach = a["reach"] == 1
        self.x0, self.z0 = float(a["x"][0, 0]) - 0.5, float(a["z"][0, 0]) - 0.5
        self.distance = ndimage.distance_transform_edt(self.reach)
        self.extent = (self.x0, self.x0 + self.reach.shape[1], self.z0, self.z0 + self.reach.shape[0])

    def distance_at(self, pts):
        ix = np.floor(pts[:, 0] - self.x0).astype(int)
        iz = np.floor(pts[:, 1] - self.z0).astype(int)
        ok = (ix >= 0) & (iz >= 0) & (ix < self.reach.shape[1]) & (iz < self.reach.shape[0])
        out = np.full(len(pts), np.nan)
        out[ok] = self.distance[iz[ok], ix[ok]]
        return out


def pauses(samples, window_ms=10000, move_m=2.0):
    """Longest run of samples where the centre moved < move_m over window_ms while is_moving."""
    t = np.array([s["t_ms"] for s in samples], dtype=float)
    xz = np.array([[s["motion"]["x"], s["motion"]["z"]] for s in samples])
    moving = np.array([s["motion"].get("is_moving") is True for s in samples])
    stuck_ms, worst_at = 0.0, None
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - window_ms, side="right") - 1
        if j >= 0 and moving[j:i + 1].all() and np.hypot(*(xz[i] - xz[j])) < move_m:
            if t[i] - t[j] > stuck_ms:
                stuck_ms, worst_at = t[i] - t[j], xz[i].round(1).tolist()
    return stuck_ms, worst_at


def leg_report(leg, block):
    s, end = leg["samples"], leg["end"]
    target = np.array([leg["target"]["x"], leg["target"]["z"]])
    start = np.array([leg["start"]["x"], leg["start"]["z"]])
    last = s[-1]
    g_last = geometry(last)
    ordered = leg["ordered"]["motion"]
    report = {
        "kind": leg["kind"], "run": leg["run"], "requested_width_m": leg["target"]["width"],
        "end_reason": end["reason"], "end_s": end["t_ms"] / 1000, "centre_path_m": round(end["path_m"], 1),
        "order_accepted": any(abs(x["motion"].get("ordered_x", 1e9) - target[0]) < 1
                              and abs(x["motion"].get("ordered_z", 1e9) - target[1]) < 1 for x in s),
        "ordered_width_read_m": last["motion"].get("ordered_width"),
        "final_width_m": round(g_last["width_m"], 1), "final_width_p95_m": round(g_last["width_p95_m"], 1),
        "final_depth_m": round(g_last["depth_m"], 1),
        "final_centre_to_target_m": round(float(np.hypot(*(np.array([last["motion"]["x"], last["motion"]["z"]]) - target))), 1),
        "final_front_to_target_m": round(float(np.hypot(*(g_last["front"] - target))), 1),
        "centre_vs_soldier_centroid_m": round(g_last["centre_to_centroid_m"], 1),
        "final_bearing_deg": round(last["motion"]["bearing"], 1),
        "moving_at_end": last["motion"].get("is_moving"),
    }
    # First time the leading rank is within 5 m of the target, and first time the unit stops after it.
    fronts = [geometry(x, leg["target"]["facing"])["front"] for x in s]
    near = [x["t_ms"] for x, f in zip(s, fronts) if np.hypot(*(f - target)) <= 5]
    report["front_within_5m_s"] = near[0] / 1000 if near else None
    stops = [x["t_ms"] for x in s if x["motion"].get("is_moving") is False and x["t_ms"] >= 3000]
    report["first_stop_s"] = stops[0] / 1000 if stops else None
    if leg["kind"] == "traverse":
        straight = float(np.hypot(*(target - start)))
        report["straight_m"] = round(straight, 1)
        report["detour_ratio"] = round(end["path_m"] / straight, 2) if straight else None
        speeds = [np.hypot(b["motion"]["x"] - a["motion"]["x"], b["motion"]["z"] - a["motion"]["z"]) /
                  max((b["t_ms"] - a["t_ms"]) / 1000, 1e-6) for a, b in zip(s, s[1:])]
        report["centre_speed_mps_median"] = round(float(np.median(speeds)), 2) if speeds else None
        stuck_ms, at = pauses(s)
        report["longest_no_progress_ms"], report["longest_no_progress_at"] = stuck_ms, at
        if block is not None:
            clearance = [block.distance_at(soldiers(x)) for x in s]
            mins = [np.nanmin(c) if np.isfinite(c).any() else np.nan for c in clearance]
            inside = [int(np.nansum(c == 0)) for c in clearance]
            centre = block.distance_at(np.array([[x["motion"]["x"], x["motion"]["z"]] for x in s]))
            report["closest_soldier_to_block_m"] = round(float(np.nanmin(mins)), 1) if np.isfinite(mins).any() else None
            report["max_soldiers_in_block_cells"] = int(max(inside))
            report["closest_centre_to_block_m"] = round(float(np.nanmin(centre)), 1) if np.isfinite(centre).any() else None
            # Which side of the block the centre passed: sign of the cross product at the closest approach.
            k = int(np.nanargmin(centre)) if np.isfinite(centre).any() else None
            if k is not None:
                d = target - start
                p = np.array([s[k]["motion"]["x"], s[k]["motion"]["z"]]) - start
                report["passed_block_on"] = "left" if d[0] * p[1] - d[1] * p[0] > 0 else "right"
    return report


def reform(leg):
    """Shape legs: how the formation changed from the order to the end."""
    before = geometry(leg["ordered"])
    return {"width_before_m": round(before["width_m"], 1), "depth_before_m": round(before["depth_m"], 1)}


def draw_shapes(out, legs):
    shapes = [leg for leg in legs if leg["kind"] == "shape"]
    fig, axes = plt.subplots(1, len(shapes), figsize=(3 * len(shapes), 5), sharey=True)
    for ax, leg in zip(np.atleast_1d(axes), shapes):
        last = leg["samples"][-1]
        p = soldiers(last)
        fwd, right = frame(last["motion"]["bearing"])
        c = p.mean(0)
        ax.scatter((p - c) @ right, (p - c) @ fwd, s=4)
        g = geometry(last)
        ax.set_title(f'{leg["name"]}\n{leg["end"]["reason"]} {leg["end"]["t_ms"] / 1000:.0f}s\n'
                     f'{g["width_m"]:.1f} x {g["depth_m"]:.1f} m', fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-35, 35)
        ax.set_ylim(-25, 25)
        ax.grid(alpha=0.3)
    fig.supxlabel("across the front, m")
    fig.supylabel("forward, m")
    fig.tight_layout()
    fig.savefig(out / "shapes.png", dpi=110)
    plt.close(fig)


def draw_traverses(out, legs, block):
    moves = [leg for leg in legs if leg["kind"] == "traverse"]
    cols = 4
    rows = math.ceil(len(moves) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for ax, leg in zip(axes.ravel(), moves):
        if block is not None:
            ax.imshow(~block.reach, origin="lower", extent=block.extent, cmap="Greys", vmin=0, vmax=1.5)
        s = leg["samples"]
        for x in s[::10]:
            p = soldiers(x)
            ax.scatter(p[:, 0], p[:, 1], s=1, c="#ff7f0e", alpha=0.5)
        ax.plot([x["motion"]["x"] for x in s], [x["motion"]["z"] for x in s], c="#1f77b4", lw=1.5)
        ax.scatter([leg["start"]["x"]], [leg["start"]["z"]], marker="o", c="green", zorder=4)
        ax.scatter([leg["target"]["x"]], [leg["target"]["z"]], marker="*", s=90, c="red", zorder=4)
        ax.set_title(f'{leg["name"]}: {leg["end"]["reason"]} {leg["end"]["t_ms"] / 1000:.0f}s, '
                     f'path {leg["end"]["path_m"]:.0f} m', fontsize=9)
        ax.set_xlim(-175, 5)
        ax.set_ylim(-160, 20)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(moves):]:
        ax.axis("off")
    fig.suptitle("grey: unreachable (1 m grid) · blue: unit centre · orange: soldiers every 10 s · star: order point")
    fig.tight_layout()
    fig.savefig(out / "traverses.png", dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid", type=Path)
    args = parser.parse_args(argv)
    start, ready, legs = load(args.events)
    block = Block(args.grid) if args.grid else None
    summary = {"build": start["build"], "speed": start["speed"], "plan": start["plan"]["name"],
               "unit": {"men": ready["men"], "slow_speed": ready["slow_speed"], "fast_speed": ready["fast_speed"]},
               "number_check": ready.get("number_check"), "legs": {}}
    for leg in legs:
        report = leg_report(leg, block)
        if leg["kind"] == "shape":
            report.update(reform(leg))
        summary["legs"][leg["name"]] = report
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_shapes(args.output, legs)
    draw_traverses(args.output, legs, block)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
