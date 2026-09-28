"""Report of a formation-probe run (src/entries/formation_probe.lua).

    python -m tools.analysis.formation_probe <events.jsonl> <output dir>

summary.json  plan vs battle for every unit after placing (front centre error,
              front and depth, bearing), closest soldiers of different units at
              every stage, and while the archers turn
stages.png    planned rectangles and real soldiers at every stage
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
from matplotlib.patches import Polygon  # noqa: E402

from tools.analysis.move_probe import soldiers  # noqa: E402
from tools.sim.formation import COLORS, corners  # noqa: E402


def frame(bearing):
    b = math.radians(bearing)
    return np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])


def measured(unit, bearing):
    """Front centre, front and depth of the real soldiers in the planned frame."""
    p = soldiers(unit)
    fwd, right = frame(bearing)
    c = p.mean(0)
    f, r = (p - c) @ fwd, (p - c) @ right
    front_centre = c + f.max() * fwd + ((r.max() + r.min()) / 2) * right
    return {"front_centre": front_centre, "front_m": float(r.max() - r.min()), "depth_m": float(f.max() - f.min())}


def closest(units):
    """Closest soldiers of two different units: metres and the pair."""
    best = (math.inf, None)
    pts = [(u["script_name"], soldiers(u)) for u in units if u.get("soldiers_dm")]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i][1], pts[j][1]
            d = np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]))
            if d < best[0]:
                best = (float(d), [pts[i][0], pts[j][0]])
    return {"closest_m": round(best[0], 2), "pair": best[1]}


# Without an enemy assessment the army must stand still after placing.
# Judged by the soldiers: unit:position() jumped 2.1 m while no soldier moved
# (measured 27.09.2026), so the native position is only reported.
STABLE_MIDDLE_M = 0.5
STABLE_ORDER_M = 0.5


def stability(rows):
    """Hold stage: soldiers' middle before/after, order point, is_moving ticks, orders we gave."""
    samples = [r for r in rows if r["event"] == "hold_sample"]
    ordered = [r for r in rows if r["event"] == "stage_snapshot"]
    names = [r["stage"] for r in ordered]
    if not samples or "hold" not in names or names.index("hold") == 0:
        return None
    # From the start of the hold: the snapshot of the stage before it (placed, or
    # align when the army was aligned first), to the end of the hold.
    start = ordered[names.index("hold") - 1]
    before = {u["script_name"]: u for u in start["units"]}
    after = {u["script_name"]: u for u in ordered[names.index("hold")]["units"]}
    first = {u["script_name"]: u["motion"] for u in samples[0]["units"]}
    units = {}
    for name, m0 in first.items():
        order, moving, native = 0.0, 0, 0.0
        for s in samples:
            m = next(u["motion"] for u in s["units"] if u["script_name"] == name)
            native = max(native, math.hypot(m["x"] - m0["x"], m["z"] - m0["z"]))
            if "ordered_x" in m and "ordered_x" in m0:
                order = max(order, math.hypot(m["ordered_x"] - m0["ordered_x"], m["ordered_z"] - m0["ordered_z"]))
            moving += m.get("is_moving") is True
        a, b = soldiers(before[name]), soldiers(after[name])
        each = np.hypot(*(b - a).T) if len(a) == len(b) else np.array([math.nan])
        units[name] = {"soldiers_middle_moved_m": round(float(np.hypot(*(b.mean(0) - a.mean(0)))), 2),
                       "soldier_moved_max_m": round(float(np.nanmax(each)), 2),
                       "order_point_moved_m": round(order, 2), "moving_ticks": moving,
                       "native_position_jump_m": round(native, 2)}
    orders = samples[-1]["orders_after_placed"]
    passed = orders == 0 and all(u["soldiers_middle_moved_m"] <= STABLE_MIDDLE_M and u["moving_ticks"] == 0
                                 and u["order_point_moved_m"] <= STABLE_ORDER_M for u in units.values())
    return {"verdict": "PASS" if passed else "FAIL", "hold_s": samples[-1]["t_ms"] / 1000,
            "orders_after_placed": orders,
            "limits": {"soldiers_middle_m": STABLE_MIDDLE_M, "order_point_m": STABLE_ORDER_M, "moving_ticks": 0},
            "units": units}


def lord_passage(snapshot, placements):
    """The passage of the first archer row in battle: planned and real width between the
    soldiers of the blocks either side of the lord, and how far the lord stands off its middle."""
    by = {u["script_name"]: u for u in snapshot["units"]}
    lord = next((p for p in placements.values() if p["role"] == "lord"), None)
    first = [p for p in placements.values() if p["role"] == "arc" and p.get("row") == 1 and p["id"] in by]
    if not lord or lord["id"] not in by or not first:
        return None
    _, right = frame(lord["bearing"])
    origin = np.array([lord["x"], lord["z"]])
    left = [p for p in first if p["along_m"] < lord["along_m"]]
    rightside = [p for p in first if p["along_m"] > lord["along_m"]]
    if not rightside:
        return None
    near_r = min(rightside, key=lambda p: p["along_m"])
    near_l = max(left, key=lambda p: p["along_m"]) if left else None
    r_edge = float(((soldiers(by[near_r["id"]]) - origin) @ right).min())
    planned_r = near_r["along_m"] - near_r["front_m"] / 2 - lord["along_m"]
    if near_l:
        l_edge = float(((soldiers(by[near_l["id"]]) - origin) @ right).max())
        planned_l = near_l["along_m"] + near_l["front_m"] / 2 - lord["along_m"]
    else:  # nothing on the left: the passage is measured as symmetric
        l_edge, planned_l = -r_edge, -planned_r
    lord_off = float((soldiers(by[lord["id"]]).mean(0) - origin) @ right)
    return {"planned_m": round(planned_r - planned_l, 1), "real_m": round(r_edge - l_edge, 1),
            "lord_off_middle_m": round(lord_off - (r_edge + l_edge) / 2, 1),
            "blocks": [near_l and near_l["id"], near_r["id"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    plan_row = next(r for r in rows if r["event"] == "plan")
    plan = plan_row["plan"]
    placements = {p["id"]: p for p in plan["placements"]}
    snapshots = [r for r in rows if r["event"] == "stage_snapshot"]
    turns = [r for r in rows if r["event"] == "turn_sample"]

    placed = next(s for s in snapshots if s["stage"] == "placed")

    def placement_errors(snapshot, by_id):
        units = {}
        for u in snapshot["units"]:
            p = by_id[u["script_name"]]
            m = measured(u, p["bearing"])
            units[u["script_name"]] = {
                "role": p["role"],
                "front_centre_error_m": round(float(np.hypot(*(m["front_centre"] - [p["x"], p["z"]]))), 1),
                "planned_front_x_depth_m": [round(p["front_m"], 1), round(p["depth_m"], 1)],
                "real_front_x_depth_m": [round(m["front_m"], 1), round(m["depth_m"], 1)],
                "bearing_error_deg": round(((u["motion"]["bearing"] - p["bearing"] + 180) % 360) - 180, 1),
            }
        return units

    # After an alignment the army walks to a new plan; stages after it are checked against that one.
    aligned_row = next((r for r in rows if r["event"] == "alignment" and r.get("plan")), None)
    aligned = {p["id"]: p for p in aligned_row["plan"]["placements"]} if aligned_row else None
    after_row = next((r for r in rows if r["event"] == "alignment_after"), None)
    alignment_summary = None
    if aligned_row or after_row:
        align_snap = next((s for s in snapshots if s["stage"] == "align"), None)
        alignment_summary = {
            "before": aligned_row and {k: aligned_row["check"][k] for k in ("angle_off_deg", "offset_m", "reasons")},
            "after": after_row and {k: after_row["check"][k] for k in ("angle_off_deg", "offset_m", "needed")},
            "orders": after_row and after_row["orders"], "walk_s": align_snap and align_snap["t_ms"] / 1000,
            "overhang_after": after_row and after_row["overhang"],
            "units_after_walking": placement_errors(align_snap, aligned) if (aligned and align_snap) else None,
            "governor": [r["decision"] for r in rows if r["event"] == "governor"]}
    units = {}
    for u in placed["units"]:
        p = placements[u["script_name"]]
        m = measured(u, p["bearing"])
        units[u["script_name"]] = {
            "role": p["role"],
            "front_centre_error_m": round(float(np.hypot(*(m["front_centre"] - [p["x"], p["z"]]))), 1),
            "planned_front_x_depth_m": [round(p["front_m"], 1), round(p["depth_m"], 1)],
            "real_front_x_depth_m": [round(m["front_m"], 1), round(m["depth_m"], 1)],
            "bearing_error_deg": round(((u["motion"]["bearing"] - p["bearing"] + 180) % 360) - 180, 1),
        }
    middles = {u["script_name"]: soldiers(u).mean(0) for u in placed["units"] if u.get("soldiers_dm")}
    archers = [n for n, p in placements.items() if p["role"] == "arc"]

    def stage_row(s):
        by = {u["script_name"]: u for u in s["units"]}
        wall = [u for u in s["units"] if placements[u["script_name"]]["role"] == "wall"]
        shift = [float(np.hypot(*(soldiers(by[n]).mean(0) - middles[n]))) for n in archers if n in by]
        return dict(closest(s["units"]), settled=s["settled"], t_s=s["t_ms"] / 1000,
                    archers_to_archers=closest([by[n] for n in archers if n in by])["closest_m"],
                    archers_to_wall=closest(wall + [by[n] for n in archers if n in by])["closest_m"]
                    if wall else None,
                    archer_middle_shift_m=round(max(shift), 1) if shift else None)
    stages = {s["stage"]: stage_row(s) for s in snapshots}
    during = {}
    for t in turns:
        c = closest(t["units"])
        if t["stage"] not in during or c["closest_m"] < during[t["stage"]]["closest_m"]:
            during[t["stage"]] = dict(c, t_s=t["t_ms"] / 1000)
    summary = {"facing_source": plan_row["facing_source"], "enemies_seen": plan_row["enemies_seen"],
               "bearing": round(plan_row["bearing"], 1), "status": plan["status"], "choice": plan["choice"],
               "units_after_placing": units, "lord_passage": lord_passage(placed, placements),
               "alignment": alignment_summary, "stability": stability(rows),
               "stages": stages,
               "archers_while_turning": during}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.2 * len(snapshots), 5.5))
    for ax, snap in zip(np.atleast_1d(axes), snapshots):
        shown = aligned if (aligned and snap["stage"] != "placed") else placements
        for p in shown.values():
            ax.add_patch(Polygon(corners(p), closed=True, fill=False, ec=COLORS[p["role"]], lw=1, ls="--"))
        for u in snap["units"]:
            if u.get("soldiers_dm"):
                pts = soldiers(u)
                ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c=COLORS[placements[u["script_name"]]["role"]])
        c = stages[snap["stage"]]
        ax.set_title(f'{snap["stage"]}\nлучники–стена {c["archers_to_wall"]} м, сдвиг {c["archer_middle_shift_m"]} м',
                     fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("пунктир — план (после выравнивания — новый план), точки — бойцы в игре")
    fig.tight_layout()
    fig.savefig(args.output / "stages.png", dpi=100)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
