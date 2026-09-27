"""Picture and stability of the battlefield built in battle (enemy-layout run).

    python -m tools.analysis.battlefield_game <run dir> --out DIR

Draws the soldiers of both sides at the end of the hold with the battlefield
the AI modules built from our side's view (apps.vision + apps.battlefield),
and tabulates how the field moved over the hold.
Output: game.png, summary.json.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    fields = [r for r in rows if r["event"] == "battlefield" and r.get("field")]
    enemy = next(r for r in rows if r["event"] == "enemy_snapshot" and r["stage"] == "end")
    own = next(r for r in rows if r["event"] == "own_snapshot")
    last = fields[-1]
    f = last["field"]

    series = [{"t_s": r["t_ms"] / 1000, "bearing": round(r["field"]["bearing"], 1),
               "centres_m": round(r["field"]["centres_m"], 1), "gap_m": round(r["field"]["gap_m"], 1),
               "half_width_m": round(r["field"]["half_width_m"], 1), "own_groups": len(r["own"]["groups"]),
               "enemy_groups": len(r["enemy"]["groups"]), "seen": r["seen"], "clock_s": r["clock_s"]} for r in fields]
    settled = [s for s in series if s["t_s"] >= 30]
    summary = {"layout": next(r["layout"] for r in rows if "layout" in r), "samples": series,
               "after_30s": {k: [min(s[k] for s in settled), max(s[k] for s in settled)]
                             for k in ("bearing", "gap_m", "centres_m", "half_width_m")} if settled else None,
               "clock_s_max": max(s["clock_s"] for s in series if isinstance(s["clock_s"], (int, float)))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 11))
    for u in own["units"]:
        if u.get("soldiers_dm"):
            p = soldiers(u)
            ax.scatter(p[:, 0], p[:, 1], s=1, c="#1f77b4")
    for u in enemy["units"]:
        if u.get("soldiers_dm"):
            p = soldiers(u)
            ax.scatter(p[:, 0], p[:, 1], s=1, c="#d62728")
    ax.add_patch(Polygon([(c["x"], c["z"]) for c in f["corners"]], closed=True, fill=False, ec="#9467bd", lw=1.2,
                         ls="-."))
    b = math.radians(f["bearing"])
    fw, r = np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])
    o = np.array([f["origin"]["x"], f["origin"]["z"]])
    for along in (f["own"]["front_m"], f["enemy"]["front_m"]):
        a, z = o + fw * along - r * f["half_width_m"], o + fw * along + r * f["half_width_m"]
        ax.plot([a[0], z[0]], [a[1], z[1]], c="#9467bd", lw=0.7)
    a, z = o + fw * f["own"]["back_m"], o + fw * f["enemy"]["back_m"]
    ax.plot([a[0], z[0]], [a[1], z[1]], c="#9467bd", lw=0.8, ls="--")
    for side, color in (("own", "#1f77b4"), ("enemy", "#d62728")):
        c = last[side]["groups"][0]["centre"]
        ax.scatter([c["x"]], [c["z"]], marker="X", s=140, c=color, ec="k", zorder=6)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set(xlabel="X, m", ylabel="Z, m",
           title=f'В игре ({summary["layout"]}, конец стояния): синие — мы, красные — штатный ИИ в обороне\n'
                 f'поле боя: между передними линиями {f["gap_m"]:.0f} м, ширина {2 * f["half_width_m"]:.0f} м, '
                 f'ось {f["bearing"]:.0f}°; расчёт {summary["clock_s_max"] * 1000:.0f} мс')
    fig.tight_layout()
    fig.savefig(args.out / "game.png", dpi=100)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
