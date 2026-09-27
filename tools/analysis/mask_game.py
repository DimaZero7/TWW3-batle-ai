"""The stand mask read in battle vs the captured map (formation-probe run).

    python -m tools.analysis.mask_game <run dir> --map moorlands-route --out DIR

Compares every cell of the 'mask' event (is_area_clear + can_reach_position
in battle) with the same point on data/maps/<map>/grid-3m.npz, and draws both.
Output: summary.json, mask.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.sim.mapgrid import MapGrid  # noqa: E402


def cells(mask):
    g = mask["grid"]
    b = math.radians(g["frame"]["bearing"])
    fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
    o = g["frame"]["origin"]
    for r, row in enumerate(mask["code"].split("/")):
        along = g["along0"] + (r + 0.5) * g["step"]
        for c, ch in enumerate(row):
            across = g["across0"] + (c + 0.5) * g["step"]
            yield o["x"] + fx * along + rx * across, o["z"] + fz * along + rz * across, ch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    mask = next(r for r in rows if r["event"] == "mask")
    grid = MapGrid(args.map)
    both = {"agree_stand": 0, "agree_blocked": 0, "game_blocked_map_stand": 0, "game_stand_map_blocked": 0,
            "unknown": 0}
    game_blocked, differ = [], []
    for x, z, ch in cells(mask):
        if ch == "?":
            both["unknown"] += 1
            continue
        game = ch == "."
        clear, _ = grid.reader(x, z)
        if game and clear:
            both["agree_stand"] += 1
        elif not game and not clear:
            both["agree_blocked"] += 1
        elif game:
            both["game_stand_map_blocked"] += 1
            differ.append((x, z))
        else:
            both["game_blocked_map_stand"] += 1
            differ.append((x, z))
        if not game:
            game_blocked.append((x, z))
    known = sum(v for k, v in both.items() if k != "unknown")
    summary = {"summary": mask["summary"], "compare": both,
               "agreement": round((both["agree_stand"] + both["agree_blocked"]) / known, 4) if known else None,
               "lane": mask["lane"], "fits": mask["fits"], "clock_s": mask["clock_s"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 10))
    xs = [x for x, z, ch in cells(mask)]
    zs = [z for x, z, ch in cells(mask)]
    ax.scatter(xs, zs, s=2, c="#dddddd", marker="s", lw=0)
    if game_blocked:
        ax.scatter(*zip(*game_blocked), s=9, c="#404040", marker="s", lw=0, label="в игре нельзя встать")
    if differ:
        ax.scatter(*zip(*differ), s=14, facecolors="none", edgecolors="#d62728", lw=0.8, label="не совпало с картой")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set(xlabel="X, m", ylabel="Z, m",
           title=f'Маска 3 м в игре: нельзя встать {mask["summary"]["blocked"]} из {mask["summary"]["cells"]}; '
                 f'совпадение с картой {summary["agreement"] * 100:.1f}%\n'
                 f'снятие {mask["clock_s"]:.2f} с процессора; полоса вперёд '
                 + ("свободна" if mask["lane"]["free"] else f'занята с {mask["lane"]["first_blocked_along"]:.0f} м'))
    fig.tight_layout()
    fig.savefig(args.out / "mask.png", dpi=100)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
