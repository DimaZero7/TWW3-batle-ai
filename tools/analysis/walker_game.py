"""The simulator's walking against the game's (task 27): the step round an obstacle.

    python -m tools.analysis.walker_game [--out DIR]

For every army with a step round an obstacle and its game runs of 27.09.2026
(formation-probe, The Moorlands Route, x20): the same step in the game and in the
simulation, two ways — the engine alone (every unit ordered at once; in the
simulation tools/sim/walker.py) and our queues (apps.logistics; in the simulation
tools/sim/logistics.walk). Per unit: how much longer than the straight line its
centre walked, how close its centre came to blocked ground; per step: the time until
the last unit stood. Writes README.md (the table) and paths_<army>.png (game grey,
simulation coloured) to DIR (default research/analysis/walker/), summary.json beside.
"""
import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools import config as project  # noqa: E402
from tools.sim import formation as sim  # noqa: E402
from tools.sim import logistics as queues  # noqa: E402
from tools.sim import walker as walking  # noqa: E402
from tools.sim.mapgrid import MapGrid  # noqa: E402

ARMIES = ["rock_march_wide", "edge_march_wide", "gap_march_wide", "spear_march_wide", "skaven_march_wide"]
RUNS = project.BUILD / "formation-probe" / "runs"


def game_runs(army):
    """{'engine': run dir, 'queues': run dir} — the latest completed run of each way."""
    out = {}
    for d in sorted(glob.glob(str(RUNS / "2026092*"))):
        try:
            cfg = json.loads(Path(d, "manifest.json").read_text(encoding="utf-8"))["config"]
        except (OSError, KeyError, ValueError):
            continue
        if cfg.get("army") != army:
            continue
        rows = [json.loads(x) for x in Path(d, "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        if not any(r["event"] == "approach_decision" and (r.get("path") or {}).get("detour") for r in rows):
            continue
        way = "queues" if cfg.get("logistics") else "engine"
        out[way] = (d, rows)
    return out


def game_step(rows):
    """Centre tracks {unit: [(t_s, x, z)]} and the time of the step round the obstacle."""
    start = next(r for r in rows if r["event"] == "approach_decision" and (r.get("path") or {}).get("detour"))
    end = next(r for r in rows if r["event"] == "approach_manoeuvre" and r["model_ms"] > start["model_ms"])
    tracks = {}
    for r in rows:
        if r["event"] == "approach_sample" and start["model_ms"] <= r["model_ms"] <= end["model_ms"]:
            for u in r["units"]:
                m = u["motion"]
                if (m.get("number_of_men_alive") or 0) > 1:
                    tracks.setdefault(u["script_name"], []).append(((r["model_ms"] - start["model_ms"]) / 1000, m["x"], m["z"]))
    return tracks, end["t_ms"] / 1000


def centre_tracks(tracks):
    """Walker tracks of the front-rank centre (t, x, z, bearing, front, depth) or (x, z, bearing, front, depth)
    to centre tracks [(x, z)]."""
    out = {}
    for uid, tr in tracks.items():
        pts = []
        for s in tr:
            x, z, b, _, depth = s[-5:]
            pts.append((x - math.sin(math.radians(b)) * depth / 2, z - math.cos(math.radians(b)) * depth / 2))
        out[uid] = pts
    return out


def measures(tracks, terrain):
    """Per unit: walked / straight, and the closest its centre came to blocked ground (m)."""
    ratio, near = [], []
    for pts in tracks.values():
        pts = [p[-2:] for p in pts]
        walked = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        straight = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        if straight > 10:
            ratio.append(walked / straight)
        near.append(min(clearance(terrain, x, z) for x, z in pts))
    return {"units": len(tracks), "ratio_median": round(statistics.median(ratio), 3) if ratio else None,
            "ratio_max": round(max(ratio), 3) if ratio else None,
            "near_median_m": round(statistics.median(near), 1), "near_min_m": round(min(near), 1)}


def clearance(terrain, x, z, max_r=20):
    i, j = terrain.cell(x, z)
    if terrain.is_blocked(i, j):
        return 0.0
    for r in range(1, max_r + 1):
        for di in range(-r, r + 1):
            for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                if terrain.is_blocked(i + di, j + dj):
                    return round(math.hypot(di, dj) * terrain.step, 1)
    return float(max_r * terrain.step)


def run(out):
    report = []
    out.mkdir(parents=True, exist_ok=True)
    for army_name in ARMIES:
        games = game_runs(army_name)
        if not games:
            continue
        army, _ = sim.load_army(army_name)
        planner = sim.Planner()
        sides, by_id = sim.simulate(army, planner)
        grid = MapGrid(army["map"])
        terrain = walking.Terrain(grid)
        move = next(m for m in sides["approach"]["moves"] if m["detour"])
        simulated = {"engine": (centre_tracks(move["tracks"]), move["walk_s"])}
        summary, qtracks, _ = queues.walk(queues.Logistics(planner.lua), move["field"], grid, move["before"],
                                          move["after"], by_id, True, track_every=2)
        simulated["queues"] = (centre_tracks(qtracks), summary["last_in_place_s"])
        row = {"army": army_name}
        for way in ("engine", "queues"):
            if way not in games:
                continue
            gt, gs = game_step(games[way][1])
            st, ss = simulated[way]
            row[way] = {"run": Path(games[way][0]).name, "game_s": gs, "sim_s": round(ss, 1),
                        "time_error": round(ss / gs - 1, 3), "game": measures(gt, terrain), "sim": measures(st, terrain)}
            draw(out / f"paths_{army_name}_{way}.png", grid, gt, st, row[way], army_name, way)
        report.append(row)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def draw(path, grid, game, simulated, row, army, way):
    fig, ax = plt.subplots(figsize=(9, 9))
    pts = [{"x": p[-2], "z": p[-1]} for tr in list(game.values()) + list(simulated.values()) for p in tr]
    blocked = grid.blocked_near(pts, margin=30)
    if blocked:
        ax.scatter(*zip(*blocked), s=6, marker="s", c="#8c564b", alpha=0.45, lw=0)
    for tr in game.values():
        ax.plot([p[1] for p in tr], [p[2] for p in tr], c="#7f7f7f", lw=1.2, alpha=0.8)
    for tr in simulated.values():
        ax.plot([p[0] for p in tr], [p[1] for p in tr], c="#1f77b4", lw=1, alpha=0.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set(xlabel="X, m", ylabel="Z, m")
    how = "только движок" if way == "engine" else "очереди (logistics)"
    ax.set_title(f"{army}, шаг в обход, {how}: серые — игра ({row['game_s']:.0f} с), синие — симуляция "
                 f"({row['sim_s']:.0f} с)\nпуть длиннее прямой (медиана): игра {row['game']['ratio_median']}, "
                 f"симуляция {row['sim']['ratio_median']}; ближе всего к камню: игра {row['game']['near_min_m']} м, "
                 f"симуляция {row['sim']['near_min_m']} м", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=project.ROOT / "research" / "analysis" / "walker")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    for row in run(args.out):
        for way in ("engine", "queues"):
            r = row.get(way)
            if r:
                print(f"{row['army']:18} {way:7} time game {r['game_s']:5.0f} s, sim {r['sim_s']:5.0f} s ({r['time_error']:+.0%}); "
                      f"longer than straight: game {r['game']['ratio_median']}, sim {r['sim']['ratio_median']}; "
                      f"closest to rock: game {r['game']['near_min_m']} m, sim {r['sim']['near_min_m']} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
