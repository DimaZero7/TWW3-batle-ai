"""Queues past an obstacle in the simulation (src/apps/logistics).

    python -m tools.sim.logistics <army config> [--out DIR]

The army goes through tools/sim/formation.py (plan, alignment, approach). For
every approach step that walks round an obstacle, src/apps/logistics plans the
queue on the mask read from the captured map (who goes which side, in which
order and width) and its dispatcher gives the orders tick by tick, exactly as
in battle. tools/sim/walker.py carries the orders out as the engine would: the
front rank centre walks at 1.5 m/s round obstacles, the block faces its way and
turns to the ordered facing at the place, soldiers squeeze off blocked ground,
and a block stands still for the measured reform time when it narrows. The same plan is
also walked with everybody released at once (no queue) for comparison.

Measured: the time the last unit stands in its place, crowding (soldiers of two
units closer than 1 m, src/apps/logistics.crowding) and soldiers on blocked
cells. Output (default research/analysis/logistics/<army>/): summary.json,
walk.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from tools import config as project  # noqa: E402
from tools.sim import formation as sim  # noqa: E402
from tools.sim import walker as walking  # noqa: E402
from tools.sim.mapgrid import MapGrid  # noqa: E402

DT_S = 0.5  # tools/sim/walker.DT_S
MAX_S = 900


class Logistics:
    """The Lua side: mask, band, plan, dispatcher; the same calls as in battle."""

    def __init__(self, lua):
        self.lua = lua
        self._prepare = lua.eval("""
            function(field, reader, before, after, shapes, queue, params)
                local m = require('apps.mask.services')
                local L = require('apps.logistics.services')
                local mask = m.new(m.grid(field))
                m.fill(mask, reader)
                local units = L.units_from(field, before, after, shapes)
                local e = L.extent(units)
                local band = L.band(mask, e.from, e.to, nil, e.left, e.right)
                local plan = L.plan(units, band, params)
                if not queue then
                    for _, row in pairs(plan.units) do row.release = {at_s = 0, after = {}} end
                end
                local depths = {}
                for _, u in ipairs(units) do depths[u.id] = u.depth_m end
                plan.check = L.verify(units, plan)
                return plan, L.new_dispatch(plan, depths), require('apps.core.json').encode(plan)
            end
        """)
        self._crowding = lua.eval("""
            function(units) return require('apps.core.json').encode(require('apps.logistics.services').crowding(units)) end
        """)

    def prepare(self, field, grid, before, after, shapes, queue, params=None):
        t = self.lua.table_from
        plan, dispatch, text = self._prepare(t(field, recursive=True), grid.reader, t(before, recursive=True),
                                             t(after, recursive=True), t(shapes, recursive=True), queue,
                                             t(params or {}, recursive=True))
        return json.loads(text), dispatch

    def crowding(self, units):
        return json.loads(self._crowding(self.lua.table_from(units, recursive=True)))


def _axes(bearing):
    b = math.radians(bearing)
    return (math.sin(b), math.cos(b)), (math.cos(b), -math.sin(b))


def to_world(field, along, across):
    f, r = _axes(field["bearing"])
    o = field["origin"]
    return o["x"] + f[0] * along + r[0] * across, o["z"] + f[1] * along + r[1] * across


def to_frame(field, x, z):
    f, r = _axes(field["bearing"])
    dx, dz = x - field["origin"]["x"], z - field["origin"]["z"]
    return dx * f[0] + dz * f[1], dx * r[0] + dz * r[1]


def walk(logistics, field, grid, before, after, units_by_id, queue=True, speed=walking.SPEED_MPS, params=None,
         track_every=4):
    """Walks one manoeuvre by the dispatcher's orders on tools/sim/walker.py (the engine's walking:
    round obstacles, facing the way, soldiers squeezed off blocked ground). Returns (summary, tracks,
    plan); a track point (x, z, bearing, front_m, depth_m of the front-rank centre) every track_every
    steps of DT_S."""
    shapes = {p["id"]: units_by_id[p["id"]]["shapes"] for p in after}
    plan, dispatch = logistics.prepare(field, grid, before, after, shapes, queue, params)
    was = {p["id"]: p for p in before}
    goal = {p["id"]: p for p in after}
    terrain = walking.Terrain(grid)
    squeezer = walking.Squeezer(terrain)
    walker = walking.Walker(terrain, speed=speed, dt=DT_S)
    for p in after:
        b = was[p["id"]]
        walker.add(p["id"], b["x"], b["z"], b["bearing"], b["front_m"], b["depth_m"])
    ordered, done_s, tracks = set(), {}, {p["id"]: [] for p in after}
    t, lua, step = 0.0, logistics.lua, 0
    crowd_series, rock_series, worst_pair, squeezed_max = [], [], {}, 0.0
    while t <= MAX_S:
        positions = {}
        for uid in tracks:
            u = walker.state(uid)
            b = math.radians(u["bearing"])
            along, across = to_frame(field, u["x"] - math.sin(b) * u["depth_m"] / 2, u["z"] - math.cos(b) * u["depth_m"] / 2)
            positions[uid] = {"along": along, "across": across, "still": uid in done_s}
        for o in dispatch.update(t, lua.table_from(positions, recursive=True)).values():
            u = walker.state(o.id)
            if o.final:
                g = goal[plan["units"][o.id].get("place") or o.id]
                front_m, depth_m, bearing = g["front_m"], g["depth_m"], g["bearing"]
            else:
                slot = plan["units"][o.id]["slot"]
                front_m, depth_m = slot["front_m"], slot["depth_m"]
                bearing = field["bearing"] + (o.heading_deg or 0)
            pause = (plan["units"][o.id]["slot"].get("reform_s") or 0) if front_m < u["front_m"] - 1e-6 else 0
            x, z = to_world(field, o.along, o.across)
            walker.order(o.id, x, z, bearing, front_m, depth_m, pause)
            ordered.add(o.id)
            if o.final:
                done_s.pop(o.id, None)
        walker.step()
        step += 1
        # Soldiers in the world for crowding and blocked cells, squeezed off the obstacle as the engine's.
        units, on_rock = [], 0
        for uid in tracks:
            u = walker.state(uid)
            if u["arrived"] and uid in ordered and uid not in done_s and dispatch.state(uid).leg == len(plan["units"][uid]["route"]):
                done_s[uid] = t
            rect = {"x": u["x"], "z": u["z"], "bearing": u["bearing"], "front_m": u["front_m"], "depth_m": u["depth_m"]}
            men = int(units_by_id[uid]["men"])
            pts, moved = squeezer.squeeze(sim.soldier_points(rect, men))
            squeezed_max = max(squeezed_max, moved / max(men, 1))
            units.append({"id": uid, "points": pts})
            for i in range(0, len(pts), 2):
                if not grid.reader(pts[i], pts[i + 1])[0]:
                    on_rock += 1
            if step % track_every == 1 or track_every == 1:
                tracks[uid].append((u["x"], u["z"], u["bearing"], u["front_m"], u["depth_m"]))
        crowd = logistics.crowding(units)
        crowd_series.append(crowd["soldiers"])
        rock_series.append(on_rock)
        for k, n in crowd["pairs"].items():
            worst_pair[k] = max(worst_pair.get(k, 0), n)
        if len(done_s) == len(tracks) and dispatch.done():
            break
        t += DT_S
    waits = [plan["units"][u]["release"]["at_s"] for u in plan["units"]]
    summary = {"queue": queue, "last_in_place_s": max(done_s.get(u, MAX_S) for u in tracks),
               "planned_s": plan["makespan_s"], "crowded_max": max(crowd_series),
               "crowded_s": sum(DT_S for c in crowd_series if c > 0), "on_blocked_max": max(rock_series),
               "squeezed_max_share": round(squeezed_max, 3),
               "longest_planned_wait_s": max(waits) if queue else 0,
               "check_s": plan["check"]["seconds"], "check_pairs": plan["check"]["pairs"],
               "kinds": {k: sum(1 for r in plan["units"].values() if r["kind"] == k) for k in ("detour", "straight", "aside")},
               "narrowed": sorted(u for u, r in plan["units"].items() if r.get("slot") and r["slot"]["front_m"] < goal[u]["front_m"] - 0.5),
               "worst_pairs": dict(sorted(worst_pair.items(), key=lambda kv: -kv[1])[:5]),
               "log": [dict(e.items()) for e in dispatch.log.values()],
               # Over time, a value every DT_S: soldiers crowded, soldiers on blocked cells.
               "series": {"dt_s": DT_S, "crowded": crowd_series, "on_blocked": rock_series}}
    return summary, tracks, plan


def draw(path, field, grid, plan, tracks, runs, before, after, title):
    fig, ax = plt.subplots(figsize=(10, 11))
    pts = before + after
    blocked = grid.blocked_near(pts, margin=60)
    if blocked:
        ax.scatter(*zip(*blocked), s=8, marker="s", c="#8c564b", alpha=0.45, lw=0)
    band = plan["band"]
    if band:
        for a in (band["along0"], band["along1"]):
            h = 2 * field["half_width_m"]
            xs, zs = zip(*(to_world(field, a, c) for c in (-h / 2, h / 2)))
            ax.plot(xs, zs, c="#9467bd", lw=0.8, ls="--")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for k, uid in enumerate(sorted(tracks)):
        tr = tracks[uid]
        ax.plot([p[0] for p in tr], [p[1] for p in tr], lw=1, c=colors[k % 20])
        row = plan["units"][uid]
        if row["kind"] == "detour" or row["release"]["at_s"] > 0:
            x, z = tr[0][0], tr[0][1]
            ax.annotate(f'{uid.split("_")[-1]}: +{row["release"]["at_s"]:.0f} с', (x, z), fontsize=7,
                        xytext=(0, -10), textcoords="offset points")
    for p in before:
        ax.add_patch(Polygon(sim.corners(p), closed=True, fill=False, ec="#7f7f7f", lw=0.6, ls="--"))
    for p in after:
        ax.add_patch(Polygon(sim.corners(p), closed=True, fc=sim.COLORS.get(p["role"], "#7f7f7f"), ec="k", lw=0.4,
                             alpha=0.5))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set(xlabel="X, m", ylabel="Z, m")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(army, out, params=None):
    planner = sim.Planner()
    sides, units_by_id = sim.simulate(army, planner)
    grid = MapGrid(army["map"])
    logistics = Logistics(planner.lua)
    moves = [m for m in (sides.get("approach") or {}).get("moves", []) if m["detour"]]
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for n, m in enumerate(moves, start=1):
        runs = {}
        for queue in (True, False):
            summary, tracks, plan = walk(logistics, m["field"], grid, m["before"], m["after"], units_by_id, queue,
                                         params=params)
            runs["queue" if queue else "at_once"] = (summary, tracks, plan)
        q, a = runs["queue"][0], runs["at_once"][0]
        plan = runs["queue"][2]
        sides_used = {}
        for uid, row in plan["units"].items():
            if row["kind"] == "detour":
                side = "лево" if row["slot"]["across"] < 0 else "право"
                sides_used[side] = sides_used.get(side, 0) + 1
        title = (f'Очередь у препятствия: в обход {q["kinds"]["detour"]}, прямо через полосу {q["kinds"]["straight"]}, '
                 f'в стороне {q["kinds"]["aside"]}; стороны: {sides_used}; сужались: {len(q["narrowed"])}\n'
                 f'с очередью: последний на месте через {q["last_in_place_s"]:.0f} с (план {q["planned_s"]:.0f}), '
                 f'давка макс. {q["crowded_max"]} бойцов, {q["crowded_s"]:.0f} с\n'
                 f'все сразу: {a["last_in_place_s"]:.0f} с, давка макс. {a["crowded_max"]} бойцов, {a["crowded_s"]:.0f} с\n'
                 f'подписи — задержка старта; пунктир — полоса препятствия (+5 м)')
        draw(out / f"walk_{n}.png", m["field"], grid, plan, runs["queue"][1], runs, m["before"], m["after"], title)
        report.append({"queue": q, "at_once": a, "plan": plan})
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("army")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    army, name = sim.load_army(args.army)
    out = args.out or project.ROOT / "research" / "analysis" / "logistics" / name
    report = run(army, out)
    for r in report:
        print(json.dumps({k: {x: r[k][x] for x in ("last_in_place_s", "planned_s", "crowded_max", "crowded_s", "check_s",
                                                  "on_blocked_max", "kinds", "narrowed", "longest_planned_wait_s")}
                          for k in ("queue", "at_once")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
