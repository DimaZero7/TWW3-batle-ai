"""A step round an obstacle, walked in queues (tools/sim/logistics.py), to battle records.

For every approach step of the army that goes round an obstacle, apps.logistics
plans the queue (who goes which side, in which order and width) and its
dispatcher gives the orders tick by tick; simple walkers carry them out every
0.5 s. Two records per step, to be played side by side: with the queue and
with everybody released at once. Only our army: the enemy is far and plays no
part in the queue. Overlays: the obstacle's band and, per unit,
its way (round / straight / aside), side and start delay. The chart: soldiers
crowded (two units closer than 1 m) and soldiers on blocked cells.
"""
from tools.sim import formation as simulation
from tools.sim import logistics as walker
from tools.sim.mapgrid import MapGrid
from tools.viewer import record as rec
from tools.viewer import sim as simrec

TRACK_EVERY = 2  # a frame every second (two steps of 0.5 s)
KINDS = {"detour": rec.tr("в обход", "round"), "straight": rec.tr("прямо", "straight"),
         "aside": rec.tr("в стороне", "aside")}


def band_overlay(field, plan):
    """The obstacle's band: two lines across the field, and each unit's plan."""
    band = plan.get("band")
    lines = []
    if band:
        h = field["half_width_m"]
        lines = [[rec.in_frame(field["origin"], field["bearing"], a, -h), rec.in_frame(field["origin"], field["bearing"], a, h)]
                 for a in (band["along0"], band["along1"])]
    units = {}
    for uid, row in plan["units"].items():
        slot = row.get("slot") or {}
        side = None
        if row["kind"] == "detour" and slot.get("across") is not None:
            side = "left" if slot["across"] < 0 else "right"
        units[simrec.side_id("own", uid)] = {"kind": row["kind"], "side": side,
                                            "release_s": (row.get("release") or {}).get("at_s", 0)}
    return {"game_ms": 0, "kind": "logistics", "band": lines, "units": units}


def convert(army_name):
    """[(record with the queue, record all at once)] for every step round an obstacle."""
    army, name = simulation.load_army(army_name)
    planner = simulation.Planner()
    sides, own_by_id = simulation.simulate(army, planner)
    grid = MapGrid(army["map"])
    logistics = walker.Logistics(planner.lua)
    enemy_by_id = {u["id"]: u for u in simulation.roster_units(army["enemy"], simulation.roster.load_roster())}
    enemy = sides["enemy"]["placements"]
    units_all = rec.tag_units(simrec.units_of("own", own_by_id, sides["own"]["placements"])
                              + simrec.units_of("enemy", enemy_by_id, enemy))
    out = []
    for n, m in enumerate([m for m in (sides.get("approach") or {}).get("moves", []) if m["detour"]], start=1):
        pair = []
        mask = None  # the ground round the walk, once the frames are known (record_of)
        for queue in (True, False):
            summary, tracks, plan = walker.walk(logistics, m["field"], grid, m["before"], m["after"], own_by_id, queue,
                                                track_every=TRACK_EVERY)
            pair.append(record_of(name, n, queue, summary, tracks, plan, m, units_all, enemy, enemy_by_id, mask, grid))
        out.append(tuple(pair))
    return out


def record_of(name, n, queue, summary, tracks, plan, move, units_all, enemy, enemy_by_id, mask, grid):
    step_ms = walker.DT_S * TRACK_EVERY * 1000
    longest = max(len(t) for t in tracks.values())
    spec = {u["id"]: u for u in units_all}
    frames = []
    for k in range(longest):
        units = []
        for uid, tr in tracks.items():
            x, z, bearing, front_m, depth_m = tr[min(k, len(tr) - 1)]
            placement = {"x": x, "z": z, "bearing": bearing, "depth_m": depth_m}
            cx, cz = simrec.centre(placement)
            moving = k + 1 < len(tr) and tr[k + 1][:2] != tr[k][:2]
            uid = simrec.side_id("own", uid)
            units.append(rec.unit_row(uid, cx, cz, bearing, front_m, depth_m, spec[uid]["men_max"], moving))
        frames.append({"game_ms": round(k * step_ms), "units": units})
    ids = {u["id"] for f in frames[:1] for u in f["units"]}
    units = [u for u in units_all if u["id"] in ids]
    kinds = summary["kinds"]
    how = rec.tr("с очередью", "with the queue") if queue else rec.tr("все сразу", "all at once")
    events = [rec.stage_event(0, "logistics"),
              rec.note(0, f"в обход {kinds['detour']}, прямо {kinds['straight']}, в стороне {kinds['aside']}; "
                          f"сужаются: {len(summary['narrowed'])}",
                       f"round {kinds['detour']}, straight {kinds['straight']}, aside {kinds['aside']}; "
                       f"narrowing: {len(summary['narrowed'])}")]
    last = summary["last_in_place_s"]
    events.append(rec.note(round(last * 1000), f"последний на месте: {last:.0f} с, давка до {summary['crowded_max']} "
                                               f"бойцов ({summary['crowded_s']:.0f} с)",
                           f"the last in place: {last:.0f} s, up to {summary['crowded_max']} soldiers crowded "
                           f"({summary['crowded_s']:.0f} s)"))
    if queue:
        for uid, row in plan["units"].items():
            at = (row.get("release") or {}).get("at_s") or 0
            if at > 0:
                tag = spec[simrec.side_id("own", uid)]["tag"]
                way = KINDS.get(row["kind"], rec.tr(row["kind"], row["kind"]))
                events.append(rec.note(round(at * 1000), f"{tag['ru']} пошёл ({way['ru']})",
                                       f"{tag['en']} goes ({way['en']})"))
    ser = summary["series"]
    dt = ser["dt_s"] * 1000
    series = [{"name": rec.tr("давка, бойцов", "crowded soldiers"),
               "points": [[round(i * dt), v] for i, v in enumerate(ser["crowded"])]},
              {"name": rec.tr("на камнях, бойцов", "soldiers on blocked cells"),
               "points": [[round(i * dt), v] for i, v in enumerate(ser["on_blocked"])]}]
    meta = {"source": "logistics", "name": f"{name}-{n}-{'queue' if queue else 'at_once'}", "army": name,
            "title": rec._join([rec.tr(f"Обход препятствия · {name} · скачок {n} · ", f"Round the obstacle · {name} · step {n} · "), how])}
    overlays = [band_overlay(move["field"], plan), rec.field_overlay(0, move["field"])]
    mask = mask or rec.map_mask(grid, frames)
    return rec.make(meta, units, frames, events, {}, mask, idle_ms=0, overlays=overlays, series=series)
