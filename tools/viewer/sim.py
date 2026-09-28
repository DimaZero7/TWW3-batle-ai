"""A simulated battle (tools/sim/formation.py) to a battle record.

Every manoeuvre is walked in time as the engine would (tools/sim/walker.py:
round obstacles, facing the way): a frame a second from its tracks. The enemy stands
still (as the game's AI did in the test battle, tools/sim/enemy.py). The
alignment at deployment takes ALIGN_S; the exchange of fire (config fire_s)
is a frame a second from apps.missile.

What the modules saw is kept as overlays at every decision: the battlefield
(apps.battlefield), the groups of both sides (apps.vision), the alignment
(apps.alignment), the lord's routes (apps.formation).
"""
import math

from tools.sim import formation as simulation
from tools.sim.mapgrid import MapGrid
from tools.viewer import record as rec

ALIGN_S = 30  # the simulated alignment's walking time (tools/sim/formation.approach)


def centre(p):
    """Placements hold the front rank's centre; the record holds the block's centre."""
    b = math.radians(p["bearing"])
    return p["x"] - math.sin(b) * p["depth_m"] / 2, p["z"] - math.cos(b) * p["depth_m"] / 2


def rows(placements, side, by_id, men=None, moving=False, ammo=None):
    out = []
    for p in placements:
        u = by_id[p["id"]]
        full = u.get("men") or 1
        now = full if men is None else men.get(f"{side}:{p['id']}", full)
        x, z = centre(p)
        depth = p["depth_m"] * (now / full if full > 1 else 1)
        out.append(rec.unit_row(f"{side}:{p['id']}", x, z, p["bearing"], p["front_m"], max(depth, 1.0), now, moving,
                                None if ammo is None else ammo.get(f"{side}:{p['id']}")))
    return out


def units_of(side, by_id, placements):
    roles = {p["id"]: p.get("role") for p in placements}
    return [rec.unit_spec(f"{side}:{uid}", side, uid,
                          roles.get(uid) or ("lord" if u.get("commanding") and u.get("men") == 1 else "wall"),
                          u.get("class"), u.get("key"), u.get("men"), u.get("range_m")) for uid, u in by_id.items()]


def side_id(side, uid):
    return f"{side}:{uid}"


def convert(army_name, planner=None):
    army, name = simulation.load_army(army_name)
    planner = planner or simulation.Planner()
    sides, own_by_id = simulation.simulate(army, planner)
    enemy_by_id = {u["id"]: u for u in simulation.roster_units(army["enemy"], simulation.roster.load_roster())}
    enemy = sides["enemy"]["placements"]
    own_final = sides["own"]["placements"]
    units = units_of("own", own_by_id, own_final) + units_of("enemy", enemy_by_id, enemy)
    frames, events, overlays = [], [], []

    def frame(ms, own, moving=False, men=None, ammo=None):
        frames.append({"game_ms": round(ms), "units": rows(own, "own", own_by_id, men, moving, ammo)
                       + rows(enemy, "enemy", enemy_by_id, men)})

    def walked(t0, move):
        """The manoeuvre's tracks (front-rank centres, a sample a second) as frames."""
        tracks = move.get("tracks") or {}
        if not tracks:
            frame(t0, move["before"], moving=True)
            frame(t0 + move["walk_s"] * 1000, move["after"])
            return
        n = max(len(tr) for tr in tracks.values())
        for k in range(n):
            units_now, at = [], 0.0
            for uid, tr in tracks.items():
                t, x, z, b, front_m, depth_m = tr[min(k, len(tr) - 1)]
                at = max(at, t)
                cx, cz = centre({"x": x, "z": z, "bearing": b, "depth_m": depth_m})
                moving = k + 1 < len(tr) and tuple(tr[k + 1][1:4]) != (x, z, b)
                units_now.append(rec.unit_row(f"own:{uid}", cx, cz, b, front_m, depth_m, own_by_id[uid]["men"], moving))
            frames.append({"game_ms": round(t0 + at * 1000), "units": units_now + rows(enemy, "enemy", enemy_by_id)})

    def seen(ms, field, vision, anchor, bearing, routes):
        overlays.extend([rec.field_overlay(ms, field), rec.groups_overlay(ms, vision, side_id),
                         rec.routes_overlay(ms, routes, anchor, bearing)])

    events.append(rec.stage_event(0, "placed"))
    start = 0
    al = sides.get("alignment")
    if al:
        overlays.append(rec.alignment_overlay(0, al))
    if sides.get("own_before"):
        c = al["check"]
        events.append({"game_ms": 0, "kind": "decision", "decision": "align", "reason": "off_line", "text": rec.tr(
            f"выравнивание на расстановке: угол {c['angle_off_deg']:+.0f}°, вбок {c['offset_m']:+.0f} м",
            f"alignment at deployment: {c['angle_off_deg']:+.0f}°, {c['offset_m']:+.0f} m aside")})
        frame(0, sides["own_before"]["placements"], moving=True)
        start = ALIGN_S * 1000
    appr = sides.get("approach")
    final_vision = {"own": sides["own"].get("vision"), "enemy": sides["enemy"].get("vision")}
    moves = appr["moves"] if appr else []
    frame(start, appr["trail"][0] if appr and appr["trail"] else own_final)
    if appr:
        s0 = appr["start"]
        seen(start, moves[0]["field"] if moves else sides.get("battlefield"),
             moves[0]["vision"] if moves else final_vision, s0["anchor"], s0["bearing"], s0["lord_routes"])
        events.append(rec.stage_event(start, "approach"))
        for r in appr["log"]:
            events.append(rec.decision_event(start + r["t_s"] * 1000, r["decision"], r["reason"], r["gap_m"],
                                             r["stop_gap_m"], r["advance_m"] if r["decision"] == "approach" else None,
                                             r["window"]))
        for i, m in enumerate(moves):
            t0, t1 = start + m["t_s"] * 1000, start + (m["t_s"] + m["walk_s"]) * 1000
            walked(t0, m)
            events.append(rec.manoeuvre_event(t1, m["decision"], "stopped"))
            nxt = moves[i + 1] if i + 1 < len(moves) else None
            seen(t1, nxt["field"] if nxt else sides.get("battlefield"), nxt["vision"] if nxt else final_vision,
                 m["anchor"], m["bearing"], m["lord_routes"])
    else:
        seen(start, sides.get("battlefield"), final_vision, sides["own"].get("anchor"), sides["own"].get("bearing"),
             sides["own"].get("lord_routes"))
    end = max(f["game_ms"] for f in frames)
    fire_s = army.get("fire_s") or 0
    if fire_s and own_final:
        events.append(rec.stage_event(end, "hold"))
        blocks = simulation.blocks(own_final, own_by_id, "own") + simulation.blocks(enemy, enemy_by_id, "enemy")
        for row in planner.fire(blocks, fire_s):
            men = {u["id"]: u["men"] for u in row["units"]}
            ammo = {u["id"]: u["ammo"] for u in row["units"] if u["side"] == "own"}
            frame(end + row["t"] * 1000, own_final, men=men, ammo=ammo)
        lost = sides.get("fire", {}).get("lost")
        if lost:
            lo, le = lost["own"]["men"], lost["enemy"]["men"]
            events.append(rec.note(end + fire_s * 1000, f"перестрелка {fire_s} с: мы −{lo}, они −{le}",
                                   f"{fire_s} s of fire: us −{lo}, them −{le}"))
    meta = {"source": "sim", "name": name, "army": name, "strategy": sides["own"].get("strategy"),
            "tree": army.get("tree") or {}, "title": rec.tr(f"Симуляция · {name}", f"Simulation · {name}")}
    # The ground the army walks on: the captured map round its way (rocks the walker goes round).
    mask = rec.map_mask(MapGrid(army["map"]), frames) if army.get("map") else None
    return rec.make(meta, units, frames, events, {}, mask, idle_ms=0, overlays=overlays)
