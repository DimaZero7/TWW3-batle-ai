"""The battle record: one JSON per battle, the same for a game run and a simulation.

    {"format": "tww3-battle-record", "version": 2,
     "meta":   {source: game|sim|logistics, name, title{ru,en}, army, strategy, skipped: [...], duration_ms},
     "units":  [{id, side, name, role, kind, type{ru,en}, tag{ru,en}, men_max, range_m, reach_m}],
     "frames": [{t_ms, game_ms, units: [{id, x, z, bearing, front_m, depth_m, men, moving, ammo?}]}],
     "soldiers": {unit id: [{t_ms, pts: [across_dm, forward_dm, ...]}]},     real soldiers, when known
     "events": [{t_ms, game_ms, kind: stage|decision|manoeuvre|note, text{ru,en}, ...}],
     "overlays": [{t_ms, game_ms, kind: field|groups|alignment|routes|logistics, ...}],  what the modules saw
     "series": [{name{ru,en}, points: [[t_ms, value], ...]}],                           extra lines on the chart
     "mask":   {x, z, bearing, along0, across0, step, rows: ["..#.", ...]} or null}

(x, z) of a unit is the CENTRE of its block (as the game gives it); bearing in
degrees, forward = (sin b, cos b), right = (cos b, -sin b). A soldier's offset is
in the unit's frame, in decimetres. t_ms is the player's clock: long stretches
where nothing moves are cut to a second (meta.skipped); game_ms is the battle's.
Every text a person reads is in both languages: {"ru": ..., "en": ...}. An
overlay stays on the page until the next one of its kind.
"""
import json
import math
from pathlib import Path

FORMAT, VERSION = "tww3-battle-record", 2
SIDES = ("own", "enemy")
ROLES = ("wall", "arc", "lord", "other")
EVENT_KINDS = ("stage", "decision", "manoeuvre", "note")
OVERLAY_KINDS = ("field", "groups", "alignment", "routes", "logistics")
PLAYER = Path(__file__).with_name("player.html")


def tr(ru, en):
    return {"ru": ru, "en": en}


# The trunk's decisions in plain words (apps.tactics / apps.approach codes).
DECISIONS = {"align": tr("выравниваюсь", "aligning"), "approach": tr("скачок вперёд", "step forward"),
             "hold": tr("стою", "holding"), "wait": tr("жду", "waiting"), "blocked": tr("места нет", "no room"),
             "aligned": tr("ровно", "in line")}
REASONS = {"off_line": tr("стоим криво", "off the line"), "step": tr("до рубежа далеко", "far from the stop line"),
           "at_stop_line": tr("на рубеже", "at the stop line"), "under_fire": tr("под обстрелом", "under fire"),
           "manoeuvre_under_way": tr("манёвр идёт", "a manoeuvre is under way"),
           "no_place": tr("некуда встать", "nowhere to stand"), "no_path": tr("нет пути", "no path"),
           "no_step": tr("шага нет", "no step"), "stopped": tr("дошли", "arrived"),
           "timeout": tr("не дошли за время", "did not arrive in time")}
STAGES = {"placed": tr("расстановка", "deployment"), "align": tr("выравнивание", "alignment"),
          "mask": tr("маска местности", "terrain mask"), "approach": tr("сближение", "approach"),
          "hold": tr("стоим и стреляем", "holding and shooting"), "logistics": tr("очереди у препятствия", "queues at the obstacle")}
WINDOWS = {"window": tr("окно", "window"), "no_window": tr("окна нет", "no window"),
           "no_enemy_shooters": tr("у них нет стрелков", "they have no shooters"),
           "no_shooters": tr("у нас нет стрелков", "we have no shooters")}
MANOEUVRES = {"align": tr("выравнивание", "alignment"), "approach": tr("скачок", "step")}
# Unit types by the game's unit class (no unit keys): type and the tag's letter.
CLASSES = {"com": (tr("Лорд", "Lord"), tr("Лорд", "Lord")), "inf_mel": (tr("Пехота", "Infantry"), tr("П", "I")),
           "inf_mis": (tr("Стрелки", "Missile infantry"), tr("С", "M")), "cav": (tr("Конница", "Cavalry"), tr("К", "C"))}
OTHER = (tr("Отряд", "Unit"), tr("О", "U"))
UNIT_GROUPS = ("inf", "cav", "cha", "art", "mon", "veh")


def _join(parts, sep=""):
    return {lang: sep.join(p[lang] if isinstance(p, dict) else str(p) for p in parts) for lang in ("ru", "en")}


def decision_text(decision, reason=None):
    parts = [DECISIONS.get(decision, tr(decision, decision))]
    if reason:
        why = REASONS.get(reason) or (tr("жду выравнивания", "waiting for the alignment")
                                      if reason.startswith("alignment_") else tr(reason, reason))
        parts += [" (", why, ")"]
    return _join(parts)


def stage_event(game_ms, stage):
    return {"game_ms": game_ms, "kind": "stage", "stage": stage,
            "text": _join([tr("этап: ", "stage: "), STAGES.get(stage, tr(stage, stage))])}


def manoeuvre_event(game_ms, kind, reason, unit=None):
    what = MANOEUVRES.get(kind, tr(kind, kind))
    text = _join([tr("манёвр «", "manoeuvre “"), what, tr("» закончен: ", "” ended: "),
                  REASONS.get(reason, tr(reason, reason))] + ([f" ({unit})"] if unit else []))
    return {"game_ms": game_ms, "kind": "manoeuvre", "manoeuvre": kind, "reason": reason, "text": text}


def note(game_ms, ru, en):
    return {"game_ms": game_ms, "kind": "note", "text": tr(ru, en)}


def decision_event(game_ms, decision, reason, gap_m, stop_gap_m, advance_m, window):
    """A decision of the trunk (apps.tactics intent) with its window (apps.reach.window) as an event."""
    w = window or {}
    span = w.get("window") or {}
    parts = [decision_text(decision, reason)]
    if gap_m is not None:
        parts.append(tr(f"; между фронтами {gap_m:.0f} м", f"; {gap_m:.0f} m between the fronts"))
    if w.get("reason"):
        parts += ["; ", WINDOWS.get(w["reason"], tr(w["reason"], w["reason"])),
                  tr(f", достают {w.get('reached', 0)} из {w.get('shooters', 0)}",
                     f", {w.get('reached', 0)} of {w.get('shooters', 0)} reach")]
    return {"game_ms": game_ms, "kind": "decision", "decision": decision, "reason": reason, "text": _join(parts),
            "gap_m": gap_m, "stop_gap_m": stop_gap_m, "advance_m": advance_m,
            "window": {"reason": w.get("reason"), "from": span.get("from"), "to": span.get("to"),
                       "safe_to": w.get("safe_to"), "advance_m": w.get("advance_m"), "reached": w.get("reached"),
                       "shooters": w.get("shooters"), "under_fire_now": w.get("under_fire_now")} if w else None}


def unit_spec(uid, side, name, role, kind, key, men_max, range_m):
    """What does not change about a unit; type and tag come from the unit class, the title from its key."""
    parts = (key or "").split("_")
    at = next((i for i, p in enumerate(parts) if p in UNIT_GROUPS), None)
    title = " ".join(parts[at + 1:-1]) if at is not None and len(parts) > at + 2 else (key or name)
    rng = range_m or 0
    return {"id": uid, "side": side, "name": name, "role": role if role in ROLES else "other", "kind": kind,
            "key": key, "title": title, "men_max": men_max or 1, "range_m": rng,
            "reach_m": first_arrow(rng) if rng else None}


def tag_units(units):
    """type{ru,en} and a short tag{ru,en} (П1, С3, Лорд) for every unit, numbered per side and type."""
    count = {}
    for u in units:
        cls = u.get("kind") or ""
        typ, letter = CLASSES.get(cls) or CLASSES.get(cls.split("_")[0]) or OTHER
        u["type"] = typ
        if letter == CLASSES["com"][1]:
            u["tag"] = dict(letter)
            continue
        n = count[(u["side"], letter["ru"])] = count.get((u["side"], letter["ru"]), 0) + 1
        u["tag"] = {lang: f"{letter[lang]}{n}" for lang in ("ru", "en")}
    return units


def mask_of(m):
    """apps.mask's {grid, code} (a game 'mask' event or the simulation's) for the page."""
    if not m or not m.get("code"):
        return None
    g = m["grid"]
    return {"x": g["frame"]["origin"]["x"], "z": g["frame"]["origin"]["z"], "bearing": g["frame"]["bearing"],
            "along0": g["along0"], "across0": g["across0"], "step": g["step"], "rows": m["code"].split("/")}


def map_mask(grid, frames, side="own", margin_m=60):
    """Blocked cells of a captured map (tools/sim/mapgrid.MapGrid) round where `side` walks, in the
    mask's shape (bearing 0: along = +z, across = +x) — the simulation's ground, not apps.mask's field."""
    pts = [(u["x"], u["z"]) for f in frames for u in f["units"] if u["id"].startswith(side + ":")]
    if not pts:
        return None
    step = float(grid.step)
    x0 = math.floor((min(p[0] for p in pts) - margin_m - grid.min_x) / step) * step + grid.min_x
    z0 = math.floor((min(p[1] for p in pts) - margin_m - grid.min_z) / step) * step + grid.min_z
    cols = int(math.ceil((max(p[0] for p in pts) + margin_m - x0) / step))
    rows = int(math.ceil((max(p[1] for p in pts) + margin_m - z0) / step))
    lines = ["".join("." if grid.reader(x0 + (c + 0.5) * step, z0 + (r + 0.5) * step)[0] else "#" for c in range(cols))
             for r in range(rows)]
    return {"x": x0, "z": z0, "bearing": 0, "along0": 0, "across0": 0, "step": step, "rows": lines}


def in_frame(origin, bearing, along, across):
    """A point of a frame (origin, bearing): `along` forward, `across` to the right."""
    f, r = axes(bearing)
    return [round(origin["x"] + f[0] * along + r[0] * across, 2), round(origin["z"] + f[1] * along + r[1] * across, 2)]


def field_overlay(game_ms, field):
    """apps.battlefield: the frame between the two main groups, its axis, both fronts."""
    if not field or not field.get("corners"):
        return None
    o, b = field["origin"], field["bearing"]
    own, enemy = field.get("own") or {}, field.get("enemy") or {}

    def front(side):
        if side.get("front_m") is None:
            return None
        return [in_frame(o, b, side["front_m"], side["left_m"]), in_frame(o, b, side["front_m"], side["right_m"])]
    return {"game_ms": game_ms, "kind": "field", "corners": [[c["x"], c["z"]] for c in field["corners"]],
            "origin": [o["x"], o["z"]], "bearing": b, "gap_m": field.get("gap_m"),
            "own_front": front(own), "enemy_front": front(enemy)}


def groups_overlay(game_ms, vision, side_ids):
    """apps.vision: groups of both sides and which is the main one; ids become record ids."""
    out = {"game_ms": game_ms, "kind": "groups"}
    for side in SIDES:
        pic = (vision or {}).get(side) or {}
        main = set((pic.get("main") or {}).get("ids") or [])
        rows = []
        for g in pic.get("groups") or []:
            bd = g.get("bounds") or {}
            rows.append({"ids": [side_ids(side, i) for i in g.get("ids", [])], "main": bool(main) and set(g["ids"]) == main,
                         "centre": [g["centre"]["x"], g["centre"]["z"]], "facing": g.get("facing"),
                         "bounds": [bd.get("min_x"), bd.get("min_z"), bd.get("max_x"), bd.get("max_z")],
                         "strength": g.get("strength")})
        out[side] = rows
    return out


def alignment_overlay(game_ms, alignment):
    """apps.alignment: how crooked we stand and where it would put us."""
    if not alignment:
        return None
    c, t = alignment.get("check") or {}, alignment.get("target")
    return {"game_ms": game_ms, "kind": "alignment", "needed": bool(c.get("needed")),
            "angle_off_deg": c.get("angle_off_deg"), "offset_m": c.get("offset_m"),
            "max_angle_deg": c.get("max_angle_deg"), "max_offset_m": c.get("max_offset_m"),
            "target": {"x": t["anchor"]["x"], "z": t["anchor"]["z"], "bearing": t["bearing"]} if t else None}


def routes_overlay(game_ms, routes, anchor, bearing):
    """apps.formation.lord_routes: the lord's ways to the flanks, from (along, back) of the formation's anchor."""
    if not routes or anchor is None:
        return None
    o = {"x": anchor[0], "z": anchor[1]}
    lines = [{"side": side, "clear": r.get("clear"), "length_m": r.get("length_m"),
              "points": [in_frame(o, bearing, -back, along) for along, back in r.get("points", [])]}
             for side, r in sorted(routes.items())]
    return {"game_ms": game_ms, "kind": "routes", "lines": lines}


_lua = None


def first_arrow(range_m):
    """apps.reach.first_arrow_m: the one rule of the game and the simulation."""
    global _lua
    if _lua is None:
        from tools.lua_runtime import new_runtime
        _lua = new_runtime().eval("function(r) return require('apps.reach.services').first_arrow_m({range_m = r}) end")
    return float(_lua(range_m))


def axes(bearing):
    b = math.radians(bearing)
    return (math.sin(b), math.cos(b)), (math.cos(b), -math.sin(b))


def local_points(flat, x, z, bearing, scale=1.0):
    """Soldiers [x, z, x, z, ...] (world, times `scale`) to [across_dm, forward_dm, ...] around (x, z)."""
    f, r = axes(bearing)
    out = []
    for i in range(0, len(flat) - 1, 2):
        dx, dz = flat[i] * scale - x, flat[i + 1] * scale - z
        out += [round((dx * r[0] + dz * r[1]) * 10), round((dx * f[0] + dz * f[1]) * 10)]
    return out


def world_points(pts, x, z, bearing):
    """The inverse of local_points: [x, z, ...] in metres."""
    f, r = axes(bearing)
    out = []
    for i in range(0, len(pts) - 1, 2):
        a, fw = pts[i] / 10, pts[i + 1] / 10
        out += [x + r[0] * a + f[0] * fw, z + r[1] * a + f[1] * fw]
    return out


def unit_row(uid, x, z, bearing, front_m, depth_m, men, moving=False, ammo=None):
    row = {"id": uid, "x": round(x, 2), "z": round(z, 2), "bearing": round(bearing % 360, 2),
           "front_m": round(front_m, 2), "depth_m": round(depth_m, 2), "men": int(men), "moving": bool(moving)}
    if ammo is not None:
        row["ammo"] = round(ammo)
    return row


def cut_idle(frames, timed, idle_ms=5000, keep_ms=1000):
    """Gaps between frames longer than idle_ms where nothing moved shrink to keep_ms.
    Sets t_ms on frames and every row of `timed` (from game_ms); returns the cuts."""
    cuts = []
    for a, b in zip(frames, frames[1:]):
        gap = b["game_ms"] - a["game_ms"]
        still = not any(u.get("moving") for u in a["units"] + b["units"]) and _same_places(a, b)
        if gap > idle_ms and still:
            cuts.append({"game_from_ms": a["game_ms"], "game_to_ms": b["game_ms"], "cut_ms": gap - keep_ms})
    start = frames[0]["game_ms"] if frames else 0

    def t_of(game_ms):
        before = 0
        for c in cuts:
            if game_ms >= c["game_to_ms"]:
                before += c["cut_ms"]
            elif game_ms > c["game_from_ms"]:
                # Inside a cut: squeezed onto its kept second.
                frac = (game_ms - c["game_from_ms"]) / (c["game_to_ms"] - c["game_from_ms"])
                return c["game_from_ms"] - start - before + round(frac * keep_ms)
        return game_ms - start - before

    for row in frames + timed:
        row["t_ms"] = max(0, t_of(row["game_ms"]))
    for c in cuts:
        c["at_ms"] = t_of(c["game_from_ms"])
    return cuts


def _same_places(a, b, tol_m=5.0):
    pa = {u["id"]: u for u in a["units"]}
    for u in b["units"]:
        p = pa.get(u["id"])
        if p and (abs(p["x"] - u["x"]) > tol_m or abs(p["z"] - u["z"]) > tol_m):
            return False
    return True


def make(meta, units, frames, events, soldiers=None, mask=None, idle_ms=5000, overlays=None, series=None):
    """A complete record; frames, events, overlays and soldier snapshots carry game_ms, sorted here;
    series points are [game_ms, value]."""
    frames = sorted(frames, key=lambda f: f["game_ms"])
    events = sorted(events, key=lambda e: e["game_ms"])
    overlays = sorted([o for o in overlays or [] if o], key=lambda o: o["game_ms"])
    soldiers = soldiers or {}
    snaps = [s for rows in soldiers.values() for s in rows]
    points = [{"game_ms": p[0], "value": p[1], "_s": n} for n, s in enumerate(series or []) for p in s["points"]]
    meta = dict(meta)
    meta["skipped"] = cut_idle(frames, events + overlays + snaps + points, idle_ms if idle_ms else 10 ** 12)
    for s in snaps:
        s.pop("game_ms")
    out_series = [{"name": s["name"], "points": []} for s in series or []]
    for p in points:
        out_series[p["_s"]]["points"].append([p["t_ms"], p["value"]])
    meta["duration_ms"] = frames[-1]["t_ms"] if frames else 0
    return {"format": FORMAT, "version": VERSION, "meta": meta, "units": tag_units(units), "frames": frames,
            "soldiers": soldiers, "events": events, "overlays": overlays, "series": out_series, "mask": mask}


def _texts_ok(v):
    return isinstance(v, dict) and all(isinstance(v.get(lang), str) and v[lang] for lang in ("ru", "en"))


def check(record):
    """Problems of a record (empty list = fine)."""
    bad = []
    if record.get("format") != FORMAT or record.get("version") != VERSION:
        bad.append("not a battle record of this version")
    ids = set()
    for u in record.get("units", []):
        if u["side"] not in SIDES or u["role"] not in ROLES:
            bad.append(f"unit {u['id']}: side/role")
        if not (_texts_ok(u.get("type")) and _texts_ok(u.get("tag"))):
            bad.append(f"unit {u['id']}: type/tag in both languages")
        ids.add(u["id"])
    last = -1
    for f in record.get("frames", []):
        if f["t_ms"] < last:
            bad.append(f"frames not in time order at {f['t_ms']}")
        last = f["t_ms"]
        for u in f["units"]:
            if u["id"] not in ids:
                bad.append(f"frame {f['t_ms']}: unknown unit {u['id']}")
            if not all(math.isfinite(u[k]) for k in ("x", "z", "bearing", "front_m", "depth_m")):
                bad.append(f"frame {f['t_ms']}: {u['id']} not finite")
    for e in record.get("events", []):
        if e["kind"] not in EVENT_KINDS:
            bad.append(f"event kind {e['kind']}")
        if not _texts_ok(e.get("text")):
            bad.append(f"event at {e.get('t_ms')}: text in both languages")
    for o in record.get("overlays", []):
        if o["kind"] not in OVERLAY_KINDS:
            bad.append(f"overlay kind {o['kind']}")
    for s in record.get("series", []):
        if not _texts_ok(s.get("name")):
            bad.append("series name in both languages")
    if not _texts_ok(record.get("meta", {}).get("title")):
        bad.append("title in both languages")
    for uid in record.get("soldiers", {}):
        if uid not in ids:
            bad.append(f"soldiers of unknown unit {uid}")
    if not record.get("frames"):
        bad.append("no frames")
    return bad


def write(record, path):
    problems = check(record)
    if problems:
        raise ValueError("bad record:\n" + "\n".join(problems[:20]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    return path


def page(records, path, preset="approach", lang="ru"):
    """One HTML file with the records inside: opens without a server."""
    for r in records:
        problems = check(r)
        if problems:
            raise ValueError(f"bad record {r.get('meta', {}).get('name')}:\n" + "\n".join(problems[:20]))
    data = json.dumps({"records": records, "preset": preset, "lang": lang}, ensure_ascii=False, separators=(",", ":"))
    html = PLAYER.read_text(encoding="utf-8").replace("/*RECORDS*/null", data.replace("</", "<\\/"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", newline="\n")
    return path
