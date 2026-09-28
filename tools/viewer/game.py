"""A game run (build/<target>/runs/<run>/events.jsonl + manifest.json) to a battle record.

Frames come from every event that carries the units (stage_snapshot,
approach_sample, approach_manoeuvre, hold_sample); the enemy is carried over
from the last event that saw it. Blocks' depth comes from the unit's shapes in
the manifest (the one nearest the ordered width); soldiers from the snapshots
that have soldiers_dm. Events: stages, the trunk's decisions with the window,
manoeuvres' ends, the alignment, the result.
"""
import json
from pathlib import Path

from tools.viewer import record as rec

UNIT_EVENTS = ("stage_snapshot", "approach_sample", "approach_manoeuvre", "hold_sample")


def load(run):
    run = Path(run)
    rows = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return rows, manifest


def _role(spec):
    if spec.get("commanding") and (spec.get("men") or 1) == 1:
        return "lord"
    return "arc" if (spec.get("range_m") or 0) > 0 else "wall"


def _shape(spec, width):
    shapes = spec.get("shapes") or [{"ordered_m": 5, "front_m": 1.0, "depth_m": 1.0}]
    return min(shapes, key=lambda s: abs(s["ordered_m"] - (width or 0)))




def convert(run, idle_ms=5000):
    rows, manifest = load(run)
    cfg = manifest["config"]
    specs = {("own", u["id"]): u for u in cfg["own"]["units"]}
    specs.update({("enemy", u["id"]): u for u in cfg["enemy"]["units"]})
    roles = {}
    for r in rows:
        if r["event"] == "stage_snapshot":
            roles.update({u["script_name"]: u["role"] for u in r["units"] if u.get("role")})
    units = [rec.unit_spec(f"{side}:{name}", side, name, (roles.get(name) if side == "own" else None) or _role(s),
                           s.get("class"), s.get("key"), s.get("men"), s.get("range_m")) for (side, name), s in specs.items()]

    def row(side, u):
        m = u["motion"]
        spec = specs.get((side, u["script_name"]))
        if not spec:
            return None
        shape = _shape(spec, m.get("ordered_width"))
        men, full = m.get("number_of_men_alive") or 0, spec.get("men") or 1
        depth = shape["depth_m"] * (men / full if full > 1 else 1)
        return rec.unit_row(f"{side}:{u['script_name']}", m["x"], m["z"], m["bearing"], shape["front_m"],
                            max(depth, 1.0), men, m.get("is_moving"), u.get("ammo") if side == "own" else None)

    frames, soldiers, enemy_last = {}, {}, None
    first_enemy = next((r["enemy"] for r in rows if r["event"] in UNIT_EVENTS and r.get("enemy")), [])
    for r in rows:
        if r["event"] not in UNIT_EVENTS or not r.get("units"):
            continue
        if r.get("enemy"):
            enemy_last = r["enemy"]
        enemy = enemy_last or first_enemy
        units_now = [x for x in ([row("own", u) for u in r["units"]] + [row("enemy", e) for e in enemy]) if x]
        frames[r["model_ms"]] = {"game_ms": r["model_ms"], "units": units_now}
        for u in r["units"]:
            if u.get("soldiers_dm"):
                m = u["motion"]
                soldiers.setdefault(f"own:{u['script_name']}", []).append(
                    {"game_ms": r["model_ms"], "pts": rec.local_points(u["soldiers_dm"], m["x"], m["z"], m["bearing"], 0.1)})

    events, overlays, governor = [], [], None
    for r in rows:
        kind, at = r["event"], r.get("model_ms", 0)
        if kind == "stage":
            events.append(rec.stage_event(at, r["stage"]))
        elif kind == "approach_decision":
            events.append(decision_event(r))
        elif kind == "approach_manoeuvre":
            events.append(rec.manoeuvre_event(at, r.get("kind"), r.get("reason"), r.get("unit")))
        elif kind in ("alignment", "alignment_after"):
            c = r.get("check") or {}
            if kind == "alignment":
                events.append({"game_ms": at, "kind": "decision", "decision": r.get("decision"),
                               "reason": "off_line" if c.get("needed") else None, "text": rec.tr(
                                   f"выравнивание на расстановке: угол {c.get('angle_off_deg', 0):+.0f}°, "
                                   f"вбок {c.get('offset_m', 0):+.0f} м",
                                   f"alignment at deployment: {c.get('angle_off_deg', 0):+.0f}°, "
                                   f"{c.get('offset_m', 0):+.0f} m aside")})
            # What apps.battlefield and apps.alignment saw (the game logs them at deployment only).
            overlays += [rec.field_overlay(at, r.get("field")),
                         rec.alignment_overlay(at, {"check": c, "target": r.get("target") if kind == "alignment" else None})]
        elif kind == "governor" and r.get("decision") != governor:
            governor = r.get("decision")
            d = rec.DECISIONS.get(governor, rec.tr(governor, governor))
            events.append(rec.note(at, "проверка строя: " + d["ru"], "formation check: " + d["en"]))
        elif kind == "result":
            events.append(rec.note(at, f"итог прогона: {r.get('status')}", f"run result: {r.get('status')}"))

    mask = rec.mask_of(next((r for r in rows if r["event"] == "mask"), None))
    plan = next((r for r in rows if r["event"] == "plan"), {})
    name = Path(run).name
    meta = {"source": "game", "name": name, "army": cfg.get("army"), "strategy": plan.get("strategy"),
            "tree": cfg.get("tree") or {},
            "title": rec.tr(f"Игра · {cfg.get('army')} · {name}", f"Game · {cfg.get('army')} · {name}")}
    return rec.make(meta, units, list(frames.values()), events, soldiers, mask, idle_ms, overlays)


def decision_event(r):
    """approach_decision -> a decision event with the window (apps.reach) and the step."""
    step = r.get("step") or {}
    return rec.decision_event(r.get("model_ms", 0), r["decision"], r.get("reason"), r.get("gap_m"), r.get("stop_gap_m"),
                              step.get("advance_m") if step.get("action") == "step" else None, r.get("window"))
