"""Check a unit-readout run against the documented unit readouts.

Two views of the same run:

  python -m tools.analysis.unit_readout <run>                   # full summary
  python -m tools.analysis.unit_readout <run> --view side --side 1

--view full (default): everything we read, checked against the docs
(states.md, state-sensors.md, missile-range.md) -> report.md, summary.json.

--view side: what ONE side saw (side_view events built by
apps/observation), compared with the full summary (full_view) as ground
truth: hidden enemies must never leak -> report-side-<n>.md, summary-side-<n>.json.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

UNITS = ["a_general", "a_spears", "a_archers", "a_forest", "a_stalkers",
         "b_general", "b_spears", "b_archers", "b_forest", "b_stalkers"]
# general / spears / archers / stalkers; the forest ambush units are spearmen
KIND = {name: {"forest": "spears"}.get(name.split("_", 1)[1], name.split("_", 1)[1]) for name in UNITS}
SIDE = {name: 1 if name.startswith("a_") else 2 for name in UNITS}
AMBUSH = {"forest": "hide_forest in forest", "stalkers": "stalk on open grass"}

# docs/*/game/units/states.md, "What the game returned for each unit".
PROFILE = {
    "general": {"attribute_stalk": False, "type": "wh_main_emp_cha_general_0", "initial_men": 1, "attribute_hide_forest": True,
                "attribute_charge_defense_vs_large": False, "attribute_charge_reflection": False,
                "attribute_encourages": True, "can_defend": True, "can_fire_at_will": False,
                "can_skirmish": False, "can_change_formation_spacing": False},
    "spears": {"attribute_stalk": False, "type": "wh_main_emp_inf_spearmen_0", "initial_men": 120, "missile_range": 0,
               "attribute_hide_forest": True, "attribute_charge_defense_vs_large": True,
               "attribute_charge_reflection": True, "attribute_encourages": False, "can_defend": True,
               "can_fire_at_will": False, "can_skirmish": False, "can_change_formation_spacing": False},
    "archers": {"attribute_stalk": False, "type": "wh2_dlc13_emp_inf_archers_0", "initial_men": 90, "missile_range": 130,
                "attribute_hide_forest": True, "attribute_charge_defense_vs_large": False,
                "attribute_charge_reflection": False, "attribute_encourages": False, "can_defend": True,
                "can_fire_at_will": True, "can_skirmish": True, "can_change_formation_spacing": False},
    # Game DB (unit_attributes_to_groups_junctions): Huntsmen have stalk.
    "stalkers": {"type": "wh2_dlc13_emp_inf_huntsmen_0", "attribute_stalk": True},
}


def load(run):
    rows = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
    return rows + expand_side_views(rows)


def expand_side_views(rows):
    """side_view carries own readings, enemy intel and shooter range; expand it
    into unit_state / intel / range rows for the full-summary checks."""
    out = []
    for r in rows:
        if r["event"] != "side_view":
            continue
        view, stage, side = r["view"], r.get("stage"), r["observer_side"]
        for name, readings in view["own"].items():
            out.append({"event": "unit_state", "side": side, "name": name, "stage": stage, "readings": readings})
        for name, record in view["enemies"].items():
            out.append(dict(record, event="intel", observer_side=side, stage=stage))
        for pair, readings in view["range"].items():
            source, target = pair.split(">", 1)
            out.append({"event": "range", "source": source, "target": target, "stage": stage, "readings": readings})
    return out


def analyse(rows):
    checks = []

    def check(group, name, expected, actual, ok=None):
        if ok is None:
            ok = expected == actual
        checks.append({"group": group, "check": name, "expected": expected, "actual": actual,
                       "result": "pass" if ok else "fail"})

    events = defaultdict(list)
    for r in rows:
        events[r["event"]].append(r)
    check("run", "no Lua errors", 0, len(events["error"]))
    check("run", "run completed", "completed", (events["result"] or [{}])[-1].get("status"))
    moved = []
    for name in UNITS:
        states = [s for s in events["unit_state"] if s["name"] == name]
        setup = next((s for s in states if s.get("stage") == "setup"), None)
        # Teleports back to the spawn apply on a following tick: compare with
        # the first sample of the march stage.
        idle = next((s for s in states if s.get("stage") == "march"), None)
        if setup and idle:
            a = setup["readings"]["sensors"]["native.position"].get("value")
            b = idle["readings"]["sensors"]["native.position"].get("value")
            if a and b and ((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5 > 15:
                moved.append(name)
    check("run", "units start where the scenario put them (<15 m)", [], moved)
    stages = [r["name"] for r in events["stage"]]
    check("run", "all stages reached", ["idle", "march", "ranged", "cease_fire", "melee", "scout_a80", "scout_a40", "scout_a15",
                                             "scout_b80", "scout_b40", "scout_b15", "halt"], stages)

    # --- static profile
    for p in events["unit_profile"]:
        for key, expected in PROFILE[KIND[p["name"]]].items():
            check("profile", f"{p['name']}.{key}", expected, p.get(key))
        if KIND[p["name"]] == "general":
            active = p.get("owned_non_passive_special_abilities")
            check("profile", f"{p['name']} active abilities (doc: Stand Your Ground, Foe Seeker)", 2,
                  len(active) if isinstance(active, list) else active,
                  isinstance(active, list) and len(active) == 2)
            passive = p.get("owned_passive_special_abilities")
            check("profile", f"{p['name']} passive abilities (doc: Single Entity, Hold the Line)", 2,
                  len(passive) if isinstance(passive, list) else passive,
                  isinstance(passive, list) and len(passive) == 2)
        for key in ("cco_CharacterRank", "cco_HasCharacterRank", "cco_ExperienceLevel"):
            check("profile", f"{p['name']}.{key} readable", "number/boolean", p.get(key),
                  not str(p.get(key)).startswith("unknown"))

    # --- 68 own-unit sensors: coverage, values, transitions per stage
    coverage = defaultdict(lambda: defaultdict(lambda: {"known": 0, "total": 0, "reasons": defaultdict(int),
                                                        "values": set(), "min": None, "max": None}))
    by_stage = defaultdict(lambda: defaultdict(list))  # (unit, sensor) -> stage -> values
    sensor_names = set()
    for r in events["unit_state"]:
        for key, s in r["readings"]["sensors"].items():
            sensor_names.add(key)
            c = coverage[key][r["name"]]
            c["total"] += 1
            if s["status"] == "known":
                c["known"] += 1
                v = s["value"]
                if isinstance(v, bool) or isinstance(v, str):
                    c["values"].add(v)
                elif isinstance(v, (int, float)):
                    c["min"] = v if c["min"] is None else min(c["min"], v)
                    c["max"] = v if c["max"] is None else max(c["max"], v)
                elif isinstance(v, list):
                    c["values"].update(v)
                by_stage[(r["name"], key)][r.get("stage")].append(v)
            else:
                c["reasons"][s["reason"]] += 1
    check("sensors", "own-unit sensors per sample", 68, len(sensor_names))
    never_known = sorted(k for k in sensor_names if all(coverage[k][u]["known"] == 0 for u in coverage[k]))
    check("sensors", "sensors known for at least one unit", len(sensor_names),
          len(sensor_names) - len(never_known), not never_known)

    def seen(unit, key, stage, predicate):
        return any(predicate(v) for v in by_stage[(unit, key)].get(stage, []))

    def drop(unit, key, stage_from, stage_to):
        a, b = by_stage[(unit, key)].get(stage_from, []), by_stage[(unit, key)].get(stage_to, [])
        return bool(a and b) and min(b) < max(a)

    dyn = [
        ("march: a_spears is_moving", seen("a_spears", "native.is_moving", "march", lambda v: v is True)),
        ("march: a_general is_moving_fast (run order)", seen("a_general", "native.is_moving_fast", "march", lambda v: v is True)),
        ("march: ordered_position known", seen("a_spears", "native.ordered_position", "march", lambda v: isinstance(v, dict))),
        ("ranged: a_archers IsFiringMissiles", seen("a_archers", "cco.IsFiringMissiles", "ranged", lambda v: v is True)),
        ("ranged: a_archers ammo_left decreases", drop("a_archers", "native.ammo_left", "march", "ranged")),
        ("ranged: b_spears under missile attack (native or CCO)",
         seen("b_spears", "native.is_under_missile_attack", "ranged", lambda v: v is True)
         or seen("b_spears", "cco.IsUnderMissileAttack", "ranged", lambda v: v is True)),
        ("ranged: b_spears HealthValue decreases", drop("b_spears", "cco.HealthValue", "march", "ranged")),
        ("melee: a_spears is_in_melee", seen("a_spears", "native.is_in_melee", "melee", lambda v: v is True)),
        ("melee: b_spears is_in_melee", seen("b_spears", "native.is_in_melee", "melee", lambda v: v is True)),
        ("melee: spearmen lose men (melee or scouting stages)",
         any(drop(u, "native.number_of_men_alive", "cease_fire", s)
             for u in ("a_spears", "b_spears") for s in ("melee",) + SCOUT_STAGES)),
        ("melee: a_general defend behaviour on", seen("a_general", "native.behaviour.defend", "melee", lambda v: v is True)),
        ("melee: current_target is a visible unit id",
         any(seen(u, "native.current_target", "melee", lambda v: isinstance(v, str) and v.startswith(("a_", "b_")))
             for u in ("a_spears", "b_spears"))),
        ("any: card stat_morale.Value known", any(coverage["cco.card.stat_morale.Value"][u]["known"] for u in UNITS)),
        ("any: MoralePercent known", any(coverage["cco.MoralePercent"][u]["known"] for u in UNITS)),
        ("any: StatusList keys read", any(coverage["cco.StatusList.Keys"][u]["known"] for u in UNITS)),
        ("halt: a_spears stops moving", seen("a_spears", "native.is_moving", "halt", lambda v: v is False)),
    ]
    for name, ok in dyn:
        check("dynamics", name, True, ok)

    # --- enemy gate: nothing but visibility
    leaks = [r for r in events["enemy_gate"]
             if r["readings"].get("access") != "withheld_own_only" or r["readings"].get("sensors")]
    check("gate", "enemy view withholds all sensors", 0, len(leaks))
    check("gate", "enemy gate samples", ">0", len(events["enemy_gate"]), bool(events["enemy_gate"]))

    # --- missile range
    ranges = defaultdict(list)
    for r in events["range"]:
        ranges[(r["source"], r["target"])].append(r)
    def known(r, key):
        s = r["readings"]["sensors"].get(key, {})
        return s.get("value") if s.get("status") == "known" else None
    archer_rows = [r for r in ranges[("a_archers", "b_spears")] if r["readings"]["access"] == "allowed"]
    check("range", "a_archers missile_range_m = 130",
          130, sorted({known(r, "missile_range_m") for r in archer_rows} - {None}),
          {known(r, "missile_range_m") for r in archer_rows} - {None} <= {130} and bool(archer_rows))
    card = {r["readings"]["sensors"]["card"]["Value"].get("value") for r in archer_rows} - {None}
    check("range", "a_archers card scalar_missile_range = 130", [130], sorted(card), card == {130})
    in_range = [r for r in archer_rows if r.get("stage") == "ranged"]
    check("range", "a_archers unit_in_range true in ranged stage (100 m)", True,
          any(known(r, "unit_in_range") for r in in_range))
    distances = [(known(r, "unit_distance_m"), known(r, "centre_distance_xz_m")) for r in archer_rows]
    distances = [d for d in distances if None not in d]
    check("range", "engine unit_distance < centre distance (footprints)", True,
          bool(distances) and all(a <= b + 0.01 for a, b in distances))
    withheld = [r for rs in ranges.values() for r in rs if r["readings"]["access"] == "withheld"]
    blocked_ok = all(r["readings"]["sensors"]["missile_range_m"]["reason"] == "endpoint_unavailable" for r in withheld)
    check("range", "withheld pairs expose no range", True, blocked_ok)

    # --- intel
    vis = defaultdict(int)
    for r in events["intel"]:
        vis[r["visibility"]] += 1
    check("intel", "visibility samples", ">0", dict(vis), bool(events["intel"]))
    bad_current = [r for r in events["intel"] if r["visibility"] != "visible" and r.get("current")]
    check("intel", "no current position for hidden units", 0, len(bad_current))
    check("intel", "last_seen present after a sighting", True,
          any(r.get("last_seen") for r in events["intel"]))

    # --- telemetry / navigation
    check("telemetry", "full summary frames", ">0", len(events["full_view"]), bool(events["full_view"]))
    check("telemetry", "side views (both sides)", ">0", len(events["side_view"]), bool(events["side_view"]))
    check("navigation", "nav diagnostics per stage", ">0", len(events["nav_state"]), bool(events["nav_state"]))
    nav_ok = [r for r in events["nav_state"]
              if r["origins"]["native_origin"]["result"].get("status") == "ok"]
    check("navigation", "can_reach_position own origin answered", True, bool(nav_ok))

    table = []
    for key in sorted(sensor_names):
        row = {"sensor": key}
        for u in UNITS:
            c = coverage[key][u]
            if not c["total"]:
                row[u] = "-"
            elif c["known"]:
                if c["min"] is not None:
                    row[u] = f"{c['min']:.4g}..{c['max']:.4g}" if c["min"] != c["max"] else f"{c['min']:.4g}"
                else:
                    values = sorted(map(str, c["values"]))
                    row[u] = ", ".join(values[:4]) + ("…" if len(values) > 4 else "")
                if c["known"] < c["total"]:
                    row[u] += f" ({c['known']}/{c['total']})"
            else:
                row[u] = "unknown: " + ", ".join(f"{k}×{n}" for k, n in c["reasons"].items())
        table.append(row)
    return checks, table, events


def write_report(run, checks, table, events):
    passed = sum(c["result"] == "pass" for c in checks)
    lines = [f"# Unit readout test — {run.name}", "",
             f"**{passed} / {len(checks)} checks passed.** Map: The Moorlands Route (catchment_03). "
             f"Samples: {len(events['unit_state'])} own-unit, {len(events['range'])} range, "
             f"{len(events['intel'])} intel, {len(events['full_view'])} full-summary frames, "
             f"{len(events['side_view'])} side views.", "",
             "| Group | Check | Expected | Actual | Result |", "|---|---|---|---|---|"]
    for c in checks:
        mark = "✅" if c["result"] == "pass" else "❌"
        lines.append(f"| {c['group']} | {c['check']} | `{c['expected']}` | `{c['actual']}` | {mark} |")
    lines += ["", "## All own-unit sensors (value range or seen values; unknown reasons)", "",
              "| Sensor | " + " | ".join(UNITS) + " |", "|---|" + "---|" * len(UNITS)]
    for row in table:
        lines.append(f"| `{row['sensor']}` | " + " | ".join(str(row[u]).replace("|", "/") for u in UNITS) + " |")
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"passed": passed, "total": len(checks), "checks": checks,
                                                  "sensors": table}, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    return passed


ALLOWED_ENEMY_FIELDS = {"id", "visibility", "sampled_ms", "current", "last_seen", "position_status"}
PRE_SCOUT = ("idle", "march", "ranged", "cease_fire", "melee")
SCOUT_STAGES = ("scout_a80", "scout_a40", "scout_a15", "scout_b80", "scout_b40", "scout_b15")


def analyse_side(rows, side):
    """One side's view against the full summary (ground truth)."""
    checks = []

    def check(group, name, expected, actual, ok=None):
        if ok is None:
            ok = expected == actual
        checks.append({"group": group, "check": name, "expected": expected, "actual": actual,
                       "result": "pass" if ok else "fail"})

    views = [r for r in rows if r["event"] == "side_view" and r["observer_side"] == side]
    fulls = [r for r in rows if r["event"] == "full_view"]
    check("run", "side views recorded", ">0", len(views), bool(views))
    check("run", "one full frame per side view", len(views), len(fulls), len(fulls) == len(views))
    enemies = [u for u in UNITS if SIDE[u] != side]
    own = [u for u in UNITS if SIDE[u] == side]

    truth_seen = defaultdict(dict)  # enemy -> time_ms -> (x, z) while truly visible
    per_stage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    mismatches, leaked, hidden_pos, stale, range_leaks = [], [], [], [], []
    own_complete = True
    for view_row, full in zip(views, fulls):
        view, stage = view_row["view"], view_row.get("stage")
        now = view["time_ms"]
        truth = {u["id"]: u for s in full["sides"] for u in s["units"]}
        for name in own:
            own_complete &= len(view["own"].get(name, {}).get("sensors", {})) == 68
        for name in enemies:
            rec = view["enemies"].get(name, {})
            real = truth.get(name, {})
            real_visible = real.get("visibility", {}).get(f"side_{side}")
            per_stage[name][stage][rec.get("visibility", "missing")] += 1
            if rec.get("visibility") == "visible" and rec.get("current"):
                truth_seen[name][now] = (rec["current"]["x"], rec["current"]["z"])
            extra = set(rec) - ALLOWED_ENEMY_FIELDS
            if extra:
                leaked.append((now, name, sorted(extra)))
            if rec.get("visibility") != "visible" and rec.get("current"):
                hidden_pos.append((now, name))
            if rec.get("visibility") in ("visible", "not_visible") and isinstance(real_visible, bool) \
                    and (rec["visibility"] == "visible") != real_visible:
                mismatches.append((now, name, rec["visibility"], real_visible))
            seen = rec.get("last_seen")
            if rec.get("visibility") != "visible" and seen:
                pos = truth_seen[name].get(seen["seen_ms"])
                if seen["seen_ms"] > now or pos is None or abs(pos[0] - seen["x"]) > 1 or abs(pos[1] - seen["z"]) > 1:
                    stale.append((now, name, seen["seen_ms"]))
        for pair, reading in view["range"].items():
            target = pair.split(">", 1)[1]
            if view["enemies"].get(target, {}).get("visibility") != "visible" and reading["access"] != "withheld":
                range_leaks.append((now, pair))

    samples = len(views) * len(enemies)
    check("leaks", "enemy records carry only visibility / position / last_seen", 0, len(leaked))
    check("leaks", "hidden enemy never has a current position", 0, len(hidden_pos))
    check("leaks", "range to a hidden enemy is withheld", 0, len(range_leaks))
    check("leaks", "last_seen = an earlier real sighting of this side", 0, len(stale))
    check("agreement", "side visibility = full summary visibility (<=2% at transitions)",
          f"<= {max(1, samples // 50)}", len(mismatches), len(mismatches) <= max(1, samples // 50))
    check("own", "own units: all 68 fields in the side view", True, own_complete)
    for kind, label in AMBUSH.items():
        unit = next(u for u in enemies if u.endswith("_" + kind))
        hidden = sum(per_stage[unit][s]["not_visible"] for s in PRE_SCOUT)
        shown = sum(per_stage[unit][s]["visible"] for s in PRE_SCOUT)
        revealed = sum(per_stage[unit][s]["visible"] for s in SCOUT_STAGES)
        check("mechanics", f"{unit} ({label}): hidden before scouting", ">0 hidden samples",
              f"{hidden} hidden / {shown} visible", hidden > 0)
        check("mechanics", f"{unit} ({label}): revealed while scouted", ">0 visible samples",
              revealed, revealed > 0)
        kept = sum(1 for v in views if v["view"]["enemies"].get(unit, {}).get("visibility") == "not_visible"
                   and v["view"]["enemies"][unit].get("last_seen"))
        check("mechanics", f"{unit}: samples hidden with a remembered last position", "info", kept, True)
    return checks, per_stage, {"mismatches": mismatches[:20], "leaked_fields": leaked[:20],
                               "hidden_with_position": hidden_pos[:20], "stale_last_seen": stale[:20],
                               "range_leaks": range_leaks[:20]}


def write_side_report(run, side, checks, per_stage):
    passed = sum(c["result"] == "pass" for c in checks)
    stages = ["idle", "march", "ranged", "cease_fire", "melee", *SCOUT_STAGES, "halt", "done"]
    lines = [f"# Side {side} view — {run.name}", "",
             f"**{passed} / {len(checks)} checks passed.** Only what side {side} may know about the enemy, "
             "checked against the full summary as ground truth.", "",
             "| Group | Check | Expected | Actual | Result |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['group']} | {c['check']} | `{c['expected']}` | `{c['actual']}` | "
                     f"{'✅' if c['result'] == 'pass' else '❌'} |")
    lines += ["", "## Enemy visibility by stage (visible / not visible samples)", "",
              "| Enemy | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for name, by_stage in per_stage.items():
        cells = [f"{by_stage[s]['visible']} / {by_stage[s]['not_visible']}" if s in by_stage else "-"
                 for s in stages]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    (run / f"report-side-{side}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--view", choices=("full", "side"), default="full")
    parser.add_argument("--side", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.view == "side":
        checks, per_stage, samples = analyse_side(load(args.run), args.side)
        passed = write_side_report(args.run, args.side, checks, per_stage)
        (args.run / f"summary-side-{args.side}.json").write_text(json.dumps(
            {"passed": passed, "total": len(checks), "checks": checks, "samples": samples},
            ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        for c in checks:
            if c["result"] != "pass":
                print("FAIL", c["group"], c["check"], "expected", c["expected"], "actual", c["actual"])
        print(f"{passed}/{len(checks)} checks passed; report: {args.run / f'report-side-{args.side}.md'}")
        return 0 if passed == len(checks) else 1
    checks, table, events = analyse(load(args.run))
    passed = write_report(args.run, checks, table, events)
    for c in checks:
        if c["result"] != "pass":
            print("FAIL", c["group"], c["check"], "expected", c["expected"], "actual", c["actual"])
    print(f"{passed}/{len(checks)} checks passed; report: {args.run / 'report.md'}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
