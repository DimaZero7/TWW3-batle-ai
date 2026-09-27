"""Check a unit-readout run against the documented unit readouts.

Usage: python -m tools.analysis.unit_readout build/unit-readout/runs/<time>

Reads events.jsonl written by src/entries/unit_readout.lua and writes
report.md and summary.json next to it. Expectations come from
docs/*/game/units (states.md, state-sensors.md, missile-range.md).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

UNITS = ["a_general", "a_spears", "a_archers", "b_general", "b_spears", "b_archers"]
KIND = {name: name.split("_", 1)[1] for name in UNITS}  # general / spears / archers

# docs/*/game/units/states.md, "What the game returned for each unit".
PROFILE = {
    "general": {"type": "wh_main_emp_cha_general_0", "initial_men": 1, "attribute_hide_forest": True,
                "attribute_charge_defense_vs_large": False, "attribute_charge_reflection": False,
                "attribute_encourages": True, "can_defend": True, "can_fire_at_will": False,
                "can_skirmish": False, "can_change_formation_spacing": False},
    "spears": {"type": "wh_main_emp_inf_spearmen_0", "initial_men": 120, "missile_range": 0,
               "attribute_hide_forest": True, "attribute_charge_defense_vs_large": True,
               "attribute_charge_reflection": True, "attribute_encourages": False, "can_defend": True,
               "can_fire_at_will": False, "can_skirmish": False, "can_change_formation_spacing": False},
    "archers": {"type": "wh2_dlc13_emp_inf_archers_0", "initial_men": 90, "missile_range": 130,
                "attribute_hide_forest": True, "attribute_charge_defense_vs_large": False,
                "attribute_charge_reflection": False, "attribute_encourages": False, "can_defend": True,
                "can_fire_at_will": True, "can_skirmish": True, "can_change_formation_spacing": False},
}


def load(run):
    return [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]


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
    stages = [r["name"] for r in events["stage"]]
    check("run", "all stages reached", ["idle", "march", "ranged", "cease_fire", "melee", "halt"], stages)

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
        ("melee: a_spears number_of_men_alive decreases", drop("a_spears", "native.number_of_men_alive", "cease_fire", "melee")),
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
    spear_rows = [r for r in ranges[("a_spears", "b_spears")] if r["readings"]["access"] == "allowed"]
    check("range", "a_spears missile_range_m = 0", [0],
          sorted({known(r, "missile_range_m") for r in spear_rows} - {None}),
          {known(r, "missile_range_m") for r in spear_rows} - {None} == {0})
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
    check("telemetry", "sampler frames", ">0", len(events["sampler_frame"]), bool(events["sampler_frame"]))
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
             f"{len(events['intel'])} intel, {len(events['sampler_frame'])} sampler frames.", "",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    checks, table, events = analyse(load(args.run))
    passed = write_report(args.run, checks, table, events)
    for c in checks:
        if c["result"] != "pass":
            print("FAIL", c["group"], c["check"], "expected", c["expected"], "actual", c["actual"])
    print(f"{passed}/{len(checks)} checks passed; report: {args.run / 'report.md'}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
