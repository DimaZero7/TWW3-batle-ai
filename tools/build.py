"""Build a battle pack from src/ for one target. Never touches the game.

Usage:
    python -m tools.build duel --runs 3 --speed 20 --timeout 300
    python -m tools.build arena
    python -m tools.build ai-vs-ai --speed 3
    python -m tools.build map-capture --step 3 --features
    python -m tools.build move-probe --plan hamlet
    python -m tools.build manual --deadline 3600 --stall-minutes 30
    python -m tools.build roster-capture        # scenario from config/roster/capture.json
    python -m tools.build formation-probe --army first_attack
    python -m tools.build enemy-layout --layout balanced_5

Output: build/<target>/ with the .pack, the bundled script and manifest.json.
Install and launch with tools/launcher/launch.ps1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from tools import config as project
from tools.pack import bundle, pfh5

# target -> entry module, pack/script names, scenario XML and its in-pack name.
TARGETS = {
    "duel": {
        "entry": "entries.duel",
        "pack": "tww3_bai_duel.pack",
        "script": "tww3_bai_duel",
        "folder": "tww3_bai",
        "scenario": "ranged_melee.xml",
        "packed_scenario": "ranged_melee.xml",
    },
    "arena": {
        "entry": "entries.arena",
        "pack": "tww3_bai_arena.pack",
        "script": "tww3_bai_arena",
        "folder": "tww3_bai",
        "scenario": "triple_melee.xml",
        "packed_scenario": "triple_melee.xml",
    },
    "ai-vs-ai": {
        "entry": "entries.ai_vs_ai",
        "pack": "tww3_bai_ai_vs_ai.pack",
        "script": "tww3_bai_ai_vs_ai",
        "folder": "tww3_bai",
        "scenario": "ai_vs_ai.xml",
        "packed_scenario": "ai_vs_ai.xml",
    },
    "unit-readout": {
        "entry": "entries.unit_readout",
        "pack": "tww3_bai_unit_readout.pack",
        "script": "tww3_bai_unit_readout",
        "folder": "tww3_bai",
        "scenario": "unit_readout.xml",
        "packed_scenario": "unit_readout.xml",
    },
    "move-probe": {
        "entry": "entries.move_probe",
        "pack": "tww3_bai_move_probe.pack",
        "script": "tww3_bai_move_probe",
        "folder": "tww3_bai",
        "scenario": "move_probe.xml",
        "packed_scenario": "move_probe.xml",
    },
    "roster-capture": {
        "entry": "entries.roster_capture",
        "pack": "tww3_bai_roster_capture.pack",
        "script": "tww3_bai_roster_capture",
        "folder": "tww3_bai",
        "scenario": "roster_capture.xml",
        "packed_scenario": "roster_capture.xml",
    },
    "archer-range": {
        "entry": "entries.archer_range",
        "pack": "tww3_bai_archer_range.pack",
        "script": "tww3_bai_archer_range",
        "folder": "tww3_bai",
        "scenario": "archer_range.xml",
        "packed_scenario": "archer_range.xml",
    },
    "formation-probe": {
        "entry": "entries.formation_probe",
        "pack": "tww3_bai_formation_probe.pack",
        "script": "tww3_bai_formation_probe",
        "folder": "tww3_bai",
        "scenario": "formation_probe.xml",
        "packed_scenario": "formation_probe.xml",
    },
    "enemy-layout": {
        "entry": "entries.enemy_layout",
        "pack": "tww3_bai_enemy_layout.pack",
        "script": "tww3_bai_enemy_layout",
        "folder": "tww3_bai",
        "scenario": "enemy_layout.xml",
        "packed_scenario": "enemy_layout.xml",
    },
    "manual": {
        "entry": "entries.manual_record",
        "pack": "tww3_bai_manual.pack",
        "script": "tww3_bai_manual",
        "folder": "tww3_bai",
        "scenario": "manual_hamlet.xml",
        "packed_scenario": "manual_hamlet.xml",
    },
    "map-capture": {
        "entry": "entries.map_capture",
        "pack": "tww3_bai_map_capture.pack",
        "script": "tww3_bai_map_capture",
        "folder": "tww3_bai_map_capture",
        "scenario": "map_capture.xml",
        "packed_scenario": "map_probe.xml",
    },
}

SCENARIO_LUA = b"load_script_libraries()\n"

# Scripted length of unit-readout (entries/unit_readout.lua, M.STAGES 'done').
READOUT_MODEL_S = 196
# move-probe: model time between legs (teleport, then the formation settles).
MOVE_SETTLE_MS = 2000
# formation-probe: least time per stage and its limit (entries/formation_probe.lua, 9 stages).
FORMATION_SETTLE_MS = 3000
FORMATION_STAGE_S = 20
# enemy-layout: game time the game AI army is watched after deployment.
ENEMY_LAYOUT_HOLD_S = 90
# formation-probe approach: one manoeuvre at most, the whole approach at most.
FORMATION_MANOEUVRE_S = 360
FORMATION_APPROACH_S = 900
# formation-probe: longest walk to the aligned place (walking pace 1.5 m/s).
FORMATION_ALIGN_S = 120
# formation-probe: the army stands this long after placing and must not move.
FORMATION_HOLD_S = 60


def load_move_plan(name):
    """config/move-plans/<name>.json; every leg must be complete."""
    plan = json.loads((project.CONFIG_DIR / "move-plans" / f"{name}.json").read_text(encoding="utf-8"))
    assert plan.get("legs"), f"move plan {name} has no legs"
    for leg in plan["legs"]:
        assert leg["kind"] in ("shape", "traverse"), leg
        assert leg["timeout_s"] > 0 and isinstance(leg["run"], bool), leg
        for point in (leg["start"], leg["target"]):
            assert all(isinstance(point[k], (int, float)) for k in ("x", "z", "facing", "width")), leg
    return plan


def move_plan_model_s(plan):
    """Longest scripted length of a plan: every leg to its timeout."""
    return sum(leg["timeout_s"] + MOVE_SETTLE_MS / 1000 for leg in plan["legs"])


def deadline_seconds(model_s, speed):
    """Real-time limit for one battle: the model time at the requested speed
    (engines reach ~x19 of x20), doubled, plus a minute for slow-downs."""
    return int(model_s / speed * 2) + 60


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_syntax(script):
    """Compiles the bundle with Lua 5.1 (lupa) when available."""
    try:
        from lupa.lua51 import LuaRuntime
    except ImportError:
        return False
    LuaRuntime().eval("function(s) local f, e = loadstring(s); assert(f, e) end")(script)
    return True


def build(target, run_config, dependencies=None, scenario=None, plain=False):
    """dependencies defaults to the required mods; every battle loads them.
    scenario overrides the target's scenario file (a name in scenarios/).
    plain: the battle only, without our script (the player's own test)."""
    if dependencies is None:
        dependencies = project.required_mods()
    spec = dict(TARGETS[target])
    if scenario:
        spec["scenario"] = scenario
    scenario_xml = (project.SCENARIOS / spec["scenario"]).read_bytes()
    # The build id covers every input, so telemetry rows name the exact code.
    probe_script, modules = bundle.bundle(project.SRC, spec["entry"], run_config)
    digest = sha256(probe_script.encode("utf-8") + scenario_xml)[:16]
    run_config = dict(run_config, build=digest)
    header = f"-- Generated by tools/build.py ({target}, build {digest}). Edit src/, not this file.\n"
    script, _ = bundle.bundle(project.SRC, spec["entry"], run_config, header)
    script_bytes = script.encode("utf-8")
    syntax_checked = check_syntax(script)

    folder = spec["folder"]
    files = {
        f"script\\battle\\mod\\{spec['script']}.lua": script_bytes,
        f"script\\battle\\{folder}\\scenario.lua": SCENARIO_LUA,
        f"script\\battle\\{folder}\\{spec['packed_scenario']}": scenario_xml,
    }
    if plain:
        del files[f"script\\battle\\mod\\{spec['script']}.lua"]
    dependency_names = [d["pack"] for d in dependencies]
    blob = pfh5.pack_files(files, dependency_names)
    assert pfh5.read_pack(blob) == (files, tuple(dependency_names))

    out = project.BUILD / target
    out.mkdir(parents=True, exist_ok=True)
    (out / spec["pack"]).write_bytes(blob)
    (out / f"{spec['script']}.lua").write_bytes(script_bytes)
    manifest = {
        "target": target,
        "build": digest,
        "config": run_config,
        "modules": modules,
        "pack": spec["pack"],
        "pack_sha256": sha256(blob),
        "entries": list(files),
        "scenario": f"script/battle/{folder}/{spec['packed_scenario']}",
        "mod_profile": project.mod_profile(),
        "dependencies": list(dependencies),
        "syntax_checked": syntax_checked,
        "engine_tested": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--runs", type=int, choices=range(1, 11), default=3, help="duel: battles in a row")
    parser.add_argument("--speed", type=int, choices=(1, 3, 10, 20), default=20)
    parser.add_argument("--timeout", type=int, default=600, help="model seconds, 30..1800")
    parser.add_argument("--tick-ms", type=int, default=1000)
    parser.add_argument("--step", type=int, choices=(1, 2, 3, 5), default=5, help="map-capture: cell size, m")
    parser.add_argument("--deadline", type=int, help="real seconds per battle before the script ends it "
                        "(default: from the scripted length and speed)")
    parser.add_argument("--stall-minutes", type=float, default=10,
                        help="end the battle when nobody takes damage for this much GAME time")
    parser.add_argument("--scenario", help="scenario file in scenarios/ instead of the target's default")
    parser.add_argument("--window", type=float, nargs=4, metavar=("MIN_X", "MAX_X", "MIN_Z", "MAX_Z"),
                        help="map-capture: capture only this area")
    parser.add_argument("--plan", default="hamlet", help="move-probe: plan in config/move-plans/")
    parser.add_argument("--capture", type=Path, help="roster-capture: capture list instead of config/roster/capture.json")
    parser.add_argument("--army", default="first_attack", help="formation-probe: army in config/armies/")
    parser.add_argument("--layout", default="balanced_5", help="enemy-layout: layout in config/armies/defender_layouts.json")
    parser.add_argument("--enemy-mode", choices=("native", "defend"), default="defend",
                        help="enemy-layout: game AI as set by the battle, or told to defend where it deployed")
    parser.add_argument("--turn-test", action="store_true", help="formation-probe: also turn the archers right/left")
    parser.add_argument("--plain", action="store_true",
                        help="formation-probe: the battle only, without our script: the player deploys, the game's AI")
    parser.add_argument("--range-mode", choices=("fire_at_will", "attack", "damage"), default="fire_at_will",
                        help="archer-range: archers stand with fire at will (the target steps closer), attack it, "
                             "or shoot a fearless target at fixed distances until out of arrows (damage)")
    parser.add_argument("--damage-rotate", type=int, default=0,
                        help="archer-range --range-mode damage: shift the distances by this many lanes")
    parser.add_argument("--facing-sweep", action="store_true",
                        help="formation-probe: research — teleport one unit with a sweep of bearings and read its facing")
    parser.add_argument("--fast", action="store_true",
                        help="formation-probe --handover: keep --speed (research runs) instead of the player's pace")
    parser.add_argument("--fire", action="store_true",
                        help="formation-probe: our shooters fire at will; losses and arrows recorded while holding")
    parser.add_argument("--handover", action="store_true",
                        help="formation-probe: our AI places the army, then the player commands it; only recorded")
    parser.add_argument("--defend-radius", type=int, default=300,
                        help="formation-probe --handover --enemy-ai defend: the enemy's defence radius, m")
    parser.add_argument("--enemy-ai", choices=("native", "defend"),
                        help="formation-probe: the enemy to the game's AI, as the battle sets it or defending")
    parser.add_argument("--engine-only", action="store_true",
                        help="formation-probe: no queues past obstacles (apps.logistics off), the engine alone")
    parser.add_argument("--features", action="store_true",
                        help="map-capture: also read objects and reachability after deployment")
    args = parser.parse_args(argv)
    if not 30 <= args.timeout <= 1800:
        parser.error("--timeout must be between 30 and 1800 seconds")

    if args.target == "map-capture":
        run_config = {"step": args.step, "features": args.features}
        if args.window:
            run_config["window"] = dict(zip(("min_x", "max_x", "min_z", "max_z"), args.window))
    else:
        run_config = {"runs": args.runs if args.target == "duel" else 1, "speed": args.speed,
                      "timeout_ms": args.timeout * 1000, "tick_ms": args.tick_ms,
                      "scenario": TARGETS[args.target]["scenario"].removesuffix(".xml")}
        model_s = READOUT_MODEL_S if args.target == "unit-readout" else args.timeout
        stall_ms = int(args.stall_minutes * 60000)
        if args.target == "move-probe":
            plan = load_move_plan(args.plan)
            model_s = move_plan_model_s(plan)
            run_config.update(plan=plan, settle_ms=MOVE_SETTLE_MS)
            # Nobody takes damage in this probe: the stall rule must not cut the plan.
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "roster-capture":
            from tools import roster
            spec = roster.write_scenario(roster.load_capture(args.capture) if args.capture else None)
            capture, model_s = roster.run_config(spec, args.speed, MOVE_SETTLE_MS)
            run_config.update(capture)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "formation-probe":
            from tools.sim import formation as sim
            army, _ = sim.load_army(args.army)
            xml, probe_config, _ = sim.probe(army, args.speed, FORMATION_SETTLE_MS)
            (project.SCENARIOS / "formation_probe.xml").write_text(xml, encoding="utf-8")
            run_config.update(probe_config, stage_timeout_s=FORMATION_STAGE_S, army=args.army,
                              hold_s=FORMATION_HOLD_S, turn_test=args.turn_test, align_timeout_s=FORMATION_ALIGN_S,
                              manoeuvre_timeout_s=FORMATION_MANOEUVRE_S, approach_timeout_s=FORMATION_APPROACH_S)
            if args.engine_only:
                run_config["logistics"] = False
            # placed + align + mask (same limit) + hold.
            model_s = (FORMATION_STAGE_S + 2 * FORMATION_ALIGN_S + FORMATION_HOLD_S
                       + (FORMATION_APPROACH_S if probe_config.get("approach") else 0)
                       + (8 * FORMATION_STAGE_S if args.turn_test else 0) + 15)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if args.enemy_ai and not args.handover:
                # Our AI plays; the enemy is the game's AI (as the battle sets it, or told to defend).
                run_config.update(enemy_ai=args.enemy_ai, defend_radius_m=args.defend_radius)
            if args.fire:
                run_config["fire"] = True
            if args.handover:
                # The player's test: our AI places the army and hands it over;
                # the player's pace, an hour, nobody fighting does not end it.
                run_config.update(handover=True, enemy_ai=args.enemy_ai, defend_radius_m=args.defend_radius)
                if not args.fast:
                    run_config.pop("speed")
                model_s = None
                stall_ms = max(stall_ms, 3600000)
            if args.facing_sweep:
                # Research: the facing the engine gives for commanded bearings
                # (every 1 deg round the circle, every 0.1 deg around 90).
                bearings = [float(b) for b in range(0, 360)] + [round(80 + 0.1 * i, 1) for i in range(201)]
                run_config["facing_sweep"] = {"unit": "own_2", "ticks": 3, "bearings": bearings}
                stall_ms = max(stall_ms, 3600000)
        if args.target == "archer-range":
            from tools import archer_range
            archer_range.write_scenario()
            range_config, model_s = archer_range.run_config(args.range_mode, args.damage_rotate)
            run_config.update(range_config)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "enemy-layout":
            from tools import enemy_layout
            enemy_layout.write_scenario(args.layout)
            run_config.update(layout=args.layout, hold_s=ENEMY_LAYOUT_HOLD_S, enemy_mode=args.enemy_mode,
                              picture_every=5, roster=enemy_layout.roster_inputs(args.layout))
            model_s = ENEMY_LAYOUT_HOLD_S + 10
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "manual":
            # The player sets the pace: no forced speed, an hour by default.
            run_config.pop("speed")
            model_s = None
        run_config["deadline_s"] = args.deadline or (3600 if model_s is None else deadline_seconds(model_s, args.speed))
        run_config["stall_ms"] = stall_ms
    if args.plain:
        run_config["deadline_s"] = args.deadline or 3600
    manifest = build(args.target, run_config, scenario=args.scenario, plain=args.plain)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
