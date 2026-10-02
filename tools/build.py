"""Build a battle pack from src/ for one target. Never touches the game.

Usage:
    python -m tools.build ai-vs-ai --speed 3
    python -m tools.build map-capture --step 3 --features
    python -m tools.build move-probe --plan hamlet
    python -m tools.build manual --deadline 3600 --stall-minutes 30
    python -m tools.build roster-capture        # scenario from config/roster/capture.json
    python -m tools.build archer-range --range-mode damage
    python -m tools.build enemy-layout --layout balanced_5
    python -m tools.build nn-arena --own-ai defend --timeout 900
    python -m tools.build nn-arena --arena pair_spear_v_slave --own-ai attack   # config/nn/arenas.json
    python -m tools.build nn-arena --own-ai net --speed 1   # the network commands our side (companion)
    python -m tools.build nn-arena --army-seed 1000900000 --own-role attack   # a generated battle, the network
    python -m tools.build lord-swarm --repeats 2   # a lord swarmed by 1-4 units (tools/nn/lord_swarm.py)

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
    "nn-arena": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_nn_arena.pack",
        "script": "tww3_bai_nn_arena",
        "folder": "tww3_bai",
        "scenario": "nn_arena.xml",
        "packed_scenario": "nn_arena.xml",
    },
    "lord-swarm": {
        "entry": "entries.lord_swarm",
        "pack": "tww3_bai_lord_swarm.pack",
        "script": "tww3_bai_lord_swarm",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "lord_swarm.xml",
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
# enemy-layout: game time the game AI army is watched after deployment.
ENEMY_LAYOUT_HOLD_S = 90
# nn-arena --own-ai net: how often the companion's orders file is read (model ms).
NET_POLL_MS = 100
# lord-swarm: ms between samples of the lords and their attackers.
LORD_SWARM_TICK_MS = 200
# nn-arena: the battle file's own time limit is past the script's (the script ends the battle first).
TIMEOUT_MARGIN_S = 60


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


def build(target, run_config, dependencies=None, scenario=None):
    """dependencies defaults to the required mods; every battle loads them.
    scenario overrides the target's scenario file (a name in scenarios/)."""
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


def nn_arena_config(args, run_config):
    """nn-arena: writes the battle file and fills run_config (units, roles, the generated armies).
    Returns the path of a battle file written outside scenarios/ (a generated battle), else None."""
    from tools.nn import scenario as nn_scenario
    # The side that wins on timeout defends: ours when the planner defends,
    # the game's AI when ours attacks; the network's role is --own-role (default defend).
    own_role = {"attack": "attack", "defend": "defend", "hold": "defend",
                "net": args.own_role or "defend"}[args.own_ai]
    enemy_role = "defend" if own_role == "attack" else "attack"
    defender = "enemy" if enemy_role == "defend" else "own"
    duration_s = max(3600, args.timeout + TIMEOUT_MARGIN_S)
    path = None
    if args.army_seed is not None:
        from tools.nn.armies import generate
        arena = generate.battle(args.army_seed)
        if args.army_swap:
            # the other half of a swapped pair (tools/nn/gate.py): our network takes the other army
            arena = dict(arena, name=f"{arena['name']}_swap",
                         sides={"own": arena["sides"]["enemy"], "enemy": arena["sides"]["own"]})
        # A generated battle is not a scenario of the repository: its file goes next to the build.
        path = project.BUILD / "nn-arena" / f"{arena['name']}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        nn_scenario.write_scenario(defender, arena, path, duration_s)
        run_config["army"] = {"seed": args.army_seed, "split": generate.split(args.army_seed), "swap": args.army_swap,
                              "budget": arena["budget"],
                              "side_budget": {s: arena["sides"][s]["budget"] for s in nn_scenario.SIDES},
                              "template": {s: arena["sides"][s]["army"] for s in nn_scenario.SIDES},
                              "cost": {s: arena["sides"][s]["cost"] for s in nn_scenario.SIDES},
                              "men": {s: sum(u["men"] for u in arena["sides"][s]["units"])
                                      for s in nn_scenario.SIDES}}
    else:
        arena = nn_scenario.write_scenario(defender, nn_scenario.load_arena(args.arena), duration_s=duration_s)
    run_config.update(nn_scenario.run_config(arena), own_ai=args.own_ai, enemy_role=enemy_role)
    if args.own_ai == "net":
        run_config.update(own_role=own_role, decide_ms=args.decide_ms, poll_ms=NET_POLL_MS)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--speed", type=int, choices=(1, 3, 10, 20), default=20)
    parser.add_argument("--timeout", type=int, default=600, help="model seconds, 30..3600")
    parser.add_argument("--tick-ms", type=int, help="ms between samples (default 1000; lord-swarm 200)")
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
    parser.add_argument("--layout", default="balanced_5", help="enemy-layout: layout in config/armies/defender_layouts.json")
    parser.add_argument("--enemy-mode", choices=("native", "defend"), default="defend",
                        help="enemy-layout: game AI as set by the battle, or told to defend where it deployed")
    parser.add_argument("--range-mode", choices=("fire_at_will", "attack", "damage"), default="fire_at_will",
                        help="archer-range: archers stand with fire at will (the target steps closer), attack it, "
                             "or shoot a fearless target at fixed distances until out of arrows (damage)")
    parser.add_argument("--damage-rotate", type=int, default=0,
                        help="archer-range --range-mode damage: shift the distances by this many lanes")
    parser.add_argument("--own-ai", choices=("attack", "defend", "hold", "net"),
                        help="nn-arena: CA's script AI planner attacks or defends with our side; "
                             "the game's AI does the other; hold: our side gets no orders and stands "
                             "(a target for the game's AI to attack); net: the network in the companion "
                             "(tools/nn/companion) commands our side (its role: --own-role). "
                             "Default: net with --army-seed, else attack")
    parser.add_argument("--own-role", choices=("attack", "defend"),
                        help="nn-arena --own-ai net: our side attacks (the game's AI defends and wins on "
                             "timeout) or defends (default: the game's AI attacks)")
    parser.add_argument("--army-seed", type=int,
                        help="nn-arena: a generated battle (tools/nn/armies, generate.battle(seed)): "
                             "armies of a lord and 0-19 units a side; EVAL seeds for checks")
    parser.add_argument("--army-swap", action="store_true",
                        help="nn-arena --army-seed: the seed's armies swapped (our side gets the generator's "
                             "enemy army): the second battle of a swapped pair (tools/nn/gate.py)")
    parser.add_argument("--decide-ms", type=int, default=1000,
                        help="nn-arena --own-ai net: model ms between two decisions (250..5000)")
    parser.add_argument("--arena", default="arena",
                        help="nn-arena: 'arena' (config/nn/arena.json, the same army on both sides) "
                             "or a named arena in config/nn/arenas.json")
    parser.add_argument("--repeats", type=int, default=2,
                        help="lord-swarm: how many times each layout runs in the battle")
    parser.add_argument("--swarm", dest="plan_swarm", choices=("infantry", "lords", "all"), default="infantry",
                        help="lord-swarm: infantry around each lord, the other lord (with units) on him, or both")
    parser.add_argument("--features", action="store_true",
                        help="map-capture: also read objects and reachability after deployment")
    args = parser.parse_args(argv)
    if args.own_ai is None:
        args.own_ai = "net" if args.army_seed is not None else "attack"
    if args.own_role and args.own_ai != "net":
        parser.error("--own-role is for --own-ai net (the planner modes set our role themselves)")
    if args.army_swap and args.army_seed is None:
        parser.error("--army-swap is for --army-seed")
    if args.army_seed is not None and args.arena != "arena":
        parser.error("--army-seed and --arena are two sources of armies: give one")
    if not 30 <= args.timeout <= 3600:
        parser.error("--timeout must be between 30 and 3600 seconds")
    if not 250 <= args.decide_ms <= 5000:
        parser.error("--decide-ms must be between 250 and 5000")

    if args.target == "map-capture":
        run_config = {"step": args.step, "features": args.features}
        if args.window:
            run_config["window"] = dict(zip(("min_x", "max_x", "min_z", "max_z"), args.window))
    else:
        scenario_file = None
        tick_ms = args.tick_ms or (LORD_SWARM_TICK_MS if args.target == "lord-swarm" else 1000)
        run_config = {"speed": args.speed, "timeout_ms": args.timeout * 1000, "tick_ms": tick_ms,
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
        if args.target == "archer-range":
            from tools import archer_range
            archer_range.write_scenario()
            range_config, model_s = archer_range.run_config(args.range_mode, args.damage_rotate)
            run_config.update(range_config)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "enemy-layout":
            from tools import enemy_layout
            enemy_layout.write_scenario(args.layout)
            run_config.update(layout=args.layout, hold_s=ENEMY_LAYOUT_HOLD_S, enemy_mode=args.enemy_mode)
            model_s = ENEMY_LAYOUT_HOLD_S + 10
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "nn-arena":
            scenario_file = nn_arena_config(args, run_config)
        if args.target == "lord-swarm":
            from tools.nn import lord_swarm
            lord_swarm.write_scenario()
            swarm, model_s = lord_swarm.run_config(args.repeats, plan=args.plan_swarm)
            run_config.update(swarm)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "manual":
            # The player sets the pace: no forced speed, an hour by default.
            run_config.pop("speed")
            model_s = None
        run_config["deadline_s"] = args.deadline or (3600 if model_s is None else deadline_seconds(model_s, args.speed))
        run_config["stall_ms"] = stall_ms
    if args.target == "nn-arena" and scenario_file and not args.scenario:
        args.scenario = str(scenario_file)
    manifest = build(args.target, run_config, scenario=args.scenario)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
