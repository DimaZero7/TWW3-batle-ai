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
    python -m tools.build human --army-seed 1000900014 --army-swap   # a human plays our side (x1, recorded)
    python -m tools.build nn-arena --army-from-run build/nn-arena/runs/20261007-091416 --army-swap   # an old run's armies
    python -m tools.build lord-fall --faction skv --treatment kill   # the lord killed / routed (tools/nn/lord_fall.py)
    python -m tools.build lord-duel --duel emp   # our network's lord v a lord under one attack order (tools/nn/lord_duel.py)
    python -m tools.build lord-duel --duel skv --duel-variant escort   # the same, each lord with 2 infantry units
    python -m tools.build lord-ai --duel skv --own-role defend   # a scripted lord v the game AI's lord (tools/nn/lord_ai.py)
    python -m tools.build charge-probe --probe-plan hit --probe-battle 1   # the melee probe (tools/nn/charge_probe.py)
    python -m tools.build missile-probe --mprobe-plan dist --mprobe-battle 1   # the missile probe (tools/nn/missile_probe.py)
    python -m tools.build morale-probe --morale-plan flank --morale-battle 1   # the morale probe (tools/nn/morale_probe.py)

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
    # A human plays our side of an arena battle (entries.nn_arena, own_ai 'human'): its own build folder,
    # so the gate's and the arena's runs (build/nn-arena/runs) never mix with a human's.
    "human": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_human.pack",
        "script": "tww3_bai_human",
        "folder": "tww3_bai",
        "scenario": "nn_arena.xml",
        "packed_scenario": "human.xml",
    },
    "lord-swarm": {
        "entry": "entries.lord_swarm",
        "pack": "tww3_bai_lord_swarm.pack",
        "script": "tww3_bai_lord_swarm",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "lord_swarm.xml",
    },
    # The melee probe (one melee indicator in lanes, game and simulator alike): tools/nn/charge_probe.py.
    "charge-probe": {
        "entry": "entries.charge_probe",
        "pack": "tww3_bai_charge_probe.pack",
        "script": "tww3_bai_charge_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "charge_probe.xml",
    },
    # The missile probe (one shooting indicator in lanes, game and simulator alike): tools/nn/missile_probe.py.
    "missile-probe": {
        "entry": "entries.missile_probe",
        "pack": "tww3_bai_missile_probe.pack",
        "script": "tww3_bai_missile_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "missile_probe.xml",
    },
    # The morale probe (one morale term in lanes, game and simulator alike): tools/nn/morale_probe.py.
    "morale-probe": {
        "entry": "entries.morale_probe",
        "pack": "tww3_bai_morale_probe.pack",
        "script": "tww3_bai_morale_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "morale_probe.xml",
    },
    "lord-fall": {
        "entry": "entries.lord_fall",
        "pack": "tww3_bai_lord_fall.pack",
        "script": "tww3_bai_lord_fall",
        "folder": "tww3_bai",
        "scenario": "lord_fall.xml",
        "packed_scenario": "lord_fall.xml",
    },
    # The lord duel (entries.nn_arena, enemy_ai 'scripted'): its own build folder, so its runs never mix
    # with the arena's and the gate's (build/nn-arena/runs: data for the simulator).
    "lord-duel": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_lord_duel.pack",
        "script": "tww3_bai_lord_duel",
        "folder": "tww3_bai",
        "scenario": "lord_duel.xml",
        "packed_scenario": "lord_duel.xml",
    },
    # The lord against the game's AI (entries.nn_arena, side 1 one scripted attack order, side 2 the game's
    # battle AI or (the control) scripted; both lords' cards and abilities recorded): tools/nn/lord_ai.py.
    "lord-ai": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_lord_ai.pack",
        "script": "tww3_bai_lord_ai",
        "folder": "tww3_bai",
        "scenario": "lord_duel.xml",
        "packed_scenario": "lord_ai.xml",
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
# human: every soldier's position every this many ticks (a reading of all men takes ~50 ms).
HUMAN_SOLDIERS_EVERY = 5
ARENA_TARGETS = ("nn-arena", "human")
# lord-fall: ms between samples (the moment's step is read at +1, +2 s).
LORD_FALL_TICK_MS = 500


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
                "net": args.own_role or "defend", "human": args.own_role or "defend"}[args.own_ai]
    enemy_role = "defend" if own_role == "attack" else "attack"
    defender = "enemy" if enemy_role == "defend" else "own"
    duration_s = max(3600, args.timeout + TIMEOUT_MARGIN_S)
    path = None
    if args.army_from_run is not None:
        # an old run's armies and deployment (tools/nn/army_from_run.py): its manifest, not the generator
        from tools.nn import army_from_run
        arena, army = army_from_run.load(args.army_from_run, swap=args.army_swap)
    if args.army_seed is not None:
        from tools.nn.armies import generate
        arena = generate.battle(args.army_seed)
        if args.army_swap:
            # the other half of a swapped pair (tools/nn/gate.py): our network takes the other army
            arena = dict(arena, name=f"{arena['name']}_swap",
                         sides={"own": arena["sides"]["enemy"], "enemy": arena["sides"]["own"]})
        army = {"seed": args.army_seed, "split": generate.split(args.army_seed), "swap": args.army_swap,
                "budget": arena["budget"],
                "side_budget": {s: arena["sides"][s]["budget"] for s in nn_scenario.SIDES},
                "template": {s: arena["sides"][s]["army"] for s in nn_scenario.SIDES}}
    if args.army_seed is not None or args.army_from_run is not None:
        if args.army_add or args.army_remove:
            # our side's army changed by hand (tools/nn/army_edit.py): a battle for the game only
            from tools.nn import army_edit
            arena = army_edit.edit(arena, "own", add=[army_edit.parse(x) for x in args.army_add or ()],
                                   remove=[army_edit.parse(x) for x in args.army_remove or ()])
        # A generated battle is not a scenario of the repository: its file goes next to the build.
        path = project.BUILD / args.target / f"{arena['name']}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        nn_scenario.write_scenario(defender, arena, path, duration_s)
        run_config["army"] = dict(army, cost={s: arena["sides"][s]["cost"] for s in nn_scenario.SIDES},
                                  men={s: sum(u["men"] for u in arena["sides"][s]["units"])
                                       for s in nn_scenario.SIDES})
        if "edit" in arena["sides"]["own"]:
            run_config["army"]["edit"] = arena["sides"]["own"]["edit"]
    elif args.target == "nn-arena":
        arena = nn_scenario.write_scenario(defender, nn_scenario.load_arena(args.arena), duration_s=duration_s)
    else:
        # the human target never rewrites scenarios/nn_arena.xml: its battle file goes next to its build
        arena = nn_scenario.load_arena(args.arena)
        path = project.BUILD / args.target / f"{arena['name']}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        nn_scenario.write_scenario(defender, arena, path, duration_s)
    run_config.update(nn_scenario.run_config(arena), own_ai=args.own_ai, enemy_role=enemy_role)
    if args.own_ai == "net":
        run_config.update(own_role=own_role, decide_ms=args.decide_ms, poll_ms=NET_POLL_MS)
    if args.own_ai == "human":
        run_config.update(own_role=own_role, soldiers_every=args.soldiers_every)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--speed", type=int, choices=(1, 3, 10, 20), help="battle speed (default 20; human: 1)")
    parser.add_argument("--timeout", type=int, help="model seconds, 30..3600 (default 600; human: 3600, the gate's)")
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
    parser.add_argument("--own-ai", choices=("attack", "defend", "hold", "net", "scripted"),
                        help="nn-arena: CA's script AI planner attacks or defends with our side; "
                             "the game's AI does the other; hold: our side gets no orders and stands "
                             "(a target for the game's AI to attack); net: the network in the companion "
                             "(tools/nn/companion) commands our side (its role: --own-role); "
                             "scripted (lord-duel): one attack order on the nearest enemy. "
                             "Default: net with --army-seed, else attack; lord-duel: net")
    parser.add_argument("--own-role", choices=("attack", "defend"),
                        help="nn-arena --own-ai net, human: our side attacks (the game's AI defends and wins on "
                             "timeout) or defends (default: the game's AI attacks)")
    parser.add_argument("--soldiers-every", type=int, default=HUMAN_SOLDIERS_EVERY,
                        help="human: every soldier's position every this many ticks (0: never)")
    parser.add_argument("--army-seed", type=int,
                        help="nn-arena: a generated battle (tools/nn/armies, generate.battle(seed)): "
                             "armies of a lord and 0-19 units a side; EVAL seeds for checks")
    parser.add_argument("--army-swap", action="store_true",
                        help="nn-arena --army-seed: the seed's armies swapped (our side gets the generator's "
                             "enemy army): the second battle of a swapped pair (tools/nn/gate.py)")
    parser.add_argument("--army-from-run", type=Path, metavar="RUN_DIR",
                        help="nn-arena: the armies, deployment, gap and defend radius of an old arena run "
                             "(its manifest.json, e.g. build/nn-arena/runs/20261007-091416) instead of the "
                             "generator: a battle of an older generator again; --army-swap exchanges its sides "
                             "(tools/nn/army_from_run.py)")
    parser.add_argument("--army-add", action="append", metavar="KEY[=N]",
                        help="nn-arena --army-seed: add N (1) units of KEY (a unit of the side's pool, "
                             "config/nn/pools.json) to our side (after --army-swap), deployed again by the "
                             "generator's rule; repeatable. The file gets '_edit' (tools/nn/army_edit.py)")
    parser.add_argument("--army-remove", action="append", metavar="KEY[=N]",
                        help="nn-arena --army-seed: remove N (1) units of KEY from our side; repeatable")
    parser.add_argument("--decide-ms", type=int, default=1000,
                        help="nn-arena --own-ai net: model ms between two decisions (250..5000)")
    parser.add_argument("--arena", default="arena",
                        help="nn-arena: 'arena' (config/nn/arena.json, the same army on both sides) "
                             "or a named arena in config/nn/arenas.json")
    parser.add_argument("--repeats", type=int, default=2,
                        help="lord-swarm: how many times each layout runs in the battle")
    parser.add_argument("--probe-plan", choices=("charge", "hit", "move", "vv", "syg2", "pair", "fatleave", "fresh", "meleeorders", "damaged", "reform", "reform2"), default="charge",
                        help="charge-probe: the plan (tools/nn/charge_probe.py)")
    parser.add_argument("--probe-battle", type=int, default=1, help="charge-probe: the plan's battle, 1-based")
    parser.add_argument("--mprobe-plan", choices=("dist", "arc", "range", "targets", "shield", "moving", "rank", "thin", "pistol", "moving2", "lof",
                                                    "newdist", "starsmove", "meleefire", "hglof", "rangenew", "lofab", "lofthresh", "lofheight", "rangestand"),
                        default="dist", help="missile-probe: the plan (tools/nn/missile_probe.py)")
    parser.add_argument("--mprobe-battle", type=int, default=1, help="missile-probe: the plan's battle, 1-based")
    parser.add_argument("--morale-plan", choices=("flank", "charge", "secure", "shoot", "rally", "strong", "penitent", "aura", "strong2", "rally2"),
                        default="flank",
                        help="morale-probe: the plan (tools/nn/morale_probe.py)")
    parser.add_argument("--morale-battle", type=int, default=1, help="morale-probe: the plan's battle, 1-based")
    parser.add_argument("--swarm", dest="plan_swarm", choices=("infantry", "lords", "all", "damage"), default="infantry",
                        help="lord-swarm: infantry around each lord, the other lord (with units) on him, both, or the "
                             "damage plan (one swordsmen / greatswords / clanrat / Stormvermin unit on a lord)")
    parser.add_argument("--faction", choices=("emp", "skv", "vmp"), default="emp",
                        help="lord-fall: the treated army (its fearless opponent: Skaven for emp, else Empire)")
    parser.add_argument("--treatment", choices=("kill", "rout", "none"), default="none",
                        help="lord-fall: the treated lord killed, routed, or left alone (the control)")
    parser.add_argument("--duel", choices=("emp", "skv"), default="emp",
                        help="lord-duel, lord-ai: both lords Empire Generals or Skaven Warlords")
    parser.add_argument("--duel-enemy", choices=("game", "scripted"), default="game",
                        help="lord-ai: the other lord under the game's battle AI, or under one attack order (the control)")
    parser.add_argument("--duel-variant", choices=("solo", "escort"), default="solo",
                        help="lord-duel: the lords alone, or each with 2 infantry units of his faction (escort)")
    parser.add_argument("--features", action="store_true",
                        help="map-capture: also read objects and reachability after deployment")
    args = parser.parse_args(argv)
    if args.target == "human":
        if args.own_ai not in (None, "human"):
            parser.error("the human target is our side under a human: no --own-ai")
        args.own_ai = "human"
    if args.speed is None:
        args.speed = 1 if args.target == "human" else 20
    if args.timeout is None:
        args.timeout = {"human": 3600, "lord-duel": 900, "lord-ai": 900}.get(args.target, 600)
    if args.own_ai is None:
        args.own_ai = ("scripted" if args.target == "lord-ai"
                       else "net" if (args.army_seed is not None or args.army_from_run is not None
                                      or args.target == "lord-duel") else "attack")
    if args.target == "lord-duel" and args.own_ai not in ("net", "scripted"):
        parser.error("lord-duel: --own-ai net (the network) or scripted (the control)")
    if args.target == "lord-ai" and args.own_ai != "scripted":
        parser.error("lord-ai: our lord is scripted (one attack order)")
    if args.own_ai == "scripted" and args.target not in ("lord-duel", "lord-ai"):
        parser.error("--own-ai scripted is for lord-duel and lord-ai")
    if args.own_role and args.own_ai not in ("net", "human") and args.target not in ("lord-duel", "lord-ai"):
        parser.error("--own-role is for --own-ai net and the human target (the planner modes set our role themselves)")
    if args.soldiers_every < 0:
        parser.error("--soldiers-every must be 0 or more")
    if args.army_from_run is not None:
        if args.target not in ARENA_TARGETS:
            parser.error("--army-from-run is for nn-arena and human")
        if args.army_seed is not None or args.arena != "arena":
            parser.error("--army-from-run, --army-seed and --arena are sources of armies: give one")
        if not (args.army_from_run / "manifest.json").exists():
            parser.error(f"--army-from-run: no manifest.json in {args.army_from_run}")
    generated = args.army_seed is not None or args.army_from_run is not None
    if args.army_swap and not generated:
        parser.error("--army-swap is for --army-seed and --army-from-run")
    if (args.army_add or args.army_remove) and not generated:
        parser.error("--army-add/--army-remove are for --army-seed and --army-from-run")
    for spec in (args.army_add or []) + (args.army_remove or []):
        from tools.nn import army_edit
        try:
            army_edit.parse(spec)
        except ValueError:
            parser.error(f"--army-add/--army-remove {spec!r}: KEY or KEY=N with N >= 1")
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
        tick_ms = args.tick_ms or {"lord-swarm": LORD_SWARM_TICK_MS, "lord-fall": LORD_FALL_TICK_MS,
                                   "charge-probe": 500, "missile-probe": 500,
                                   "morale-probe": 500}.get(args.target, 1000)
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
        if args.target in ARENA_TARGETS:
            scenario_file = nn_arena_config(args, run_config)
        if args.target == "lord-swarm":
            from tools.nn import lord_swarm
            if args.plan_swarm == "damage":
                # its own armies: the battle file goes next to the build, scenarios/lord_swarm.xml stays
                path = project.BUILD / "lord-swarm" / "lord_swarm_damage.xml"
                path.parent.mkdir(parents=True, exist_ok=True)
                lord_swarm.write_scenario(path, plan="damage")
                if not args.scenario:
                    args.scenario = str(path)
            else:
                lord_swarm.write_scenario()
            swarm, model_s = lord_swarm.run_config(args.repeats, plan=args.plan_swarm)
            run_config.update(swarm)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target == "charge-probe":
            from tools.nn import charge_probe
            path = charge_probe.write_scenario(args.probe_plan, args.probe_battle)
            probe, model_s, _ = charge_probe.run_config(args.probe_plan, args.probe_battle)
            run_config.update(probe)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "missile-probe":
            from tools.nn import missile_probe
            path = missile_probe.write_scenario(args.mprobe_plan, args.mprobe_battle)
            probe, model_s, _, _ = missile_probe.run_config(args.mprobe_plan, args.mprobe_battle)
            run_config.update(probe)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "morale-probe":
            from tools.nn import morale_probe
            path = morale_probe.write_scenario(args.morale_plan, args.morale_battle)
            probe, model_s, _ = morale_probe.run_config(args.morale_plan, args.morale_battle)
            run_config.update(probe)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "lord-fall":
            from tools.nn import lord_fall
            path = lord_fall.write_scenario(args.faction)
            fall, model_s = lord_fall.run_config(args.faction, args.treatment)
            run_config.update(fall)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "lord-duel":
            from tools.nn import lord_duel
            path, duel = lord_duel.build_config(args.duel, args.own_ai, args.own_role or "attack", args.decide_ms,
                                                NET_POLL_MS, args.timeout, args.duel_variant)
            run_config.update(duel)
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "lord-ai":
            from tools.nn import lord_ai
            path, duel = lord_ai.build_config(args.duel, args.own_role or "attack", args.duel_enemy, args.timeout)
            run_config.update(duel)
            if not args.scenario:
                args.scenario = str(path)
        if args.target == "manual":
            # The player sets the pace: no forced speed, an hour by default.
            run_config.pop("speed")
            model_s = None
        run_config["deadline_s"] = args.deadline or (3600 if model_s is None else deadline_seconds(model_s, args.speed))
        run_config["stall_ms"] = stall_ms
    if args.target in ARENA_TARGETS and scenario_file and not args.scenario:
        args.scenario = str(scenario_file)
    manifest = build(args.target, run_config, scenario=args.scenario)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
