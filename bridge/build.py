"""Build a battle pack from bridge/lua/ for one target. Never touches the game.

Usage:
    python -m bridge.build nn-arena --army-seed 1000702112 --armies v3 --map ksl_plains_01 --deploy strategist \\
        --own-ai net --speed 10 --timeout 1200          # the chain's battle (bridge/launcher/watch.ps1 -Chain)
    python -m bridge.build nn-arena --army-from-run data/human/runs/20261107-171600 --map ksl_plains_01 \\
        --deploy strategist --own-ai net                 # an old run's armies and places (game/arena/army_from_run.py)
    python -m bridge.build human --army-seed 1000900014 --army-swap --live-label --free-speed   # the person plays
    python -m bridge.build charge-probe --probe-plan hit --probe-battle 1 --mem   # the melee probe (game/probe)
    python -m bridge.build missile-probe --mprobe-plan dist --mprobe-battle 1     # the missile probe
    python -m bridge.build morale-probe --morale-plan timer --kv waver_base_timeout=x10   # a probe with game rules changed
    python -m bridge.build map-capture --step 3 --nodes --features      # a map captured once (game/maps/map_grid.py)
    python -m bridge.build move-probe --plan ksl_detour                 # bridge/config/move-plans/<plan>.json

Output: build/<target>/ with the .pack, the bundled script and manifest.json.
Install and launch with bridge/launcher/launch.ps1 (the chain's battle: bridge/launcher/watch.ps1).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from bridge import config as project
from bridge.pack import bundle, kvtable, pfh5

# target -> entry module, pack/script names, scenario XML (bridge/scenarios) and its in-pack name.
TARGETS = {
    "nn-arena": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_nn_arena.pack",
        "script": "tww3_bai_nn_arena",
        "folder": "tww3_bai",
        "scenario": "nn_arena.xml",
        "packed_scenario": "nn_arena.xml",
    },
    # A human plays our side of an arena battle (entries.nn_arena, own_ai 'human'): its own build folder,
    # so the network's runs (build/nn-arena/runs) never mix with a human's.
    "human": {
        "entry": "entries.nn_arena",
        "pack": "tww3_bai_human.pack",
        "script": "tww3_bai_human",
        "folder": "tww3_bai",
        "scenario": "nn_arena.xml",
        "packed_scenario": "human.xml",
    },
    # The melee probe (one melee indicator in lanes, game and simulator alike): game/probe/charge_probe.py. The base
    # battle file lord_swarm.xml is rewritten by the probe's write_scenario (its lanes) next to the build.
    "charge-probe": {
        "entry": "entries.charge_probe",
        "pack": "tww3_bai_charge_probe.pack",
        "script": "tww3_bai_charge_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "charge_probe.xml",
    },
    # The missile probe: game/probe/missile_probe.py.
    "missile-probe": {
        "entry": "entries.missile_probe",
        "pack": "tww3_bai_missile_probe.pack",
        "script": "tww3_bai_missile_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "missile_probe.xml",
    },
    # The morale probe: game/probe/morale_probe.py.
    "morale-probe": {
        "entry": "entries.morale_probe",
        "pack": "tww3_bai_morale_probe.pack",
        "script": "tww3_bai_morale_probe",
        "folder": "tww3_bai",
        "scenario": "lord_swarm.xml",
        "packed_scenario": "morale_probe.xml",
    },
    "map-capture": {
        "entry": "entries.map_capture",
        "pack": "tww3_bai_map_capture.pack",
        "script": "tww3_bai_map_capture",
        "folder": "tww3_bai_map_capture",
        "scenario": "map_capture.xml",
        "packed_scenario": "map_probe.xml",
    },
    "move-probe": {
        "entry": "entries.move_probe",
        "pack": "tww3_bai_move_probe.pack",
        "script": "tww3_bai_move_probe",
        "folder": "tww3_bai",
        "scenario": "move_probe.xml",
        "packed_scenario": "move_probe.xml",
    },
}

SCENARIO_LUA = b"load_script_libraries()\n"

# move-probe: model time between legs (teleport, then the formation settles).
MOVE_SETTLE_MS = 2000
# nn-arena --own-ai net: how often the companion's orders file is read (model ms).
NET_POLL_MS = 100
# nn-arena: the battle file's own time limit is past the script's (the script ends the battle first).
TIMEOUT_MARGIN_S = 60
# nn-arena --deploy strategist: real seconds the deployment waits for the companion's plan (then as before: no plan),
# and real ms after the teleports before the places are read (the 26.09.2026 check read them 1000 ms after).
DEPLOY_WAIT_S = 120
DEPLOY_SETTLE_MS = 1000
# human: every soldier's position every this many ticks (a reading of all men takes ~50 ms).
HUMAN_SOLDIERS_EVERY = 5
ARENA_TARGETS = ("nn-arena", "human")
PROBE_TICK_MS = 500
# --kv (the game's rule rows changed in the pack, bridge/pack/kvtable.py): probes only, never a battle for data.
KV_TARGETS = ("charge-probe", "missile-probe", "morale-probe")
# --mem (memreader Plus: the game's memory read in a probe; bridge/docs/apps/memreader.md): probes whose entry reads it.
# Never the network's battles or exams.
MEM_TARGETS = ("missile-probe", "charge-probe", "morale-probe")
# the probes' modules (game/probe): their PLANS, run_config, write_scenario
PROBES = {"charge-probe": ("charge_probe", "probe_plan", "probe_battle"),
          "missile-probe": ("missile_probe", "mprobe_plan", "mprobe_battle"),
          "morale-probe": ("morale_probe", "morale_plan", "morale_battle")}
# the strategist's cases (their deployment zones: strategist_zone)
STRATEGIST_CONFIG = project.ROOT / "net" / "config" / "strategist.json"


def load_move_plan(name):
    """bridge/config/move-plans/<name>.json; every leg must be complete."""
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


def build(target, run_config, dependencies=None, scenario=None, kv=None, out=None):
    """dependencies defaults to the required mods; every battle loads them.
    scenario overrides the target's scenario file (a name in bridge/scenarios/, or a path).
    kv ({section: {key: value}}, a probe only): the pack carries its own rule tables with these rows
    (bridge/pack/kvtable.py); the run's config and manifest name them (kv, kv_label).
    out: the output folder (default build/<target>/)."""
    if dependencies is None:
        dependencies = project.required_mods()
    if kv:
        assert target in KV_TARGETS, f"--kv is for the probes {KV_TARGETS}"
        run_config = dict(run_config, kv=kv, kv_label=kvtable.label(kv))
    spec = dict(TARGETS[target])
    if scenario:
        spec["scenario"] = scenario
    scenario_xml = (project.SCENARIOS / spec["scenario"]).read_bytes()
    # The build id covers every input, so telemetry rows name the exact code.
    probe_script, modules = bundle.bundle(project.LUA, spec["entry"], run_config)
    digest = sha256(probe_script.encode("utf-8") + scenario_xml)[:16]
    run_config = dict(run_config, build=digest)
    header = f"-- Generated by bridge/build.py ({target}, build {digest}). Edit bridge/lua/, not this file.\n"
    script, _ = bundle.bundle(project.LUA, spec["entry"], run_config, header)
    script_bytes = script.encode("utf-8")
    syntax_checked = check_syntax(script)

    folder = spec["folder"]
    files = {
        f"script\\battle\\mod\\{spec['script']}.lua": script_bytes,
        f"script\\battle\\{folder}\\scenario.lua": SCENARIO_LUA,
        f"script\\battle\\{folder}\\{spec['packed_scenario']}": scenario_xml,
    }
    files.update(kvtable.entries(kv or {}))
    if (run_config or {}).get("live_label"):
        # the labelling's marks over the battle (bridge/lua/apps/battle/label_overlay.lua): the group icons; the enemy
        # cards' pictures (bridge/lua/apps/battle/enemy_cards.lua) by the units' keys
        from game.label import icons
        files.update(icons.pack_files())
        keys = {u.get("key") for side in ("own", "enemy") for u in ((run_config.get("units") or {}).get(side) or [])}
        files.update(icons.unit_files(k for k in keys if k))
    dependency_names = [d["pack"] for d in dependencies]
    blob = pfh5.pack_files(files, dependency_names)
    assert pfh5.read_pack(blob) == (files, tuple(dependency_names))

    out = Path(out) if out is not None else project.BUILD / target
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
    if kv:
        manifest["kv_override"] = kv
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def side_labels(args):
    """{side: the general's name over the army in the game}: who commands the side (--label-own/--label-enemy
    override). The game runs in Russian (Steam language), so its fonts have Cyrillic."""
    own = {"net": "Нейросеть", "human": "Человек"}.get(args.own_ai, args.own_ai)
    return {"own": args.label_own or own, "enemy": args.label_enemy or "ИИ игры (нормальная)"}


def strategist_zone(arena, path=STRATEGIST_CONFIG):
    """The arena with the strategist's deployment areas (sim/arena.py deploy_zone): the zones of its cases
    (net/config/strategist.json cases.zone_m, gap_m), so the zones it sees in the game are those it learnt with."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    front, depth = cases["zone_m"]
    return dict(arena, deploy_zone={"front": float(front), "depth": float(depth), "gap": float(cases["gap_m"])})


def nn_arena_config(args, run_config):
    """nn-arena / human: writes the battle file next to the build (build/<target>/<arena>.xml, --out) and fills run_config
    (units, roles, the generated armies). Returns the battle file's path."""
    from sim import arena as nn_scenario
    # The side that wins on timeout defends: the game's AI when ours attacks (--own-role; default defend).
    own_role = args.own_role or "defend"
    enemy_role = "defend" if own_role == "attack" else "attack"
    defender = "enemy" if enemy_role == "defend" else "own"
    duration_s = max(3600, args.timeout + TIMEOUT_MARGIN_S)
    labels = side_labels(args)
    army = None
    if getattr(args, "arena_json", None) is not None:
        # an arena dict written elsewhere (game: the stage 0 armies against the game's AI)
        arena = json.loads(Path(args.arena_json).read_text(encoding="utf-8"))
        army = {"from_file": str(args.arena_json)}
    elif args.army_from_run is not None:
        # an old run's armies and deployment (game/arena/army_from_run.py): its manifest, not the generator
        from game.arena import army_from_run
        arena, army = army_from_run.load(args.army_from_run, swap=args.army_swap,
                                         replace=dict(x.split("=", 1) for x in args.army_replace or ()))
    elif args.army_seed is not None:
        from sim.armies import generate
        arena = generate.battle(args.army_seed, armies=args.armies)
        if args.army_swap:
            # the other half of a swapped pair: our side takes the generator's enemy army
            arena = dict(arena, name=f"{arena['name']}_swap",
                         sides={"own": arena["sides"]["enemy"], "enemy": arena["sides"]["own"]})
        army = {"seed": args.army_seed, "split": generate.split(args.army_seed), "swap": args.army_swap,
                "armies": args.armies,
                "budget": arena["budget"],
                "side_budget": {s: arena["sides"][s]["budget"] for s in nn_scenario.SIDES},
                "template": {s: arena["sides"][s]["army"] for s in nn_scenario.SIDES}}
    else:
        arena = nn_scenario.load_arena()          # sim/config/arena.json: the same army on both sides
    if args.deploy == "strategist":
        arena = strategist_zone(arena)
    if args.map:
        # a captured map (sim/maps/<name>): its battle file, catchment and deployment zones; the places are moved onto
        # the zones after the battle file is written
        from sim import maps as arena_map
        arena = arena_map.on_map(arena, args.map)
    # The battle file goes next to the build: bridge/scenarios/nn_arena.xml stays as it is in git.
    out = Path(args.out) if args.out else project.BUILD / args.target
    path = out / f"{arena['name']}{'_strategist' if args.deploy == 'strategist' else ''}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    nn_scenario.write_scenario(defender, arena, path, duration_s, labels)
    if arena.get("origin"):
        from sim import maps as arena_map
        arena_map.shift_xml(path, arena["origin"])
    if army is not None:
        run_config["army"] = dict(army, cost={s: arena["sides"][s]["cost"] for s in nn_scenario.SIDES},
                                  men={s: sum(u["men"] for u in arena["sides"][s]["units"])
                                       for s in nn_scenario.SIDES})
    run_config.update(nn_scenario.run_config(arena), own_ai=args.own_ai, enemy_role=enemy_role, own_role=own_role)
    if args.deploy == "strategist":
        # Deployment: the companion's strategist places our side (bridge/lua/apps/bridge/deploy_adapter.lua); the zones
        # its input reads are the battle file's (sim.arena.zones), the enemy's side stays where the battle file puts it
        run_config.update(deploy="strategist", deploy_zones={s: dict(z) for s, z in nn_scenario.zones(arena).items()},
                          deploy_wait_s=DEPLOY_WAIT_S, deploy_settle_ms=DEPLOY_SETTLE_MS)
    if args.live_label:
        # the live labelling (bridge/lua/apps/battle/live_label.lua, game/label/live.py): the person's groups and the
        # enemy cards' groups out with the state every second; live_test: the scripted check's commands
        run_config["live_label"] = True
        # the deployment: the person deploys as in the game (until his 'Start battle'); the zones the strategist's input
        # reads (game/label/live_deploy.py) are the battle file's
        run_config.setdefault("deploy_zones", {s: dict(z) for s, z in nn_scenario.zones(arena).items()})
        if args.live_test:
            run_config["live_test"] = True
    if args.own_ai == "net":
        # skirmish: the game's skirmish mode off for every unit the bridge commands by default - the simulator's network
        # units have it off (sim/state.py "sk"; contract/parity.md); 'game': as the game sets it
        run_config.update(decide_ms=args.decide_ms, poll_ms=NET_POLL_MS, skirmish=args.skirmish or "off")
    if args.own_ai == "human":
        run_config["soldiers_every"] = HUMAN_SOLDIERS_EVERY if args.soldiers_every is None else args.soldiers_every
        if args.free_speed:
            # the human sets the pace: may change the speed and pause (no speed guard)
            run_config.update(free_speed=True)
    if arena.get("origin"):
        # the captured map's zones sit off the map's centre: the places and the zones as the battle file has them
        from sim import maps as arena_map
        arena_map.shift_config(run_config, arena["origin"])
        run_config["map_name"] = arena["map_name"]
    return path


def extra_mod(workshop_id):
    """{pack, workshop_id, sha256} of a Workshop mod's one pack (the launcher's dependency entry)."""
    folder = Path(project.load()["workshop_dir"]) / str(workshop_id)
    packs = sorted(folder.glob("*.pack"))
    if len(packs) != 1:
        raise SystemExit(f"--extra-mod {workshop_id}: {len(packs)} packs in {folder}")
    return {"pack": packs[0].name, "workshop_id": str(workshop_id), "sha256": sha256(packs[0].read_bytes())}


def probe_module(target):
    import importlib
    return importlib.import_module(f"game.probe.{PROBES[target][0]}")


def parser_of():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--speed", type=int, choices=(1, 3, 10, 20), help="battle speed (default 20; human: 1)")
    parser.add_argument("--timeout", type=int, help="model seconds, 30..3600 (default 600; human: 3600)")
    parser.add_argument("--tick-ms", type=int, help="ms between samples (default 1000; the probes 500)")
    parser.add_argument("--step", type=int, choices=(1, 2, 3, 5), default=5, help="map-capture: cell size, m")
    parser.add_argument("--deadline", type=int, help="real seconds per battle before the script ends it "
                        "(default: from the scripted length and speed)")
    parser.add_argument("--stall-minutes", type=float, default=10,
                        help="end the battle when nobody takes damage for this much GAME time")
    parser.add_argument("--scenario", help="scenario file in bridge/scenarios/ (or a path) instead of the target's default")
    parser.add_argument("--window", type=float, nargs=4, metavar=("MIN_X", "MAX_X", "MIN_Z", "MAX_Z"),
                        help="map-capture: capture only this area")
    parser.add_argument("--plan", default="hamlet", help="move-probe: plan in bridge/config/move-plans/")
    parser.add_argument("--own-ai", choices=("net", "ai"),
                        help="nn-arena: the network in the companion (bridge/companion: the chain, the strategist's "
                             "deployment) commands our side ('net', the default); 'ai' - the game's own tactical AI "
                             "(CA's script_ai_planner, 'attack the enemy force': no companion) - the game's AI against "
                             "the game's AI; the game's AI the other side always")
    parser.add_argument("--own-role", choices=("attack", "defend"),
                        help="nn-arena, human: our side attacks (the game's AI defends and wins on timeout) or defends "
                             "(default: the game's AI attacks)")
    parser.add_argument("--label-own", default="",
                        help="the general's name over our army (default: who commands it - 'Нейросеть', 'Человек')")
    parser.add_argument("--label-enemy", default="",
                        help="the general's name over the enemy army (default: 'ИИ игры (нормальная)')")
    parser.add_argument("--free-speed", action="store_true",
                        help="human: the battle starts at --speed and the human may change the speed and pause "
                             "(no speed guard)")
    parser.add_argument("--skirmish", choices=("game", "off"), default=None,
                        help="nn-arena: the game's skirmish mode (a unit steps back from approaching enemies by itself, "
                             "over its order) off for every unit the bridge commands (default: as the simulator's "
                             "network units, sim/state.py 'sk') or as the game sets it ('game')")
    parser.add_argument("--soldiers-every", type=int,
                        help=f"human: every soldier's position every this many ticks (0: never; default "
                             f"{HUMAN_SOLDIERS_EVERY})")
    parser.add_argument("--army-seed", type=int,
                        help="nn-arena, human: a generated battle (sim/armies, generate.battle(seed))")
    parser.add_argument("--armies", choices=("v1", "v2", "v3"), default="v1",
                        help="--army-seed: the generator's scheme (sim/armies/generate.py); v3 the training's")
    parser.add_argument("--army-swap", action="store_true",
                        help="--army-seed / --army-from-run: the armies swapped (our side gets the enemy army)")
    parser.add_argument("--arena-json", type=Path, metavar="FILE",
                        help="nn-arena, human: the arena dict (sim/arena.py shape: sides own / enemy with units' slot, "
                             "key, men, forward, lateral, width) from a JSON file instead of the generator")
    parser.add_argument("--army-from-run", type=Path, metavar="RUN_DIR",
                        help="nn-arena, human: the armies, deployment, gap of an old run (its manifest.json) instead of "
                             "the generator (game/arena/army_from_run.py)")
    parser.add_argument("--army-replace", action="append", default=[], metavar="OLD=NEW",
                        help="with --army-from-run: every unit of key OLD becomes one of key NEW on the same place")
    parser.add_argument("--map", default=None, metavar="NAME",
                        help="a captured map (sim/maps/NAME: meta.json): the battle on it, its deployment zones "
                             "(sim/maps/__init__.py)")
    parser.add_argument("--extra-mod", action="append", default=[], metavar="WORKSHOP_ID",
                        help="human: one more Workshop mod for this battle (pinned by its SHA-256 now; the launcher "
                             "copies it into data/ and removes it after) - the person's own mods, never the network's")
    parser.add_argument("--decide-ms", type=int, default=1000,
                        help="nn-arena --own-ai net: model ms between two decisions (250..5000)")
    parser.add_argument("--deploy", choices=("script", "strategist"), default="script",
                        help="nn-arena: who places our side in the deployment - 'script' (default: the battle file's "
                             "places, the phase ends after 1 s) or 'strategist' (the companion's strategist, "
                             "companion --strategist: its zones - the strategist's cases' - in the battle file)")
    parser.add_argument("--live-label", action="store_true",
                        help="human: the live labelling - the enemy's cards, the person's groups out with the state "
                             "every second (bridge/lua/apps/battle/live_label.lua, game/label/live.py)")
    parser.add_argument("--live-test", action="store_true",
                        help="human --live-label: the scripted check's commands (tww3_bai_live_cmd.txt in the game folder)")
    parser.add_argument("--probe-plan", default="charge", help="charge-probe: the plan (game/probe/charge_probe.py PLANS)")
    parser.add_argument("--probe-battle", type=int, default=1, help="charge-probe: the plan's battle, 1-based")
    parser.add_argument("--mprobe-plan", default="dist", help="missile-probe: the plan (game/probe/missile_probe.py)")
    parser.add_argument("--mprobe-battle", type=int, default=1, help="missile-probe: the plan's battle, 1-based")
    parser.add_argument("--morale-plan", default="flank", help="morale-probe: the plan (game/probe/morale_probe.py)")
    parser.add_argument("--morale-battle", type=int, default=1, help="morale-probe: the plan's battle, 1-based")
    parser.add_argument("--mem", action="store_true",
                        help="a probe only (" + ", ".join(MEM_TARGETS) + "): load memreader Plus "
                             "(bridge/config/memreader.json) next to our pack and read the projectiles in flight "
                             "(missile-probe), the men of the lanes' units (charge-probe) or the units' morale "
                             "(morale-probe) from the game's memory; the code sites: memreader.json sites_file "
                             "(reverse/mem/sites_exe1103.json; BAI_MEM_SITES to try another)")
    parser.add_argument("--kv", action="append", metavar="[SECTION.]KEY=VALUE|xN",
                        help="a probe only: a row of the game's rule tables (_kv_morale/_kv_rules/_kv_fatigue: section "
                             "morale/battle/fatigue) set to VALUE or N times the game's value, by a table of the probe's "
                             "own pack (bridge/pack/kvtable.py); repeatable. Never for a battle that records data")
    parser.add_argument("--out", type=Path, default=None,
                        help="the output folder (default build/<target>/, where the launcher takes it): another one for a "
                             "dry build (tests)")
    parser.add_argument("--nodes", action="store_true",
                        help="map-capture: sample the multiples of the step (the engine's height nodes), not cell centres")
    parser.add_argument("--features", action="store_true",
                        help="map-capture: also read objects and reachability after deployment")
    return parser


def check_args(parser, args):
    arena = args.target in ARENA_TARGETS
    if args.target == "human":
        if args.own_ai is not None:
            parser.error("the human target is our side under a human: no --own-ai")
        args.own_ai = "human"
    elif args.target == "nn-arena":
        args.own_ai = args.own_ai or "net"
    elif args.own_ai is not None:
        parser.error("--own-ai is for nn-arena")
    if args.speed is None:
        args.speed = 1 if args.target == "human" else 20
    if args.timeout is None:
        args.timeout = 3600 if args.target == "human" else 600
    if args.own_role and not arena:
        parser.error("--own-role is for nn-arena and human")
    if args.skirmish is not None and args.target != "nn-arena":
        parser.error("--skirmish is for nn-arena (the bridge's units)")
    for flag, ok in (("live_label", args.target == "human"), ("live_test", args.live_label),
                     ("free_speed", args.target == "human"), ("extra_mod", args.target == "human"),
                     ("soldiers_every", args.target == "human")):
        if getattr(args, flag) not in (None, False, []) and not ok:
            parser.error(f"--{flag.replace('_', '-')} is for the human target" + (" with --live-label" if
                                                                                    flag == "live_test" else ""))
    if args.soldiers_every is not None and args.soldiers_every < 0:
        parser.error("--soldiers-every must be 0 or more")
    if args.army_from_run is not None:
        if not arena:
            parser.error("--army-from-run is for nn-arena and human")
        if args.army_seed is not None:
            parser.error("--army-from-run and --army-seed are two sources of armies: give one")
        if not (args.army_from_run / "manifest.json").exists():
            parser.error(f"--army-from-run: no manifest.json in {args.army_from_run}")
    generated = args.army_seed is not None or args.army_from_run is not None
    if args.army_seed is not None and not arena:
        parser.error("--army-seed is for nn-arena and human")
    if args.army_swap and not generated:
        parser.error("--army-swap is for --army-seed and --army-from-run")
    if args.map and not arena:
        parser.error("--map is for nn-arena and human")
    if args.deploy != "script" and args.target != "nn-arena":
        parser.error("--deploy is for nn-arena")
    if not 30 <= args.timeout <= 3600:
        parser.error("--timeout must be between 30 and 3600 seconds")
    if not 250 <= args.decide_ms <= 5000:
        parser.error("--decide-ms must be between 250 and 5000")
    if args.mem and args.target not in MEM_TARGETS:
        parser.error(f"--mem is for the probes: {', '.join(MEM_TARGETS)}")
    if args.kv and args.target not in KV_TARGETS:
        parser.error(f"--kv is for the probes: {', '.join(KV_TARGETS)}")
    if args.target in PROBES:
        plans = tuple(probe_module(args.target).PLANS)
        plan = getattr(args, PROBES[args.target][1])
        if plan not in plans:
            parser.error(f"{args.target}: no plan {plan!r} ({', '.join(plans)})")


def main(argv=None):
    parser = parser_of()
    args = parser.parse_args(argv)
    check_args(parser, args)
    kv = None
    if args.kv:
        try:
            kv = kvtable.parse(args.kv)
        except ValueError as e:
            parser.error(str(e))

    if args.target == "map-capture":
        run_config = {"step": args.step, "features": args.features, "nodes": args.nodes}
        if args.window:
            run_config["window"] = dict(zip(("min_x", "max_x", "min_z", "max_z"), args.window))
    else:
        tick_ms = args.tick_ms or (PROBE_TICK_MS if args.target in PROBES else 1000)
        run_config = {"speed": args.speed, "timeout_ms": args.timeout * 1000, "tick_ms": tick_ms,
                      "scenario": TARGETS[args.target]["scenario"].removesuffix(".xml")}
        model_s = args.timeout
        stall_ms = int(args.stall_minutes * 60000)
        if args.target == "move-probe":
            plan = load_move_plan(args.plan)
            model_s = move_plan_model_s(plan)
            run_config.update(plan=plan, settle_ms=MOVE_SETTLE_MS)
            # Nobody takes damage in this probe: the stall rule must not cut the plan.
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
        if args.target in ARENA_TARGETS:
            path = nn_arena_config(args, run_config)
            if not args.scenario:
                args.scenario = str(path)
        if args.target in PROBES:
            probe = probe_module(args.target)
            plan, battle = getattr(args, PROBES[args.target][1]), getattr(args, PROBES[args.target][2])
            path = probe.write_scenario(plan, battle)
            cfg, model_s = probe.run_config(plan, battle)[:2]
            run_config.update(cfg)
            stall_ms = max(stall_ms, int((model_s + 120) * 1000))
            if not args.scenario:
                args.scenario = str(path)
        run_config["deadline_s"] = args.deadline or deadline_seconds(model_s, args.speed)
        run_config["stall_ms"] = stall_ms
    dependencies = None
    if args.extra_mod:
        dependencies = project.required_mods() + [extra_mod(i) for i in args.extra_mod]
    if args.mem:
        from bridge import mem_sites
        exe = Path(project.load()["game_dir"]) / "Warhammer3.exe"
        run_config["mem"] = mem_sites.lua_config(exe=exe if exe.is_file() else None)
        dependencies = project.required_mods() + [mem_sites.dependency()]
    manifest = build(args.target, run_config, scenario=args.scenario, **({"kv": kv} if kv else {}),
                     **({"dependencies": dependencies} if dependencies else {}), out=args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
