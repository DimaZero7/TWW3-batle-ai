"""Build a battle pack from bridge/lua/ for one target. Never touches the game.

Usage:
    python -m bridge.build nn-arena --army-seed 1000702112 --armies v3 --map ksl_plains_01 --deploy strategist \\
        --own-ai net --speed 10 --timeout 1200          # the chain's battle (bridge/launcher/watch.ps1 -Chain)
    python -m bridge.build nn-arena --army-from-run data/human/runs/20261107-171600 --map ksl_plains_01 \\
        --deploy strategist --own-ai net                 # an old run's armies and places (game/arena/army_from_run.py)
    python -m bridge.build human --army-seed 1000900014 --army-swap --live-label --free-speed   # the person plays
    python -m bridge.build charge-probe --probe-plan hit --probe-battle 1 --mem   # the melee probe (game/probe)
    python -m bridge.build missile-probe --mprobe-plan dist --mprobe-battle 1     # the missile probe
    python -m bridge.build charge-probe --probe-plan duel --no-draw              # no frames drawn, x40 (V2)
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
# nn-arena --own-ai net - lock-step (bridge/lua/entries/nn_arena.lua M.lockstep), on in every build where the network
# drives our side (--no-lockstep: off); a build with no network side (the probes, 'ai', human) has none. A decision tick
# waits for its answer at most LOCKSTEP_TIMEOUT_MS of real time (LOCKSTEP_FIRST_TIMEOUT_MS for the first: the companion
# may still load its networks), reading the orders file every LOCKSTEP_SPIN_MS; strict: the first failed hold voids the
# battle (an unattended series takes only battles with result ls_ok). The deadline: as if the battle ran at
# LOCKSTEP_PACE model s a wall s (deadline_seconds: doubled + 60 s) - lock-step realised 5.0-8.9 on an 8-unit scene
# (tmp/farm/l, 22.11), a battle far slower than 1 is not a lock-step battle worth keeping.
LOCKSTEP_TIMEOUT_MS, LOCKSTEP_FIRST_TIMEOUT_MS, LOCKSTEP_SPIN_MS, LOCKSTEP_STRIKES = 5000, 30000, 2, 3
LOCKSTEP_PACE = 1
# nn-arena: the battle file's own time limit is past the script's (the script ends the battle first).
TIMEOUT_MARGIN_S = 60
# nn-arena --deploy strategist: real seconds the deployment waits for the companion's plan (then as before: no plan),
# and real ms after the teleports before the places are read (the 26.09.2026 check read them 1000 ms after).
DEPLOY_WAIT_S = 120
DEPLOY_SETTLE_MS = 1000
# nn-arena --park-lords: the corners' margin inside the map's square, the least gap wanted from the other side's units
# (the user's 21.11: far from the fight), the drift that makes the script take the lord again (m).
PARK_MARGIN_M, PARK_MIN_M, PARK_DRIFT_M = 60.0, 400.0, 10.0
# human: every soldier's position every this many ticks (a reading of all men takes ~50 ms).
HUMAN_SOLDIERS_EVERY = 5
ARENA_TARGETS = ("nn-arena", "human")
PROBE_TICK_MS = 500
# --kv (the game's rule rows changed in the pack, bridge/pack/kvtable.py): probes only, never a battle for data.
KV_TARGETS = ("charge-probe", "missile-probe", "morale-probe")
# --mem (memreader Plus: the game's memory read in a probe; bridge/docs/apps/memreader.md): probes whose entry reads it.
# Never the network's battles or exams.
MEM_TARGETS = ("missile-probe", "charge-probe", "morale-probe")
# --mem-diag (bridge/config/memreader.json 'mem_diag'): ONE diagnostic exception - nn-arena --own-ai net built with the
# entry entries.nn_arena_memdiag (the same battle, the enemy lord and our men read from memory beside it, read only);
# mem_diag = true in the manifest (and config.mem_diag) and on every row the diagnosis writes: never a series, an exam
# or the owner's battle
MEM_DIAG_ENTRY = "entries.nn_arena_memdiag"
# --no-draw (bridge/lua/apps/battle/no_draw.lua; the policy bridge/config/memreader.json, decided 23.11.2026): the probes
# and lanes, and nn-arena under the game's planner ('ai') or the network under lock-step - not an exam; never 'human'.
# ON NO_DRAW_DELAY_MS of model time after the start; the Lua's own safety net turns it off NO_DRAW_SAFETY_S of real time
# after the battle's deadline. Its deadline is reckoned at NO_DRAW_PACE (a refused patch: the battle runs drawn, which is
# frame-bound at x20 and x40 alike - tmp/farm/l/REPORT.md); x40 only with it.
NO_DRAW_TARGETS = ("charge-probe", "missile-probe", "morale-probe", "nn-arena")
NO_DRAW_DELAY_MS, NO_DRAW_SAFETY_S, NO_DRAW_PACE, NO_DRAW_SPEED = 3000, 60, 20, 40
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


def build(target, run_config, dependencies=None, scenario=None, kv=None, out=None, entry=None, mem_diag=False):
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
    if entry:                                # (--mem-diag only)
        spec["entry"] = entry
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
    if mem_diag:                             # (--mem-diag: the mark on the build itself, beside config.mem_diag)
        manifest["mem_diag"] = True
    if kv:
        manifest["kv_override"] = kv
    if (run_config or {}).get("no_draw"):
        manifest["no_draw"] = True
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


def east_zone_maps(args):
    """--map-zones east: a maps folder (build/<target>/maps) where the map's meta.json has the game's east.xml zones
    ("game_deployment_zones", the land battle's of a map without its own: sim/maps/__init__.py GAME_DIRECTIONS) taken
    from a captured map that has them (they are the same file for every map); a map that has them itself: sim/maps as
    is. For a map captured without them (kho_wastes_01: only its capture's 200 x 120 m zones at z = -190), so its battle
    file declares the zones the other maps' battles have - sim/maps (frozen) unchanged."""
    from sim import maps as arena_map
    m = arena_map.meta(args.map)
    if m.get("game_deployment_zones"):
        return arena_map.MAPS
    src = next(n for n in sorted(p.name for p in arena_map.MAPS.iterdir() if (p / "meta.json").exists())
               if "east.xml" in str(arena_map.meta(n).get("game_deployment_source") or ""))
    folder = (Path(args.out) if args.out else project.BUILD / args.target) / "maps" / args.map
    folder.mkdir(parents=True, exist_ok=True)
    m = dict(m, game_deployment_zones=arena_map.meta(src)["game_deployment_zones"],
             game_deployment_source=f"terrain.pack terrain\deployment\east.xml, as {src}'s (bridge.build --map-zones east)")
    (folder / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return folder.parent


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
        arena = arena_map.on_map(arena, args.map, maps=east_zone_maps(args) if args.map_zones == "east" else arena_map.MAPS)
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
        # lock-step: the dict when on, false with --no-lockstep (the manifest says either way)
        run_config["lockstep"] = False
        if args.lockstep:
            run_config["lockstep"] = {"timeout_ms": LOCKSTEP_TIMEOUT_MS, "first_timeout_ms": LOCKSTEP_FIRST_TIMEOUT_MS,
                                      "spin_ms": LOCKSTEP_SPIN_MS, "strikes": LOCKSTEP_STRIKES, "strict": True}
    if getattr(args, "park_lords", False):
        # both lords parked in two far corners for the whole battle (bridge/lua/entries/nn_arena.lua park_lords): the
        # square - a captured map's frame (sim/maps/<name>/meta.json), else the flat map's edge (sim/config/sim.json
        # map_half_m: routing units leave it at 1020 m)
        if args.map:
            from sim import maps as arena_map
            fr = arena_map.meta(args.map)["frame"]
            half = min(fr["max_x"] - fr["min_x"], fr["max_z"] - fr["min_z"]) / 2
            cx, cz = (fr["max_x"] + fr["min_x"]) / 2, (fr["max_z"] + fr["min_z"]) / 2
        else:
            sim_cfg = json.loads((Path(__file__).resolve().parents[1] / "sim/config/sim.json").read_text(encoding="utf-8"))
            half, cx, cz = float(sim_cfg["map_half_m"]["value"]), 0.0, 0.0
        run_config["park_lords"] = {"half": float(half), "cx": float(cx), "cz": float(cz), "margin": PARK_MARGIN_M,
                                    "min_m": PARK_MIN_M, "drift_m": PARK_DRIFT_M}
        if args.park_lords == "enemy":
            # only the enemy's lord (side 2) parked; ours stays in play (the lord line's lane E). 'both' writes no
            # 'sides': the config and the pack's config as before
            run_config["park_lords"]["sides"] = [2]
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
    parser.add_argument("--speed", type=int, choices=(1, 3, 10, 20, 40),
                        help="battle speed (default 20; human: 1; --no-draw: 40, and 40 only with it)")
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
    parser.add_argument("--lockstep", action="store_true",
                        help="nn-arena --own-ai net: lock-step, the default (kept so that older calls work)")
    parser.add_argument("--no-lockstep", action="store_true",
                        help="nn-arena --own-ai net: lock-step off (by default each decision tick waits - the model "
                             "stands - until the companion's answer for that move is applied, at most "
                             + str(LOCKSTEP_TIMEOUT_MS) + " ms of real time (the first " + str(LOCKSTEP_FIRST_TIMEOUT_MS)
                             + "); a failed wait voids the battle (no result; bridge/lua/entries/nn_arena.lua "
                             "M.lockstep); the default deadline as at " + str(LOCKSTEP_PACE) + " model s a wall s)")
    parser.add_argument("--park-lords", action="store_const", const="both", default=None,
                        help="nn-arena --own-ai net: both lords taken out of the fight - teleported to two far corners at "
                             "the start and held there by the script (no orders from the game's AI or the companion; "
                             "bridge/lua/entries/nn_arena.lua): the infantry alone. '--park-lords enemy' (the word right "
                             "after it; main() reads it as --park-side enemy): only the enemy's lord parked, ours in play "
                             "(the state's lords 'park_enemy'); '--park-lords both' = the bare flag")
    parser.add_argument("--park-side", choices=("both", "enemy"), default=None, help=argparse.SUPPRESS)
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
    parser.add_argument("--map-zones", choices=("map", "east"), default="map",
                        help="with --map: 'map' - the map's own zones as sim/maps gives them (default); 'east' - the "
                             "game's east.xml zones (416 x 176 m) also on a map captured without them (kho_wastes_01)")
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
    parser.add_argument("--mem-diag", action="store_true",
                        help="nn-arena --own-ai net only, a DIAGNOSTIC build (bridge/config/memreader.json mem_diag): "
                             "the same battle with the enemy lord and our men read from memory every 0.1 s, read only "
                             "(entries/nn_arena_memdiag.lua); the manifest and the diagnosis' rows say mem_diag - never a series, "
                             "an exam or the owner's battle")
    parser.add_argument("--no-draw", action="store_true",
                        help="the game draws no frame while the battle runs (V2; " + ", ".join(NO_DRAW_TARGETS) + "; "
                             "nn-arena: --own-ai ai, or net under lock-step; never human nor an exam): memreader Plus "
                             "next to our pack turns the view / present call into a nop after the start and back at "
                             "the end (bridge/lua/apps/battle/no_draw.lua; site no_draw of memreader.json sites_file); "
                             "refused in the game - drawn, event v2_refused; speed 40 by default")
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
    net = args.target == "nn-arena" and args.own_ai == "net"
    if (args.lockstep or args.no_lockstep) and not net:
        parser.error("--lockstep / --no-lockstep are for nn-arena --own-ai net (only the companion's answers are "
                     "waited for)")
    if args.lockstep and args.no_lockstep:
        parser.error("--lockstep (the default) and --no-lockstep: give one")
    # lock-step on in every build where the network drives our side, unless --no-lockstep
    args.lockstep = net and not args.no_lockstep
    if args.no_draw:
        if args.target == "human":
            parser.error("--no-draw is never for the human target (the owner's own battle: bridge/config/memreader.json)")
        if args.target not in NO_DRAW_TARGETS:
            parser.error(f"--no-draw is for {', '.join(NO_DRAW_TARGETS)}")
        if net and not args.lockstep:
            parser.error("--no-draw with nn-arena --own-ai net needs lock-step (the model must wait for the companion): "
                         "no --no-lockstep")
    if args.speed == NO_DRAW_SPEED and not args.no_draw:
        parser.error(f"--speed {NO_DRAW_SPEED} is only with --no-draw (drawn, the game is frame-bound at x20)")
    if args.speed is None:
        args.speed = 1 if args.target == "human" else (NO_DRAW_SPEED if args.no_draw else 20)
    if args.timeout is None:
        args.timeout = 3600 if args.target == "human" else 600
    if args.own_role and not arena:
        parser.error("--own-role is for nn-arena and human")
    if args.skirmish is not None and args.target != "nn-arena":
        parser.error("--skirmish is for nn-arena (the bridge's units)")
    if args.park_lords and not (args.target == "nn-arena" and args.own_ai == "net"):
        parser.error("--park-lords is for nn-arena --own-ai net (the planner of 'ai' would order the lord again)")
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
    if args.mem_diag and not net:
        parser.error("--mem-diag is for nn-arena --own-ai net only (bridge/config/memreader.json mem_diag)")
    if args.kv and args.target not in KV_TARGETS:
        parser.error(f"--kv is for the probes: {', '.join(KV_TARGETS)}")
    if args.target in PROBES:
        plans = tuple(probe_module(args.target).PLANS)
        plan = getattr(args, PROBES[args.target][1])
        if plan not in plans:
            parser.error(f"{args.target}: no plan {plan!r} ({', '.join(plans)})")


def park_words(argv):
    """'--park-lords enemy|both' -> '--park-lords --park-side <word>' (any other next word - the target nn-arena, an
    option - stays as it was: the bare flag is 'both', as before)."""
    out, i = [], 0
    while i < len(argv):
        out.append(argv[i])
        if argv[i] == "--park-lords" and i + 1 < len(argv) and argv[i + 1] in ("both", "enemy"):
            out += ["--park-side", argv[i + 1]]
            i += 1
        i += 1
    return out


def main(argv=None):
    parser = parser_of()
    args = parser.parse_args(park_words(list(sys.argv[1:] if argv is None else argv)))
    if args.park_side is not None:
        if not args.park_lords:
            parser.error("--park-side comes only with --park-lords")
        args.park_lords = args.park_side
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
        pace = LOCKSTEP_PACE if args.lockstep else (
            min(args.speed, NO_DRAW_PACE) if args.no_draw else args.speed)
        run_config["deadline_s"] = args.deadline or deadline_seconds(model_s, pace)
        run_config["stall_ms"] = stall_ms
    dependencies = None
    if args.extra_mod:
        dependencies = project.required_mods() + [extra_mod(i) for i in args.extra_mod]
    if args.mem:
        from bridge import mem_sites
        exe = Path(project.load()["game_dir"]) / "Warhammer3.exe"
        run_config["mem"] = mem_sites.lua_config(exe=exe if exe.is_file() else None)
        dependencies = project.required_mods() + [mem_sites.dependency()]
    if args.mem_diag:
        from bridge import mem_sites
        exe = Path(project.load()["game_dir"]) / "Warhammer3.exe"
        run_config["mem"] = mem_sites.lua_config(exe=exe if exe.is_file() else None)
        run_config["mem_diag"] = True
        dependencies = (dependencies or project.required_mods()) + [mem_sites.dependency()]
    if args.no_draw:
        from bridge import mem_sites
        exe = Path(project.load()["game_dir"]) / "Warhammer3.exe"
        nd = mem_sites.no_draw_config(exe=exe if exe.is_file() else None)
        if nd["site"] is None:
            print(f"--no-draw: {nd['why']} - the battle will go on drawn (v2_refused)", file=sys.stderr)
        # (the bundle takes no None: an absent key is the Lua's nil)
        run_config["no_draw"] = dict({k: v for k, v in nd.items() if v is not None}, delay_ms=NO_DRAW_DELAY_MS,
                                     safety_ms=(run_config["deadline_s"] + NO_DRAW_SAFETY_S) * 1000)
        if not (args.mem or args.mem_diag):
            dependencies = (dependencies or project.required_mods()) + [mem_sites.dependency()]
    manifest = build(args.target, run_config, scenario=args.scenario, **({"kv": kv} if kv else {}),
                     **({"dependencies": dependencies} if dependencies else {}), out=args.out,
                     **({"entry": MEM_DIAG_ENTRY, "mem_diag": True} if args.mem_diag else {}))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
