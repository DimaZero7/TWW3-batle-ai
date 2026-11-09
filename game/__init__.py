"""Battles in the real game, mechanics probes, live labelling (game/README.md).

    battle/   the chain's battle in the game (chain.sh) and its analysis against the person's base
    probe/    the mechanics probes: lanes in the game, the simulator's twin on the same lanes (charge, missile, morale,
              move; memreader Plus rows: mem_probe)
    label/    the live labelling: the person plays, the recorder writes the levels' samples; digest*, bridge_infer
    maps/     a new map captured from the game once (map_grid.py) -> sim/maps/<name>/
    arena/    a battle of the game from a kept run (army_from_run.py; bridge.build --army-from-run)
    vision/   a battle for the orchestrator's eyes: a strip of frames (PNG) + events and measures (text), ~3 k tokens

The game is started only through the bridge's launcher (bridge/launcher/launch.ps1: Normal difficulty, the user's
settings restored); the marks of state/ (HUMAN_PLAYING, probe_game.lock) are checked by ops/guard.*.
"""
from pathlib import Path

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent                     # the project's root
CONFIG = HOME / "config"               # label.json
BUILD = ROOT / "build"                 # packs and runs the bridge writes (build/<target>/runs/<time>/; not in git)
DATA = ROOT / "data"                   # kept runs and records (not in git): nn-arena/, human/, label/, label_icons/
STATE = ROOT / "state"                 # HUMAN_PLAYING, probe_game.lock (not in git; ops/guard.py)
CONFIG_DIR = CONFIG


def load():
    """This machine's settings (game_dir, workshop_dir, ...): the bridge's (bridge/config)."""
    from bridge import config
    return config.load()


def required_mods():
    from bridge import config
    return config.required_mods()
