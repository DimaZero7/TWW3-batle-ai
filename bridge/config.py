"""The bridge's paths and this machine's settings (the project's common settings: sim.passports and game read them too).

bridge/config/default.json is committed; bridge/config/local.json (not in git) overrides it per machine:
    {"game_dir": "D:\\SteamLibrary\\steamapps\\common\\Total War WARHAMMER III",
     "workshop_dir": "D:\\SteamLibrary\\steamapps\\workshop\\content\\1142710"}
"""
import json
from pathlib import Path

HOME = Path(__file__).resolve().parent          # bridge/
ROOT = HOME.parent                              # the project's root
LUA = HOME / "lua"                              # the mod's Lua: apps/*, entries/* (the bundler's root)
SCENARIOS = HOME / "scenarios"                  # the battle files
CONFIG_DIR = HOME / "config"
BUILD = ROOT / "build"                          # build/<target>/ the pack, runs/ (not in git)
STATE = ROOT / "state"                          # state/HUMAN_PLAYING, state/probe_game.lock (not in git)
MOD_DEPENDENCIES = CONFIG_DIR / "mod-dependencies.json"


def required_mods():
    """Mods every battle must load, pinned by SHA-256 (bridge/config/mod-dependencies.json).

    Returns [{pack, workshop_id, sha256}] in load order.
    """
    pinned = json.loads(MOD_DEPENDENCIES.read_text(encoding="utf-8"))
    return [{"pack": m["pack_name"], "workshop_id": m["workshop_id"], "sha256": m["sha256"]}
            for m in pinned["mods"]]


def mod_profile():
    return json.loads(MOD_DEPENDENCIES.read_text(encoding="utf-8"))["profile"]


def resolve(path):
    """A path in the project (a checkpoint of the configs, a run): absolute as given, a relative one from the root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load():
    """{game_dir, workshop_dir, ...}: default.json overridden by local.json."""
    settings = json.loads((CONFIG_DIR / "default.json").read_text(encoding="utf-8"))
    local = CONFIG_DIR / "local.json"
    if local.exists():
        settings.update(json.loads(local.read_text(encoding="utf-8")))
    return settings
