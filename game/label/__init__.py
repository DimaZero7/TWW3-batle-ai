"""Разметка: a graphical simulator where a person plays one level of the network (the operative, the tactician; the
strategist) and every decision is written as a sample for imitation learning
(docs/ru/training/label.md; the numbers: game/config/label.json).

    matchup.py   the matchup: both armies from the pool, the map, the places, the command from above, the enemy's
                 script; checked, saved and loaded as json (no torch: the host's tests)
    pool.py      the units the pool offers (sim/config/pools.json + the passports), the maps (sim/maps)
    session.py   one battle in the simulator, second by second, paused on the human level's decisions (torch)
    human.py     the person's edit -> the level's choices in the walk's format, checked by the walk's own bans and by
                 tactic/parcel.check (the same checks the network's decision goes through)
    strategist.py   the strategist's level: the deployment once before the battle through net/strategist (imported)
    mapimg.py    the captured map's picture for the page (heights, ground, 'cannot stand', trunks)
    record.py    the folder data/label/<date>_<name>/: the samples (the raw numbers the level's input rebuilds from), the
                 decisions as text, the replay, the result; load() and rebuild() for training
    view.py      what the page draws: the units, the groups, the zones, the targets, the frames (hidden enemies left out)
    server.py    the HTTP server: the page (web/) and the JSON API; serve.sh runs it in the container
    browser_check.py   the scripted check of the page through Chrome's DevTools protocol (host python, stdlib)
    game.py      Разметка в игре: the person as the strategist and the operative of a live battle in the game (the
                 companion's chain below, the bridge pauses the battle on the operative's seconds); game.ps1 runs the
                 server with the game folder and launches the game on the page's request; game_check.py - its scripted
                 check through the page
    icons.py     the in-game marks' pictures (the group's number over a unit; bridge/lua/apps/battle/label_overlay.lua)
"""
import json
from functools import lru_cache

import game as project

CONFIG = project.CONFIG / "label.json"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    """game/config/label.json without its comments."""
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
