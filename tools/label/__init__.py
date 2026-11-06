"""Разметка: a graphical simulator where a person plays one level of the network (the operative, the tactician; the
strategist) and every decision is written as a sample for imitation learning
(docs/ru/training/label.md; the numbers: config/label.json).

    matchup.py   the matchup: both armies from the pool, the map, the places, the command from above, the enemy's
                 script; checked, saved and loaded as json (no torch: the host's tests)
    pool.py      the units the pool offers (config/nn/pools.json + the passports), the maps (build/maps)
    session.py   one battle in the simulator, second by second, paused on the human level's decisions (torch)
    human.py     the person's edit -> the level's choices in the walk's format, checked by the walk's own bans and by
                 tactic/parcel.check (the same checks the network's decision goes through)
    strategist.py   the strategist's level: the deployment once before the battle through tools/nn/strategist (imported)
    mapimg.py    the captured map's picture for the page (heights, ground, 'cannot stand', trunks)
    record.py    the folder build/label/<date>_<name>/: the samples (the raw numbers the level's input rebuilds from), the
                 decisions as text, the replay, the result; load() and rebuild() for training
    view.py      what the page draws: the units, the groups, the zones, the targets, the frames (hidden enemies left out)
    server.py    the HTTP server: the page (web/) and the JSON API; serve.sh runs it in the container
    browser_check.py   the scripted check of the page through Chrome's DevTools protocol (host python, stdlib)
"""
import json
from functools import lru_cache

from tools import config as project

CONFIG = project.ROOT / "config" / "label.json"       # (not config/nn: that folder is in the simulator's version hash)


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    """config/label.json without its comments."""
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
