"""The user's matchups in the simulator (sim/evidence/diag_m1/night.md; net/config/matchup.json): the armies and places the game's
battle is built from (tools.build human: army_from_run / armies.generate, arena_map.on_map), on the captured Kislev
plains with the game's deployment zones; a battle of two sides, each a chain of our levels or a script of the
simulator.

    armies.py   base(name) -> {side: [unit]} the matchup's armies and places in the map's frame (the game's); the human's
                own deployment of his run where there is one; augment(base, gen) - places, mirror, units out
    duel.py     play(...) a batch of battles to their end: each side a chain (the operative's formula or network, the
                tactician's formula or network, the frozen unit network) or a script (opponents.ai_like, nearest,
                hold_shoot); the readings: the winner, the gold lost, the time
"""
import json
from functools import lru_cache

from net import CONFIG as NET_CONFIG

CONFIG = NET_CONFIG / "matchup.json"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
