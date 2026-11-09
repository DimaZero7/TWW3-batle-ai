"""Stage 0 'infantry only' (tmp/old-build/diag_stage1/plan.md; net/config/stage0.json): the stage's general battles, the drills
that do not depend on the formation's width (S0-1 the line on the height, T0-2 pin and flank, O0-3 the reserve closes a
breach), the scripted behaviours each drill is ranked with, the rank check and the 'before' reading.

    specs.py      plain python: the stage's roster, armies equal in gold, the general battles' and the drills' specs
    battle.py     a batch of specs played to its end: my side the chain of our levels (the strategist's formula ->
                  the operative's -> the tacticians' -> the frozen unit network) with a behaviour on top, the lords
                  150 m behind holding (the chain does not see its own), the enemy a script or the formula chain
    watch.py      the readings every second -> the stage's measures (trade, flank / rear hits, gaps in the line, the
                  height at contact, the reserve's time, order changes per unit a minute) and the drills' own
    behave.py     the scripted behaviours: the strategist's placement (S0-1), the tactician's flank (T0-2), the
                  operative's reserve (O0-3) - each a rule that works in a drill and in a general battle alike
    rank.py       the CLI: the behaviours in the drill and in the general battles, Spearman; 'before' - the chain's
                  measures in the general battles
"""
import json
from functools import lru_cache

from net import CONFIG as NET_CONFIG

CONFIG = NET_CONFIG / "stage0.json"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    """net/config/stage0.json without its comments (keys starting with '_', at every level)."""
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
