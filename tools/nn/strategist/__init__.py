"""The strategist: the blind operative before the battle (build/diag_strategist/plan.md). Once, at the deployment, it
cuts my army into groups, sets each group's zone in my deployment zone (a cell, a turn, a form) and gives the operative
its starting command (verb + pace). It sees my army, the enemy's composition (as the pre-battle screen), the map and both
deployment zones - never where the enemy stands. Under it the operative (its formula by default, config operative; its network for the chain), the tactician's formula and
the frozen unit network.

    zones.py    the deployment zones (turned rectangles), the strategist's axes, the cells' bans
    deploy.py   the groups' zones -> every unit's starting place (code, the tiles of the tactician), the checks
    inputs.py   the blind rows, the deployment map 32 x 32 x 8
    net.py      the encoder, the heads (members -> per group form, turn, cell -> verb, pace), the walk; the critic
    teacher.py  the strategist's formula and the plain strategies the exam compares with; the teacher's term
    cases.py    the cases: wide, go (hold or attack)
    battle.py   a batch: the decision -> the places -> the groups -> the operative from second 0 to the end
    exam.py, train.py (imitation without battles, then PPO of one decision a battle)

The numbers: config/nn/strategist.json.
"""
import json
from functools import lru_cache

from tools import config as project

CONFIG = project.ROOT / "config" / "nn" / "strategist.json"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


def default():
    """The strategist by default (config default): 'formula' or the absolute path of a checkpoint."""
    who = str(cfg()["default"])
    return who if who == "formula" else str(project.ROOT / who)


@lru_cache(maxsize=1)
def cfg():
    """config/nn/strategist.json without its comments (keys starting with '_', at every level)."""
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
