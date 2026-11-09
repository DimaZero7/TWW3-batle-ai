"""The operative: the network over the tacticians (tmp/old-build/diag_operative/plan.md; design/tactic, the left zone "Зона
оперативника"). It cuts the army into groups, gives each a target, a zone, a form and an order - the parcel to a
tactician exactly in net/tactic/parcel.py's format - once in 15 s; under it a tactician (the formula
tactic/teacher.drive, or a TacticNet) every 5 s and the frozen unit network every second.

    frame.py    the field's axes (the field's own axis nearest "towards the enemy", a multiple of 90 deg) and a unit's place
    rows.py     the rows of every own unit and every seen enemy (the tactician's row formula, the origin at the unit
                itself) + place 3 + the last groups 14
    fmap.py     the whole field's map 32 x 32 x 33 (the tactician's window builders, a cell of 64 m)
    groups.py   the groups' state between decisions: members, task, the tiles' count, age; deaths
    target.py   the target grown from a seed by the squares' rule; the turn's bans
    step.py     the zone's size, the cells' bans (the zone out of the field), the step towards the goal by code
    net.py      the input, the records, the group tokens, the council, the heads, the walk; the critic
    send.py     the parcels and orders of a decision, checked by parcel.check
    lower.py    the tacticians of all groups (formula or a TacticNet), their tiles and frames, the units' areas
    cases.py    the three cases: gather, split, go-or-stay
    watch.py    the cases' readings (contacts per clump, gathered, obeyed) and the end
    teacher.py  the operative's formula (and the plain operators the exam compares with), the teacher's term
    env.py      a batch of battles played to the end; exam.py; train.py (PPO)

The bridge's side: bridge/companion/operative_input.py. The numbers: net/config/operative.json.
"""
import json
from functools import lru_cache

from net import CONFIG as NET_CONFIG

CONFIG = NET_CONFIG / "operative.json"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    """net/config/operative.json without its comments (keys starting with '_', at every level)."""
    return _strip(json.loads(CONFIG.read_text(encoding="utf-8")))
