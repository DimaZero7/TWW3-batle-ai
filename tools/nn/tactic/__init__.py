"""The tactician: its input and analysis, the council, the zones of its units, the first drill (design/tactic/index.html;
build/diag_g49/unit_base.md "Заметки для старших уровней"): a group of 1-10 own units with one task from the operative.

    raw.py      the tactician's raw state of every slot: the unit level's view.RAW + the effects' clocks, the buttons,
                the received abilities (one formula for the simulator and the bridge)
    rows.py     the units' rows (passport 69 + live 22 + who 5 + effects 25 x 2 + stats now 15 + arsenal 24 = 185) in
                the zone's axes, the rows' zone of attention (K rows)
    parcel.py   the operative's parcel (my slots, the target's, the zone's centre, turn and form, the group's number),
                its checks and a random generator of correct parcels
    zone.py     the zone as tiles (80 x 80 m a unit), the tile dropped where a unit fell out
    windows.py  the three map windows (near 34, far 28, target 17 layers; 30 x 30) and the 4 numbers
    inputs.py   the whole input of a group from the raw state, the passports and the parcel
    net.py      'Анализ отряда' (185 -> 256 -> 128), the card on the map (128 -> 8), 'Анализ карты' (three 3x3 convs)
    pretrain_card.py  the card's squeeze trained on ~10 facts (is 8 enough?)
    uzone.py    'Зона отряда': frames from a size and a cell, the bans, the nudge, the last resort (code, no weights)
    decide.py   the summary (map 128 + numbers 4 + order 7 -> 128), the council (unit.net.Block x 2), the size and 'where'
                heads, the walk in slot order, TacticNet, TacticCritic
    teacher.py  the first drill's formation as a formula: per decision a label, the term of the training, the yardstick
    drill1.py   the first drill: 2 infantry + 1 lobbing shooters against 3 infantry under a script; the contact's reading
    env.py      a batch of battles: the tactician every 5 s, the frozen unit network a second, the reward
    exam1.py    the first drill's exam; train.py PPO over the tactician's decisions

The bridge's side: tools/nn/companion/tactic_input.py (the same input from the bridge's documents).
The numbers: config/nn/tactic.json.
"""
import json
from functools import lru_cache

from tools import config as project

CONFIG = project.ROOT / "config" / "nn" / "tactic.json"


@lru_cache(maxsize=1)
def cfg():
    """config/nn/tactic.json without its comments (keys starting with '_')."""
    d = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}
