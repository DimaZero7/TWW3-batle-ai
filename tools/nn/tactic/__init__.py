"""The tactician's input and its analysis (design/tactic/index.html; build/diag_g49/unit_base.md "Заметки для старших
уровней"): a group of 1-10 own units with one task from the operative.

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
