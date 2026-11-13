"""The chain's battle in the game (chain.sh) and its analysis (host python, no torch):

    analyze.py   one battle: step time, orders reaching the units, units in their areas, contact, outcome (--json)
    compare.py   the base's numbers: winner, time, men and gold lost of each side, first contact
    units.py     per own unit: first melee, rout, firing seconds, ammunition, standing still, distance walked
    churn.py     per shooter / gun: orders given to the game and the bridge's own re-gives
    table.py     one row a run from the four above (the after-N tables)
    snip.py      own snipers, their guards and the enemy's snipers
    aivai.py     the game's AI against the game's AI with stage 0 armies (aivai.sh): arenas, measures, the twin

A run is a folder or a run name (yyyymmdd-hhmmss): build/nn-arena/runs/<name>, else the kept data/nn-arena/runs/<name>,
else data/human/runs/<name>.
"""
import json
from pathlib import Path

import game

UNITS_JSON = game.ROOT / "sim" / "config" / "units.json"


def run_dir(run):
    """A run folder from a path or a run name."""
    p = Path(run)
    if p.is_dir():
        return p
    for base in (game.BUILD / "nn-arena" / "runs", game.DATA / "nn-arena" / "runs", game.DATA / "human" / "runs"):
        if (base / str(run)).is_dir():
            return base / str(run)
    raise FileNotFoundError(f"no run {run}: not a folder, not in build/nn-arena/runs, data/nn-arena/runs, data/human/runs")


def costs():
    """{unit key: passport} of sim/config/units.json (the multiplayer cost)."""
    return json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"]
