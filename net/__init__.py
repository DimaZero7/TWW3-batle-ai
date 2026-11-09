"""The network's levels and their training on drills (net/README.md): unit -> tactician -> operative -> strategist,
each its own package; the stage's general battles and drills (stage0/), the user's matchups in the simulator (matchup/).

    unit/ tactic/ operative/ strategist/   a level: its input, network, teacher (formula), battle, training, exam;
                                           <level>/ckpt.py loads its weights without the training (the bridge's way in)
    matchup/   M1-M3 (frozen armies and places, net/matchup/m*.json), the chain against a script, the operative's drills
    stage0/    stage 0 'infantry only': general battles, drills, scripted behaviours, the rank check
    cards/     the units' cards (passport, abilities, effects) the inputs read
    common/    PPO's pieces, the cadence, the GPU lock
    config/    the levels' numbers (each with its _why); weights.json - the networks (files in weights/, out of git)
"""
from pathlib import Path

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent                     # the project's root
CONFIG = HOME / "config"               # tactic.json, operative.json, strategist.json, matchup.json, stage0.json
WEIGHTS = ROOT / "weights"             # the networks' files (out of git; the list: net/weights.json)
