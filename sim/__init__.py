"""Battle simulator of TWW3 (sim/README.md): many battles at once as batched torch tensors, one unit = one row (men,
health, place, facing, morale, fatigue, ammunition).

    state.py     the state layout (the contract with the network)
    orders.py    the order layout (the contract with the network)
    params.py    numbers: passports, the game's rules, measurements (sim/config/*.json)
    scenario.py  armies -> the first state
    battle.py    one step of all battles (stepper, run), the end of a battle
    geometry.py, movement.py, melee.py, missile.py, artillery.py, impact.py, morale.py, fatigue.py, ... the rules
    terrain.py, ground.py, detour.py   the captured map: height, ground types, trunks, 'cannot stand'
    scripts.py   scripted opponents (ai_game, nearest, hold_shoot, ...)
    reward.py    the battle's score (gold, lord) and the reward's terms
    arena.py     the arena format (sides, places, the game's battle file); maps/ the captured maps (MAPS)
    armies/      random armies (pools, templates); passports/ the game's database -> sim/config/*.json
"""
from pathlib import Path

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent                      # the project's root
CONFIG = HOME / "config"                # sim.json (measured, each number with its _why), units / abilities / effects ...
DATA = HOME / "data"                    # probe-units.json, roster/ (the game's unit cards)
