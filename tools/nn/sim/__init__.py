"""Battle simulator v1 (docs/en/training/simulator.md): many battles at once as batched torch
tensors, one unit = one object (men, health, place, facing, morale, fatigue, ammunition).

    state.py     the state layout (the contract with the network)
    orders.py    the order layout (the contract with the network)
    params.py    numbers: passports, the game's rules, calibration (config/nn/sim.json)
    scenario.py  armies -> the first state
    geometry.py  distances, facing, contact
    movement.py  moving, fleeing, the map's edge
    melee.py     blows in melee
    missile.py   shooting
    morale.py    morale, wavering, rout, rally
    fatigue.py   fatigue
    battle.py    one step of all battles, the end of a battle
    replay.py    orders from recorded battles (open-loop)
    check.py     checks against the game: python -m tools.nn.sim.check
"""
