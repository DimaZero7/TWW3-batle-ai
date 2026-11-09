"""Mechanics probes in the game and the simulator's twin on the same lanes (game/README.md):

    charge_probe.py   melee, movement, morale, the new units' plans (PLANS; unit_probe.py the new units' lanes and tables)
    missile_probe.py  fire: spread, ranges, friends in the line of fire
    morale_probe.py   morale: flanks, timers, effects
    move_probe.py     a move-probe run's report (bridge.build move-probe: paths, detours, formation shapes)
    mem_probe.py      memreader Plus rows of a probe run (--mem): projectiles, men, morale from the game's memory
    run.sh            plans in turn, one battle each, with the game lock and --mem
    evidence/         the stage 0-1 probes' tables behind sim/config/sim.json _why (hill, flank, lob, forest)
"""
