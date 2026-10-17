"""The fighters lab (docs/en/training/fighters_lab.md): every man of a unit a point with his own health, inside the
simulator's step (hook.py) - apart from training; the simulator's files are not changed.

  state.py   the men: places (offsets from the unit's centre), health, attack clocks; formation places; conform
  cells.py   neighbours by cells (fixed sizes)
  melee.py   blows man by man
  missile.py projectiles man by man
  move.py    the men's own moves
  hook.py    into the simulator's step
  shape.py   the shape numbers from men's places (numpy: the game's probes and the lab alike)
  probes.py  the probe lanes (charge_probe) with the men
  replay.py  the step replay with the men; card.py the comparison
  soldiersdata.py  the game's recorded soldiers' places (nn_soldiers), the men-in-reach geometry
  places.py  the step replay's windows from those places
  shapecheck.py  shape.unit_features (the unit's shape for the network): the lab against the game; its cost
  bench.py   speed
"""
