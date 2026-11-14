"""The unit level (the personal level): the unit script (scripts.py) carries out the tactician's order; the old unit
network (one unit, its input only its own zone; weights/unit/mixmap_run2) still drives the units of the levels above and
the bridge.
Design: sim/evidence/diag_g49/unit_base.md (the user's decisions), sim/evidence/diag_g49/plan_unit_net.md (the work plan).

    zmap.py   the zone map: one builder for every level (centre, turn front-up, cell size; 30 x 30 cells; layers
              enemies / own / self as density 0..1, the fronts' direction as cos / sin, my area, the map's edge, a
              shooter's range circle and fire arc now, the ground's height against mine and its slope forward / right
              (sim/terrain.py); the cell 8 m, a shooter's its range / 14: view.cell_size)
    view.py   the unit's view from the raw state: its zone (who can reach it within 15 s, by any part of the formation),
              up to 16 rows of raw numbers in its frame (itself, enemies, own), 10 snapshots of memory put into its
              frame of now, its zone map, the bans' masks (cells and targets outside its area). Training stores only
              raw() per decision and rebuilds the view
    net.py    the network (~0.32 M weights: map convs, row encoder, memory GRU, attention, core; heads where / face 64 /
              target pointer / pace with hard bans / width: keep or 7 steps) and the critic (the same trunk, own
              weights, the whole skirmish)
    decode.py the choices -> the simulator's orders (a target: ATTACK; a cell: MOVE to its centre, my own cell: HOLD; the
              facing; the width between the unit's narrowest and widest, Orders.width); the leash: a pursuit about to carry the unit out of its area stops at the edge facing the
              target until the target is back (not in melee, not routing); the unit's row shows it
    ckpt.py   the network's checkpoint (kind unit_v1), its cards and views, without training (the bridge's way in)
    drill_hold.py  the skirmish of the old drill U1 'hold the line': melee infantry Empire x Skaven (roster), an area
              of 6..10 x 10 cells on the unit's grid, the enemy from any side 75-160 m away under `nearest`, the end
              rule (Ender); the generator of the tactician's and the levels' cases above, u2.py and skills.py
    drill_go.py    the skirmish of the old drill U2 'reach and hold' and the enemy's script to_area (u2.py, skills.py)
    realmap.py     skirmishes laid on the captured maps at a random place and turn (u2.py)
    scripts.py     the unit level now: the script that carries out the tactician's order (no network)
    u2.py, skills.py, exam_u2.py   the unit script's drills and exams (METHOD.md 'Личный уровень - без сети')

Outside tools/nn/model on purpose: that folder is part of the simulator's version (tools/nn/train/version.py).
"""
