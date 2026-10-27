"""The unit level (g49, the personal level): one network that commands one unit, its input only its own zone.
Design: build/diag_g49/unit_base.md (the user's decisions), build/diag_g49/plan_unit_net.md (the work plan).

    zmap.py   the zone map: one builder for every level (centre, turn front-up, cell size; 30 x 30 cells; layers
              enemies / own / self as density 0..1, the fronts' direction as cos / sin, my area, the map's edge)
    view.py   the unit's view from the raw state: its zone (who can reach it within 15 s, by any part of the formation),
              up to 16 rows of raw numbers in its frame (itself, enemies, own), 10 snapshots of memory put into its
              frame of now, its zone map, the bans' masks (cells and targets outside its area). Training stores only
              raw() per decision and rebuilds the view
    net.py    the network (~0.32 M weights: map convs, row encoder, memory GRU, attention, core; heads where / face 64 /
              target pointer / pace with hard bans) and the critic (the same trunk, own weights, the whole skirmish)
    decode.py the choices -> the simulator's orders (a target: ATTACK; a cell: MOVE to its centre, my own cell: HOLD; the
              facing); the leash: a pursuit about to carry the unit out of its area stops at the edge facing the
              target until the target is back (not in melee, not routing); the unit's row shows it
    train.py  the level's own small PPO loop and the smoke battle (spearmen vs clanrat spearmen under `nearest`);
              stores only the raw state per decision and rebuilds the views in the update; checkpoint kind unit_v1

Outside tools/nn/model on purpose: that folder is part of the simulator's version (tools/nn/train/version.py).
"""
