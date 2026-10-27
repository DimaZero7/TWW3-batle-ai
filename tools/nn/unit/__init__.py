"""The unit level (g49, the personal level): one network that commands one unit, its input only its own zone.
Design: build/diag_g49/unit_base.md (the user's decisions), build/diag_g49/plan_unit_net.md (the work plan).

    zmap.py   the zone map: one builder for every level (centre, turn front-up, cell size; 30 x 30 cells; layers
              enemies / own / self as density 0..1, the fronts' direction as cos / sin, my area, the map's edge)

Outside tools/nn/model on purpose: that folder is part of the simulator's version (tools/nn/train/version.py).
"""
