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
    drill_hold.py  drill U1 'hold the line': melee infantry Empire x Skaven, an area of 6..10 x 10 cells on the unit's
              grid, the enemy from any side 75-160 m away under `nearest`, the end rule (our rout; the enemy's rout
              goes on until it leaves the zone; a 3 min cut)
    exam_hold.py   its fixed exam (360 skirmishes): met head-on, melee damage to the front, out of the area, trade
    drill_go.py    drill U2 'reach and hold': the unit starts out of its area (10-700 m), the order 'hold' at the stage
              'going' -> 'in place' on arrival, the pace fast / slow by the order, the enemy goes to the area (to_area)
              with a race early / close / held; the limit 3 min after the arrival
    exam_go.py     its fixed exam (360): reached, when, before the enemy, the area ours, head-on, out after arrival, trade
    drill_multi.py drill U3 'several enemies': as U1 with two enemies A and B (front + flank / front + rear / both ahead
              at an angle; together or B 5-20 s later), both under `nearest`
    exam_multi.py  its fixed exam (240): chose the best of 4 variants, which is best, damage front / flank-rear, head-on
    drill_pair.py  drill U4 'friends beside': two of our units (both the network's) with areas side by side or in
              echelon, 1-2 enemies under `nearest`; the side's reward shared
    exam_pair.py   its fixed exam (240): jostling, damage front / flank-rear, head-on, the pair's trade, out of the area
    drill_shoot.py drill U5 'shooters - fire at one target': a shooter in a small position with a fire sector (any target
              in the sector and in range, wherever; it never walks to a target - decode.Leash), one infantry enemy
              going at it or past
    exam_shoot.py  its fixed exam (240): fire time while in range, damage by fire (before contact), melee losses, trade
    drill_withdraw.py  drill U6 'withdraw from melee': in melee (5 s of it), ordered to a new area 40-150 m behind or
              aside at a random pace; the simulator's melee exit, the enemy's pursuit under `nearest`
    exam_withdraw.py   its fixed exam (240): reached, when, losses on the way, routed, damage in the back, trade
    branch.py the approach's branches (U1, U3): the skirmish copied when an enemy is within its charge distance + 4 s
              of its run, played on per variant (U1 stand / attack; U3 attack A / B, stand facing A / B); a decaying
              bonus x the best's margin over the second when the network did the best

Outside tools/nn/model on purpose: that folder is part of the simulator's version (tools/nn/train/version.py).
"""
