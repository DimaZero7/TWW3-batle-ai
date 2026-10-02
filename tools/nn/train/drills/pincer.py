"""Drill "pincer": our infantry outnumbers an enemy group two to one; piled on an enemy unit's front
it loses, enveloping it (one of ours pins it in front, the other strikes its flank) breaks it.
Needs the flank batch (build/sim-pending/flank.json: melee.flank_face "striker", flank_slope 3.8,
rear_slope 5.5): with the old rule a flank attacker strikes no harder than a frontal one and
the drill does not hold (drills.READY leaves it out until that batch is in the simulator).

Frame (one battle):
* the enemy: K = 2-3 units of Empire greatswords standing SPREAD_M apart (each its own fight),
  facing us; they hold (stand, fight back when attacked: the `hold` script) and defend - the enemy
  wins at the battle limit;
* ours: two units per enemy unit, in a column opposite it (the second COLUMN_M behind the first),
  GAP_M away, all Empire swordsmen or all Skaven clanrats (with clanrats K is at most 2); we attack;
* random: the roster, K, the gap, the spread, the column's depth, small lateral offsets, the place
  and bearing on the map, our side (1 or 2).
Scripts:
* naive: every unit attacks the nearest enemy, running: the pair piles on its enemy's front;
* skilled: per enemy unit (each unit's nearest), the nearer of ours pins it (walks up to PIN_M in
  front, waits there until its flanker is near its place, then attacks); the other goes round to a
  point beside the enemy's open flank and slightly behind its centre (a waypoint keeps it clear of
  the enemy's front), then attacks into the flank.
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.train import drills as D

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
# The matchup: one enemy unit that beats two of ours piled on its front, and loses to the same two
# when one pins it and the other strikes its flank or rear (measured in the simulator with the flank
# batch, build/sim-pending/flank.json; 32 battles each, build/drills/flank_exp.py: Empire greatswords
# against two Empire swordsmen - frontal pile 0.09 wins, front + flank 0.94, front + rear 0.91; against
# two Skaven clanrats (shields) 0.03 / 0.84 / 0.50).
ENEMY = ("wh_main_emp_inf_greatswords", EMPIRE)
# (key, faction, most enemy units): clanrats against three greatswords won 0.35 of the battles piling in
# frontally (256 battles), so with them at most two
OURS = (("wh_main_emp_inf_swordsmen", EMPIRE, 3), ("wh2_main_skv_inf_clanrats_1", SKAVEN, 2))
K = (2, 3)               # enemy units, each its own fight
SPREAD_M = (120.0, 180.0)    # between the enemy units (fights apart: a pair's neighbours do not join)
GAP_M = (140.0, 240.0)   # our front units to the enemy line
COLUMN_M = (35.0, 55.0)  # our second unit of a pair this far behind the first (a column)
JITTER_M = 8.0           # lateral offsets of ours from their enemy's line
WIDTH_M = 30.0
PIN_M = 45.0             # the pinner waits this far in front of its target
FLANK_M = 18.0           # the flank point: this far beside the target's flank ...
BEHIND_M = 6.0           # ... and this far behind its centre
READY_M = 12.0           # a flanker this near its point attacks
REAR_M = 25.0            # rank 3+: a point this far behind the target
GO_M = 35.0              # the pinner goes when its flanker is this near its point


LAYOUT = "apart"         # "line": the enemy units side by side (no open flank between them); "apart"
LINE_GAP_M = (2.0, 6.0)  # between the enemy units of a line


def frame(rng):
    mine = OURS[int(rng.integers(len(OURS)))]
    k = int(rng.integers(K[0], min(K[1], mine[2]) + 1))
    gap = float(rng.uniform(*GAP_M))
    step = WIDTH_M + float(rng.uniform(*LINE_GAP_M)) if LAYOUT == "line" else float(rng.uniform(*SPREAD_M))
    enemy, ours = [], []
    for i in range(k):
        lat = (i - (k - 1) / 2) * step
        enemy.append(D.unit(ENEMY[0], *D.local(gap / 2, lat), 180.0, WIDTH_M))
        col = float(rng.uniform(*COLUMN_M))
        for back in (0.0, col):
            j = float(rng.uniform(-JITTER_M, JITTER_M))
            ours.append(D.unit(mine[0], *D.local(-gap / 2 - back, lat + j), 0.0, WIDTH_M))
    return D.army(ours, enemy, attacker=1, ours_faction=mine[1], enemy_faction=ENEMY[1])


def enemy(st):
    return D.hold(st)


def naive(st):
    return D.nearest(st)


def skilled(st):
    v = D.View(st)
    u = st.u
    x, z = u["x"], u["z"]
    slot = torch.arange(st.N, device=st.device)
    tgt, _ = v.nearest()
    has = v.melee & (v.nearest()[1] < 1e9)
    cur = u["order_target"].clamp(min=0)
    committed = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & v.foe.gather(2, cur[:, :, None]).squeeze(2)
    tgt = torch.where(committed, cur, tgt)
    dist = v.d.gather(2, tgt[:, :, None]).squeeze(2)
    # rank among own units on the same target by distance: 0 pins it in front, 1 and 2 its flanks, 3+ its rear
    same = v.friend & has[:, None, :] & has[:, :, None] & (tgt[:, :, None] == tgt[:, None, :])
    closer = same & ((dist[:, None, :] < dist[:, :, None])
                     | ((dist[:, None, :] == dist[:, :, None]) & (slot[None, None, :] < slot[None, :, None])))
    rank = closer.sum(2)
    take = lambda a: a.gather(1, tgt)
    tx, tz, tb, tw = take(x), take(z), torch.deg2rad(take(u["b"])), take(u["width"])
    fx, fz = torch.sin(tb), torch.cos(tb)              # the target's facing
    rx, rz = torch.cos(tb), -torch.sin(tb)             # its right
    lat = (x - tx) * rx + (z - tz) * rz
    # the target's open side: away from its own group's centre (the outer end of a line); alone: the
    # flanker's own side; a target in the middle of a line has none - its flanker goes to its rear
    (_, _), (gx, gz) = v.side_centroid(v.standing)
    gxt, gzt = take(gx), take(gz)
    g_lat = (tx - gxt) * rx + (tz - gzt) * rz
    alone = g_lat.abs() < 1.0
    outer = torch.where(g_lat >= 0, torch.ones_like(lat), -torch.ones_like(lat))
    own = torch.where(lat >= 0, torch.ones_like(lat), -torch.ones_like(lat))
    side = torch.where(alone, own, outer)
    off = tw / 2 + FLANK_M
    px = torch.where(rank >= 2, tx - REAR_M * fx, tx + side * off * rx - BEHIND_M * fx)
    pz = torch.where(rank >= 2, tz - REAR_M * fz, tz + side * off * rz - BEHIND_M * fz)
    # on its way round a flanker keeps clear of the enemy's front: a waypoint beside the target's open
    # side and ahead of it first
    wx, wz = tx + side * (off + 10.0) * rx + 10.0 * fx, tz + side * (off + 10.0) * rz + 10.0 * fz
    ahead = (x - tx) * fx + (z - tz) * fz
    to_point = torch.sqrt((px - x) ** 2 + (pz - z) ** 2)
    flanker = has & (rank > 0)
    pinner = has & (rank == 0)
    go_in = flanker & ((to_point <= READY_M) | committed | u["m"])
    o = O.hold(st.B, st.N, st.device)
    via = flanker & ~go_in & (ahead > 0) & (lat * side < off)
    D.move(o, flanker & ~go_in, px, pz, run=True)
    D.move(o, via, wx, wz, run=True)
    D.attack(o, go_in, tgt, run=True)
    # the pinner waits PIN_M in front until a flanker on its target is near its point (or there is none)
    fl_near = (same & (flanker & (to_point <= GO_M))[:, None, :]).any(2)
    fl_any = (same & flanker[:, None, :]).any(2)
    go = pinner & (committed | u["m"] | fl_near | ~fl_any)
    D.move(o, pinner & ~go, tx + PIN_M * fx, tz + PIN_M * fz, run=False)
    D.attack(o, go, tgt, run=True)
    return o


DRILL = D.Drill("pincer", frame, enemy, naive, skilled,
                "more infantry than an enemy group: pin in front, strike the flank or rear")
