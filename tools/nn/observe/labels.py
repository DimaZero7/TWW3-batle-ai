"""Another player's new orders read from a recording, in the network's language (v2's chained heads, chain.py).

Only the seconds with a NEW order get a label (docs/en/training/training.md "Learning by observation"): a human cannot
command every unit every second, and his "nothing new" is mostly a lack of hands, not a choice - copying it would teach
passivity. The label sits on the decision BEFORE the change (the input the player saw when he gave it).

A new order of unit i between two recorded seconds k - 1 and k (both: men, not routing or shattered, on the map):
* the engine target became another present enemy -> attack it. Not for a missile unit firing at will (firing, not
  moving, its order point not moved: the engine picks the target itself - the "hold" of a shooter read as attack, 17 %
  in build/observe/infer_acc.py). The target must be one the side may attack at k - 1 (Obs.target_ok: seen, alive);
* else the order point moved more than SHIFT_M -> a hold when the unit does not move the first recorded second after
  the change and the new point is within STOP_M of it (the game's halt: on the network's battles its holds stood at k in
  99 % of the cases, its moves moved in 99 %, the point of a hold up to 40 m off); else a move to the point (the formation's front -> its centre, as tools/nn/sim/replay.py half_depth),
  running when the run flag is on at k or k + 1. A running move and a withdraw are one order in the game
  (bridge/adapter.lua: withdraw = move with run): the label is the set {move + run, withdraw} (loss.py takes the log of
  the sum of both probabilities). The game AI's and CA's planner's melee units (not missile units) get no label in
  melee: their recorded point lies anywhere while they fight on (replay.py "leavers"); a human's do (he leaves).
* nothing else is labelled (the commitment's duration, the abilities, formation and width: not taught).
The point -> the nearest cell of the network's grid inside the map (sectors.point_index; the point clipped to the map
first, record.clip_points). The run head is taught only when the unit moves at k or k + 1.

The heads taught are those the reading gets right (infer_acc.py on 306 network battles): kind 89 % (hold 73 / move 87
/ attack 98 %; withdraw only as the move's set), attack target 93 %, the point 100 % within 12.5 m, run 99.9 %.
"""
import math

import numpy as np

from tools.nn.sim.orders import ATTACK, HOLD, MOVE

SHIFT_M = 1.0          # m: the order point moved more than this -> a new order (tools/nn/human_orders.py)
STOP_M = 40.0          # m: ... a hold when the unit does not move the first second after it and the point is this near
NONE = -1


def grid_cells(cfg, geo):
    """(point index [P], centre (forward, lateral) [P, 2]) of the grid's cells inside the map (geo [8]: sectors.geo)."""
    import torch
    from tools.nn.model import sectors
    inside = sectors.inside(cfg, torch.as_tensor(np.asarray(geo, np.float32)).reshape(1, -1))[0].numpy()   # [S, k2]
    c = sectors.cells(cfg).numpy()                                                              # [S, k2, 2]
    s, q = np.nonzero(inside)
    idx = sectors.point_index(cfg, torch.as_tensor(s), torch.as_tensor(q)).numpy()
    return idx, c[s, q]


def cell_of(frame, px, pz, cells):
    """The grid cell (point index) of world points px, pz [L] - the nearest cell inside the map (cells: grid_cells)."""
    idx, centre = cells
    f, l = frame.point(np.asarray(px, float)[None], np.asarray(pz, float)[None])
    p = np.stack([f[0], l[0]], -1)                                                   # [L, 2]
    d = ((p[:, None, :] - centre[None]) ** 2).sum(-1)
    return idx[d.argmin(1)]


def half_depth(men, width, spacing=1.5):
    """Half the depth of a formation (tools/nn/sim/replay.py half_depth)."""
    if not width or men <= 1:
        return 0.0
    files = min(max(1.0, math.floor(width / spacing + 1e-4)), max(men, 1.0))
    return math.ceil(max(men, 1.0) / files) * spacing / 2


def read(raw, side, own, ctrl, target_ok, missile, human, widths=None, spacing=1.5):
    """The labels of side `own` -> dict of [T, N]: kind (NONE: no label), target, px, pz (the move point, world; NaN),
    run, run_known (the run head taught), alt (the set {move + run, withdraw}).
    raw: {field: [T, N]} the recorded fields (record.RAW; ox / oz clipped to the map); side [N]; ctrl, target_ok
    [T, N] of the side's observations; missile [N] (a missile range); human: a human's side (his melee units leave);
    widths [N] the formations' widths (None: no front -> centre)."""
    T, N = raw["x"].shape
    kind = np.full((T, N), NONE, np.int64)
    target = np.full((T, N), NONE, np.int64)
    px = np.full((T, N), np.nan)
    pz = np.full((T, N), np.nan)
    run = np.zeros((T, N), bool)
    run_known = np.zeros((T, N), bool)
    alt = np.zeros((T, N), bool)
    men = np.nan_to_num(raw["men"])
    up = (men > 0) & ~raw["r"] & ~raw["s"] & np.isfinite(raw["x"]) & np.isfinite(raw["z"])
    for k in range(1, T):
        nxt = min(k + 1, T - 1)
        for i in np.nonzero((side == own) & up[k - 1] & up[k] & ctrl[k - 1])[0]:
            tk, tp = int(raw["target"][k, i]), int(raw["target"][k - 1, i])
            enemy = 0 <= tk < N and side[tk] != own and men[k, tk] > 0
            o0 = (raw["ox"][k - 1, i], raw["oz"][k - 1, i])
            o1 = (raw["ox"][k, i], raw["oz"][k, i])
            moved_pt = (all(np.isfinite(o0)) and all(np.isfinite(o1))
                        and math.hypot(o1[0] - o0[0], o1[1] - o0[1]) > SHIFT_M)
            moving = bool(raw["mv"][k, i] or raw["mv"][nxt, i])
            fast = bool(raw["f"][k, i] or raw["f"][nxt, i])
            if enemy:
                if tk == tp:
                    continue
                if missile[i] and raw["fire"][k, i] and not raw["mv"][k, i] and not moved_pt:
                    continue                      # a shooter firing at will
                if not target_ok[k - 1, tk]:
                    continue
                kind[k - 1, i], target[k - 1, i] = ATTACK, tk
                run[k - 1, i], run_known[k - 1, i] = fast, moving
                continue
            if not moved_pt:
                continue
            if not human and not missile[i] and (raw["m"][k - 1, i] or raw["m"][k, i]):
                continue                          # the AI's melee unit: its point in melee is no order
            x, z = raw["x"][k, i], raw["z"][k, i]
            if not raw["mv"][k, i] and math.hypot(o1[0] - x, o1[1] - z) < STOP_M:
                kind[k - 1, i] = HOLD
                continue
            back = half_depth(men[k, i], widths[i] if widths is not None else None, spacing)
            br = math.radians(float(np.nan_to_num(raw["b"][k, i])))
            kind[k - 1, i] = MOVE
            px[k - 1, i], pz[k - 1, i] = o1[0] - back * math.sin(br), o1[1] - back * math.cos(br)
            run[k - 1, i], run_known[k - 1, i] = fast, True
            alt[k - 1, i] = fast
    return dict(kind=kind, target=target, px=px, pz=pz, run=run, run_known=run_known, alt=alt)
