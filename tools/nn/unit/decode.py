"""The unit network's choices -> the simulator's orders, and the leash (build/diag_g49/plan_unit_net.md step 5; the
user's decision on the pursuit: not a penalty, a hard ban in the decoder).

to_orders: a target chosen -> ATTACK it (run by the pace); else the cell chosen: the cell I stand in -> HOLD, another
-> MOVE to its centre; both with the facing chosen (64 sectors from my front: tools/nn/sim/orders.py Orders.face).
The bans (cells and targets outside the area, the pace) are in the network's logits (net.py); the decoder never
widens them.

Leash: while a unit attacks, its target may run out of its area and the pursuit would carry it out. Each simulator
step, a unit under an ATTACK order of its own choice whose target is beyond the area's edge (more than EDGE_M out) and
which is about to leave the area (its centre within INSET_M of the edge or out, LOOK_S ahead at its speed) is leashed:
its order is replaced by MOVE to the nearest point INSET_M inside the area, facing the target. It stays leashed while
its target stays out; once the target is back (inside or within EDGE_M of the edge) the attack goes on. A unit in
melee or routing is never leashed (pushed out in a fight or fleeing in panic is not its choice: no leash, no penalty);
the network sees the leash in its own row (view.py 'leashed').
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

EDGE_M = 10.0      # m: a target this far beyond the edge still is 'at the edge': the attack goes on
INSET_M = 2.0      # m: the leash's stand inside the edge; a unit closer to the edge than this is 'about to leave'
LOOK_S = 1.0       # s: 'about to leave' looks this far ahead at the unit's speed


def to_orders(act, v, now, cell_m, B, N, face_sectors=64, cells=Z.CELLS):
    """Orders [B, N] for the viewers v["bi"], v["ii"] (the rest: KEEP) from the sampled act (net.sample)."""
    dev = now.device
    bi, ii = v["bi"], v["ii"]
    R = bi.shape[0]
    me = now[bi, ii]
    o = O.hold(B, N, dev)
    o.kind[:] = O.KEEP
    K = v["slot"].shape[1]
    t = act["target"]
    attack = t < K
    slot = v["slot"].gather(1, t.clamp(max=K - 1)[:, None])[:, 0]
    attack = attack & (slot >= 0)
    row, col = act["where"] // cells, act["where"] % cells
    cm = cell_m if torch.is_tensor(cell_m) else torch.full((R,), float(cell_m), device=dev)
    s, f = Z.cell_centre(row.float(), col.float(), cm, cells)
    x, z = Z.world(s, f, me[:, V.I["x"]], me[:, V.I["z"]], me[:, V.I["b"]])
    here_r, here_c = Z.cell_of(torch.zeros(R, device=dev), torch.zeros(R, device=dev), cm, cells)
    here = (row == here_r) & (col == here_c)
    face = torch.remainder(me[:, V.I["b"]] + act["face"].float() * (360.0 / face_sectors), 360.0)
    kind = torch.where(attack, O.ATTACK, torch.where(here, O.HOLD, O.MOVE))
    o.kind[bi, ii] = kind
    o.target[bi, ii] = torch.where(attack, slot, torch.full_like(slot, -1))
    o.x[bi, ii] = torch.where(kind == O.MOVE, x, me[:, V.I["x"]])
    o.z[bi, ii] = torch.where(kind == O.MOVE, z, me[:, V.I["z"]])
    o.run[bi, ii] = act["pace"] == 1
    o.face[bi, ii] = torch.where(attack, torch.full_like(face, -1.0), face)
    return o


def signed_dist(x, z, area):
    """Signed distance (m, inside positive) of points to the areas (turned rectangles; fields broadcast with x)."""
    s, f = Z.local(x, z, area["x"], area["z"], area["b"])
    dx, dz = s.abs() - area["front"] / 2, f.abs() - area["depth"] / 2
    inside = (dx <= 0) & (dz <= 0)
    return torch.where(inside, -torch.maximum(dx, dz), -torch.sqrt(dx.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2))


def clamp_in(x, z, area, inset=INSET_M):
    """The nearest point inset m inside each area (a point already that far inside stays)."""
    s, f = Z.local(x, z, area["x"], area["z"], area["b"])
    hs = (area["front"] / 2 - inset).clamp(min=0)
    hf = (area["depth"] / 2 - inset).clamp(min=0)
    return Z.world(torch.maximum(torch.minimum(s, hs), -hs), torch.maximum(torch.minimum(f, hf), -hf),
                   area["x"], area["z"], area["b"])


class Leash:
    """The leash's state per slot (on [B, N]) and its rule, applied every simulator step to the decoder's orders."""

    def __init__(self, B, N, device="cpu"):
        self.on = torch.zeros((B, N), dtype=torch.bool, device=device)

    def apply(self, st, orders, area, own):
        """orders: the decoder's orders in force (ATTACK of the unit's choice re-given each step); area {x, z, b, front,
        depth[, on]} [B, N]; own [B, N] the units the network commands. Returns the orders with the leashed ones
        replaced (a new Orders; orders itself is not changed) and updates self.on."""
        u = st.u
        on_area = area.get("on", torch.ones_like(own)) & own
        attack = (orders.kind == O.ATTACK) & (orders.target >= 0) & on_area
        free = attack & ~u["m"] & ~u["r"] & (u["men"] > 0)
        tg = orders.target.clamp(min=0)
        tx, tz = u["x"].gather(1, tg), u["z"].gather(1, tg)
        target_out = signed_dist(tx, tz, area) < -EDGE_M
        ahead = signed_dist(u["x"] + u["vx"] * LOOK_S, u["z"] + u["vz"] * LOOK_S, area)
        leaving = (signed_dist(u["x"], u["z"], area) < INSET_M) | (ahead < INSET_M)
        self.on = free & target_out & (self.on | leaving)
        shoot_hold = torch.zeros_like(self.on)
        if "sw" in area:
            # a shooter with a fire sector (U5): it fires, it does not walk to its target - an attack on an enemy beyond
            # its range (the simulator would close in) is a hold facing it until the enemy comes within range; the
            # pursuit's leash above is not its (its targets lie beyond its position by design)
            shooter = (area["sw"] > 0) & attack
            far = torch.sqrt((tx - u["x"]) ** 2 + (tz - u["z"]) ** 2) > u["range"]
            shoot_hold = shooter & far & ~u["r"]
            self.on = (self.on & ~shooter) | shoot_hold
        out = orders.clone()
        px, pz = clamp_in(u["x"], u["z"], area)
        face = torch.remainder(torch.rad2deg(torch.atan2(tx - u["x"], tz - u["z"])), 360.0)
        on = self.on & ~shoot_hold
        out.kind = torch.where(shoot_hold, O.HOLD, out.kind)
        out.target = torch.where(shoot_hold, torch.full_like(out.target, -1), out.target)
        out.face = torch.where(shoot_hold, face, out.face)
        out.kind = torch.where(on, O.MOVE, out.kind)
        out.target = torch.where(on, torch.full_like(out.target, -1), out.target)
        out.x = torch.where(on, px, out.x)
        out.z = torch.where(on, pz, out.z)
        out.face = torch.where(on, face, out.face)
        return out
