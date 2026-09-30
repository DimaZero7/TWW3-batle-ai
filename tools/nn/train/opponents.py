"""Scripted opponents in the simulator (docs/en/training/training.md). The game's AI takes no part
in training: it stays an independent check.

Each is policy(state) -> Orders for every unit of both sides; the rollout keeps only the orders
of the side it plays.

    nearest     every unit attacks the nearest standing enemy, running (missile units close to
                their range and shoot)
    hold_shoot  hold and shoot at will; a melee unit counter-charges the nearest enemy once it is
                within `react_m`; as the attacker, after `patience_s` everyone attacks (or the
                attacker would just lose on time)
    hold        every unit holds (shoots at will, fights back when attacked)
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.sim import replay

REACT_M = 80.0
PATIENCE_S = 300.0


def _nearest(st):
    """(index [B, N] of the nearest standing enemy, its distance [B, N]; 1e9 when none)."""
    u = st.u
    standing = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]
    dx = u["x"][:, None, :] - u["x"][:, :, None]
    dz = u["z"][:, None, :] - u["z"][:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & standing[:, None, :]
    d = torch.where(enemy, d, torch.full_like(d, 1e9))
    return d.argmin(2), d.min(2).values


def nearest(st):
    return replay.nearest_attack(st)


def hold(st):
    return O.hold(st.B, st.N, st.device)


def hold_shoot(st, react_m=REACT_M, patience_s=PATIENCE_S):
    u = st.u
    near, dist = _nearest(st)
    has = dist < 1e9
    missile = (u["range"] > 0) & (u["a"] > 0)
    attacker = u["side"] == st.attacker[:, None]
    late = attacker & (st.t[:, None] >= patience_s)
    charge = has & ((~missile & ((dist <= react_m) | u["m"])) | late)
    o = O.hold(st.B, st.N, st.device)
    o.kind = torch.where(charge, torch.full_like(o.kind, O.ATTACK), o.kind)
    o.target = torch.where(charge, near, o.target)
    o.run = charge
    return o


SCRIPTS = {"nearest": nearest, "hold_shoot": hold_shoot, "hold": hold}
