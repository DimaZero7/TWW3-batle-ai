"""Policies without a network: hold, attack the nearest enemy, and replaying the orders of recorded
battles open-loop (docs/en/training/simulator.md).

Replay: for every recorded second and unit, the order that recording implies -
    a recorded target (fought or shot)   -> ATTACK it (a missile unit closes to its range)
    in melee without a target            -> HOLD (fight whoever is in contact)
    otherwise                            -> MOVE to the recorded order point (ox, oz),
                                            running if the unit was running After the recording ends the
last recorded orders stay for `grace_s` (120 s), then every unit attacks the nearest enemy.
"""
import numpy as np
import torch

from tools.nn.sim import orders as O


def hold(st):
    return O.hold(st.B, st.N, st.device)


def nearest_attack(st, run=True):
    """Every standing unit attacks the nearest standing enemy (missile units close to their range)."""
    u = st.u
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    dx = u["x"][:, None, :] - u["x"][:, :, None]
    dz = u["z"][:, None, :] - u["z"][:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & standing[:, None, :]
    d = torch.where(enemy, d, torch.full_like(d, 1e9))
    near = d.argmin(2)
    has = d.min(2).values < 1e9
    o = O.hold(st.B, st.N, st.device)
    o.kind = torch.where(has & standing, O.ATTACK, O.HOLD)
    o.target = torch.where(has & standing, near, torch.full_like(near, -1))
    o.run = torch.full_like(o.run, run)
    return o


def half_depth(men, width, spacing=1.5):
    """Half the depth of a formation of `men` with a `width` front (numpy); 0 for a single man."""
    files = np.maximum(1, np.round(np.asarray(width, float) / spacing))
    files = np.minimum(files, np.maximum(men, 1))
    return np.where(men > 1, np.ceil(np.maximum(men, 1) / files) * spacing / 2, 0.0)


def recorded_orders(battle, slot_of, N, widths=None, spacing=1.5):
    """Orders implied by a recorded battle (tools/nn/gamedata.Battle), one row per recorded second:
    dict of arrays [T, N] kind, x, z, target, run in the simulator's slots (slot_of: recorded index
    -> slot). The game records the order's point at the formation's front (measured: half a depth
    ahead of the centre); widths [recorded index] (m) turn it into the centre the simulator goes to."""
    f = battle.f
    T = len(battle.t)
    kind = np.full((T, N), O.HOLD, dtype=np.int64)
    x = np.zeros((T, N), dtype=np.float32)
    z = np.zeros((T, N), dtype=np.float32)
    target = np.full((T, N), -1, dtype=np.int64)
    run = np.zeros((T, N), dtype=bool)
    ammo = np.nan_to_num(f["a"], nan=0.0)
    for i, s in enumerate(slot_of):
        tg = battle.target[:, i]
        ok_t = tg >= 0
        mapped = np.where(ok_t, np.array(slot_of)[np.clip(tg, 0, None)], -1)
        attack = ok_t & (battle.side[np.clip(tg, 0, None)] != battle.side[i])
        ox, oz = np.nan_to_num(f["ox"][:, i]), np.nan_to_num(f["oz"][:, i])
        if widths is not None and widths[i]:
            back = half_depth(np.nan_to_num(f["men"][:, i]), widths[i], spacing)
            br = np.radians(np.nan_to_num(f["b"][:, i]))
            ox, oz = ox - back * np.sin(br), oz - back * np.cos(br)
        fighting = f["m"][:, i] & ~attack
        kind[:, s] = np.where(attack, O.ATTACK, np.where(fighting, O.HOLD, O.MOVE))
        target[:, s] = np.where(attack, mapped, -1)
        x[:, s], z[:, s] = ox, oz
        running = f["f"][:, i]
        run[:, s] = running | np.roll(running, -1) | np.roll(running, -2)
    return {"kind": kind, "x": x, "z": z, "target": target, "run": run}


class Replay:
    """A policy that replays recorded orders per battle of the batch (open-loop); grace_s after a
    battle's recording ends, `after` (default: attack the nearest enemy)."""

    def __init__(self, rows, device="cpu", after=nearest_attack, grace_s=120.0):
        T = max(r["kind"].shape[0] for r in rows)
        N = rows[0]["kind"].shape[1]
        B = len(rows)
        self.length = torch.tensor([r["kind"].shape[0] for r in rows], device=device)
        stack = {}
        for k in ("kind", "x", "z", "target", "run"):
            fill = {"kind": O.HOLD, "target": -1}.get(k, 0)
            arr = np.full((B, T, N), fill, dtype=rows[0][k].dtype)
            for b, r in enumerate(rows):
                arr[b, :r[k].shape[0]] = r[k]
            stack[k] = torch.as_tensor(arr, device=device)
        self.stack = stack
        self.after = after
        self.grace = grace_s

    def __call__(self, st):
        sec = st.t.floor().long()
        idx = torch.minimum(sec, self.length - 1).clamp(min=0)
        gather = lambda a: a[torch.arange(st.B, device=st.device), idx]
        o = O.Orders(kind=gather(self.stack["kind"]), x=gather(self.stack["x"]), z=gather(self.stack["z"]),
                     target=gather(self.stack["target"]), run=gather(self.stack["run"]))
        over = (sec >= self.length + self.grace)[:, None].expand(-1, st.N)
        if bool(over.any()):
            o = O.merge(o, self.after(st), over)
        return o
