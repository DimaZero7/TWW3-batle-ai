"""Orders to the game (METHOD.md 'Уровень': the code re-issues an order only when its point has moved past a threshold):
a unit's intended order goes to the game only when it differs from the one last sent there; otherwise the last one is
re-given as it was (the simulator takes an identical order as no new one: battle.py new_order - the same kind, target,
a point within 1 m, the same facing; the leash still sees the attack in force). The same gate in the drills, the
exams, the general battles and the bridge, so 'orders a minute' counts what the game would get, not the decisions.

A new order (against the last one sent): another kind (hold / move / attack / withdraw); another attack target; a move
or withdraw point MOVE_M or more away; a facing given FACE_DEG or more off the one sent (or one given where none was);
a width given WIDTH_M or more off (or one given where none was). A hold stays a hold wherever the unit stands; a facing
or width taken away is no new order (the game keeps the one in force). KEEP sends nothing.

    Gate(B, N, device)        per slot: the last order sent, the counts
    gate.apply(intended, live) -> the orders to give (intended where new, else the last sent); counts new ones
    gate.per_minute(mask)     [B] new orders a unit a minute of its live seconds (mask [B, N]: whose)

The thresholds: MOVE_M - game/vision's 'a new order' (events.ORDER_M, the person's ~1 a minute); FACE_DEG - four of the
game's 64 facing sectors (net/unit/train.FACE_CHANGE_DEG); WIDTH_M - one file. Not yet from a probe: the probe 'a turn
when the course changes' (tmp/critique/method_learn.md V1) will set them.
"""
import torch

from sim import orders as O

MOVE_M = 8.0
FACE_DEG = 22.5
WIDTH_M = 1.0


def _angle(a, b):
    return (torch.remainder(a - b + 180.0, 360.0) - 180.0).abs()


def differs(new, old, move_m=MOVE_M, face_deg=FACE_DEG, width_m=WIDTH_M):
    """[B, N] bool: the order new is another order to the game than old (module doc); new KEEP: False; old KEEP (none
    sent yet): any order is new."""
    k, k0 = new.kind, old.kind
    point = (k == O.MOVE) | (k == O.WITHDRAW)
    moved = point & ((new.x - old.x) ** 2 + (new.z - old.z) ** 2 >= move_m ** 2)
    target = (k == O.ATTACK) & (new.target != old.target)
    f, f0 = new.face, old.face
    turned = (k != O.ATTACK) & (f >= 0) & ((f0 < 0) | (_angle(f, f0) >= face_deg))
    w, w0 = new.width, old.width
    wide = (k != O.ATTACK) & (w >= 0) & ((w0 < 0) | ((w - w0).abs() >= width_m))
    return (k != O.KEEP) & ((k0 == O.KEEP) | (k != k0) | target | moved | turned | wide)


class Gate:
    def __init__(self, B, N, device, move_m=MOVE_M, face_deg=FACE_DEG, width_m=WIDTH_M):
        self.move_m, self.face_deg, self.width_m = move_m, face_deg, width_m
        self.last = O.hold(B, N, device)
        self.last.kind[:] = O.KEEP
        self.prev = self.last.clone()
        z = lambda: torch.zeros((B, N), device=device)          # noqa: E731
        self.sent, self.raw, self.live_s = z(), z(), z()

    def apply(self, intended, live, seconds=1.0):
        """The orders to give for intended [B, N] (a decision's): new ones as they are, the rest the last sent (KEEP
        stays KEEP); live [B, N] the units counted (alive, standing, ours) for seconds."""
        th = (self.move_m, self.face_deg, self.width_m)
        new = differs(intended, self.last, *th)
        self.raw += (differs(intended, self.prev, *th) & live).float()
        self.prev = intended.clone()
        keep = intended.kind == O.KEEP
        out = O.Orders(*(torch.where(new | keep, getattr(intended, f), getattr(self.last, f)) for f in O.FIELDS))
        self.last = O.Orders(*(torch.where(new, getattr(intended, f), getattr(self.last, f)) for f in O.FIELDS))
        self.sent += (new & live).float()
        self.live_s += live.float() * seconds
        return out

    def per_minute(self, mask=None, raw=False):
        """[B] new orders (raw: changes of the intended order, no gate) a unit a minute of its live seconds."""
        c = self.raw if raw else self.sent
        m = torch.ones_like(c, dtype=torch.bool) if mask is None else mask
        return 60.0 * (c * m).sum(1) / (self.live_s * m).sum(1).clamp(min=1e-6)
