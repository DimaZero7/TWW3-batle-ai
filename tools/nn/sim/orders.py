"""Orders to units: the simulator's action layout (docs/en/training/simulator.md).

Each decision step every unit of the batch gets one order, as six tensors [B, N]:

    kind     int64  HOLD, MOVE, ATTACK, WITHDRAW or KEEP (below)
    x, z     float  MOVE / WITHDRAW: the point to go to, m (same frame as the state's x, z)
    target   int64  ATTACK: the enemy's slot (the state's layout), -1 none
    run      bool   run (True) or walk (False)
    ability  int64  the unit's ability slot to use now (0..SLOTS-1, tools/nn/sim/abilities.py), -1 none.
                    Optional: Orders made without it get -1 everywhere (the five-field contract
                    still works). Independent of kind: a lord may move and use an ability at once.

    HOLD      stay; shoot at will at enemies in range; fight back when attacked.
    MOVE      go to (x, z) and stop there; missile units do not shoot on the move. A unit in
              melee with the point contact.leave_m (10 m) or more away breaks off (as WITHDRAW:
              it strikes nobody, the enemies in contact strike it); a nearer point it fights on.
    ATTACK    melee units run (or walk) at the target and fight it; missile units close to
              their range and shoot it. Pursues a routing target.
    WITHDRAW  break off melee and go to (x, z); while leaving, the enemies in contact strike
              its rear.
    KEEP      no new order: the order in force goes on unchanged (x, z, target, run are
              ignored). A unit that never had an order holds. Lets a network leave a unit alone
              instead of re-issuing (and jittering) its order every decision.

An order is given every decision step and stays in force until the next one that is not KEEP.
Orders to empty slots and to routing or shattered units are ignored (routing units flee on their
own). An ability order is not kept: it fires the ability once if it is ready (self-cast, not
active, recharged), else nothing happens.
A network that sees its own units first (state.own_first) turns its target index back with
state.slot_from_own_first.
"""
from dataclasses import dataclass

from tools.nn.sim.abilities import SLOTS

try:
    import torch
except ImportError:
    torch = None

HOLD, MOVE, ATTACK, WITHDRAW, KEEP = 0, 1, 2, 3, 4
KINDS = ("hold", "move", "attack", "withdraw", "keep")
FIELDS = {
    "kind": ("i", "HOLD 0, MOVE 1, ATTACK 2, WITHDRAW 3, KEEP 4"),
    "x": ("f", "MOVE/WITHDRAW point x, m"),
    "z": ("f", "MOVE/WITHDRAW point z, m"),
    "target": ("i", "ATTACK: enemy slot, -1 none"),
    "run": ("b", "run instead of walk"),
    "ability": ("i", "ability slot to use now, -1 none"),
}


@dataclass
class Orders:
    kind: "torch.Tensor"
    x: "torch.Tensor"
    z: "torch.Tensor"
    target: "torch.Tensor"
    run: "torch.Tensor"
    ability: "torch.Tensor" = None

    def __post_init__(self):
        if self.ability is None:
            self.ability = torch.full_like(self.kind, -1)

    def to(self, device):
        return Orders(*(getattr(self, k).to(device) for k in FIELDS))

    def clone(self):
        return Orders(*(getattr(self, k).clone() for k in FIELDS))


def hold(B, N, device="cpu"):
    """Every unit holds."""
    return Orders(kind=torch.full((B, N), HOLD, dtype=torch.int64, device=device),
                  x=torch.zeros((B, N), device=device), z=torch.zeros((B, N), device=device),
                  target=torch.full((B, N), -1, dtype=torch.int64, device=device),
                  run=torch.zeros((B, N), dtype=torch.bool, device=device),
                  ability=torch.full((B, N), -1, dtype=torch.int64, device=device))


def merge(first, second, use_second):
    """Orders from `second` where use_second [B, N] is true, else from `first` (e.g. two sides'
    policies into one batch)."""
    return Orders(*(torch.where(use_second, getattr(second, k), getattr(first, k)) for k in FIELDS))


def check(orders, N):
    """Raise ValueError when an order is malformed."""
    kind, target = orders.kind, orders.target
    if ((kind < HOLD) | (kind > KEEP)).any():
        raise ValueError("order kind out of range")
    if ((target < -1) | (target >= N)).any():
        raise ValueError("order target out of range")
    if ((kind == ATTACK) & (target < 0)).any():
        raise ValueError("ATTACK without a target")
    if not (torch.isfinite(orders.x).all() and torch.isfinite(orders.z).all()):
        raise ValueError("order point not finite")
    if ((orders.ability < -1) | (orders.ability >= SLOTS)).any():
        raise ValueError("order ability slot out of range")
