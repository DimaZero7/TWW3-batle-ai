"""Orders to units: the simulator's action layout (docs/en/training/simulator.md).

Each decision step every unit of the batch gets one order, as six tensors [B, N]:

    kind     int64  HOLD, MOVE, ATTACK, WITHDRAW or KEEP (below)
    x, z     float  MOVE / WITHDRAW: the point to go to, m (same frame as the state's x, z)
    target   int64  ATTACK: the enemy's slot (the state's layout), -1 none
    run      bool   run (True) or walk (False)
    ability  int64  the unit's ability slot to use now, -1 none: 0..SLOTS-1 a lord ability (tools/nn/sim/abilities.py),
                    SPELL_AT + k its spell k (tools/nn/sim/magic.py: a cast at cast_target) - the slots of the
                    network's ability input (tools/nn/model/abilities.py layout: ABILITY_SLOTS actives, the spells
                    last). Optional: Orders made without it get -1 everywhere (the five-field contract
                    still works). Independent of kind: a lord may move and use an ability at once.
    cast_target int64  a spell's target: the unit (a slot of the state's layout) whose centre the bombardment falls
                    on (magic.request), -1 none. Optional (-1 everywhere).
    face     float  HOLD / MOVE / WITHDRAW: the facing to stand at, degrees of the world's bearing 0..360 (as the
                    state's b), < 0 none (the default: Orders made without it face nowhere - the old behaviour bit
                    for bit). The simulator snaps it to the middle of one of the game's 64 sectors (snap_face) and
                    turns a standing unit to it as the game does on goto_location_angle_width (battle.py: the
                    formation re-forms, turn.face_delay_s, then turns); ignored under ATTACK and in melee.

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
from tools.nn.sim.magic import SLOTS as SPELLS

try:
    import torch
except ImportError:
    torch = None

HOLD, MOVE, ATTACK, WITHDRAW, KEEP = 0, 1, 2, 3, 4
# The order's ability slots (the network's active slots, tools/nn/model/abilities.py ACTIVE): the lord abilities from 0,
# the spells in the last SPELLS (SPELL_AT + k: spell k)
ABILITY_SLOTS = 15
SPELL_AT = ABILITY_SLOTS - SPELLS
KINDS = ("hold", "move", "attack", "withdraw", "keep")
FIELDS = {
    "kind": ("i", "HOLD 0, MOVE 1, ATTACK 2, WITHDRAW 3, KEEP 4"),
    "x": ("f", "MOVE/WITHDRAW point x, m"),
    "z": ("f", "MOVE/WITHDRAW point z, m"),
    "target": ("i", "ATTACK: enemy slot, -1 none"),
    "run": ("b", "run instead of walk"),
    "ability": ("i", "ability slot to use now, -1 none (SPELL_AT + k: spell k)"),
    "cast_target": ("i", "a spell's target slot, -1 none"),
    "face": ("f", "the facing to stand at, degrees 0..360, < 0 none (snapped to the game's 64 sectors)"),
}
# The game keeps a unit's facing in 64 sectors of 5.625 deg and stands at the sector's middle (measured 28.09.2026:
# src/apps/orders/facing.lua engine / command).
FACE_SECTORS = 64
FACE_SECTOR_DEG = 360.0 / FACE_SECTORS


@dataclass
class Orders:
    kind: "torch.Tensor"
    x: "torch.Tensor"
    z: "torch.Tensor"
    target: "torch.Tensor"
    run: "torch.Tensor"
    ability: "torch.Tensor" = None
    cast_target: "torch.Tensor" = None
    face: "torch.Tensor" = None

    def __post_init__(self):
        if self.ability is None:
            self.ability = torch.full_like(self.kind, -1)
        if self.cast_target is None:
            self.cast_target = torch.full_like(self.kind, -1)
        if self.face is None:
            self.face = torch.full_like(self.x, -1.0)

    def spell(self):
        """[B, N] the spell slot ordered (0..SPELLS-1), -1 none."""
        k = self.ability - SPELL_AT
        return torch.where((k >= 0) & (k < SPELLS), k, torch.full_like(k, -1))

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
                  ability=torch.full((B, N), -1, dtype=torch.int64, device=device),
                  cast_target=torch.full((B, N), -1, dtype=torch.int64, device=device),
                  face=torch.full((B, N), -1.0, device=device))


def snap_face(face):
    """[B, N] the facing the game stands at for an ordered one (degrees): the middle of its 64-sector
    (facing.lua engine: 0..5.62 -> 2.81, 90..95.62 -> 92.81); < 0 (none) stays -1."""
    k = torch.floor(torch.remainder(face, 360.0) / FACE_SECTOR_DEG)
    mid = torch.remainder((k + 0.5) * FACE_SECTOR_DEG, 360.0)
    return torch.where(face >= 0, mid, torch.full_like(face, -1.0))


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
    a = orders.ability
    if ((a < -1) | ((a >= SLOTS) & (a < SPELL_AT)) | (a >= ABILITY_SLOTS)).any():
        raise ValueError("order ability slot out of range")
    if ((orders.cast_target < -1) | (orders.cast_target >= N)).any():
        raise ValueError("order cast target out of range")
    if not torch.isfinite(orders.face).all() or (orders.face >= 360).any():
        raise ValueError("order facing not finite or >= 360 (degrees 0..360, < 0 none)")
