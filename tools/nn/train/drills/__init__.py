"""Drills: curriculum battles that teach one skill each (docs/en/training/training.md "Drills").

A drill is a FRAME (a condition), not a fixed scene: every battle inside it is generated - random
rosters and unit types that satisfy the condition, sizes, distances, the place and bearing of the
whole battle on the map, which side we play. A drill module (tools/nn/train/drills/<name>.py) gives
a DRILL = Drill(...):

    frame(rng) -> army description     one generated battle with OUR side = side 1 (the format of
                                       tools/nn/sim/scenario.py: {"attacker", "sides": {1, 2}})
    enemy(st) -> Orders                the drill's enemy script (orders for every unit of every
                                       battle; the side the enemy plays takes its own)
    naive(st), skilled(st) -> Orders   our side's two check scripts: without the skill the drill
                                       must be lost clearly, with it won clearly (verify.py)

A drill battle runs under the standard battle limit (the simulator's battle_limit_s, as every training
battle: the attacker loses at it): the naive play must lose the fight itself, not the clock (the
network sees no time limit).

The drill's enemy is a training opponent (tools/nn/train/league.py "drill_<name>"): the battles
of its rows come from the drill's bank (Source below). Drills only change which battles the
network plays; only a drill given to the teacher (drills/teach.py, run.py --drill-teach) also adds an
imitation term of its `skilled` script (annealed, or adaptive: tools/nn/train/teach_auto.py).
"""
import importlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from tools.nn.sim import orders as O

NAMES = ("pincer", "kiting", "counter", "hold_fire", "defend")   # the drills, in the order they were built
READY = ("kiting", "counter", "hold_fire")   # the drills that passed verify.py (naive loses, skilled wins): run.py --drills default
PREFIX = "drill_"
MAP_HALF_M = 700.0       # a generated battle stays within this of the map's centre (the network's map frame,
#                          tools/nn/model/sources.py CROSSROADS, is -768..768 x -800..736)
DRILL_SEEDS = range(0, 1_000_000_000)           # training (as tools/nn/armies/generate.py)
DRILL_EVAL_SEEDS = range(1_000_000_000, 1_001_000_000)


@dataclass(frozen=True)
class Drill:
    name: str
    frame: Callable                 # rng -> army description (our side = 1)
    enemy: Callable                 # State -> Orders
    naive: Callable                 # State -> Orders
    skilled: Callable               # State -> Orders
    about: str = ""
    roles: Callable = None          # optional: State -> (correct, bad) [B, N, N] targets (drills/metrics.py)


def opponent(name):
    """The league's opponent name of a drill."""
    return PREFIX + name


def load(names=None):
    """{name: Drill} of the drills whose modules exist (all of NAMES by default)."""
    out = {}
    for n in names or NAMES:
        try:
            mod = importlib.import_module(f"tools.nn.train.drills.{n}")
        except ModuleNotFoundError as e:
            if e.name == f"tools.nn.train.drills.{n}":
                continue
            raise
        out[n] = mod.DRILL
    return out


def enemy_scripts(drills=None):
    """{league opponent name: the drill's enemy script}."""
    return {opponent(n): d.enemy for n, d in (drills or load()).items()}


# --- making a battle -------------------------------------------------------------------------------

def unit(key, x, z, b, width=None, general=False, men=None):
    """A unit of an army description (x, z: the formation's centre; b: bearing, degrees)."""
    out = {"key": key, "x": float(x), "z": float(z), "b": float(b) % 360.0, "general": bool(general)}
    if width:
        out["width"] = float(width)
    if men:
        out["men"] = float(men)
    return out


def army(ours, enemy, attacker, ours_faction=None, enemy_faction=None, **meta):
    """An army description: our units (side 1), the enemy's (side 2), who attacks (1 ours, 2 the enemy)."""
    return {"attacker": int(attacker),
            "sides": {1: {"faction": ours_faction, "units": list(ours)},
                      2: {"faction": enemy_faction, "units": list(enemy)}}, **meta}


def local(fx, lat, forward_deg=0.0):
    """(x, z) of a point `fx` metres forward and `lat` to the right, in a frame facing forward_deg."""
    br = math.radians(forward_deg)
    return fx * math.sin(br) + lat * math.cos(br), fx * math.cos(br) - lat * math.sin(br)


def transform(desc, rng, half=MAP_HALF_M):
    """The whole battle turned by a random angle about its centre and moved to a random place on the
    map, kept within `half` of the centre (in place; returns it)."""
    units = [u for s in (1, 2) for u in desc["sides"][s]["units"]]
    cx = float(np.mean([u["x"] for u in units]))
    cz = float(np.mean([u["z"] for u in units]))
    ang = float(rng.uniform(0, 360))
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for u in units:
        dx, dz = u["x"] - cx, u["z"] - cz
        # a turn by `ang` clockwise (bearings grow clockwise: b 90 = east = +x)
        pts.append((dx * ca + dz * sa, -dx * sa + dz * ca))
    ext_x = max(abs(p[0]) for p in pts) + 40.0
    ext_z = max(abs(p[1]) for p in pts) + 40.0
    ox = float(rng.uniform(-max(0.0, half - ext_x), max(0.0, half - ext_x)))
    oz = float(rng.uniform(-max(0.0, half - ext_z), max(0.0, half - ext_z)))
    for u, (px, pz) in zip(units, pts):
        u["x"], u["z"] = px + ox, pz + oz
        u["b"] = (u["b"] + ang) % 360.0
    return desc


def mirror(desc):
    """The same battle with the sides swapped (ours becomes side 2)."""
    out = dict(desc)
    out["sides"] = {1: desc["sides"][2], 2: desc["sides"][1]}
    out["attacker"] = 3 - int(desc["attacker"])
    return out


def battles(drill, seeds, both_sides=True):
    """[(army description, our side)] of a drill, one per seed: our side alternates 1, 2 (both_sides)."""
    out = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        desc = transform(drill.frame(rng), rng)
        ours = 2 if both_sides and i % 2 else 1
        out.append((mirror(desc) if ours == 2 else desc, ours))
    return out


# --- what scripts see -------------------------------------------------------------------------------

class View:
    """Pairwise geometry of a state for the scripts: d [B, i, j] distances, foe [B, i, j] (j a
    standing enemy of i), standing, missile, melee [B, N]."""

    def __init__(self, st):
        u = st.u
        self.u = u
        self.side = u["side"]
        self.alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
        self.standing = self.alive & ~u["r"]
        self.missile = (u["range"] > 0) & (u["a"] > 0)
        self.melee = self.standing & ~self.missile
        dx = u["x"][:, None, :] - u["x"][:, :, None]
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        self.dx, self.dz = dx, dz
        self.d = torch.sqrt(dx * dx + dz * dz)
        self.foe = (self.side[:, :, None] != self.side[:, None, :]) & (self.side[:, :, None] > 0) & self.standing[:, None, :]
        self.friend = (self.side[:, :, None] == self.side[:, None, :]) & (self.side[:, :, None] > 0)

    def nearest(self, mask=None):
        """(index [B, N], distance [B, N]) of the nearest standing enemy of each unit (in `mask` [B, j] or
        [B, i, j] when given); distance 1e9 when none."""
        ok = self.foe if mask is None else self.foe & (mask if mask.dim() == 3 else mask[:, None, :])
        d = torch.where(ok, self.d, torch.full_like(self.d, 1e9))
        v, i = d.min(2)
        return i, v

    def centroid(self, mask):
        """(x [B], z [B]) of the units in mask [B, N] (0 when none)."""
        w = mask.float()
        n = w.sum(1).clamp(min=1)
        return (self.u["x"] * w).sum(1) / n, (self.u["z"] * w).sum(1) / n

    def side_centroid(self, mask):
        """(x [B, N], z [B, N]): for each unit the centroid of its own side's units in mask, and
        (ex, ez) of the enemy side's, per unit."""
        out = []
        for own in (True, False):
            xs, zs = torch.zeros_like(self.u["x"]), torch.zeros_like(self.u["z"])
            for s in (1, 2):
                pick = mask & (self.side == (s if own else 3 - s))
                cx, cz = self.centroid(pick)
                xs = torch.where(self.side == s, cx[:, None], xs)
                zs = torch.where(self.side == s, cz[:, None], zs)
            out.append((xs, zs))
        return out


def attack(o, mask, target, run=True):
    """Orders o with ATTACK target [B, N] where mask (in place; returns o)."""
    o.kind = torch.where(mask, torch.full_like(o.kind, O.ATTACK), o.kind)
    o.target = torch.where(mask, target, o.target)
    o.run = torch.where(mask, torch.full_like(o.run, run), o.run)
    return o


def move(o, mask, x, z, run=True, kind=O.MOVE):
    """Orders o with MOVE (or WITHDRAW) to (x, z) [B, N] where mask (in place; returns o)."""
    o.kind = torch.where(mask, torch.full_like(o.kind, kind), o.kind)
    o.x = torch.where(mask, x, o.x)
    o.z = torch.where(mask, z, o.z)
    o.target = torch.where(mask, torch.full_like(o.target, -1), o.target)
    o.run = torch.where(mask, torch.full_like(o.run, run), o.run)
    return o


def hold(st):
    """Every unit holds: stands, shoots at will, fights back when attacked."""
    return O.hold(st.B, st.N, st.device)


def nearest(st):
    """Every standing unit attacks the nearest standing enemy, running (missile units close to range
    and shoot) - the league's `nearest`."""
    v = View(st)
    i, d = v.nearest()
    o = O.hold(st.B, st.N, st.device)
    return attack(o, v.standing & (d < 1e9), i)


def merged(st, ours_side, ours, enemy):
    """Orders [B, N]: `ours` for the units of ours_side [B], `enemy` for the rest."""
    return O.merge(enemy, ours, st.u["side"] == ours_side[:, None])
