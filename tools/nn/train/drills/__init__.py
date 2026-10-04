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
    broad(rng) -> army description     optional: the same situation embedded in a messier battle (more
                                       unit types, uninvolved units on both sides, lords, other sizes);
                                       battles() draws it for a share BROAD of the seeds, frame() for the rest
    embedded(rng) -> army description  optional: the drill's situation INSIDE a normal battle - both sides the
                                       usual generated armies (tools/nn/armies: normal sizes, both lords, the
                                       arena's deployment, no turn or move on the map) with the drill's units
                                       inserted (marked by the unit's "tag": TAG_OURS / TAG_ENEMY, state STATIC
                                       "tag", which the network never sees) and the gold kept even (balance());
                                       battles() draws it for a share EMBED of the seeds (before the broad draw)
    teacher(st) -> Orders              optional: the script whose orders label our units for the teacher
                                       (drills/teach.py; default `skilled`): the skill alone, tag-blind
    moments(st, orders) -> [B, N]      optional: the units at the situation's moments, where the teacher's
                                       orders are labels in an embedded or a normal battle (the clean and broad
                                       frames label every unit); also the teacher in normal battles
                                       (run.py --teach-normal)
    transfer(st) -> (situation, applied, mistake) [B, N]
                                       optional: the drill's situation detected in an ORDINARY battle, per
                                       unit and step, and whether the unit applies the skill there / makes
                                       the drill's mistake (drills/transfer.py: measured in the normal
                                       evaluation battles, the network's units and the opponent script's)

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
READY = ("kiting", "counter", "hold_fire")   # the drills that passed verify.py (naive loses, skilled wins): evaluated by test5
# the drills trained by default (run.py --drill-weights): READY without counter - counter-picking as the drill
# defines it COSTS in normal battles (build/why, 04.10.2026: overriding the network with it at the detector's
# moments -0.09 gold trade, -12 pp win: it pulls units out of their fights); its code stays
TRAIN = ("kiting", "hold_fire")
PREFIX = "drill_"
MAP_HALF_M = 700.0       # a generated battle stays within this of the map's centre (the network's map frame,
#                          tools/nn/model/sources.py CROSSROADS, is -768..768 x -800..736)
BROAD_DEFAULT = 0.5      # (run.py --drill-broad's default; BROAD below is the share in force)
BROAD = BROAD_DEFAULT    # share of a drill's battles from its broad frame (when it has one; the rest: the
#                          clean frame); drills.BROAD is read by battles() at every call (run.py --drill-broad)
EMBED_DEFAULT = 1.0      # share of a drill's battles from its embedded frame (the drills that have one: kiting,
#                          hold_fire), drawn before the broad / clean draw (run.py --drill-embed; 0: the old mix)
EMBED = EMBED_DEFAULT    # the share in force (run.train / test5 set it from --drill-embed)
TAG_OURS, TAG_ENEMY = 1, 2                      # unit tags of an embedded frame (state STATIC "tag")
MAX_UNITS = 19           # a side's units besides the lord (tools/nn/armies/generate.py MAX_UNITS)
EMBED_TOLERANCE = 0.05   # an embedded battle's two sides' gold differ by at most this share (as the generator's)
EMBED_TRIES = 40         # draws of an embedded battle until its gold is even (else the most even one)
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
    broad: Callable = None          # optional: rng -> a messier army description of the same situation
    transfer: Callable = None       # optional: State -> (situation, applied, mistake) [B, N] (drills/transfer.py)
    embedded: Callable = None       # optional: rng -> the situation inside a normal battle (tagged units)
    teacher: Callable = None        # optional: State -> Orders the teacher labels with (default skilled)
    moments: Callable = None        # optional: (State, Orders) -> [B, N] where the labels count outside clean frames


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

def unit(key, x, z, b, width=None, general=False, men=None, tag=0):
    """A unit of an army description (x, z: the formation's centre; b: bearing, degrees; tag: an embedded
    frame's mark, TAG_OURS / TAG_ENEMY)."""
    out = {"key": key, "x": float(x), "z": float(z), "b": float(b) % 360.0, "general": bool(general)}
    if width:
        out["width"] = float(width)
    if men:
        out["men"] = float(men)
    if tag:
        out["tag"] = int(tag)
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


def is_broad(drill, seed, share=None):
    """Whether the drill's battle of this seed comes from its broad frame: a draw of its own (the clean
    battles of a seed stay what they were before broad frames existed), below `share` (default BROAD)."""
    share = BROAD if share is None else share
    return drill.broad is not None and share > 0 and np.random.default_rng([int(seed), 7]).random() < share


def is_embedded(drill, seed, share=None):
    """Whether the drill's battle of this seed comes from its embedded frame: a draw of its own, below
    `share` (default EMBED); drawn before is_broad (EMBED 0: every battle as before embedded frames)."""
    share = EMBED if share is None else share
    return drill.embedded is not None and share > 0 and np.random.default_rng([int(seed), 11]).random() < share


FRAMES = ("clean", "broad", "embedded")


def frame_of(drill, seed, broad=None, embed=None):
    """"embedded", "broad" or "clean": the frame of the drill's battle of this seed."""
    if is_embedded(drill, seed, embed):
        return "embedded"
    return "broad" if is_broad(drill, seed, broad) else "clean"


def battles(drill, seeds, both_sides=True, broad=None, embed=None):
    """[(army description, our side)] of a drill, one per seed: our side alternates 1, 2 (both_sides);
    a share `embed` (default EMBED) from the drill's embedded frame (as generated: the arena's place), of the
    rest a share `broad` (default BROAD) from its broad frame, else the clean one (both turned and moved on
    the map: transform). The description's "frame" says which ("broad": True for the broad one). An embedded
    battle with our side 2 is also turned half round the map's centre (half_turn): our army deploys where a
    normal battle's side 2 does."""
    out = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        kind = frame_of(drill, s, broad, embed)
        if kind == "embedded":
            desc = drill.embedded(rng)
        else:
            desc = transform((drill.broad if kind == "broad" else drill.frame)(rng), rng)
        desc["broad"] = kind == "broad"
        desc["frame"] = kind
        ours = 2 if both_sides and i % 2 else 1
        if ours == 2 and kind == "embedded":
            half_turn(desc)
        out.append((mirror(desc) if ours == 2 else desc, ours))
    return out


def half_turn(desc):
    """The battle turned by 180 degrees about the map's centre (in place; returns it): the arena's two
    deployment zones are symmetric about it, so side 1's army stands where side 2's would."""
    for s in (1, 2):
        for u in desc["sides"][s]["units"]:
            u["x"], u["z"], u["b"] = -u["x"], -u["z"], (u["b"] + 180.0) % 360.0
    return desc


# --- embedded frames: the situation inside a normal battle -------------------------------------------

def generated(rng, ours, enemy, swap=None):
    """An army description of a normal generated battle (tools/nn/armies/generate.py: budget, templates,
    deployment, both lords) with our side (1) of faction `ours` against `enemy`, attacking or defending
    (one half each) - the battles scenes.Generated plays, laid out by scenario.from_arena. swap: {1: [keys],
    2: [keys]} the units the drill will insert per side: for each, the side gives up the unit of nearest cost
    (not its lord) and is deployed again without them (tools/nn/armies/place.py) - with the inserted units the
    battle has a normal battle's unit count and gold, and no hole in its lines."""
    from tools.nn.armies import generate
    from tools.nn.armies import place as placement
    from tools.nn.armies import pools as P
    from tools.nn.sim import scenario
    arena = generate.default().generate(rng, sides=(ours, enemy), name="embedded")
    pools = P.load()
    for s, tag in ((1, "own"), (2, "enemy")):
        keys = (swap or {}).get(s) or []
        if not keys:
            continue
        side = arena["sides"][tag]
        pool = pools[side["faction"]]
        by_key = {u.key: u for u in pool.units}
        units = [by_key[u["key"]] for u in side["units"] if not u.get("general")]
        for key in keys:
            if not units:
                break
            c = cost(key)
            units.remove(min(units, key=lambda u: abs(u.cost - c)))
        side["units"] = placement.place(pool.lord, units, arena["deployment_m"])
        side["cost"] = pool.lord.cost + sum(u.cost for u in units)
    desc = scenario.from_arena(arena, "attack" if rng.random() < 0.5 else "defend")
    for s in (1, 2):
        for u in desc["sides"][s]["units"]:
            u.pop("name", None)
    desc["sides"][1]["ai"], desc["sides"][2]["ai"] = False, True
    return desc


@dataclass(frozen=True)
class Axes:
    """A side's frame on the map: origin (its units' centroid) and forward (towards the enemy's centroid);
    lateral is to the right of forward (bearings grow clockwise, as local())."""
    ox: float
    oz: float
    fx: float
    fz: float

    def local(self, x, z):
        dx, dz = x - self.ox, z - self.oz
        return dx * self.fx + dz * self.fz, dx * self.fz - dz * self.fx

    def world(self, f, lat):
        return self.ox + f * self.fx + lat * self.fz, self.oz + f * self.fz - lat * self.fx


def axes(desc, side=1):
    """The Axes of a side of an army description."""
    pts = {s: [(u["x"], u["z"]) for u in desc["sides"][s]["units"]] for s in (1, 2)}
    ox, oz = (float(np.mean([p[k] for p in pts[side]])) for k in (0, 1))
    ex, ez = (float(np.mean([p[k] for p in pts[3 - side]])) for k in (0, 1))
    n = max(1e-6, math.hypot(ex - ox, ez - oz))
    return Axes(ox, oz, (ex - ox) / n, (ez - oz) / n)


def width_of(u):
    """A unit description's frontage, m (a lord 5, a unit without one 30)."""
    return float(u.get("width") or (5.0 if u.get("general") else 30.0))


def extent(units, ax, s):
    """How far the units reach to the side s (+1 right, -1 left of ax's forward), m."""
    return max(s * ax.local(u["x"], u["z"])[1] + width_of(u) / 2 for u in units)


def front(units, ax, enemy=False):
    """The forward coordinate of the units' front line in ax (the most forward non-lord unit; enemy: the
    nearest to us, the least forward)."""
    fs = [ax.local(u["x"], u["z"])[0] for u in units if not u.get("general")] or \
        [ax.local(u["x"], u["z"])[0] for u in units]
    return min(fs) if enemy else max(fs)


def shift(units, ax, lat):
    """The units moved by `lat` metres along ax's lateral (in place)."""
    for u in units:
        f, l = ax.local(u["x"], u["z"])
        u["x"], u["z"] = ax.world(f, l + lat)


_COST = {}


def cost(key):
    """A unit's multiplayer cost (config/nn/pools.json pools, lords included)."""
    if not _COST:
        from tools.nn.armies import pools as P
        for pool in P.load().values():
            for u in (pool.lord,) + tuple(pool.units):
                _COST[u.key] = u.cost
    return _COST[key]


def gold(desc):
    """{side: the gold of its units}."""
    return {s: sum(cost(u["key"]) for u in desc["sides"][s]["units"]) for s in (1, 2)}


def balance(desc, tolerance=EMBED_TOLERANCE):
    """Even the two sides' gold (in place): while the richer side has more than `tolerance` of its gold over
    the other's, it drops the untagged non-lord unit that brings the difference nearest zero (only while
    that helps) -> whether it is within the tolerance now."""
    while True:
        g = gold(desc)
        rich = 1 if g[1] >= g[2] else 2
        diff = g[rich] - g[3 - rich]
        if diff <= tolerance * g[rich] + 1e-9:
            return True
        units = desc["sides"][rich]["units"]
        cand = [u for u in units if not u.get("general") and not u.get("tag")]
        if not cand:
            return False
        best = min(cand, key=lambda u: abs(diff - cost(u["key"])))
        if abs(diff - cost(best["key"])) >= diff:
            return False
        units.remove(best)


def imbalance(desc):
    """The gold difference of the two sides / the richer side's gold."""
    g = gold(desc)
    return abs(g[1] - g[2]) / max(g[1], g[2], 1)


NORMAL = "@normal"        # the teacher in normal battles: "<drill>@normal" (rollout.Battles teach_normal)


def normal_name(name):
    """The name of a drill's teacher in normal battles (its share, its log)."""
    return name + NORMAL


def embedded_rows(st):
    """[B] the battles of an embedded frame (any tagged unit)."""
    return (st.u["tag"] > 0).any(1)


def tagged(st, tag):
    """[B, N] the units with this tag."""
    return st.u["tag"] == tag


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
