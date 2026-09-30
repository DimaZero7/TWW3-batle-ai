"""What one side sees: the network's input (docs/en/training/model.md).

The rule: the AI sees only what a human player sees.
* Before battle (static): both rosters (passports, experience rank), lord levels, who attacks,
  the map's bounds, the side's faction character and role.
* In battle: own units — everything, exact morale too. Enemy units — only while visible:
  position, facing, movement, men and health, what they do (melee, moving, running, firing) and the
  morale STATE (steady / wavering / routing / shattered), never the exact morale. Not visible:
  the last seen position and how long ago.
Everything is in the side's frame (tools/nn/model/frame.py), scaled to about -1..1.

State: a dict of arrays [B, N] with the names of recorded `nn_sample` (tools/nn/gamedata.py):
x, z, b, men, hp, mp, ms, m, mv, f, fire, a, k, ox, oz, lf, rf, bf, target, plus `t` [B] (s).
Optional: `vis` [B, N] (the unit is visible to the other side; missing = all visible),
`fat` [B, N] (fatigue state 0-5; missing = unknown). Units of both sides in one row; `setup.side`
says whose each is. Works on numpy arrays and torch tensors: the same code for recorded
battles and the batched simulator.
"""
from dataclasses import dataclass

import numpy as np

from tools.nn.model import factions, passport
from tools.nn.model.frame import Frame, army_axis, edge_distances, xp

POS = 500.0      # m: positions and order points
VEL = 5.0        # m/s
EDGE = 1000.0    # m: distance to the map edge (capped)
AGE = 60.0       # s: age of the last sighting (capped at 1)
KILLS = 200.0
TIME = 3600.0    # s: the battle limit, 60 minutes
MORALE = 2.0     # MoralePercent (-2..2 seen) / 2, clipped to -1.5..1.5
FATIGUE = 6      # fresh, active, winded, tired, very tired, exhausted

# The token's dynamic features: (name, who sees it). "both": own units and visible enemies;
# "own": own units only (zero for enemies; the critic's full view fills them for all units).
DYNAMIC = (
    ("fwd", "both"), ("lat", "both"), ("face_cos", "both"), ("face_sin", "both"),
    ("vel_fwd", "both"), ("vel_lat", "both"), ("men", "both"), ("hp", "both"),
    ("state_steady", "both"), ("state_wavering", "both"), ("state_routing", "both"), ("state_shattered", "both"),
    ("melee", "both"), ("moving", "both"), ("running", "both"), ("firing", "both"),
    ("edge_fwd", "both"), ("edge_back", "both"), ("edge_right", "both"), ("edge_left", "both"),
    ("morale", "own"), *((f"ms_{i}", "own") for i in range(1, 8)),
    ("ammo", "own"), ("kills", "own"), *((f"fatigue_{i}", "own") for i in range(FATIGUE)), ("fatigue_known", "own"),
    ("order_fwd", "own"), ("order_lat", "own"), ("has_order", "own"), ("has_target", "own"),
    ("threat_left", "own"), ("threat_right", "own"), ("threat_rear", "own"),
)
FLAGS = ("is_own", "visible", "seen", "age", "rank")
NAMES = tuple(n for n, _ in DYNAMIC) + FLAGS + tuple(f"passport_{i}" for i in range(passport.SIZE))
INDEX = {n: i for i, n in enumerate(NAMES)}
TOKEN = len(NAMES)
OWN_ONLY = tuple(INDEX[n] for n, who in DYNAMIC if who == "own")
CONTEXT = factions.SIZE + 9   # character; attack, defend, time, 2 lord levels, map width and depth, 2 counts
CONTEXT_FULL = CONTEXT + factions.SIZE   # the critic's: + the enemy's character


@dataclass
class Setup:
    """What is known before battle, per battle of the batch (numpy)."""
    keys: list                 # [B][N] unit keys ("" = padding)
    side: np.ndarray           # [B, N] 1 or 2, 0 = padding
    bounds: np.ndarray         # [B, 4] xmin, xmax, zmin, zmax (m)
    factions: list             # [B] (faction of side 1, faction of side 2)
    attacker: np.ndarray       # [B] side that attacks (1 or 2)
    rank: np.ndarray = None    # [B, N] experience rank 0-9
    lord_level: np.ndarray = None   # [B, 2] per side

    def __post_init__(self):
        B, N = self.side.shape
        self.rank = np.zeros((B, N), np.float32) if self.rank is None else np.asarray(self.rank, np.float32)
        self.bounds = np.asarray(self.bounds, np.float32)
        self.attacker = np.asarray(self.attacker)
        self.lord_level = np.ones((B, 2), np.float32) if self.lord_level is None else np.asarray(self.lord_level,
                                                                                                    np.float32)
        self.passport = np.stack([passport.table(k) for k in self.keys])      # [B, N, P]
        self.men0 = np.stack([passport.men(k) for k in self.keys])            # [B, N]
        self.ammo0 = np.stack([passport.ammo(k) for k in self.keys])          # [B, N]
        self.present = self.side > 0
        self._cache = {}

    def like(self, a):
        """The setup's arrays in the backend (and device) of array a."""
        m = xp(a)
        if m is np:
            return self
        key = str(a.device)
        if key not in self._cache:
            t = lambda v: m.as_tensor(np.asarray(v), device=a.device)
            self._cache[key] = _Arrays(side=t(self.side), bounds=t(self.bounds.astype(np.float32)),
                                       rank=t(self.rank), lord_level=t(self.lord_level),
                                       passport=t(self.passport), men0=t(self.men0), ammo0=t(self.ammo0),
                                       present=t(self.present), attacker=t(self.attacker))
        return self._cache[key]

    def character(self, side):
        """[B, traits] of the given side (1 or 2)."""
        return np.stack([factions.character(f[side - 1]) for f in self.factions])

    def adapter(self, side):
        """[B] LoRA adapter index of (faction, role) for the side."""
        return np.array([factions.adapter(f[side - 1], a == side) for f, a in zip(self.factions, self.attacker)])


@dataclass
class _Arrays:
    side: object
    bounds: object
    rank: object
    lord_level: object
    passport: object
    men0: object
    ammo0: object
    present: object
    attacker: object


@dataclass
class Memory:
    """What a side remembers between decisions (world coordinates)."""
    frame: Frame
    last_x: object     # [B, N] last seen position
    last_z: object
    last_t: object     # [B, N] time of the last sighting (s)
    seen: object       # [B, N] seen at least once
    dead: object       # [B, N] seen destroyed
    prev_x: object     # previous decision: position, time, visible (for velocity)
    prev_z: object
    prev_t: object
    prev_vis: object


@dataclass
class Obs:
    tokens: object     # [B, N, TOKEN]
    ctx: object        # [B, CONTEXT]
    own: object        # [B, N] the side's units (present)
    attend: object     # [B, N] tokens attention may use (present, not known dead)
    ctrl: object       # [B, N] own units that take orders (alive, not routing or shattered)
    target_ok: object  # [B, N] enemies that may be attacked now (visible, alive)
    pos: object        # [B, N, 2] (forward, lateral) / POS: current or last seen
    adapter: object    # [B] LoRA adapter index
    frame: Frame
    side: int


def _f(m, a):
    a = m.nan_to_num(a * 1.0, nan=0.0)
    return a.float() if m is not np else a.astype(np.float32)


def _field(state, name, like):
    v = state.get(name)
    return like * 0 if v is None else v


def _visible(m, state, S, side):
    """[B, N] units the side sees now: its own, and enemies visible to it."""
    own = S.side == side
    vis = state.get("vis")
    enemy_vis = S.present if vis is None else (vis & S.present)
    return own | (enemy_vis & ~own)


def start(state, setup, side):
    """Memory at the start of battle: the side's frame from the armies' positions."""
    x = state["x"]
    m = xp(x)
    S = setup.like(x)
    xs, zs = m.nan_to_num(x * 1.0), m.nan_to_num(state["z"] * 1.0)
    own, enemy = (S.side == side) & S.present, (S.side != side) & S.present
    ux, uz = army_axis(xs, zs, own, enemy)
    b = S.bounds
    frame = Frame((b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2, ux, uz)
    zero, false = xs * 0, xs != xs
    return Memory(frame, zero, zero, zero, false, false, zero, zero, zero - 1, false)


def observe(state, setup, side, memory=None, full=False):
    """(Obs, new Memory) of `side` (1 or 2). full=True: the critic's view — every unit, every field."""
    x = state["x"]
    m = xp(x)
    S = setup.like(x)
    memory = memory or start(state, setup, side)
    fr = memory.frame
    B, N = x.shape
    t = state["t"] * 1.0
    tn = t.reshape(B, 1) + x * 0
    xs, zs = _f(m, x), _f(m, state["z"])
    men = _f(m, state["men"])
    alive = (men > 0) & S.present
    own = (S.side == side) & S.present
    enemy = (S.side != side) & S.present
    vis = _visible(m, state, S, side) | (S.present if full else own)
    sees = vis & S.present

    seen = memory.seen | sees
    dead = memory.dead | (sees & ~alive & enemy)
    last_x = m.where(sees, xs, memory.last_x)
    last_z = m.where(sees, zs, memory.last_z)
    last_t = m.where(sees, tn, memory.last_t)

    fwd, lat = fr.point(last_x, last_z)
    face_c, face_s = fr.facing(_f(m, state["b"]))
    dt = tn - memory.prev_t
    moved = sees & memory.prev_vis & (memory.prev_t >= 0) & (dt > 0)
    dts = m.where(moved, dt, dt * 0 + 1)
    vf, vl = fr.vector((xs - memory.prev_x) / dts, (zs - memory.prev_z) / dts)
    ms = _f(m, state["ms"])
    edges = edge_distances(last_x, last_z, fr, [S.bounds[:, i] for i in range(4)], EDGE)
    ofw, olt = fr.vector(_f(m, state["ox"]) - xs, _f(m, state["oz"]) - zs)
    has_order = (m.abs(ofw) + m.abs(olt)) > 1.0
    fat = state.get("fat")
    fat_known = xs * 0 if fat is None else _f(m, fat == fat)
    fat = xs * 0 - 1 if fat is None else m.nan_to_num(fat * 1.0, nan=-1.0)
    target = state.get("target")
    ammo0 = S.ammo0 + (S.ammo0 == 0) * 1.0

    cols = {
        "fwd": fwd / POS, "lat": lat / POS, "face_cos": face_c, "face_sin": face_s,
        "vel_fwd": m.where(moved, vf, vf * 0) / VEL, "vel_lat": m.where(moved, vl, vl * 0) / VEL,
        "men": men / (S.men0 + (S.men0 == 0) * 1.0), "hp": _f(m, state["hp"]),
        "state_steady": _f(m, (ms >= 1) & (ms <= 4)), "state_wavering": _f(m, ms == 5),
        "state_routing": _f(m, ms == 6), "state_shattered": _f(m, ms == 7),
        "melee": _f(m, state["m"]), "moving": _f(m, state["mv"]), "running": _f(m, state["f"]),
        "firing": _f(m, state["fire"]),
        "edge_fwd": edges[0] / EDGE, "edge_back": edges[1] / EDGE, "edge_right": edges[2] / EDGE,
        "edge_left": edges[3] / EDGE,
        "morale": m.clip(_f(m, state["mp"]) / MORALE, -1.5, 1.5),
        **{f"ms_{i}": _f(m, ms == i) for i in range(1, 8)},
        "ammo": _f(m, _field(state, "a", xs)) / ammo0, "kills": _f(m, _field(state, "k", xs)) / KILLS,
        **{f"fatigue_{i}": _f(m, fat == i) for i in range(FATIGUE)}, "fatigue_known": fat_known,
        "order_fwd": m.where(has_order, ofw, ofw * 0) / POS, "order_lat": m.where(has_order, olt, olt * 0) / POS,
        "has_order": _f(m, has_order), "has_target": xs * 0 if target is None else _f(m, target >= 0),
        "threat_left": _f(m, state["lf"]), "threat_right": _f(m, state["rf"]), "threat_rear": _f(m, state["bf"]),
    }
    # Who sees what: "both" fields only for units seen now; "own" fields only for own units
    # (every unit in the critic's full view). Position and edges stay for the last sighting.
    keep_last = ("fwd", "lat", "edge_fwd", "edge_back", "edge_right", "edge_left")
    see_all = S.present if full else own
    for name, who in DYNAMIC:
        allowed = seen if name in keep_last else (sees if who == "both" else see_all)
        cols[name] = m.where(allowed & S.present, cols[name], cols[name] * 0)
    age = m.clip((tn - last_t) / AGE, 0, 1)
    flags = {"is_own": _f(m, own), "visible": _f(m, sees), "seen": _f(m, seen),
             "age": m.where(seen & ~sees, age, age * 0), "rank": S.rank / passport.MAX_RANK}
    dyn = m.stack([cols[n] for n, _ in DYNAMIC] + [_f(m, flags[n]) for n in FLAGS], -1)
    tokens = _cat(m, [dyn, _f(m, S.passport)], -1) * _f(m, S.present)[..., None]

    attend = S.present & ~dead & ~(own & ~alive)
    ctrl = own & alive & (ms < 6)
    target_ok = enemy & sees & alive
    ctx = _context(m, setup, S, side, t, own & alive, enemy & ~dead & seen, xs)
    pos = m.stack([fwd, lat], -1) / POS
    pos = m.where(seen[..., None], pos, pos * 0)
    new = Memory(fr, last_x, last_z, last_t, seen, dead, m.where(sees, xs, xs * 0), m.where(sees, zs, zs * 0),
                 tn, sees)
    adapter = setup.adapter(side)
    adapter = adapter if m is np else m.as_tensor(adapter, device=x.device)
    if full:   # the critic also knows the enemy's character
        other = setup.character(3 - side)
        ctx = _cat(m, [ctx, _f(m, other if m is np else m.as_tensor(other, device=x.device))], -1)
    return Obs(_f(m, tokens), _f(m, ctx), own, attend, ctrl, target_ok, _f(m, pos), adapter, fr, side), new


def _cat(m, xs, dim):
    return np.concatenate(xs, dim) if m is np else m.cat(xs, dim)


def _context(m, setup, S, side, t, own_alive, enemy_known, like):
    char = setup.character(side)
    char = char if m is np else m.as_tensor(char, device=like.device)
    attack = S.attacker == side
    b = S.bounds
    cols = [attack, ~attack, m.clip(t / TIME, 0, 1),
            S.lord_level[:, side - 1] / 50, S.lord_level[:, 2 - side] / 50,
            (b[:, 1] - b[:, 0]) / 2000, (b[:, 3] - b[:, 2]) / 2000,
            own_alive.sum(-1) / 20.0, enemy_known.sum(-1) / 20.0]
    return _cat(m, [_f(m, char), m.stack([_f(m, c) for c in cols], -1)], -1)
