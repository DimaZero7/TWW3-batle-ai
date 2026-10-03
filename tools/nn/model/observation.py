"""What one side sees: the network's input (docs/en/training/model.md).

The rule: the AI sees only what a human player sees.
* Before battle (static): both rosters (passports, experience rank), lord levels, who attacks,
  the map's bounds, the side's faction character and role.
* In battle: own units — everything, exact morale too. Enemy units — only while visible:
  position, facing, movement, men and health, what they do (melee, moving, running, firing) and the
  morale STATE (steady / wavering / routing / shattered), never the exact morale. Not visible:
  the last seen position and how long ago.
* Events a player is told or sees: own and enemy lord slain and how long ago (the game announces
  it), and per unit how recently it fought in melee and how recently it routed (enemies: while
  seen).
* Damage timers (context, TIMERS): whether the side has dealt any damage yet and how long ago it
  last did, the same for the enemy (damage it dealt to us), and a fine battle clock for the first
  minutes. A human sees his units fight, the kill counters and the balance-of-power bar. Damage
  dealt by a side = some unit of the other side lost health (`hp` fell) since the previous
  observation (Memory.prev_hp; all units, seen or not: the bar shows the totals; a unit whose `hp`
  is not known (NaN) counts nothing until known again). The same rule for the simulator, recorded
  battles and the companion; in training it is the reward's attacker damage (reward.struck: the
  defender's health fell in the step), seen by the attacker as `dealt` and by the defender as `taken`.
* The attacker's progress (context, PROGRESS; both sides see it): the clock of the reward's idle cost
  (tools/nn/train/reward.py, idle_rate > 0). Its damage rate - the defender's gold lost (gold_lost:
  cost x the share lost, a routing unit ROUT_SHARE of what it has left besides, one dead, gone or
  shattered whole; counted once: only beyond the unit's worst so far, so a rally gives nothing back
  and a rout after a rally adds nothing new), share of the budget (the mean of the two armies' cost)
  a minute, an exponential mean over RATE_WINDOW s - divided by RATE_MIN; whether it has reached RATE_MIN yet; seconds since it
  last was at least RATE_MIN (the reward's last_hit). From every unit's health and state (a unit
  whose `hp` is not known counts nothing until known again), so the companion computes it from the
  game's state as the simulator does. It is the reward's clock while the run's --idle-rate,
  --idle-window and rout share are these numbers (0.05, 30 s, 0.5: rollout.Battles warns otherwise).
* Abilities (Obs.abil, per unit and ability slot): the ability's passport (tools/nn/model/abilities.py:
  both sides, it is on the unit's card) and its state. Own units: owned, ready, seconds until ready,
  seconds active left (the ability bar). Enemies: owned, and active now only while the unit is seen
  (the game draws the ability's effect on the unit and lists it among the unit's active effects;
  its timers are not shown). Obs.abil_ok: own abilities the network may use now.
* Innate effects (tools/nn/model/effects.py, the token's last columns): per effect of
  config/nn/effects.json, owned (both sides: the unit's card) and on now (own units; enemies while
  seen: the game lists a seen unit's active effects). On comes from the simulator's `fx_on` or, in a
  recorded battle or the game, from the token's own fields (health, morale state, melee, own morale).
Everything is in the side's frame (tools/nn/model/frame.py), scaled to about -1..1.

State: a dict of arrays [B, N] with the names of recorded `nn_sample` (tools/nn/gamedata.py):
x, z, b, men, hp, mp, ms, m, mv, f, fire, a, k, ox, oz, lf, rf, bf, target, plus `t` [B] (s).
Optional: `vis` [B, N] (the unit is visible to the other side; missing = all visible),
`fat` [B, N] (fatigue state 0-5; missing = unknown), `s` [B, N] (shattered; missing = ms 7), `gone`
[B, N] (left the map; missing = none: in the game such a unit reads no men), `ab{k}_on` /
`ab{k}_cd` [B, N] for ability slot k (seconds active left / until ready: tools/nn/sim/state.py observation(); missing = unknown: no
ability is ready), `fx_on` [B, N] (the innate effects on now, a bitmask: missing = worked out from the
other fields). Units of both sides in one row; `setup.side`
says whose each is. Works on numpy arrays and torch tensors: the same code for recorded
battles and the batched simulator.
"""
from dataclasses import dataclass

import numpy as np

from tools.nn.model import abilities, effects, factions, passport
from tools.nn.model.frame import Frame, army_axis, edge_distances, xp

POS = 500.0      # m: positions and order points
VEL = 5.0        # m/s
EDGE = 1000.0    # m: distance to the map edge (capped)
AGE = 60.0       # s: age of the last sighting (capped at 1)
KILLS = 200.0
MORALE = 2.0     # MoralePercent (-2..2 seen) / 2, clipped to -1.5..1.5
FATIGUE = 6      # fresh, active, winded, tired, very tired, exhausted
EVENT = 120.0    # s: how long an event stays "recent" (1 now, falling to 0 after EVENT)
SINCE = 300.0    # s: seconds since a side last dealt damage / SINCE, capped at 1
EARLY = 30.0     # s: the fine clock log1p(t / EARLY) / log1p(20): 0.23 at 30 s, 0.59 at 150 s, 1 from 10 min
RATE_MIN = 0.05      # the attacker's progress (PROGRESS): the reward's idle_rate (budget share a minute) ...
RATE_WINDOW = 30.0   # ... its idle_window_s (s) ...
ROUT_SHARE = 0.5     # ... and its rout_share (reward.Weights)
RATE_CAP = 4.0       # the rate input: rate / RATE_MIN, at most this

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
    ("melee_recent", "both"), ("rout_recent", "both"),
)
FLAGS = ("is_own", "visible", "seen", "age", "rank")
# Innate effects appended after the passport (02.10.2026; older checkpoints load with zero weights for
# them, encoder.py): per effect (owned, on), tools/nn/model/effects.py.
NAMES = tuple(n for n, _ in DYNAMIC) + FLAGS + tuple(f"passport_{i}" for i in range(passport.SIZE)) + effects.NAMES
INDEX = {n: i for i, n in enumerate(NAMES)}
TOKEN = len(NAMES)
OWN_ONLY = tuple(INDEX[n] for n, who in DYNAMIC if who == "own")
# character; attack, defend, 2 lord levels, map width and depth, 2 counts; own lord slain and how
# recently, the enemy lord the same (CONTEXT_BASE: the context before the damage timers); then TIMERS:
# the fine clock, we dealt damage yet and seconds since we last did (0 before), the enemy the same.
# Then PROGRESS: the attacker's damage rate / RATE_MIN, it reached RATE_MIN yet, seconds since it last
# did (the reward's idle clock).
# Nothing refers to the battle's time limit (a campaign battle may have none): only time elapsed.
# A checkpoint of the older context (a t / 3600 column after defend, no TIMERS) loads with that
# column dropped and zero weights for TIMERS and PROGRESS; one without PROGRESS with zero weights
# for it (encoder.TokenEncoder).
CONTEXT_BASE = factions.SIZE + 12
OLD_TIME = factions.SIZE + 2   # the older context's t / 3600 column (removed)
TIMERS = ("clock_fine", "dealt_any", "dealt_since", "taken_any", "taken_since")
PROGRESS = ("progress_rate", "progress_any", "progress_since")
CONTEXT = CONTEXT_BASE + len(TIMERS) + len(PROGRESS)
CONTEXT_FULL = CONTEXT + factions.SIZE   # the critic's: + the enemy's character
CTX = {n: CONTEXT_BASE + i for i, n in enumerate(TIMERS + PROGRESS)}


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
        self.lord = np.stack([passport.lords(k) for k in self.keys])          # [B, N] the army's general
        self.men0 = np.stack([passport.men(k) for k in self.keys])            # [B, N]
        self.ammo0 = np.stack([passport.ammo(k) for k in self.keys])          # [B, N]
        ab = [abilities.slots(k) for k in self.keys]
        self.abil = np.stack([a[0] for a in ab])                               # [B, N, SLOTS, STATIC]
        self.abil_owned = np.stack([a[1] for a in ab])                         # [B, N, SLOTS]
        self.abil_use = np.stack([a[2] for a in ab])                           # [B, N, SLOTS] may be ordered
        self.fx_owned = np.stack([effects.owned(k) for k in self.keys])       # [B, N, E] innate effects
        self.cost = np.stack([passport.cost(k) for k in self.keys])           # [B, N] gold (PROGRESS)
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
                                       present=t(self.present), attacker=t(self.attacker), lord=t(self.lord),
                                       abil=t(self.abil), abil_owned=t(self.abil_owned),
                                       abil_use=t(self.abil_use), fx_owned=t(self.fx_owned), cost=t(self.cost))
        return self._cache[key]

    def character(self, side):
        """[B, traits] of the given side (1 or 2). Computed once per side (a loop over the batch)."""
        key = ("character", side)
        if key not in self._cache:
            self._cache[key] = np.stack([factions.character(f[side - 1]) for f in self.factions])
        return self._cache[key]


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
    lord: object
    abil: object = None        # abilities (Setup.abil ...); None: the units have no abilities
    abil_owned: object = None
    abil_use: object = None
    fx_owned: object = None    # [B, N, E] innate effects owned (Setup.fx_owned); None: none known
    cost: object = None        # [B, N] multiplayer cost (Setup.cost); None: unknown, every unit counts 1


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
    last_melee_t: object = None   # [B, N] time last seen in melee (-1 never)
    last_rout_t: object = None    # [B, N] time last seen routing (-1 never)
    lord_dead_t: object = None    # [B, 2] time own / enemy lord was slain (-1 alive or none)
    prev_hp: object = None        # [B, N] health share at the previous observation (-1 unknown)
    hit_t: object = None          # [B, 2] time own / enemy side last dealt damage (-1 not yet)
    prev_gold: object = None      # [B, N] the worst gold each unit has lost so far (-1 never known; _progress)
    rate: object = None           # [B] the attacker's damage rate (PROGRESS; budget share a minute)
    rate_t: object = None         # [B] time it last was at least RATE_MIN (-1 not yet)


@dataclass
class Obs:
    tokens: object     # [B, N, TOKEN]
    ctx: object        # [B, CONTEXT]
    own: object        # [B, N] the side's units (present)
    attend: object     # [B, N] tokens attention may use (present, not known dead)
    ctrl: object       # [B, N] own units that take orders (alive, not routing or shattered)
    target_ok: object  # [B, N] enemies that may be attacked now (visible, alive)
    pos: object        # [B, N, 2] (forward, lateral) / POS: current or last seen
    frame: Frame
    side: int
    abil: object = None     # [B, N, SLOTS, abilities.SIZE] per ability slot: state, then passport
    abil_ok: object = None  # [B, N, SLOTS] own abilities that may be used now (owned, self-cast, ready,
    #                         the unit takes orders). Both None when the setup has no abilities.


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
    return Memory(frame, zero, zero, zero, false, false, zero, zero, zero - 1, false, zero - 1, zero - 1,
                  zero[:, :2] - 1, zero - 1, zero[:, :2] - 1, zero - 1, zero[:, 0], zero[:, 0] - 1)


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
    # Events: melee and rout of units seen now; lords slain (announced: known without seeing).
    last_melee = m.where(sees & _b(m, state["m"]), tn, memory.last_melee_t)
    last_rout = m.where(sees & _b(m, state["r"]), tn, memory.last_rout_t)
    cols["melee_recent"] = _recent(m, tn, last_melee)
    cols["rout_recent"] = _recent(m, tn, last_rout)
    slain = S.lord & S.present & ~alive
    slain2 = m.stack([(slain & own).any(-1), (slain & enemy).any(-1)], -1)          # [B, 2] own, enemy
    tb = tn[:, :2]
    lord_dead = m.where(slain2 & (memory.lord_dead_t < 0), tb, memory.lord_dead_t)
    # Damage: some unit of the other side lost health since the previous observation (module doc).
    hp_now = m.nan_to_num(state["hp"] * 1.0, nan=-1.0)
    hp_now = m.where(S.present, hp_now, hp_now * 0 - 1)
    fell = (hp_now >= 0) & (memory.prev_hp >= 0) & (hp_now < memory.prev_hp)
    struck2 = m.stack([(fell & enemy).any(-1), (fell & own).any(-1)], -1)           # [B, 2] we, the enemy
    hit_t = m.where(struck2, tb, memory.hit_t)
    rate, rate_t, gold = _progress(m, state, S, memory, men, tb[:, 0])
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
    fx_own = getattr(S, "fx_owned", None)
    if fx_own is None:                           # a setup without innate effects: none owned
        E = effects.SIZE // 2
        fx_own = np.zeros(xs.shape + (E,), np.float32) if m is np else xs.new_zeros(xs.shape + (E,))
    fx = effects.features(m, state, fx_own, S.present if full else (own | sees))
    tokens = _cat(m, [dyn, _f(m, S.passport), fx], -1) * _f(m, S.present)[..., None]

    attend = S.present & ~dead & ~(own & ~alive)
    ctrl = own & alive & (ms < 6)
    target_ok = enemy & sees & alive
    ctx = _context(m, setup, S, side, own & alive, enemy & ~dead & seen, xs)
    lords = m.stack([_f(m, lord_dead[:, 0] >= 0), _recent(m, tb, lord_dead)[:, 0],
                     _f(m, lord_dead[:, 1] >= 0), _recent(m, tb, lord_dead)[:, 1]], -1)
    hit = hit_t >= 0
    since = m.where(hit, m.clip((tb - hit_t) / SINCE, 0, 1), tb * 0)
    clock = m.log1p(m.clip(t / EARLY, 0, 20)) / np.log1p(20.0)
    timers = m.stack([clock, _f(m, hit[:, 0]), since[:, 0], _f(m, hit[:, 1]), since[:, 1]], -1)
    reached = rate_t >= 0
    progress = m.stack([m.clip(rate / RATE_MIN, 0, RATE_CAP), _f(m, reached),
                        m.where(reached, m.clip((tb[:, 0] - rate_t) / SINCE, 0, 1), rate * 0)], -1)
    ctx = _cat(m, [ctx, lords, _f(m, timers), _f(m, progress)], -1)
    pos = m.stack([fwd, lat], -1) / POS
    pos = m.where(seen[..., None], pos, pos * 0)
    new = Memory(fr, last_x, last_z, last_t, seen, dead, m.where(sees, xs, xs * 0), m.where(sees, zs, zs * 0),
                 tn, sees, last_melee, last_rout, lord_dead, hp_now, hit_t, gold, rate, rate_t)
    if full:   # the critic also knows the enemy's character
        other = setup.character(3 - side)
        ctx = _cat(m, [ctx, _f(m, other if m is np else m.as_tensor(other, device=x.device))], -1)
    abil, abil_ok = _abilities(m, state, S, own, sees, ctrl, full, xs)
    return Obs(_f(m, tokens), _f(m, ctx), own, attend, ctrl, target_ok, _f(m, pos), fr, side,
               abil, abil_ok), new


def _abilities(m, state, S, own, sees, ctrl, full, like):
    """(abil [B, N, SLOTS, SIZE], abil_ok [B, N, SLOTS]): each slot's state and passport (module doc)."""
    K = abilities.SLOTS
    if getattr(S, "abil", None) is None:          # a setup without abilities (e.g. an older LiveSetup):
        return None, None                         # no ability input at all (and nothing to store)
    owned = S.abil_owned & S.present[..., None]
    known = all(state.get(f"ab{k}_{t}") is not None for k in range(K) for t in ("on", "cd"))
    zero = like * 0
    on = m.stack([_f(m, state[f"ab{k}_on"]) if known else zero for k in range(K)], -1)       # [B, N, K]
    cd = m.stack([_f(m, state[f"ab{k}_cd"]) if known else zero for k in range(K)], -1)
    mine = (S.present if full else own)[..., None] & owned
    ready = mine & S.abil_use & (cd <= 0) & (on <= 0) & known
    seen_on = owned & (on > 0) & ((S.present if full else own | sees)[..., None])
    none = zero[..., None] * 0
    dyn = m.stack([_f(m, owned), _f(m, ready), m.where(mine, m.clip(cd / abilities.RECHARGE, 0, 3), none),
                   m.where(mine, m.clip(on / abilities.ACTIVE, 0, 2), none), _f(m, seen_on)], -1)
    abil = _cat(m, [dyn, _f(m, S.abil) * _f(m, owned)[..., None]], -1)
    abil_ok = ready & own[..., None] & ctrl[..., None]
    return _f(m, abil), abil_ok


def _b(m, a):
    return a > 0.5 if m is np else a.bool() if a.dtype != m.bool else a


def gold_lost(m, state, S, men):
    """[B, N] the gold each unit has lost, from the observed fields (as tools/nn/train/reward.py
    gold_lost): its cost x the share lost - the health lost; dead (no men), gone or shattered: whole;
    routing: ROUT_SHARE of what it has left besides. -1 where not known (`hp` not read and not out of
    the fight) and for padding."""
    hp = state["hp"] * 1.0
    hp_lost = m.clip(1 - m.nan_to_num(hp, nan=1.0), 0, 1)
    shattered, gone = state.get("s"), state.get("gone")
    shattered = m.nan_to_num(state["ms"] * 1.0, nan=0.0) == 7 if shattered is None else _b(m, shattered)
    out = (men <= 0) | shattered
    if gone is not None:
        out = out | _b(m, gone)
    share = m.where(out, hp_lost * 0 + 1, hp_lost + (1 - hp_lost) * ROUT_SHARE * _f(m, _b(m, state["r"])))
    cost = getattr(S, "cost", None)
    g = _f(m, share * (_f(m, S.present) if cost is None else cost))
    return m.where(S.present & ((hp == hp) | out), g, g * 0 - 1)


def _progress(m, state, S, memory, men, now):
    """(rate [B], rate_t [B], gold [B, N]): the attacker's damage rate and the time it last was at least
    RATE_MIN after this observation (PROGRESS; tools/nn/train/reward.py hit_rate and idle_cost's
    last_hit), and every unit's worst gold lost so far (Memory.prev_gold: the next observation's
    reference; -1 never known). A loss counts once, as in reward.gold_lost: only beyond the unit's
    worst - a rally is not damage and a rout after a rally adds nothing until the unit has lost more
    than at its worst. The defender's loss counts over the units known at both observations (a unit
    whose health was not known at the previous one: its worst is raised to its loss now, uncounted)."""
    gold = gold_lost(m, state, S, men)
    known = gold >= 0
    worst = m.where(known, m.maximum(gold, memory.prev_gold), memory.prev_gold)
    cost = getattr(S, "cost", None)
    cost = _f(m, S.present) if cost is None else cost * _f(m, S.present)
    budget = m.clip(cost.sum(-1) / 2, 1.0, 1e12)                     # the mean of the two armies' cost
    defender = S.present & (S.side == (3 - S.attacker).reshape(-1, 1))
    prev_t = memory.prev_t[:, 0]
    dt = now - prev_t
    step = (prev_t >= 0) & (dt > 0)
    both = defender & known & (memory.prev_gold >= 0) & (memory.prev_hp >= 0)
    share = m.clip(m.where(both, worst - memory.prev_gold, gold * 0).sum(-1) / budget, 0, 1e12)
    dts = m.where(step, dt, dt * 0 + 1)
    a = m.exp(-dts / RATE_WINDOW)
    rate = m.where(step, a * memory.rate + (1 - a) * share * (60.0 / dts), memory.rate)
    rate_t = m.where(step & (rate >= RATE_MIN), now, memory.rate_t)
    again = (prev_t >= 0) & ~step                       # the same time again: the reference stays
    return rate, rate_t, m.where(again[:, None], memory.prev_gold, worst)


def _recent(m, now, when):
    """1 when the event is now, falling to 0 after EVENT s; 0 when it never happened."""
    return m.where(when >= 0, m.clip(1 - (now - when) / EVENT, 0, 1), now * 0)


def _cat(m, xs, dim):
    return np.concatenate(xs, dim) if m is np else m.cat(xs, dim)


def _context(m, setup, S, side, own_alive, enemy_known, like):
    char = setup.character(side)
    char = char if m is np else m.as_tensor(char, device=like.device)
    attack = S.attacker == side
    b = S.bounds
    cols = [attack, ~attack,
            S.lord_level[:, side - 1] / 50, S.lord_level[:, 2 - side] / 50,
            (b[:, 1] - b[:, 0]) / 2000, (b[:, 3] - b[:, 2]) / 2000,
            own_alive.sum(-1) / 20.0, enemy_known.sum(-1) / 20.0]
    return _cat(m, [_f(m, char), m.stack([_f(m, c) for c in cols], -1)], -1)
