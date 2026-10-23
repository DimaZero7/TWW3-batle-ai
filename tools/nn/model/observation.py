"""What one side sees: the network's input (docs/en/training/model.md).

The rule: the AI sees only what a human player sees.
* Before battle (static): both rosters (passports, experience rank), lord levels, who attacks,
  the map's bounds, the side's faction character and role.
* In battle: own units — everything. Enemy units — only while visible, everything the game's battle UI
  shows for a selected or hovered enemy unit (Normal difficulty; Legendary hides the enemy's leadership):
  position, facing, movement, men and health, what they do (melee, moving, running, firing), the exact
  morale (the leadership bar under the banner and the leadership stat with its breakdown) and its state
  1-7, ammunition, kills, fatigue state, flank / rear threats (the leadership breakdown lists them),
  volley readiness (the volleys are seen: from the ammunition read while the unit is seen). Never the
  enemy's orders (order point, has a target, the order in force: OWN_ONLY) or its ability timers (the bridge cannot read them).
  Not visible: the last seen position and how long ago.
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
* Abilities (Obs.abil, per unit and ability slot: every ability of the unit, tools/nn/model/abilities.py layout -
  the actives and game-fired ones from slot 0, the passives from slot ACTIVE): its state, and the ability's index into
  the passports' table (Obs.abil_idx: the networks put the passport back, abilities.full; both sides, it is on the
  unit's card). Own units: owned, ready, seconds until ready, seconds active left (the ability bar). Enemies: owned,
  and active now only while the unit is seen (the game draws the ability's effect on the unit and lists it among the
  unit's active effects; its timers are not shown). On now (own units, seen enemies): an active's timer runs, a
  passive's condition holds (the innate effect's on, below). The timers of the actives' slots come from the state's
  `ab{k}_on` / `ab{k}_cd` (the simulator's first SLOTS; a slot without them is not ready). Obs.abil_ok [B, N,
  ACTIVE]: own abilities the network may use now.
* Spells (16.10.2026, S27; the user's rule: what a player sees of magic): the side's own winds of magic, current and
  reserve (the context's WINDS: the game's winds bar; never the enemy's); a caster's spells are slots of its abilities
  (the last SPELLS active slots, abilities.py: their passports - both sides, the unit card lists an enemy mage's spells),
  their timers from the state's `sp{k}_cd` / `sp{k}_on` (own units), affordable (the side's current winds at least the
  spell's cost) and casting (own units: its cast is under way); Obs.abil_ok of a spell slot: owned, recharged, affordable,
  not casting, the unit takes orders and some enemy may be attacked (where it falls: the target pointer; no range mask -
  the caster walks into range, pairs.py shows how far). MAGIC (the token's last columns; own units and seen enemies):
  casting now (the state's `cast_now`: winding up, the staff glows - not which spell nor where) and struck by a spell
  (`bm_hit`: spell projectiles fell on it in the last step - the strikes are seen once they fall). Missing fields: 0.
* The unit's shape (SHAPE, the token's last columns, 17.10.2026, S27; own units and seen enemies: the soldiers are seen
  on the field): tools/nn/fighters/shape.unit_features from the soldiers' places - men per m², width and depth, the
  mass off the middle forward and sideways, the share within an enemy's reach, empty cells of the unit's 4 x 4 box and
  of its front row, wounds (from health and men, both shown on the card), the 4 x 4 density map - scaled (SHAPE_SCALE),
  and shape_known (a reading exists: the fighters lab in the simulator, the companion's --shape in the game). The
  state's `shape` [B, N, 25] and `shape_ok` [B, N]; without them (the rectangle simulator, older recordings) zeros and
  shape_known 0. Older checkpoints: zero weights (they play as before).
* Effects on me from others (RECEIVED, the token's last columns, 13.10.2026; own units and seen enemies: the game lists
  them among the unit's active effects): the stats the abilities of other units lay on the unit now - a friend's aura
  or cast, an enemy's - summed as the passports' allies group (an enemy's: its enemies group), abilities.received;
  from the state's `ab_rx` (the abilities on it, a bitmask: the simulator's; the companion's from the unit card;
  missing: none). The unit's own abilities and innate effects are in its slots and its innate effects.
* What the effects on the unit do (FX_STATS, the token's last columns but FIGHT, 23.10.2026; own units and seen enemies:
  the game lists a seen unit's active effects with their numbers): the stats of its innate and contact effects on now
  (the Innate effects' `on` below: Frenzy, Strength in Numbers, poison, contagion, suppression...) summed as RECEIVED -
  multipliers multiply (value - 1), additions add (/ scale) - effects.stats_on; so poison is "-15 % speed, damage" and
  not only effect number 19. An aura's friends get it through RECEIVED, its owner here.
* The fight (FIGHT, the token's last columns, 23.10.2026; own units and seen enemies): under fire - a living unit of
  the other side, not in melee, fires (the state's `fire`) with this unit its target (`target`: the simulator's shot
  target, the game's engine target; the general's FIGHT under_fire, tools/nn/general/runtime.py fight_of) - and the
  seconds the unit's current melee has gone on / MELEE_S, at most 1 (from the observations: Memory.melee_t0, the
  first observation that saw it in melee; a unit not seen or out of melee starts again; the companion the same).
  Missing `target`: not under fire.
* Volley readiness (VOLLEY, own units and seen enemies, the column before ARC): seconds since the
  unit's projectiles left (`a`) last fell between two observations (its last volley) / its passport
  reload_s, clipped to 0..1; 1 before its first shot; 0 for units without missiles. Seen at the
  decisions only (Memory.prev_ammo, volley_t), so the simulator (its ammo counter at the network's 1 s
  cadence) and the companion (the game's ammo_left() each second) compute it the same way, from the
  same code. An enemy's ammunition is read only while it is seen: hidden, its reference is forgotten
  (a fall across the time it was hidden is no volley; the last seen volley stays). The simulator's men load fully only after its measured reload (missile.py: ~1.1-1.3x the
  passport's), so 1 means "the passport's reload has passed", not "every man is loaded".
* Fire arc (ARC, both sides, the token's last column, 07.10.2026): the passport's fire arc each side / 180
  (passport.fire_arc): 1.0 for the Night Runners' throwing stars, which shoot all round and on the move; 0.17 for
  archers and slings. It stands at the token's end, after VOLLEY, so older checkpoints load with zero weights for it.
* Innate effects (tools/nn/model/effects.py, the columns before VOLLEY): per effect of
  config/nn/effects.json, owned (both sides: the unit's card) and on now (own units; enemies while
  seen: the game lists a seen unit's active effects). On comes from the simulator's `fx_on` or, in a
  recorded battle or the game, from the token's own fields (health, morale state, melee, own morale).
* The order in force (ORDER, own units only, the token's last columns, 07.10.2026): its kind one-hot (hold, move,
  attack, withdraw) and how long it has been in force, s / ORDER_AGE capped at 1. From the state's optional
  `order_kind` and `order_target` [B, N] (the simulator's order in force; the companion's last order given,
  exchange.order_points) and the order point ox, oz: a new kind, a new attack target or a point moved more than
  ORDER_MOVE_M (as reward.order_changes) starts the clock again (Memory.ord_*). Missing: all 0. Older
  checkpoints load with zero weights for these columns.
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
from tools.nn.sim.orders import ATTACK, HOLD, MOVE, WITHDRAW

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
ORDER_AGE = 60.0     # s: the order in force's age / ORDER_AGE, capped at 1 (ORDER)
ORDER_MOVE_M = 10.0  # m: a move point moved more than this is a new order (= reward.Weights.order_move_m)
MELEE_S = 30.0       # s: the current melee's seconds / MELEE_S, capped at 1 (FIGHT; the charge bonus fades in ~13 s)

# The token's dynamic features: (name, who sees it). "both": own units and visible enemies (the game's UI shows
# them for an enemy unit, module doc); "own": own units only - the orders (zero for enemies; the critic's full view
# fills them for all units). The columns keep their places (trained checkpoints load as before).
DYNAMIC = (
    ("fwd", "both"), ("lat", "both"), ("face_cos", "both"), ("face_sin", "both"),
    ("vel_fwd", "both"), ("vel_lat", "both"), ("men", "both"), ("hp", "both"),
    ("state_steady", "both"), ("state_wavering", "both"), ("state_routing", "both"), ("state_shattered", "both"),
    ("melee", "both"), ("moving", "both"), ("running", "both"), ("firing", "both"),
    ("edge_fwd", "both"), ("edge_back", "both"), ("edge_right", "both"), ("edge_left", "both"),
    ("morale", "both"), *((f"ms_{i}", "both") for i in range(1, 8)),
    ("ammo", "both"), ("kills", "both"), *((f"fatigue_{i}", "both") for i in range(FATIGUE)),
    ("fatigue_known", "both"),
    ("order_fwd", "own"), ("order_lat", "own"), ("has_order", "own"), ("has_target", "own"),
    ("threat_left", "both"), ("threat_right", "both"), ("threat_rear", "both"),
    ("melee_recent", "both"), ("rout_recent", "both"),
)
FLAGS = ("is_own", "visible", "seen", "age", "rank")
# Innate effects appended after the passport (02.10.2026; older checkpoints load with zero weights for
# them, encoder.py): per effect (owned, on), tools/nn/model/effects.py. Then VOLLEY (03.10.2026, own units and
# seen enemies; older checkpoints: zero weights): seconds since the last volley / passport reload (module doc).
VOLLEY = ("volley_ready",)
# Then ARC (07.10.2026, both sides: the unit's card; older checkpoints: zero weights): the passport's fire arc each
# side / 180 (passport.fire_arc) - appended at the token's end, not among the passport's features, so the columns
# before it keep their places.
ARC = ("fire_arc",)
# Then ORDER (07.10.2026, own units only; older checkpoints: zero weights): the order in force, kind and age (module doc).
ORDER = ("order_hold", "order_move", "order_attack", "order_withdraw", "order_age")
# Then WEAPON (11.10.2026, both sides: the unit's card; older checkpoints: zero weights, encoder.py): the passport's
# shots a man fires in a volley (passport.volley_shots: the Ratling Gun's burst of 18 -> 0.9, 1 shot -> 0.05) and its
# melee weapon's magical attacks (passport.magic).
WEAPON = ("volley_shots", "magic_attacks")
# Then AIM (13.10.2026, both sides: the unit's card; older checkpoints: zero weights): the passport's seconds of aim
# before the first shot from a halt and of silence after an attack order on another target, / passport.AIM_S
# (passport.aim: the simulator's numbers, tools/nn/sim/params.aim_times; the reload is not reset by a new target).
AIM = ("aim_first", "aim_retarget")
# Then RECEIVED (13.10.2026, own units and seen enemies; older checkpoints: zero weights): the stat bonuses other units'
# abilities lay on the unit now (module doc; abilities.received).
RECEIVED = abilities.RECEIVED_NAMES
# Then MAGIC (16.10.2026, own units and seen enemies; older checkpoints: zero weights): casting now, struck by a spell
# (module doc).
MAGIC = ("casting_now", "spell_strike")
# Then SHAPE (17.10.2026, own units and seen enemies; older checkpoints: zero weights): the unit's shape from its
# soldiers' places (module doc): known, the 9 numbers of tools/nn/fighters/shape.FEATURES, the 4 x 4 map.
SHAPE_FEATURES = ("men_m2", "width_m", "depth_m", "skew_fwd_m", "skew_side_m", "contact", "holes", "front_holes",
                  "wounded")
SHAPE_GRID = 16
SHAPE = ("shape_known",) + tuple(f"shape_{n}" for n in SHAPE_FEATURES) + tuple(f"shape_cell{i}" for i in
                                                                                  range(SHAPE_GRID))
# the raw numbers' scales: men per m² x 2 (~0.5), metres / 30 (width, depth) and / 10 (the offsets), shares as they
# are, a cell's share x 4 (an even spread 1 / 16 -> 0.25)
SHAPE_SCALE = (2.0, 1 / 30, 1 / 30, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0) + (4.0,) * SHAPE_GRID
# Then FX_STATS (23.10.2026, own units and seen enemies; older checkpoints: zero weights): what the effects on the unit
# do, as numbers (module doc; effects.stats_on). Then FIGHT (23.10.2026, the same): under fire, the current melee's
# seconds (module doc).
FX_STATS = effects.FX_STAT_NAMES
FIGHT = ("under_fire", "melee_s")
NAMES = (tuple(n for n, _ in DYNAMIC) + FLAGS + tuple(f"passport_{i}" for i in range(passport.SIZE)) + effects.NAMES
         + VOLLEY + ARC + ORDER + WEAPON + AIM + RECEIVED + MAGIC + SHAPE + FX_STATS + FIGHT)
INDEX = {n: i for i, n in enumerate(NAMES)}
TOKEN = len(NAMES)
# Effect pairs of the tokens of trained checkpoints before 11.10.2026 (14 effects, no WEAPON): old_columns maps them.
OLD_EFFECTS = (14,)
# Column groups appended at the token's end since the new units (11.10.2026), oldest first: a token without the last
# of them keeps its columns where they are (old_columns), zeros (zero weights) for the missing ones.
TAIL = (AIM, RECEIVED, MAGIC, SHAPE, FX_STATS, FIGHT)


def old_columns(width):
    """[the new token's column of each column of an older token `width` wide] for a token from before the effects
    grew past OLD_EFFECTS (11.10.2026: the new units' attributes and contact effects, then WEAPON after ORDER) that had
    columns after its effect pairs (VOLLEY 03.10, ARC and ORDER 07.10.2026): its effect pairs keep their places, those
    columns move behind the new pairs to their places by name, the new columns get zero weights (encoder.TokenEncoder).
    A token of today's columns without the last groups of TAIL (AIM, RECEIVED, 13.10.2026): its columns stay where
    they are.
    None: not such a token (today's; a token that ended with its effect pairs, or older still, grew at the end only:
    encoder.pad_inputs)."""
    width_now = TOKEN
    for group in reversed(TAIL):
        width_now -= len(group)
        if width == width_now:
            return list(range(width))
    base = len(DYNAMIC) + len(FLAGS) + passport.SIZE
    E, after = effects.SIZE // 2, VOLLEY + ARC + ORDER
    for e_old in OLD_EFFECTS:
        head = base + 2 * e_old
        k = width - head
        if e_old < E and k in (len(VOLLEY), len(VOLLEY) + len(ARC), len(after)):
            return list(range(head)) + [INDEX[n] for n in after[:k]]
    return None
# The token columns an enemy never shows (its orders): zero for enemies in the side's view.
OWN_ONLY = tuple(INDEX[n] for n, who in DYNAMIC if who == "own") + tuple(INDEX[n] for n in ORDER)
# character; attack, defend, 2 lord levels, map width and depth, 2 counts; own lord slain and how
# recently, the enemy lord the same (CONTEXT_BASE: the context before the damage timers); then TIMERS:
# the fine clock, we dealt damage yet and seconds since we last did (0 before), the enemy the same.
# Then PROGRESS: the attacker's damage rate / RATE_MIN, it reached RATE_MIN yet, seconds since it last
# did (the reward's idle clock). Then WINDS (16.10.2026): the side's own winds of magic, current and reserve /
# abilities.WINDS_SCALE (the state's wm_cur, wm_res; 0 missing). A checkpoint without WINDS loads with zero weights
# for them (encoder.TokenEncoder; the critic's enemy character after them keeps its weights).
# Nothing refers to the battle's time limit (a campaign battle may have none): only time elapsed.
# A checkpoint of the older context (a t / 3600 column after defend, no TIMERS) loads with that
# column dropped and zero weights for TIMERS and PROGRESS; one without PROGRESS with zero weights
# for it (encoder.TokenEncoder).
CONTEXT_BASE = factions.SIZE + 12
OLD_TIME = factions.SIZE + 2   # the older context's t / 3600 column (removed)
TIMERS = ("clock_fine", "dealt_any", "dealt_since", "taken_any", "taken_since")
PROGRESS = ("progress_rate", "progress_any", "progress_since")
WINDS = ("winds_current", "winds_reserve")
CONTEXT = CONTEXT_BASE + len(TIMERS) + len(PROGRESS) + len(WINDS)
CONTEXT_FULL = CONTEXT + factions.SIZE   # the critic's: + the enemy's character
CTX = {n: CONTEXT_BASE + i for i, n in enumerate(TIMERS + PROGRESS + WINDS)}


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
        self.reload = np.stack([passport.reload(k) for k in self.keys])       # [B, N] s (VOLLEY; 0: no missile)
        self.arc = np.stack([passport.fire_arc(k) for k in self.keys])        # [B, N] fire arc / 180 (ARC)
        self.weapon = np.stack([passport.weapon(k) for k in self.keys])       # [B, N, len(WEAPON)] (WEAPON)
        self.aim = np.stack([passport.aim(k) for k in self.keys])             # [B, N, len(AIM)] (AIM)
        ab = [abilities.slots(k) for k in self.keys]
        self.abil_idx = np.stack([a[0] for a in ab])                           # [B, N, SLOTS] row in abilities.table
        self.abil_use = np.stack([a[1] for a in ab])                           # [B, N, SLOTS] may be ordered
        self.abil_fx = np.stack([a[2] for a in ab])                            # [B, N, SLOTS] a passive's effect
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
                                       abil_idx=t(self.abil_idx), abil_use=t(self.abil_use),
                                       abil_fx=t(self.abil_fx), fx_owned=t(self.fx_owned), cost=t(self.cost),
                                       reload=t(self.reload), arc=t(self.arc), weapon=t(self.weapon),
                                       aim=t(self.aim))
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
    abil_idx: object = None    # abilities (Setup.abil_idx ...); None: the units have no abilities
    abil_use: object = None
    abil_fx: object = None
    fx_owned: object = None    # [B, N, E] innate effects owned (Setup.fx_owned); None: none known
    cost: object = None        # [B, N] multiplayer cost (Setup.cost); None: unknown, every unit counts 1
    reload: object = None      # [B, N] passport reload_s (Setup.reload, VOLLEY); None: unknown, the input 0
    arc: object = None         # [B, N] passport fire arc / 180 (Setup.arc, ARC); None: unknown, the input 0
    weapon: object = None      # [B, N, len(WEAPON)] (Setup.weapon, WEAPON); None: unknown, the inputs 0
    aim: object = None         # [B, N, len(AIM)] (Setup.aim, AIM); None: unknown, the inputs 0


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
    prev_ammo: object = None      # [B, N] projectiles left at the last observation that read them (-1 unknown)
    volley_t: object = None       # [B, N] time they last fell (VOLLEY: the last volley; -1 not yet)
    ord_kind: object = None       # [B, N] the order in force at the last observation (ORDER; -1 unknown) ...
    ord_tgt: object = None        # ... its attack target, its point
    ord_x: object = None
    ord_z: object = None
    ord_t: object = None          # [B, N] time it was first seen in force (-1 never)
    melee_t0: object = None       # [B, N] time the current melee was first seen (FIGHT; -1 not in melee or not seen)


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
    abil: object = None     # [B, N, SLOTS, abilities.STATE] per ability slot: its state (abilities.full: + passport)
    abil_ok: object = None  # [B, N, ACTIVE] own abilities that may be used now (owned, self-cast, ready,
    #                         the unit takes orders). Both None when the setup has no abilities.
    abil_idx: object = None  # [B, N, SLOTS] int64 the slot's ability: its row in abilities.table (-1 none)


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
                  zero[:, :2] - 1, zero - 1, zero[:, :2] - 1, zero - 1, zero[:, 0], zero[:, 0] - 1, zero - 1,
                  zero - 1, ord_kind=zero - 1, ord_tgt=zero - 1, ord_x=zero, ord_z=zero, ord_t=zero - 1,
                  melee_t0=zero - 1)


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
    cols["volley_ready"], prev_ammo, volley_t = _volley(m, state, S, memory, tn, xs, sees)
    # Who sees what: "both" fields only for units seen now (own units always); "own" fields (the orders)
    # only for own units (every unit in the critic's full view). Position and edges stay for the last sighting.
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
    fx_on = effects.active(m, state, fx_own)
    fx = effects.features(m, state, fx_own, S.present if full else (own | sees), on=fx_on)
    late = m.stack([m.where(sees, cols[n], cols[n] * 0) for n in VOLLEY], -1)
    arc = getattr(S, "arc", None)
    arc = xs * 0 if arc is None else _f(m, arc)
    order, ord_mem = _order(m, state, memory, tn, xs)
    order = m.stack([m.where(see_all & S.present, c, c * 0) for c in order], -1)
    weapon = getattr(S, "weapon", None)
    weapon = m.stack([xs * 0] * len(WEAPON), -1) if weapon is None else _f(m, weapon)
    aim = getattr(S, "aim", None)
    aim = m.stack([xs * 0] * len(AIM), -1) if aim is None else _f(m, aim)
    rx = _received(m, state, S, S.present if full else (own | sees), xs)
    shown = S.present if full else (own | sees)
    magic = m.stack([m.where(shown, _f(m, _field(state, k, xs) > 0), xs * 0) for k in ("cast_now", "bm_hit")], -1)
    form = _shape(m, state, shown, xs)
    fxs = _f(m, effects.stats_on(m, fx_on & shown[..., None]))
    fight, melee_t0 = _fight(m, state, S, memory, tn, alive, sees, shown, xs)
    tokens = (_cat(m, [dyn, _f(m, S.passport), fx, late, arc[..., None], order, weapon, aim, rx, magic, form, fxs,
                       fight], -1) * _f(m, S.present)[..., None])

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
    winds = _winds(m, state, own, xs)
    ctx = _cat(m, [ctx, lords, _f(m, timers), _f(m, progress), winds], -1)
    pos = m.stack([fwd, lat], -1) / POS
    pos = m.where(seen[..., None], pos, pos * 0)
    new = Memory(fr, last_x, last_z, last_t, seen, dead, m.where(sees, xs, xs * 0), m.where(sees, zs, zs * 0),
                 tn, sees, last_melee, last_rout, lord_dead, hp_now, hit_t, gold, rate, rate_t, prev_ammo, volley_t,
                 *ord_mem, melee_t0=melee_t0)
    if full:   # the critic also knows the enemy's character
        other = setup.character(3 - side)
        ctx = _cat(m, [ctx, _f(m, other if m is np else m.as_tensor(other, device=x.device))], -1)
    abil, abil_ok, abil_idx = _abilities(m, state, S, own, sees, ctrl, full, xs, fx_on, target_ok)
    return Obs(_f(m, tokens), _f(m, ctx), own, attend, ctrl, target_ok, _f(m, pos), fr, side,
               abil, abil_ok, abil_idx), new


def _fight(m, state, S, memory, tn, alive, sees, shown, like):
    """([B, N, len(FIGHT)] the FIGHT columns, the Memory's melee_t0): module doc. Under fire: a living unit of the other
    side, not in melee, firing with this unit its target; the current melee's seconds from the first observation that
    saw the unit (seen now) in melee. Only where shown (own units, seen enemies; the critic's: all)."""
    zero = like * 0
    tgt, fire = state.get("target"), state.get("fire")
    if tgt is None or fire is None:
        under = zero > 0
    else:
        tgt = m.nan_to_num(tgt * 1.0, nan=-1.0)
        side = S.side
        shoots = alive & _b(m, fire) & ~_b(m, state["m"]) & (tgt >= 0)                         # [B, N] shooter i
        N = like.shape[1]
        cols = np.arange(N) if m is np else m.arange(N, device=like.device)
        hit = shoots[..., None] & (tgt[..., None] == cols)                                   # [B, i, j]: i shoots j
        foe = side[..., :, None] != side[..., None, :]
        under = (hit & foe).any(-2) & S.present
    melee = sees & _b(m, state["m"])
    old = zero - 1 if memory.melee_t0 is None else memory.melee_t0
    t0 = m.where(melee, m.where(old >= 0, old, tn), zero - 1)
    secs = m.where(melee, m.clip((tn - t0) / MELEE_S, 0, 1), zero)
    out = m.stack([_f(m, under), _f(m, secs)], -1)
    return m.where(shown[..., None], out, out * 0), t0


def _shape(m, state, shown, like):
    """[B, N, len(SHAPE)] the units' shape (module doc): known (the state's shape_ok and shown: own units, seen
    enemies), then the state's shape [B, N, 25] x SHAPE_SCALE where known; zeros without them."""
    v, ok = state.get("shape"), state.get("shape_ok")
    if v is None or ok is None:
        return m.stack([like * 0] * len(SHAPE), -1)
    known = shown & (_f(m, ok) > 0.5)
    scale = SHAPE_SCALE if m is np else m.as_tensor(SHAPE_SCALE, device=like.device)
    vals = _f(m, v) * scale
    vals = m.where(known[..., None], vals, vals * 0)
    return _cat(m, [_f(m, known)[..., None], vals], -1)


def _winds(m, state, own, like):
    """[B, len(WINDS)] the side's own current winds and reserve / abilities.WINDS_SCALE (module doc): the state's wm_cur,
    wm_res of its units (the same on every unit of a side; the largest over its own units); 0 when missing."""
    cols = []
    for k in ("wm_cur", "wm_res"):
        v = state.get(k)
        if v is None:
            cols.append(like[:, 0] * 0)
            continue
        v = m.where(own, _f(m, v), like * 0)
        cols.append((v.max(-1) if m is np else v.max(-1).values) / abilities.WINDS_SCALE)
    return m.stack(cols, -1)


def _abilities(m, state, S, own, sees, ctrl, full, like, fx_on, target_ok=None):
    """(abil [B, N, SLOTS, STATE], abil_ok [B, N, ACTIVE], abil_idx [B, N, SLOTS]): each slot's state and ability
    (module doc). fx_on [B, N, E]: the innate effects on (effects.active; the passives' on now); target_ok [B, N]:
    the enemies that may be attacked now (a spell needs one to fall on; None: any)."""
    K, A = abilities.SLOTS, abilities.ACTIVE
    if getattr(S, "abil_idx", None) is None:      # a setup without abilities (e.g. an older LiveSetup):
        return None, None, None                   # no ability input at all (and nothing to store)
    idx = S.abil_idx
    owned = (idx >= 0) & S.present[..., None]
    idx = m.where(owned, idx, idx * 0 - 1)
    # the actives' timers: the slots the state has them for (the simulator's first SLOTS, the companion's), the rest
    # none (never ready); the passives' slots have none
    # (a spell slot's timers: the state's sp{j}_on / sp{j}_cd)
    src = {k: f"ab{k}" for k in range(A)}
    src.update({k: f"sp{j}" for j, k in enumerate(abilities.SPELL_SLOTS)})
    timed = [k for k in range(A) if state.get(f"{src[k]}_on") is not None and state.get(f"{src[k]}_cd") is not None]
    zero = like * 0
    on = m.stack([_f(m, state[f"{src[k]}_on"]) if k in timed else zero for k in range(K)], -1)      # [B, N, K]
    cd = m.stack([_f(m, state[f"{src[k]}_cd"]) if k in timed else zero for k in range(K)], -1)
    has_timer = m.stack([(zero == zero) if k in timed else (zero != zero) for k in range(K)], -1)
    mine = (S.present if full else own)[..., None] & owned
    ready = mine & S.abil_use & (cd <= 0) & (on <= 0) & has_timer
    # spells: the side's current winds pay for it, the caster is not casting already; it needs an enemy to fall on
    spell = m.stack([(zero == zero) if k in abilities.SPELL_SLOTS else (zero != zero) for k in range(K)], -1) & owned
    cost = abilities.column("mana", idx) * abilities.WINDS_SCALE                                      # [B, N, K]
    wm = state.get("wm_cur")
    wm = zero if wm is None else _f(m, wm)
    affordable = mine & spell & (wm[..., None] >= cost - 1e-4)
    ck = state.get("cast_k")
    ck = zero - 1 if ck is None else _f(m, ck)
    casting_any = ck >= 0
    casting = mine & spell & m.stack([(ck == k - abilities.SPELL_AT) if k in abilities.SPELL_SLOTS else (zero != zero)
                                      for k in range(K)], -1)
    foe = (zero[:, 0] == zero[:, 0]) if target_ok is None else target_ok.any(-1)                     # [B]
    ready = m.where(spell, ready & affordable & ~casting_any[..., None] & foe[:, None, None], ready)
    vis = owned & ((S.present if full else own | sees)[..., None])
    seen_on = vis & (on > 0)
    fx = S.abil_fx
    if fx_on.shape[-1]:
        passive_on = m.where(fx >= 0, _take(m, fx_on, m.where(fx >= 0, fx, fx * 0)), fx < -1)
    else:
        passive_on = fx < -1
    on_now = vis & (seen_on | passive_on)
    none = zero[..., None] * 0
    state_ = m.stack([_f(m, owned), _f(m, ready), m.where(mine, m.clip(cd / abilities.RECHARGE, 0, 3), none),
                      m.where(mine, m.clip(on / abilities.ACTIVE_S, 0, 2), none), _f(m, seen_on), _f(m, on_now),
                      _f(m, affordable), _f(m, casting)], -1)
    abil_ok = (ready & own[..., None] & ctrl[..., None])[..., :A]
    return _f(m, state_), abil_ok, idx


def _take(m, a, i):
    """a [..., E] gathered at i [..., K] (int) -> [..., K]."""
    if m is np:
        return np.take_along_axis(a, i, -1)
    return a.gather(-1, i.long())


def _received(m, state, S, visible, like):
    """[B, N, len(RECEIVED)] the RECEIVED columns (module doc): from the state's `ab_rx`; only where visible."""
    rx = state.get("ab_rx")
    idx = getattr(S, "abil_idx", None)
    if rx is None or idx is None:
        return m.stack([like * 0] * len(RECEIVED), -1)
    rx = rx.astype(np.int64) if m is np else rx.long()
    out = abilities.received(rx, S.side, idx)
    return m.where(visible[..., None], _f(m, out), _f(m, out) * 0)


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


def _volley(m, state, S, memory, tn, like, sees=None):
    """(ready [B, N], prev_ammo, volley_t): VOLLEY (module doc). The unit's projectiles left `a` fell
    since the previous observation that read them -> its last volley is now; ready = seconds since it /
    the passport reload_s, clipped to 0..1, 1 before the first; 0 without a missile (or a reload not
    known). `a` not read (NaN) or missing: nothing changes. The same time again (the critic's view or a
    repeated state) finds no fall: the reference is the same number. sees [B, N]: the units seen now
    (None: all present) - a unit not seen is not read and its reference is forgotten (-1)."""
    a = state.get("a")
    a = like * 0 - 1 if a is None else m.nan_to_num(a * 1.0, nan=-1.0)
    sees = S.present if sees is None else sees & S.present
    known = (a >= 0) & sees
    fell = known & (memory.prev_ammo >= 0) & (a < memory.prev_ammo)
    volley_t = m.where(fell, tn, memory.volley_t)
    reload = getattr(S, "reload", None)
    reload = like * 0 if reload is None else _f(m, reload)
    shooter = reload > 0
    since = m.clip((tn - volley_t) / m.where(shooter, reload, reload * 0 + 1), 0, 1)
    ready = m.where(shooter, m.where(volley_t >= 0, since, since * 0 + 1), since * 0)
    prev = m.where(known, a, m.where(sees, memory.prev_ammo, memory.prev_ammo * 0 - 1))
    return _f(m, ready), prev, volley_t


def _order(m, state, memory, tn, like):
    """(the ORDER columns [B, N] each, (ord_kind, ord_tgt, ord_x, ord_z, ord_t) for the Memory): module doc."""
    zero = like * 0
    old = [zero - 1, zero - 1, zero, zero, zero - 1] if memory.ord_kind is None else         [memory.ord_kind, memory.ord_tgt, memory.ord_x, memory.ord_z, memory.ord_t]
    k = state.get("order_kind")
    if k is None:
        return [zero] * len(ORDER), old
    k = m.nan_to_num(k * 1.0, nan=-1.0)
    k = m.where(k == k, k, zero - 1)
    tgt = state.get("order_target")
    tgt = zero - 1 if tgt is None else m.nan_to_num(tgt * 1.0, nan=-1.0)
    ox, oz = _f(m, state["ox"]), _f(m, state["oz"])
    o_kind, o_tgt, o_x, o_z, o_t = old
    point = (k == MOVE) | (k == WITHDRAW)
    moved = point & ((ox - o_x) ** 2 + (oz - o_z) ** 2 > ORDER_MOVE_M ** 2)
    changed = (k != o_kind) | ((k == ATTACK) & (tgt != o_tgt)) | moved | (o_t < 0)
    o_t = m.where(changed, tn, o_t)
    known = k >= 0
    age = m.where(known, m.clip((tn - o_t) / ORDER_AGE, 0, 1), zero)
    cols = [_f(m, k == c) for c in (HOLD, MOVE, ATTACK, WITHDRAW)] + [_f(m, age)]
    keep = changed | ~point
    return cols, [k, tgt, m.where(keep, ox, o_x), m.where(keep, oz, o_z), o_t]


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
