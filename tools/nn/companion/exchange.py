"""The files the game and the companion exchange (docs/en/apps/bridge.md). Numpy only, no torch.

    game  -> tww3_bai_nn_state.json   every decision: move number, battle time, every unit's row
             (with fx: the phases active on it), abilities_used: {unit: {ability: ms of its last use}}
    companion -> tww3_bai_nn_orders.txt   the answer for that move: one line per own unit, and a line
             'ability <unit> <key>' per ability to use now

Both files are written to a temp file first and then renamed, so a reader never sees half of one.
The orders file ends with a line 'end': the game takes nothing without it.
"""
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.nn.model import abilities as model_abilities
from tools.nn.model import passport
from tools.nn.model.observation import Setup
from tools.nn.model.sources import CROSSROADS, EMPIRE, SKAVEN
from tools.nn.sim.abilities import SLOTS, slot_keys
from tools.nn.sim.orders import KINDS   # hold, move, attack, withdraw, keep (codes 0-4)

STATE = "tww3_bai_nn_state.json"
ORDERS = "tww3_bai_nn_orders.txt"
# The enemy side under a script in the companion (nn-arena --enemy-ai ai_like; tools/nn/companion/script.py):
# the bridge's second instance, side 2, writes and reads these.
ENEMY_STATE = "tww3_bai_nn_state_enemy.json"
ENEMY_ORDERS = "tww3_bai_nn_orders_enemy.txt"
FORMAT = "tww3_bai_nn_orders 1"
FLOAT_FIELDS = ("x", "z", "b", "men", "hp", "mp", "ms", "a", "k", "ox", "oz")
BOOL_FIELDS = ("r", "s", "w", "m", "mv", "f", "fire", "lf", "rf", "bf")
FATIGUE_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                  "threshold_very_tired", "threshold_exhausted")


def read_state(path):
    """The state document, or None while the file is missing or not complete."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        doc = json.loads(text)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or "move" not in doc or "batch" not in doc:
        return None
    return doc


@dataclass
class Battle:
    """What stays the same for a battle: unit names in the state's order, and the Setup."""
    batch: str
    names: list
    side: np.ndarray
    setup: Setup
    slots: list = None     # per unit: its ability keys by slot ("" empty), as the network's input has them
    walk: np.ndarray = None   # [N] walk speed (m/s, passport; 0 unknown): running_by_speed
    shape: dict = None     # the formations as the simulator lays them out (formation_shape): engaged_targets' contact
    threat: dict = None    # the simulator's flank / rear threat rule (threat_rule): threat_flags

    @property
    def own(self):
        return [n for n, s in zip(self.names, self.side) if s == 1]


def battle(doc, bounds=CROSSROADS):
    """The Battle of a state document: setup from unit keys, sides, factions and who attacks."""
    units = doc["units"]
    names = [u["n"] for u in units]
    side = np.array([[int(u["side"]) for u in units]])
    factions = doc.get("factions") or {}
    setup = Setup(keys=[[u.get("key") or "" for u in units]], side=side,
                  bounds=np.asarray([bounds], np.float32),
                  factions=[(factions.get("own", EMPIRE), factions.get("enemy", SKAVEN))],
                  attacker=np.array([int(doc.get("attacker", 2))]))
    units_db, passports = passport.load(), model_abilities.load()
    slots = [slot_keys(u.get("key") or "", units_db, passports) if u.get("key") in units_db else [""] * SLOTS
             for u in units]
    walk = np.array([float(((units_db.get(u.get("key") or "") or {}).get("speed") or {}).get("walk") or 0.0)
                     for u in units])
    return Battle(doc["batch"], names, side[0], setup, slots, walk, formation_shape(doc), threat_rule())


def formation_shape(doc, params=None):
    """{width, sp_h, sp_v, men0, radius, range: [N] arrays; reach_m, lord_reach_m, hold_m, spacing; exit: the melee
    exit's rules} - each unit's formation as the simulator's scenario.build lays it out (the passport's STATIC fields;
    the width of the state's layout when it has one, as the enemy's script builds its simulator state,
    tools/nn/companion/script.py), its missile range (0: none) and the simulator's contact distances and melee exit
    (config/nn/sim.json contact; the database's melee_breakoff_secs: Breakoff); a key the simulator does not know: NaN
    (it touches nobody)."""
    from tools.nn.sim.params import load
    params = params or load()
    layout = {u["n"]: u for u in doc.get("layout") or ()}
    out = {k: np.full(len(doc["units"]), np.nan) for k in ("width", "sp_h", "sp_v", "men0", "radius", "range")}
    for i, u in enumerate(doc["units"]):
        key = u.get("key") or ""
        if key not in params.units:
            continue
        row = params.static(key)
        width = (layout.get(u["n"]) or {}).get("width")
        if width and row["men0"] > 1:
            row["width"] = float(width)
        for k in out:
            out[k][i] = float(row[k])
    c = params.sim["contact"]
    out.update(reach_m=float(c["reach_m"]), lord_reach_m=float(c.get("lord_reach_m", c["reach_m"])),
               hold_m=float(c["hold_m"]), spacing=float(params.sim["formation"]["spacing_m"]))
    leave_m = float(c.get("leave_m", 0.0))
    out["exit"] = {"window_s": float(params.battle.get("melee_breakoff_secs", 0.0)) if c.get("breakoff") else 0.0,
                   "leave_m": leave_m, "away_only": bool(c.get("leave_away_only")),
                   "away_m": float(c.get("leave_away_m", leave_m)), "move_melee": bool(c.get("move_melee_leave")),
                   "latch": bool(c.get("leave_latch")), "attack": bool(c.get("attack_leave"))}
    return out


def gaps(state, shape):
    """([N, N] the gap between the edges of unit i's and unit j's formations, m (negative: overlap), [N, N] the gap
    at which they touch) as the simulator measures them (geometry.pairwise, battle.py `touch`): rectangles of
    geometry.dims (files = floor(width / h), at most the men; ranks = ceil(men / files); a lone man 2 x radius);
    contact.reach_m (a lone man lord_reach_m), hold_m more when either is in melee (state m)."""
    x, z = state["x"][0].astype(float), state["z"][0].astype(float)
    b = np.radians(np.nan_to_num(state["b"][0].astype(float)))
    men = np.nan_to_num(state["men"][0].astype(float))
    h = np.where(shape["sp_h"] > 0, shape["sp_h"], shape["spacing"])
    v = np.where(shape["sp_v"] > 0, shape["sp_v"], shape["spacing"])
    with np.errstate(invalid="ignore"):
        files = np.minimum(np.maximum(np.floor(shape["width"] / h + 1e-4), 1), np.maximum(men, 1))
        ranks = np.ceil(np.maximum(men, 1) / files)
    single = shape["men0"] <= 1
    front = np.where(single, 2 * shape["radius"], files * h)
    depth = np.where(single, 2 * shape["radius"], ranks * v)
    dx, dz = x[None, :] - x[:, None], z[None, :] - z[:, None]
    dist = np.sqrt(dx * dx + dz * dz + 1e-9)
    theta = np.arctan2(dx, dz)

    def wrap(a):
        return np.remainder(a + math.pi, 2 * math.pi) - math.pi

    def half(f, d, phi):
        s, c = np.maximum(np.abs(np.sin(phi)), 1e-6), np.maximum(np.abs(np.cos(phi)), 1e-6)
        return np.minimum((f / 2) / s, (d / 2) / c)
    ext = half(front[:, None], depth[:, None], wrap(theta - b[:, None])) + \
        half(front[None, :], depth[None, :], wrap(theta + math.pi - b[None, :]))
    reach = np.where(single[:, None] | single[None, :], shape["lord_reach_m"], shape["reach_m"])
    m = np.asarray(state["m"][0], dtype=bool)
    held = (m[:, None] | m[None, :]) * shape["hold_m"]
    return dist - ext, reach + held


def touching(state, shape):
    """[N, N] unit i touches enemy j as in the simulator (tools/nn/sim/battle.py `touch`, geometry.pairwise): the edges
    of the two formations' rectangles (gaps) within contact.reach_m of each other (an overlap of 2.5 m; a lone man
    lord_reach_m), hold_m more when either is in melee (state m); both alive (men > 0, a position) on different
    sides. shape: formation_shape with `side` [N]."""
    x, z = state["x"][0].astype(float), state["z"][0].astype(float)
    men = np.nan_to_num(state["men"][0].astype(float))
    side = np.asarray(shape["side"])
    gap, at = gaps(state, shape)
    alive = (men > 0) & np.isfinite(x) & np.isfinite(z)
    enemy = (side[:, None] != side[None, :]) & (side[:, None] > 0) & (side[None, :] > 0)
    with np.errstate(invalid="ignore"):
        return enemy & alive[:, None] & alive[None, :] & (gap <= at)


def threat_rule(params=None):
    """The simulator's flank / rear threat rule (tools/nn/sim/battle.py threat_flags, config/nn/sim.json threat):
    {radius m, front deg, rear deg, facing deg or None (the enemy need not face the unit)}."""
    from tools.nn.sim.params import load
    cal = (params or load()).sim["threat"]
    trial = cal.get("calibration", {})
    if trial.get("on", False):
        return {"radius": float(trial["range_m"]), "front": float(trial["front_deg"]),
                "rear": float(trial["rear_deg"]), "facing": float(trial["facing_deg"])}
    return {"radius": float(cal["range_m"]), "front": 60.0, "rear": 120.0, "facing": None}


def threat_flags(state, side, rule=None):
    """lf / rf / bf (the network's threat_left / right / rear) as the simulator sets them, in place in `state`
    (exchange.arrays), from the game's positions and bearings (tools/nn/sim/battle.py threat_flags with rule:
    threat_rule): a standing enemy (men, a position, not routing or shattered) within `radius` m centre to centre,
    facing the unit (the unit within `facing` deg of its front), seen from the unit between front and rear deg off
    its facing on the left (lf) or right (rf), or rear deg or more off it (bf); every unit with men and a position
    (routers too), both sides.

    The game's own flags (the bridge reads them from the unit card) mean something else: on for the nearest enemy in
    a ~40 m sector whether it faces the unit or not, none in the sector in 12-27 % of them. On the same positions the
    simulator's rule held in 17-47 % of the game's flags (build/audit_in/table.md); the network learned the rule."""
    rule = rule or threat_rule()
    x, z = state["x"][0].astype(float), state["z"][0].astype(float)
    b = np.radians(state["b"][0].astype(float))
    men = np.nan_to_num(state["men"][0].astype(float))
    side = np.asarray(side)
    alive = (side > 0) & (men > 0) & np.isfinite(x) & np.isfinite(z)
    standing = alive & ~(np.asarray(state["r"][0], dtype=bool) | np.asarray(state["s"][0], dtype=bool))
    dx, dz = x[None, :] - x[:, None], z[None, :] - z[:, None]
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(dx * dx + dz * dz + 1e-9)
        theta = np.arctan2(dx, dz)
        rel = np.degrees(np.remainder(theta - b[:, None] + math.pi, 2 * math.pi) - math.pi)
        rel_j = np.degrees(np.remainder(theta + math.pi - b[None, :] + math.pi, 2 * math.pi) - math.pi)
        enemy = (side[:, None] != side[None, :]) & (side[:, None] > 0) & (side[None, :] > 0)
        threat = enemy & standing[None, :] & (dist <= rule["radius"])
        if rule.get("facing") is not None:
            threat = threat & (np.abs(rel_j) <= rule["facing"])
        front, rear = rule["front"], rule["rear"]
        state["lf"][0] = alive & (threat & (rel < -front) & (rel > -rear)).any(1)
        state["rf"][0] = alive & (threat & (rel > front) & (rel < rear)).any(1)
        state["bf"][0] = alive & (threat & (np.abs(rel) >= rear)).any(1)
    return state


def keep_still(memory, state, prev_mv):
    """The network's speed input (vel_fwd / vel_lat: the centre's shift between two decisions, observation.py) as the
    simulator has it: a unit's centre moves there only while the unit moves (battle.py: a unit with no point, held in
    melee or arrived has no velocity; mv = moving), so a unit the game shows not moving (mv off) at this state and
    at the previous one (prev_mv [N], None: the battle's first) gets no shift - its memory's previous position is
    set to where it is now (in place; memory: the observation's Memory, None: nothing to do). Returns this state's
    mv for the next call.

    The game's centre of a formation that stands drifts: in melee 0.5-0.7 m/s median (> 0.3 m/s in 58-63 % of the
    seconds; the simulator's twin 0, 17-21 %), standing out of melee > 0.5 m/s in 28 % of the seconds (twin 5.5 %;
    build/audit_in/table.md). A unit that moved at either end keeps its measured shift (it really walked)."""
    mv = np.asarray(state["mv"][0], dtype=bool).copy()
    if memory is not None and prev_mv is not None and len(prev_mv) == len(mv):
        x, z = state["x"][0].astype(float), state["z"][0].astype(float)
        still = ~mv & ~np.asarray(prev_mv, dtype=bool) & np.isfinite(x) & np.isfinite(z)
        px, pz = np.asarray(memory.prev_x), np.asarray(memory.prev_z)     # the observation's own precision
        memory.prev_x = np.where(still[None], x[None].astype(px.dtype), px)
        memory.prev_z = np.where(still[None], z[None].astype(pz.dtype), pz)
    return mv


def effects_on(state, doc, names, setup, catalogue=None, passports=None):
    """The innate effects on now (`fx_on`, the bitmask the simulator keeps: tools/nn/model/effects.py), in place in
    `state`: a timed effect (Strength of the Penitent: the game fires it itself, 20 s, the simulator keeps its timer)
    is on while the unit's card shows its phase (row fx; not read: off); the others as the observation works them
    out from the state without a bitmask (effects.active: health, morale state, melee - the card agrees 99.3-99.8 %).
    Without it the companion never showed a timed effect: the flagellants' card showed it in 21-53 % of their
    seconds (build/audit_in/table.md)."""
    from tools.nn.model import effects
    cat = catalogue or effects.load()
    passports = passports or model_abilities.load()
    have = np.asarray(setup.fx_owned) > 0.5                       # [1, N, E]
    on = np.array(effects.active(np, {k: v for k, v in state.items() if k != "fx_on"}, have.astype(np.float32), cat))
    rows = {u["n"]: u for u in doc["units"]}
    for e, key in enumerate(effects.keys(cat)):
        if not (cat["effects"].get(key) or {}).get("timed"):
            continue
        phases = set((passports.get(key) or {}).get("phases") or (key,))
        for i, name in enumerate(names):
            fx = _effects(rows.get(name, {}))
            on[0, i, e] = bool(have[0, i, e]) and fx is not None and bool(phases & fx)
    state["fx_on"] = (on.astype(np.int64) << np.arange(on.shape[-1], dtype=np.int64)).sum(-1)
    return state


RUN_MARGIN = 0.3   # m/s: running = moving faster than the walk + this (the simulator's `f`, tools/nn/sim/battle.py)


def running_by_speed(state, walk, prev=None):
    """The `running` input as the simulator has it: moving (mv) and faster than the unit's walk +
    RUN_MARGIN, the speed measured between the previous state (prev: (x, z, t) of the last call)
    and this one; in place in `state` (exchange.arrays). Returns the reference for the next call.

    The game's `f` is unit:is_moving_fast(), the unit's run mode, not its speed: in the gate of
    03.10.2026 it was on in 74 % of the unit-seconds a unit stood (< 0.3 m/s) in melee and in 99 % of
    the game AI's melee seconds; the simulator's `f` is on in 3-4 % of melee seconds. Measured by speed
    the game AI's units in melee run 9 % of the time (docs/en/apps/bridge.md "Running"). Without a
    previous state (the battle's first) or with a unit's position unknown, it is not running."""
    x, z, t = state["x"][0].astype(float), state["z"][0].astype(float), float(state["t"][0])
    run = np.zeros(x.shape, dtype=bool)
    same = prev is not None and len(prev[0]) == len(x)
    if same and t > prev[2]:
        speed = np.hypot(x - prev[0], z - prev[1]) / (t - prev[2])
        with np.errstate(invalid="ignore"):
            run = np.asarray(state["mv"][0], dtype=bool) & np.isfinite(speed) & (speed > np.asarray(walk) + RUN_MARGIN)
    state["f"][0] = run
    if same and t <= prev[2]:
        return prev            # the same moment again: the reference stays
    return x.copy(), z.copy(), t


def engaged_targets(state, side, own=1, shape=None):
    """Own units' `target` as the simulator has it (tools/nn/sim/battle.py: the enemy fought or shot at
    now, -1 otherwise), in place in `state` (exchange.arrays); the observation's has_target input.

    The game's `t` is the engine's target: an attack order's target from the moment it is given (free
    units under an attack: 99 % with a target), and mostly none for a unit fighting under a hold (26 %)
    or a move (0 %). In the simulator a unit has a target exactly while it is in melee or shooting.
    Fed the game's, the network saw 'in melee, no target' (never in training) and 'free, target' (gate
    of 03.10.2026, build/gap2/hastarget.py); the same shift in the simulator (build/gap2/sim2.py
    --gametarget) lowered its trade against ai_like by 0.105 a battle and moved its orders to the
    game's mix (hold 0.73 -> 0.63, attack 0.17 -> 0.24, move 0.10 -> 0.13; game 0.61 / 0.23 / 0.16).
    So: an own unit in melee takes the simulator's opponent - the nearest (centre to centre) standing enemy it
    touches (touching, with shape: Battle.shape / formation_shape), as tools/nn/sim/battle.py `opp`; touching none
    by that geometry while the game says melee (the game's fronts reach further), the engine's target when it is a
    present enemy, else the nearest present enemy (the simulator never shows 'in melee, no target'); a firing unit
    keeps the engine's target; any other own unit has none. Before (no shape), a unit in melee kept the engine's
    target whenever present: in the game that is the attack ORDER's target, kept ~25 s, not the enemy it fights - the
    enemy's script then saw its unit 'fighting' our shooter it was ordered onto and kept that order (62 % of re-orders
    held 5 s, the twin 7 %; build/bench2/o_revert_p1.txt), and the network saw the same shift in its has_target.
    Enemy rows are left as read (has_target is an own-only input). own: the side whose rows are set
    (1 the network's; the enemy's script sets side 2's too, tools/nn/companion/script.py)."""
    tg = state["target"][0]
    x, z, men = state["x"][0], state["z"][0], state["men"][0]
    enemy = (np.asarray(side) != own) & (men > 0) & np.isfinite(x) & np.isfinite(z)
    opp = np.full(len(tg), -1)
    if shape is not None:
        standing = enemy & ~np.asarray(state["r"][0], dtype=bool) & ~np.asarray(state["s"][0], dtype=bool)
        touch = touching(state, dict(shape, side=np.asarray(side))) & standing[None, :]
        d = np.where(touch, np.hypot(x[None, :] - x[:, None], z[None, :] - z[:, None]), np.inf)
        opp = np.where(touch.any(1), d.argmin(1), -1)
    for i in np.nonzero(np.asarray(side) == own)[0]:
        j = int(tg[i])
        valid = 0 <= j < len(tg) and bool(enemy[j])
        if state["m"][0, i]:
            if opp[i] >= 0:
                j, valid = int(opp[i]), True
            elif not valid and enemy.any() and np.isfinite(x[i]) and np.isfinite(z[i]):
                d = np.where(enemy, np.hypot(x - x[i], z - z[i]), np.inf)
                j, valid = int(d.argmin()), True
            tg[i] = j if valid else -1
        elif state["fire"][0, i]:
            tg[i] = j if valid else -1
        else:
            tg[i] = -1
    return state


def arrays(doc, names, slots=None):
    """The state for tools/nn/model/observation.py: dict of arrays [1, N] (NaN: not read), t [1] in s;
    with slots (Battle.slots) also the abilities' timers (ability_timers)."""
    N = len(names)
    index = {n: i for i, n in enumerate(names)}
    out = {k: np.full((1, N), np.nan) for k in FLOAT_FIELDS}
    out.update({k: np.zeros((1, N), dtype=bool) for k in BOOL_FIELDS})
    out["fat"] = np.full((1, N), np.nan)
    out["target"] = np.full((1, N), -1, dtype=np.int64)
    out["order_kind"] = np.full((1, N), -1, dtype=np.int64)     # own units: order_points
    out["order_target"] = np.full((1, N), -1, dtype=np.int64)
    out["vis"] = np.ones((1, N), dtype=bool)
    fatigue = {name: float(i) for i, name in enumerate(FATIGUE_LEVELS)}
    for u in doc["units"]:
        i = index[u["n"]]
        for k in FLOAT_FIELDS:
            v = u.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k][0, i] = v
        for k in BOOL_FIELDS:
            out[k][0, i] = bool(u.get(k))
        out["target"][0, i] = index.get(u.get("t") or "", -1)
        out["fat"][0, i] = fatigue.get(u.get("fat"), np.nan)
        if u.get("v") is False:
            out["vis"][0, i] = False
    out["men"] = np.nan_to_num(out["men"], nan=0.0)
    out["t"] = np.array([doc.get("t", 0) / 1000.0])
    if slots is not None:
        out.update(ability_timers(doc, names, slots))
    return out


def order_points(state, names, side, given, last=None, own=1):
    """The order point (ox, oz) of own units as the simulator reports it (tools/nn/sim/battle.py), in
    place in `state` (exchange.arrays); returns the points kept for the next call (`last`).

    The network was trained on the simulator's point: a holding unit's own place, an attacking unit's
    target, a move / withdraw point. The game's ordered_position is a move's point too, but ~10 m
    from the unit itself for a hold or an attack (gate it2, 8 battles: 28 309 attack samples, median
    9.8 m from the unit, 13.5 m from the target). Fed that, the network does not recognise the order
    in force: the same network changed orders 15.7 times a unit-minute in game vs 5.8 in the
    simulator (docs/en/apps/bridge.md "Order point"). given: {name: the last order the companion gave
    the unit, not keep} (orders_list's dicts); a unit without one holds. An attack on a target that
    is gone (no men) holds, as the simulator turns it to HOLD. Routing or shattered units keep the
    point they had (the simulator does not update them). Enemy rows are left as read. Also the order in force
    as the simulator keeps it (`order_kind` code, `order_target` slot; the observation's ORDER input): the last
    order given, hold without one; an attack whose target is gone holds. own: the side whose rows are set
    (1 the network's; tools/nn/companion/script.py sets both sides')."""
    last = dict(last or {})
    index = {n: i for i, n in enumerate(names)}
    x, z, men = state["x"][0], state["z"][0], state["men"][0]
    for i, name in enumerate(names):
        if side[i] != own:
            continue
        o = given.get(name) or {}
        code = KINDS.index(o["kind"]) if o.get("kind") in KINDS[:4] else 0
        j = index.get(o.get("target"), -1) if code == 2 else -1
        if code == 2 and not (j >= 0 and men[j] > 0):
            code, j = 0, -1
        if "order_kind" in state:
            state["order_kind"][0, i], state["order_target"][0, i] = code, j
        if (state["r"][0, i] or state["s"][0, i]) and name in last:
            state["ox"][0, i], state["oz"][0, i] = last[name]
            continue
        o = given.get(name) or {}
        k = o.get("kind")
        px, pz = x[i], z[i]
        if k in ("move", "withdraw"):
            px, pz = o["x"], o["z"]
        elif k == "attack":
            j = index.get(o.get("target"), -1)
            if j >= 0 and men[j] > 0 and np.isfinite(x[j]) and np.isfinite(z[j]):
                px, pz = x[j], z[j]
        state["ox"][0, i], state["oz"][0, i] = px, pz
        last[name] = (px, pz)
    return last


class Breakoff:
    """The melee exit's window as the simulator keeps it (tools/nn/sim/battle.py contact.breakoff, the database's
    melee_breakoff_secs 24 s): a unit leaving melee that still touches a standing enemy that long after it began to
    leave drops its order and holds (it fights again). The companion keeps the same clocks over the game's states and
    turns the order in force (given, as order_points reads it) to hold, so the network (its ORDER input, ox / oz) and
    the enemy's script see what they saw in training; the game's engine drops the order by the same rule. Before, the
    order in force stayed an attack on a target > 40 m away after >= 24 s in melee: ~1 % of the attack seconds,
    both sides (build/audit_in/table.md).

    Leaving (battle.py, config/nn/sim.json contact, the switches as set there): a withdraw; a move whose point lies
    away from every enemy it touches (leave_away_only, from leave_away_m on), or any move given in melee within the
    window (move_melee_leave: the order's clock order_s, 0 for an order given in melee, the window for one given
    out of melee), or a move that began as a leave while it stands (leave_latch); an attack on an enemy it does not
    touch, leave_m or more away, by a formation without a missile weapon that fights (attack_leave). In contact:
    the game's melee flag (m) on a standing unit with more than one man (the simulator's `engaged`; touching by the
    simulator's geometry, gaps, only says which enemy). exit_s runs while it leaves in contact, and on out of
    contact while it keeps leaving; at the window in contact the order is dropped. One clock per decision: the time
    between the game's states."""

    def __init__(self, names, side, shape, own=1):
        self.names, self.side, self.own = list(names), np.asarray(side), own
        self.shape = dict(shape, side=np.asarray(side))
        self.rule = shape["exit"]
        N = len(self.names)
        self.exit_s = np.zeros(N)
        self.order_s = np.full(N, self.rule["window_s"])
        self.last, self.t, self.m_prev = {}, None, np.zeros(N, dtype=bool)

    def update(self, state, given):
        """The clocks to this state (exchange.arrays, the game's m, positions, men); orders in force whose window is
        over become hold in `given` (in place). Returns the names whose order was dropped."""
        rule, window = self.rule, self.rule["window_s"]
        t = float(state["t"][0])
        if window <= 0 or (self.t is not None and t <= self.t):
            return []
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        index = {n: i for i, n in enumerate(self.names)}
        x, z = state["x"][0].astype(float), state["z"][0].astype(float)
        men = np.nan_to_num(state["men"][0].astype(float))
        m = np.asarray(state["m"][0], dtype=bool)
        alive = (men > 0) & np.isfinite(x) & np.isfinite(z)
        standing = alive & ~(np.asarray(state["r"][0], dtype=bool) | np.asarray(state["s"][0], dtype=bool))
        touch = touching(state, self.shape)
        gap, _ = gaps(state, self.shape)
        formation = np.nan_to_num(self.shape["men0"]) > 1
        dropped = []
        for i, name in enumerate(self.names):
            if self.side[i] != self.own:
                continue
            o = given.get(name) or {}
            kind = o.get("kind") if o.get("kind") in ("move", "attack", "withdraw") else "hold"
            j = index.get(o.get("target"), -1) if kind == "attack" else -1
            if kind == "attack" and not (j >= 0 and men[j] > 0):          # gone: holds (order_points)
                kind, j = "hold", -1
            px, pz = (float(o["x"]), float(o["z"])) if kind in ("move", "withdraw") else (x[i], z[i])
            if kind == "attack":
                px, pz = x[j], z[j]
            before = self.last.get(name)
            new = before is None and kind != "hold" or before is not None and (
                before[0] != kind or before[1] != j or (kind in ("move", "withdraw") and
                                                         math.hypot(px - before[2], pz - before[3]) > 1.0))
            self.last[name] = (kind, j, px, pz)
            # the order's clock: a new order given in melee starts at 0, out of melee at the window
            self.order_s[i] = (0.0 if self.m_prev[i] else window) + dt if new else self.order_s[i] + dt
            engaged = bool(standing[i] and m[i])
            leaving = kind == "withdraw"
            if kind == "move" and rule["leave_m"] > 0:
                point_d = math.hypot(px - x[i], pz - z[i])
                if rule["away_only"]:
                    toward = ((px - x[i]) * (x - x[i]) + (pz - z[i]) * (z - z[i])) > 0
                    far = not (touch[i] & toward).any() and point_d >= rule["away_m"]
                    if rule["move_melee"]:
                        far = far or (self.order_s[i] < window and point_d >= rule["away_m"])
                else:
                    far = point_d >= rule["leave_m"]
                leaving = far or (rule["latch"] and self.exit_s[i] > 0 and not new)
            if kind == "attack" and rule["attack"] and rule["leave_m"] > 0:
                away = not touch[i, j] and gap[i, j] >= rule["leave_m"]
                leaving = bool(away and engaged and not np.nan_to_num(self.shape["range"][i]) > 0 and formation[i])
            in_exit = leaving and engaged and bool(formation[i])
            self.exit_s[i] = self.exit_s[i] + dt if leaving and (in_exit or self.exit_s[i] > 0) else 0.0
            if in_exit and self.exit_s[i] >= window:
                given[name] = {"unit": name, "kind": "hold"}
                self.last[name] = ("hold", -1, x[i], z[i])
                self.exit_s[i] = 0.0
                dropped.append(name)
        self.m_prev = m.copy()
        return dropped


def remember_orders(given, orders, ctrl_names):
    """The orders in force after an answer: given updated in place with every order to a unit that
    takes orders (ctrl_names), except keep (the order in force goes on). Orders to routing units are
    not given in the game (nor in the simulator)."""
    for o in orders:
        if o["kind"] != "keep" and o["unit"] in ctrl_names:
            given[o["unit"]] = o
    return given


def _effects(row):
    """The phases active on a unit (row fx; JSON gives an empty list as {}), or None when not read."""
    fx = row.get("fx")
    if fx is None:
        return None
    return set(fx) if isinstance(fx, list) else set()


TAKE_S = 1.5   # s after a use by which the card shows its phase (in game: by the next state, <= 1 s)


def ability_timers(doc, names, slots, passports=None, own=1):
    """ab{k}_on / ab{k}_cd [1, N] (s) for tools/nn/model/observation.py.

    Own units (side 1): from the bridge's last use of each ability (abilities_used) and its passport:
    active for active_s, then ready again recharge_s later (matched the game to ~1 s, 01.10.2026).
    The unit's card (fx) is trusted over that count: a use whose phase is not on the unit TAKE_S
    after it did not take (ready again: on and cd 0; the game's can_perform_special_ability only says
    the lord owns it, so the bridge cannot refuse a use in recharge); a phase on the unit with no use
    known counts 1 s active. Enemies: only whether it is active now (fx: the game shows it on the
    unit; the observation keeps it only while the unit is seen): `on` 1 s, timers 0. own: the side whose
    uses the bridge counts (1; the enemy's script: 2, its own bridge's abilities_used)."""
    passports = passports or model_abilities.load()
    N = len(names)
    out = {f"ab{k}_{t}": np.zeros((1, N)) for k in range(SLOTS) for t in ("on", "cd")}
    t = doc.get("t", 0) / 1000.0
    used = doc.get("abilities_used") or {}
    rows = {u["n"]: u for u in doc["units"]}
    for i, name in enumerate(names):
        row = rows.get(name, {})
        fx = _effects(row)
        mine = int(row.get("side", 0)) == own
        last = used.get(name) if isinstance(used.get(name), dict) else {}
        for k, key in enumerate(slots[i]):
            if not key or key not in passports:
                continue
            p = passports[key]
            seen_on = fx is not None and any(ph in fx for ph in p.get("phases") or ())
            on = cd = 0.0
            if mine and key in last and not p.get("passive"):
                since = t - float(last[key]) / 1000.0
                active, recharge = max(float(p["active_s"]), 0.0), max(float(p["recharge_s"]), 0.0)
                on, cd = max(0.0, active - since), max(0.0, active + recharge - since)
                if fx is not None and not seen_on and since >= TAKE_S and on > 0:
                    on = cd = 0.0                      # the card does not show it: it did not take
            if seen_on and on <= 0 and not p.get("passive"):
                on = 1.0
                cd = max(cd, on)
            out[f"ab{k}_on"][0, i], out[f"ab{k}_cd"][0, i] = on, cd
    return out


def orders_list(names, side, kind, x, z, target, run, own=1):
    """The network's orders [N] (numpy, the simulator's layout) -> one dict per unit of side `own`."""
    out = []
    for i, name in enumerate(names):
        if side[i] != own:
            continue
        k = KINDS[int(kind[i])]
        o = {"unit": name, "kind": k}
        if k in ("move", "withdraw"):
            o.update(x=round(float(x[i]), 1), z=round(float(z[i]), 1), run=bool(run[i]) or k == "withdraw")
        elif k == "attack":
            t = int(target[i])
            if t < 0 or t >= len(names):
                o = {"unit": name, "kind": "hold"}
            else:
                o.update(target=names[t], run=bool(run[i]))
        out.append(o)
    return out


def ability_list(names, side, ability, slots):
    """The network's ability choice [N] (slot, -1 none) -> [{unit, key}] for own units."""
    out = []
    for i, name in enumerate(names):
        k = int(ability[i])
        if side[i] == 1 and 0 <= k < len(slots[i]) and slots[i][k]:
            out.append({"unit": name, "key": slots[i][k]})
    return out


def orders_text(batch, move, orders, think_ms=None, abilities=()):
    """The orders file for the game (parsed by src/apps/bridge/services.lua). abilities: [{unit, key}]
    to use now, one line each. A unit that takes no orders (marked "out" by the loop: dead, routing,
    shattered) gets no line: the network's HOLD for it is only a filler, and given to a unit that
    rallied before the answer came it halted it (gate 02.10.2026: 42 of 102 rallies; the bridge
    also ignores it, docs/en/apps/bridge.md)."""
    lines = [FORMAT, f"move {int(move)}", f"batch {batch}"]
    if think_ms is not None:
        lines.append(f"think_ms {think_ms:.1f}")
    for o in orders:
        if o.get("out"):
            continue
        k = o["kind"]
        if k in ("move", "withdraw"):
            if not (math.isfinite(o["x"]) and math.isfinite(o["z"])):
                raise ValueError(f"order point not finite: {o}")
            lines.append(f"unit {o['unit']} {k} {o['x']:.1f} {o['z']:.1f} {int(bool(o['run']))}")
        elif k == "attack":
            lines.append(f"unit {o['unit']} attack {o['target']} {int(bool(o['run']))}")
        elif k in ("hold", "keep"):
            lines.append(f"unit {o['unit']} {k}")
        else:
            raise ValueError(f"unknown order kind: {k}")
    for a in abilities:
        if not a["unit"] or not a["key"] or any(c.isspace() for c in a["unit"] + a["key"]):
            raise ValueError(f"bad ability line: {a}")
        lines.append(f"ability {a['unit']} {a['key']}")
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_orders(text):
    """The orders file -> {move, batch, think_ms, orders: {unit: order}} (tests and tools; the game
    has its own parser). None when the file is not complete."""
    lines = text.splitlines()
    if not lines or lines[0] != FORMAT or lines[-1] != "end":
        return None
    doc = {"orders": {}, "abilities": []}
    for line in lines[1:-1]:
        w = line.split()
        if w[0] == "move":
            doc["move"] = int(w[1])
        elif w[0] == "batch":
            doc["batch"] = w[1]
        elif w[0] == "think_ms":
            doc["think_ms"] = float(w[1])
        elif w[0] == "unit":
            o = {"kind": w[2]}
            if w[2] in ("move", "withdraw"):
                o.update(x=float(w[3]), z=float(w[4]), run=w[5] == "1")
            elif w[2] == "attack":
                o.update(target=w[3], run=w[4] == "1")
            doc["orders"][w[1]] = o
        elif w[0] == "ability":
            doc["abilities"].append({"unit": w[1], "key": w[2]})
    return doc


def write_atomic(path, text, attempts=50, wait_s=0.004):
    """Write text to a temp file and rename it over path. The game may hold the old file open for a
    moment (Windows then refuses the rename): try again. Returns the number of attempts used."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for attempt in range(1, attempts + 1):
        try:
            os.replace(tmp, path)
            return attempt
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(wait_s)
    return attempts


def summary(orders):
    """A short line: how many units got each kind of order; units that take no orders (dead, routing,
    shattered: the network gives them HOLD, the game nothing; marked "out" by the loop) count as out.
    Before 02.10.2026 they counted as hold: "hold 15 attack 5" late in a gate battle was 13 units out."""
    counts = {k: 0 for k in KINDS}
    out = 0
    for o in orders:
        if o.get("out"):
            out += 1
        else:
            counts[o["kind"]] += 1
    return " ".join([f"{k} {n}" for k, n in counts.items() if n] + ([f"out {out}"] if out else []))
