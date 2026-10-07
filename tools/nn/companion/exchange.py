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
    return Battle(doc["batch"], names, side[0], setup, slots, walk)


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


def engaged_targets(state, side):
    """Own units' `target` as the simulator has it (tools/nn/sim/battle.py: the enemy fought or shot at
    now, -1 otherwise), in place in `state` (exchange.arrays); the observation's has_target input.

    The game's `t` is the engine's target: an attack order's target from the moment it is given (free
    units under an attack: 99 % with a target), and mostly none for a unit fighting under a hold (26 %)
    or a move (0 %). In the simulator a unit has a target exactly while it is in melee or shooting.
    Fed the game's, the network saw 'in melee, no target' (never in training) and 'free, target' (gate
    of 03.10.2026, build/gap2/hastarget.py); the same shift in the simulator (build/gap2/sim2.py
    --gametarget) lowered its trade against ai_like by 0.105 a battle and moved its orders to the
    game's mix (hold 0.73 -> 0.63, attack 0.17 -> 0.24, move 0.10 -> 0.13; game 0.61 / 0.23 / 0.16).
    So: an own unit in melee keeps the engine's target when it is a present enemy, else takes the
    nearest present enemy; a firing unit keeps the engine's target; any other own unit has none.
    Enemy rows are left as read (has_target is an own-only input)."""
    tg = state["target"][0]
    x, z, men = state["x"][0], state["z"][0], state["men"][0]
    enemy = (np.asarray(side) != 1) & (men > 0) & np.isfinite(x) & np.isfinite(z)
    for i in np.nonzero(np.asarray(side) == 1)[0]:
        j = int(tg[i])
        valid = 0 <= j < len(tg) and bool(enemy[j])
        if state["m"][0, i]:
            if not valid and enemy.any() and np.isfinite(x[i]) and np.isfinite(z[i]):
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


def order_points(state, names, side, given, last=None):
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
    order given, hold without one; an attack whose target is gone holds."""
    last = dict(last or {})
    index = {n: i for i, n in enumerate(names)}
    x, z, men = state["x"][0], state["z"][0], state["men"][0]
    for i, name in enumerate(names):
        if side[i] != 1:
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


def ability_timers(doc, names, slots, passports=None):
    """ab{k}_on / ab{k}_cd [1, N] (s) for tools/nn/model/observation.py.

    Own units (side 1): from the bridge's last use of each ability (abilities_used) and its passport:
    active for active_s, then ready again recharge_s later (matched the game to ~1 s, 01.10.2026).
    The unit's card (fx) is trusted over that count: a use whose phase is not on the unit TAKE_S
    after it did not take (ready again: on and cd 0; the game's can_perform_special_ability only says
    the lord owns it, so the bridge cannot refuse a use in recharge); a phase on the unit with no use
    known counts 1 s active. Enemies: only whether it is active now (fx: the game shows it on the
    unit; the observation keeps it only while the unit is seen): `on` 1 s, timers 0."""
    passports = passports or model_abilities.load()
    N = len(names)
    out = {f"ab{k}_{t}": np.zeros((1, N)) for k in range(SLOTS) for t in ("on", "cd")}
    t = doc.get("t", 0) / 1000.0
    used = doc.get("abilities_used") or {}
    rows = {u["n"]: u for u in doc["units"]}
    for i, name in enumerate(names):
        row = rows.get(name, {})
        fx = _effects(row)
        mine = int(row.get("side", 0)) == 1
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


def orders_list(names, side, kind, x, z, target, run):
    """The network's orders [N] (numpy, the simulator's layout) -> one dict per own unit."""
    out = []
    for i, name in enumerate(names):
        if side[i] != 1:
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
