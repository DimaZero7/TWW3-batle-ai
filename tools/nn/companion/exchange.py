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
    return Battle(doc["batch"], names, side[0], setup, slots)


def arrays(doc, names, slots=None):
    """The state for tools/nn/model/observation.py: dict of arrays [1, N] (NaN: not read), t [1] in s;
    with slots (Battle.slots) also the abilities' timers (ability_timers)."""
    N = len(names)
    index = {n: i for i, n in enumerate(names)}
    out = {k: np.full((1, N), np.nan) for k in FLOAT_FIELDS}
    out.update({k: np.zeros((1, N), dtype=bool) for k in BOOL_FIELDS})
    out["fat"] = np.full((1, N), np.nan)
    out["target"] = np.full((1, N), -1, dtype=np.int64)
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
    to use now, one line each."""
    lines = [FORMAT, f"move {int(move)}", f"batch {batch}"]
    if think_ms is not None:
        lines.append(f"think_ms {think_ms:.1f}")
    for o in orders:
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
    """A short line: how many units got each kind of order."""
    counts = {k: 0 for k in KINDS}
    for o in orders:
        counts[o["kind"]] += 1
    return " ".join(f"{k} {n}" for k, n in counts.items() if n)
