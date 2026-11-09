"""The files the game and the companion exchange (bridge/docs/apps/bridge.md). Numpy only, no torch.

    game  -> tww3_bai_nn_state.json   every decision: move number, battle time, every unit's row
             (with fx: the phases active on it), abilities_used: {unit: {ability: ms of its last use}},
             winds: the own army's Winds of Magic {cur, rate, res} (absent when the game gives none)
    companion -> tww3_bai_nn_orders.txt   the answer for that move: one line per own unit, and a line
             'ability <unit> <key> [<target>]' per ability to use now (with a target: a spell on that unit)

Both files are written to a temp file first and then renamed, so a reader never sees half of one.
The orders file ends with a line 'end': the game takes nothing without it. The format's test: the text this module
writes is parsed by the bridge's Lua (bridge/lua/apps/bridge/services.lua parse_orders; contract/README.md).
"""
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from net.cards import abilities as model_abilities
from net.cards import passport
from sim.orders import KINDS   # hold, move, attack, withdraw, keep (codes 0-4)

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
    """What stays the same for a battle: unit names and sides in the state's order, the abilities' slots."""
    batch: str
    names: list
    side: np.ndarray
    slots: list = None     # per unit: its ability keys by slot ("" empty), as the network's input has them
    #                        (net/cards/abilities.py layout: every ability; the actives' slot k is the simulator's)
    walk: np.ndarray = None   # [N] walk speed (m/s, passport; 0 unknown)

    @property
    def own(self):
        return [n for n, s in zip(self.names, self.side) if s == 1]


def battle(doc):
    """The Battle of a state document: names, sides, the abilities' slots and walk speeds by unit key."""
    units = doc["units"]
    names = [u["n"] for u in units]
    side = np.array([int(u["side"]) for u in units])
    units_db, passports = passport.load(), model_abilities.load()
    slots = [model_abilities.layout(u.get("key") or "", units_db, passports) if u.get("key") in units_db
             else [""] * model_abilities.SLOTS for u in units]
    walk = np.array([float(((units_db.get(u.get("key") or "") or {}).get("speed") or {}).get("walk") or 0.0)
                     for u in units])
    return Battle(doc["batch"], names, side, slots, walk)


RUN_MARGIN = 0.3   # m/s: running = moving faster than the walk + this (the simulator's `f`, sim/battle.py;
#                    bridge/companion/unit_input.py game_raw)


def arrays(doc, names, slots=None, own=1):
    """The document's rows as arrays [1, N] (NaN: not read), t [1] in s (unit_input.game_raw reads them);
    with slots (Battle.slots) also the abilities' timers (ability_timers; own: the side whose bridge wrote doc)."""
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
        out.update(ability_timers(doc, names, slots, own=own))
        out["ab_rx"] = received(doc, names)
    w = winds_of(doc)
    if w is not None:
        # the own side's winds: on its units (the simulator's per-unit copy of the side's pool)
        mine = np.zeros((1, N), dtype=bool)
        for u in doc["units"]:
            mine[0, index[u["n"]]] = int(u.get("side", 0)) == own
        out["wm_cur"] = np.where(mine, w[0], 0.0)
        out["wm_res"] = np.where(mine, w[1], 0.0)
    return out


def winds_of(doc, params=None):
    """(current, reserve) of the own army's winds of magic from the state document (the bridge's winds {cur, res}); the
    reserve when the game gives none: the battle's start reserve less its flow so far (sim/config/sim.json magic.winds:
    68 - 0.1 x t, the magic probe: flat while it lasts); None without winds."""
    w = doc.get("winds")
    if not isinstance(w, dict) or not isinstance(w.get("cur"), (int, float)):
        return None
    res = w.get("res")
    if not isinstance(res, (int, float)):
        from sim.params import load
        cal = ((params or load()).sim.get("magic") or {}).get("winds") or {}
        res = max(0.0, float(cal.get("reserve", 0.0)) - float(cal.get("flow_per_s", 0.0)) * doc.get("t", 0) / 1000.0)
    return float(w["cur"]), float(res)


def received(doc, names, passports=None, units_db=None):
    """[1, N] int64: the abilities of other units on each unit now - the network's effects from others (the
    simulator's ab_rx; tactic_input): a bitmask over the passports' order (net/cards/abilities.py keys) of the abilities
    whose phase the unit's card shows (row fx: the bridge's ActiveEffectList, whoever laid it) and that the unit does not
    own itself (its own are its slots: a friend's aura, Hold the Line or Rally, an enemy's on it). fx not read: none."""
    passports = passports or model_abilities.load()
    units_db = units_db or passport.load()
    keys = model_abilities.keys(passports)
    phases = [set(passports[k].get("phases") or (k,)) for k in keys]
    rows = {u["n"]: u for u in doc["units"]}
    out = np.zeros((1, len(names)), np.int64)
    for i, name in enumerate(names):
        row = rows.get(name, {})
        fx = _effects(row)
        if not fx:
            continue
        own = set((units_db.get(row.get("key") or "") or {}).get("abilities") or ())
        for a, k in enumerate(keys):
            if k not in own and phases[a] & fx:
                out[0, i] |= 1 << a
    return out


def _effects(row):
    """The phases active on a unit (row fx; JSON gives an empty list as {}), or None when not read."""
    fx = row.get("fx")
    if fx is None:
        return None
    return set(fx) if isinstance(fx, list) else set()


TAKE_S = 1.5   # s after a use by which the card shows its phase (in game: by the next state, <= 1 s)


def ability_timers(doc, names, slots, passports=None, own=1):
    """ab{k}_on / ab{k}_cd [1, N] (s): the buttons' timers (tactic_input.observe_game).

    Own units (side 1): from the bridge's last use of each ability (abilities_used) and its passport:
    active for active_s, then ready again recharge_s later (matched the game to ~1 s, 01.10.2026).
    The unit's card (fx) is trusted over that count: a use whose phase is not on the unit TAKE_S
    after it did not take (ready again: on and cd 0; the game's can_perform_special_ability only says
    the lord owns it, so the bridge cannot refuse a use in recharge); a phase on the unit with no use
    known counts 1 s active. Enemies: only whether it is active now (fx: the game shows it on the
    unit; the observation keeps it only while the unit is seen): `on` 1 s, timers 0. own: the side whose
    uses the bridge counts (1; the enemy's script: 2, its own bridge's abilities_used). A spell's slot (abilities.py
    SPELL_AT + j) also gives sp{j}_on / sp{j}_cd (its passport's active_s and recharge_s from its last cast: the
    recharge runs from the effect's end, the magic probe); the caster shows no phase of it. An ability of limited uses
    (the passport's uses > 0: Concealment Bombs; abilities_count: {unit: {ability: how many times the bridge used it}}): with
    none left the recharge is frozen at recharge_s, as the simulator's (sim/abilities.py) - not ready to the end."""
    passports = passports or {**model_abilities.load_spells(), **model_abilities.load()}
    N = len(names)
    K = max((len(s) for s in slots), default=0)
    out = {f"ab{k}_{t}": np.zeros((1, N)) for k in range(K) for t in ("on", "cd")}
    t = doc.get("t", 0) / 1000.0
    used = doc.get("abilities_used") or {}
    counted = doc.get("abilities_count") or {}
    rows = {u["n"]: u for u in doc["units"]}
    for i, name in enumerate(names):
        row = rows.get(name, {})
        fx = _effects(row)
        mine = int(row.get("side", 0)) == own
        last = used.get(name) if isinstance(used.get(name), dict) else {}
        times = counted.get(name) if isinstance(counted.get(name), dict) else {}
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
                if fx is not None and not seen_on and since >= TAKE_S and on > 0 and p.get("phases"):
                    on = cd = 0.0                      # the card does not show it: it did not take
                limit = int(p.get("uses", -1) or -1)
                if limit > 0 and int(times.get(key, 0)) >= limit:
                    cd = max(cd, recharge)             # none left: the recharge does not run out
            if seen_on and on <= 0 and not p.get("passive"):
                on = 1.0
                cd = max(cd, on)
            out[f"ab{k}_on"][0, i], out[f"ab{k}_cd"][0, i] = on, cd
    for j, k in enumerate(model_abilities.SPELL_SLOTS):
        if k < K:
            out[f"sp{j}_on"], out[f"sp{j}_cd"] = out[f"ab{k}_on"], out[f"ab{k}_cd"]
    return out


def orders_list(names, side, kind, x, z, target, run, own=1, face=None, width=None):
    """The network's orders [N] (numpy, the simulator's layout) -> one dict per unit of side `own`. face, width [N]
    (Orders.face / Orders.width: degrees of the world's bearing / m, < 0 none): a hold's, a move's or a withdraw's
    "face" / "width" when given (the unit network's formation: net/unit/decode.to_orders)."""
    out = []
    for i, name in enumerate(names):
        if side[i] != own:
            continue
        k = KINDS[int(kind[i])]
        o = {"unit": name, "kind": k}
        if k in ("hold", "move", "withdraw"):
            if face is not None and float(face[i]) >= 0:
                o["face"] = round(float(face[i]), 2)
            if width is not None and float(width[i]) >= 0:
                o["width"] = round(float(width[i]), 1)
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


def orders_text(batch, move, orders, think_ms=None, abilities=(), mode=None):
    """The orders file for the game (parsed by bridge/lua/apps/bridge/services.lua). abilities: [{unit, key,
    target (optional: the unit a spell is cast on; none: the unit itself)}] to use now, one line each. A unit that takes no orders (marked "out" by the loop: dead, routing,
    shattered) gets no line: the network's HOLD for it is only a filler, and given to a unit that
    rallied before the answer came it halted it (gate 02.10.2026: 42 of 102 rallies; the bridge
    also ignores it, bridge/docs/apps/bridge.md)."""
    lines = [FORMAT, f"move {int(move)}", f"batch {batch}"]
    if mode:
        # 'mode chain' (companion --chain): the bridge gives the orders as the simulator does - no release of an idle
        # shooter under an attack (bridge/lua/apps/bridge/services.lua missile_duty; 08.11, sim/evidence/diag_mem/plan.md
        # Н-прямой-огонь)
        lines.append(f"mode {mode}")
    if think_ms is not None:
        lines.append(f"think_ms {think_ms:.1f}")
    for o in orders:
        if o.get("out"):
            continue
        k = o["kind"]
        if k in ("move", "withdraw"):
            if not (math.isfinite(o["x"]) and math.isfinite(o["z"])):
                raise ValueError(f"order point not finite: {o}")
            line = f"unit {o['unit']} {k} {o['x']:.1f} {o['z']:.1f} {int(bool(o['run']))}"
        elif k == "attack":
            line = f"unit {o['unit']} attack {o['target']} {int(bool(o['run']))}"
        elif k in ("hold", "keep"):
            line = f"unit {o['unit']} {k}"
        else:
            raise ValueError(f"unknown order kind: {k}")
        if k in ("hold", "move", "withdraw"):
            # the formation (orders_list): "face <deg>" (the world's bearing 0..360) and / or "width <m>" at the end
            for f, fmt in (("face", "{:.2f}"), ("width", "{:.1f}")):
                if o.get(f) is not None:
                    if not (math.isfinite(o[f]) and o[f] >= 0):
                        raise ValueError(f"order {f} not finite or negative: {o}")
                    line += f" {f} " + fmt.format(o[f])
        lines.append(line)
    for a in abilities:
        target = a.get("target") or ""
        if not a["unit"] or not a["key"] or any(c.isspace() for c in a["unit"] + a["key"] + target):
            raise ValueError(f"bad ability line: {a}")
        lines.append(f"ability {a['unit']} {a['key']}" + (f" {target}" if target else ""))
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
            rest = []
            if w[2] in ("move", "withdraw"):
                o.update(x=float(w[3]), z=float(w[4]), run=w[5] == "1")
                rest = w[6:]
            elif w[2] == "attack":
                o.update(target=w[3], run=w[4] == "1")
            elif w[2] == "hold":
                rest = w[3:]
            for f, v in zip(rest[::2], rest[1::2]):
                if f in ("face", "width"):
                    o[f] = float(v)
            doc["orders"][w[1]] = o
        elif w[0] == "ability":
            doc["abilities"].append({"unit": w[1], "key": w[2], **({"target": w[3]} if len(w) > 3 else {})})
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
