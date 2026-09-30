"""The files the game and the companion exchange (docs/en/apps/bridge.md). Numpy only, no torch.

    game  -> tww3_bai_nn_state.json   every decision: move number, battle time, every unit's row
    companion -> tww3_bai_nn_orders.txt   the answer for that move: one line per own unit

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

from tools.nn.model.observation import Setup
from tools.nn.model.sources import CROSSROADS, EMPIRE, SKAVEN
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
    return Battle(doc["batch"], names, side[0], setup)


def arrays(doc, names):
    """The state for tools/nn/model/observation.py: dict of arrays [1, N] (NaN: not read), t [1] in s."""
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


def orders_text(batch, move, orders, think_ms=None):
    """The orders file for the game (parsed by src/apps/bridge/services.lua)."""
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
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_orders(text):
    """The orders file -> {move, batch, think_ms, orders: {unit: order}} (tests and tools; the game
    has its own parser). None when the file is not complete."""
    lines = text.splitlines()
    if not lines or lines[0] != FORMAT or lines[-1] != "end":
        return None
    doc = {"orders": {}}
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
