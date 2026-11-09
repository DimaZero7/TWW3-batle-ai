"""A battle as frames for the vision (host python, no torch): a game recording (events.jsonl nn_sample / nn_final,
manifest.json, companion.jsonl of the chain) or the simulator's states (from_sim) -> Battle.

Battle.frames[i] = {name: unit} at second Battle.t[i]; a unit is the recording's nn_sample row (x, z, b, men, hp, r, m,
mv, f, a, fire, t, ox, oz, ow, ...). Roles from the unit passport (sim/config/units.json): lord, sn (missile >= 200 m:
long rifles, jezzails), art, sh (other missile), fast (run >= 7.5 m/s or a war beast: cavalry, wolf rats), mon, inf.
Battle.groups = [(sec, [operative group: g, units, verb, pace, cx, cz, heading, task, moved, reformed])] - what the
chain decided every 15 s (only the chain writes it; a human's battle has none).
"""
import json
import math
from pathlib import Path

import game

UNITS_JSON = game.ROOT / "sim" / "config" / "units.json"
ROLE_CODE = {"lord": "L", "sn": "S", "art": "A", "sh": "M", "fast": "F", "mon": "X", "inf": "I"}


def passports():
    return json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"]


def role_of(p):
    """The unit's role from its passport (None -> inf)."""
    if not p:
        return "inf"
    mi = p.get("missile") or {}
    run = (p.get("speed") or {}).get("run") or 0
    if p.get("caste") == "lord":
        return "lord"
    if p.get("category") == "artillery":
        return "art"
    if mi and (mi.get("range_m") or 0) >= 200:
        return "sn"
    if mi:
        return "sh"
    if run >= 7.5 or p.get("caste") == "war_beast":
        return "fast"
    if p.get("caste") in ("monster", "monstrous_infantry") or p.get("size") in ("large", "very_large"):
        return "mon"
    return "inf"


class Battle:
    def __init__(self, name, frames, t, keys, map_name=None, groups=None, kind="game", winner=None, own_side=1):
        self.name, self.frames, self.t, self.keys = name, frames, t, keys
        self.map, self.groups, self.kind, self.winner, self.own = map_name, groups or [], kind, winner, own_side
        self.inside = {}
        db = passports()
        self.pp = {n: db.get(k) or {} for n, k in keys.items()}
        self.role = {n: role_of(self.pp.get(n)) for n in keys}
        first = frames[0]
        self.men0 = {n: (first.get(n) or {}).get("men") or (self.pp.get(n) or {}).get("men") or 1 for n in keys}
        for f in frames:                                   # units missing from the manifest (should not happen)
            for n in f:
                if n not in self.role:
                    self.role[n], self.pp[n], self.men0[n] = "inf", {}, f[n].get("men") or 1
        self.side = {n: (first.get(n) or {}).get("side") or (1 if n.startswith("own_") else 2) for n in self.role}
        self._frame_axes()

    # --- the view: up = towards the enemy, right = our right flank
    def _frame_axes(self):
        f = self.frames[0]
        own = [u for n, u in f.items() if self.side[n] == self.own]
        en = [u for n, u in f.items() if self.side[n] != self.own]
        ox, oz = sum(u["x"] for u in own) / len(own), sum(u["z"] for u in own) / len(own)
        ex, ez = sum(u["x"] for u in en) / len(en), sum(u["z"] for u in en) / len(en)
        a = math.atan2(ex - ox, ez - oz)                              # bearing own -> enemy (sin, cos) in (x, z)
        a = round(a / (math.pi / 2)) * (math.pi / 2)                  # snapped to the map's axes
        self.fwd = (round(math.sin(a)), round(math.cos(a)))
        self.rgt = (-round(math.cos(a)), round(math.sin(a)))          # facing +x the +z is on the right
        self.fwd_deg = math.degrees(a) % 360

    def view(self, x, z):
        """(right, up) metres of a world point in the view."""
        return x * self.rgt[0] + z * self.rgt[1], x * self.fwd[0] + z * self.fwd[1]

    def at(self, sec):
        """The frame index nearest to the second."""
        best = min(range(len(self.t)), key=lambda i: abs(self.t[i] - sec))
        return best

    def alive(self, u):
        return u is not None and (u.get("men") or 0) > 0

    def dims(self, n, u):
        """(front, depth) m of the unit's box: files = ordered width / h, ranks = men / files (sim/geometry.py dims)."""
        sp = (self.pp.get(n) or {}).get("spacing") or {}
        h, v = sp.get("h") or 1.5, sp.get("v") or 1.6
        men = max(u.get("men") or 1, 1)
        if men <= 1:
            r = (self.pp.get(n) or {}).get("radius_m") or 1.0
            return max(2 * r, 3.0), max(2 * r, 3.0)
        w = u.get("ow") or u.get("width") or 30.0
        files = max(1, min(math.floor(w / h + 1e-4), men))
        ranks = math.ceil(men / files)
        if self.role.get(n) == "art":
            return max(files * h, 6.0), v
        return max(files * h, 3.0), max(ranks * v, 2.0)


def load(run):
    """A Battle of a game recording: a run folder or a run name (game.battle.run_dir)."""
    from game.battle import run_dir
    R = run_dir(run)
    m = json.loads((R / "manifest.json").read_text(encoding="utf-8"))["config"]
    keys = {u["script_name"]: u["key"] for s in ("own", "enemy") for u in m["units"][s]}
    frames, ts, winner = [], [], None
    for line in (R / "events.jsonl").open(encoding="utf-8", errors="ignore"):
        if '"nn_sample"' not in line and '"nn_final"' not in line and '"result"' not in line:
            continue
        d = json.loads(line)
        if d.get("event") == "result":
            winner = d.get("winner")
            continue
        if d.get("event") not in ("nn_sample", "nn_final"):
            continue
        sec = round(d["t"] / 1000)
        if ts and sec <= ts[-1]:
            if d["event"] == "nn_final":                 # the last state replaces the same second's sample
                frames[-1] = {u["n"]: u for u in d["units"]}
            continue
        frames.append({u["n"]: u for u in d["units"]})
        ts.append(sec)
    groups, inside = [], {}
    comp = R / "companion.jsonl"
    if comp.exists():
        for line in comp.open(encoding="utf-8", errors="ignore"):
            if '"chain"' not in line:
                continue
            c = json.loads(line).get("chain") or {}
            if c.get("operative"):
                groups.append((c["sec"], c["operative"]))
            if c.get("inside"):
                inside[c["sec"]] = c["inside"]
    mp = m.get("map_name") or Path(str(m.get("map") or "").rstrip("\\/")).name.replace("wh3_main_macro_", "")
    if not (game.ROOT / "sim" / "maps" / mp / "heights.npz").exists():
        mp = None
    kind = "net" if m.get("own_ai") in (None, "net") and groups else (m.get("own_ai") or "game")
    b = Battle(R.name, frames, ts, keys, map_name=mp, groups=groups, kind=kind, winner=winner)
    b.inside = inside                                  # {sec: {unit: inside its area}} (the chain's areas)
    return b


def sim_frame(st, b=0, names=None):
    """One frame {name: unit} of the simulator's state st (sim.state: st.u [B, N] tensors, the observed fields named as
    nn_sample's; target - a slot), battle b of the batch. names[slot] - script names (default: own_/enemy_ + the unit
    key's tail + the slot). Call it every simulated second; the vision then draws the frames like a recording."""
    u = st.u
    fields = [k for k in ("x", "z", "b", "men", "hp", "mp", "ms", "r", "s", "w", "m", "mv", "f", "a", "fire", "ox", "oz",
                          "side") if k in u]
    cols = {k: u[k][b].tolist() for k in fields}
    tg = u["target"][b].tolist() if "target" in u else [-1] * len(cols["side"])
    wid = u["width"][b].tolist() if "width" in u else [30.0] * len(cols["side"])
    keys = st.keys[b] if hasattr(st, "keys") else [""] * len(cols["side"])
    if names is None:
        names = [("own_" if int(sd) == 1 else "enemy_") + ("_".join((k or "unit").split("_")[3:]) or k) + f"_{i}"
                 if int(sd) else ""
                 for i, (sd, k) in enumerate(zip(cols["side"], keys))]
    f = {}
    for i, n in enumerate(names):
        if not n or int(cols["side"][i]) == 0:
            continue
        row = {k: cols[k][i] for k in fields}
        row.update(side=int(row["side"]), n=n, ow=wid[i], key=keys[i],
                   t=names[tg[i]] if 0 <= tg[i] < len(names) else "")
        for k in ("r", "s", "w", "m", "mv", "f", "fire"):
            if k in row:
                row[k] = bool(row[k])
        f[n] = row
    return f


def from_sim(frames, secs, map_name=None, groups=None, name="sim"):
    """A Battle of the simulator's frames (sim_frame every second) at the seconds secs."""
    keys = {n: u.get("key") or "" for n, u in frames[0].items()}
    return Battle(name, frames, [round(float(s)) for s in secs], keys, map_name=map_name, groups=groups, kind="sim")
