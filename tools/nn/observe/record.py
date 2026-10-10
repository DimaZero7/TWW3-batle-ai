"""A recorded battle (build/nn-arena/runs/<run>, build/human/runs/<run>) second by second as the network and its critic
see it: the same path as the companion's (tools/nn/companion/loop.py Brain.decide: exchange.arrays and its fixes), for
any side, with the orders in force of both sides as the simulator keeps them.

Who gave the orders of a side:
* the network (its companion log: companion.jsonl for side 1, companion_enemy.jsonl for side 2) - known exactly;
* anybody else (a human, the game's AI, CA's planner) - read from the recording (infer_orders: an enemy engine target ->
  attack it; the order point > 15 m away or moving -> move, withdraw when running away from the nearest standing enemy;
  else hold; build/observe/infer_acc.py, 89 % of the kinds on the network's battles).
The order point of the recording beyond the map (the game AI's skirmishers walk to points ~5 km off) is clipped to the
map's edge first: the simulator's points never leave the map.

    rec = load(run_dir)
    pipe = Pipeline(rec)
    for k, frame in enumerate(rec.frames):
        step = pipe.step(k)        # Step: actor and critic observations of the observed sides, raw fields, gold lost
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.nn import gamedata
from tools.nn.companion import exchange
from tools.nn.model import observation as ob

ROUT_SHARE = 0.5          # reward.Weights.rout_share: a routing unit loses this share of what it has left
FAR_M = 1500.0            # the critic's input: an order point farther than this from the unit is zeroed (never seen in
#                           training; build/critic_arch/farfix.py, the critic's EV on the game 0.28 -> 0.35)
HOLD_M = 15.0             # infer_orders: the point this near a standing unit is a hold
RAW = ("x", "z", "b", "men", "r", "s", "m", "mv", "f", "fire", "ox", "oz", "target", "vis")
LOGS = {1: "companion.jsonl", 2: "companion_enemy.jsonl"}


@dataclass
class Recording:
    run: str
    path: Path
    cfg: dict
    battle: gamedata.Battle            # names, keys, side
    frames: list                       # [(t_ms, units, fx, abilities_used)]
    logs: dict                         # {side: [companion entries]} of the sides the network commanded
    winner: int
    attacker: int
    observed: tuple = ()               # the sides of another player (not the network)
    widths: dict = field(default_factory=dict)   # {name: the formation's width in the layout}
    human: int = 0                     # the side a human played (tools.build human: side 1), 0 none


def load(run_dir):
    """The recording of a run folder (module doc)."""
    from tools.nn.sim import scenario as sim_scenario       # (torch: only here, the rest is numpy)
    d = Path(run_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    g = gamedata.load(d)
    samples, effects, used_ev, result = [], [], [], {}
    with open(d / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"nn_sample"' in line or '"nn_final"' in line:
                samples.append(json.loads(line))
            elif '"nn_effects"' in line:
                e = json.loads(line)
                effects.append((e["t"], e["u"], e["fx"]))
            elif '"nn_ability"' in line:
                e = json.loads(line)
                if e.get("status") == "used":
                    used_ev.append((e["t"], e["u"], e["key"]))
            elif '"event":"result"' in line:
                result = json.loads(line)
    logs = {}
    for side, name in LOGS.items():
        p = d / name
        if not p.exists():
            continue
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        rows.sort(key=lambda e: e.get("t", 0))
        logs[side] = rows
    frames, fx, used, ei, ui = [], {}, {}, 0, 0
    effects.sort(key=lambda e: e[0])
    used_ev.sort(key=lambda e: e[0])
    for s in samples:
        while ei < len(effects) and effects[ei][0] <= s["t"]:
            _, u, v = effects[ei]
            fx[u] = None if isinstance(v, str) else (list(v) if isinstance(v, list) else [])
            ei += 1
        while ui < len(used_ev) and used_ev[ui][0] <= s["t"]:
            _, u, k = used_ev[ui]
            used.setdefault(u, {})[k] = used_ev[ui][0]
            ui += 1
        frames.append((s["t"], s["units"], dict(fx), {k: dict(v) for k, v in used.items()}))
    widths = {u["script_name"]: u.get("width") for tag in ("own", "enemy") for u in (cfg.get("units") or {}).get(tag, [])}
    human = 1 if manifest.get("target") == "human" else 0
    # a human's battle: only his side (against the game's AI too: that side is not the one we learn from there)
    observed = (human,) if human else tuple(s for s in (1, 2) if s not in logs)
    return Recording(run=d.name, path=d, cfg=cfg, battle=g, frames=frames, logs=logs,
                     winner=int(result.get("winner", 0) or 0), attacker=sim_scenario.attacker_of(g, cfg),
                     observed=observed, widths=widths, human=human)


def clip_points(ox, oz, bounds):
    """The order points (arrays) clipped to the map (xmin, xmax, zmin, zmax); NaN stays NaN."""
    xmin, xmax, zmin, zmax = (float(v) for v in bounds)
    with np.errstate(invalid="ignore"):
        return np.clip(ox, xmin, xmax), np.clip(oz, zmin, zmax)


def infer_orders(raw, names, side, own):
    """The order in force of side `own` read from the recording (module doc) -> exchange's given-dicts."""
    x, z = raw["x"], raw["z"]
    men = np.nan_to_num(raw["men"])
    rr = raw["r"] | raw["s"]
    enemy_ok = (side != own) & (men > 0) & ~rr & np.isfinite(x)
    out = {}
    for i in np.nonzero(side == own)[0]:
        j = int(raw["target"][i])
        if 0 <= j < len(names) and side[j] != own and men[j] > 0:
            out[names[i]] = {"unit": names[i], "kind": "attack", "target": names[j]}
            continue
        ox, oz = raw["ox"][i], raw["oz"][i]
        if not (np.isfinite(x[i]) and np.isfinite(ox)):
            out[names[i]] = {"unit": names[i], "kind": "hold"}
            continue
        if math.hypot(ox - x[i], oz - z[i]) < HOLD_M and not raw["mv"][i]:
            out[names[i]] = {"unit": names[i], "kind": "hold"}
            continue
        kind = "move"
        if enemy_ok.any() and raw["f"][i]:
            dd = np.where(enemy_ok, np.hypot(x - x[i], z - z[i]), np.inf)
            e = int(dd.argmin())
            if (ox - x[i]) * (x[e] - x[i]) + (oz - z[i]) * (z[e] - z[i]) < 0:
                kind = "withdraw"
        out[names[i]] = {"unit": names[i], "kind": kind, "x": float(ox), "z": float(oz)}
    return out


def far_zeroed(obs, far_m=FAR_M):
    """The critic's input with order points farther than far_m from the unit zeroed (in place; module FAR_M)."""
    t = obs.tokens
    cols = [ob.INDEX["order_fwd"], ob.INDEX["order_lat"]]
    far = np.hypot(t[..., cols[0]], t[..., cols[1]]) * ob.POS > far_m
    for c in cols:
        t[..., c] = np.where(far, 0.0, t[..., c])
    return obs


@dataclass
class Step:
    obs: dict          # {side: Obs} the actor's input of each observed side (its own view)
    critic: dict       # {side: Obs} the critic's input (full view, far points zeroed)
    raw: dict          # the recorded fields of this second [N] (RAW; ox / oz clipped to the map)
    gold: np.ndarray   # [2] gold lost so far by side 1 and side 2 (worst share kept), share of the budget
    standing: np.ndarray   # [N]


class Pipeline:
    """Recording -> per second the observations of the observed sides (module doc)."""

    def __init__(self, rec, sides=None, effects=True):
        self.rec = rec
        g = rec.battle
        self.names, self.side = list(g.names), np.asarray(g.side)
        self.sides = tuple(sides) if sides is not None else rec.observed
        self.layout = [{"n": u["script_name"], "slot": u.get("slot"), "width": u.get("width")}
                       for tag in ("own", "enemy") for u in rec.cfg.get("units", {}).get(tag, [])]
        self.keys = dict(zip(g.names, g.keys))
        self.factions = rec.cfg.get("factions") or {}
        self.effects = effects
        self.b = exchange.battle(self.doc(0))
        self.bounds = np.asarray(self.b.setup.bounds[0], float)
        self.breakoff = {s: exchange.Breakoff(self.b.names, self.b.side, self.b.shape, own=s) for s in rec.logs}
        self.moved = None
        self.points = {1: {}, 2: {}}
        self.given = {1: {}, 2: {}}            # the orders in force of the network's sides (its logs)
        self.li = {s: 0 for s in rec.logs}
        self.amem = {s: None for s in self.sides}
        self.cmem = {s: None for s in self.sides}
        self.amv = {s: None for s in self.sides}
        self.cmv = {s: None for s in self.sides}
        cost = np.asarray(self.b.setup.cost[0], float)
        self.cost = cost
        self.budget = max(1.0, float(np.mean([cost[self.b.side == s].sum() for s in (1, 2)])))
        self.worst = np.zeros(len(self.names))

    def doc(self, k):
        t_ms, units, fx, used = self.rec.frames[k]
        rows = {u["n"]: u for u in units}
        us = []
        for n, s in zip(self.names, self.side):
            u = dict(rows.get(n) or {"n": n, "side": int(s), "men": 0})
            u["key"] = self.keys[n]
            u["side"] = int(s)
            if fx is not None and self.effects and fx.get(n) is not None:
                u["fx"] = fx[n]
            us.append(u)
        return {"batch": self.rec.run, "move": k, "t": t_ms, "units": us, "factions": self.factions,
                "attacker": self.rec.attacker, "layout": self.layout,
                "abilities_used": (used or {}) if self.effects else {}}

    def _net_given(self, side, t_ms):
        rows, given = self.rec.logs[side], self.given[side]
        while self.li[side] < len(rows) and rows[self.li[side]].get("t", 0) < t_ms:
            for o in rows[self.li[side]].get("orders") or []:
                if o.get("kind") != "keep" and not o.get("out") and o.get("unit") in self.keys:
                    given[o["unit"]] = o
            self.li[side] += 1
        return given

    def step(self, k):
        b = self.b
        doc = self.doc(k)
        st = exchange.arrays(doc, b.names, b.slots if self.effects else None)
        ox, oz = clip_points(st["ox"][0], st["oz"][0], self.bounds)
        st["ox"][0], st["oz"][0] = ox, oz
        raw = {f: np.asarray(st[f][0]).copy() for f in RAW if f != "target"}
        raw["target"] = st["target"][0].copy()
        raw["r"], raw["s"] = raw["r"].astype(bool), raw["s"].astype(bool)
        self.moved = exchange.running_by_speed(st, b.walk, self.moved)
        for s in (1, 2):
            exchange.engaged_targets(st, b.side, own=s, shape=b.shape)
        exchange.threat_flags(st, b.side, b.threat)
        if self.effects:
            exchange.effects_on(st, doc, b.names, b.setup)
        for s in (1, 2):
            if s in self.rec.logs:
                given = self._net_given(s, doc["t"])
                self.breakoff[s].update(st, given)
            else:
                given = infer_orders(raw, b.names, b.side, own=s)
            self.points[s] = exchange.order_points(st, b.names, b.side, given, self.points[s], own=s)
        obs, crit = {}, {}
        for s in self.sides:
            self.amv[s] = exchange.keep_still(self.amem[s], st, self.amv[s])
            obs[s], self.amem[s] = ob.observe(st, b.setup, s, self.amem[s])
            self.cmv[s] = exchange.keep_still(self.cmem[s], st, self.cmv[s])
            c, self.cmem[s] = ob.observe(st, b.setup, s, self.cmem[s], full=True)
            crit[s] = far_zeroed(c)
        # gold lost (reward.gold_lost on the recorded fields, the worst share kept)
        men = np.nan_to_num(st["men"][0])
        gone = (men > 0) & ~np.isfinite(st["x"][0])
        sh = np.asarray(st["s"][0], bool)
        r = np.asarray(st["r"][0], bool) | sh
        hp_lost = np.clip(1 - np.nan_to_num(st["hp"][0], nan=1.0), 0, 1)
        lost = np.where((men <= 0) | gone | sh, 1.0, hp_lost + (1 - hp_lost) * ROUT_SHARE * r)
        self.worst = np.maximum(self.worst, lost)
        g = self.cost * self.worst
        gold = np.array([g[b.side == 1].sum(), g[b.side == 2].sum()]) / self.budget
        return Step(obs, crit, raw, gold, (men > 0) & ~r & ~gone)
