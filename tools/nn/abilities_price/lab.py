"""The bonus lab: the rate k of each bonus kind (gold per counter unit) measured in controlled simulator experiments,
independent of the network's presses (docs/en/training/ability_price.md, "Bonus lab").

A lab experiment is a PAIR of copies of one small battle that differ only by one bonus: in copy "on" the lord fires a lab
ability that carries just that bonus (its passport: one real ability's stats of that kind, its duration, radius and
targets, the value times a dose), in copy "off" he does not. Both copies have the same battle id (the simulator's
randomness is a hash of it: common random numbers), the same start and the same scripted orders - the orders are made
on copy "on" and replayed in copy "off" (open loop, as the branch prices of collect.py). The pair's gold
y = (enemy gold destroyed - own gold lost) on - (the same) off, 60 s after the press (the trade of the reward and of the
branch prices x the budget), and the counter c of the bonus's brick (layer 1, bonuses.py) on copy "on"'s trajectory,
through layer 2 (ability.py: the lab passport's Ability, its masks and counters) - the very code that prices a press.
The rate of a kind: k = sum y / sum c over its pairs (the mean gold per counter unit over the drawn conditions), 95 %
by bootstrap over the pairs; where it clearly depends on the conditions, the curve (k per bin of a condition).

The conditions are drawn from the real ones: the lords that carry the bonus (the Empire's General, the Skaven
Warlord), friends and enemies from our armies' pools (config/nn/pools.json, both factions), two families of start -
"line" (the lord with 0-4 friends against 1-4 enemies, lines 10-80 m apart, some enemies on a flank; the press 0-45 s
in, so before, at or deep in the fight) and "runup" (the lord 20-60 m - 20-150 m for speed and vigour - from an enemy
unit's edge, at any angle from its front to its rear, the enemy often already fighting a friend; the press as he
starts) - fatigue of the lord from the presses' own distribution (or every tier, for vigour), the bonus at 0.5 / 1 / 2
times its real value (proportionality). Each side keeps a far reserve (4-10 units 700 m back, holding) so a local fight
does not end the battle and the army's strength (army destruction) is that of a real army. Beside these small fights,
whole battles (lab_sim.draw_battles: the v2 armies of training, ai_like on both sides; the press at a battle time of the
network's presses, or when the bonus's condition starts): the rates come from those (MAIN), the small fights give the
controlled curves (run-up, angle, the enemy's attack).

This file is numpy only (the host): the lab's variants and passports, the scenario draw, the counters from the saved
trajectories, the rates and curves, the layer-3 check against the branch prices, the card. The simulator runs are
lab_sim.py (torch, the container).

    python -m tools.nn.abilities_price.lab --dir build/abilities_price/lab      # rates, curves, layer 3 -> lab.md
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import validate as va

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "nn"
PREFIX = "zz_lab_"                    # sorted after the real keys: their indices in the simulator's table stay
KINDS = ("melee_damage", "charge", "melee_defence", "morale", "speed", "charge_speed", "vigour")
DOSES = (0.5, 1.0, 2.0)
DOSE_P = (0.2, 0.6, 0.2)
HORIZON_S = 60
FRIENDS_MAX, ENEMIES_MAX = 4, 4
RESERVE = (4, 10)                     # far reserve units per side (drawn)
RESERVE_X = 700.0
PER_SIDE = 1 + FRIENDS_MAX + 2 + RESERVE[1]      # lord, friends (+2 extra in runup), reserve
TRACE_H = 1 + FRIENDS_MAX + 2         # the traced slots of side 1: the lord and his lab friends (not the reserve)
# the families of start per kind (share of "line", the rest "runup") and the run-up range, m
FAMILY = {"melee_damage": 0.6, "charge": 0.2, "melee_defence": 1.0, "morale": 1.0, "speed": 0.4, "charge_speed": 0.2,
          "vigour": 0.6}
RUNUP = {"charge": (20.0, 60.0), "charge_speed": (20.0, 60.0), "melee_damage": (20.0, 60.0), "speed": (20.0, 150.0),
         "vigour": (20.0, 150.0)}
PRESS_LINE = (0, 45)                  # the press, s after the start (line)
PRESS_RUNUP = (0, 3)
FATIGUE_TIERS = (0.0, 2800.0, 6600.0, 12600.0, 18000.0, 27000.0)
# script modes of a slot (lab_sim.script)
NEAREST, HOLD, TARGET, RESERVE_HOLD = 0, 1, 2, 3


# --- the lab's abilities ---

@dataclass(frozen=True)
class Variant:
    """One real ability's part of one bonus kind (the lab ability is it alone)."""
    kind: str
    source: str               # the real ability's key
    groups: tuple             # "self", "friends"
    active_s: float
    range_m: float
    update_targets: bool
    lords: tuple              # the lords (unit keys) that carry the source ability


def variants(passports, bricks, units):
    """{kind: [Variant]} of the pressable abilities' parts (ability.py's mapping of a passport to bricks)."""
    out = {}
    for key, a in abl.abilities(passports, bricks).items():
        lords = tuple(sorted(k for k, u in units.items()
                             if u.get("caste") == "lord" and key in (u.get("abilities") or ())))
        for part in a.parts:
            out.setdefault(part.kind, []).append(Variant(part.kind, key, part.groups, a.active_s, a.range_m,
                                                         a.update_targets, lords))
    return out


def lab_key(v, dose):
    return f"{PREFIX}{v.kind}__{v.source.split('abilities_')[-1]}__{dose:g}"


def lab_passport(v, dose, passports, bricks):
    """The lab ability of variant v at dose x its real value: the source's passport with only v's stats (on v's groups),
    or only its vigour, scaled; never fired by the game's AI (it is in no trigger list and no side is the AI's)."""
    p = passports[v.source]
    stats = set(bricks[v.kind].desc.stats)
    eff = []
    for e in p["effects"]:
        if (e["stat"], e["how"]) in stats and tuple(g for g in abl.GROUPS if g in e["on"]) == v.groups:
            x = float(e["value"])
            eff.append(dict(e, value=1.0 + (x - 1.0) * dose if e["how"] == "mult" else x * dose))
    vig = float(p.get("vigour_per_s") or 0.0) * dose if v.kind in ("vigour", "fatigue_cost") else 0.0
    return {"active_s": float(p["active_s"]), "recharge_s": 9999.0, "initial_s": 0.0, "uses": -1, "vigour_per_s": vig,
            "cp": [0, 0], "range_m": float(p["range_m"]), "passive": False, "targets_own": True, "friendly_units": -1,
            "enemy_units": 0, "self_cast": True, "update_targets": bool(p.get("update_targets", True)), "auto": False,
            "recharge_when": [], "off_when": [], "phases": [], "targets": dict(p["targets"]), "effects": eff,
            "attributes": [], "lab": {"kind": v.kind, "source": v.source, "dose": dose}}


def ability_of(passport, bricks):
    """Layer 2's Ability of a lab passport (one part: its kind)."""
    return abl.Ability.from_passport("lab", passport, bricks)


# --- the conditions ---

def load_config():
    rd = lambda n: json.loads((CONFIG / n).read_text(encoding="utf-8"))
    pools = rd("pools.json")["factions"]
    units = rd("units.json")
    units = units["units"]
    passports = rd("abilities.json")["abilities"]
    sim = rd("sim.json")
    rules = rd("game_rules.json")
    return pools, units, passports, sim, rules


def pool_of(pools):
    """{faction: {"lord": key, "units": [(key, width)]}}."""
    out = {}
    for f, v in pools.items():
        us = [(u["key"], float(u.get("width") or 30.0)) for u in list(v["units"]) + list(v.get("units_v2") or [])]
        out[f] = {"lord": v["lord"]["key"], "units": list(dict.fromkeys(us))}
    return out


def depth_of(key, width, units):
    """A formation's depth, m (tools/nn/sim/geometry.dims; a lord: 2 x his radius)."""
    u = units[key]
    men = int(u.get("men") or 1)
    if men <= 1:
        return 2.0 * float(u.get("radius_m") or 0.7)
    sp = u.get("spacing") or {}
    h, v = float(sp.get("h") or 1.5), float(sp.get("v") or 1.5)
    files = max(1, min(men, math.floor(width / h + 1e-4)))
    return math.ceil(men / files) * v


def empirical_fatigue(paths):
    """The presser's fatigue at the press over collect.py's files (None when there are none)."""
    out = []
    for p in paths or ():
        if Path(p).exists():
            d = np.load(p)
            P = len(d["owner"])
            out.append(d["fatigue"][np.arange(P), 0, 0, d["owner"]].astype(float))
    return np.concatenate(out) if out else None


def _bearing(dx, dz):
    return math.degrees(math.atan2(dx, dz)) % 360.0


def draw(kind, n, rng, pool, units, vlist, fatigue=None, seed_base=0):
    """n scenarios of `kind` (module doc): each a dict - "army" (tools/nn/sim/scenario.build's description; names
    "lord", "f<k>", "e<k>", "elord", "r1_<k>", "r2_<k>"), "mode" {name: script mode}, "target" {name: name},
    "fatigue" {name: points}, "press" (s), "key" (the lab ability), "variant", "dose", "family" and the conditions."""
    factions = list(pool)
    fac_of = {pool[f]["lord"]: f for f in factions}
    out = []
    for i in range(n):
        v = vlist[rng.integers(len(vlist))]
        dose = float(rng.choice(DOSES, p=DOSE_P))
        lord = v.lords[rng.integers(len(v.lords))]
        own_f = fac_of[lord]
        foe_f = factions[rng.integers(len(factions))]
        own_u, foe_u = pool[own_f]["units"], pool[foe_f]["units"]
        family = "line" if rng.random() < FAMILY[kind] else "runup"
        s1 = [{"key": lord, "x": 0.0, "z": 0.0, "general": True, "name": "lord", "b": 90.0}]
        s2 = []
        mode, target, fat = {"lord": NEAREST}, {}, {}
        cond = {"family": family}
        if family == "line":
            nf = int(rng.choice(5, p=(0.1, 0.2, 0.25, 0.25, 0.2)))
            ne = int(rng.choice(4, p=(0.25, 0.3, 0.25, 0.2))) + 1
            half = max(nf - 1, 0) * 35.0 / 2
            for k in range(nf):
                key, w = own_u[rng.integers(len(own_u))]
                s1.append({"key": key, "x": float(rng.uniform(-3, 3)), "z": float(k * 35.0 - half + rng.uniform(-3, 3)),
                           "width": w, "name": f"f{k}", "b": 90.0})
                mode[f"f{k}"] = NEAREST if rng.random() < 0.8 else HOLD
            s1[0]["x"] = float(-rng.uniform(0, 30))
            s1[0]["z"] = float(rng.uniform(-1, 1) * (half + 10))
            gap = float(rng.uniform(10, 80))
            hold_foe = rng.random() < 0.25 and any(mode.get(f"f{k}") == NEAREST for k in range(nf))
            for k in range(ne):
                key, w = foe_u[rng.integers(len(foe_u))]
                if rng.random() < 0.15:
                    x, z = float(rng.uniform(-20, 20)), float(rng.choice((-1, 1)) * (half + rng.uniform(30, 50)))
                else:
                    x = gap + depth_of(key, w, units) / 2 + float(rng.uniform(0, 10))
                    z = float(k * 35.0 - max(ne - 1, 0) * 17.5 + rng.uniform(-15, 15))
                s2.append({"key": key, "x": x, "z": z, "width": w, "name": f"e{k}", "b": 270.0})
                mode[f"e{k}"] = HOLD if hold_foe else NEAREST
            cond.update(friends=nf, enemies=ne, gap=gap)
            press = int(rng.integers(PRESS_LINE[0], PRESS_LINE[1] + 1))
        else:
            lo, hi = RUNUP[kind]
            key, w = foe_u[rng.integers(len(foe_u))]
            dE = depth_of(key, w, units)
            s2.append({"key": key, "x": 0.0, "z": 0.0, "width": w, "name": "e0", "b": 270.0})
            engaged = rng.random() < 0.6
            nf = 0
            if engaged:
                fk, fw = own_u[rng.integers(len(own_u))]
                s1.append({"key": fk, "x": -(dE + depth_of(fk, fw, units)) / 2 - 0.5, "z": 0.0, "width": fw,
                           "name": "f0", "b": 90.0})
                mode["f0"], target["f0"] = TARGET, "e0"
                mode["e0"], target["e0"] = TARGET, "f0"
                nf = 1
            else:
                mode["e0"] = HOLD if rng.random() < 0.5 else NEAREST
            R = float(rng.uniform(lo, hi))
            phi = float(rng.uniform(60 if engaged else 0, 180))
            side = float(rng.choice((-1, 1)))
            # the rest of the enemy's line beside E0, on the side away from the lord's way in; our other units facing it
            ne = 1 + int(rng.integers(0, 3))
            for k in range(1, ne):
                ek, ew = foe_u[rng.integers(len(foe_u))]
                s2.append({"key": ek, "x": float(rng.uniform(-5, 5)), "z": float(-side * k * rng.uniform(35, 45)),
                           "width": ew, "name": f"e{k}", "b": 270.0})
                mode[f"e{k}"] = HOLD if rng.random() < 0.5 else NEAREST
            for k in range(nf, nf + int(rng.integers(0, 3))):
                fk, fw = own_u[rng.integers(len(own_u))]
                s1.append({"key": fk, "x": float(rng.uniform(-45, -25)),
                           "z": float(-side * (k + 1) * rng.uniform(35, 45)), "width": fw, "name": f"f{k}", "b": 90.0})
                mode[f"f{k}"] = NEAREST
                nf += 1
            a = math.radians(phi)
            # E0 faces west (-x): its front is at angle 0 (west), its rear at 180 (east)
            dx, dz = -math.cos(a), side * math.sin(a)
            edge = min((dE / 2) / max(abs(math.cos(a)), 1e-6), (w / 2) / max(abs(math.sin(a)), 1e-6))
            s1[0]["x"], s1[0]["z"] = (R + edge) * dx, (R + edge) * dz
            s1[0]["b"] = _bearing(-dx, -dz)
            mode["lord"], target["lord"] = TARGET, "e0"
            cond.update(friends=nf, enemies=ne, runup=R, angle=phi, engaged=bool(engaged))
            press = int(rng.integers(PRESS_RUNUP[0], PRESS_RUNUP[1] + 1))
        # the enemy's lord: in the fight (a quarter) or with its reserve
        elord = pool[foe_f]["lord"]
        if family == "line" and rng.random() < 0.25:
            s2.append({"key": elord, "x": float(cond["gap"] + rng.uniform(10, 30)), "z": float(rng.uniform(-20, 20)),
                       "general": True, "name": "elord", "b": 270.0})
            mode["elord"] = NEAREST
        else:
            s2.append({"key": elord, "x": RESERVE_X + 40.0, "z": 0.0, "general": True, "name": "elord", "b": 270.0})
            mode["elord"] = RESERVE_HOLD
        for side_, units_, sgn, mine in ((1, own_u, -1, s1), (2, foe_u, 1, s2)):
            for k in range(int(rng.integers(RESERVE[0], RESERVE[1] + 1))):
                rk, rw = units_[rng.integers(len(units_))]
                mine.append({"key": rk, "x": sgn * RESERVE_X, "z": float((k - 5) * 40.0), "width": rw,
                             "name": f"r{side_}_{k}", "b": 90.0 if side_ == 1 else 270.0})
                mode[f"r{side_}_{k}"] = RESERVE_HOLD
        # fatigue at the start: the lord's from the presses' own distribution (every tier for vigour), the lab units'
        # fresh to tired, the reserves' fresh
        if kind == "vigour" or fatigue is None:
            fat["lord"] = float(rng.uniform(0, 30000))
        else:
            fat["lord"] = float(fatigue[rng.integers(len(fatigue))])
        for u in s1[1:] + s2:
            if mode.get(u["name"]) != RESERVE_HOLD:
                fat[u["name"]] = float(rng.uniform(0, 12600))
        ab = lab_key(v, dose)
        sides = {1: {"faction": own_f, "ai": False, "skirmish": False, "units": s1},
                 2: {"faction": foe_f, "ai": False, "skirmish": False, "units": s2}}
        out.append({"army": {"attacker": 1, "sides": sides},
                    "mode": mode, "target": target, "fatigue": fat, "press": press, "key": ab, "kind": kind,
                    "variant": v.source, "dose": dose, "lord": lord, "id": seed_base + i, **cond,
                    "foe_keys": [u["key"] for u in s2 if mode.get(u["name"]) != RESERVE_HOLD]})
    return out


# --- counters and rates ---

def trace_of(data, i, dt_s=1.0):
    """The trajectory (bonuses.TRACE) of pair i of a lab_sim file."""
    tr = {k: np.asarray(data[k][i], dtype=bool if k in bz.FLAGS else np.float64)
          for k in ("x", "z", "alive", "r", "m", "mv", "sprint", "charge", "morale", "fatigue", "mmax", "on")}
    tr["owner"], tr["dt_s"] = 0, dt_s
    return tr


def counters(data, passports_lab, bricks):
    """[n] the counter of each pair's bonus (layer 1 through layer 2, on copy "on"'s trajectory)."""
    keys = [str(k) for k in data["key"]]
    out = np.zeros(len(keys))
    cache = {}
    for i, k in enumerate(keys):
        a = cache.get(k) or cache.setdefault(k, ability_of(passports_lab[k], bricks))
        out[i] = sum(a.counters(trace_of(data, i), bricks).values())
    return out


def unit_counters(data, i, passport, bricks):
    """[U] the counter of pair i split by the affected unit (the brick counted on each unit's column of the mask)."""
    a = ability_of(passport, bricks)
    tr = trace_of(data, i)
    m = a.masks(tr)
    part = a.parts[0]
    mask = np.zeros_like(tr["alive"])
    for g in part.groups:
        mask = mask | m[g]
    out = np.zeros(mask.shape[1])
    for u in range(mask.shape[1]):
        if mask[:, u].any():
            mu = np.zeros_like(mask)
            mu[:, u] = mask[:, u]
            out[u] = bricks[part.kind].count(tr, mu, part.value)
    return out


def rate(y, c, boot=1000, seed=0):
    """{"k": sum y / sum c, "lo", "hi" (95 % bootstrap over the pairs), "ls" (least squares through 0), "n", "n_pos"
    (pairs with c > 0), "y0" (mean gold of the pairs with c = 0: work the counter does not see)}."""
    y, c = np.asarray(y, float), np.asarray(c, float)
    n = len(y)
    out = {"n": int(n), "n_pos": int((c > 0).sum())}
    zero = c <= 0
    out["y0"] = float(y[zero].mean()) if zero.any() else float("nan")
    out["y0_se"] = float(y[zero].std() / math.sqrt(zero.sum())) if zero.sum() > 1 else float("nan")
    if c.sum() <= 0:
        return dict(out, k=float("nan"), lo=float("nan"), hi=float("nan"), ls=float("nan"))
    k = float(y.sum() / c.sum())
    if boot <= 0:
        return dict(out, k=k, lo=float("nan"), hi=float("nan"), ls=float((c * y).sum() / (c * c).sum()))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n, (boot, n))
    cs = c[ix].sum(1)
    ks = np.where(cs > 0, y[ix].sum(1) / np.maximum(cs, 1e-12), np.nan)
    lo, hi = np.nanquantile(ks, [0.025, 0.975])
    return dict(out, k=k, lo=float(lo), hi=float(hi), ls=float((c * y).sum() / (c * c).sum()))


def curve(y, c, x, edges, boot=1000, seed=0):
    """[{"bin", "lo_x", "hi_x", **rate}] the rate per bin of a condition x (edges: the bins' bounds)."""
    out = []
    x = np.asarray(x, float)
    for j in range(len(edges) - 1):
        sel = (x >= edges[j]) & (x < edges[j + 1])
        if sel.sum() < 10:
            continue
        r = rate(np.asarray(y)[sel], np.asarray(c)[sel], boot, seed)
        out.append(dict(r, lo_x=float(edges[j]), hi_x=float(edges[j + 1])))
    return out


def separated(rows):
    """True when two bins' 95 % intervals do not overlap (the rate clearly depends on the condition)."""
    rows = [r for r in rows if np.isfinite(r["lo"]) and r["n_pos"] >= 30]
    return any(a["hi"] < b["lo"] or b["hi"] < a["lo"] for i, a in enumerate(rows) for b in rows[i + 1:])


# --- the conditions of a pair (for the curves) ---

def conditions(data, scen, units, passports_lab, bricks):
    """{name: [n]} per pair: the lord's fatigue tier at the press, his and the affected units' morale at the press
    (counter-weighted), the enemies' mean melee attack and armour, the run-up, angle, dose."""
    n = len(scen)
    P = data["fatigue"][:, 0, 0].astype(float)
    out = {"tier": np.searchsorted(np.asarray(FATIGUE_TIERS), P, side="right") - 1.0,
           "fatigue": P,
           "dose": np.array([s["dose"] for s in scen]),
           "runup": np.array([s.get("runup", np.nan) for s in scen]),
           "angle": np.array([s.get("angle", np.nan) for s in scen]),
           "press": np.array([s["press"] for s in scen], float),
           "foe_attack": np.array([np.mean([units[k]["melee"]["attack"] for k in s["foe_keys"]]) for s in scen]),
           "foe_armour": np.array([np.mean([units[k]["armour"] for k in s["foe_keys"]]) for s in scen]),
           "lord_morale": data["morale"][:, 0, 0].astype(float)}
    if scen and scen[0]["kind"] == "morale":
        mor = np.full(n, np.nan)
        for i, s in enumerate(scen):
            w = unit_counters(data, i, passports_lab[s["key"]], bricks)
            if w.sum() > 0:
                mor[i] = float((w * data["morale"][i, 0].astype(float)).sum() / w.sum())
        out["morale_at_press"] = mor
    return out


CURVES = {"melee_damage": [("foe_armour", (0, 40, 70, 100, 200)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "charge": [("runup", (20, 30, 40, 50, 61)), ("angle", (0, 45, 135, 181)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "melee_defence": [("foe_attack", (0, 25, 32, 40, 100)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "morale": [("morale_at_press", (-200, 16, 32, 48, 64, 1000)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "speed": [("runup", (20, 50, 80, 110, 151)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "charge_speed": [("runup", (20, 30, 40, 50, 61)), ("dose", (0.4, 0.75, 1.5, 2.5))],
          "vigour": [("tier", (-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)), ("dose", (0.4, 0.75, 1.5, 2.5))]}


# --- layer 3: the lab's rates against the branch prices ---

def morale_curve_counters(data, names, abilities, bricks, edges, rates_bin, k_flat):
    """[P] the morale part of each press priced by the curve: each affected unit's counter x the rate of its morale
    bin at the press, divided by the flat rate (so it plugs into the morale column in place of the counter)."""
    P = len(data["diff"])
    out = np.zeros(P)
    for p in range(P):
        a = abilities.get(names[int(data["key"][p])])
        if a is None or "morale" not in a.kinds():
            continue
        part = next(x for x in a.parts if x.kind == "morale")
        tr = va.trace(data, p, 0)
        m = a.masks(tr)
        mask = np.zeros_like(tr["alive"])
        for g in part.groups:
            mask = mask | m[g]
        tot = 0.0
        for u in np.nonzero(mask.any(0))[0]:
            mu = np.zeros_like(mask)
            mu[:, u] = mask[:, u]
            cu = bricks["morale"].count(tr, mu, part.value)
            if cu <= 0:
                continue
            j = int(np.clip(np.searchsorted(edges, float(tr["morale"][0, u]), side="right") - 1, 0, len(rates_bin) - 1))
            tot += cu * rates_bin[j]
        out[p] = tot / k_flat if k_flat > 0 else 0.0
    return out


def vigour_curve_factor(data, edges, rates_bin, k_flat):
    """[P] the vigour column's factor by the presser's fatigue tier at the press (the curve's rate / the flat one)."""
    P = len(data["diff"])
    f = data["fatigue"][np.arange(P), 0, 0, data["owner"]].astype(float)
    tier = np.searchsorted(np.asarray(FATIGUE_TIERS), f, side="right") - 1.0
    j = np.clip(np.searchsorted(np.asarray(edges, float), tier, side="right") - 1, 0, len(rates_bin) - 1)
    return np.asarray(rates_bin, float)[j] / k_flat if k_flat > 0 else np.zeros(P)


def curve_rates(curves, kd, name, k_flat):
    """(edges, [rate per bin]) of a kind's curve, a bin without enough pairs at the flat rate (None: no curve)."""
    rows = (curves.get(kd) or {}).get(name)
    if not rows:
        return None
    edges = dict(CURVES[kd])[name]
    by = {r["lo_x"]: max(0.0, r["k"]) for r in rows if np.isfinite(r["k"])}
    return edges, [by.get(float(e), max(0.0, k_flat)) for e in edges[:-1]]


def layer3(paths, rate_sets, bricks, passports, sim, boot=200, seed=0):
    """The card rows (validate.card) of the network's presses priced by the regression's cross-fitted rates and by each
    set of the lab's rates: rate_sets {name: (rates {kind: {"k"}}, curves or None)}; with curves, the morale counter is
    priced per unit by its morale at the press and vigour by the presser's fatigue tier. -> ({way: rows}, the
    regression's rates, the kinds, the number of presses)."""
    abilities = abl.abilities(passports, bricks)
    data, names, meta = va.load(paths)
    dt_s = float(meta[0].get("dt_s", 1.0))
    kinds, C, _ = va.counters(data, names, abilities, bricks, dt_s)
    y = data["diff"].mean(1) * data["budget"].mean(1)
    reg = va.fit_rates(C, y, kinds, bricks, boot, seed)
    pred_reg = va.crossfit(C, y, kinds, bricks, seed=seed)
    out = {"regression": va.card(data, names, abilities, bricks, kinds, C, reg, pred_reg, passports, sim["abilities"])}
    cache = {}
    for way, (rates, curves) in rate_sets.items():
        k = np.nan_to_num(np.array([rates.get(kd, {}).get("k", 0.0) for kd in kinds], float))
        rl = {kd: {"k": float(k[i])} for i, kd in enumerate(kinds)}
        Cw = C.copy()
        if curves:
            if "morale" in kinds and curve_rates(curves, "morale", "morale_at_press", k[kinds.index("morale")]):
                j = kinds.index("morale")
                edges, rb = curve_rates(curves, "morale", "morale_at_press", k[j])
                key = ("morale", tuple(rb), k[j])
                if key not in cache:
                    cache[key] = morale_curve_counters(data, names, abilities, bricks, np.asarray(edges, float), rb,
                                                       k[j])
                Cw[:, j] = cache[key]
            if "vigour" in kinds and curve_rates(curves, "vigour", "tier", k[kinds.index("vigour")]):
                j = kinds.index("vigour")
                edges, rb = curve_rates(curves, "vigour", "tier", k[j])
                Cw[:, j] = C[:, j] * vigour_curve_factor(data, edges, rb, k[j])
        X = va.design(Cw, kinds, bricks)
        out[way] = va.card(data, names, abilities, bricks, kinds, Cw, rl, X @ k, passports, sim["abilities"])
    return out, reg, kinds, len(y)


# --- the card ---

def _f(x, nd=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}g}"


FAMILY_FILES = (("small", "pairs_{}.npz"), ("battle", "battle_{}.npz"), ("battle_cond", "battle_cond_{}.npz"))


def load_family(path, units, bricks):
    """One lab_sim file -> {"y", "c", "fired", "cond", "y_own", "y_foe", "sprint_busy", "seconds", "n"}."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    m = json.loads(str(d["meta"]))
    scen, plab = m["scenarios"], m["passports"]
    sp = d["sprint"][:, :, 0]
    return {"y": d["y"].astype(float), "c": counters(d, plab, bricks), "fired": d["fired"].astype(bool),
            "cond": conditions(d, scen, units, plab, bricks), "y_own": d["y_own"].astype(float),
            "y_foe": d["y_foe"].astype(float), "sprint_busy": float((sp & d["tbusy"]).sum() / max(1, sp.sum())),
            "seconds": m.get("seconds"), "n": len(scen)}


def summarise(fams, boot=1000, seed=0):
    """The rate (at dose 1) and the curves of one or more families' pairs pooled."""
    cat = lambda k: np.concatenate([f[k] for f in fams])
    y, c, fired = cat("y"), cat("c"), cat("fired")
    names = set.intersection(*(set(f["cond"]) for f in fams))
    cond = {k: np.concatenate([f["cond"][k] for f in fams]) for k in names}
    sel = fired & (cond["dose"] == 1.0)              # the rate at the real value; the doses' curve beside it
    r = rate(y[sel], c[sel], boot, seed)
    r["k_all"] = rate(y[fired], c[fired], 0, seed)["k"]
    r["own_saved"] = float(-cat("y_own")[sel].mean())
    r["foe_destroyed"] = float(cat("y_foe")[sel].mean())
    r["fired"] = float(fired.mean())
    r["c_mean"] = float(c[sel].mean())
    return r, (y, c, fired, sel, cond)


def curves_of(kd, pooled, boot=1000, seed=0):
    y, c, fired, sel, cond = pooled
    out = {}
    for name, edges in CURVES.get(kd, []):
        if name in cond:
            s_ = fired if name == "dose" else sel
            out[name] = curve(y[s_], c[s_], cond[name][s_], edges, boot, seed)
    return out


def markdown(rates, fam_rates, curves, l3, reg, kinds, P, seconds):
    fams = [f for f, _ in FAMILY_FILES if any(f in fr for fr in fam_rates.values())]
    L = ["# Bonus lab: rates of the bonus kinds from controlled simulator pairs", ""]
    L.append("Pairs per kind (copies on / off of one battle, the same battle id, the orders of 'on' replayed in 'off'), "
             "60 s after the press; gold = (enemy gold destroyed - own lost) on - off; counter = layer 1's brick through "
             "layer 2 on 'on'. Families: small (line / runup fights, the lab script); battle (whole v2 battles, ai_like "
             "both sides, the press at a battle time of the network's presses); battle_cond (the same, the press when "
             "the bonus's condition starts, at most 120 s later). The rate k = sum gold / sum counter over the pairs at "
             "the real value (dose 1), 95 % by bootstrap; 'battles' pools the two whole-battle families. "
             f"Simulator time: {', '.join(f'{k} {v} s' for k, v in seconds.items())}.")
    L += ["", "## Rates k (gold per counter unit) at the real value: lab vs the regression on the network's presses", "",
          "| kind | unit | lab k, battles | 95 % | " + " | ".join(fams) + " | all doses | pairs dose 1 (c > 0): "
          + "; ".join(fams) + " | mean gold at c = 0 | gold a pair: own saved / enemy destroyed | regression k | "
          "regression 95 % |", "|---|---|---|---|" + "---|" * len(fams) + "---|---|---|---|---|---|"]
    b = bz.bricks()
    for kd in KINDS:
        r = rates.get(kd)
        if r is None:
            continue
        g = reg.get(kd, {})
        fr = fam_rates.get(kd, {})
        cell = lambda f: (f"{_f(fr[f]['k'])} ({_f(fr[f]['lo'])} .. {_f(fr[f]['hi'])})" if f in fr else "-")
        npairs = "; ".join(f"{fr[f]['n']} ({fr[f]['n_pos']})" if f in fr else "-" for f in fams)
        L.append(f"| {kd} | {b[kd].desc.unit} | {_f(r['k'])} | {_f(r['lo'])} .. {_f(r['hi'])} | "
                 + " | ".join(cell(f) for f in fams) + f" | {_f(r.get('k_all'))} | {npairs} | {_f(r['y0'])} +- "
                 f"{_f(r['y0_se'], 2)} | {_f(r.get('own_saved'), 2)} / {_f(r.get('foe_destroyed'), 2)} | "
                 f"{_f(g.get('k'))} | {_f(g.get('lo'))} .. {_f(g.get('hi'))} |")
    sb = {f"{kd} {f}": v["sprint_busy"] for kd in ("charge", "charge_speed") for f, v in fam_rates.get(kd, {}).items()}
    if sb:
        L += ["", "Share of the lord's trace `sprint` seconds whose attack target is already in melee (the simulator "
              "gives no sprint and no charge then, charge.free_target_only): "
              + ", ".join(f"{k} {v:.2f}" for k, v in sb.items())]
    for group, cs_all in curves.items():
        L += ["", f"## Curves, {group} (k per bin of a condition; * = two bins' 95 % intervals apart)", ""]
        for kd, cs in cs_all.items():
            for name, rows in cs.items():
                if not rows:
                    continue
                sep = "*" if separated(rows) else ""
                cells = "; ".join(f"[{_f(r['lo_x'])}, {_f(r['hi_x'])}): {_f(r['k'])} ({_f(r['lo'])} .. {_f(r['hi'])}, "
                                  f"n {r['n_pos']})" for r in rows)
                L.append(f"- **{kd}** by {name}{sep}: {cells}")
    if l3:
        L += ["", f"## Layer 3: prices of the network's {P} presses against the branch prices", "",
              "| ability | way | sign agree | Spearman (ceiling) | V | verdict | price > 0 | branch > 0 | "
              "mean net / branch, gold |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a in next(iter(l3.values())):
            for way, rows in l3.items():
                r = rows.get(a)
                if r is None:
                    continue
                L.append(f"| {va.short(a)} | {way} | {r['sign']:.2f} | {r['spearman']:.2f} ({r['ceiling']:.2f}) | "
                         f"{r['V']:.2f} | {r['verdict']} | {r['share_price_pos']:.2f} | {r['share_branch_pos']:.2f} | "
                         f"{r['net_mean']:.1f} / {r['branch_mean']:.1f} |")
    return "\n".join(L) + "\n"


MAIN = ("battle", "battle_cond")      # the families the lab's rates come from (whole battles); the others beside them


def report(lab_dir, paths, boot=1000, seed=0, out=None):
    """Reads lab_sim's files in lab_dir (FAMILY_FILES), writes lab.md and lab.json there. The rates: the whole battles'
    families pooled (MAIN; all families when there are none), each family's beside them; curves of the whole battles and
    of the small fights."""
    pools, units, passports, sim, rules_ = load_config()
    bricks = bz.bricks(bz.rules(rules_, sim))
    lab_dir = Path(lab_dir)
    rates, fam_rates, seconds = {}, {}, {}
    curves = {"whole battles": {}, "small fights": {}}
    for kd in KINDS:
        fams = {}
        for fam, pat in FAMILY_FILES:
            f = lab_dir / pat.format(kd)
            if f.exists():
                fams[fam] = load_family(f, units, bricks)
                seconds[f"{kd} {fam}"] = fams[fam]["seconds"]
        if not fams:
            continue
        fam_rates[kd] = {}
        for fam, fd in fams.items():
            fam_rates[kd][fam], _ = summarise([fd], boot, seed)
            fam_rates[kd][fam]["sprint_busy"] = fd["sprint_busy"]
        main = [fams[f] for f in MAIN if f in fams] or list(fams.values())
        rates[kd], pooled = summarise(main, boot, seed)
        curves["whole battles"][kd] = curves_of(kd, pooled, boot, seed)
        if "small" in fams:
            curves["small fights"][kd] = curves_of(kd, summarise([fams["small"]], 0, seed)[1], boot, seed)
    l3, reg, kinds, P = ({}, {}, [], 0)
    if paths and all(Path(p).exists() for p in paths):
        sets = {"lab": (rates, None), "lab + curves": (rates, curves["whole battles"])}
        for fam, _ in FAMILY_FILES:
            if any(fam in fr for fr in fam_rates.values()):
                sets[f"lab {fam}"] = ({kd: fr[fam] for kd, fr in fam_rates.items() if fam in fr}, None)
        l3, reg, kinds, P = layer3(paths, sets, bricks, passports, sim, seed=seed)
    md = markdown(rates, fam_rates, curves, l3, reg, kinds, P, seconds)
    (Path(out) if out else lab_dir / "lab.md").write_text(md, encoding="utf-8")
    (lab_dir / "lab.json").write_text(json.dumps({"rates": rates, "families": fam_rates, "curves": curves,
                                                  "layer3": l3, "regression": reg}, indent=1, default=float),
                                      encoding="utf-8")
    return md, rates, curves, l3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="build/abilities_price/lab")
    ap.add_argument("--data", nargs="*", default=["build/abilities_price/natural.npz", "build/abilities_price/hold.npz"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    md, *_ = report(a.dir, a.data, a.boot, a.seed)
    print(md)


if __name__ == "__main__":
    main()
