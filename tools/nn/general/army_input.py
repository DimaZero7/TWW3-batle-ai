"""The general's army input, "Армии" (design/general/index.html, "Слоты армий", "Слот отряда", "Считаем сами",
"Правила входа «Армии»"): 80 slots, 40 of the side the general plays and 40 of the enemy, one unit a slot.

Always as the general's side sees it: slots 0..39 its own army, 40..79 the enemy's; for side 2 the armies swap places
(as the map's layers do, map_input.py). The slots' order means nothing (the lord need not be in slot 0): the army
reader (army_reader.py) reads every slot with the same weights. An empty slot is flagged `empty` and holds zeros.
Only what is known before the battle: no place, facing or visibility (the enemy's roster is shown); the state group is a
fresh unit's until units.dynamic puts the battle's in.

A slot = NUMBERS (49 floats, the table's groups in its order) + LABELS (3 indices, each read as a learned vector):

    group                 numbers                                                      scale
    side                  own 1 / enemy 0                                              flag
    cost                  multiplayer cost                                             log, 5000 -> 1
    size and health (5)   men in the full unit, men come / full, health of the full     log 200; share; log 20000;
                          unit, health come / full, health of a man                    share; log 10000
    body and speed (4)    mass of a man, walk, run, charge speed                       log 10000; m/s / 5, / 10, / 10
    melee (10)            attack, defence, charge bonus, weapon damage, its AP part,   / 100 x3; log 500 x2;
                          bonus v large, v infantry, time between blows, splash        / 50 x2; s / 6; / 10;
                          targets, magical attacks                                     flag
    defence, morale (8)   armour, shield's block chance, leadership, resistances:      / 200; % / 100; / 100;
                          all, physical, missile, magic, fire                          % / 100 x5 (fire may be < 0)
    missile (12 + label)  range, ammo a man, projectile damage, its AP part, reload,   m / 500; log 100; log 500 x2;
                          spread at 100 m (computed), projectiles a shot, shots a      s / 20; m / 10; / 10; / 20;
                          volley, projectile speed, fire arc each side, aim before     m/s / 100; deg / 180;
                          the first shot, silence after a new target                   s / 10; s / 10
    properties (4)        expendable, charge reflection, unbreakable, fires on the move  flags
    state (4)             morale (MoralePercent), wavering or worse (morale state 5-7),  / 2 (within -1.5..1.5);
                          fatigue state (0 fresh .. 5 exhausted), ammunition come /     flag; / 5; share
                          full (0 without a missile weapon)
    labels                trajectory (TRAJECTORIES; 0 = no missile weapon), caste (CASTES, 12), size (SIZES, 4)

The state group (STATE, appended last: a checkpoint read 45 numbers a slot, its reader gets zero weights for these,
army_reader.py) is the battle's now (units.dynamic: the simulator's mp, ms, fat, a; the companion's from the game's
state, tools/nn/companion/general.py) for own units and the enemies seen; before the battle and for a hidden enemy the
fresh unit's: morale 1 (0.5), not wavering, fresh, its ammunition full (1; 0 without a missile weapon). A shooter is a
shooter only while it has ammunition (armed(): range > 0 and ammunition come > 0, the simulator's rule: battle.py
fires while `a` > 0) - the general's shooters' shares, the "shoot" mask and the tiles' shooters' strength.

The scales are the unit network's passport's (tools/nn/model/passport.py) wherever it has the field; new ones: the
come shares (0..1 already), the spread at 100 m (m / SPREAD_SCALE_M: handguns 0.30, crossbows 0.44, slings 0.60,
throwing stars 0.78), shots a volley / passport.VOLLEY_SHOTS and the aim times / passport.AIM_S (the unit token's
WEAPON / AIM columns), the fire arc / 180 (the token's ARC). A unit without a missile weapon has 0 in every missile number and trajectory 0, whose vector is
fixed at zero (army_reader.py).

No "other" variant: a caste, size or trajectory not in the lists is an error when the data is built (UnknownVariant),
not a silent "other". A new variant is appended to its list (old indices keep their places) and the reader gets a
new vector with zero weights (ArmyReader.add_variant): the old cards stay the same to the bit.

Where the numbers come from (final_stats): the card's final numbers (rank, items, skills already inside) - the bridge
will read them from the game's unit card later and pass them as `card` {CARD_KEYS name: value}; what the card lacks
comes from the database's passport (config/nn/units.json) or is computed ("Считаем сами"):
* spread at 100 m: the simulator's own spread (tools/nn/sim/missile.py square_side, config/nn/sim.json
  missile.accuracy.spread "square") at d = 100 m - the side of the uniform square in the target's plane,
  sqrt(calibration_area / p), p = a / 100 x (calibration_distance / 100)^2 x (1 - launch angle / 180 deg) within
  [0.01, 1], a = the land unit's accuracy + the projectile's marksmanship + the rank's accuracy (game_rules.json
  experience_bonus stat_accuracy: +3 a rank) - the card shows no accuracy. (The design page's wording "weapon's
  spread x (1 - accuracy / 100)" is the older Gaussian "radius" reading; with spread "gauss" this gives the Gaussian's
  equal-variance side sqrt 12 x sigma, sigma cut by acc_shrink's rule, so the number means the same in both.)
* walk and charge speed: the database's x the card's run / the database's run (1 now);
* men and health come: the card's now / the passport's full (1 in our battles).
With no card (the simulator, our battles) the passport's numbers are the final ones: rank 0, nothing missing.

    build(armies, side)        -> ArmyInput: numbers [80, 49] float32, labels [80, 3] int64, empty [80] bool
    stack([ArmyInput, ...])    -> the same with a batch axis
    python -m tools.nn.general.army_input        # the slot of every unit of units.json, one line each
"""
import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tools import config as project
from tools.nn.model import passport as pp

SIDE_SLOTS = 40
SLOTS = 2 * SIDE_SLOTS

# --- the labels: fixed lists, new variants appended at the end (old indices keep their places) ---------------------
CASTES = pp.CASTES                    # 12: lord, hero, melee_infantry, ..., war_beast, chariot, war_machine
SIZES = pp.SIZES                      # 4: small, medium, large, very_large
# The flight (the database's projectiles.trajectory_sight): index 0 is "no missile weapon" (a zero vector, not a
# variant). direct: flat fire, hits the first in its line, never over friends (low: guns, militia, throwing stars);
# high: a fixed high arc over obstacles (fixed: mortars, catapults - none of ours); dual: flat, over friends an arc
# (dual_low_fixed: bows, crossbows, slings). The database's "high" is spells' only: an error until a unit has it.
TRAJECTORIES = ("none", "direct", "high", "dual")
DB_TRAJECTORY = {"low": "direct", "fixed": "high", "dual_low_fixed": "dual"}
LABELS = ("trajectory", "caste", "size")
VARIANTS = {"trajectory": TRAJECTORIES, "caste": CASTES, "size": SIZES}
# the properties: rules for the unit itself (passport attributes); the others are gone from the input (design page,
# "Доработки") or go through "Эффекты"
PROPERTIES = (("expendable", "expendable"), ("charge_reflection", "charge_reflection"),
              ("unbreakable", "unbreakable"), ("fire_on_move", "mounted_fire_move"))

SPREAD_D_M = 100.0       # the spread's distance
SPREAD_SCALE_M = 10.0    # its scale: m / this
RESISTS = ("all", "physical", "missile", "magic", "flame")

# (name, group, scale): the slot's numbers in order; scale "flag", "share", ("log", top) or a divisor
NUMBERS = (
    ("side", "side", "flag"),
    ("cost", "cost", ("log", 5000)),
    ("men_full", "size", ("log", 200)), ("men_come", "size", "share"),
    ("hp_full", "size", ("log", 20000)), ("hp_come", "size", "share"), ("hp_per_man", "size", ("log", 10000)),
    ("mass", "body", ("log", 10000)), ("walk", "body", 5), ("run", "body", 10), ("charge", "body", 10),
    ("attack", "melee", 100), ("defence", "melee", 100), ("charge_bonus", "melee", 100),
    ("damage", "melee", ("log", 500)), ("ap_damage", "melee", ("log", 500)),
    ("bonus_v_large", "melee", 50), ("bonus_v_infantry", "melee", 50), ("attack_interval", "melee", 6),
    ("splash", "melee", 10), ("magic", "melee", "flag"),
    ("armour", "defence", 200), ("shield", "defence", 100), ("leadership", "defence", 100),
    *((f"resist_{k}", "defence", 100) for k in RESISTS),
    ("range", "missile", 500), ("ammo", "missile", ("log", 100)),
    ("missile_damage", "missile", ("log", 500)), ("missile_ap", "missile", ("log", 500)),
    ("reload", "missile", 20), ("spread_100", "missile", SPREAD_SCALE_M),
    ("projectiles", "missile", 10), ("volley_shots", "missile", pp.VOLLEY_SHOTS),
    ("muzzle_velocity", "missile", 100), ("fire_arc", "missile", 180),
    ("aim", "missile", pp.AIM_S), ("silence", "missile", pp.AIM_S),
    *((name, "properties", "flag") for name, _ in PROPERTIES),
    # the battle's state now (module doc; appended last: older checkpoints read the 45 above)
    ("morale", "state", 2), ("wavering", "state", "flag"), ("fatigue", "state", 5), ("ammo_come", "state", "share"),
)
NAMES = tuple(n for n, _, _ in NUMBERS)
INDEX = {n: i for i, n in enumerate(NAMES)}
SIZE = len(NUMBERS)                   # 49
STATE = tuple(n for n, g, _ in NUMBERS if g == "state")
OLD_SIZE = SIZE - len(STATE)          # 45: the slot before the state group
MORALE_CLIP = 1.5                     # morale / 2 within +- this (the unit token's: observation.MORALE)
SIDE = INDEX["side"]
# what every number may span on real units (test_numbers_span): fire resistance may be negative
SPAN = (-1.0, 1.5)

# The card's final numbers the bridge may give (final_stats card): the slot's numbers from the card, plus the unit's
# rank (its accuracy) and what is now come (men_now, hp_now). Not here: what is computed (walk, charge: by the run;
# spread_100: by accuracy and rank; the come shares) and what the card has no line for.
CARD_KEYS = frozenset({"rank", "men_now", "hp_now", "men_full", "hp_full", "hp_per_man", "mass", "run", "cost",
                       "attack", "defence", "charge_bonus", "damage", "ap_damage", "bonus_v_large",
                       "bonus_v_infantry", "attack_interval", "splash", "magic", "armour", "shield", "leadership",
                       *(f"resist_{k}" for k in RESISTS), "range", "ammo", "missile_damage", "missile_ap", "reload",
                       "projectiles", "volley_shots", "muzzle_velocity", "fire_arc",
                       *(name for name, _ in PROPERTIES)})


def armed(num):
    """[...] bool: a shooter with ammunition left - range > 0 and ammunition come > 0 (module doc) - of slot numbers
    num [..., SIZE] (numpy or torch)."""
    return (num[..., INDEX["range"]] > 0) & (num[..., INDEX["ammo_come"]] > 0)


class UnknownVariant(ValueError):
    """A unit's caste, size or trajectory is not in its list: add a new variant (module doc), never "other"."""


@lru_cache(maxsize=1)
def _sim_missile():
    """config/nn/sim.json's missile section (the aim times, the spread's rule)."""
    from tools.nn.sim import params as sim_params
    return sim_params.load(units=pp.PASSPORTS).sim["missile"]


@lru_cache(maxsize=1)
def rank_accuracy():
    """Accuracy a rank adds (config/nn/game_rules.json experience_bonus stat_accuracy growth_scalar: 3; the lonerank
    probe fits rank 9 +18 better than the database's +27 - open, the simulator has no ranks yet)."""
    rules = json.loads((project.CONFIG_DIR / "nn" / "game_rules.json").read_text(encoding="utf-8"))
    return float(rules["experience_bonus"]["stat_accuracy"]["b"])


def spread_100(accuracy, cal_area, cal_dist, muzzle_v, rule=None, d=SPREAD_D_M):
    """The spread at d m, m (module doc): the side of the simulator's uniform square (missile.accuracy.spread
    "square", tools/nn/sim/missile.py square_side), or with "gauss" the Gaussian's equal-variance side sqrt 12 x
    sigma (missile._spread, acc_shrink). 0 without a calibration (no missile weapon)."""
    if not cal_dist or not cal_area or cal_area <= 0:
        return 0.0
    rule = rule if rule is not None else _sim_missile()["accuracy"]
    v = max(float(muzzle_v or 0), 1.0)
    theta = max(0.5 * math.asin(min(d * 9.81 / (v * v), 1.0)), 1e-3)
    if rule.get("spread") == "square":
        p = min(max(accuracy / 100 * (cal_dist / d) ** 2 * (1 - theta / math.pi), 0.01), 1.0)
        return max(math.sqrt(cal_area / p), 1e-3)
    a = max(1 - accuracy / 100, 0.0)
    shrink = a if rule.get("shrink") == "radius" else math.sqrt(a)
    return max(math.sqrt(cal_area) * d / max(cal_dist, 1.0) * shrink, 1e-3) * math.sqrt(12.0)


def variant(kind, value, unit=""):
    """The index of a label's value; UnknownVariant if it is not in the list."""
    names = VARIANTS[kind]
    if value not in names or (kind == "trajectory" and value == "none"):
        raise UnknownVariant(f"unit {unit!r}: {kind} {value!r} is not in {names[1:] if kind == 'trajectory' else names}"
                             f" - add a new variant (army_input.py), there is no 'other'")
    return names.index(value)


def trajectory_of(missile, unit=""):
    if not missile:
        return "none"
    t = missile.get("trajectory")
    if t not in DB_TRAJECTORY:
        raise UnknownVariant(f"unit {unit!r}: the database's trajectory {t!r} is not in {tuple(DB_TRAJECTORY)}"
                             f" - add a new variant (army_input.py), there is no 'other'")
    return DB_TRAJECTORY[t]


def base_stats(p, unit=""):
    """The database's numbers of a passport (config/nn/units.json) as final_stats' names: rank 0, nothing missing."""
    speed, melee, resist = p["speed"], p["melee"], p.get("damage_resist") or {}
    miss = p.get("missile") or {}
    attrs = set(p.get("attributes") or ())
    s = {
        "cost": p["multiplayer_cost"], "men_full": p["men"], "men_now": p["men"],
        "hp_full": p["hp_total"], "hp_now": p["hp_total"], "hp_per_man": p["hp_per_man"], "mass": p["mass"],
        "walk": speed["walk"], "run": speed["run"], "charge": speed["charge"],
        "attack": melee["attack"], "defence": melee["defence"], "charge_bonus": melee["charge_bonus"],
        "damage": melee["damage"], "ap_damage": melee["ap_damage"],
        "bonus_v_large": melee["bonus_v_large"], "bonus_v_infantry": melee["bonus_v_infantry"],
        "attack_interval": melee["attack_interval_s"], "splash": melee["splash_max_attacks"] or 1,
        "magic": float(bool(melee.get("magic"))),
        "armour": p["armour"], "shield": (p.get("shield") or {}).get("missile_block_chance", 0),
        "leadership": p["leadership"], **{f"resist_{k}": resist.get(k, 0) for k in RESISTS},
        "range": miss.get("range_m") or 0, "ammo": miss.get("ammo") or 0,
        "missile_damage": miss.get("damage") or 0, "missile_ap": miss.get("ap_damage") or 0,
        "reload": miss.get("reload_s") or 0, "projectiles": miss.get("projectile_number") or 0,
        "volley_shots": miss.get("shots_per_volley") or 0, "muzzle_velocity": miss.get("muzzle_velocity") or 0,
        "fire_arc": miss.get("fire_arc_deg") or 0,
        **{name: float(attr in attrs) for name, attr in PROPERTIES},
        # the state of a fresh unit (module doc: the battle's now replaces it, units.dynamic)
        "morale": 1.0, "wavering": 0.0, "fatigue": 0.0, "ammo_come": 1.0 if miss.get("ammo") else 0.0,
        "rank": 0,
        # not on the slot themselves: the spread's makings (the card shows none of them)
        "accuracy": (miss.get("accuracy", 0) + miss.get("marksmanship", 0)) if miss else 0,
        "cal_area": miss.get("calibration_area_m") or 0, "cal_dist": miss.get("calibration_distance_m") or 0,
        "trajectory": trajectory_of(miss, unit), "caste": p["caste"], "size": p["size"],
    }
    if miss:
        from tools.nn.sim import params as sim_params
        ms = _sim_missile()
        aim_s, rt_s, quiet_s = sim_params.aim_times(miss, ms)
        s["aim"], s["silence"] = aim_s, sim_params.retarget_silence(aim_s, rt_s, quiet_s, ms)
    else:
        s["aim"] = s["silence"] = 0.0
    return s


def final_stats(p, card=None, unit="", rule=None):
    """The unit's final numbers (module doc): the passport's, the card's over them (CARD_KEYS; another key is an
    error), then the computed ones - walk and charge by the run's change, the spread at 100 m by the accuracy and
    the rank, the come shares."""
    base = base_stats(p, unit)
    s = dict(base)
    if card:
        bad = set(card) - CARD_KEYS
        if bad:
            raise KeyError(f"unit {unit!r}: the card's {sorted(bad)} are not card numbers (army_input.CARD_KEYS)")
        s.update(card)
    k = s["run"] / base["run"] if base["run"] else 1.0
    s["walk"], s["charge"] = base["walk"] * k, base["charge"] * k
    s["spread_100"] = spread_100(base["accuracy"] + rank_accuracy() * s["rank"], base["cal_area"], base["cal_dist"],
                                 s["muzzle_velocity"], rule) if base["range"] else 0.0
    s["men_come"] = s["men_now"] / s["men_full"] if s["men_full"] else 0.0
    s["hp_come"] = s["hp_now"] / s["hp_full"] if s["hp_full"] else 0.0
    return s


def _scaled(v, scale):
    if scale in ("flag", "share"):
        return float(v)
    if isinstance(scale, tuple):
        return pp._log(v, scale[1])
    return float(v) / scale


def slot(stats, own, unit=""):
    """(numbers [SIZE] float32, labels [3] int64) of a unit's final numbers (final_stats); own: the general's side."""
    s = dict(stats, side=float(bool(own)))
    nums = np.array([_scaled(s[n], scale) for n, _, scale in NUMBERS], dtype=np.float32)
    if not np.isfinite(nums).all():
        raise ValueError(f"unit {unit!r}: a slot number is not finite: {dict(zip(NAMES, nums))}")
    traj = s["trajectory"]
    labels = np.array([0 if traj == "none" else variant("trajectory", traj, unit),
                       variant("caste", s["caste"], unit), variant("size", s["size"], unit)], dtype=np.int64)
    return nums, labels


@dataclass
class ArmyInput:
    """numbers [.., 80, SIZE] float32, labels [.., 80, 3] int64 (LABELS), empty [.., 80] bool."""
    numbers: np.ndarray
    labels: np.ndarray
    empty: np.ndarray


def _entry(e):
    """A slot's entry: None / "" (empty), a unit key, or {"key", "card"} -> (key or None, card or None)."""
    if not e:
        return None, None
    if isinstance(e, str):
        return e, None
    return e["key"], e.get("card")


def build(armies, side=1, passports=None, rule=None):
    """The 80 slots as side `side` sees them. armies {1: [...], 2: [...]} (or a pair): each side's entries in slot
    order (_entry), at most 40; the rest of the 40 are empty."""
    passports = passports or pp.load()
    if not isinstance(armies, dict):
        armies = {1: armies[0], 2: armies[1]}
    assert side in (1, 2), side
    nums = np.zeros((SLOTS, SIZE), np.float32)
    labels = np.zeros((SLOTS, len(LABELS)), np.int64)
    empty = np.ones(SLOTS, bool)
    for half, s in enumerate((side, 3 - side)):
        entries = list(armies.get(s) or ())
        if len(entries) > SIDE_SLOTS:
            raise ValueError(f"side {s}: {len(entries)} units, at most {SIDE_SLOTS} on the field")
        for i, e in enumerate(entries):
            key, card = _entry(e)
            if key is None:
                continue
            if key not in passports:
                raise KeyError(f"unit {key!r} has no passport")
            j = half * SIDE_SLOTS + i
            nums[j], labels[j] = slot(final_stats(passports[key], card, key, rule), own=half == 0, unit=key)
            empty[j] = False
    return ArmyInput(nums, labels, empty)


def stack(items):
    """[ArmyInput] -> ArmyInput with a batch axis first."""
    return ArmyInput(np.stack([a.numbers for a in items]), np.stack([a.labels for a in items]),
                     np.stack([a.empty for a in items]))


def to_tensors(a, device=None):
    """(numbers float32, labels int64, empty bool) torch tensors of an ArmyInput."""
    import torch
    return (torch.as_tensor(a.numbers, device=device), torch.as_tensor(a.labels, device=device),
            torch.as_tensor(a.empty, device=device))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    passports = pp.load()
    print(f"{SIZE} numbers + {len(LABELS)} labels a slot")
    print("unit".ljust(45) + " ".join(n[:7].rjust(7) for n in NAMES))
    for key, p in passports.items():
        nums, labels = slot(final_stats(p, unit=key), own=True, unit=key)
        print(key[:44].ljust(45) + " ".join(f"{v:7.3f}" for v in nums), labels.tolist())


if __name__ == "__main__":
    main()
