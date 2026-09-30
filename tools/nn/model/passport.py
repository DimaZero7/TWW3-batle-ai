"""Static features of a unit from its passport (config/nn/units.json): what both sides know before battle.

One vector per unit key, the same for own and enemy units (the enemy's roster is shown before
battle). Numbers are scaled to about 0-1; wide ones (health, mass, cost) on a log scale. Only numpy.
"""
import json
import math
from functools import lru_cache

import numpy as np

from tools import config as project

PASSPORTS = project.ROOT / "config" / "nn" / "units.json"

CASTES = ("lord", "hero", "melee_infantry", "missile_infantry", "melee_cavalry", "missile_cavalry",
          "monstrous_infantry", "monstrous_cavalry", "monster", "warbeast", "chariot", "war_machine")
SIZES = ("small", "medium", "large", "very_large")
ATTRIBUTES = ("encourages", "expendable", "charge_defense_vs_large", "charge_reflection",
              "hide_forest", "stalk", "causes_fear", "causes_terror", "unbreakable", "strider")
MAX_RANK = 9   # experience ranks 0-9 (game_rules.json experience_levels)


def _log(v, top):
    """log(1 + v) scaled so that v = top gives 1."""
    return math.log1p(max(float(v or 0), 0.0)) / math.log1p(top)


def _hot(value, names):
    return [float(value == n) for n in names] + [float(value not in names)]


def features(p):
    """The passport of one unit (a dict from units.json) as a list of floats."""
    speed, melee, shield = p["speed"], p["melee"], p["shield"]
    resist = p.get("damage_resist") or {}
    miss = p.get("missile") or {}
    out = [
        _log(p["men"], 200), _log(p["hp_total"], 20000), _log(p["hp_per_man"], 10000), _log(p["mass"], 10000),
        speed["walk"] / 5, speed["run"] / 10, speed["charge"] / 10,
        melee["attack"] / 100, melee["defence"] / 100, melee["charge_bonus"] / 100,
        _log(melee["damage"], 500), _log(melee["ap_damage"], 500),
        melee["bonus_v_large"] / 50, melee["bonus_v_infantry"] / 50,
        melee["attack_interval_s"] / 6, (melee["splash_max_attacks"] or 1) / 10,
        p["armour"] / 200, shield["missile_block_chance"] / 100, p["leadership"] / 100,
        *[resist.get(k, 0) / 100 for k in ("all", "physical", "missile", "magic", "flame")],
        float(bool(p.get("can_skirmish"))), float(bool(p.get("can_brace"))), float(p.get("hiding_scalar") or 1.0),
        _log(p["multiplayer_cost"], 5000),
        float(bool(miss)),
        (miss.get("range_m") or 0) / 500, _log(miss.get("ammo"), 100),
        _log(miss.get("damage"), 500), _log(miss.get("ap_damage"), 500),
        (miss.get("reload_s") or 0) / 20, (miss.get("accuracy") or 0) / 100,
        (miss.get("projectile_number") or 0) / 10,
    ]
    out += _hot(p["caste"], CASTES) + _hot(p["size"], SIZES)
    attrs = set(p.get("attributes") or ())
    out += [float(a in attrs) for a in ATTRIBUTES]
    return out


SIZE = len(features({"men": 1, "hp_total": 1, "hp_per_man": 1, "mass": 1, "armour": 0, "leadership": 0,
                     "multiplayer_cost": 0, "caste": "", "size": "",
                     "speed": {"walk": 0, "run": 0, "charge": 0},
                     "melee": {"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0,
                               "bonus_v_large": 0, "bonus_v_infantry": 0, "attack_interval_s": 0,
                               "splash_max_attacks": 1},
                     "shield": {"missile_block_chance": 0}}))


@lru_cache(maxsize=4)
def load(path=None):
    return json.loads((path or PASSPORTS).read_text(encoding="utf-8"))["units"]


def table(keys, passports=None):
    """Features of units by key: [len(keys), SIZE] float32; an empty key (padding) gives zeros."""
    passports = passports or load()
    out = np.zeros((len(keys), SIZE), dtype=np.float32)
    for i, k in enumerate(keys):
        if k:
            out[i] = features(passports[k])
    return out


def men(keys, passports=None):
    """Men at full strength per unit (0 for padding)."""
    passports = passports or load()
    return np.array([passports[k]["men"] if k else 0 for k in keys], dtype=np.float32)


def ammo(keys, passports=None):
    """Projectiles of the whole unit at full strength (0: no missile weapon)."""
    passports = passports or load()
    return np.array([(passports[k].get("missile") or {}).get("ammo", 0) * passports[k]["men"] if k else 0
                     for k in keys], dtype=np.float32)


def lords(keys, passports=None):
    """True for the army's general (caste lord) per unit; False for padding."""
    passports = passports or load()
    return np.array([bool(k) and passports[k].get("caste") == "lord" for k in keys], dtype=bool)
