"""Static features of an ability from its passport (config/nn/abilities.json): what both sides know
before battle (a unit's abilities are on its card, the enemy's too). Only numpy.

An ability is described by what it does, never by its key: times, reach, targets and its effects on
allies (the owner and his friends) and on enemies, one number per stat of the game's database
(STATS) and a flag per attribute it gives (ATTRIBUTES). A stat or attribute not listed counts for
nothing; a new ability with listed stats needs only its passport. Numbers are scaled to about -1..1:
multipliers as value - 1, additions divided by a typical size.

The slots of a unit (which of its abilities, in what order) are the simulator's
(tools/nn/sim/abilities.py slot_keys): the network's slot k is the simulator's slot k.
"""
import json
from functools import lru_cache

import numpy as np

from tools import config as project
from tools.nn.model import passport as unit_passport
from tools.nn.sim.abilities import SLOTS, slot_keys

PASSPORTS = project.ROOT / "config" / "nn" / "abilities.json"

# (stat, how, scale): how = mult -> value - 1 (multiplied over effects); add -> sum / scale
STATS = (
    ("scalar_speed", "mult", 1), ("scalar_charge_speed", "mult", 1),
    ("stat_melee_attack", "add", 50), ("stat_melee_attack", "mult", 1),
    ("stat_melee_defence", "add", 50), ("stat_melee_defence", "mult", 1),
    ("stat_melee_damage_base", "mult", 1), ("stat_melee_damage_ap", "mult", 1),
    ("stat_charge_bonus", "mult", 1), ("stat_charge_bonus", "add", 50),
    ("stat_morale", "add", 50), ("stat_armour", "add", 100),
    ("stat_resistance_all", "add", 100), ("stat_resistance_physical", "add", 100),
    ("stat_resistance_missile", "add", 100), ("stat_resistance_magic", "add", 100),
    ("stat_resistance_flame", "add", 100), ("stat_weakness_flame", "add", 100),
    ("scalar_missile_damage_base", "mult", 1), ("scalar_missile_damage_ap", "mult", 1),
    ("stat_reloading", "add", 100), ("stat_accuracy", "add", 100), ("scalar_missile_range", "mult", 1),
    ("stat_missile_block_chance", "add", 100), ("stat_bonus_vs_large", "add", 50),
    ("stat_bonus_vs_infantry", "add", 50), ("stat_mass", "mult", 1), ("stat_damage_reflection", "add", 100),
)
ATTRIBUTES = ("immune_to_psychology", "unbreakable", "rampage", "stalk", "causes_terror", "causes_fear",
              "unspottable", "charge_defense", "cannot_die", "silenced", "strider", "invulnerable",
              "fatigue_immune", "devastating_flanker", "charge_reflection")
GROUPS = ("allies", "enemies")     # allies: the owner and his friends (phase targets self or friends)
HEAD = ("passive", "active_s", "recharge_s", "uses_limited", "uses", "range_m", "self_cast", "targets_own",
        "friendly_all", "friendly_n", "enemy_all", "enemy_n", "target_self", "target_friends", "target_enemies")
# When the game fires it and when it is off (02.10.2026; appended last so that older checkpoints load
# with zero weights for them, encoder.py): auto = a timed passive the game fires by itself (never by
# order), its context (auto_when) and the conditions that hold it off (off_when), from the database.
WHEN = (("auto", None), ("when_losing_melee", "losing_melee_combat"), ("when_in_melee", "engaged_in_melee"),
        ("off_out_of_melee", "out_of_melee"), ("off_morale_below_half", "morale_is_lower_than_half_of_base_morale"),
        ("off_not_wavering", "morale_is_higher_than_wavering"), ("off_health_below_half", "health_below_50%_base"),
        ("other_condition", None))
STATIC_NAMES = HEAD + tuple(f"{g}_{s}_{h}" for g in GROUPS for s, h, _ in STATS) + tuple(
    f"{g}_{a}" for g in GROUPS for a in ATTRIBUTES) + tuple(n for n, _ in WHEN)
STATIC = len(STATIC_NAMES)
# The state of a slot in battle (tools/nn/model/observation.py): owned, ready to use, seconds until
# ready / 60, seconds active left / 30, active now. Enemies: owned and active now (while seen).
DYNAMIC_NAMES = ("owned", "ready", "recharge_left", "active_left", "active_now")
DYNAMIC = len(DYNAMIC_NAMES)
SIZE = DYNAMIC + STATIC
INDEX = {n: i for i, n in enumerate(DYNAMIC_NAMES + STATIC_NAMES)}
RECHARGE, ACTIVE = 60.0, 30.0


def _group(on):
    return [g for g, hit in (("allies", "self" in on or "friends" in on), ("enemies", "enemies" in on)) if hit]


def features(p):
    """The passport of one ability (a dict from abilities.json) as a list of floats."""
    t = p.get("targets") or {}
    uses, friendly, enemy = p.get("uses", -1), p.get("friendly_units", 0), p.get("enemy_units", 0)
    passive = bool(p.get("passive"))
    out = [float(passive), 0.0 if passive else max(p.get("active_s", 0), 0) / 60,
           max(p.get("recharge_s", 0), 0) / 120, float(uses > 0), max(uses, 0) / 5,
           max(p.get("range_m", 0), 0) / 100, float(bool(p.get("self_cast"))), float(bool(p.get("targets_own"))),
           float(friendly == -1), max(friendly, 0) / 10, float(enemy == -1), max(enemy, 0) / 10,
           float(bool(t.get("self"))), float(bool(t.get("friends"))), float(bool(t.get("enemies")))]
    mult = {(g, s, h): 1.0 for g in GROUPS for s, h, _ in STATS if h == "mult"}
    add = {(g, s, h): 0.0 for g in GROUPS for s, h, _ in STATS if h == "add"}
    for e in p.get("effects") or ():
        for g in _group(e.get("on") or ()):
            k = (g, e["stat"], e["how"])
            if k in mult:
                mult[k] *= float(e["value"])
            elif k in add:
                add[k] += float(e["value"])
    for g in GROUPS:
        for s, h, scale in STATS:
            out.append(mult[(g, s, h)] - 1 if h == "mult" else add[(g, s, h)] / scale)
    given = {(g, a["attribute"]) for a in p.get("attributes") or () for g in _group(a.get("on") or ())}
    for g in GROUPS:
        out += [float((g, a) in given) for a in ATTRIBUTES]
    conds = set(p.get("auto_when") or ()) | set(p.get("off_when") or ())
    known = {c for _, c in WHEN if c}
    out += [float(bool(p.get("auto")))] + [float(c in conds) for n, c in WHEN[1:-1]] + [float(bool(conds - known))]
    return out


assert len(features({})) == STATIC


@lru_cache(maxsize=4)
def load(path=None):
    return json.loads((path or PASSPORTS).read_text(encoding="utf-8"))["abilities"]


def slots(unit_keys, units=None, passports=None):
    """For units by key: (features [n, SLOTS, STATIC] float32, owned [n, SLOTS] bool, usable [n, SLOTS]
    bool: an active self-cast ability the network may order). Padding ("") and units without
    abilities give zeros."""
    units = units or unit_passport.load()
    passports = passports or load()
    n = len(unit_keys)
    feat = np.zeros((n, SLOTS, STATIC), np.float32)
    owned = np.zeros((n, SLOTS), bool)
    usable = np.zeros((n, SLOTS), bool)
    for i, key in enumerate(unit_keys):
        if not key:
            continue
        for k, a in enumerate(slot_keys(key, units, passports)):
            if a:
                p = passports[a]
                feat[i, k] = features(p)
                owned[i, k] = True
                usable[i, k] = bool(p.get("self_cast")) and not p.get("passive")
    return feat, owned, usable
