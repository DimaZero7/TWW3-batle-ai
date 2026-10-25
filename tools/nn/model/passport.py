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

# (the database's caste of war beasts is "war_beast": "warbeast" here until 11.10.2026 sent the Giant Rats to "other";
# its column keeps its place)
CASTES = ("lord", "hero", "melee_infantry", "missile_infantry", "melee_cavalry", "missile_cavalry",
          "monstrous_infantry", "monstrous_cavalry", "monster", "war_beast", "chariot", "war_machine")
SIZES = ("small", "medium", "large", "very_large")
ATTRIBUTES = ("encourages", "expendable", "charge_defense_vs_large", "charge_reflection",
              "hide_forest", "stalk", "causes_fear", "causes_terror", "unbreakable", "strider",
              # 02.10.2026 (appended: older checkpoints load with zero weights for them, encoder.py)
              "mounted_fire_move", "guerrilla_deploy")
# Shooting's flight, appended after the attributes on 02.10.2026: direct (flat, trajectory low: needs a
# clear line past friends), spread = calibration_area / calibration_distance x 20 (arrow 0.78, pistol
# 0.62), muzzle velocity / 100 m/s.
FLIGHT = ("direct", "spread", "muzzle_velocity")
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
    cal_d = miss.get("calibration_distance_m") or 0
    out += [float(bool(miss.get("direct"))),
            20 * (miss.get("calibration_area_m") or 0) / cal_d if cal_d else 0.0,
            (miss.get("muzzle_velocity") or 0) / 100]
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
    """The unit passports (config/nn/units.json). The default file's also the casters the armies may field
    (config/nn/pools.json "mages", their passports in data/units/probe-units.json; whatever the share: the network
    knows them as it knows any unit, the simulator's units get them only with a share above 0, sim/params.py)."""
    units = json.loads((path or PASSPORTS).read_text(encoding="utf-8"))["units"]
    if path is not None:
        return units
    from tools.nn.armies import pools
    m = json.loads(pools.POOLS.read_text(encoding="utf-8")).get("mages") or {}
    extra = pools.mage_passports_of(m)
    return {**units, **{k: v for k, v in extra.items() if k not in units}}     # (units.json's first, in its order)


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


def reload(keys, passports=None):
    """The missile weapon's reload_s per unit (0: no missile weapon, or padding): the observation's VOLLEY."""
    passports = passports or load()
    return np.array([float((passports[k].get("missile") or {}).get("reload_s") or 0) if k else 0
                     for k in keys], dtype=np.float32)


def fire_arc(keys, passports=None):
    """The fire arc per unit, each side of the facing / 180 deg (0: no missile weapon, or padding): the token's ARC
    (tools/nn/model/observation.py, 07.10.2026) - archers and slings 30 deg (0.17), militia 35, Night Runners'
    throwing stars 180 (1.0: they shoot all round, on the move too). The passport's missile.fire_arc_deg."""
    passports = passports or load()
    return np.array([float((passports[k].get("missile") or {}).get("fire_arc_deg") or 0) / 180 if k else 0
                     for k in keys], dtype=np.float32)


SIZE_RANK = {"very_small": 0, "small": 1, "medium": 2, "large": 3, "very_large": 4}   # (tools/nn/sim/state.py's)
PIERCE_MEN = 4.0      # SIGHT's men a projectile strikes past the first / this (the Jezzails' 1 -> 0.25)
SPOT_M = 160.0        # SIGHT's spotting distance / this (the Jezzails' 160 -> 1, the rest 80 -> 0.5)


def men_struck(p, size="small", resistance=1, limit=8):
    """The men one projectile of passport p strikes in a formation of men of this size and penetration resistance
    (the game's rule, tools/nn/sim/missile.py men_struck): 1 without a missile weapon."""
    m = p.get("missile") or {}
    mx = m.get("penetration_max") or 0
    cap = m.get("penetration_cap")
    cap_r = SIZE_RANK.get(cap, 99) if cap else (99 if mx < 0 else 0)
    if not m or SIZE_RANK.get(size, 1) >= cap_r:
        return 1
    return int(limit if mx < 0 else min(limit, 1 + mx // max(float(resistance), 1e-3)))


def sight(keys, passports=None):
    """[n, 2] the token's SIGHT passport numbers (tools/nn/model/observation.py, 25.10.2026, line g43; both sides: the
    roster's passports): the infantrymen one projectile strikes past the first (men_struck against a small man of
    resistance 1, less 1) / PIERCE_MEN, and the spotting distance (land_units spot_dist_scrub) / SPOT_M; padding 0."""
    passports = passports or load()
    out = np.zeros((len(keys), 2), np.float32)
    for i, k in enumerate(keys):
        if k:
            p = passports[k]
            out[i] = ((men_struck(p) - 1) / PIERCE_MEN, float((p.get("spotting") or {}).get("scrub", 80.0)) / SPOT_M)
    return out


VOLLEY_SHOTS = 20.0   # the volley's shots a man / this (the Ratling Gun's 18 -> 0.9)


def weapon(keys, passports=None):
    """[len(keys), 2] float32: the token's WEAPON columns (tools/nn/model/observation.py, 11.10.2026) - the shots a man
    fires in a volley (missile.shots_per_volley / VOLLEY_SHOTS; 0 without a missile weapon) and the melee weapon's
    magical attacks (melee.magic, 0 / 1); 0 for padding."""
    passports = passports or load()
    out = np.zeros((len(keys), 2), dtype=np.float32)
    for i, k in enumerate(keys):
        if k:
            p = passports[k]
            miss = p.get("missile") or {}
            out[i] = ((miss.get("shots_per_volley") or 0) / VOLLEY_SHOTS if miss else 0.0,
                      float(bool((p.get("melee") or {}).get("magic"))))
    return out


AIM_S = 10.0   # the AIM columns' seconds / this (the Ratling Gun's 4.5 s spin-up -> 0.45)


def aim(keys, passports=None, missile_cal=None):
    """[len(keys), 2] float32: the token's AIM columns (tools/nn/model/observation.py, 13.10.2026), seconds / AIM_S:
    - the aim before the first shot from a halt (the simulator's aim_s: the Ratling Gun's spin-up 4.5, arrows 3.3,
      slings 4.3, muskets 2.0, the rest 3.8);
    - the silence after an attack order on another target (sim params.retarget_silence: arrows and slings 8.5 - their
      quiet window, the Ratling Gun 4.5, muskets 0, the rest 3.8); the reload is not reset by it, it runs on.
    0 without a missile weapon, and for padding. The same numbers as the simulator's (tools/nn/sim/params.aim_times,
    config/nn/sim.json missile; missile_cal: that section, default today's file): static per unit key, so the companion
    gets them from the key as the simulator does."""
    from tools.nn.sim import params as sim_params
    passports = passports or load()
    ms = missile_cal if missile_cal is not None else sim_params.load().sim["missile"]
    out = np.zeros((len(keys), 2), dtype=np.float32)
    for i, k in enumerate(keys):
        miss = passports[k].get("missile") if k else None
        if miss:
            aim_s, rt_s, quiet_s = sim_params.aim_times(miss, ms)
            out[i] = (aim_s / AIM_S, sim_params.retarget_silence(aim_s, rt_s, quiet_s, ms) / AIM_S)
    return out


def cost(keys, passports=None):
    """Multiplayer cost per unit, gold (0 for padding): the simulator's `cost` (tools/nn/sim/params.py)."""
    passports = passports or load()
    return np.array([passports[k].get("multiplayer_cost", 0) if k else 0 for k in keys], dtype=np.float32)


def combat_potential(keys, passports=None, abilities=None):
    """(fixed, missile) [len(keys)] float32: the database's combat potential per unit (0 for padding) - main_units
    melee_cp + its abilities' cp (config/nn/abilities.json cp), and missile_cp (counts by the ammunition left): the
    simulator's cp_fixed / cp_missile (tools/nn/sim/params.py, the army destruction's strength). A passport without
    combat_potential counts its cost as fixed."""
    passports = passports or load()
    if abilities is None:
        path = PASSPORTS.parent / "abilities.json"
        abilities = json.loads(path.read_text(encoding="utf-8"))["abilities"] if path.exists() else {}
    fixed, missile = [], []
    for k in keys:
        p = passports[k] if k else {}
        cp = p.get("combat_potential")
        abil = sum(sum((abilities.get(a) or {}).get("cp") or (0, 0)) for a in p.get("abilities") or ())
        fixed.append((cp["melee"] if cp else p.get("multiplayer_cost", 0)) + abil if k else 0)
        missile.append(cp["missile"] if cp and k else 0)
    return np.array(fixed, dtype=np.float32), np.array(missile, dtype=np.float32)


def lords(keys, passports=None):
    """True for the army's general (caste lord) per unit; False for padding."""
    passports = passports or load()
    return np.array([bool(k) and passports[k].get("caste") == "lord" for k in keys], dtype=bool)
