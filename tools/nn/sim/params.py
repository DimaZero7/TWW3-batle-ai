"""The simulator's numbers, from three files (docs/en/training/simulator.md):

    config/nn/units.json       unit passports (the game's database)
    config/nn/abilities.json   ability passports (the game's database)
    config/nn/effects.json     innate effects: attributes and passives, linked to units (the game's database)
    config/nn/game_rules.json  the game's battle, morale and fatigue rules (the game's database)
    config/nn/sim.json         measured or calibrated numbers the others lack

Only json; no torch here.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tools.nn.sim import abilities
from tools.nn.sim import effects as innate

ROOT = Path(__file__).resolve().parents[3]
# Projectile categories a shield blocks (small arms: tw-modding "Missiles and You", fandom "Shielded"; artillery,
# misc, grenades and spells go through; build/shields/spec.md S4).
SMALL_ARMS = ("arrow", "musket", "sling", "javelin", "axe")
CONFIG = ROOT / "config" / "nn"


def _value(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass
class Params:
    units: dict      # key -> passport
    rules: dict      # game_rules.json: battle, morale, fatigue
    sim: dict        # sim.json
    abilities: dict = None   # abilities.json: ability key -> passport
    effects: dict = None     # effects.json: {"order", "effects", "units"}

    # --- the game's rules (config/nn/game_rules.json) ---
    @property
    def battle(self):
        return self.rules["battle"]

    @property
    def morale(self):
        return self.rules["morale"]

    @property
    def fatigue(self):
        return self.rules["fatigue"]

    # --- calibration (config/nn/sim.json) ---
    @property
    def dt(self):
        return float(_value(self.sim["step_s"]))

    @property
    def limit_s(self):
        return float(_value(self.sim["battle_limit_s"]))

    @property
    def map_half(self):
        return float(_value(self.sim["map_half_m"]))

    def cal(self, section, name):
        return self.sim[section][name]

    def with_cal(self, section, **values):
        """A copy with some calibrated numbers replaced (for fitting)."""
        sim = json.loads(json.dumps(self.sim))
        sim[section].update(values)
        return Params(self.units, self.rules, sim, self.abilities, self.effects)

    # --- per unit ---
    def spacing_of(self, key):
        """(h, v), m: a man's place across the front and between ranks in the unit's close formation (the
        database's unit_spacings of its template, config/nn/units.json spacing); sim.json formation.spacing_m for a
        unit without one."""
        sp = (self.units.get(key) or {}).get("spacing") or {}
        d = float(self.sim["formation"]["spacing_m"])
        return float(sp.get("h") or d), float(sp.get("v") or d)

    def static(self, key, faction=None):
        """The STATIC fields (tools/nn/sim/state.py) of a unit key, without its place."""
        u = self.units[key]
        m = u["melee"]
        missile = u.get("missile") or {}
        cat = missile.get("category")
        ms = self.sim["missile"]
        # measured reload of the category, for the passport reload it was measured on (reload_ref_s); another
        # unit of the category scales with its passport (Night Runners' sling 8 s against the slaves' 9 s)
        # a projectile measured on its own (missile.reload_projectile_s) takes its measured cycle
        reload_s = 0
        own = (ms.get("reload_projectile_s") or {}).get(missile.get("projectile")) if missile else None
        if own is not None:
            reload_s = float(own)
        elif missile:
            ref = (ms.get("reload_ref_s") or {}).get(cat)
            reload_s = (ms["reload_s"][cat] * (missile.get("reload_s", ref) / ref if ref else 1.0)
                        if cat in ms["reload_s"] else missile.get("reload_s", 0) * ms["reload_scale_other"])
        resist = u.get("damage_resist") or {}
        h, v = self.spacing_of(key)
        men = u["men"]
        return {
            "men0": men, "hp_man": u["hp_per_man"], "hp0": u["hp_total"], "mass": u["mass"],
            "radius": u["radius_m"], "sp_h": h, "sp_v": v,
            # the default order: men / rank_depth files (the database's ranks), h apart
            "width": (men / max(1, u.get("rank_depth") or 1)) * h if men > 1 else 2 * u["radius_m"],
            "walk": u["speed"]["walk"], "run": u["speed"]["run"], "charge_speed": u["speed"]["charge"],
            "charge_dist": u["speed"].get("charge_distance", 0.0),
            "charge_pose": u["speed"].get("charge_pose_distance", 0.0),
            "accel": u["speed"]["acceleration"], "decel": u["speed"]["deceleration"],
            "turn": u["speed"].get("turn_deg_s", 120.0),
            "attack": m["attack"], "defence": m["defence"], "charge_bonus": m["charge_bonus"],
            "damage": m["damage"], "ap_damage": m["ap_damage"], "bonus_v_large": m["bonus_v_large"],
            "bonus_v_inf": m["bonus_v_infantry"], "interval": m["attack_interval_s"],
            "splash": max(1, m.get("splash_max_attacks") or 1),
            "armour": u["armour"], "shield": (u.get("shield") or {}).get("missile_block_chance", 0) / 100,
            "leadership": u["leadership"], "resist_missile": resist.get("missile", 0) / 100,
            # damage_mod_all (a ward save) adds to physical: every damage here is physical (melee, and missiles
            # through missile.physical_resist), so it counts once (0 for every unit today)
            "resist_physical": (resist.get("physical", 0) + resist.get("all", 0)) / 100,
            "large": u.get("size", "small") != "small",
            # rule flags (expendable, encourages, reflect, unbreakable, fire_move, fatigue_immune), the
            # effects bitmask and the timed effects: from the unit's innate effects (config/nn/effects.json)
            **innate.static(self, key),
            "direct": bool(missile.get("direct")) if missile else False,
            "ammo0": men * missile.get("ammo", 0) if missile else 0,
            "range": missile.get("range_m", 0) if missile else 0,
            "reload": reload_s,
            "m_damage": missile.get("damage", 0) if missile else 0,
            "m_ap": missile.get("ap_damage", 0) if missile else 0,
            "hit_rate": ms["hit_rate"].get(cat, ms["hit_rate_other"]) if missile else 0,
            # the spread (missile.py hit_chance): the database's projectile and the unit's accuracy
            "arc": missile.get("fire_arc_deg", 180.0) if missile else 0,
            "acc": (missile.get("accuracy", 0) + missile.get("marksmanship", 0)) if missile else 0,
            "cal_area": missile.get("calibration_area_m", 0.0) if missile else 0,
            "cal_dist": missile.get("calibration_distance_m", 0.0) if missile else 0,
            "muzzle_v": missile.get("muzzle_velocity", 0.0) if missile else 0,
            "height": u.get("height_m", 1.8),
            "small_arms": cat in SMALL_ARMS if missile else False,
            "aim_s": ms["aim_s"].get(cat, ms["aim_s_other"]) if missile else 0,
            "friendly_fire": ms["friendly_fire"].get(cat, ms["friendly_fire_other"]) if missile else 0,
            "morale_bonus": self.sim["morale"]["faction_bonus"].get(faction, 0),
            "rout_death_s": (self.sim["morale"]["lord_fall"].get("rout_death_s") or {}).get(faction, 0),
            "cost": u.get("multiplayer_cost", 0),
            # the army destruction's strength (morale.army_collapse): the database's combat potential
            "cp_fixed": (u.get("combat_potential") or {}).get("melee", u.get("multiplayer_cost", 0)) + sum(
                sum((self.abilities.get(k) or {}).get("cp") or (0, 0)) for k in u.get("abilities") or ()),
            "cp_missile": (u.get("combat_potential") or {}).get("missile", 0),
            "ai": False,
            **{f"ab{k}": i for k, i in enumerate(abilities.slots_of(self, key))},
        }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load(units=None, rules=None, sim=None, abilities=None, effects=None):
    """Params from the five files (defaults: config/nn/)."""
    return Params(units=_read(units or CONFIG / "units.json")["units"],
                  rules=_read(rules or CONFIG / "game_rules.json"),
                  sim=_read(sim or CONFIG / "sim.json"),
                  abilities=_read(abilities or CONFIG / "abilities.json")["abilities"],
                  effects=_read(effects or CONFIG / "effects.json"))
