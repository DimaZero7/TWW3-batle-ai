"""The simulator's numbers, from three files (docs/en/training/simulator.md):

    config/nn/units.json       unit passports (the game's database)
    config/nn/game_rules.json  the game's battle, morale and fatigue rules (the game's database)
    config/nn/sim.json         measured or calibrated numbers the other two lack

Only json; no torch here.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "nn"


def _value(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass
class Params:
    units: dict      # key -> passport
    rules: dict      # game_rules.json: battle, morale, fatigue
    sim: dict        # sim.json

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
        return Params(self.units, self.rules, sim)

    # --- per unit ---
    def static(self, key, faction=None):
        """The STATIC fields (tools/nn/sim/state.py) of a unit key, without its place."""
        u = self.units[key]
        m = u["melee"]
        missile = u.get("missile") or {}
        cat = missile.get("category")
        ms = self.sim["missile"]
        reload_s = ms["reload_s"].get(cat, missile.get("reload_s", 0) * ms["reload_scale_other"]) if missile else 0
        resist = u.get("damage_resist") or {}
        spacing = self.sim["formation"]["spacing_m"]
        men = u["men"]
        return {
            "men0": men, "hp_man": u["hp_per_man"], "hp0": u["hp_total"], "mass": u["mass"],
            "radius": u["radius_m"],
            "width": (men / max(1, u.get("rank_depth") or 1)) * spacing if men > 1 else 2 * u["radius_m"],
            "walk": u["speed"]["walk"], "run": u["speed"]["run"], "charge_speed": u["speed"]["charge"],
            "accel": u["speed"]["acceleration"], "decel": u["speed"]["deceleration"],
            "attack": m["attack"], "defence": m["defence"], "charge_bonus": m["charge_bonus"],
            "damage": m["damage"], "ap_damage": m["ap_damage"], "bonus_v_large": m["bonus_v_large"],
            "bonus_v_inf": m["bonus_v_infantry"], "interval": m["attack_interval_s"],
            "splash": max(1, m.get("splash_max_attacks") or 1),
            "armour": u["armour"], "shield": (u.get("shield") or {}).get("missile_block_chance", 0) / 100,
            "leadership": u["leadership"], "resist_missile": resist.get("missile", 0) / 100,
            "resist_physical": resist.get("physical", 0) / 100,
            "large": u.get("size", "small") != "small",
            "expendable": "expendable" in u.get("attributes", []),
            "encourages": "encourages" in u.get("attributes", []),
            "ammo0": men * missile.get("ammo", 0) if missile else 0,
            "range": missile.get("range_m", 0) if missile else 0,
            "reload": reload_s,
            "m_damage": missile.get("damage", 0) if missile else 0,
            "m_ap": missile.get("ap_damage", 0) if missile else 0,
            "hit_rate": ms["hit_rate"].get(cat, ms["hit_rate_other"]) if missile else 0,
            "aim_s": ms["aim_s"].get(cat, ms["aim_s_other"]) if missile else 0,
            "friendly_fire": ms["friendly_fire"].get(cat, ms["friendly_fire_other"]) if missile else 0,
            "morale_bonus": self.sim["morale"]["faction_bonus"].get(faction, 0),
            "cost": u.get("multiplayer_cost", 0),
        }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load(units=None, rules=None, sim=None):
    """Params from the three files (defaults: config/nn/)."""
    return Params(units=_read(units or CONFIG / "units.json")["units"],
                  rules=_read(rules or CONFIG / "game_rules.json"),
                  sim=_read(sim or CONFIG / "sim.json"))
