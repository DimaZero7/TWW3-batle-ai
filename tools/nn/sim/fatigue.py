"""Fatigue (docs/en/training/simulator.md, docs/en/game/units/pace.md).

Points per activity from the game's database (_kv_fatigue_tables): charging +34, melee +19,
shooting +18, running +4, walking -1, standing ready -7. The database gives no time; the
recordings give 5x a second (config/nn/sim.json fatigue.per_second). States by the database
thresholds: fresh 0, active 2800, winded 6600, tired 12600, very tired 18000, exhausted 27000
(max 30000). Tired states cost morale (morale.py) and scale speed, melee attack and defence,
armour, charge bonus, AP damage and reload by the database's unit_fatigue_effects_tables
(effects(); config/nn/sim.json fatigue.effects).
"""
import torch

LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
          "threshold_exhausted")


def step(u, activity, params, dt):
    """activity: dict of [B, N] bools charging, melee, shooting, running, walking. Changes u."""
    F = params.fatigue
    scale = params.sim["fatigue"]["per_second"] * dt
    rate = torch.full_like(u["fatigue"], F["ready"])
    rate = torch.where(activity["walking"], F["walking"], rate)
    rate = torch.where(activity["running"], F["running"], rate)
    rate = torch.where(activity["shooting"], F["shooting"], rate)
    rate = torch.where(activity["melee"], F["combat"], rate)
    rate = torch.where(activity["charging"], F["charging"], rate)
    u["fatigue"] = (u["fatigue"] + rate * scale).clamp(0, F["threshold_max"])
    th = torch.tensor([F[k] for k in LEVELS], dtype=u["fatigue"].dtype, device=u["fatigue"].device)
    u["fat"] = (torch.bucketize(u["fatigue"], th, right=True) - 1).clamp(min=0).float()


# The stats unit_fatigue_effects_tables scales, and the unit field each one scales.
EFFECT_FIELDS = {"scalar_speed": ("walk", "run", "charge_speed"), "stat_melee_attack": ("attack",),
                 "stat_melee_defence": ("defence",), "stat_armour": ("armour",),
                 "stat_charge_bonus": ("charge_bonus",), "stat_melee_damage_ap": ("ap_damage",)}


def effects(u, params):
    """Lay the fatigue state's stat multipliers on u for this step (the game's database,
    unit_fatigue_effects_tables, copied to config/nn/sim.json fatigue.effects; missing rows x1).
    stat_reloading scales the reload skill: the reload time is divided by it. Returns the fields
    it changed with their old values (restore with u.update)."""
    table = params.sim["fatigue"].get("effects")
    if not table:
        return {}
    old = {}
    level = u["fat"].long().clamp(0, len(LEVELS) - 1)
    for stat, fields in list(EFFECT_FIELDS.items()) + [("stat_reloading", ("reload",))]:
        mult = [float(table.get(lv, {}).get(stat, 1.0)) for lv in LEVELS]
        if all(m == 1.0 for m in mult):
            continue
        m = torch.tensor(mult, dtype=u["attack"].dtype, device=u["attack"].device)[level]
        for f in fields:
            old.setdefault(f, u[f])
            u[f] = u[f] / m if stat == "stat_reloading" else u[f] * m
    return old
