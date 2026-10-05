"""Fatigue (docs/en/training/simulator.md, docs/en/game/units/pace.md).

Points per activity from the game's database (_kv_fatigue_tables): charging +34, melee +19,
shooting +18, running +4, walking -1, standing ready -7, idle -18. The database gives no time;
isolated activity and recovery transitions give 10 ticks/s. The unaccepted calibration is
switched off: production retains 5 ticks/s (config/nn/sim.json fatigue). States by the database
thresholds: fresh 0, active 2800, winded 6600, tired 12600, very tired 18000, exhausted 27000
(max 30000). Tired states cost morale (morale.py) and scale speed, melee attack and defence,
armour, charge bonus, AP damage and reload by the database's unit_fatigue_effects_tables
(effects(); config/nn/sim.json fatigue.effects).
"""
import torch

LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
          "threshold_exhausted")


def step(u, activity, params, dt):
    """Activity bools [B, N], highest priority charging > melee > shooting > run > walk > idle.
    The calibration switch enables idle recovery, contact-weighted melee and the active mask.
    Missing idle means ready; missing contact_share means whole-unit melee.
    OFF retains the legacy clock and ready recovery, including inactive slots.
    """
    F = params.fatigue
    cal = params.sim["fatigue"]
    trial = cal.get("calibration", {})
    enabled = trial.get("on", False)
    scale = (trial["per_second"] if enabled else cal["per_second"]) * dt
    rate = torch.full_like(u["fatigue"], F["ready"])
    if enabled and "idle" in activity:
        rate = torch.where(activity["idle"], F["idle"], rate)
    rate = torch.where(activity["walking"], F["walking"], rate)
    rate = torch.where(activity["running"], F["running"], rate)
    rate = torch.where(activity["shooting"], F["shooting"], rate)
    combat = F["combat"]
    if enabled and trial.get("contact_weighted") and "contact_share" in activity:
        share = activity["contact_share"].clamp(0, 1)
        # Non-contact men use the trial's DB activity; the contact estimate is the melee model's
        # weapon-bearing population, before damage/ramp/hold rate multipliers.
        rest = F[trial.get("noncontact", "ready")]
        combat = share * F["combat"] + (1 - share) * rest
    rate = torch.where(activity["melee"], combat, rate)
    rate = torch.where(activity["charging"], F["charging"], rate)
    # Perfect Vigour (attribute fatigue_immune, an innate effect): never tires
    if "fatigue_immune" in u:
        rate = torch.where(u["fatigue_immune"], torch.clamp(rate, max=0.0), rate)
    if enabled and "active" in activity:
        rate = torch.where(activity["active"], rate, torch.zeros_like(rate))
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
