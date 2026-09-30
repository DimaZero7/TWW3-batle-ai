"""Fatigue (docs/en/training/simulator.md, docs/en/game/units/pace.md).

Points per activity from the game's database (_kv_fatigue_tables): charging +34, melee +19,
shooting +18, running +4, walking -1, standing ready -7. The database gives no time; the
recordings give 5x a second (config/nn/sim.json fatigue.per_second). States by the database
thresholds: fresh 0, active 2800, winded 6600, tired 12600, very tired 18000, exhausted 27000
(max 30000). Tired states cost morale (morale.py); their effect on attack, defence and speed is
not in the database and is not modelled.
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
