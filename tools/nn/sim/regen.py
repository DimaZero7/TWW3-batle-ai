"""Regeneration and the fire that cuts it (docs/ru/training/simulator.md "Монстры скавенов"; the game's program:
build/diag_skv_mon/mechanic.md M11-M14; config/nn/sim.json regen). Only a unit with a regeneration (STATIC regen > 0: the
Brood Horror) heals; a battle without one is unchanged bit for bit.

  heal       a living unit (routing too: the game's tick has no condition, in melee and on the run alike) heals every
             second round(heal x P x its starting health) (the database's heal_amount 0.001 every hp_change_frequency 1 s:
             the Brood Horror's 6009 HP 6 HP/s, 360 a minute), P = 1 - flaming_damage_taken_hp_regen_multiplier 0.5 while
             its fire clock runs (3 HP/s); no more than its men's health (a dead man stays dead) and no more than its pool:
             heal_cap (regen.cap 0.75, the database's healing_percentage_cap: the community's '75 % of the maximum') x its
             starting health in the whole battle - the program's pool never refills (the Horror: 4506 HP, 12.5 min of it)
  the fire   a flaming blow's damage on it (STATIC flaming: the Knights of the Blazing Sun's lances) starts its fire clock at
             regen.burn_s 10 s (the program's 100 ticks; where the clock runs down was not found - read as 10 s); the
             blow itself is x1.2 by Regeneration's weakness to fire -20 % (effects.py resist_flame, melee.resist_of)
Needs torch."""
import torch


def on(params):
    return bool((params.sim.get("regen") or {}).get("on"))


def burn(u, params, dt, hp_melee, enemy):
    """The fire clock (module doc), in place: down by dt, and back to regen.burn_s on a unit a flaming enemy's blows took
    HP from this step (hp_melee [B, i, j]: HP i's blows took from j, enemy [B, i, j])."""
    if "burn_s" not in u or "flaming" not in u:
        return
    hit = ((hp_melee * enemy) * u["flaming"][:, :, None].float()).sum(1) > 0
    left = (u["burn_s"] - dt).clamp(min=0)
    u["burn_s"] = torch.where(hit & (u["regen"] > 0), torch.full_like(left, float(params.sim["regen"]["burn_s"])), left)


def heal(u, params, dt):
    """HP each unit heals this step [B, N] (module doc); adds it to hp_abs and its pool's count `healed`, in place."""
    R = params.battle
    P = torch.where(u["burn_s"] > 0, 1.0 - float(R["flaming_damage_taken_hp_regen_multiplier"]), 1.0)
    tick = torch.floor(u["regen"] * P * u["hp0"] + 0.5)            # HP a second: the program's round (6009 x 0.001 -> 6)
    room = (u["men"] * u["hp_man"] - u["hp_abs"]).clamp(min=0)     # its living men's wounds
    pool = (u["heal_cap"] * u["hp0"] - u["healed"]).clamp(min=0)
    alive = (u["men"] > 0) & (u["side"] > 0) & ~u["gone"]
    got = torch.minimum(torch.minimum(tick * dt, room), pool)
    got = torch.where(alive & (u["regen"] > 0), got, torch.zeros_like(got))
    u["hp_abs"] = torch.where(got > 0, u["hp_abs"] + got, u["hp_abs"])
    u["healed"] = u["healed"] + got
    return got
