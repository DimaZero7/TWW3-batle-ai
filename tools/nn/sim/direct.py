"""Direct damage of an innate effect on its owner (docs/ru/training/simulator.md "Гноевики"; the game's program:
build/diag_roster/wh3_dlc29_skv_inf_pusbags/mechanic.md B1-B6; config/nn/sim.json direct). The class of the database's
phase damage (special_ability_phases damage_amount, max_damaged_entities, hp_change_frequency: the ability passport's
`direct`, config/nn/abilities.json; the catalogue's, config/nn/effects.json): a unit's effect with one (STATIC dd_fx:
the Pusbags' Unshakable Pathogen) strikes the unit itself while the effect is on (its conditions: fx_on of this step,
tools/nn/sim/effects.py - the Pathogen is off out of melee, its database flag out_of_melee, by the unit's melee flag):

  tick     at the phase's start and every every_s (the effects table's dd_s, 1 s) while it stays on: the clock dd_t
           (INTERNAL) is 0 when the effect is off, so a unit coming into melee again takes a tick at once (B1, B6)
  men      min(entities, its living men) different random men of the whole unit (dd_n: 3; -1 all), not the ones next to
           the enemy (B2)
  HP       each max(1, trunc(trunc(max(1, amount x scale)) x U[min_unary, 1])) - the battle's unit size's scale (ultra
           1.0) of _kv_unit_ability_scaling_rules: 2 / 3 / 4 HP, the mean dd_hp 3.2 (9.6 HP a second); magical (the
           program's damage flags 6: the Warp Grinder's reading of the same code, build/diag_roster/
           wh2_dlc14_skv_inf_warp_grinder_0/mechanic.md W4): no armour, no shield, no physical resistance, the spell
           resistance and the ward cut it - round(HP x (100 - resistance) / 100), at least 0 (resist_magic: the
           Pusbags have none); no contact effect (B3); it may kill (B4) - the kills Poisson on the living men
           (melee.shot_kills; the random men picked again every tick, the same may be struck again: the hits spread
           evenly), and a man it kills fires his vortex too (vortex.py)
Its HP counts in the unit's casualties (morale's casualty windows: a loss of its health like any other, B5), not in the
fight's balance, not as an enemy's kill. A battle without such a unit is unchanged bit for bit. Needs torch."""
import torch

from tools.nn.sim import effects as innate
from tools.nn.sim import melee


def on(params):
    """sim.json direct.on (the rule's switch)."""
    return bool((params.sim.get("direct") or {}).get("on"))


def active(u, params):
    """[B, N] bool: the unit's direct-damage effect is on this step (fx_on, set by effects.apply) and it is alive."""
    has = u["dd_fx"] >= 0
    bit = ((u["fx_on"] >> u["dd_fx"].clamp(min=0)) & 1) > 0
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    return has & bit & alive


def top_base(params):
    """The largest hit base (trunc(max(1, amount x scale))) of the catalogue's direct damage: the loop's bound of
    hit_mean, from the json (no tensor read)."""
    import math
    vs = [math.floor(max(1.0, float(d["amount"]) * float(d.get("scale", 1.0))))
          for e in (params.effects.get("effects") or {}).values() for d in [(e or {}).get("direct") or {}] if d]
    return int(max(vs + [1]))


def hit_mean(base, low, keep, top):
    """[B, N] the mean HP of a man's hit (module doc: HP): max(1, trunc(base x U[low, 1])) = k with its share, then
    round(k x keep) - keep 1 - resistance."""
    lo, hi = base * low, base
    w = (hi - lo).clamp(min=1e-6)
    out = torch.zeros_like(base)
    for k in range(0, top + 1):
        share = ((torch.minimum(hi, torch.full_like(hi, k + 1.0)) - torch.maximum(lo, torch.full_like(lo, float(k))))
                 .clamp(min=0) / w)
        share = torch.where(hi - lo > 1e-6, share, (torch.floor(hi) == k).float())
        out = out + share * torch.floor(max(1, k) * keep + 0.5)
    return out


def strike(u, params, dt):
    """(hp [B, N], kills [B, N], hits [B, N]): the HP this step's tick of each unit's own direct damage takes from it,
    the men it kills and the men it strikes; advances the tick clock dd_t and the bookkeeping dd_sum, in place."""
    T = innate.table(params, u["men"].device)
    E = T.shape[0] - 1
    e = u["dd_fx"].clamp(min=0, max=E)
    n_cap, every = T[e, innate.COL["dd_n"]], T[e, innate.COL["dd_s"]]
    keep = 1 - u["resist_magic"].clamp(-1.0, 0.9)
    hp_hit = hit_mean(T[e, innate.COL["dd_base"]], T[e, innate.COL["dd_low"]], keep, top_base(params))
    go = active(u, params) & (every > 0)
    tick = go & (u["dd_t"] <= 1e-4)
    men = u["men"].clamp(min=0)
    n = torch.where(n_cap < 0, men, torch.minimum(n_cap, men))
    hits = torch.where(tick, n, torch.zeros_like(n))
    hp = (hits * hp_hit).clamp(max=u["hp_abs"].clamp(min=0))
    kills = melee.shot_kills(hits, men, u["hp_man"], u["hp_abs"], hp_hit.clamp(min=1e-6))
    kills = torch.where(hp_hit > 0, kills, torch.zeros_like(kills))
    kills = torch.where(tick & (u["men0"] > 1), kills, torch.zeros_like(kills))
    # the clock: + every_s at a tick, - dt each step on; 0 when off (the next melee strikes at once)
    left = torch.where(tick, u["dd_t"] + every, u["dd_t"]) - dt
    u["dd_t"] = torch.where(go, left.clamp(min=0), torch.zeros_like(u["dd_t"]))
    u["dd_sum"] = u["dd_sum"] + hp
    return hp, kills, hits
