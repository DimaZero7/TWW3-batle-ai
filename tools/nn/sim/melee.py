"""Melee: who strikes whom and how much health it takes (docs/en/training/simulator.md).

Per pair of units in contact (i strikes j), per second:

    men striking F  = fighting_files x (length of the sides in contact / spacing), at most men;
                      against a single man (a lord) at most lord_max_attackers (measured ~8);
                      a single man strikes once (his blow hits up to `splash` men); a unit in
                      contact with several enemies shares out no more than its own front holds
    hit chance p    = 35 + hit_slope x (attack + charge - defence x direction) within 8-90 %
                      (the database rule is hit_slope 1; calibrated, config/nn/sim.json)
    per hit         = ap + base x (1 - 0.75 armour / 100)   (armour stops 50-100 %: mean 75 %),
                      not more than a man's health
    HP/s            = F x splash x p x per hit / interval x (1 + impact x charge)

Direction: defence x0.6 from the flank, x0.3 from the rear or against a routing unit (database);
the defence lost counts at flank_slope / rear_slope of the rule (measured in whole battles; a lone
man: the rule). A unit brings to each side of its formation no more men than that side holds.
Charge: a unit that meets the enemy running gets its charge bonus to attack and damage and hits
harder (x (1 + impact)), fading over charge_decay_duration (13 s); a unit that did not charge
brings its men to bear over ramp_s (calibrated on the first 15 s of the pairs); a braced unit
with charge_reflection meets a frontal charge as a charge (battle.py).
"""
import torch

from tools.nn.sim import geometry


def hit_chance(attack, defence, slope, base=35.0, lo=8.0, hi=90.0):
    """The hit chance 0-1."""
    return torch.clamp(base + slope * (attack - defence), lo, hi) / 100


def per_hit(base, ap, armour, hp_man, resist=0.0):
    """HP a hit takes: armour stops on average 75 % of its value in % of the base damage."""
    dmg = ap + base * torch.clamp(1 - 0.75 * armour / 100, min=0)
    return torch.minimum(dmg * (1 - resist), hp_man)


def kill_share(hp_man, hit, exponent):
    """Men lost per man's worth of HP lost: 1 when a hit kills, less when blows only wound."""
    n = (hp_man / hit.clamp(min=1e-6)).clamp(min=1)
    return n.pow(-exponent)


def strikes(u, pw, contact, params, charge_now, contact_s):
    """HP per second each unit i takes from each j: rate [B, N, N] (i strikes j), the per-hit
    damage [B, N, N], the direction sector of i seen from j and the men striking F [B, N, N]. charge_now [B, N]: the striker's
    charge left (0-1); contact_s [B, N]: seconds since its contact began (a unit that did not
    charge brings its men to bear over ramp_s)."""
    B = params.battle
    cal = params.sim["melee"]
    cc = params.sim["contact"]
    spacing = params.sim["formation"]["spacing_m"]
    men_i = u["men"][:, :, None]
    single_i = u["men0"][:, :, None] <= 1
    single_j = u["men0"][:, None, :] <= 1
    length = torch.minimum(pw["face_i"], pw["face_j"])
    F = cal["fighting_files"] * length / spacing
    F = torch.minimum(F, men_i)
    F = torch.where(single_j, torch.minimum(men_i, torch.full_like(F, float(cc["lord_max_attackers"]))), F)
    F = torch.where(single_i, torch.ones_like(F), F)
    # Fewer men left, fewer bring their weapons to bear (and fewer can be reached).
    share_i = (u["men"] / u["men0"].clamp(min=1)).clamp(0, 1)[:, :, None]
    share_j = (u["men"] / u["men0"].clamp(min=1)).clamp(0, 1)[:, None, :]
    F = F * torch.where(single_i, torch.ones_like(F), share_i.pow(cal.get("men_exp_striker", 0.0)))
    F = F * torch.where(single_j, torch.ones_like(F), share_j.pow(cal.get("men_exp_target", 0.0)))
    F = torch.where(contact, F, torch.zeros_like(F))
    # Across several contacts a unit brings to each side of its formation (front, left, right, back)
    # no more men than that side holds, and no more than it has in all; a single man is split
    # between his opponents. (Men on a flank turn to fight: measured, a unit already fighting hits a
    # newcomer on its flank as hard as a free unit does.)
    front_i, depth_i = pw["front"][:, :, None], pw["depth"][:, :, None]
    s, c = torch.sin(pw["rel_i"]), torch.cos(pw["rel_i"])
    through_front = s.abs() * depth_i <= c.abs() * front_i
    side_of = torch.where(through_front, torch.where(c >= 0, 0, 3), torch.where(s >= 0, 1, 2))
    for g in range(4):
        in_g = side_of == g
        hold = cal["fighting_files"] * (front_i if g in (0, 3) else depth_i) / spacing
        tot = torch.where(in_g, F, torch.zeros_like(F)).sum(dim=2, keepdim=True)
        F = torch.where(in_g & ~single_i & (tot > hold), F * hold / tot.clamp(min=1e-6), F)
    own = torch.where(single_i, torch.ones_like(men_i), men_i)
    total = F.sum(dim=2, keepdim=True)
    F = F * torch.where(total > own, own / total.clamp(min=1e-6), torch.ones_like(total))
    # At most lord_max_attackers around a single man, whoever they belong to.
    around = F.sum(dim=1, keepdim=True)
    cap = float(cc["lord_max_attackers"])
    F = torch.where(single_j & (around > cap), F * cap / around.clamp(min=1e-6), F)

    sector = geometry.sector(pw["rel_j"], cc["front_deg"], cc["rear_deg"])
    routing_j = u["r"][:, None, :]
    sector = torch.where(routing_j, torch.full_like(sector, 2), sector)
    coef = torch.where(sector == 0, 1.0, torch.where(sector == 1, B["melee_defence_direction_penalty_coefficient_flank"],
                                                     B["melee_defence_direction_penalty_coefficient_rear"]))
    ch = charge_now[:, :, None]
    large_j = u["large"][:, None, :]
    bonus = torch.where(large_j, u["bonus_v_large"][:, :, None], u["bonus_v_inf"][:, :, None])
    attack = u["attack"][:, :, None] + u["charge_bonus"][:, :, None] * ch + bonus
    defence = u["defence"][:, None, :]
    # Defence lost to a flank / rear attack counts at flank_slope / rear_slope (1 = the database rule).
    # A lord turns to fight whoever reaches him: the database rule (slope 1) for him.
    slope = torch.where(sector == 2, cal.get("rear_slope", cal["flank_slope"]), cal["flank_slope"])
    slope = torch.where(single_j, torch.ones_like(slope), slope)
    exposed = defence * (1 - coef) * (slope / max(cal["hit_slope"], 1e-6) - 1)
    p = hit_chance(attack + exposed, defence * coef, cal["hit_slope"], B["melee_hit_chance_base"],
                   B["melee_hit_chance_min"], B["melee_hit_chance_max"])
    dmg, ap = u["damage"][:, :, None], u["ap_damage"][:, :, None]
    share = dmg / (dmg + ap).clamp(min=1e-6)
    extra = u["charge_bonus"][:, :, None] * ch + bonus
    hit = per_hit(dmg + extra * share, ap + extra * (1 - share), u["armour"][:, None, :], u["hp_man"][:, None, :],
                  u["resist_physical"][:, None, :])
    splash = torch.minimum(u["splash"][:, :, None], u["men"][:, None, :].clamp(min=1))
    ramp = torch.where(u["charge"] > 0, torch.ones_like(contact_s), (contact_s / cal["ramp_s"]).clamp(0, 1))
    # The charge's impact brings more men to bear: not for a single man (his charge is his bonus),
    # nor against one (no more than lord_max_attackers reach him anyway).
    impact = torch.where(single_i | single_j, torch.zeros_like(ch), cal["impact"] * ch)
    rate = F * splash * p * hit / u["interval"][:, :, None].clamp(min=1e-6) * (1 + impact) * ramp[:, :, None]
    return rate, hit, sector, F
