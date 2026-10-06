"""Melee: who strikes whom and how much health it takes (docs/en/training/simulator.md).

Per pair of units in contact (i strikes j), per second:

    men striking F  = fighting_files x (length of the sides in contact / spacing), at most men
                      (through the target's flank: the striker's own side, melee.flank_face "striker");
                      against a single man (a lord) at most lord_max_attackers in all, however many
                      units surround him (measured: the lord swarm probe, docs/en/game/units/lord-swarm.md);
                      a single man strikes once (his blow hits up to `splash` men); a unit in
                      contact with several enemies shares out no more than its own front holds;
                      a unit attacking another enemy strikes one it only touches at
                      contact.unit_incidental (a lord: lord_incidental) of its rate; a unit without
                      a missile weapon under HOLD (no attack order) strikes at contact.hold_rate of it
    hit chance p    = 35 + hit_slope x (attack + charge - defence x direction) within 8-90 %
                      (the database rule is hit_slope 1; calibrated, config/nn/sim.json); a lone
                      man striking a lone man (lord against lord): contact.lord_hit_slope (measured 1)
    per hit         = ap + base x (1 - 0.75 armour / 100)   (armour stops 50-100 %: mean 75 %),
                      not more than a man's health; a splash blow's damage is divided among its targets
    HP/s            = F x splash x p x per hit / interval x (1 + impact x charge)

Direction: defence x0.6 from the flank, x0.3 from the rear or against a routing unit (database);
the defence lost counts at flank_slope / rear_slope of the rule (measured in whole battles; a lone
man: lord_direction of it). A unit brings to each side of its formation no more men than that side
holds. A lone man fought by the enemy lord takes lord_rival_others of the infantry's rate.
Charge: a unit that meets the enemy running gets its charge bonus to attack and damage and hits
harder (x (1 + impact)), fading over charge_decay_duration (13 s); a unit that did not charge
brings its men to bear over ramp_s (calibrated on the first 15 s of the pairs); a braced unit
with charge_reflection meets a frontal charge as a charge (battle.py).
Pursuit: a routing target is struck from the rear with all the men in contact at contact.pursuit_rate
of the rule and no charge (measured).
"""
import torch

from tools.nn.sim import geometry
from tools.nn.sim import orders as O


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
    if cal.get("flank_face", "min") == "striker":
        # Through the target's flank (its side towards i is its depth) the striker brings men by
        # its own face, not by the target's short flank (measured: a lone flank attacker takes
        # 1.53x what a frontal one does; melee.flank_face, config/nn/sim.json).
        s_j, c_j = torch.sin(pw["rel_j"]).abs(), torch.cos(pw["rel_j"]).abs()
        flank_j = s_j * pw["depth"][:, None, :] > c_j * pw["front"][:, None, :]
        length = torch.where(flank_j, pw["face_i"], length)
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
    # At most lord_max_attackers around a single man, however many units they belong to (the
    # lord swarm probe); an enemy lord among them keeps his place (one man) and the infantry
    # share the rest.
    lone = (single_i & (F > 0)).float().sum(dim=1, keepdim=True)
    around = torch.where(single_i, torch.zeros_like(F), F).sum(dim=1, keepdim=True)
    cap = (float(cc["lord_max_attackers"]) - lone).clamp(min=0)
    F = torch.where(single_j & ~single_i & (around > cap), F * cap / around.clamp(min=1e-6), F)

    sector = geometry.sector(pw["rel_j"], cc["front_deg"], cc["rear_deg"])
    routing_j = u["r"][:, None, :]
    sector = torch.where(routing_j, torch.full_like(sector, 2), sector)
    coef = torch.where(sector == 0, 1.0, torch.where(sector == 1, B["melee_defence_direction_penalty_coefficient_flank"],
                                                     B["melee_defence_direction_penalty_coefficient_rear"]))
    ch = charge_now[:, :, None]
    # Pursuit (contact.pursuit_rate; absent: off): a routing enemy is struck with every man the contact holds at
    # pursuit_rate of the rule, without the striker's charge (measured: a router loses 0.64 of what a standing target
    # loses to one pursuer, and the first 5 s of a pursuit hit no harder than later; config/nn/sim.json contact.pursuit_why).
    # Without it a unit touching only routers struck nothing (its contact clock and so its ramp stay 0).
    pursuit = cc.get("pursuit_rate")
    if pursuit is not None:
        ch = torch.where(routing_j, torch.zeros_like(pw["gap"]), ch)
    large_j = u["large"][:, None, :]
    bonus = torch.where(large_j, u["bonus_v_large"][:, :, None], u["bonus_v_inf"][:, :, None])
    attack = u["attack"][:, :, None] + u["charge_bonus"][:, :, None] * ch + bonus
    defence = u["defence"][:, None, :]
    # Defence lost to a flank / rear attack counts at flank_slope / rear_slope (1 = the database rule).
    # A lone man (a lord) at contact.lord_direction of the rule (measured in the lord swarm probe:
    # infantry on his back or flank hurts him hardly more than on his front).
    slope = torch.where(sector == 2, cal.get("rear_slope", cal["flank_slope"]), cal["flank_slope"])
    slope = torch.where(single_j, torch.full_like(slope, float(cc.get("lord_direction", 1.0))), slope)
    # The hit chance's weight of attack - defence: hit_slope (calibrated, flat) for formations; a lone man
    # striking a lone man (lord against lord) at contact.lord_hit_slope (1: the database rule; measured in the
    # lone lord duels: General v General 0.60-0.71 %HP/s, Warlord v Warlord 0.23-0.34, config/nn/sim.json).
    k_hit = torch.full_like(slope, float(cal["hit_slope"]))
    k_hit = torch.where(single_i & single_j, torch.full_like(k_hit, float(cc.get("lord_hit_slope", cal["hit_slope"]))),
                        k_hit)
    exposed = defence * (1 - coef) * (slope / k_hit.clamp(min=1e-6) - 1)
    p = hit_chance(attack + exposed, defence * coef, k_hit, B["melee_hit_chance_base"],
                   B["melee_hit_chance_min"], B["melee_hit_chance_max"])
    dmg, ap = u["damage"][:, :, None], u["ap_damage"][:, :, None]
    share = dmg / (dmg + ap).clamp(min=1e-6)
    extra = u["charge_bonus"][:, :, None] * ch + bonus
    splash = torch.minimum(u["splash"][:, :, None], u["men"][:, None, :].clamp(min=1))
    # A splash blow's damage is divided among the men it strikes (melee.splash_divides; CA forum, the
    # community): each takes 1/targets of it, then armour and the man's health cap it.
    div = splash.clamp(min=1) if cal.get("splash_divides") else torch.ones_like(splash)
    hit = per_hit((dmg + extra * share) / div, (ap + extra * (1 - share)) / div, u["armour"][:, None, :],
                  u["hp_man"][:, None, :], u["resist_physical"][:, None, :])
    ramp = torch.where(u["charge"] > 0, torch.ones_like(contact_s), (contact_s / cal["ramp_s"]).clamp(0, 1))[:, :, None]
    if pursuit is not None:
        ramp = torch.where(routing_j, torch.full_like(F, float(pursuit)), ramp.expand_as(F))
    # The charge's impact brings more men to bear: not for a single man (his charge is his bonus),
    # nor against one (no more than lord_max_attackers reach him anyway).
    impact = torch.where(single_i | single_j, torch.zeros_like(ch), cal["impact"] * ch)
    rate = F * splash * p * hit / u["interval"][:, :, None].clamp(min=1e-6) * (1 + impact) * ramp
    # A lone man (a lord) fought by infantry takes the sum of what the men around him strike (no
    # more than lord_max_attackers of them, however many units: the lord swarm probe). Fought by
    # the enemy lord too, the infantry's share counts only lord_rival_others (the probe's lord
    # against lord and units; the whole battles: 12.2 HP/s from the enemy lord, 13.5 with a unit).
    # A lone man striking a lone man (lord against lord) hits at lord_v_lord of the rule (measured
    # in the whole and network battles: a lord fought by the enemy lord alone).
    rate = torch.where(single_i & single_j, rate * float(cc.get("lord_v_lord", 1.0)), rate)
    rival = (single_i & (rate > 0)).any(dim=1, keepdim=True)
    k = float(cc.get("lord_rival_others", 1.0))
    rate = torch.where(single_j & rival & ~single_i, rate * k, rate)
    # A unit told to attack another enemy fights that one: a lone man it only touches takes
    # lord_incidental of its rate (whole battles: a lord in contact with one enemy unit loses
    # 1.7 HP/s when the unit's current target is another, 4.1 when it is him).
    slot = torch.arange(u["men"].shape[1], device=rate.device)[None, None, :]
    order_t = u["order_target"][:, :, None]
    busy = (u["order_kind"][:, :, None] == O.ATTACK) & (order_t >= 0) & (order_t != slot)
    rate = torch.where(single_j & ~single_i & busy, rate * float(cc.get("lord_incidental", 1.0)), rate)
    # The same for an enemy unit it only touches: unit_incidental of its rate (gate battles, 02.10.2026:
    # a unit fought by one enemy unit took 19.6 HP/s in the game, 33.9 in the open-loop replay, when
    # other enemy units stood within 35 m of it; 16.5 against 20.7 when none did).
    rate = torch.where(~single_j & ~single_i & busy, rate * float(cc.get("unit_incidental", 1.0)), rate)
    # A unit without a missile weapon standing under HOLD in melee (no attack order) fights only with the men
    # that happen to be in contact: hold_rate of its rate on every enemy it touches (the network's units in the
    # game: matched on own and enemy unit keys 0.80 of the kills of the same unit attacking; calibrated to the
    # kills of held units in the gate battles, where they touch 2.3 enemy units; missile units 0.99-1.01;
    # config/nn/sim.json contact.why).
    held = (u["order_kind"] == O.HOLD) & (u["range"] <= 0)
    rate = torch.where(held[:, :, None], rate * float(cc.get("hold_rate", 1.0)), rate)
    return rate, hit, sector, F
