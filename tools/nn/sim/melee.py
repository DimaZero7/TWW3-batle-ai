"""Melee: who strikes whom and how much health it takes (docs/en/training/simulator.md,
docs/en/game/mechanics/melee.md).

Per pair of units in contact (i strikes j), per second:

    men striking F  = fighting_files x (length of the sides in contact / the striker's spacing along its side:
                      h across its front or back, v along a flank; its formation template, config/nn/units.json
                      spacing), at most men (through the target's flank: the striker's own side, melee.flank_face
                      "striker"); against a single man (a lord) at most lord_max_attackers in all, however many
                      units surround him (the lord swarm probe, docs/en/game/units/lord-swarm.md); a single man
                      strikes once; a unit in contact with several enemies shares out no more than its own front
                      holds; a unit attacking another enemy strikes one it only touches at contact.unit_incidental
                      (a lord: lord_incidental) of its rate; a unit without a missile weapon under HOLD (no attack
                      order) strikes at contact.hold_rate of it
    hit chance p    = 35 + attack + charge bonus x charge left + bonus v target - defence x direction, within
                      8-90 % (the database rule, CA Feature Focus #2: weight 1, melee_hit_chance_*)
    hits a second   = p / (p x interval + miss_s) a fighting man: the interval runs after a hit, a miss costs
                      melee.miss_s (build/hitchance: 121 battles of formations); a lone man striking a lone man
                      (lord against lord): p / interval (the lone lord duels: a blow every 4 s, 41 % hit)
    per hit         = the armour-piercing part whole + the base part less the armour roll (U(0.5, 1) x armour
                      %, at most 100 %), then resistance (at most 90 %); damage beyond the struck man's health is
                      lost (overkill, smoothed: per_hit); bonus v infantry / large and the charge bonus split by
                      the weapon's armour-piercing share (CA)
    struck          = men a blow strikes: 1; a lord's splash blow on a formation contact.lord_splash_struck
                      (measured 2.07), each taking 1 / splash_max_attacks of the blow (CA 5.1.0: divided)
    HP/s            = F x struck x hits a second x per hit

Direction: defence x0.6 from the flank, x0.3 from the rear or against a routing unit (database, weight 1;
a lone man: contact.lord_direction of it - the lord swarm probe). A unit brings to each side of its formation no
more men than that side holds. A lone man fought by the enemy lord takes lord_rival_others of the infantry's rate.
Charge (battle.py): an attack order, a run-up and the charge's own clock, linear to 0 over charge_decay_duration
(13 s); nothing beyond the bonus (no impact, no ramp: CA, build/charge/spec.md). A braced unit with
charge_reflection strikes a charger in front of it (within bracing_attack_angle) whose charge is still at least
charge_reflect_min_charge_factor_threshold at charge_reflect_damage_multiplier of its damage (database).
Pursuit: a routing target is struck from the rear with all the men in contact at contact.pursuit_rate of the
rule and no charge (measured).
"""
import torch

from tools.nn.sim import geometry
from tools.nn.sim import orders as O


def hit_chance(attack, defence, slope=1.0, base=35.0, lo=8.0, hi=90.0):
    """The hit chance 0-1 (slope: the database's melee_hit_chance_normalisation_coefficient, 1)."""
    return torch.clamp(base + slope * (attack - defence), lo, hi) / 100


def armour_cut(armour):
    """Mean share of the base damage the armour roll stops: U(0.5, 1) x armour %, each roll at most 100 %
    (CA Feature Focus #2: Damage; the database's armour_roll_lower_cap 0.5): 0.75 a up to a = 1 (armour 100),
    2 - 1/a - a/4 up to 2, then 1."""
    a = (torch.as_tensor(armour) / 100).clamp(min=0)
    mid = 2 - 1 / a.clamp(min=1e-6) - a / 4
    return torch.where(a <= 1, 0.75 * a, torch.where(a < 2, mid, torch.ones_like(a)))


def per_hit(base, ap, armour, hp_man, resist=0.0, single=None):
    """HP a hit takes from the struck unit (build/damage/spec.md D1-D3). The armour-piercing part whole, the base
    part less the armour roll (armour_cut), then resistance (the caller sums them, at most 90 %). Damage beyond the
    struck man's health is lost (overkill, CA): a unit of many men loses hp / E[hits to kill a man], smoothed - the
    first hit exactly (it kills when the roll leaves at least his health), the rest on average:
    E[N] = 1 + P(the first hit leaves him alive) x max(1, (hp - E[first hit, at most hp]) / mean + 1/2).
    single (bool, broadcastable): a lone man (a lord) - his health is one pool, he loses the mean."""
    keep = 1 - torch.as_tensor(resist)
    a = (torch.as_tensor(armour) / 100).clamp(min=0)
    lo = ((ap + base * (1 - a).clamp(min=0)) * keep).clamp(min=1)
    hi = ((ap + base * (1 - a / 2).clamp(min=0)) * keep).clamp(min=1)
    mu = ((ap + base * (1 - armour_cut(armour))) * keep).clamp(min=1)
    w = hi - lo
    p1 = torch.where(w > 1e-6, ((hp_man - lo) / w.clamp(min=1e-6)).clamp(0, 1), (lo < hp_man).to(lo.dtype))
    tail = ((hp_man - (lo + torch.minimum(hi, hp_man)) / 2) / mu + 0.5).clamp(min=1)
    eff = hp_man / (1 + p1 * tail)
    return eff if single is None else torch.where(single, mu, eff)


def kill_share(hp_man, hit, exponent):
    """Men lost per man's worth of HP lost: 1 when a hit kills, less when blows only wound."""
    n = (hp_man / hit.clamp(min=1e-6)).clamp(min=1)
    return n.pow(-exponent)


def spacing_of(u, spacing):
    """(h, v) [B, N]: a man's place across the front and between ranks, m (the unit's formation template,
    config/nn/units.json spacing; `spacing` where a unit has none)."""
    if "sp_h" not in u:
        return torch.full_like(u["men"], float(spacing)), torch.full_like(u["men"], float(spacing))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    return h, v


def strikes(u, pw, contact, params, charge_now):
    """HP per second each unit i takes from each j: rate [B, N, N] (i strikes j), the per-hit damage [B, N, N],
    the direction sector of i seen from j and the men striking F [B, N, N]. charge_now [B, N]: the striker's
    charge left (0-1, battle.py: its own clock from the charge's first blow)."""
    B = params.battle
    cal = params.sim["melee"]
    cc = params.sim["contact"]
    h_u, v_u = spacing_of(u, params.sim["formation"]["spacing_m"])
    men_i = u["men"][:, :, None]
    single_i = u["men0"][:, :, None] <= 1
    single_j = u["men0"][:, None, :] <= 1
    # Which side of i faces j: its front or back (men h apart) or a flank (v apart).
    front_i, depth_i = pw["front"][:, :, None], pw["depth"][:, :, None]
    s, c = torch.sin(pw["rel_i"]), torch.cos(pw["rel_i"])
    through_front = s.abs() * depth_i <= c.abs() * front_i
    along = torch.where(through_front, h_u[:, :, None], v_u[:, :, None])
    length = torch.minimum(pw["face_i"], pw["face_j"])
    if cal.get("flank_face", "min") == "striker":
        # Through the target's flank (its side towards i is its depth) the striker brings men by
        # its own face, not by the target's short flank (measured: a lone flank attacker takes
        # 1.53x what a frontal one does; melee.flank_face, config/nn/sim.json).
        s_j, c_j = torch.sin(pw["rel_j"]).abs(), torch.cos(pw["rel_j"]).abs()
        flank_j = s_j * pw["depth"][:, None, :] > c_j * pw["front"][:, None, :]
        length = torch.where(flank_j, pw["face_i"], length)
    F = cal["fighting_files"] * length / along
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
    side_of = torch.where(through_front, torch.where(c >= 0, 0, 3), torch.where(s >= 0, 1, 2))
    for g in range(4):
        in_g = side_of == g
        hold = cal["fighting_files"] * (front_i / h_u[:, :, None] if g in (0, 3) else depth_i / v_u[:, :, None])
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
    ch = charge_now[:, :, None].expand_as(F)
    # Pursuit (contact.pursuit_rate; absent: off): a routing enemy is struck with every man the contact holds at
    # pursuit_rate of the rule, without the striker's charge (measured: a router loses 0.64 of what a standing target
    # loses to one pursuer, and the first 5 s of a pursuit hit no harder than later; config/nn/sim.json contact.pursuit_why).
    pursuit = cc.get("pursuit_rate")
    if pursuit is not None:
        ch = torch.where(routing_j, torch.zeros_like(ch), ch)
    large_j = u["large"][:, None, :]
    bonus = torch.where(large_j, u["bonus_v_large"][:, :, None], u["bonus_v_inf"][:, :, None])
    extra = u["charge_bonus"][:, :, None] * ch + bonus
    attack = u["attack"][:, :, None] + extra
    # Defence from the flank / rear: x0.6 / x0.3 (the database, weight 1); against a lone man (a lord)
    # contact.lord_direction of the loss (measured in the lord swarm probe: infantry on his back or flank hurts him
    # hardly more than on his front).
    w = torch.where(single_j, torch.full_like(coef, float(cc.get("lord_direction", 1.0))), torch.ones_like(coef))
    defence = u["defence"][:, None, :] * (1 - w * (1 - coef))
    p = hit_chance(attack, defence, B["melee_hit_chance_normalisation_coefficient"], B["melee_hit_chance_base"],
                   B["melee_hit_chance_min"], B["melee_hit_chance_max"])
    dmg, ap = u["damage"][:, :, None], u["ap_damage"][:, :, None]
    share = dmg / (dmg + ap).clamp(min=1e-6)
    # Charge reflection (database): a braced unit with charge_reflection (standing: slower than melee.braced_speed)
    # strikes a charger within bracing_attack_angle of its front, while the charger's charge is at least
    # charge_reflect_min_charge_factor_threshold, at charge_reflect_damage_multiplier of its damage (CA: spearmen
    # do double damage within 3.9 s of the charge); it gets no charge of its own.
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    braced = (u["reflect"] & (speed < cal["braced_speed"]))[:, :, None]
    facing = pw["rel_i"].abs() <= B["bracing_attack_angle"] * geometry.DEG
    charging_j = (charge_now >= B["charge_reflect_min_charge_factor_threshold"])[:, None, :]
    reflect = torch.where(braced & facing & charging_j & ~routing_j,
                          torch.full_like(F, float(B["charge_reflect_damage_multiplier"])), torch.ones_like(F))
    # A splash blow (a lord's) on a formation strikes contact.lord_splash_struck men on average (measured; at most
    # splash_max_attacks and the men left), each taking 1 / splash_max_attacks of it (CA 5.1.0: divided among the
    # targets; measured: single blows of 47-57 HP on stormvermin); a lone man takes the whole blow.
    n_max = u["splash"][:, :, None].expand_as(F)
    struck = torch.minimum(n_max, torch.full_like(n_max, float(cc.get("lord_splash_struck", 4.0))))
    struck = torch.minimum(struck, u["men"][:, None, :].clamp(min=1))
    struck = torch.where(single_j, torch.ones_like(struck), struck)
    div = n_max.clamp(min=1) if cal.get("splash_divides") else torch.ones_like(n_max)
    div = torch.where(single_j, torch.ones_like(div), div)
    hit = per_hit((dmg + extra * share) * reflect / div, (ap + extra * (1 - share)) * reflect / div,
                  u["armour"][:, None, :], u["hp_man"][:, None, :], u["resist_physical"][:, None, :],
                  single=single_j)
    # Blows a second a fighting man: the interval runs after a hit, a miss costs melee.miss_s (build/hitchance);
    # a lone man striking a lone man (lord against lord) swings every interval (the lone lord duels).
    interval = u["interval"][:, :, None].clamp(min=1e-6)
    per_s = torch.where(single_i & single_j, p / interval, p / (p * interval + float(cal["miss_s"])))
    rate = F * struck * per_s * hit
    # Men gather round a lone man (a lord) over contact.lord_gather_s from the striker's contact (0: at once).
    gather = float(cc.get("lord_gather_s", 0.0))
    if gather > 0:
        grow = (u["contact_s"] / gather).clamp(0, 1)[:, :, None]
        rate = torch.where(single_j & ~single_i, rate * grow, rate)
    if pursuit is not None:
        rate = torch.where(routing_j, rate * float(pursuit), rate)
    # A lone man (a lord) fought by infantry takes the sum of what the men around him strike (no
    # more than lord_max_attackers of them, however many units: the lord swarm probe). Fought by
    # the enemy lord too, the infantry's share counts only lord_rival_others (the probe's lord
    # against lord and units; the whole battles: 12.2 HP/s from the enemy lord, 13.5 with a unit).
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
