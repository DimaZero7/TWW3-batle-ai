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
                      (a lord: lord_incidental) of its rate; a formation without a missile weapon under HOLD (no
                      attack order; a missile unit too: contact.hold_missile) or under a MOVE still in contact (contact.hold_move) strikes at contact.hold_rate
                      of it (measured: the melee probe's held units 0.49-0.52 of the rule, move lanes 0.70 of
                      attacking), except its blows reflecting a charge (contact.hold_reflect_full); a lone man (a lord)
                      under HOLD strikes in full (the damage plan)
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
    first strike    = F x struck x p x per hit, once (swing), not cut by hold_rate (contact.first_strike_full): the
                      interval runs only after a blow, so a unit coming into a fight (moving in, or its charge landing)
                      or a standing one reached by an enemy it faces strikes once at once with every man in contact
                      (battle.py decides when)

Direction: defence x0.6 from the flank, x0.3 from the rear or against a routing unit (database, weight 1;
a lone man: contact.lord_direction of it - the lord swarm probe). A unit brings to each side of its formation no
more men than that side holds. A lone man fought by the enemy lord takes lord_rival_others of the infantry's rate.
Charge (battle.py): an attack order, a run-up and the charge's own clock, linear to 0 over charge_decay_duration
(13 s); its first blows are the first strike above; nothing beyond the bonus (no impact, no ramp: CA,
build/charge/spec.md). A braced unit with
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


def shot_kills(hits, men, hp_man, hp_abs, per_hit):
    """Men killed by `hits` projectile hits on a unit [B, N] (kills.missile_uniform; build/open2: the thin probe).
    Every man has his own health and a projectile strikes one man; the hits spread evenly over the living men, so
    a man's hits are Poisson: with tau hits a man so far and K = HP a man / per_hit hits to kill him (E[N],
    smoothed as per_hit), the living share is Q(K, tau) (the regularised upper gamma: P(fewer than K hits)) and
    `hits` more on A living men kill A x (1 - Q(K, tau + hits / A) / Q(K, tau)). tau comes from the unit's own
    wounded: the living men's mean hits so far E[k | k < K] = tau Q(K - 1, tau) / Q(K, tau) = W / (A x per_hit),
    W = men x HP a man - the unit's HP (the wounds of any source). A hit that kills (K <= 1) is a man."""
    A = men.clamp(min=1e-6)
    K = (hp_man / per_hit.clamp(min=1e-6)).clamp(min=1.0)
    m_obs = ((men * hp_man - hp_abs).clamp(min=0) / (A * per_hit.clamp(min=1e-6)))
    Ks = K.clamp(min=1.02)
    m_obs = torch.minimum(m_obs, 0.95 * (Ks - 1))
    lo, hi = torch.zeros_like(K), 4 * Ks + 30
    lg = torch.lgamma(Ks)
    for _ in range(16):                     # E[k | k < K] grows with tau: bisection (to ~1e-3 of a hit)
        mid = (lo + hi) / 2
        q = torch.special.gammaincc(Ks, mid).clamp(min=1e-30)
        # Q(K - 1, tau) = Q(K, tau) - tau^(K-1) e^-tau / Gamma(K)
        pk = torch.exp((Ks - 1) * torch.log(mid.clamp(min=1e-30)) - mid - lg)
        m = mid * (1 - pk / q)
        up = m < m_obs
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    tau = (lo + hi) / 2
    s0 = torch.special.gammaincc(Ks, tau).clamp(min=1e-30)
    s1 = torch.special.gammaincc(Ks, tau + hits / A)
    k = A * (1 - s1 / s0).clamp(0, 1)
    return torch.where(K < 1.02, hits, k).clamp(min=0) * (men > 0)


def spacing_of(u, spacing):
    """(h, v) [B, N]: a man's place across the front and between ranks, m (the unit's formation template,
    config/nn/units.json spacing; `spacing` where a unit has none)."""
    if "sp_h" not in u:
        return torch.full_like(u["men"], float(spacing)), torch.full_like(u["men"], float(spacing))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    return h, v


def strikes(u, pw, contact, params, charge_now, first=False):
    """HP per second each unit i takes from each j: rate [B, N, N] (i strikes j), the per-hit damage [B, N, N],
    the direction sector of i seen from j and the men striking F [B, N, N]; with first, also swing [B, N, N]: the
    HP of one blow of every man in contact at once (the first strike, battle.py). charge_now [B, N]: the striker's
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
    if cal.get("target_face_cap"):
        # The target's side is only so long (melee.target_face_cap): the units coming at j through the same side of j
        # (front, left, right, back - seen from j) share its length - each brings fighting_files of the files it
        # lines up along it (F x its own spacing / fighting_files metres), together no more than the side is long (or
        # than the longest single attacker brings), a lone target keeps lord_max_attackers.
        # Measured (build/routgap/overlap_melee.py, all fair recordings): an enemy formation struck by two of ours
        # standing in each other (centres within 8 m) loses 26.5 HP/s, by one 26.9, by two apart 32.5; three 29.0 /
        # 39.4.
        s_j, c_j = torch.sin(pw["rel_j"]), torch.cos(pw["rel_j"])
        front_j, depth_j = pw["front"][:, None, :], pw["depth"][:, None, :]
        thru_j = s_j.abs() * depth_j <= c_j.abs() * front_j
        side_j = torch.where(thru_j, torch.where(c_j >= 0, 0, 3), torch.where(s_j >= 0, 1, 2))
        used = F * along / cal["fighting_files"]                 # metres of j's side each striker's men take
        for g in range(4):
            in_g = side_j == g
            u_g = torch.where(in_g & ~single_i, used, torch.zeros_like(used))
            # (a lone attacker keeps what it brings - through a flank by its own front, melee.flank_face: the cap is
            # the side's length or the longest single attacker's, whichever is more)
            side_len = torch.maximum(front_j if g in (0, 3) else depth_j, u_g.amax(dim=1, keepdim=True))
            tot = u_g.sum(dim=1, keepdim=True)
            F = torch.where(in_g & ~single_j & ~single_i & (tot > side_len), F * side_len / tot.clamp(min=1e-6), F)
    wave = cal.get("wave")
    if wave and "contact_s" in u:
        # The opening wave (melee.wave): two formations' fronts meet closer than they fight later and part over
        # tau_s, so more men stand within reach early: men striking x (1 + amp exp(-t / tau_s)), t the pair's time in
        # melee (the younger of the two clocks: a fresh unit meets its enemy's front anew); formations only.
        t_pair = torch.minimum(u["contact_s"][:, :, None], u["contact_s"][:, None, :])
        w = 1.0 + float(wave["amp"]) * torch.exp(-t_pair / float(wave["tau_s"]))
        floor = wave.get("missile_attack_floor")
        if floor:
            # A missile unit fighting under an attack order (the wavemiss probes, build/routmorale/wavemiss_report.py):
            # its loose block does not part from the enemy as formations do - the fronts stay 2.1-2.6 m apart (unordered
            # shooters and formations: 3.0-3.3 m) and the enemy's men within reach of it stay at 22.6 against 16.4 for the
            # same shooters unordered, 10-90 s after contact (8 / 4 lanes): the men striking it keep at least
            # missile_attack_floor (1.38) of the steady level.
            shot_at = ((u["range"] > 0) & (u["order_kind"] == O.ATTACK))[:, None, :]
            w = torch.where(shot_at, w.clamp(min=float(floor)), w)
        F = torch.where(single_i | single_j, F, F * w)
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
    # Men gather round a lone man (a lord) who ran into their formation over contact.lord_gather_s from the
    # striker's contact (0: at once); infantry that runs onto a standing lord strikes him in full at once (the lord
    # swarm probe: his first 15 s 1.11 x his steady loss; build/melee2/spec.md L4).
    gather = float(cc.get("lord_gather_s", 0.0))
    if gather > 0:
        grow = (u["contact_s"] / gather).clamp(0, 1)[:, :, None]
        ran_j = u["ran_in"][:, None, :] if "ran_in" in u else torch.ones_like(single_j)
        rate = torch.where(single_j & ~single_i & ran_j, rate * grow, rate)
    if pursuit is not None:
        share = torch.full_like(rate, float(pursuit))
        if cc.get("rout_pin_s") and cc.get("rout_pin_rate") is not None and "rpin_s" in u:
            # a router in its rout's exit (battle.py, contact.rout_pin_s), still among the men it fought, is struck at
            # contact.rout_pin_rate of the rule (the recordings: 0.77 of a standing unit's loss in melee)
            share = torch.where((u["rpin_s"] > 0)[:, None, :], float(cc["rout_pin_rate"]), share)
        rate = torch.where(routing_j, rate * share, rate)
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
    inc = torch.full_like(rate, float(cc.get("unit_incidental", 1.0)))
    if cc.get("fresh_incidental") is not None and "order_s" in u:
        # Within the database's melee_breakoff_secs (24 s) of an attack order given in melee on another enemy, it
        # strikes the units it only touches at contact.fresh_incidental (the fresh-order probe,
        # build/charge-probe/runs/20261008-145559 / -145641: swordsmen told to attack clanrats 4 m beside the clanrats
        # they fought dealt those 0 HP for 25 s in 2 lanes of 2, still touching them, then fought them again).
        fresh = (u["order_s"] < float(B.get("melee_breakoff_secs", 0.0)))[:, :, None]
        inc = torch.where(fresh, float(cc["fresh_incidental"]), inc)
    rate = torch.where(~single_j & ~single_i & busy, rate * inc, rate)
    # A formation without a missile weapon standing under HOLD in melee (no attack order) strikes at hold_rate of
    # the rule (measured: the melee probe's held units - braced spearmen, spearmen facing away, swordsmen - strike
    # clanrats at 0.49-0.52 of it from 5 s on, 6 lanes; the network's held units in the gates 0.80 of the kills of
    # the same units attacking); missile units 0.99-1.01 and a lone man (a lord held by script strikes at 0.77-1.19
    # of the rule, the damage plan): in full. The database's melee_attack_threshold_modifier_* (vanilla idle 0.14,
    # ordered 0.22; all 1.0 with the required mod True Sight, config/nn/game_rules.json "_mods") are when a unit joins
    # a fight, not its rate (build/melee2/spec.md 3): no rule for the rate found.
    # A formation under a MOVE order still in contact (not leaving it: a far point is a leaving unit, which strikes
    # nobody) strikes like a held one (contact.hold_move; build/open_melee/spec.md R3: blocked by the enemy its men do
    # not step into the gaps - the probe's move lanes struck 0.70 of the same units attacking, men within 2.5 m 0.73).
    held_kind = u["order_kind"] == O.HOLD
    if cc.get("hold_move"):
        move = u["order_kind"] == O.MOVE
        if cc.get("move_far_full") and cc.get("leave_m"):
            # a MOVE to a far point through the enemy (contact.leave_away_only: not a leave) keeps pushing and strikes
            # in full (contact.move_far_full; build/probes7 P3: spearmen walking to a point 60 m beyond their attacker
            # struck 29 HP/s, answering with an attack order 21); a point within leave_m (R3's 5 m) - held
            far = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) >= float(cc["leave_m"])
            move = move & ~far
        held_kind = held_kind | move
    # contact.hold_missile: a missile unit under HOLD is held too (the wavemiss probes: unordered shooters struck clanrats
    # at 0.45 HP/s a man within reach, the same shooters under an attack order 0.80 - 0.56 of it; the twin's held
    # shooters dealt 3.4x the game's); absent: missile units strike in full (the old note: 0.99-1.01, source not kept)
    missile_held = bool(cc.get("hold_missile"))
    held = held_kind & ((u["range"] <= 0) | missile_held) & (u["men0"] > 1)
    # A braced unit's blows while it reflects a charge are not cut (contact.hold_reflect_full; R2: braced spearmen
    # struck 52 HP/s in 1-5 s after a charge = the full rule x2, held swordsmen 12 HP/s = hold_rate).
    cut = held[:, :, None]
    if cc.get("hold_reflect_full"):
        cut = cut & (reflect <= 1.0)
    full = rate
    rate = torch.where(cut, rate * float(cc.get("hold_rate", 1.0)), rate)
    if not first:
        return rate, hit, sector, F
    # One blow of every man in contact (the first strike): the rate's men and multipliers, one swing each; every man in
    # reach swings, hold_rate is about the men who step in later (contact.first_strike_full; R1).
    swing_rate = full if cc.get("first_strike_full") else rate
    return rate, hit, sector, F, swing_rate * (p / per_s.clamp(min=1e-9))
