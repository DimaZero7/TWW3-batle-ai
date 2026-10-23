"""Melee: who strikes whom and how much health it takes (docs/en/training/simulator.md,
docs/en/game/mechanics/melee.md).

Per pair of units in contact (i strikes j), per second:

    men striking F  = fighting_files x (length of the sides in contact / the striker's spacing along its side:
                      h across its front or back, v along a flank; its formation template, config/nn/units.json
                      spacing; an ellipse body - a beast - takes its width there, melee.body_across), at most
                      men (through the target's flank: the striker's own side, melee.flank_face "striker");
                      against a single man (a lord) at most lord_max_attackers in all, however many units surround him (the lord swarm probe, docs/en/game/units/lord-swarm.md); a single man
                      strikes once; a unit in contact with several enemies shares out no more than its own front
                      holds; a unit attacking another enemy strikes one it only touches at contact.unit_incidental
                      (a lord: lord_incidental) of its rate; a formation without a missile weapon under HOLD (no
                      attack order; a missile unit too: contact.hold_missile) or under a MOVE still in contact (contact.hold_move) strikes at contact.hold_rate
                      of it (measured: the melee probe's held units 0.49-0.52 of the rule, move lanes 0.70 of
                      attacking), except its blows reflecting a charge (contact.hold_reflect_full); a lone man (a lord)
                      under HOLD strikes in full (the damage plan)
    hit chance p    = 35 + attack + charge bonus x charge left + bonus v target - defence x direction (rounded half
                      away from zero), within 8-90, less melee.hit_roll_offset (1), / 100 (the game's rules; the
                      database's melee_hit_chance_*)
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
                      (battle.py decides when: a formation at its first contact of the battle only, contact.opening_once);
                      F without the opening wave (contact.first_strike_steady)

Direction: defence x0.6 from the flank, x0.3 from the rear or against a routing unit (database, weight 1;
a standing lone man: contact.lord_direction of it - the lord swarm probe's measure, the game's rules have no separate
branch for lone men -, a routing lone man: lone_rout.direction, in full). A unit brings to each side of its formation no
more men than that side holds. A lone man fought by the enemy lord takes lord_rival_others of the infantry's rate.
Charge (battle.py): an attack order, a run-up and the charge's own clock, linear to 0 over charge_decay_duration
(13 s); its first blows are the first strike above; nothing beyond the bonus (no impact, no ramp: CA,
build/charge/spec.md). A braced unit with
charge_reflection strikes a charger in front of it (within bracing_attack_angle) whose charge is still at least
charge_reflect_min_charge_factor_threshold at charge_reflect_damage_multiplier of its damage (database).
Pursuit: a routing target is struck from the rear with all the men in contact at contact.pursuit_rate of the
rule and no charge (measured; the game's rules give every pursuit blow the whole charge - a known gap); a lone
striker on a router at the rule in full, on his routing attack target with the whole charge (the game's rules:
lone_rout, config/nn/sim.json contact.lone_rout); a chaser that keeps up engulfs it (contact.pursue_envelop), its men running in
from their ranks since the rout (contact.pursue_envelop_run).
"""
import torch

from tools.nn.sim import fatigue, geometry, knobs, lone_rout
from tools.nn.sim import orders as O


def hit_chance(attack, defence, slope=1.0, base=35.0, lo=8.0, hi=90.0, offset=0.0):
    """The hit chance 0-1 (slope: the database's melee_hit_chance_normalisation_coefficient, 1): the clamped score less
    offset, / 100 (the game's rules: (score - 1) / 100 - melee.hit_roll_offset 1)."""
    return (torch.clamp(base + slope * (attack - defence), lo, hi) - offset) / 100


def round_half_away(x):
    """x rounded to the nearest whole number, halves away from zero (13.5 -> 14, -2.5 -> -3)."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def directional_defence(defence, factor):
    """The defence a blow meets: defence x factor (the flank's 0.6, the rear's 0.3), rounded half away from zero where
    the factor cuts it (the game's rules); unchanged where factor is 1."""
    cut = defence * factor
    return torch.where(factor < 1.0, round_half_away(cut), cut)


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


def _tau_of(m_obs, Ks):
    """tau with E[k | k < K] = tau Q(K - 1, tau) / Q(K, tau) = m_obs (bisection, as shot_kills)."""
    m_obs = torch.minimum(m_obs, 0.95 * (Ks - 1))
    lo, hi = torch.zeros_like(Ks), 4 * Ks + 30
    lg = torch.lgamma(Ks)
    for _ in range(16):
        mid = (lo + hi) / 2
        q = torch.special.gammaincc(Ks, mid).clamp(min=1e-30)
        pk = torch.exp((Ks - 1) * torch.log(mid.clamp(min=1e-30)) - mid - lg)
        m = mid * (1 - pk / q)
        up = m < m_obs
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    return (lo + hi) / 2


def _pool(n, tau, x, Ks):
    """One pool of n living men with Poisson(tau) hits so far (each under K) taking x more hits a man: (men killed,
    hits beyond the dead men's K - lost, a man's health is all he loses -, the survivors' mean hits)."""
    s0 = torch.special.gammaincc(Ks, tau).clamp(min=1e-30)
    t1 = tau + x
    s1 = torch.special.gammaincc(Ks, t1)
    kills = n * (1 - s1 / s0).clamp(0, 1)

    def excess(t):                       # E[(S - K)+], S ~ Poisson(t)
        return t * (1 - torch.special.gammaincc(Ks - 1, t)) - Ks * (1 - torch.special.gammaincc(Ks, t))
    # E[(S - K)+ ; k0 < K] = E+(tau + x) - E+(tau) - x P(k0 >= K), over the living at the start (/ s0)
    waste = n * ((excess(t1) - excess(tau) - x * (1 - s0)).clamp(min=0) / s0)
    waste = torch.minimum(waste, n * x)
    mean = t1 * torch.special.gammaincc(Ks - 1, t1) / s1.clamp(min=1e-30)
    return kills, waste, torch.minimum(mean, Ks)


def front_kills(h_front, h_rest, men, front, hp_man, hp_abs, per_hit, fw):
    """Projectile kills with the front rank struck first (kills.missile_front; missile.front_share): the unit's living
    men are two pools - its front rank (`front` men: its files, at most its men) and the rest - each man with his own
    health and Poisson hits (shot_kills). h_front hits fall on the front rank only, h_rest spread evenly over all the
    living. fw: the front rank's wounds (HP it lacks; the rest's are the unit's wounds W less these). A front man who
    dies is replaced by a man of the rest stepping up (the game's file closing up), his wounds with him. Returns
    (men killed, hits lost beyond the dead men's health - a man takes at most his health, CA -, the front's new
    wounds). No fitted number: the front's share is the spread model's, the pools Poisson as shot_kills."""
    A = men.clamp(min=1e-6)
    a = torch.minimum(front, men).clamp(min=1e-6)
    rest = (men - a).clamp(min=0)
    ph = per_hit.clamp(min=1e-6)
    K = (hp_man / ph).clamp(min=1.0)
    Ks = K.clamp(min=1.02)
    W = (men * hp_man - hp_abs).clamp(min=0)
    Wa = torch.minimum(fw.clamp(min=0), W)
    Wr = W - Wa
    xr = h_rest / A
    xa = h_front / a + xr
    ta = _tau_of(Wa / (a * ph), Ks)
    tr = _tau_of(Wr / (rest.clamp(min=1e-6) * ph), Ks)
    ka, wa, ma = _pool(a, ta, xa, Ks)
    kr, wr, mr = _pool(rest, tr, xr, Ks)
    kr = torch.where(rest > 0, kr, torch.zeros_like(kr))
    wr = torch.where(rest > 0, wr, torch.zeros_like(wr))
    up = torch.minimum(ka, (rest - kr).clamp(min=0))          # the men stepping up into the front's gaps
    fw_new = ph * ((a - ka).clamp(min=0) * ma + up * mr)
    one = K < 1.02                                            # a hit that kills: a man a hit, nothing lost
    hits = h_front + h_rest
    kills = torch.where(one, hits, ka + kr).clamp(min=0) * (men > 0)
    lost = torch.where(one, torch.zeros_like(hits), wa + wr) * (men > 0)
    return kills, lost, torch.where(one | (men <= 0), torch.zeros_like(fw_new), fw_new)


def spacing_of(u, spacing):
    """(h, v) [B, N]: a man's place across the front and between ranks, m (the unit's formation template,
    config/nn/units.json spacing; `spacing` where a unit has none)."""
    if "sp_h" not in u:
        return torch.full_like(u["men"], float(spacing)), torch.full_like(u["men"], float(spacing))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    return h, v


def contact_spacing(u, h, v, on):
    """(h, v) [B, N]: a man's place along a contact, m (melee.body_across; on False: the formation's h, v). In melee the
    men close up to their neighbours body to body, keeping the template's gap between bodies, and face the enemy: an
    ellipse body (the database's battle_entities shape 'ellipse', radius long, radius x radii_ratio across - a beast, a
    weapon team) takes its width, not its length, so its place is h - 2 r (1 - radii_ratio) (v the same); a circle
    (every man of the infantry: radii_ratio 1) keeps h, v."""
    if not on or "body_ratio" not in u:
        return h, v
    ratio = u["body_ratio"]
    ratio = torch.where(ratio > 0, ratio, torch.ones_like(ratio)).clamp(max=1)
    cut = 2 * u["radius"] * (1 - ratio)
    return (h - cut).clamp(min=0.3), (v - cut).clamp(min=0.3)


def strikes(u, pw, contact, params, charge_now, first=False, ran=None):
    """HP per second each unit i takes from each j: rate [B, N, N] (i strikes j), the per-hit damage [B, N, N],
    the direction sector of i seen from j and the men striking F [B, N, N]; with first, also swing [B, N, N]: the
    HP of one blow of every man in contact at once (the first strike, battle.py). charge_now [B, N]: the striker's
    charge left (0-1, battle.py: its own clock from the charge's first blow). ran [B, N] (with first): the strikers
    whose charge lands now (battle.py: the first strike of a charge), whose swing is their front rank
    (contact.first_strike_rank)."""
    B = params.battle
    cal = params.sim["melee"]
    cc = params.sim["contact"]
    h_u, v_u = spacing_of(u, params.sim["formation"]["spacing_m"])
    # a man's place along a contact (melee.body_across): the formation's h, v for circles, the body's width for an ellipse
    hc_u, vc_u = contact_spacing(u, h_u, v_u, cal.get("body_across"))
    men_i = u["men"][:, :, None]
    single_i = u["men0"][:, :, None] <= 1
    single_j = u["men0"][:, None, :] <= 1
    # Which side of i faces j: its front or back (men h apart) or a flank (v apart).
    front_i, depth_i = pw["front"][:, :, None], pw["depth"][:, :, None]
    s, c = torch.sin(pw["rel_i"]), torch.cos(pw["rel_i"])
    through_front = s.abs() * depth_i <= c.abs() * front_i
    edge = cal.get("frontage_edge_m")
    sat_j = None
    if edge is not None:
        # (melee.frontage_edge_m, below: the side of contact by the rectangles' least overlap, geometry.front_contact,
        # not by the line between the centres; a lone man's keeps the centres' line)
        sat = geometry.front_contact(u["x"], u["z"], u["b"], pw["front"], pw["depth"])
        lone = single_i | single_j
        through_front = torch.where(lone, through_front, sat)
        sat_j = torch.where(lone, pw["face_j"] == pw["front"][:, None, :], sat.transpose(1, 2))
    along = torch.where(through_front, hc_u[:, :, None], vc_u[:, :, None])
    # The front's fill (melee.front_fill_ranks, lam; 0 / absent: off): a thinned formation does not close up - it keeps
    # its ordered width (the dmgmelee probe: 25-28 of 30 m) and only part of its men reach the enemy. Its frontal face
    # is the ordered width (files0 = floor(width / h), not the men's), and its men striking are scaled by the fill of
    # that front, occ(r) = 1 - exp(-r / lam), r = men / files0 (men a file), against the whole unit's occ(r0): a whole
    # unit keeps its calibrated F. MEASURED from the soldiers' places (config/nn/sim.json melee.front_fill_why).
    lam = float(cal.get("front_fill_ranks") or 0.0)
    if lam > 0:
        files0 = torch.clamp(torch.floor(u["width"] / h_u + 1e-4), min=1)
        r_i = (u["men"].clamp(min=0) / files0)[:, :, None]
        r0_i = (u["men0"].clamp(min=1) / files0)[:, :, None]
        face_i = torch.where(through_front, (files0 * h_u)[:, :, None],
                             pw["face_i"] if sat_j is None else depth_i.expand_as(pw["face_i"]))
        lam = knobs.get(u, "front_fill_ranks", lam, 3)      # (a per-battle draw in training: tools/nn/train/dr.py)
        fill = (1 - torch.exp(-r_i / lam)) / (1 - torch.exp(-r0_i / lam))
    else:
        face_i = pw["face_i"] if sat_j is None else torch.where(through_front, front_i, depth_i)
        fill = None
    face_j = pw["face_j"] if sat_j is None else torch.where(sat_j, pw["front"][:, None, :], pw["depth"][:, None, :])
    length = torch.minimum(face_i, face_j)
    length_meet = length          # the two sides' common length (the impact's front: contact.first_strike_rank)
    flank_j = torch.zeros_like(through_front)
    wrap = flank_j
    if cal.get("flank_face", "min") == "striker":
        # Through the target's flank (its side towards i is its depth) the striker brings men by
        # its own face, not by the target's short flank (measured: a lone flank attacker takes
        # 1.53x what a frontal one does; melee.flank_face, config/nn/sim.json).
        s_j, c_j = torch.sin(pw["rel_j"]).abs(), torch.cos(pw["rel_j"]).abs()
        flank_j = s_j * pw["depth"][:, None, :] > c_j * pw["front"][:, None, :] if sat_j is None else ~sat_j
        wrap = flank_j
        if cal.get("flank_face_busy"):
            # A target held through its front or back by another unit (melee.flank_face_busy; config/nn/sim.json
            # melee.flank_face_busy_why): its men face that enemy and do not turn to mesh with the newcomer on their flank,
            # so the newcomer reaches them only along that flank - the two sides' common length (the busycharge probe:
            # swordsmen charging the flank of clanrats fighting our spearmen - 15 of their men within 2.5 m of a clanrat
            # against 26-28 through the back, the clanrats' HP lost 0.5 of a charge into their back after the impact).
            # Only a newcomer (i in melee for less time than j): a pair that met before j turned its flank (to leave, to
            # attack a neighbour - the defender probe) stays meshed.
            held = (contact & ~flank_j).any(dim=1, keepdim=True)       # j fought through its front or back by anyone
            if "contact_s" in u:
                held = held & (u["contact_s"][:, :, None] + float(params.dt) < u["contact_s"][:, None, :])
            wrap = flank_j & ~held
        length = torch.where(wrap, face_i, length)
    if edge is not None:
        # The contact's frontage (melee.frontage_edge_m; absent: off): a formation's men strike the enemies within their
        # reach, whichever unit those belong to - so i brings to j the men of the part of its side that j covers, not its
        # whole side: i's side towards j (its front or back, h apart, or a flank) against j's box seen along that side
        # (j's front and depth at j's bearing), their overlap, plus frontage_edge_m for the men at the overlap's ends who
        # reach on past it; at most the length above (a formation square on another: unchanged). Formations only, and not
        # through the target's flank under melee.flank_face "striker" (its own measured rule: the men wrap round).
        # MEASURED (the sidecontact probe, build/adjmelee; config/nn/sim.json melee.frontage_edge_why).
        b = u["b"] * geometry.DEG
        sb, cb = torch.sin(b), torch.cos(b)
        # i's side direction: across its front (cos b, -sin b) through the front or back, along its facing on a flank
        lx = torch.where(through_front, cb[:, :, None], sb[:, :, None])
        lz = torch.where(through_front, -sb[:, :, None], cb[:, :, None])
        half = torch.where(through_front, face_i, depth_i.expand_as(face_i)) / 2
        rx = u["x"][:, None, :] - u["x"][:, :, None]
        rz = u["z"][:, None, :] - u["z"][:, :, None]
        off = rx * lx + rz * lz                                    # j's centre along i's side
        ext = (pw["front"][:, None, :] / 2 * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
               + pw["depth"][:, None, :] / 2 * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
        ov = (torch.minimum(half, off + ext) - torch.maximum(-half, off - ext)).clamp(min=0)
        length = torch.where(single_i | single_j | wrap, length, torch.minimum(length, ov + float(edge)))
    F = cal["fighting_files"] * length / along
    F = torch.minimum(F, men_i)
    if fill is not None:
        F = torch.where(single_i, F, F * fill)
    lord_max = knobs.get(u, "lord_max_attackers", float(cc["lord_max_attackers"]), 3)
    F = torch.where(single_j, torch.minimum(men_i, knobs.full_like(F, lord_max)), F)
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
        hold = cal["fighting_files"] * (front_i / hc_u[:, :, None] if g in (0, 3) else depth_i / vc_u[:, :, None])
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
    # the men striking without the opening wave: the first strike's (contact.first_strike_steady)
    F0 = F if (first and cc.get("first_strike_steady")) else None
    F_rank = None
    if first and ran is not None and cc.get("first_strike_rank"):
        # The charge's front rank (contact.first_strike_rank; config/nn/sim.json contact.first_strike_rank_why): a
        # formation whose charge lands meets the enemy with its whole front rank at once - a man of every file in
        # contact runs into an enemy and swings (his interval is ready: CA Feature Focus #2) - so its first strike is
        # the files in contact (the steady men without the fighting_files share: F / fighting_files, the same caps), on
        # the length where the two sides meet: through the target's flank the files that run into that side, not the
        # striker's whole front (melee.flank_face striker is the men wrapping round later). A unit that comes in without
        # a charge (a move order, a blocked way, a walk-in under the run-up) and a standing one keep the steady front's
        # swing (the probes: swordsmen running in under a move order 96 HP/s at 0-1 s, the steady swing 89). Formations
        # only.
        F_rank = F / float(cal["fighting_files"])
        F_rank = torch.where(flank_j, F_rank * (length_meet / length.clamp(min=1e-6)).clamp(max=1.0), F_rank)
        F_rank = torch.minimum(F_rank, men_i)
        rank_on = ran[:, :, None] & ~single_i & ~single_j
        F_rank = torch.where(rank_on, F_rank, torch.zeros_like(F_rank))
    wave = cal.get("wave")
    if wave and "contact_s" in u:
        # The opening wave (melee.wave): two formations' fronts meet closer than they fight later and part over
        # tau_s, so more men stand within reach early: men striking x (1 + amp exp(-t / tau_s)), t the pair's time in
        # melee (the younger of the two clocks: a fresh unit meets its enemy's front anew); formations only.
        # (contact.opening_once: the clock is the unit's seconds in melee over the battle, melee_s - a formation back
        # in melee after a time out gets no new wave; battle.py. melee_s >= contact_s in a run; the larger of the two
        # keeps a contact clock set from outside - a recording, a test - in force)
        clock = (torch.maximum(u["contact_s"], u["melee_s"]) if cc.get("opening_once") and "melee_s" in u
                 else u["contact_s"])
        t_pair = torch.minimum(clock[:, :, None], clock[:, None, :])
        w = 1.0 + knobs.get(u, "wave_amp", float(wave["amp"]), 3) * torch.exp(-t_pair / float(wave["tau_s"]))
        floor = wave.get("missile_attack_floor")
        if floor:
            # A missile unit fighting under an attack order (the wavemiss probes, build/routmorale/wavemiss_report.py):
            # its loose block does not part from the enemy as formations do - the fronts stay 2.1-2.6 m apart (unordered
            # shooters and formations: 3.0-3.3 m) and the enemy's men within reach of it stay at 22.6 against 16.4 for the
            # same shooters unordered, 10-90 s after contact (8 / 4 lanes): the men striking it keep at least
            # missile_attack_floor (1.38) of the steady level.
            shot_at = ((u["range"] > 0) & (u["order_kind"] == O.ATTACK))[:, None, :]
            w = torch.where(shot_at, w.clamp(min=float(floor)), w)
            if F0 is not None:          # (the floor is the steady level)
                F0 = torch.where(single_i | single_j | ~shot_at, F0, F0 * float(floor))
        F = torch.where(single_i | single_j, F, F * w)
    if cc.get("pursue_envelop"):
        # A caught router is engulfed (contact.pursue_envelop; config/nn/sim.json contact.pursue_envelop_why): a chaser
        # with its attack order on a routing formation it touches, whose run is at least the router's free flight
        # (morale.rout_speed of the router's run), does not fight it front to front - its men pick charge targets among
        # the routing mob's men (the database: pursue_charge_max_unit_separation 8 m, max_pursue_charges_ratio 1 - every
        # man of the chaser may) and run into the loose mob, so a router man has a chaser man on him: men striking =
        # min(the chaser's men, the router's men), past the formation's side caps (the pursuit probe: stormvermin after
        # Empire spearmen, wolf rats after swordsmen - chaser men within 2.5 m of a router man 26-33, 0.95-1.4 of the
        # router's men, against 7-9 on a 30 m front standing; a slower chaser, swordsmen after clanrats, does not catch).
        routing = u["r"][:, None, :] & ~single_i & ~single_j
        on_j = (u["order_kind"][:, :, None] == O.ATTACK) & \
            (u["order_target"][:, :, None] == torch.arange(u["men"].shape[1], device=F.device)[None, None, :])
        flee = u["run"][:, None, :] * float(params.sim["morale"]["rout_speed"])
        keeps_up = u["run"][:, :, None] >= flee
        caught = routing & on_j & keeps_up & contact & (u["range"][:, :, None] <= 0)
        engulf = torch.minimum(men_i, u["men"][:, None, :])
        if cc.get("pursue_envelop_run") and "rout_s" in u:
            # The chaser's men run in (contact.pursue_envelop_run): at the rout they stand in their formation's ranks and
            # each runs from his place to a router man, from a standstill at the unit's acceleration to its run (the
            # database's battle_entities), so the men it brings grow with the ground covered since the rout s(t) over
            # its formation's depth: F + (engulf - F) x min(1, s(t) / depth) (a router caught later: long run, whole)
            t = u["rout_s"][:, None, :]
            a = u["accel"][:, :, None].clamp(min=0.1)
            v = u["run"][:, :, None].clamp(min=0.1)
            ground = torch.where(t < v / a, 0.5 * a * t * t, v * t - v * v / (2 * a))
            frac = (ground / depth_i.clamp(min=0.5)).clamp(0, 1)
            engulf = F + (engulf - F).clamp(min=0) * frac
        F = torch.where(caught, torch.maximum(F, engulf), F)
    if (cc.get("rout_stuck") or {}).get("on") and "stk_n" in u:
        # A router's men stuck among a standing formation's (battle.py, contact.rout_stuck) each have a man of it on them,
        # wherever the rest of the router is: men striking = min(its men, the stuck men).
        slot = torch.arange(u["men"].shape[1], device=F.device)[None, :, None]
        among = (u["stk_k"][:, None, :] == slot) & (u["stk_n"] > 0)[:, None, :] & contact
        F = torch.where(among, torch.maximum(F, torch.minimum(men_i, u["stk_n"][:, None, :])), F)
    own =torch.where(single_i, torch.ones_like(men_i), men_i)
    total = F.sum(dim=2, keepdim=True)
    F = F * torch.where(total > own, own / total.clamp(min=1e-6), torch.ones_like(total))
    if F0 is not None:
        total0 = F0.sum(dim=2, keepdim=True)
        F0 = F0 * torch.where(total0 > own, own / total0.clamp(min=1e-6), torch.ones_like(total0))
    if F_rank is not None:
        totr = F_rank.sum(dim=2, keepdim=True)
        F_rank = F_rank * torch.where(totr > own, own / totr.clamp(min=1e-6), torch.ones_like(totr))
    # At most lord_max_attackers around a single man, however many units they belong to (the
    # lord swarm probe); an enemy lord among them keeps his place (one man) and the infantry
    # share the rest.
    lone = (single_i & (F > 0)).float().sum(dim=1, keepdim=True)
    around = torch.where(single_i, torch.zeros_like(F), F).sum(dim=1, keepdim=True)
    cap = (lord_max - lone).clamp(min=0)
    F = torch.where(single_j & ~single_i & (around > cap), F * cap / around.clamp(min=1e-6), F)
    if F0 is not None:
        around0 = torch.where(single_i, torch.zeros_like(F0), F0).sum(dim=1, keepdim=True)
        F0 = torch.where(single_j & ~single_i & (around0 > cap), F0 * cap / around0.clamp(min=1e-6), F0)

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
        k_chase = lone_rout.charge(cc, B)
        if k_chase is not None:
            # A lone striker whose attack target routs (the enemy lord chasing our lord) strikes it with his whole
            # charge bonus x the database's pursuit_charge_bonus_modifier, no decay (the game's rules; lone_rout.charge).
            # Formations keep the measured pursuit without it (contact.pursuit_why).
            slot_j = torch.arange(u["men"].shape[1], device=ch.device)[None, None, :]
            chase = lone_rout.chasing(single_i, u["order_kind"][:, :, None], u["order_target"][:, :, None], slot_j,
                                      routing_j, O.ATTACK)
            # (the stat without the fatigue: effects() has laid its multiplier on u["charge_bonus"])
            full = k_chase / fatigue.stat_mult(u, params, "stat_charge_bonus")[:, :, None]
            ch = torch.where(chase, full.expand_as(ch), ch)
    large_j = u["large"][:, None, :]
    bonus = torch.where(large_j, u["bonus_v_large"][:, :, None], u["bonus_v_inf"][:, :, None])
    extra = u["charge_bonus"][:, :, None] * ch + bonus
    attack = u["attack"][:, :, None] + extra
    # Defence from the flank / rear: x0.6 / x0.3 (the database, weight 1); against a lone man (a lord)
    # contact.lord_direction of the loss (measured in the lord swarm probe: infantry on his back or flank hurts him
    # hardly more than on his front).
    # The game's rules have no separate branch for lone men (the measure above is kept for a standing one); a routing
    # lone man runs with his back to his chasers and does not turn to them: lone_rout.direction of the rule
    # (config/nn/sim.json contact.lone_rout; absent: lord_direction). The cut defence is rounded half away from zero.
    w = torch.where(single_j, lone_rout.direction(cc, routing_j.expand_as(coef), coef.dtype), torch.ones_like(coef))
    defence = directional_defence(u["defence"][:, None, :].expand_as(coef), 1 - w * (1 - coef))
    p = hit_chance(attack, defence, B["melee_hit_chance_normalisation_coefficient"], B["melee_hit_chance_base"],
                   B["melee_hit_chance_min"], B["melee_hit_chance_max"], float(cal.get("hit_roll_offset", 0.0)))
    dmg, ap = u["damage"][:, :, None], u["ap_damage"][:, :, None]
    share = dmg / (dmg + ap).clamp(min=1e-6)
    # Charge reflection (database): a braced unit with charge_reflection (standing: slower than melee.braced_speed)
    # strikes a charger within bracing_attack_angle of its front, while the charger's charge is at least
    # charge_reflect_min_charge_factor_threshold, at charge_reflect_damage_multiplier of its damage (CA: spearmen
    # do double damage within 3.9 s of the charge); it gets no charge of its own.
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    braced = (u["reflect"] & (speed < cal["braced_speed"]))[:, :, None]
    facing = pw["rel_i"].abs() <= B["bracing_attack_angle"] * geometry.DEG
    if cal.get("reflect_any_angle"):
        # (melee.reflect_any_angle) the reflection asks only that the striker is braced and the charger's charge is on,
        # not where the charger came from: the brace probe - spearmen's single blows on clanrats that charged them at
        # 100 deg (their flank) 41-43 HP in the charge's first seconds (7 of 8), 17-23 later (one blow 21); melee.reflect_why
        facing = torch.ones_like(facing)
    charging_j = (charge_now >= B["charge_reflect_min_charge_factor_threshold"])[:, None, :]
    reflect = torch.where(braced & facing & charging_j & ~routing_j,
                          torch.full_like(F, float(B["charge_reflect_damage_multiplier"])), torch.ones_like(F))
    # A splash blow (a lord's) on a formation strikes contact.lord_splash_struck men on average (measured; at most
    # splash_max_attacks and the men left), each taking 1 / splash_max_attacks of it (CA 5.1.0: divided among the
    # targets; measured: single blows of 47-57 HP on stormvermin); a lone man takes the whole blow.
    n_max = u["splash"][:, :, None].expand_as(F)
    struck = torch.minimum(n_max, knobs.full_like(n_max, knobs.get(u, "lord_splash_struck",
                                                                   float(cc.get("lord_splash_struck", 4.0)), 3)))
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
    if cal.get("swing_interval") and "ran_in" in u and "contact_s" in u:
        # The first strike starts the men's intervals (melee.swing_interval; CA Feature Focus #2: a man's attack
        # interval runs after his blow): a unit that came into its fight moving or charging (ran_in) struck once with
        # every man in contact at the contact (battle.py, the first strike), so its men who hit wait their interval
        # before the next blow and those who missed melee.miss_s - the steady rate starts as they come back: x(1 - p)
        # from miss_s, whole from the interval (t: the striker's seconds since that contact; a formation at its first
        # fight of the battle only - contact.opening_once gives no later first strike). Formations only: a lone man's
        # first blow is still ~3x weaker than the game's (OPEN), the pause would only deepen that.
        # The busycharge probe (config/nn/sim.json melee.swing_why): swordsmen charging free clanrats head-on, the
        # clanrats' HP a second at 0-1 / 1-2 / 2-3 / 3-5 / 5-10 s after the impact 259 / 40 / 64 / 35 / 50 - a pause after
        # the impact, not a charge rate.
        t = (u["contact_s"] - float(params.dt))[:, :, None]
        swung = u["ran_in"][:, :, None] & ~single_i
        if "melee_s" in u:
            swung = swung & (u["melee_s"] - u["contact_s"] < 0.5 * float(params.dt))[:, :, None]
        back = torch.where(t >= float(cal["miss_s"]), 1 - p, torch.zeros_like(p)) + torch.where(t >= interval, p,
                                                                                               torch.zeros_like(p))
        back = torch.where(swung, back, torch.ones_like(back))
    else:
        back = None
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
        share = knobs.full_like(rate, knobs.get(u, "pursuit_rate", float(pursuit), 3))
        if cc.get("rout_pin_s") and cc.get("rout_pin_rate") is not None and "rpin_s" in u:
            # a router in its rout's exit (battle.py, contact.rout_pin_s), still among the men it fought, is struck at
            # contact.rout_pin_rate of the rule (the recordings: 0.77 of a standing unit's loss in melee)
            share = torch.where((u["rpin_s"] > 0)[:, None, :], float(cc["rout_pin_rate"]), share)
        # a lone man striking a lone router (the enemy lord chasing our lord): lone_rout's share (config/nn/sim.json
        # contact.lone_rout; absent: the formations' pursuit_rate)
        share = lone_rout.share(cc, share, single_i)
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
    rate = torch.where(cut, rate * knobs.get(u, "hold_rate", float(cc.get("hold_rate", 1.0)), 3), rate)
    swing_from = rate
    if back is not None:
        rate = rate * back
    if not first:
        return rate, hit, sector, F
    # One blow of every man in contact (the first strike): the rate's men and multipliers, one swing each; every man in
    # reach swings, hold_rate is about the men who step in later (contact.first_strike_full; R1).
    swing_rate = full if cc.get("first_strike_full") else swing_from
    if F0 is not None:
        # The first strike swings the men of the steady front, not the opening wave's (contact.first_strike_steady;
        # config/nn/sim.json contact.first_strike_steady_why): the swing was checked against the game before the wave
        # came in (the melee probe: 0-1 s 189 HP against the game's 220, swordsmen charging clanrats); multiplied by the
        # wave (x2.48 at the contact) it gave 441 against 184-219, the recontact probe's 18 first contacts 211.
        swing_rate_steady = torch.where(F > 0, swing_rate * F0 / F.clamp(min=1e-9), torch.zeros_like(swing_rate))
    else:
        swing_rate_steady = swing_rate
    if F_rank is not None:
        # (contact.first_strike_rank, above: the front rank of a unit that ran in; the rest keep the steady swing)
        swing_rate = torch.where(rank_on, torch.where(F > 0, swing_rate * F_rank / F.clamp(min=1e-9),
                                                      torch.zeros_like(swing_rate)), swing_rate_steady)
    else:
        swing_rate = swing_rate_steady
    return rate, hit, sector, F, swing_rate * (p / per_s.clamp(min=1e-9))
