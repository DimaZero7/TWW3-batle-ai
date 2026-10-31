"""A large body's blows on smaller men: the impact of its charge and the knock of its weapon (docs/ru/game/mechanics/
cavalry.md; the game's program: build/diag_cav/mechanic.md M1-M11, R-work charge_module.md 3 and 5). Every number is the
database's (config/nn/game_rules.json battle); config/nn/sim.json contact.impact switches it (impact_why). Only a unit of
size above small (the cavalry: medium; the Skaven monsters: large, very large) striking a smaller one: a battle without such
a unit is unchanged bit for bit. A struck man keeps his feet with his body's knock_interrupts_ignore_chance
(ignore(), contact.impact knock_ignore: the lords 40 / 75 %, the monsters 20-65 %, infantry 0); a man who keeps his feet
takes no collision damage of a large / very large charger either (it lands on the men it knocks down alone).

The impact (land(): a large charger's charge lands this step - battle.py `fire`; a new charge lands it again):
  bracing     B of a target man: a unit that can brace (the database's can_brace, or charge_defense(_vs_large),
              charge_reflection) standing (slower than melee.braced_speed), not routing and not in its own charge:
              1 + (bracing_calibration_ranks_multiplier - 1) clamp((ranks - 1) / (bracing_calibration_ranks - 1), 0, 1)
              (1 + 3 clamp((ranks - 1) / 7): 3.14 for 6 ranks), x bracing_charge_reflector_bonus 2 with charge defence,
              within 1 .. bracing_max_multiplier_clamp 10; the blow from within impact.brace_deg (75) of its facing; else 1
  speed       a man the charger meets: dv = (1 + e) M / (m B + M) v (e: collision_restitution_coefficient 0.3; M, m the
              masses, v the pair's speed difference); the charger loses (1 + e) m B / (m B + M) v x k(the size
              difference: 1 -> 0.8, 2 -> 0.2, 3 -> 0.1, 4 -> 0) on each man it meets
  ranks       a charger meets h_charger / h_target men of a rank (its file's lane in the formation templates; a monster - size
              large and above - its body's width across: per_rank, contact.impact lane_body); where they go
              down (dv over collision_speed_delta_threshold_knocked_down 7 m/s; the flight past 10.5 is a knock-down too) it
              rides on into the next rank at its speed less its loss a man met, and stops at the first rank that stays up
              (a target man landing his own charge now: his dv halved - the program's dv^2 x 0.25)
  damage      each man met takes min(collision_damage_maximum 70, (v M / (m B))^collision_damage_normaliser 0.8 x
              collision_damage_modifier 0.6), collision_damage_armour_penetration_ratio 0.7 of it past the armour, once a
              landing (the 4 s immunity per attacker); a charger of size above medium only on men it knocks down, x0.5.
              The B here has its own angle (contact.impact damage_brace_deg 120; build/diag_cav/mechanic.md M30): the
              same brace B as the speed change's, taken while the charger is within 120 deg of the man's facing (the
              caller of the damage function tests f.d > 0 or cos^2 <= 0.25) - the speed change keeps brace_deg 75; from
              the rear cone (beyond 120 deg) no brace. The cavcheck probe's flank charge (90 deg; build/diag_cav/
              check_results.md) lost the clanrats 271 HP in the first second, 1515 in 13 s, against the twin's 882 / 1908
              with the 75 deg in the damage and 403 / 1440 with 120 (front lanes unchanged)
  lying       the men down lie U(knockdown_getting_up_delay_min 1.0, _max 2.7) s and then get up (contact.impact wave
              rise_s 1.1: the animation's length, MEASURED - rise()): the target's front share down at t s after the
              landing is down_1 x clamp((2.7 + rise - t) / 1.7, 0, 1) (kd_q, kd_t); a man lying or getting up strikes
              nobody (up_share) and is no target (M31-M32, the program: the 'fit as a target' flag, every tick - no new
              blow, no swing on him; the old M11 'the swing strikes him' was wrong)
  the chain   a man knocked down hits the man behind at his flight speed, at most 3.3 m/s (M8): (1 + e) m / (m B + m)
              x 3.3 <= 2.2 m/s, below the step back (3) - no chain in numbers; the horse riding on is the ranks above
The ride-in (contact.impact ride_in; build/diag_cav/mechanic.md M25-M27): the brace is each man's own, recomputed every
              tick - a man standing within his radius + 10 m of his place is braced, a man who steps is not (that same
              tick), a man in a reaction keeps the flag he had. Where the charger knocks the first rank down and rides
              on (ride = 1: kd_q of the landing), those men get up off their places and walk back and into the fight -
              out of the brace - while the horses stand among them: the charge defence and the charge reflection, which
              look at the struck / striking man's own brace, are off for that unit, and so is the bracing of a new
              landing (bracing()). A held unit's men strike such a charger in full (melee.strikes, ride_full: the
              horses stand in reach of the men round them, who need not step in - contact.hold_rate, the fitted half of
              a held unit's rate, was measured on enemies at its front). A charger the first rank stops (the target's
              m B above the horse's M: its dv < 0.65 v, below 7 m/s at a horse's charge 10.8 - M28) stays in front: no
              ride. ride goes back to 0 when no larger enemy touches the unit (the men re-form). MEASURED check: the
              cavcharge probe (build/charge-probe/runs/20261029-114229): braced clanrat spearmen, knocked down (dv 8.6),
              lost 1169 HP in 13 s - between the twin's 731 with the defence held all the fight and 1290 without it -,
              braced stormvermin, knocked back only (6.4), 378 (the twin with the defence 407); the knights lost 248 to
              held clanrats they rode into (the full rule 257, x hold_rate 129) and 1190 to the stormvermin that stopped
              them (x hold_rate 1221).
The second wave (contact.impact wave; build/diag_cav/mechanic.md M31-M34, the program; melee.strikes, deferred): after a
              ride-in no man of the charger strikes the unit until the men it knocked down get up - the men in its reach
              lie or get up (no target), the men behind are beyond them (a man down is an enemy's body for
              collision_disable_delay_knocked_down_vs_enemies 2 s) - so the first ~0.9 s are the collision damage alone and
              then nothing; as the men get up (the landing + U(1.0, 2.7) + rise 1.1 s: 2.1-3.8 s) every man of the
              charger in contact (F, the men within reach: the twin's 18-19 knights at 3 s, the recordings' 20-23 within
              3 m of a clanrat 1.5-5 s after the impact) swings his first blow - none of them struck since the impact -,
              his intervals from the mean of it (3 s). Off (wave on 0): the front rank's first blow over 1.0-2.7 s, the
              ranks behind at once (bit for bit the old rule).
The front rank first (contact.impact collide_down / strike_first; build/diag_cav/mechanic.md M35-M39, the program):
              the collision damage lands only on a man the push knocks down or sends flying (dv over 7 m/s) with the
              closing speed over 7 m/s (before the exchange), never on a man who keeps his feet, nor on one lying, flying or
              getting up (collide_down); the front rank's lances land at the landing on men still standing - the attack
              starts well before the bodies touch - and the men they kill meet no horse (lance_first); one attack timer a
              man, not reset by a new target: the front rank strikes again an interval after the landing (once the men are
              up); the ranks behind swing their first blow as the men get up with every man of them within reach
              (F / fighting_files less the front rank: 37 knights at 3 s, the recording 36-37 within 4.55 m, M38), their
              intervals from it (strike_first; melee.strikes, battle.py). Both off: the second wave above bit for bit.
The knock of a blow (knock(): a large striker's hits on a smaller man, the 'decided' blow, charge_module 5.1):
  P(down) = melee_on_damage_effect_knockdown_chance 0.05 x r x (1 + knock_chance_charging_bonus 2 x f) for r above
  interrupt_mass_ratio_clamp_knockdown 0.5 (r: the striker's mass / the struck man's, f: the striker's charge left), of
  a hit that does not kill; the men down lie the mean 1.85 s - the front's share down kd_p (a pool: in at the knocks a
  front man a second, out at 1 / 1.85 s). Small strikers never knock here (the lords' r 6 would: not switched on).
Infantry never interrupts a horse (r = 0.1 < interrupt_mass_ratio_clamp_hit_reaction 0.4): the simulator interrupts
nobody (melee.cycle off), nothing to do. Needs torch."""
import torch

from tools.nn.sim import geometry, melee

SIZE_K = {1: 0.8, 2: 0.2, 3: 0.1, 4: 0.0}     # the larger body's share of a push by the size difference (M3)


def on(params):
    return bool((params.sim["contact"].get("impact") or {}).get("on"))


def _cal(params):
    return params.sim["contact"].get("impact") or {}


def rise(params):
    """The getting-up animation's length, s, with contact.impact wave on (else 0): a man knocked down lies
    U(knockdown_getting_up_delay_min, _max) s and then gets up (state 0x11 'встаёт' till the animation's end) - no target
    for a blow or a swing all that time (build/diag_cav/mechanic.md M31-M32, the program). Its length is in the animations'
    data, not the program: MEASURED from the cavcharge recording (the first strong blows 2.4 s after the first knock-downs
    at 0.3 s: 2.4 - 0.3 - 1.0 = 1.1 s; M33)."""
    w = _cal(params).get("wave") or {}
    return float(w.get("rise_s", 0.0)) if w.get("on") else 0.0


def strike_first(params):
    """contact.impact strike_first (with ride_in and wave on; build/diag_cav/mechanic.md M36-M37, M39 1, 3, 4): the front
    rank of a large charger that rides in swings at the landing - its lance reaches the men before the bodies touch (the
    attack starts with the target nearer than r1 + r2 + 3 s x the closing speed, within 15 deg) and they still stand -,
    and its intervals run from that blow (one timer a man, not reset by a new target); only the ranks behind wait for the
    men to get up (their first blow then, their intervals from it)."""
    c = _cal(params)
    return bool(c.get("strike_first")) and bool(c.get("ride_in")) and bool((c.get("wave") or {}).get("on"))


def ridden(u, params):
    """[B, N] the front men a large charger rode in among are up and out of the brace (module doc, ride_in; False with it
    off): from the mean getting-up time after the landing (knockdown_getting_up_delay_min / _max: 1.85 s) - lying, a man
    keeps the brace he had (M26: a reaction freezes it)."""
    if not _cal(params).get("ride_in"):
        return torch.zeros_like(u["r"])
    R = params.battle
    up = (float(R["knockdown_getting_up_delay_min"]) + float(R["knockdown_getting_up_delay_max"])) / 2 + rise(params)
    return (u["ride"] >= 0.5) & (u["kd_t"] >= up)


def lane_place(u, params):
    """[B, N] the width of a charger's lane, m: its place across its front (h, the formation template's). With
    contact.impact lane_body (build/diag_skv_mon/sim_gap.md 5) a charger of size large and above (the monsters) runs into
    the men its BODY covers, not its place in the template: its width across (2 r x radii_ratio: the Brood Horror's 2.74 m,
    the Mutant's 4.0, a Rat Ogre's 2.0), at most its place; the cavalry (medium) keeps the template's (3.0 m against a
    horse's 2.8: bit for bit)."""
    h, _ = melee.spacing_of(u, params.sim["formation"]["spacing_m"])
    if not _cal(params).get("lane_body"):
        return h
    body = 2 * u["radius"] * (u["body_ratio"].clamp(0.05, 1.0) if "body_ratio" in u else 1.0)
    return torch.where(u["size_k"] >= 3, torch.minimum(body, h), h)


def per_rank(u, params):
    """[B, i, j] the men of a rank of j a charger i meets in its lane: its lane's width (lane_place) over j's file
    spacing, at least 1 - the Horror's template 18 m gave 12 Empire spearmen a rank (its body: 1.85), the Rat Ogres' 6.5 m
    4.4 a man (1.35)."""
    h, _ = melee.spacing_of(u, params.sim["formation"]["spacing_m"])
    return (lane_place(u, params)[:, :, None] / h[:, None, :]).clamp(min=1.0)


def ignore(u, params):
    """[B, N] a struck man's chance to stay on his feet when a larger body would knock him (the database's
    knock_interrupts_ignore_chance, STATIC knock_ign; the game's program: a roll 0-100 below it, build/diag_skv_mon/
    mechanic.md M5), for the targets contact.impact knock_ignore names: "large" - of size large and above (the monsters:
    the Brood Horror 0.6, the Mutant 0.65, Rat Ogres 0.2), "all" - every unit (the game's rule, on: the lords 0.4 / 0.75
    against the cavalry's and the monsters' knocks, infantry 0); None: off."""
    scope = _cal(params).get("knock_ignore")
    if not scope or "knock_ign" not in u:
        return None
    if scope == "large":
        return torch.where(u["size_k"] >= 3, u["knock_ign"], torch.zeros_like(u["knock_ign"]))
    return u["knock_ign"]


def bigger(u):
    """[B, N, N] i is a large body (size above small) and j smaller: the pairs these rules act on."""
    return u["large"][:, :, None] & (u["size_k"][:, :, None] > u["size_k"][:, None, :])


def size_factor(diff):
    """The larger body's share of a push by the size difference (0 -> 1)."""
    out = torch.ones_like(diff)
    for d, k in SIZE_K.items():
        out = torch.where(diff >= d, torch.full_like(out, k), out)
    return out


def ranks_of(u, spacing):
    """(files, ranks) [B, N] of each unit's formation (geometry.dims' rule: files = floor(width / h), at most the men)."""
    h, _ = melee.spacing_of(u, spacing)
    men = u["men"].clamp(min=1)
    files = torch.minimum(torch.floor(u["width"] / h + 1e-4).clamp(min=1), men)
    return files, torch.ceil(men / files)


def bracing(u, params, speed, charge_now):
    """B [B, N]: a man's mass multiplier while braced (module doc), 1 where not braced."""
    R = params.battle
    _, ranks = ranks_of(u, params.sim["formation"]["spacing_m"])
    n0 = float(R["bracing_calibration_ranks"])
    k = float(R["bracing_calibration_ranks_multiplier"])
    mult = 1 + (k - 1) * ((ranks - 1) / max(n0 - 1, 1e-6)).clamp(0, 1)
    defence = u["cdl"] | u["cd"]
    mult = torch.where(defence, mult * float(R["bracing_charge_reflector_bonus"]), mult)
    mult = mult.clamp(1.0, float(R["bracing_max_multiplier_clamp"]))
    can = u["brace_ok"] | defence | u["reflect"]
    braced = can & (speed < float(params.sim["melee"]["braced_speed"])) & ~u["r"] & (charge_now <= 0) & (u["men0"] > 1)
    braced = braced & ~ridden(u, params)                               # (ride_in: the front men up and walking)
    return torch.where(braced, mult, torch.ones_like(mult))


def deferred(u, params, contact, dt):
    """(pairs [B, N, N], share [B, N, N]) of the ride-in's delayed first blow (module doc): i large and bigger than j, in
    contact, j ridden into (ride) - the front rank's first blow of i does not land at the landing (the men it met lie: a
    lying man is no target for a new blow, M11) but as they get up, U(knockdown_getting_up_delay_min, _max) s after it:
    the share of that blow landing over this step [kd_t - dt, kd_t) (kd_t: since the landing, after this step's)."""
    R = params.battle
    lo, hi = float(R["knockdown_getting_up_delay_min"]), float(R["knockdown_getting_up_delay_max"])
    lo, hi = lo + rise(params), hi + rise(params)                    # (wave: no target till the animation's end too)
    pair = contact & bigger(u) & (u["ride"] >= 0.5)[:, None, :]
    t1 = u["kd_t"][:, None, :]
    share = ((t1.clamp(max=hi) - (t1 - dt).clamp(min=lo)).clamp(min=0) / max(hi - lo, 1e-6)).expand_as(pair.float())
    return pair, torch.where(pair, share, torch.zeros_like(share))


def lying(u, params):
    """[B, N] the share of a unit's front men lying now (the impact's kd_q by its clock, the blows' pool kd_p)."""
    R = params.battle
    lo, hi = float(R["knockdown_getting_up_delay_min"]), float(R["knockdown_getting_up_delay_max"])
    q = u["kd_q"] * ((hi + rise(params) - u["kd_t"]) / max(hi - lo, 1e-6)).clamp(0, 1)
    return (q + u["kd_p"] * (1 - q)).clamp(0, 1)


def up_share(u, params, rate):
    """rate [B, N, N] (i strikes j) with the lying men of i striking nobody: x (1 - lying_i) on its blows at a larger
    enemy (its front men facing that enemy are the ones knocked down)."""
    ly = lying(u, params)[:, :, None]
    on_big = bigger(u).transpose(1, 2)              # [B, i, j]: j large and bigger than i
    # (in place, x 1 where nothing lies: the rate keeps its memory layout, so the sums made of it later keep their order
    # and a battle without a large unit stays bit for bit - a new tensor from torch.where changed a bookkeeping sum's
    # last digit, gold_out)
    return rate.mul_(torch.where(on_big, 1 - ly, torch.ones_like(ly)))


def land(u, pw, contact, fire, params, dt, speed, charge_now, parts=False):
    """The impact of the large chargers whose charge lands this step (fire [B, N]) on the smaller enemies they touch
    (contact [B, N, N], i on j): sets the targets' kd_q / kd_t (their front share knocked down, the clock) and counts
    the men down in kd_n; returns hp [B, N, N] (the collision damage i deals j now); with parts (hp, men0, per0): also
    the men of j's first rank i meets and the collision HP each of them takes (lance_first). Module doc."""
    R = params.battle
    cal = _cal(params)
    e = float(R["collision_restitution_coefficient"])
    down_v = float(R["collision_speed_delta_threshold_knocked_down"])
    sp = params.sim["formation"]["spacing_m"]
    u["kd_t"] = u["kd_t"] + dt                                        # (a landing below sets it again)
    # (contact.impact collide_down; build/diag_cav/mechanic.md M35, the program) the collision damage lands on a man the push
    # knocks down or sends flying - his dv over collision_speed_delta_threshold_knocked_down 7 m/s with the closing speed
    # over the same 7 m/s (the speeds before the exchange: after it the closing speed is e v, 3.2 m/s at a horse's 10.8 -
    # no damage ever; which of the two the program takes is not read) -; a man who keeps his feet (knocked back, a step
    # back, a knock ignored) takes none, and a man lying, flying or getting up neither (no reaction, no damage)
    c_down = bool(cal.get("collide_down"))
    ly_j = lying(u, params)[:, None, :] if c_down else None
    pair = contact & fire[:, :, None] & bigger(u) & (u["men"] > 0)[:, None, :]
    # (no branch on the data: the step is compiled - torch.compile, battle.stepper; the masks do the gating)
    B_j = bracing(u, params, speed, charge_now)[:, None, :]
    B_own = B_j
    ahead = pw["rel_j"].abs() <= float(cal.get("brace_deg", 75.0)) * geometry.DEG
    B_j = torch.where(ahead, B_j, torch.ones_like(B_j))
    # (damage_brace_deg, module doc: M30) the collision damage divides by the man's brace within its own wider angle
    within = pw["rel_j"].abs() <= float(cal.get("damage_brace_deg", cal.get("brace_deg", 75.0))) * geometry.DEG
    mB_dmg = u["mass"][:, None, :] * torch.where(within, B_own, torch.ones_like(B_own))
    M = u["mass"][:, :, None]
    mB = u["mass"][:, None, :] * B_j
    vx = u["vx"][:, :, None] - u["vx"][:, None, :]
    vz = u["vz"][:, :, None] - u["vz"][:, None, :]
    v = torch.sqrt(vx * vx + vz * vz)
    gain = (1 + e) * M / (mB + M)                                     # a target man's dv a m/s of the difference
    k = size_factor(u["size_k"][:, :, None] - u["size_k"][:, None, :])
    loss = ((1 + e) * mB / (mB + M) * k).clamp(max=1.0)              # the charger's speed lost a man met, a share
    h, _ = melee.spacing_of(u, sp)
    c = per_rank(u, params)                                           # men of a rank in a charger's lane
    files, ranks = ranks_of(u, sp)
    length = torch.minimum(pw["face_i"], pw["face_j"])
    # the chargers in the contact's length: its lanes side by side (a monster's lane its body: 16 Rat Ogres on 30 m of
    # spearmen - all 6 files of their front, 8.1 men met in the first rank, not 20; the lone Horror's box 2.74 m - one)
    place = lane_place(u, params)
    horses = torch.minimum(torch.minimum(length / place[:, :, None], files[:, :, None]), u["men"][:, :, None])
    lane = torch.minimum(horses * c, files[:, None, :])               # target men met a rank
    half = torch.where(fire[:, None, :], torch.full_like(v, 0.5), torch.ones_like(v))
    big = u["size_k"][:, :, None] >= 3                                # large / very large: damage only on the down, x0.5
    nmax = float(R["collision_damage_maximum"])
    norm, mod = float(R["collision_damage_normaliser"]), float(R["collision_damage_modifier"])
    ap_r = float(R["collision_damage_armour_penetration_ratio"])
    hp = torch.zeros_like(v)
    down_men = torch.zeros_like(v)
    reach = pair.clone()                                              # the charger still rides on into this rank
    first_down = torch.zeros_like(pair)
    left_men = u["men"][:, None, :].expand_as(v)
    stay = ignore(u, params)
    # (contact.impact knock_ignore) the share of a target's men a down would knock that do fall: a large charger's
    # collision damage lands on them alone (a man who keeps his feet is not knocked: no damage either)
    falls = (1 - stay)[:, None, :] if stay is not None else torch.ones_like(v)
    for r in range(int(cal.get("max_ranks", 8))):
        met = reach & (r < ranks[:, None, :]) & (left_men > 0)
        dv = gain * v * half
        down = met & (dv > down_v)
        men_r = torch.minimum(lane, left_men)
        d = (torch.pow((v * M / mB_dmg.clamp(min=1e-6)).clamp(min=0), norm) * mod).clamp(max=nmax)
        d = torch.where(big, torch.where(down, 0.5 * d, torch.zeros_like(d)), d)
        if c_down:
            d = torch.where(down & (v > down_v), d, torch.zeros_like(d))
        per = melee.per_hit(d * (1 - ap_r), d * ap_r, u["armour"][:, None, :], u["hp_man"][:, None, :],
                            u["resist_physical"][:, None, :])
        per = torch.minimum(per, d.clamp(min=0))                      # (per_hit floors a blow at 1 HP)
        per = torch.where(big, per * falls, per)                      # (a large charger's: on the men that fall)
        if c_down:
            per = torch.where(big, per, per * falls) * (1 - ly_j)     # (collide_down: the men that fall, not lying)
        hp = hp + torch.where(met, men_r * per, torch.zeros_like(hp))
        down_men = down_men + torch.where(down, men_r, torch.zeros_like(down_men))
        if r == 0:
            first_down = down
            men0 = torch.where(met, men_r, torch.zeros_like(men_r))
            per0 = torch.where(met, per, torch.zeros_like(per))
        left_men = left_men - torch.where(met, men_r, torch.zeros_like(men_r))
        v = v * (1 - loss) ** c                                       # past the rank's men in its lane
        reach = down
    # the target's front share down: its first rank in the chargers' lanes (the lanes cover the contact's length: 1)
    q = first_down.to(v.dtype).amax(dim=1)
    if stay is not None:
        # (contact.impact knock_ignore) a man of the target stays on his feet with his body's knock_interrupts_ignore_chance
        q = q * (1 - stay)
        down_men = down_men * (1 - stay)[:, None, :]
    hit = pair.any(dim=1)
    if cal.get("ride_in"):
        # (module doc, the ride-in) the first rank down: the horses ride in among the men; no larger enemy touching: 0
        touched = (contact & bigger(u)).any(dim=1)
        u["ride"] = torch.where(touched, torch.where(hit, torch.maximum(u["ride"], q), u["ride"]), torch.zeros_like(q))
    u["kd_q"] = torch.where(hit, torch.maximum(q, lying(u, params)), u["kd_q"])
    u["kd_t"] = torch.where(hit, torch.zeros_like(u["kd_t"]), u["kd_t"])
    u["kd_n"] = u["kd_n"] + down_men.sum(1)
    return (hp, men0, per0) if parts else hp


def lance_first(u, hp, men0, per0, swing):
    """(contact.impact strike_first; build/diag_cav/mechanic.md M36) the collision HP hp [B, N, N] of a landing less what
    the men of the first rank that the charger's lances kill before the bodies touch would have taken: the attack starts
    with the target's centre nearer than r1 + r2 + 3 s x the closing speed (on a charge ~34 m) and the lance reaches the
    men at 3-4.5 m between centres, the bodies at 2.55 m - a man dead is no body to meet. swing [B, N, N]: the HP of the
    landing's first blow (the front rank's), its kills swing / the man's health, at most the first rank's men met (men0;
    a horse whose lane is emptied riding on into the next rank at full speed: not modelled)."""
    killed = torch.minimum(swing / u["hp_man"][:, None, :].clamp(min=1e-6), men0)
    return hp - killed * per0


def knock(u, params, dt, rate, swing, mhit, F, charge_now):
    """The knock of a large striker's blows (module doc): the front share of each smaller target lying from blows, kd_p,
    after this step's hits (rate [B, N, N] HP/s, swing [B, N, N] HP once, mhit the HP a hit, F the men striking)."""
    R = params.battle
    tau = (float(R["knockdown_getting_up_delay_min"]) + float(R["knockdown_getting_up_delay_max"])) / 2
    decay = torch.exp(torch.full_like(u["kd_p"], -dt / tau))
    big = bigger(u)
    r = u["mass"][:, :, None] / u["mass"][:, None, :].clamp(min=1e-6)
    p = float(R["melee_on_damage_effect_knockdown_chance"]) * r * (
        1 + float(R["melee_on_damage_effect_knock_chance_charging_bonus"]) * charge_now[:, :, None])
    p = torch.where(r > float(R["interrupt_mass_ratio_clamp_knockdown"]), p.clamp(max=1.0), torch.zeros_like(p))
    stay = ignore(u, params)
    if stay is not None:
        # (contact.impact knock_ignore) less the struck man's chance to stay on his feet
        p = p * (1 - stay)[:, None, :]
    kill = (mhit / u["hp_man"][:, None, :].clamp(min=1e-6)).clamp(0, 1)       # a hit's chance to kill
    hits = torch.where(mhit > 0, (rate * dt + swing) / mhit.clamp(min=1e-6), torch.zeros_like(rate))
    knocks = torch.where(big, hits * p * (1 - kill), torch.zeros_like(hits)).sum(1)               # [B, N] men down
    ff = melee.files_share(params.sim["melee"])
    front = torch.where(big.transpose(1, 2), F, torch.zeros_like(F)).sum(2) / ff                 # j's men at them
    front = torch.where(front > 0, front, torch.where(big, F, torch.zeros_like(F)).sum(1) / ff)  # (or theirs at j)
    add = (knocks / front.clamp(min=1.0)).clamp(0, 1)
    u["kd_p"] = (u["kd_p"] + add * (1 - u["kd_p"])) * decay
    u["kd_n"] = u["kd_n"] + knocks
