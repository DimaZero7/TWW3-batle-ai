"""A large body's blows on smaller men: the impact of its charge and the knock of its weapon (docs/ru/game/mechanics/
cavalry.md; the game's program: build/diag_cav/mechanic.md M1-M11, R-work charge_module.md 3 and 5). Every number is the
database's (config/nn/game_rules.json battle); config/nn/sim.json contact.impact switches it (impact_why). Only a unit of
size above small (the cavalry: medium) striking a smaller one: a battle without such a unit is unchanged bit for bit.

The impact (land(): a large charger's charge lands this step - battle.py `fire`; a new charge lands it again):
  bracing     B of a target man: a unit that can brace (the database's can_brace, or charge_defense(_vs_large),
              charge_reflection) standing (slower than melee.braced_speed), not routing and not in its own charge:
              1 + (bracing_calibration_ranks_multiplier - 1) clamp((ranks - 1) / (bracing_calibration_ranks - 1), 0, 1)
              (1 + 3 clamp((ranks - 1) / 7): 3.14 for 6 ranks), x bracing_charge_reflector_bonus 2 with charge defence,
              within 1 .. bracing_max_multiplier_clamp 10; the blow from within impact.brace_deg (75) of its facing; else 1
  speed       a man the charger meets: dv = (1 + e) M / (m B + M) v (e: collision_restitution_coefficient 0.3; M, m the
              masses, v the pair's speed difference); the charger loses (1 + e) m B / (m B + M) v x k(the size
              difference: 1 -> 0.8, 2 -> 0.2, 3 -> 0.1, 4 -> 0) on each man it meets
  ranks       a charger meets h_charger / h_target men of a rank (its file's lane in the formation templates); where they go
              down (dv over collision_speed_delta_threshold_knocked_down 7 m/s; the flight past 10.5 is a knock-down too) it
              rides on into the next rank at its speed less its loss a man met, and stops at the first rank that stays up
              (a target man landing his own charge now: his dv halved - the program's dv^2 x 0.25)
  damage      each man met takes min(collision_damage_maximum 70, (v M / (m B))^collision_damage_normaliser 0.8 x
              collision_damage_modifier 0.6), collision_damage_armour_penetration_ratio 0.7 of it past the armour, once a
              landing (the 4 s immunity per attacker); a charger of size above medium only on men it knocks down, x0.5
  lying       the men down lie U(knockdown_getting_up_delay_min 1.0, _max 2.7) s: the target's front share down at t s
              after the landing is down_1 x clamp((2.7 - t) / 1.7, 0, 1) (kd_q, kd_t); a man lying strikes nobody
              (up_share), the blows on him land (M11: the swing strikes him)
  the chain   a man knocked down hits the man behind at his flight speed, at most 3.3 m/s (M8): (1 + e) m / (m B + m)
              x 3.3 <= 2.2 m/s, below the step back (3) - no chain in numbers; the horse riding on is the ranks above
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
    return torch.where(braced, mult, torch.ones_like(mult))


def lying(u, params):
    """[B, N] the share of a unit's front men lying now (the impact's kd_q by its clock, the blows' pool kd_p)."""
    R = params.battle
    lo, hi = float(R["knockdown_getting_up_delay_min"]), float(R["knockdown_getting_up_delay_max"])
    q = u["kd_q"] * ((hi - u["kd_t"]) / max(hi - lo, 1e-6)).clamp(0, 1)
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


def land(u, pw, contact, fire, params, dt, speed, charge_now):
    """The impact of the large chargers whose charge lands this step (fire [B, N]) on the smaller enemies they touch
    (contact [B, N, N], i on j): sets the targets' kd_q / kd_t (their front share knocked down, the clock) and counts
    the men down in kd_n; returns hp [B, N, N] (the collision damage i deals j now). Module doc."""
    R = params.battle
    cal = _cal(params)
    e = float(R["collision_restitution_coefficient"])
    down_v = float(R["collision_speed_delta_threshold_knocked_down"])
    sp = params.sim["formation"]["spacing_m"]
    u["kd_t"] = u["kd_t"] + dt                                        # (a landing below sets it again)
    pair = contact & fire[:, :, None] & bigger(u) & (u["men"] > 0)[:, None, :]
    # (no branch on the data: the step is compiled - torch.compile, battle.stepper; the masks do the gating)
    B_j = bracing(u, params, speed, charge_now)[:, None, :]
    ahead = pw["rel_j"].abs() <= float(cal.get("brace_deg", 75.0)) * geometry.DEG
    B_j = torch.where(ahead, B_j, torch.ones_like(B_j))
    M = u["mass"][:, :, None]
    mB = u["mass"][:, None, :] * B_j
    vx = u["vx"][:, :, None] - u["vx"][:, None, :]
    vz = u["vz"][:, :, None] - u["vz"][:, None, :]
    v = torch.sqrt(vx * vx + vz * vz)
    gain = (1 + e) * M / (mB + M)                                     # a target man's dv a m/s of the difference
    k = size_factor(u["size_k"][:, :, None] - u["size_k"][:, None, :])
    loss = ((1 + e) * mB / (mB + M) * k).clamp(max=1.0)              # the charger's speed lost a man met, a share
    h, _ = melee.spacing_of(u, sp)
    c = (h[:, :, None] / h[:, None, :]).clamp(min=1.0)                # men of a rank in a charger's lane
    files, ranks = ranks_of(u, sp)
    length = torch.minimum(pw["face_i"], pw["face_j"])
    horses = torch.minimum(torch.minimum(length / h[:, :, None], files[:, :, None]), u["men"][:, :, None])
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
    for r in range(int(cal.get("max_ranks", 8))):
        met = reach & (r < ranks[:, None, :]) & (left_men > 0)
        dv = gain * v * half
        down = met & (dv > down_v)
        men_r = torch.minimum(lane, left_men)
        d = (torch.pow((v * M / mB.clamp(min=1e-6)).clamp(min=0), norm) * mod).clamp(max=nmax)
        d = torch.where(big, torch.where(down, 0.5 * d, torch.zeros_like(d)), d)
        per = melee.per_hit(d * (1 - ap_r), d * ap_r, u["armour"][:, None, :], u["hp_man"][:, None, :],
                            u["resist_physical"][:, None, :])
        per = torch.minimum(per, d.clamp(min=0))                      # (per_hit floors a blow at 1 HP)
        hp = hp + torch.where(met, men_r * per, torch.zeros_like(hp))
        down_men = down_men + torch.where(down, men_r, torch.zeros_like(down_men))
        if r == 0:
            first_down = down
        left_men = left_men - torch.where(met, men_r, torch.zeros_like(men_r))
        v = v * (1 - loss) ** c                                       # past the rank's men in its lane
        reach = down
    # the target's front share down: its first rank in the chargers' lanes (the lanes cover the contact's length: 1)
    q = first_down.to(v.dtype).amax(dim=1)
    hit = pair.any(dim=1)
    u["kd_q"] = torch.where(hit, torch.maximum(q, lying(u, params)), u["kd_q"])
    u["kd_t"] = torch.where(hit, torch.zeros_like(u["kd_t"]), u["kd_t"])
    u["kd_n"] = u["kd_n"] + down_men.sum(1)
    return hp


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
    kill = (mhit / u["hp_man"][:, None, :].clamp(min=1e-6)).clamp(0, 1)       # a hit's chance to kill
    hits = torch.where(mhit > 0, (rate * dt + swing) / mhit.clamp(min=1e-6), torch.zeros_like(rate))
    knocks = torch.where(big, hits * p * (1 - kill), torch.zeros_like(hits)).sum(1)               # [B, N] men down
    ff = melee.files_share(params.sim["melee"])
    front = torch.where(big.transpose(1, 2), F, torch.zeros_like(F)).sum(2) / ff                 # j's men at them
    front = torch.where(front > 0, front, torch.where(big, F, torch.zeros_like(F)).sum(1) / ff)  # (or theirs at j)
    add = (knocks / front.clamp(min=1.0)).clamp(0, 1)
    u["kd_p"] = (u["kd_p"] + add * (1 - u["kd_p"])) * decay
    u["kd_n"] = u["kd_n"] + knocks
