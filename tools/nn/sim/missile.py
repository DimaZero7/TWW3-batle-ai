"""Shooting (docs/en/training/simulator.md).

A missile unit shoots when it stands still, is not in melee, has projectiles and an enemy within
range, centre to centre (the scripted range: the first arrow with the archers' middle 129-130 m from the target's
centre for blocks 6-47 m deep, and then all their men shoot, the rear ranks beyond range too; config/nn/sim.json
missile.range_why). It first aims for aim_s seconds after halting (measured 3.3 s arrows, 4.3 s sling); then each
man shoots once per reload (the database's 10 s archers, measured on the range; slings 11.5 s measured against 9).
A man shoots only at a target centre within the unit's fire arc seen from where he stands (arc_share: the database's
battle_entities fire arc, +-30 deg, militia +-35): a wide line facing a target off to one side fires with its near
flank only. A standing unit keeps its facing and turns to its target only beyond missile.stand_fire_arc_deg (45,
measured) until within missile.turn_done_deg (battle.py). It keeps its target while it stays in range and stands;
a new target (an order's or its own) costs missile.retarget_s (3 s) without fire (measured; missile.retarget_why).
Volleys: the men reload all the time (moving too) and the unit shoots once missile.volley_load of them
(1: all) are loaded, so it fires whole-unit volleys one reload apart (measured: archers on the range,
build/archer-range/runs 28.09.2026, ~80 of 90 arrows within 1 s of the first, then ~10 s nothing; in
battle a halt gives about one volley in the next 12 s, not a volley and then the steady rate at once).
A new order (another kind or another attack target) makes it aim again (missile.aim_reset_on_order); by category
(missile.order_quiet_why): arcing fire (arrows, bolts, slings) already aiming then keeps 8.5 s without a shot whatever
its reload, each new order or target starting it again; direct fire and throwing weapons lose nothing.

    hits   = shots x hit chance (hit_chance: the database's spread and the target's men; units fall back to
             hit_rate x distance factor with missile.accuracy.units "rates"); aimed
             at a unit in melee, a measured share lands on the shooter's own units in contact
             with it (friendly fire: 0.26 arrows, 0.56 sling); and a share by distance
             lands on the target's neighbours out of melee (spill: 0.115 within 30 m,
             0.034 at 30-60 m, 0.015 at 60-90 m)
    per hit = ap + base x (1 - 0.75 armour / 100); a shield blocks its chance of small arms from the front
              (within shield_defence_angle_missile, 60 deg); x (1 - missile resistance)

The hit chance (hit_chance, missile.accuracy.plane; build/open_missile/spec.md S1): a projectile aimed at the middle
of a random man of the target lands at a Gaussian offset in the plane across the line of fire at the target (a
calibration target, like a range's), sigma^2 = calibration_area x (d / calibration_distance)^2 x (1 - accuracy / 100):
the database's calibration area is an AREA in m^2 (modders: "valuated in square meters", Empire's setting "area of
calibration target in square metres") and accuracy (land_units accuracy + the projectile's marksmanship) cuts that area
by its percent - no fitted number. It comes down at the low arc's angle theta (muzzle velocity): on the ground the
spread along the line is sigma_a = sigma / sin(theta) and the aim point (a man's middle, height h / 2) lies
c = h / (2 tan theta) beyond his feet; a man of height h is hit within his radius r or in his shadow L = h / tan(theta).
    a lone man   p = erf(r / (sqrt2 sigma)) x [Phi((L + r - c) / sigma_a) - Phi((-r - c) / sigma_a)]
    a formation  p = max(cover, erf(r / (sqrt2 sigma))) x mean over its ranks y0 of
                     [Phi((D + L - c - y0) / sigma_a) - Phi((-r - c - y0) / sigma_a)]
                 files s_h apart across the line and ranks s_v apart along it, D = (ranks - 1) s_v: the target's own
                 formation now (geometry.dims; the database's close spacing), so a loose or thinned unit is hit less;
                 the men are not in exact files: a projectile passing at head height over n = clamp((min(L, D) + 2r) /
                 s_v, 1, ranks) ranks meets a man with cover = 1 - (1 - min(2r / s_h, 1))^n.
The spread's shape (missile.accuracy.spread, now "square"; the probe shots, build/shots): not a Gaussian but a uniform
square in that plane, side^2 = calibration_area / p, p = accuracy / 100 x (calibration_distance / d)^2 x (1 - launch
angle / 180 deg) within [0.01, 1] (square_side): the same d^2 growth far, but nearer than calibration_distance x
sqrt(accuracy / 100) the spread stays the calibration area (crossbows at a lone man 30 m off: the game 0.35 hits a shot,
the Gaussian 0.68, the square 0.40). The formulas above with Phi / erf replaced by the uniform's shares (_share: the
offset uniform +-side / 2 across, +-side / (2 sin theta) along the ground); the spill and the friends' catch keep a
Gaussian of the same variance (side / sqrt 12). "gauss": the Gaussian sigma above.
(missile.accuracy.plane false: the older ground model, sigma = k x calibration_area x d / calibration_distance x
sqrt(1 - accuracy / 100) with a fitted k, aimed at the feet, exact files.)
A lone man in melee keeps hit_rate x distance factor x single_entity_in_melee (measured 0: his share is lost, sim.json single_in_melee_why); spill and
friendly fire are measured shares of those old-rule hits.

Target: the ATTACK order's target if in range, else the nearest standing enemy in range, else
the nearest routing one (fire at will; into melee too: allow_fire_at_will_into_melee 1). On the move
(fire whilst moving) only targets within missile.move_fire_arc_deg of the facing (battle.py).

Line of fire (direct fire only: projectiles with trajectory `low`, the passport's missile.direct;
arrows and slings arc over friends): the shooter's men aim at the target's centre; a friendly unit
between them (nearer than the target's edge) blocks the lines that pass through its width across
the line, widened by the friendly man radius coefficient (projectile_friendly_fire_man_radius_
coefficient: vanilla 2.2, 1.0 with the required mod True Sight - friends at their own size). Blocked men
do not shoot; at unit_firing_line_of_sight_considered_obstructed_ratio (vanilla 0.75, True Sight 1.0: only a
wholly blocked unit) the unit holds fire at that target and takes the next one in range (an ordered target too:
the bridge releases a blocked shooter to fire at will), or holds fire. (config/nn/game_rules.json "_mods".) Enemies in the way do not block (the game's clear-shot
test is about friends). Flat map: no fire over friends from higher ground.
With missile.arc_los (07.10.2026) the line is the bullet's arc (arc_lines): a friend in the way blocks only the lines
whose arc does not clear its men's heads (a flat shot keeps its speed, so a far target's arc rises higher - the probes:
a target ~70 m behind friends 40 m ahead stops the fire, ~85-90 m does not); the shots of the clearing lines that pass
below the friends' heads by the spread hit them (friend_catch; the game: 0.22-0.27 hits a shot of a covered unit).

Friends in the way (missile.friend_los, the game's rules; it replaces the line of fire above - the obstructed ratio
then decides nothing - and the catch stays arc_lines'): friend_rays, friend_lines, clear_shot.
- The unit's line: three rays from its front (ends and middle) to the target along the projectile's low arc; the line
  is clear when 2 of 3 are. Closed: direct fire (bullets) does not shoot at that target (the next one in range, else it
  holds); arcing fire (arrows, bolts, slings) shoots at 45 deg if the target is beyond lob_min_m (10 m), else as direct.
- Each man, after every reload: his path (the low arc to 0.75 of the target man's height, or the 45 deg arc when
  lobbing) blocked by a man of another own unit costs him the shot - he loses his target, takes the same one again
  and tries at the next reload; the friends stand still, so the same men stay blocked volley after volley. The unit's
  own men never block (every rank shoots) but for a unit in melee with its target. Enemies never block.
- A path is blocked by a friend's man (a cylinder: radius and height x the mod's 1.0) it passes below the head of. A
  friend's men stand in files and ranks: a man's path lies in a file's band (2 r of the file spacing) or between two
  files - then it passes every rank of that friend - and it is blocked when the band holds a man of the friend where
  the arc is below the heads (the share of the rank spacing that window covers); a body that holds the muzzle or
  stands within its radius of it horizontally (beside, behind) blocks at any arc. The unit's line ignores a friend
  whose formation is inside its own (the circumscribed circles intersect and the turned rectangles touch): no lob for
  it, only the men's paths. (The game's path is 16 m chords along the arc; ours the exact parabola.)
- Hits on friends take the full damage (can_damage_allies 1 for all our shooters).
"""
import math

import torch

from tools.nn.sim import geometry, melee

FAR = 1e9


def distance_factor(dist, table):
    """Piecewise-linear factor over centre distance (config/nn/sim.json missile.distance_factor)."""
    d = torch.tensor([p[0] for p in table], dtype=dist.dtype, device=dist.device)
    v = torch.tensor([p[1] for p in table], dtype=dist.dtype, device=dist.device)
    x = dist.clamp(min=float(table[0][0]), max=float(table[-1][0]))
    idx = torch.searchsorted(d, x.contiguous()).clamp(1, len(table) - 1)
    d0, d1 = d[idx - 1], d[idx]
    v0, v1 = v[idx - 1], v[idx]
    w = (x - d0) / (d1 - d0)
    return v0 + w * (v1 - v0)


def reach_of(params):
    """(reach_m, rate): missile.fire_move_reach_m / fire_move_reach_rate (config/nn/sim.json fire_move_reach_why);
    (0, 0) without them."""
    ms = params.sim["missile"]
    return float(ms.get("fire_move_reach_m") or 0.0), float(ms.get("fire_move_reach_rate") or 0.0)


def rank_share(u, pw, reach=0.0, rate=0.0):
    """[B, i, j] share of direct-fire shooter i's men that fire at target j (missile.per_man_range_direct; the probe
    build/probes7 P2, militia with pistols, range 90, 7 ranks 1.7 m apart, on skavenslaves 10 ranks deep):
    - the whole unit fires (the IsFiringMissiles flag on) while j's centre is within range of i's FRONT RANK (i's centre
      + (ranks - 1) / 2 x v_i ahead): full volleys at 85 / 88 / 90 m centre to centre and 0.94-0.99 at 94.5 m (front
      rank 89.4 m from the target centre), the flag off at 97.6 m (92.5 m);
    - beyond that only the ranks within range of j's nearest men fire (the flag off): rank r (0 the front) stands
      g0 + r v_i from them, g0 = the edge gap + v_i / 2 + v_j / 2 (the men stand half a spacing inside the edges);
      at 97.6 m 4 of 7 ranks = 0.57, the game a steady stream at 0.54 of the full rate for 200 s.
    A lone shooter or target: no ranks (spacing 0).
    reach, rate (missile.fire_move_reach_m / _rate, reach_of): beyond that, at least `rate` of the men fire while j's
    centre is within range + reach of i's centre (the units that fire whilst moving, per_man_mask)."""
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))
    lone = u["men0"] <= 1
    vi = torch.where(lone, torch.zeros_like(v), v)
    ranks = torch.where(lone, torch.ones_like(v), (pw["depth"] / v).round().clamp(min=1))
    front_off = ((ranks - 1) / 2 * vi)[:, :, None]
    rng = u["range"][:, :, None]
    whole = pw["dist"] - front_off <= rng
    g0 = pw["gap"] + vi[:, :, None] / 2 + vi[:, None, :] / 2
    n = torch.floor((rng - g0) / vi.clamp(min=1e-3)[:, :, None]) + 1
    n = torch.where(lone[:, :, None], (g0 <= rng).float(), n)
    part = (n.clamp(min=0) / ranks[:, :, None]).clamp(0, 1)
    if reach > 0:
        far = pw["dist"] <= rng + reach
        part = torch.where(far, part.clamp(min=rate), part)
    return torch.where(whole, torch.ones_like(part), part)


def per_man_mask(u, params):
    """[B, N] the shooters whose range counts per man (rank_share): direct fire (missile.per_man_range_direct), and
    with missile.per_man_range_fire_move_only only the units that fire whilst moving (the militia's pistols, the
    throwing stars); the others (handgunners) by the centre rule like arrows. None: nobody."""
    ms = params.sim["missile"]
    if not ms.get("per_man_range_direct"):
        return None
    mask = u["direct"]
    if ms.get("per_man_range_fire_move_only"):
        mask = mask & u["fire_move"]
    return mask


def in_range(u, pw, per_man=False, reach=(0.0, 0.0)):
    """[B, i, j] j within i's range: centre to centre; with per_man (True: every direct-fire shooter; a [B, N] mask:
    those shooters, per_man_mask) a shooter's men each to the target's nearest men (rank_share > 0)."""
    centre = pw["dist"] <= u["range"][:, :, None]
    if per_man is None or per_man is False:
        return centre & min_range_ok(u, pw)
    mask = u["direct"] if per_man is True else per_man
    return torch.where(mask[:, :, None], rank_share(u, pw, *reach) > 0, centre) & min_range_ok(u, pw)


def min_range_ok(u, pw):
    """[B, i, j] j is not nearer than i's minimum range (the projectile's minimum_range, given only to artillery:
    artillery.py), edge to edge (the formations' gap; R-work artillery_rules.md 6); True without one."""
    if "min_range" not in u:
        return torch.ones_like(pw["enemy"])
    m = u["min_range"][:, :, None]
    return (m <= 0) | (pw["gap"] >= m)


def router_share(age, ages, shares):
    """The measured share of shooters still on a target `age` s [B, N] after it began to rout (missile.router_keep_*):
    1 at 0, linear between the nodes (ages, shares), the last share beyond the last node."""
    xs = [0.0] + [float(a) for a in ages]
    ys = [1.0] + [float(s) for s in shares]
    out = torch.full_like(age, ys[-1])
    for k in range(len(xs) - 1, 0, -1):
        f = ((age - xs[k - 1]) / max(xs[k] - xs[k - 1], 1e-9)).clamp(0, 1)
        out = torch.where(age <= xs[k], ys[k - 1] + (ys[k] - ys[k - 1]) * f, out)
    return out


def router_keep(u, prev, params, t, dt):
    """[B, N] shooters that keep last step's target this step though it routs (missile.router_keep_s / _share; None:
    off): free shooters stay on a target that began to rout as the game's engine does, by the measured share still on
    it `a` s into its rout - a step from a to a + dt stays with S(a + dt) / S(a) (an hash draw per battle, unit and
    step, as morale's, so nothing random to replay). Routing and shattered targets alike (measured the same)."""
    cal = params.sim["missile"]
    ages, shares = cal.get("router_keep_s"), cal.get("router_keep_share")
    if prev is None or not ages:
        return None
    from tools.nn.sim import morale
    p = prev.clamp(min=0)
    on_router = (prev >= 0) & u["r"].gather(1, p)
    a = u["rout_s"].gather(1, p).clamp(min=0)
    s_now = router_share(a, ages, shares)
    stay = torch.where(s_now > 1e-9, router_share(a + dt, ages, shares) / s_now.clamp(min=1e-9),
                       torch.zeros_like(s_now))
    B, N = prev.shape
    k = torch.round(t / dt).long() if isinstance(t, torch.Tensor) else torch.full((B,), round(float(t) / dt),
                                                                                     dtype=torch.long, device=prev.device)
    draw = morale._hash_uniform(B, N, k, prev.device, salt=40357, ids=morale.battle_ids(u))
    return on_router & (draw < stay)


def hit_slow_lay(u, params, dt, hp_missile, shit, enemy, t):
    """The slowing from hits (missile.hit_slow; config/nn/sim.json hit_slow.why), in place: the timer hs_s runs down
    by dt, then starts at hit_slow.duration_s on every unit hit this step by an enemy whose projectiles slow (STATIC
    hslow: the database's projectile category in hit_slow.categories). hp_missile [B, i, j]: HP i's projectiles took
    from j this step, shit [B, i, j] the HP a hit, enemy [B, i, j]. Hits land as Poisson counts of the expected ones:
    at least one with 1 - exp(-hits), a hash draw per battle, unit and step (morale._hash_uniform), as effects.lay."""
    hs = params.sim["missile"].get("hit_slow") or {}
    if not hs.get("on") or "hs_s" not in u:
        return
    from tools.nn.sim import morale
    B, N = u["men"].shape
    k = torch.round(t / dt).long() if isinstance(t, torch.Tensor) else torch.full(
        (B,), round(float(t) / dt), dtype=torch.long, device=u["men"].device)
    n = torch.where(shit > 0, hp_missile / shit.clamp(min=1e-6), torch.zeros_like(hp_missile)) * enemy
    hits = (u["hslow"].float()[:, :, None] * n).sum(1)
    draw = morale._hash_uniform(B, N, k, u["men"].device, salt=61417, ids=morale.battle_ids(u))
    hit = (hits > 0) & (draw < 1 - torch.exp(-hits))
    left = (u["hs_s"] - dt).clamp(min=0)
    u["hs_s"] = torch.where(hit, torch.full_like(left, float(hs["duration_s"])), left)


def hit_slow_factor(u, params, fast):
    """[B, N] the share of its speed a unit keeps under the slowing from hits (missile.hit_slow; 1: none): hs_s > 0 on
    a small formation (men0 > 1, not large: the measured clanrats; lords ignore most hit reactions - the database's
    battle_entities hit_reactions_ignore_chance 70-75 %, infantry 0) - hit_slow.run while it runs (fast [B, N]), else
    hit_slow.walk."""
    hs = params.sim["missile"].get("hit_slow") or {}
    one = torch.ones_like(u["hp"])
    if not hs.get("on") or "hs_s" not in u:
        return one
    on = (u["hs_s"] > 0) & (u["men0"] > 1) & ~u["large"]
    k = torch.where(fast, torch.full_like(one, float(hs["run"])), torch.full_like(one, float(hs["walk"])))
    return torch.where(on, k, one)


def choose_target(u, pw, can_shoot, order_target, order_attack, exclude=None, prev=None, per_man=False,
                  reach=(0.0, 0.0), keep_router=None):
    """[B, N] slot each shooter aims at (-1 none). exclude [B, N, N]: targets i may not take. prev [B, N]: last
    step's target, kept while it is in range (centre to centre; per_man: in_range), standing and not excluded (the
    game keeps its target; without it the nearest-each-step flips between two close enemies). keep_router [B, N]
    (router_keep): shooters that keep prev though it routs (still in range and not excluded)."""
    # (a hidden enemy is no target: stealth.py)
    alive = (u["men"] > 0) & ~u["gone"] & (~u["hid"] if "hid" in u else True)
    rng = pw["enemy"] & alive[:, None, :] & in_range(u, pw, per_man, reach)
    if exclude is not None:
        rng = rng & ~exclude
    standing = rng & ~u["r"][:, None, :]
    score = torch.where(standing, pw["dist"], torch.where(rng, pw["dist"] + 1e5, torch.full_like(pw["dist"], FAR)))
    nearest = score.argmin(dim=2)
    has = score.min(dim=2).values < FAR
    if prev is not None:
        p = prev.clamp(min=0)
        keep = (prev >= 0) & standing.gather(2, p[:, :, None]).squeeze(2)
        if keep_router is not None:
            keep = keep | ((prev >= 0) & keep_router & rng.gather(2, p[:, :, None]).squeeze(2))
        nearest = torch.where(keep, p, nearest)
        has = has | keep
    ordered = order_attack & (order_target >= 0)
    ot = order_target.clamp(min=0)
    ordered_ok = ordered & rng.gather(2, ot[:, :, None]).squeeze(2)
    target = torch.where(ordered_ok, ot, torch.where(has, nearest, torch.full_like(nearest, -1)))
    return torch.where(can_shoot, target, torch.full_like(target, -1))


def arc_share(u, pw, target):
    """[B, N] share of shooter i's men with a man of its target (slot, -1: 0) within their fire arc. The men stand
    along the unit's front W (geometry.dims); the target's centre lies `lat` across and `fwd` ahead of the unit's
    centre and its front spans e = front / 2 x |cos(its bearing - the shooter's)| across (a lone man: 0); a man at s
    across fires if some of it is within fwd x tan(arc) of him: share = |[lat - e - R, lat + e + R] n [-W/2, W/2]| / W,
    R = fwd x tan(arc) (0 behind the unit). A lone shooter: 1 within the arc of his facing. An arc of 90 deg or more:
    every man. (The probe, build/missile: a wide target 35 deg off was shot by 0.93 of the archers, a narrow one
    by 0.42; the centre alone gave 0.49 and 0.44.)"""
    t = target.clamp(min=0)[:, :, None]
    rel = pw["rel_i"].gather(2, t).squeeze(2)
    dist = pw["dist"].gather(2, t).squeeze(2)
    arc = u["arc"].clamp(max=89.0) * geometry.DEG
    W = pw["front"].clamp(min=1e-3)
    front_t = torch.where(u["men0"] <= 1, torch.zeros_like(W), pw["front"]).gather(1, target.clamp(min=0))
    b_t = u["b"].gather(1, target.clamp(min=0))
    e = front_t / 2 * torch.cos((b_t - u["b"]) * geometry.DEG).abs()
    lat, fwd = dist * torch.sin(rel), dist * torch.cos(rel)
    reach = fwd.clamp(min=0) * torch.tan(arc) + e
    lo = torch.maximum(lat - reach, -W / 2)
    hi = torch.minimum(lat + reach, W / 2)
    share = torch.where(fwd > 0, (hi - lo).clamp(min=0) / W, torch.zeros_like(W))
    share = torch.where(u["men0"] <= 1, (rel.abs() <= arc).float(), share)
    share = torch.where(u["arc"] >= 90, torch.ones_like(share), share)
    return torch.where(target >= 0, share.clamp(0, 1), torch.zeros_like(share))


def _phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _G(z):
    """The integral of the normal CDF: d/dz G = Phi(z) (G(z) = z Phi(z) + phi(z))."""
    return z * _phi(z) + torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def square_spread(params):
    """missile.accuracy.spread "square" (the probe shots, build/shots): the spread a uniform square in the target's plane,
    not a Gaussian (square_side); "gauss" or absent: the Gaussian sigma (acc_shrink)."""
    return params is not None and params.sim["missile"]["accuracy"].get("spread") == "square"


def aim_frac(params):
    """missile.accuracy.aim_frac: the aim point's height on the man, a share of his height (0.5 or absent: his middle)."""
    return 0.5 if params is None else float(params.sim["missile"]["accuracy"].get("aim_frac", 0.5))


def square_side(u, d, theta):
    """[B, i, j] the side, m, of shooter i's square spread at distance d (missile.accuracy.spread "square"): the
    calibration AREA over p, p = accuracy / 100 x (calibration distance / d)^2 x (1 - launch angle / 180 deg) within
    [0.01, 1] - the area grows as d^2 past the calibration distance x sqrt(accuracy / 100) and stays the calibration
    area nearer (the probe shots: a lone Warlock Engineer ~30 m off, crossbows 0.37 hits a shot, the Gaussian 0.83)."""
    p = (u["acc"][:, :, None] / 100 * (u["cal_dist"][:, :, None] / d) ** 2 * (1 - theta / math.pi)).clamp(0.01, 1.0)
    return torch.sqrt(u["cal_area"].clamp(min=0)[:, :, None] / p).clamp(min=1e-3)


def acc_theta(theta, v, d, dh):
    """The launch angle the accuracy's (1 - angle / 180 deg) takes (square_side): on the flat theta (the low arc's, as
    given); with the ground's height (terrain.py, pw["dh"]: the target's ground less the shooter's) the low arc's that
    puts the shot dh higher at d (_arc_tan; mechanic.md 3: the game solves its ballistics with the real height - a
    target below is shot flatter, more accurately). The fall's geometry (the shadow, the spread along the ground) keeps
    the flat arc's angle."""
    if dh is None:
        return theta
    return torch.atan(_arc_tan(v, d, dh)).clamp(min=1e-3)


def _share(lo, hi, m, w):
    """The share of a uniform [m - w, m + w] within [lo, hi]."""
    return ((torch.minimum(hi, m + w) - torch.maximum(lo, m - w)) / (2 * w)).clamp(0, 1)


def _spread(u, pw_d, theta, params):
    """(sigma, half): the Gaussian sigma across the line of fire at distance pw_d, half None; with the square spread
    sigma its equal-variance side / sqrt 12 (for the spill and the friends' catch, which stay Gaussian) and half the
    square's half side."""
    if square_spread(params):
        side = square_side(u, pw_d, theta)
        return side / math.sqrt(12.0), side / 2
    sig = (torch.sqrt(u["cal_area"].clamp(min=0))[:, :, None] * pw_d / u["cal_dist"].clamp(min=1.0)[:, :, None]
           * acc_shrink(u, params)).clamp(min=1e-3)
    return sig, None


def width_capture(u, pw, sig, rx, params, half=None):
    """[B, i, j] the share of i's shots at formation j that pass across within its span (missile.accuracy.width): the aim
    is a random man of j, his centre uniform over the span the shooter sees (front |cos phi| + depth |sin phi|, phi off
    j's facing: a column seen end on is narrow, the same men side on wide) less one man's spacing, the lateral offset
    Gaussian sigma; within the span +- a body's half-width the shot meets the men's cover, outside it nobody. The span
    a half-width a: (1 / 2a) int_{-a}^{a} [Phi((a + rx - x) / s) - Phi((-a - rx - x) / s)] dx in closed form; a -> 0 is
    the lone man's erf(rx / (sqrt 2 s)). None with the switch off (the span taken as endless, the old reading)."""
    if params is None or not params.sim["missile"]["accuracy"].get("width"):
        return None
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    seen = (pw["front"][:, None, :] * torch.cos(pw["rel_j"]).abs()
            + pw["depth"][:, None, :] * torch.sin(pw["rel_j"]).abs())
    a = ((seen - h) / 2).clamp(min=1e-3)
    if half is not None:
        # the square spread: the lateral offset uniform +-half; the aim's centre over the span by 16 midpoints
        A = a + rx
        F = torch.zeros_like(A)
        for q in range(16):
            x = a * (2 * (q + 0.5) / 16 - 1)
            F = F + _share(-A, A, x, half) / 16
        return torch.maximum(F, (rx / half).clamp(max=1.0)).clamp(max=1.0)
    zo, zi = (2 * a + rx) / sig, rx / sig
    F = sig / (2 * a) * (_G(zo) - _G(zi) - _G(-zi) + _G(-zo))
    lone = torch.erf(rx / (math.sqrt(2) * sig))
    return torch.maximum(F, lone).clamp(max=1.0)


def acc_shrink(u, params):
    """[B, i, 1] how shooter i's accuracy (land_units accuracy + the projectile's marksmanship, a) narrows its spread
    sigma (missile.accuracy.shrink): "radius" - sigma x (1 - a / 100), the spread's radius cut by the percent (the
    lonerank probe: a lone Warlock Engineer under handguns and crossbows at rank 0 and rank 9, +27 accuracy - hits
    x1.75 / x1.87, the radius reading x1.92 / x2.04, the area reading x1.43 / x1.46); otherwise "area" - sigma x
    sqrt(1 - a / 100), the area cut (the modders' wording; config/nn/sim.json missile.accuracy_why)."""
    a = (1 - u["acc"] / 100).clamp(min=0)
    radius = params is not None and params.sim["missile"]["accuracy"].get("shrink") == "radius"
    return (a if radius else torch.sqrt(a))[:, :, None]


def hit_chance(u, pw, params):
    """[B, i, j] chance that a projectile of i aimed at j hits a man of j (module docstring: the spread model).
    missile.accuracy.spacing "db": the target's own close spacing (sp_h, sp_v) and formation (geometry.dims);
    a number: that spacing for all, the same files and ranks (the prototype's 2.0, build/accuracy/proto.py)."""
    cfg = params.sim["missile"]["accuracy"]
    if cfg.get("plane"):
        return hit_chance_plane(u, pw, ellipse=bool(params.sim["missile"].get("body_ellipse")), params=params)
    k = float(cfg["k"])
    d = pw["dist"].clamp(min=5.0)
    shrink = torch.sqrt((1 - u["acc"] / 100).clamp(min=0))[:, :, None]
    sig = (k * u["cal_area"][:, :, None] * d / u["cal_dist"].clamp(min=1.0)[:, :, None] * shrink).clamp(min=1e-3)
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = 0.5 * torch.asin((d * 9.81 / (v * v)).clamp(max=1.0))
    shadow = u["height"][:, None, :] / torch.tan(theta.clamp(min=1e-3))
    r = u["radius"][:, None, :]
    p_lone = 1 - torch.exp(-(math.pi * r * r + 2 * r * torch.minimum(shadow, 3 * sig)) / (2 * math.pi * sig * sig))
    spacing = cfg.get("spacing", "db")
    if spacing == "db":
        h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))
        sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))
        D = (pw["depth"] - sv).clamp(min=0)                    # the target's formation (geometry.dims)
    else:
        h = sv = torch.full_like(u["men"], float(spacing))
        own_v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))
        D = (torch.round(pw["depth"] / own_v) - 1).clamp(min=0) * sv
    D = D[:, None, :]
    lateral = torch.maximum((2 * r / h[:, None, :]).clamp(max=1.0), torch.erf(r / (math.sqrt(2) * sig)))
    along = torch.zeros_like(sig)
    n = 7
    for q in range(n):
        y0 = D * (q + 0.5) / n
        along = along + _phi((D + shadow - y0) / sig) - _phi((-r - y0) / sig)
    p_form = lateral * along / n
    return torch.where((u["men0"] <= 1)[:, None, :], p_lone, p_form).clamp(0, 1)


def body_radii(u, pw):
    """([B, i, j] half-width across i's line of fire, half-length along it) of a man of j, m (missile.body_ellipse).
    The database's battle_entities body is a circle of `radius` (infantry, lords: radii_ratio 1) or an ellipse radius
    long along the man's facing and radius x radii_ratio across it (beasts, weapon teams) - the hit box of projectiles
    (community: an entity's hitbox is its shape, radius and radii ratio). Seen from i at the angle phi off j's facing
    (pw rel_j): half-width r sqrt(sin^2 phi + q^2 cos^2 phi), half-length r sqrt(cos^2 phi + q^2 sin^2 phi), q the
    ratio; a circle keeps r both ways."""
    r = u["radius"][:, None, :].expand_as(pw["rel_j"])
    q = u["body_ratio"][:, None, :].expand_as(r)
    s, c = torch.sin(pw["rel_j"]), torch.cos(pw["rel_j"])
    ell = q < 1
    across = torch.where(ell, r * torch.sqrt(s * s + q * q * c * c), r)
    along = torch.where(ell, r * torch.sqrt(c * c + q * q * s * s), r)
    return across, along


def hit_chance_plane(u, pw, ellipse=False, params=None):
    """[B, i, j] the hit chance with the spread in the plane across the line of fire (module docstring; missile.accuracy
    plane): sigma from the calibration AREA, the aim at a man's middle, ranks not in exact files. ellipse: a man's body
    is his battle entity's ellipse seen from the shooter (body_radii; missile.body_ellipse): its half-width across the
    line of fire where the formula has r across (the lone man's erf, the cover), its half-length along it elsewhere.
    params: the accuracy's cut (acc_shrink; None: the area's)."""
    d = pw["dist"].clamp(min=5.0)
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = (0.5 * torch.asin((d * 9.81 / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    sig, half = _spread(u, d, acc_theta(theta, v, d, pw.get("dh")), params)
    L = u["height"][:, None, :] / torch.tan(theta)
    sa = sig / torch.sin(theta)                      # the spread along the line, on the ground
    c = aim_frac(params) * L                         # the aim point (a man's middle) beyond his feet
    if ellipse:
        rx, r = body_radii(u, pw)                    # across the line of fire, along it
    else:
        rx = r = u["radius"][:, None, :]
    if half is not None:                             # the square spread: uniform +-half across, +-ha along the ground
        ha = half / torch.sin(theta)
        lat_lone = (rx / half).clamp(max=1.0)
        p_lone = lat_lone * _share(-r, L + r, c, ha)
    else:
        lat_lone = torch.erf(rx / (math.sqrt(2) * sig))
        p_lone = lat_lone * (_phi((L + r - c) / sa) - _phi((-r - c) / sa))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))[:, None, :]
    depth = pw["depth"][:, None, :]
    D = (depth - sv).clamp(min=0)                    # the target's formation (geometry.dims)
    ranks = (depth / sv).clamp(min=1)
    n = torch.minimum(((torch.minimum(L, D) + 2 * r) / sv).clamp(min=1.0), ranks)
    cover = 1 - (1 - (2 * rx / h).clamp(max=1.0)) ** n
    lateral = torch.maximum(cover, lat_lone)
    span = width_capture(u, pw, sig, rx, params, half=half)
    if span is not None:
        lateral = torch.maximum(span * cover, lat_lone)
    along = torch.zeros_like(sig)
    q_n = 7
    for q in range(q_n):
        y0 = D * (q + 0.5) / q_n
        if half is not None:
            along = along + _share(-r, D + L, y0 + c, ha)
        else:
            along = along + _phi((D + L - c - y0) / sa) - _phi((-r - c - y0) / sa)
    p_form = lateral * along / q_n
    return torch.where((u["men0"] <= 1)[:, None, :], p_lone, p_form).clamp(0, 1)


def front_share(u, pw, params):
    """[B, i, j] the share of i's hits on formation j that strike j's front rank (kills.missile_front): a projectile
    strikes the first man on its path (CA: a projectile is a body that must touch a model), and a formation stands
    file behind file (the game's places: standing swordsmen's ranks within 0.1 m of each other's files), so a flat
    shot that comes in at a man's height meets the front man of its file; only what passes between the front men or
    over their heads goes deeper. The same spread model as hit_chance_plane (no new number): the front rank's chance
    - its cover across, min(1, 2 r / its spacing), x the landing points whose low path passes the front rank (aimed
    at a man y0 deep, the window [-r, L + r] at the front) - over the formation's chance. Arcing fire (a short low
    path L) spreads over the depth (the thin probe's even spread of arrows); a flat bullet's L of 37-61 m puts most
    of its hits on the front rank. 1 for a lone man."""
    d = pw["dist"].clamp(min=5.0)
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = (0.5 * torch.asin((d * 9.81 / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    sig, half = _spread(u, d, acc_theta(theta, v, d, pw.get("dh")), params)
    L = u["height"][:, None, :] / torch.tan(theta)
    sa = sig / torch.sin(theta)
    c = aim_frac(params) * L
    if params.sim["missile"].get("body_ellipse"):
        rx, r = body_radii(u, pw)
    else:
        rx = r = u["radius"][:, None, :]
    if half is not None:
        ha = half / torch.sin(theta)
        lat_lone = (rx / half).clamp(max=1.0)
    else:
        lat_lone = torch.erf(rx / (math.sqrt(2) * sig))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))[:, None, :]
    depth = pw["depth"][:, None, :]
    D = (depth - sv).clamp(min=0)
    ranks = (depth / sv).clamp(min=1)
    n = torch.minimum(((torch.minimum(L, D) + 2 * r) / sv).clamp(min=1.0), ranks)
    cover1 = (2 * rx / h).clamp(max=1.0)
    lateral = torch.maximum(1 - (1 - cover1) ** n, lat_lone)
    lat_first = torch.maximum(cover1, lat_lone)
    span = width_capture(u, pw, sig, rx, params, half=half)
    if span is not None:
        lateral = torch.maximum(span * (1 - (1 - cover1) ** n), lat_lone)
        lat_first = torch.maximum(span * cover1, lat_lone)
    along = torch.zeros_like(sig)
    first = torch.zeros_like(sig)
    q_n = 7
    for q in range(q_n):
        y0 = D * (q + 0.5) / q_n
        if half is not None:
            along = along + _share(-r, D + L, y0 + c, ha)
            first = first + _share(-r, L + r, y0 + c, ha)
        else:
            along = along + _phi((D + L - c - y0) / sa) - _phi((-r - c - y0) / sa)
            first = first + _phi((L + r - c - y0) / sa) - _phi((-r - c - y0) / sa)
    p_form = lateral * along / q_n
    p_first = lat_first * first / q_n
    share = (p_first / p_form.clamp(min=1e-9)).clamp(0, 1)
    return torch.where((u["men0"] <= 1)[:, None, :] | (ranks <= 1.0), torch.ones_like(share), share)


def _hermite_table(n_max=32):
    """Standard-normal quadrature points and weights (Gauss-Hermite) for 1..n_max points, as plain floats. Built once
    at import, outside torch.compile: numpy inside the compiled simulator step is traced as CPU tensors, and inductor
    then needs a C++ compiler for them (none in the image)."""
    import numpy as np
    out = {}
    for n in range(1, n_max + 1):
        x, w = np.polynomial.hermite_e.hermegauss(n)
        out[n] = ([float(v) for v in x], [float(v) for v in w / w.sum()])
    return out


_GH_TABLE = _hermite_table()


def _legendre_table(n_max=32):
    """Gauss-Legendre points and weights on [-1, 1] for 1..n_max points, as plain floats (built at import, as the
    Hermite table)."""
    import numpy as np
    out = {}
    for n in range(1, n_max + 1):
        x, w = np.polynomial.legendre.leggauss(n)
        out[n] = ([float(v) for v in x], [float(v) for v in w])
    return out


_GL_TABLE = _legendre_table()


def _hermite(device, dtype, n):
    """Standard-normal quadrature points and weights (Gauss-Hermite) on the state's device."""
    x, w = _GH_TABLE[n]
    return torch.tensor(x, device=device, dtype=dtype), torch.tensor(w, device=device, dtype=dtype)


def _legendre(device, dtype, n):
    """Gauss-Legendre points and weights on [-1, 1] on the state's device."""
    x, w = _GL_TABLE[n]
    return torch.tensor(x, device=device, dtype=dtype), torch.tensor(w, device=device, dtype=dtype)


def _uniform_phi(s, half, sig):
    """E over y uniform on [-half, half] of Phi((s - y) / sig) (closed form: G(t) = t Phi(t) + phi(t))."""
    def g(t):
        return t * _phi(t) + torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    return sig / (2 * half).clamp(min=1e-6) * (g((s + half) / sig) - g((s - half) / sig))


def spill_geometry(u, pw, target, params):
    """(on_k [B, i, k], keep_j [B, i]): of shooter i's shots at its target j, the expected hits on each other formation
    k of j's side, and the share of j's own hits that stays j's (k in front of it or among its men takes the shot
    first). The same spread as hit_chance_plane (missile.spill_geometry; build/routgap/spill_analytic2.py checks it
    against a Monte Carlo of real men): the aim a random man of j (uniform over its rectangle), the landing point
    Gaussian across sigma and along sigma / sin(theta), the shot low over its last L = height / tan(theta) m. Across:
    the share of shots landing in k's span (closed form). Along: quadrature over the landing point; each formation's
    ranks inside the low band [x - L - r, x + r] give its chance 1 - (1 - 2r / files' step)^ranks; the one whose men in
    the band start nearer the shooter is hit first, interleaved bands share the union."""
    cfg = params.sim["missile"]["spill_geometry"]
    qa, qg = int(cfg.get("aim_points", 4)), int(cfg.get("gauss_points", 8))
    t = target.clamp(min=0)
    B, N = target.shape
    tj = t[:, :, None]
    d_ij = pw["dist"].gather(2, tj).clamp(min=5.0)                          # [B, i, 1]
    th_ij = pw["theta"].gather(2, tj)                                       # bearing i -> j
    v0 = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = (0.5 * torch.asin((d_ij * 9.81 / (v0 * v0)).clamp(max=1.0))).clamp(min=1e-3)
    dh_ij = pw["dh"].gather(2, tj) if "dh" in pw else None
    sig, _ = _spread(u, d_ij, acc_theta(theta, v0, d_ij, dh_ij), params)   # (the square: its equal-variance Gaussian)
    sa = sig / torch.sin(theta)
    r = u["radius"][:, None, :]                                             # [B, 1, k]
    Lk = u["height"][:, None, :] / torch.tan(theta)                         # [B, i, k]
    Lj = Lk.gather(2, tj)
    c = aim_frac(params) * Lj
    # each formation in shooter i's line-of-fire frame (along +: away from i; across)
    xj, zj = u["x"].gather(1, t)[:, :, None], u["z"].gather(1, t)[:, :, None]
    ex, ez = torch.sin(th_ij), torch.cos(th_ij)
    dx, dz = u["x"][:, None, :] - xj, u["z"][:, None, :] - zj              # [B, i, k] from j's centre
    b = dx * ex + dz * ez
    a = dx * ez - dz * ex
    phi = u["b"][:, None, :] * geometry.DEG - th_ij                          # k's facing against the line
    cs, sn = torch.cos(phi).abs(), torch.sin(phi).abs()
    front, depth = pw["front"][:, None, :], pw["depth"][:, None, :]
    half_w = front / 2 * cs + depth / 2 * sn                               # across
    half_d = front / 2 * sn + depth / 2 * cs                               # along
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))[:, None, :]
    step_across = h * cs + sv * sn
    step_along = sv * cs + h * sn
    cov1 = (2 * r / step_across).clamp(max=1.0)
    hw_j, hd_j = half_w.gather(2, tj), half_d.gather(2, tj)
    # across: the share of j-aimed shots landing in k's span, and in j's own
    lat_k = (_uniform_phi(a + half_w, hw_j, sig) - _uniform_phi(a - half_w, hw_j, sig)).clamp(0, 1)
    lat_j = lat_k.gather(2, tj)
    both = torch.minimum(lat_k, lat_j)
    # along: landing points x = aim (uniform over j's depth) + c + sa * g
    gx, gw = _hermite(sig.device, sig.dtype, qg)
    aim = (torch.arange(qa, device=sig.device, dtype=sig.dtype) + 0.5) / qa * 2 - 1     # [-1, 1]
    xl = (aim[:, None] * hd_j[..., None, None] + c[..., None, None] + sa[..., None, None] * gx[None, :])   # [B, i, 1, qa, qg]
    w = (gw[None, :] / qa).expand(qa, qg)

    def band(L_, c0, hd, step, cv, x=None):
        x = xl if x is None else x
        lo = torch.maximum(x - L_[..., None, None] - r[..., None, None], (c0 - hd)[..., None, None])
        hi = torch.minimum(x + r[..., None, None], (c0 + hd)[..., None, None])
        inside = (hi - lo).clamp(min=0)
        n = torch.where(inside > 0, (inside / step[..., None, None]).clamp(min=1.0), torch.zeros_like(inside))
        return 1 - (1 - cv[..., None, None]) ** n, lo

    def first(x):
        pk, sk = band(Lk, b, half_d, step_along, cov1, x)                    # [B, i, k, qa, qg]
        pj, sj = band(Lj, torch.zeros_like(hd_j), hd_j, step_along.gather(2, tj), cov1.gather(2, tj), x)
        sv_k = step_along[..., None, None]
        k_first = sk < sj - sv_k
        j_first = sj < sk - sv_k
        union = 1 - (1 - pj) * (1 - pk)
        share = union / (pj + pk).clamp(min=1e-9)
        hk = torch.where(k_first, pk, torch.where(j_first, pk * (1 - pj), pk * share))
        hj = torch.where(j_first, pj, torch.where(k_first, pj * (1 - pk), pj * share))
        return hk, hj, pj, pk
    hk, hj, pj, pk = first(xl)
    pk_alone = (pk * w).sum((-1, -2))                                        # [B, i, k]
    hk = (hk * w).sum((-1, -2))                                              # [B, i, k]
    hj = (hj * w).sum((-1, -2))
    pj_alone = (pj * w).sum((-1, -2))                                        # [B, i, 1]
    own_n = int(cfg.get("own_points") or 0)
    if own_n:
        # (missile.spill_geometry.own_points) j's own hits - what it keeps - over the landing points where its band is
        # not empty, [-hd_j - r, hd_j + L_j + r] from its centre (Gauss-Legendre, own_n points, weighted by the landing
        # density: the aim uniform over its depth, Gaussian along). The Hermite nodes above spread over the whole
        # spread lie sigma / sin(theta) apart (a flat shot: 40 m at 130 m), wider than a lone man's band (his height /
        # tan(theta), 34 m): his own hits came out 0, so any formation of his side ahead of him took them all - the
        # Ratling Guns of the itS26_15 exam never hit our General behind his line
        gx_, gw_ = _legendre(sig.device, sig.dtype, own_n)
        lo_w = -hd_j - r.expand(-1, N, -1).gather(2, tj)                                       # [B, i, 1]
        hi_w = hd_j + Lj + r.expand(-1, N, -1).gather(2, tj)
        mid, half = 0.5 * (lo_w + hi_w), 0.5 * (hi_w - lo_w)
        xo = mid[..., None, None] + half[..., None, None] * gx_[None, None, None, None, :]     # [B, i, 1, 1, n]
        mu = aim[:, None] * hd_j[..., None, None] + c[..., None, None]                         # [B, i, 1, qa, 1]
        dens = torch.exp(-0.5 * ((xo - mu) / sa[..., None, None]) ** 2) / (math.sqrt(2 * math.pi) * sa[..., None, None])
        wo = dens * half[..., None, None] * gw_ / qa                                            # [B, i, 1, qa, n]
        _, hj_o, pj_o, _ = first(xo)
        hj = (hj_o * wo).sum((-1, -2))
        pj_alone = (pj_o * wo).sum((-1, -2))
        # ... and each other formation k's hits over its own window [b - hd_k - r, b + hd_k + L_k + r]
        lo_k, hi_k = b - half_d - r, b + half_d + Lk + r                     # [B, i, k]
        mid_k, half_k = 0.5 * (lo_k + hi_k), 0.5 * (hi_k - lo_k)
        xk = mid_k[..., None, None] + half_k[..., None, None] * gx_                            # [B, i, k, 1, n]
        dens_k = torch.exp(-0.5 * ((xk - mu) / sa[..., None, None]) ** 2) / (math.sqrt(2 * math.pi) * sa[..., None, None])
        hk_o, _, _, pk_o = first(xk)
        wk = dens_k * half_k[..., None, None] * gw_ / qa
        hk = (hk_o * wk).sum((-1, -2))
        pk_alone = (pk_o * wk).sum((-1, -2))
        # across, k takes j's own hits only where its men stand in j's span: the share of j-aimed shots landing in the
        # overlap of the two spans (min(lat_k, lat_j) counted a formation beside a lone man as standing in his way)
        lo_c = torch.maximum(a - half_w - r, -hw_j)
        hi_c = torch.minimum(a + half_w + r, hw_j)
        both_keep = torch.where(hi_c > lo_c, (_uniform_phi(hi_c, hw_j, sig) - _uniform_phi(lo_c, hw_j, sig)).clamp(0, 1),
                                torch.zeros_like(hi_c))
        both_keep = torch.minimum(both_keep, lat_j)
    else:
        both_keep = both
    on_k = both * hk
    if cfg.get("lateral_rest"):
        # (missile.spill_geometry.lateral_rest) the shots that land in k's span but beside j's (a lone man's 1.2 m:
        # most of the gun's spread at him) meet k alone: they were lost before (min(lat_k, lat_j) took only the
        # shots through both spans)
        on_k = on_k + (lat_k - both).clamp(min=0) * pk_alone
    # j keeps its hits outside k's span in full; inside, hj of its pj_alone
    keep_jk = 1 - both_keep * (1 - hj / pj_alone.clamp(min=1e-9)) / lat_j.clamp(min=1e-9)
    side_j = u["side"].gather(1, t)[:, :, None]
    eye = torch.arange(N, device=t.device)[None, None, :] == tj
    other = (u["side"][:, None, :] == side_j) & ~eye & ((u["men"] > 0) & ~u["gone"])[:, None, :] & (u["men0"] > 1)[:, None, :]
    far = pw["dist"].gather(1, t[:, :, None].expand(B, N, N)) > float(cfg.get("max_m", 120.0))   # j to k
    other = other & ~far & (target >= 0)[:, :, None]
    on_k = torch.where(other, on_k, torch.zeros_like(on_k))
    keep_j = torch.where(other, keep_jk.clamp(0, 1), torch.ones_like(keep_jk)).prod(2)
    return on_k, keep_j


G = 9.81


def _arc_tan(v, L, dh):
    """tan of the low launch angle that puts a shot of speed v (m/s) dh m higher at L m: a flat shot keeps its speed and
    its angle rises with the range (tw-modding 'Missiles and You'); out of reach: 45 deg."""
    disc = v ** 4 - G * (G * L * L + 2 * dh * v * v)
    tan = (v * v - torch.sqrt(disc.clamp(min=0))) / (G * L.clamp(min=1e-3))
    return torch.where(disc >= 0, tan, torch.ones_like(tan))


def _cover(u, pw, target, params):
    """(cover, along, between) [B, i, k]: the share of shooter i's lines to its target that friend k covers across the
    line, k's distance along the line, and whether k stands between i and the target's edge (module docstring).
    Rectangles across the line: front x |cos| + depth x |sin|."""
    t = target.clamp(min=0)[:, :, None]
    theta = pw["theta"].gather(2, t)                              # [B, i, 1] bearing to the target
    dist = pw["dist"].gather(2, t).clamp(min=1e-3)
    near_edge = pw["reach_j"].gather(2, t)                         # to the target's edge
    rel = pw["theta"] - theta                                      # [B, i, k] k seen off the line
    along = pw["dist"] * torch.cos(rel)
    across = pw["dist"] * torch.sin(rel)
    b = u["b"] * geometry.DEG
    front, depth = pw["front"], pw["depth"]
    phi = b[:, None, :] - theta                                    # k's facing against the line
    coef = params.battle.get("projectile_friendly_fire_man_radius_coefficient", 2.2)
    half = (front[:, None, :] / 2 * torch.cos(phi).abs() + depth[:, None, :] / 2 * torch.sin(phi).abs()
            + (coef - 1) * u["radius"][:, None, :])
    phi_i = b[:, :, None] - theta
    width = (front[:, :, None] * torch.cos(phi_i).abs() + depth[:, :, None] * torch.sin(phi_i).abs()).clamp(min=1e-3)
    # The lines from the shooter's front converge on the target's centre: at `along` the line from
    # offset s across passes at s x (1 - along / dist); k covers the offsets lo..hi.
    shrink = (1 - along / dist).clamp(min=1e-3)
    lo, hi = (across - half) / shrink, (across + half) / shrink
    cover = (torch.minimum(hi, width / 2) - torch.maximum(lo, -width / 2)).clamp(min=0) / width
    N = u["men"].shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=u["men"].device)[None]
    friend = ((u["side"][:, :, None] == u["side"][:, None, :]) & ~eye & (u["side"][:, None, :] > 0)
              & ((u["men"] > 0) & ~u["gone"])[:, None, :])
    between = friend & (along > 0) & (along < near_edge)
    return cover, along, between


def arc_lines(u, pw, target, params):
    """(blocked [B, i, k], low [B, i, k]) of direct-fire shooter i at its target past friend k (missile.arc_los,
    build/step4c): the shot's arc (a fixed speed - the passport's muzzle velocity, gravity 9.81) from a muzzle muzzle_m
    above the ground to an aim point aim_m above the ground at a target man, checked over k's span along the line
    against its men's height x projectile_friendly_fire_man_height_coefficient (1.0 with True Sight: friends at their
    own size). Over a 2 x 2 grid of the shooter's ranks and the target's men (the two-point Gauss nodes of a uniform
    depth, +-0.577 of each half depth): blocked - the share of those lines that do not clear k; low - the share of the
    shots that pass below k's heads by the spread (hit_chance_plane's sigma, scaled to k's distance along the line),
    over the grid (a blocked line counts 0). The arc is lowest at an end of k's span: its two ends are checked."""
    cal = params.sim["missile"]["arc_los"]
    hm, ha = float(cal["muzzle_m"]), float(cal["aim_m"])
    coef_h = params.battle.get("projectile_friendly_fire_man_height_coefficient", 1.15)
    t = target.clamp(min=0)
    theta = pw["theta"].gather(2, t[:, :, None])
    dist = pw["dist"].gather(2, t[:, :, None])                     # [B, i, 1]
    along = pw["dist"] * torch.cos(pw["theta"] - theta)            # [B, i, k]
    phi = (u["b"] * geometry.DEG)[:, None, :] - theta
    span = (pw["depth"][:, None, :] * torch.cos(phi).abs() + pw["front"][:, None, :] * torch.sin(phi).abs()) / 2
    hf = (u["height"] * coef_h)[:, None, :]
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    ds = (pw["depth"] / 2)[:, :, None]                             # the shooter's half depth
    dt_ = pw["depth"].gather(1, t)[:, :, None] / 2                 # the target's half depth
    th_d = (0.5 * torch.asin((dist.clamp(min=5.0) * 9.81 / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    sig, _ = _spread(u, dist, th_d, params)         # (the square spread: its equal-variance Gaussian here)
    blocked = torch.zeros_like(along)
    low = torch.zeros_like(along)
    nodes = (-0.57735, 0.57735)
    for sr in nodes:
        s = sr * ds
        for tr in nodes:
            L = (dist + tr * dt_ - s).clamp(min=1.0)
            tan = _arc_tan(v, L, ha - hm)
            clear = torch.ones_like(along, dtype=torch.bool)
            below = torch.zeros_like(along)
            for e in (-1.0, 1.0):
                x = torch.minimum((along + e * span - s).clamp(min=0.0), L)
                c = hm + x * tan - G * x * x * (1 + tan * tan) / (2 * v * v) - hf
                clear = clear & (c > 0)
                below = torch.maximum(below, _phi(-c / (sig * x / L).clamp(min=1e-3)))
            blocked = blocked + (~clear).float() / 4
            low = low + torch.where(clear, below, torch.zeros_like(below)) / 4
    return blocked, low


def _lines(u, pw, target, params, arc_on):
    """(blocked [B, N], catch [B, N, N] or None) for shooter i at its target: blocked - the share of i's men whose line
    a friend between covers (module docstring; with missile.arc_los only the lines whose arc does not clear it); catch -
    the share of i's shots (of the men that fire) that hit friend k: k's cover across the line x the share of the
    clearing lines' shots passing below k's heads (arc_lines), over the share of i's men that fire; direct fire only,
    never more than all the shots."""
    cover, _, between = _cover(u, pw, target, params)
    between = between.float()
    catch = None
    if arc_on:
        arc_blocked, low = arc_lines(u, pw, target, params)
        eff = cover * arc_blocked * between
        blocked = eff.sum(2).clamp(max=1)
        catch = cover * low * between / (1 - blocked).clamp(min=1e-3)[:, :, None]
        catch = torch.where((u["direct"] & (target >= 0))[:, :, None], catch, torch.zeros_like(catch))
        catch = catch / catch.sum(2, keepdim=True).clamp(min=1.0)
    else:
        blocked = (cover * between).sum(2).clamp(max=1)
    return torch.where(target >= 0, blocked, torch.zeros_like(blocked)), catch


def blocked_share(u, pw, target, params, arc=None):
    """[B, N] share of shooter i's men whose line to its target (slot, -1 none: 0) passes through a friendly unit
    (module docstring); arc (any value): by the bullet's arc (missile.arc_los, arc_lines)."""
    return _lines(u, pw, target, params, arc is not None)[0]


def friend_catch(u, pw, target, params):
    """[B, i, k] share of shooter i's shots that hit friend k between it and its target (_lines, missile.arc_los)."""
    return _lines(u, pw, target, params, True)[1]


# The friends' men in the way (missile.friend_los; module docstring "Friends in the way"): the depth nodes of a
# shooter's men (Gauss-Legendre over his depth) and their weights; the unit's rays across its front (ends and middle).
_NODES = ((-0.7745967, 5 / 18), (0.0, 8 / 18), (0.7745967, 5 / 18))
_RAYS = (-0.5, 0.0, 0.5)


def _path_block(lo, hi, xa, xb, rk, dmin=None):
    """The length of the friends' centre interval [lo, hi] along a path whose bodies (radius rk, the check's) reach
    its blocked part - the path below their heads before xa and after xb (the arc's two crossings of their height).
    dmin (the bodies' collision, the two men's radii: a friend's man never stands nearer to the shooter; None or 0:
    no collision): within dmin of the shooter there are no centres - the window ahead starts at the band's mean
    distance to that circle, m = mean over |u| < rk of sqrt(dmin^2 - u^2) - and the men the collision pushed out of
    it along the line between the two stand on that circle: those within the path's band (asin(rk / dmin) of the
    circle each side) add asin(rk / dmin) dmin^2 / (2 rk) of the window when the arc is below the heads there. A
    body within rk of the muzzle would block at any arc; with dmin >= rk the collision leaves none there."""
    span = (hi - lo).clamp(min=0)
    pushed = torch.zeros_like(span)
    if dmin is not None:
        d = torch.maximum(dmin, rk).expand_as(span)
        rk_ = (rk.expand_as(span) if torch.is_tensor(rk) else torch.full_like(span, float(rk))).clamp(min=1e-3)
        d = torch.maximum(d, rk_)
        on = (dmin.expand_as(span) > 0) & (rk_ > 1e-3)        # (an empty slot: no body)
        ang = torch.asin((rk_ / d).clamp(max=1.0))
        m = (rk_ * torch.sqrt((d * d - rk_ * rk_).clamp(min=0)) + d * d * ang) / (2 * rk_)
        lo = torch.where(on, torch.maximum(lo, m), lo)
        # (the pushed men: the friend's centres reached into the circle, and the arc is below the heads at it)
        into = on & (lo - m < 1e-6) & (hi > m) & (xa + rk_ > m)
        pushed = torch.where(into, ang * d * d / (2 * rk_), pushed)
    near = (torch.minimum(hi, torch.maximum(xa + rk, math.pi / 4 * rk)) - lo).clamp(min=0)
    far = (hi - torch.maximum(lo, xb - rk)).clamp(min=0)
    return torch.minimum(near + far + pushed, span + pushed)


def _lob_tan(u):
    """[B, N, 1] tan of a lobbing shooter's arc: its fixed elevation (artillery: the Mortar's 50 deg; STATIC fix_el),
    else 45 deg (1)."""
    if "fix_el" not in u:
        return None
    el = u["fix_el"] * (math.pi / 180)
    return torch.where(u["fix_el"] > 0, torch.tan(el), torch.ones_like(el))[:, :, None]


def _friends(u, pw, target, params, contact=None, lob=None, men=True, rays=True):
    """(men [B, N] or None, closed [B, N] bool or None) of shooter i at its target (slot, -1: zeros) past the men of
    its other own units (missile.friend_los, module docstring "Friends in the way"):
    men - the share of i's men whose path a friend's man blocks: the low arc of its projectile, the 45 deg arc for the
    shooters of lob [B, N] (the forced arc of clear_shot; None: nobody);
    closed - the unit's line of fire: of the three rays from its front (its ends and middle) along the low arc, fewer
    than rays_clear clear (a ray is blocked when a friend's man blocks it with a chance of 0.5 or more).
    A path is blocked by a friend's man whose body (his radius x projectile_friendly_fire_man_radius_coefficient, his
    height x ..._man_height_coefficient: 1.0 with True Sight) it passes below the head of: the arc (missile.arc_los's
    muzzle and aim point; without arc_los a straight line) crosses the friends' height at xa and xb, so the men whose
    centres lie before xa + r or after xb - r along it block it; a friend's men stand in its files and ranks: a path in
    a file's band (min(2 r / file spacing, 1) of the paths) meets a man there with the share of the rank spacing the
    window covers (at most 1), a path between the files passes all the ranks - the same men blocked every volley;
    a body within its radius of the muzzle (beside, behind) blocks at any arc - the window starts pi / 4 rk behind it
    (_path_block). The aim point: aim_frac (0.75) of the target's height. The men of i's own unit never block, but
    for a unit in melee with its target (contact [B, N, N]); the unit's rays ignore a friend whose formation is inside
    i's: their circumscribed circles intersect and the turned rectangles touch (geometry.rect_gap <= 0)."""
    ms = params.sim["missile"]
    arc = ms.get("arc_los")
    fcfg = ms["friend_los"]
    hm = float(arc["muzzle_m"]) if arc else float(fcfg.get("muzzle_m", 1.5))
    coef_h = float(params.battle.get("projectile_friendly_fire_man_height_coefficient", 1.15))
    coef_r = float(params.battle.get("projectile_friendly_fire_man_radius_coefficient", 2.2))
    N = u["men"].shape[1]
    dev = u["men"].device
    t = target.clamp(min=0)[:, :, None]
    th = pw["theta"].gather(2, t)                                       # [B, i, 1] bearing to the target
    dist = pw["dist"].gather(2, t).clamp(min=1e-3)
    edge = pw["reach_j"].gather(2, t)                                   # to the target's near edge
    # the aim point: aim_frac of the target man's height (0.75; the game's very large units 0.6 - none in our roster)
    ha = float(fcfg.get("aim_frac", 0.75)) * u["height"].gather(1, target.clamp(min=0))[:, :, None]
    eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
    rel = pw["theta"] - th
    along = torch.where(eye, torch.zeros_like(rel), pw["dist"] * torch.cos(rel))   # [B, i, k]
    across = torch.where(eye, torch.zeros_like(rel), pw["dist"] * torch.sin(rel))
    alive = (u["men"] > 0) & ~u["gone"]
    friend = ((u["side"][:, :, None] == u["side"][:, None, :]) & ~eye & (u["side"][:, None, :] > 0)
              & alive[:, None, :])
    own = torch.zeros_like(friend)
    if contact is not None:
        fights = contact.gather(2, t).squeeze(2) & (target >= 0)          # [B, i]: in melee with its target
        own = eye & fights[:, :, None]
    b = u["b"] * geometry.DEG
    sp_h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))
    sp_v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))
    rad = u["radius"] * coef_r
    lone = u["men0"] <= 1
    # the friends k: their men's centres along the line (+-span) and their bodies across it (+-half), their spacing
    phi = b[:, None, :] - th
    c, s = torch.cos(phi).abs(), torch.sin(phi).abs()
    front, depth = pw["front"][:, None, :], pw["depth"][:, None, :]
    rk = rad[:, None, :]
    lone_k = lone[:, None, :]
    s_al = torch.where(lone_k, 2 * rk, c * sp_v[:, None, :] + s * sp_h[:, None, :]).clamp(min=1e-3)
    s_lat = c * sp_h[:, None, :] + s * sp_v[:, None, :]
    span = torch.where(lone_k, torch.zeros_like(s_al), (depth / 2 * c + front / 2 * s - s_al / 2).clamp(min=0))
    half = torch.where(lone_k, rk, (front / 2 * c + depth / 2 * s - s_lat / 2).clamp(min=0) + rk)
    rank_cover = torch.where(lone_k, torch.ones_like(s_al), (2 * rk / s_lat.clamp(min=1e-3)).clamp(max=1.0))
    q = (u["height"] * coef_h)[:, None, :] - hm                         # the friends' heads above the muzzle
    # the shooter i: its men's centres along the line (+-span_i) and across it (width_i)
    phi_i = b[:, :, None] - th
    ci, si = torch.cos(phi_i).abs(), torch.sin(phi_i).abs()
    fi, di = pw["front"][:, :, None], pw["depth"][:, :, None]
    hi_, vi_ = sp_h[:, :, None], sp_v[:, :, None]
    lone_i = lone[:, :, None]
    span_i = torch.where(lone_i, torch.zeros_like(fi),
                         (di / 2 * ci + fi / 2 * si - (ci * vi_ + si * hi_) / 2).clamp(min=0))
    width_i = torch.where(lone_i, torch.zeros_like(fi), (fi * ci + di * si - (ci * hi_ + si * vi_)).clamp(min=0))
    ri = rad[:, :, None]
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    dh = ha - hm
    if "dh" in pw:
        # the ground's height (terrain.py): the aim point and the friends' heads over the shooter's ground - a shooter on a
        # hill shoots over its own infantry below, one below a rise shoots into the friends on it
        dh = dh + pw["dh"].gather(2, t)
        q = q + pw["dh"]

    # the bodies' collision: a man of another unit stands no nearer than the two men's radii (the database's entity
    # radius, its shape a circle; the probe 'stacked': none nearer than 0.6 m, the least ~1.1 m once settled)
    dmin = torch.where(eye, torch.zeros_like(along), (u["radius"][:, :, None] + u["radius"][:, None, :]).expand_as(along))

    def place(y):
        """The path of i's man y m ahead of i's centre along the line: (L, the friends' centre interval along it lo..hi,
        how far across the path has come towards the target's centre at each friend)."""
        L = (dist - y).clamp(min=1.0)
        # (a body within rk of the muzzle blocks the shot at any arc - behind or beside it too: from pi / 4 rk behind,
        # the mean half chord, but the bodies' collision keeps a friend's men dmin off (_path_block); the unit's own
        # men in melee from the next man on, ri + rk: the shooter is not his own obstacle)
        lo = torch.maximum(along - span - y, torch.where(eye, ri + rk, -math.pi / 4 * rk))
        hi = torch.minimum(along + span - y, edge - y)
        return L, lo, hi, (1 - (along - y) / L).clamp(min=0)

    def block(L, lo, hi, high=None):
        """[B, i, k] the chance a path that crosses friend k (place) passes below one of its men's heads; high [B, i, 1]
        bool: the shooters on the 45 deg arc (None: all on the low arc)."""
        if arc is None:
            tan, a = dh / L, torch.zeros_like(L)                         # a straight line
        else:
            tan = _arc_tan(v, L, dh)
            a = ((tan * L - dh) / (L * L)).clamp(min=0)                 # height hm + tan x - a x^2: through the aim
        if high is not None:
            # (the 45 deg arc; a fixed trajectory's own elevation: _lob_tan)
            th = _lob_tan(u)
            if th is None:
                tan = torch.where(high, torch.ones_like(tan), tan)
                a = torch.where(high, ((L - dh) / (L * L)).clamp(min=0), a)
            else:
                tan = torch.where(high, th.expand_as(tan), tan)
                a = torch.where(high, ((th * L - dh) / (L * L)).clamp(min=0), a)
        disc = tan * tan - 4 * a * q
        sq = torch.sqrt(disc.clamp(min=0))
        aa = (2 * a).clamp(min=1e-9)
        big = torch.full_like(disc, FAR)
        xa_q = torch.where(disc >= 0, (tan - sq) / aa, big)
        xb_q = torch.where(disc >= 0, (tan + sq) / aa, big)
        # a straight line: below the heads up to q / tan when it rises, everywhere when it does not
        rising = tan > 1e-9
        xa_l = torch.where(q <= 0, -big, torch.where(rising, q / tan.clamp(min=1e-9), big))
        curved = a > 1e-12
        xa = torch.where(curved, xa_q, xa_l)
        xb = torch.where(curved, xb_q, big)
        # a path in a file's band meets a man of it within the window with the share of the rank spacing it covers;
        # between the files it passes every rank (the files stand in line: the same men blocked every volley)
        return rank_cover * (_path_block(lo, hi, xa, xb, rk, dmin) / s_al).clamp(max=1.0)

    def cover(shrink, s_off=None):
        """[B, i, k] the path s_off m across i's line passes through friend k's bodies (None: the share of i's men
        across its width whose paths do)."""
        if s_off is not None:
            return ((across - s_off * shrink).abs() <= half).float()
        sh = shrink.clamp(min=1e-3)
        lo_s, hi_s = (across - half) / sh, (across + half) / sh
        w = width_i.expand_as(lo_s)
        share = (torch.minimum(hi_s, w / 2) - torch.maximum(lo_s, -w / 2)).clamp(min=0) / w.clamp(min=1e-3)
        return torch.where(w > 1e-3, share.clamp(max=1.0), ((lo_s <= 0) & (hi_s >= 0)).float())

    on = target >= 0
    zero3 = torch.zeros_like(along)
    share = closed = None
    if men:
        who = friend | own
        high = None if lob is None else lob[:, :, None]
        share = torch.zeros_like(u["men"])
        for g, w in _NODES:
            L, lo, hi, shrink = place(g * span_i)
            cov = torch.where(who, cover(shrink), zero3)
            share = share + w * (1 - torch.prod(1 - cov * block(L, lo, hi, high), dim=2))
        share = torch.where(on, share, torch.zeros_like(share))
    if rays:
        L, lo, hi, shrink = place(span_i)
        # "a formation inside another own one": their circumscribed circles intersect and the turned rectangles touch
        # somewhere (rect_gap <= 0); a lone man: his circle
        ri_c = torch.where(lone, rad, torch.sqrt((pw["front"] / 2) ** 2 + (pw["depth"] / 2) ** 2))
        circles = pw["dist"] < ri_c[:, :, None] + ri_c[:, None, :]
        touch = geometry.rect_gap(u["x"], u["z"], u["b"], pw["front"], pw["depth"]) <= 0
        lone_pair = lone[:, :, None] | lone[:, None, :]
        touch = torch.where(lone_pair, pw["gap"] <= 0, touch)
        apart = ~(circles & touch)
        front_block = torch.where(friend & apart, block(L, lo, hi), zero3)
        shut = torch.zeros_like(u["men"])
        for r in _RAYS:
            ray = 1 - torch.prod(1 - cover(shrink, r * width_i) * front_block, dim=2)
            shut = shut + (ray >= 0.5).float()
        closed = ((len(_RAYS) - shut) < int(fcfg.get("rays_clear", 2))) & on
    return share, closed


def friend_rays(u, pw, target, params):
    """[B, N] bool: shooter i's line of fire at its target is closed by the men of its other own units (_friends:
    fewer than rays_clear of the three rays from its front clear)."""
    return _friends(u, pw, target, params, men=False)[1]


def friend_lines(u, pw, target, params, contact=None, lob=None):
    """[B, N] the share of shooter i's men whose path to its target a man of another own unit blocks (_friends): the
    low arc, the 45 deg arc for the shooters of lob [B, N]; contact: a unit in melee with its target - its own men too."""
    return _friends(u, pw, target, params, contact=contact, lob=lob, rays=False)[0]


def clear_shot(u, pw, target, can_shoot, order_target, order_attack, params, with_catch=False, contact=None,
               info=None):
    """The line of fire past friends (module docstring): (target [B, N], clear [B, N] share of the men
    that shoot) and, with_catch, the shots' catch by friends between ([B, N, N], None without missile.arc_los).
    With missile.friend_los (the game's rules, "Friends in the way"): a unit whose line of fire is closed (friend_rays)
    at its target - direct fire always, arcing fire within lob_min_m of it - takes the next target in range; closed
    there too, it holds fire (-1); arcing fire with a closed line beyond lob_min_m shoots at 45 deg; the men whose own
    path a friend's man blocks miss that shot (clear = 1 - their share). contact [B, N, N]: units in melee (a unit in
    melee with its target: its own men block too). info (a dict): filled with blocked [B, N] (the share of the men a
    friend blocks at the final target), held [B, N] (holding fire for friends) and lob [B, N] (the forced 45 deg arc).
    Without friend_los: direct fire only, a unit blocked at its target beyond the obstructed ratio takes the next
    target in range; blocked there too, it holds fire (-1). Other shooters: as given, clear 1."""
    ms = params.sim["missile"]
    direct = u["direct"]
    arc_on = ms.get("arc_los") is not None
    per_man = per_man_mask(u, params)
    reach = reach_of(params)
    fcfg = ms.get("friend_los")
    catch = None
    if fcfg:
        lob_min = float(fcfg.get("lob_min_m", 10.0))

        def judge(tg):
            """(lob, bad) [B, N]: the unit's line at tg is closed (friend_rays) - arcing fire beyond lob_min_m lobs,
            the rest may not shoot at it."""
            closed = friend_rays(u, pw, tg, params)
            if "dh" in pw:
                # the ground in the way (terrain.ground_closed: a hill between - direct fire does not shoot at that
                # target, arcing fire lobs at 45 deg, as past friends)
                from tools.nn.sim import terrain
                closed = closed | terrain.ground_closed(u, pw, tg, params)
            d = pw["dist"].gather(2, tg.clamp(min=0)[:, :, None]).squeeze(2)
            lob = ~direct & closed & (d > lob_min)
            if "fix_el" in u:
                # a fixed trajectory (artillery: the Mortar's 50 deg) always arcs: over friends and hills, its men's
                # paths on its own arc (artillery_rules.md 6)
                lob = lob | ((u["fix_el"] > 0) & (tg >= 0))
            return lob, closed & ~lob
        lob, bad = judge(target)
        exclude = torch.zeros_like(pw["enemy"]).scatter_(2, target.clamp(min=0)[:, :, None], True) & bad[:, :, None]
        again = choose_target(u, pw, can_shoot & bad, order_target, order_attack, exclude=exclude, per_man=per_man,
                              reach=reach)
        target = torch.where(bad, again, target)
        lob2, bad2 = judge(target)
        lob = torch.where(bad, lob2, lob)
        held = bad & ((target < 0) | bad2)
        target = torch.where(held, torch.full_like(target, -1), target)
        lob = lob & (target >= 0)
        # the men whose own path a friend blocks, at the final target and on its arc (one pass)
        blocked = friend_lines(u, pw, target, params, contact, lob=lob)
        clear = 1 - blocked
        if with_catch and arc_on:
            # the bullets that pass below a friend's heads by the spread hit him (friend_catch): direct fire only
            catch = _lines(u, pw, target, params, True)[1]
        if info is not None:
            info.update(blocked=blocked, held=held, lob=lob)
    else:
        ratio = params.battle.get("unit_firing_line_of_sight_considered_obstructed_ratio", 0.75)
        blocked, catch = _lines(u, pw, target, params, arc_on)
        bad = direct & (target >= 0) & (blocked >= ratio)
        exclude = torch.zeros_like(pw["enemy"]).scatter_(2, target.clamp(min=0)[:, :, None], True) & bad[:, :, None]
        again = choose_target(u, pw, can_shoot & bad, order_target, order_attack, exclude=exclude, per_man=per_man,
                              reach=reach)
        target = torch.where(bad, again, target)
        blocked2, catch2 = _lines(u, pw, target, params, arc_on)
        blocked = torch.where(bad, blocked2, blocked)
        if catch is not None:
            catch = torch.where(bad[:, :, None], catch2, catch)
        held = direct & (target >= 0) & (blocked >= ratio)
        target = torch.where(held, torch.full_like(target, -1), target)
        clear = torch.where(direct, 1 - blocked, torch.ones_like(blocked))
    if per_man is not None:
        # only the ranks within range of the target's nearest men shoot (rank_share)
        share = rank_share(u, pw, *reach).gather(2, target.clamp(min=0)[:, :, None]).squeeze(2)
        clear = torch.where(per_man, clear * share, clear)
    out = (target, torch.where(target >= 0, clear, torch.zeros_like(clear)))
    return out + (catch,) if with_catch else out


def burst_share(u, volley_target, m_target, loaded, dt):
    """A burst weapon's step (spv > 1 shots a man in a volley at bst_rate a second: the Ratling Gun's 18 at 4 a second,
    measured - its bullets at a lone target: 64 every 0.5 s from 32 men, 576 = 32 x 18 a burst, build/newunits).
    -> (burst [B, N] the burst units, start [B, N] a burst starts now, loaded [B, N] the share of a man's shots to fire
    this step as volley() takes it, target [B, N]). A burst starts where a volley would (volley_target: the men loaded
    and aimed) and goes on spv / bst_rate s at the unit's target now (m_target; none: no shot that step, the clock
    runs); the other units keep their volley (loaded, volley_target). Updates bst, bst_s in place."""
    burst = (u["spv"] > 1) & (u["bst_rate"] > 0)
    start = burst & (volley_target >= 0) & (u["bst_s"] <= 1e-6)
    length = u["spv"] / u["bst_rate"].clamp(min=1e-6)
    bst_s = torch.where(start, length, u["bst_s"])
    bst = torch.where(start, loaded, u["bst"])
    going = burst & (bst_s > 1e-6) & (m_target >= 0)
    now = torch.minimum(bst_s, torch.full_like(bst_s, dt))
    # the men aiming again (burst_reaim: their target man died) come back at aim_s (on average: exp(-dt / aim_s)); a
    # new burst starts with every man aimed
    paused = torch.zeros_like(bst)
    if "bst_p" in u:
        back = torch.exp(-dt / u["aim_s"].clamp(min=1e-3))
        paused = torch.where(start, torch.zeros_like(bst), u["bst_p"] * back)
        u["bst_p"] = torch.where(burst, paused, u["bst_p"])
    share = torch.where(burst, torch.where(going, bst * (1 - paused) * u["bst_rate"] * now, torch.zeros_like(bst)),
                        loaded)
    target = torch.where(burst, torch.where(going, m_target, torch.full_like(m_target, -1)), volley_target)
    u["bst_s"] = torch.where(burst, (bst_s - dt).clamp(min=0), u["bst_s"])
    u["bst"] = torch.where(burst & (u["bst_s"] > 0), bst, torch.zeros_like(bst))
    return burst, start, share, target


def burst_cut(u, mask, spread_s=0.0):
    """A new order mid-burst (missile.burst_cut_reload; battle.py, the order block): the burst units of mask [B, N]
    whose burst is going end it. The men who had begun it reload from the cut as from a burst's own end (the cycle
    less the burst: the gun's 12 - 4.5 = 7.5 s); the others stay loaded. The share that had begun: the burst's
    elapsed seconds / spread_s (the game's men begin a burst over ~2 s; 0: all of them), the unit's wait that share of
    the rest (one clock for the unit). Its next burst, whole (spv a man), comes once it is loaded and the spin-up is
    done (the aim clock the order reset); the reload cycle counts from it. -> [B, N] the units cut."""
    if "bst_s" not in u:
        return torch.zeros_like(mask)
    cut = mask & (u["spv"] > 1) & (u["bst_rate"] > 0) & (u["bst_s"] > 1e-6)
    length = u["spv"] / u["bst_rate"].clamp(min=1e-6)
    begun = torch.ones_like(u["bst_s"])
    if spread_s > 0:
        begun = ((length - u["bst_s"]) / spread_s).clamp(0, 1)
    wait = begun * (u["reload"] - length).clamp(min=0)
    u["bst_s"] = torch.where(cut, torch.zeros_like(u["bst_s"]), u["bst_s"])
    u["bst"] = torch.where(cut, torch.zeros_like(u["bst"]), u["bst"])
    u["unready"] = torch.where(cut, (wait / u["reload"].clamp(min=1e-6)).clamp(0, 1), u["unready"])
    if "bst_p" in u:
        u["bst_p"] = torch.where(cut, torch.zeros_like(u["bst_p"]), u["bst_p"])
    return cut


def shield_pierce(u, params, shooter=None):
    """[B, N] (with shooter [B, X]: those men's) the share of a shield's block chance left against the projectiles:
    1 - missile_shield_piercing_coefficient x their armour-piercing share of the damage (missile.shield_ap_pierce; off:
    1). Measured on the Ratling Gun (6 of 8 damage armour-piercing) at a lone General (55 % shield): his HP per bullet
    facing the gun / his back to it 0.66 at 135 m, 0.68 at 100 m (the ratlord probe; config/nn/sim.json
    missile.shield_ap_pierce_why), the rule 1 - 0.55 x (1 - 0.5 x 0.75) = 0.66, the plain block 0.45."""
    ms = params.sim["missile"]
    if shooter is None:
        m_d, m_a = u["m_damage"], u["m_ap"]
    else:
        m_d, m_a = u["m_damage"].gather(1, shooter), u["m_ap"].gather(1, shooter)
    if not ms.get("shield_ap_pierce"):
        return torch.ones_like(m_d)
    coef = float(params.battle.get("missile_shield_piercing_coefficient", 0.0))
    return 1 - coef * m_a / (m_d + m_a).clamp(min=1e-6)


def men_struck(u, limit):
    """[B, i, j] the men one projectile of i strikes in j at most (the game's penetration, line g43: the database's
    projectile_penetration_junctions and battle_entities): the first man, and the next along its path while the man it
    struck is smaller than its size cap (pen_cap > size_k) and its counter - each man struck adds his
    projectile_penetration_resistance (pen_res) - is at most max_penetration (pen_max; -1 no limit): 1 + floor(pen_max
    / pen_res) men, at most `limit` (sim.json missile.penetration.max_men). The Jezzails' and Long Rifles' medium_1 (max
    1, cap medium) strike 2 infantrymen, the Ironsides' low (cap small) 1, very_low 1; the Giant Rats (resistance 2) stop
    medium_1 in the first. A projectile's speed lost per man (battle_entities projectile_penetration_speed_change 0.9) does
    not change its damage (the database's damage is the projectile's)."""
    passes = u["size_k"][:, None, :] < u["pen_cap"][:, :, None]
    mx = u["pen_max"][:, :, None]
    more = torch.where(mx < 0, torch.full_like(mx, float(limit)), torch.floor(mx / u["pen_res"][:, None, :].clamp(min=1e-3)))
    return 1 + torch.where(passes, more.clamp(0, float(limit) - 1), torch.zeros_like(more))


def pen_limit(params):
    """The most men a projectile of the passports can strike (men_struck over every shooter and every target of
    config/nn/units.json), at most missile.penetration.max_men: 1 when none penetrates (penetration() then does
    nothing). Worked out once when the parameters are made (params.pen_limit_of: Params.pen_men), read here: the step's
    loop has a fixed length and no Python over the passports runs in the compiled step."""
    return int(getattr(params, "pen_men", 1))


def penetration(u, pw, params, shield, fs=None):
    """[B, i, j] the HP of i's hits on formation j that land past the first man, as a factor on its first men's (1: none):
    1 + sum over k = 1 .. men_struck - 1 of P(a man stands k ranks behind the first struck along the line) x (1 -
    shield)^k (a blocked projectile stops; each man behind faces the shooter too). The ranks along the line of fire R:
    the formation's depth / sp_v seen from the front, its front / sp_h from the flank (|cos|, |sin| of the bearing); the
    first man struck is in the front rank by front_share's chance (a flat projectile meets the front man of its file),
    else in a rank behind it evenly. A lone man (a lord): 1 (a man behind him is another unit's; not modelled)."""
    cfg = params.sim["missile"].get("penetration") or {}
    limit = pen_limit(params)
    if not cfg.get("on") or limit <= 1:
        return torch.ones_like(pw["dist"])
    n = men_struck(u, limit)
    sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))[:, None, :]
    sh = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    c, s_ = torch.cos(pw["rel_j"]).abs(), torch.sin(pw["rel_j"]).abs()
    R = (c * (pw["depth"][:, None, :] / sv) + s_ * (pw["front"][:, None, :] / sh)).clamp(min=1.0)
    if fs is None:
        fs = front_share(u, pw, params) if params.sim["missile"].get("accuracy", {}).get("plane") else 1.0 / R
    keep = 1 - shield
    out = torch.ones_like(n)
    for k in range(1, limit):
        behind = fs * (R - 1 >= k).float() + (1 - fs) * ((R - 1 - k).clamp(min=0) / (R - 1).clamp(min=1.0))
        out = out + torch.where(n > k, behind * keep ** k, torch.zeros_like(behind))
    lone = (u["men0"] <= 1)[:, None, :]
    return torch.where(lone, torch.ones_like(out), out)


def volley(u, pw, target, dt, params, contact=None, clear=None, loaded=None, catch=None, info=None):
    """Shots, and HP taken per pair [B, N, N] (i shoots, f is hit) this step; per-hit damage
    [B, N, N]. clear [B, N]: share of the shooter's men with a clear line (clear_shot; None: all).
    loaded [B, N]: share of the men loaded, who all shoot now (None: men x dt / reload, the steady rate).
    contact [B, N, N]: which units touch in melee. Of the hits aimed at a unit in
    melee, the shooter's friendly_fire share lands on its own units in contact with the target
    (measured, docs/en/training/simulator.md). info (a dict): filled with pen [B, N, N], the penetration's factor
    (penetration(): the pair's HP over its first men's) when missile.penetration is on; its "front" (given: front_share
    of the step, battle.py) is used for the first man's rank."""
    ms = params.sim["missile"]
    B = params.battle
    shooting = target >= 0
    per_man = dt / u["reload"].clamp(min=1e-6) if loaded is None else loaded
    # (an artillery unit's projectiles: its firing guns x shots_per_volley; artillery.shooters)
    from tools.nn.sim import artillery
    shots = torch.where(shooting, artillery.shooters(u) * per_man, torch.zeros_like(u["men"]))
    if clear is not None:
        shots = shots * clear
    shots = torch.minimum(shots, u["a"])
    t = target.clamp(min=0)
    onehot = torch.zeros_like(pw["dist"]).scatter_(2, t[:, :, None], 1.0) * shooting[:, :, None]
    dist = pw["dist"]
    factor = distance_factor(dist, ms["distance_factor"])
    old = (u["hit_rate"][:, :, None] * factor).clamp(max=1.0)  # the measured rates (missile.accuracy.units "rates")
    men = u["men"].clamp(min=0)
    lone = torch.where(u["men0"] <= 1, torch.full_like(men, ms["single_entity_factor"]), torch.ones_like(men))
    acc = ms.get("accuracy")
    if acc:
        # the spread model (hit_chance) for a lone man always, for formations unless missile.accuracy.units "rates";
        # its lone-man chance replaces single_entity_factor out of melee
        p = hit_chance(u, pw, params)
        if acc.get("units", "model") == "rates":
            p = torch.where((u["men0"] <= 1)[:, None, :], p, old)
        rate = p
        own_out = torch.ones_like(men)
    else:
        rate, own_out = old, lone
    caught = None
    shots_on = shots
    if ms.get("arc_los") is not None:
        # direct fire past friends (missile.arc_los): the shots that pass below a friend's heads hit it, the rest go on
        if catch is None:
            catch = friend_catch(u, pw, target, params)             # [B, i, k] (clear_shot's, when given)
        caught = catch * shots[:, :, None]
        shots_on = shots * (1 - catch.sum(2))
    geo = None
    if ms.get("spill_geometry") and acc and acc.get("plane"):
        # the spill from the spread itself (spill_geometry): k's hits, and j's own hits left after k took its share
        geo = spill_geometry(u, pw, target, params)
        rate = rate * geo[1][:, :, None]
    aimed = onehot * shots_on[:, :, None] * rate              # [B, i, j] hits aimed at j
    aimed_old = onehot * shots_on[:, :, None] * old
    if contact is None:
        landed = aimed * own_out[:, None, :]
    else:
        # Hits aimed at a unit in melee: the shooter's `friendly_fire` share lands on its own
        # units in contact with the target (split by their men), the rest on the target.
        near = contact.transpose(1, 2).float() * men[:, None, :]  # [B, j, f] men of f in contact with j
        crowd = near.sum(2)
        engaged = (crowd > 0).float()
        friends = near / crowd.clamp(min=1e-6)[:, :, None]
        ff = u["friendly_fire"][:, :, None] * engaged[:, None, :]  # [B, i, j]
        # (with the spread model, hits aimed at a lone man in melee keep the measured rule: hit_rate x distance
        # factor, his share x single_entity_in_melee)
        lone_melee = ((u["men0"] <= 1) & (engaged > 0))[:, None, :]
        # ... but the shooters of missile.single_in_melee_projectiles (the Ratling Gun: lone_hit) hit a lone man in
        # melee as out of it: the spread model's hits, his whole share (config/nn/sim.json single_in_melee_projectiles_why)
        as_out = (u["lone_hit"][:, :, None] & lone_melee) if "lone_hit" in u else torch.zeros_like(lone_melee)
        aimed = torch.where(lone_melee & ~as_out, aimed_old, aimed)
        # A lone man among them (a lord) is hit as a lone target is: single_entity_factor of his
        # share (without it a lord in melee with shot enemies took the whole friendly fire).
        # The target's own share: a lone man in melee takes single_entity_in_melee of it (measured 0:
        # config/nn/sim.json missile.single_in_melee_why); out of melee single_entity_factor.
        in_melee = (u["men0"] <= 1) & (engaged > 0)
        own = torch.where(in_melee, lone, own_out)
        if ms.get("single_entity_in_melee") is not None:
            own = torch.where(in_melee, torch.full_like(men, float(ms["single_entity_in_melee"])), own_out)
        own = torch.where(as_out, own_out[:, None, :], own[:, None, :])        # [B, i, j]
        landed = torch.bmm(aimed * ff, friends * lone[:, None, :]) + aimed * (1 - ff) * own
    if caught is not None:
        landed = landed + caught
    if geo is not None:
        landed = landed + geo[0] * shots_on[:, :, None]
    # Spill: hits aimed at j also land on j's neighbours of its own side that are out of melee,
    # a share by distance between centres (measured; off with spill_geometry).
    elif ms.get("spill"):
        same = (u["side"][:, :, None] == u["side"][:, None, :]) & (u["side"][:, :, None] > 0)
        eye = torch.eye(men.shape[1], dtype=torch.bool, device=men.device)[None]
        free = (men > 0) if contact is None else (men > 0) & ~contact.any(2)
        spill = distance_factor(dist, ms["spill"]) * (same & ~eye & free[:, None, :]).float()
        if contact is not None and ms.get("spill_melee"):
            # Hits aimed at a unit in melee also land on its own side's units in melee near it
            # (measured: the spearmen around a General the slingers shoot at).
            fighting = (men > 0) & contact.any(2)
            near = distance_factor(dist, ms["spill_melee"]) * (same & ~eye & fighting[:, None, :]).float()
            spill = spill + near * fighting[:, :, None].float()
        landed = landed + torch.bmm(aimed_old, spill * lone[:, None, :])
    # a shield blocks small arms only (build/shields/spec.md S4); every projectile of our roster is one
    front = (pw["rel_j"].abs() <= B["shield_defence_angle_missile"] * geometry.DEG) & u["small_arms"][:, :, None]
    shield = torch.where(front, u["shield"][:, None, :], torch.zeros_like(dist))
    shield = shield * shield_pierce(u, params)[:, :, None]
    # Resistance: missile; with missile.physical_resist also physical (CA: physical resistance works against all
    # non-magical damage, the sum at most 90 %; config/nn/sim.json missile.physical_resist_why).
    resist = u["resist_missile"]
    if ms.get("physical_resist"):
        resist = (resist + u["resist_physical"]).clamp(max=0.9)
    # the same damage rule as a blow (melee.per_hit: armour roll, overkill; a lone man loses the mean)
    md, ma = u["m_damage"][:, :, None], u["m_ap"][:, :, None]
    if "m_bvl" in u:
        # the projectile's bonus against a large target (projectiles.bonus_v_large: the Warp Lightning Cannon +70) on
        # its base damage, as a blow's (given only to artillery: config/nn/sim.json artillery.why)
        md = md + u["m_bvl"][:, :, None] * u["large"][:, None, :].float()
    if "dh" in pw:
        # the height between shooter and target (terrain.missile_factor: x (1 + clamp(0.3 x (shooter's ground - the
        # target's) / 40 m, +-0.3)) on the projectile's damage before the armour; a battle without a height map: none)
        from tools.nn.sim import terrain
        hf = terrain.missile_factor(pw["dh"], params)
        md, ma = md * hf, ma * hf
    hit = melee.per_hit(md, ma, u["armour"][:, None, :],
                        u["hp_man"][:, None, :], resist[:, None, :], single=(u["men0"] <= 1)[:, None, :])
    hp = landed * hit * (1 - shield)
    if "fix_el" in u:
        # a fixed trajectory's shell (the Mortar) strikes when it lands, with its blast (artillery.land): none here
        hp = torch.where((u["fix_el"] > 0)[:, :, None], torch.zeros_like(hp), hp)
    if (ms.get("penetration") or {}).get("on") and "pen_max" in u:
        # a projectile through several men (penetration: the Jezzails, the Long Rifles)
        pen = penetration(u, pw, params, shield, fs=(info or {}).get("front"))
        hp = hp * pen
        if info is not None:
            info["pen"] = pen
    return shots, hp, hit
