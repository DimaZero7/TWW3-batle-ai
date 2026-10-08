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
A new order (another kind or another attack target) makes it aim again (missile.aim_reset_on_order).

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
(missile.accuracy.plane false: the older ground model, sigma = k x calibration_area x d / calibration_distance x
sqrt(1 - accuracy / 100) with a fitted k, aimed at the feet, exact files.)
A lone man in melee keeps hit_rate x distance factor x single_entity_in_melee (measured, batch 3); spill and
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


def rank_share(u, pw):
    """[B, i, j] share of direct-fire shooter i's men that fire at target j (missile.per_man_range_direct; the probe
    build/probes7 P2, militia with pistols, range 90, 7 ranks 1.7 m apart, on skavenslaves 10 ranks deep):
    - the whole unit fires (the IsFiringMissiles flag on) while j's centre is within range of i's FRONT RANK (i's centre
      + (ranks - 1) / 2 x v_i ahead): full volleys at 85 / 88 / 90 m centre to centre and 0.94-0.99 at 94.5 m (front
      rank 89.4 m from the target centre), the flag off at 97.6 m (92.5 m);
    - beyond that only the ranks within range of j's nearest men fire (the flag off): rank r (0 the front) stands
      g0 + r v_i from them, g0 = the edge gap + v_i / 2 + v_j / 2 (the men stand half a spacing inside the edges);
      at 97.6 m 4 of 7 ranks = 0.57, the game a steady stream at 0.54 of the full rate for 200 s.
    A lone shooter or target: no ranks (spacing 0)."""
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


def in_range(u, pw, per_man=False):
    """[B, i, j] j within i's range: centre to centre; with per_man (True: every direct-fire shooter; a [B, N] mask:
    those shooters, per_man_mask) a shooter's men each to the target's nearest men (rank_share > 0)."""
    centre = pw["dist"] <= u["range"][:, :, None]
    if per_man is None or per_man is False:
        return centre
    mask = u["direct"] if per_man is True else per_man
    return torch.where(mask[:, :, None], rank_share(u, pw) > 0, centre)


def choose_target(u, pw, can_shoot, order_target, order_attack, exclude=None, prev=None, per_man=False):
    """[B, N] slot each shooter aims at (-1 none). exclude [B, N, N]: targets i may not take. prev [B, N]: last
    step's target, kept while it is in range (centre to centre; per_man: in_range), standing and not excluded (the
    game keeps its target; without it the nearest-each-step flips between two close enemies)."""
    alive = (u["men"] > 0) & ~u["gone"]
    rng = pw["enemy"] & alive[:, None, :] & in_range(u, pw, per_man)
    if exclude is not None:
        rng = rng & ~exclude
    standing = rng & ~u["r"][:, None, :]
    score = torch.where(standing, pw["dist"], torch.where(rng, pw["dist"] + 1e5, torch.full_like(pw["dist"], FAR)))
    nearest = score.argmin(dim=2)
    has = score.min(dim=2).values < FAR
    if prev is not None:
        p = prev.clamp(min=0)
        keep = (prev >= 0) & standing.gather(2, p[:, :, None]).squeeze(2)
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


def hit_chance(u, pw, params):
    """[B, i, j] chance that a projectile of i aimed at j hits a man of j (module docstring: the spread model).
    missile.accuracy.spacing "db": the target's own close spacing (sp_h, sp_v) and formation (geometry.dims);
    a number: that spacing for all, the same files and ranks (the prototype's 2.0, build/accuracy/proto.py)."""
    cfg = params.sim["missile"]["accuracy"]
    if cfg.get("plane"):
        return hit_chance_plane(u, pw)
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


def hit_chance_plane(u, pw):
    """[B, i, j] the hit chance with the spread in the plane across the line of fire (module docstring; missile.accuracy
    plane): sigma from the calibration AREA, the aim at a man's middle, ranks not in exact files."""
    d = pw["dist"].clamp(min=5.0)
    shrink = torch.sqrt((1 - u["acc"] / 100).clamp(min=0))[:, :, None]
    sig = (torch.sqrt(u["cal_area"].clamp(min=0))[:, :, None] * d / u["cal_dist"].clamp(min=1.0)[:, :, None]
           * shrink).clamp(min=1e-3)
    v = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = (0.5 * torch.asin((d * 9.81 / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    L = u["height"][:, None, :] / torch.tan(theta)
    sa = sig / torch.sin(theta)                      # the spread along the line, on the ground
    c = 0.5 * L                                      # the aim point (a man's middle) beyond his feet
    r = u["radius"][:, None, :]
    lat_lone = torch.erf(r / (math.sqrt(2) * sig))
    p_lone = lat_lone * (_phi((L + r - c) / sa) - _phi((-r - c) / sa))
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 2.0))[:, None, :]
    sv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 2.0))[:, None, :]
    depth = pw["depth"][:, None, :]
    D = (depth - sv).clamp(min=0)                    # the target's formation (geometry.dims)
    ranks = (depth / sv).clamp(min=1)
    n = torch.minimum(((torch.minimum(L, D) + 2 * r) / sv).clamp(min=1.0), ranks)
    cover = 1 - (1 - (2 * r / h).clamp(max=1.0)) ** n
    lateral = torch.maximum(cover, lat_lone)
    along = torch.zeros_like(sig)
    q_n = 7
    for q in range(q_n):
        y0 = D * (q + 0.5) / q_n
        along = along + _phi((D + L - c - y0) / sa) - _phi((-r - c - y0) / sa)
    p_form = lateral * along / q_n
    return torch.where((u["men0"] <= 1)[:, None, :], p_lone, p_form).clamp(0, 1)


_GH = None


def _hermite(device, dtype, n):
    """Standard-normal quadrature points and weights (Gauss-Hermite), cached."""
    global _GH
    if _GH is None or _GH[0].device != device or _GH[0].dtype != dtype or len(_GH[0]) != n:
        import numpy as np
        x, w = np.polynomial.hermite_e.hermegauss(n)
        _GH = (torch.tensor(x, device=device, dtype=dtype), torch.tensor(w / w.sum(), device=device, dtype=dtype))
    return _GH


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
    shrink = torch.sqrt((1 - u["acc"] / 100).clamp(min=0))[:, :, None]
    sig = (torch.sqrt(u["cal_area"].clamp(min=0))[:, :, None] * d_ij / u["cal_dist"].clamp(min=1.0)[:, :, None]
           * shrink).clamp(min=1e-3)
    v0 = u["muzzle_v"].clamp(min=1.0)[:, :, None]
    theta = (0.5 * torch.asin((d_ij * 9.81 / (v0 * v0)).clamp(max=1.0))).clamp(min=1e-3)
    sa = sig / torch.sin(theta)
    r = u["radius"][:, None, :]                                             # [B, 1, k]
    Lk = u["height"][:, None, :] / torch.tan(theta)                         # [B, i, k]
    Lj = Lk.gather(2, tj)
    c = 0.5 * Lj
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

    def band(L_, c0, hd, step, cv):
        lo = torch.maximum(xl - L_[..., None, None] - r[..., None, None], (c0 - hd)[..., None, None])
        hi = torch.minimum(xl + r[..., None, None], (c0 + hd)[..., None, None])
        inside = (hi - lo).clamp(min=0)
        n = torch.where(inside > 0, (inside / step[..., None, None]).clamp(min=1.0), torch.zeros_like(inside))
        return 1 - (1 - cv[..., None, None]) ** n, lo
    pk, sk = band(Lk, b, half_d, step_along, cov1)                           # [B, i, k, qa, qg]
    pj, sj = band(Lj, torch.zeros_like(hd_j), hd_j, step_along.gather(2, tj), cov1.gather(2, tj))
    sv_k = step_along[..., None, None]
    k_first = sk < sj - sv_k
    j_first = sj < sk - sv_k
    union = 1 - (1 - pj) * (1 - pk)
    share = union / (pj + pk).clamp(min=1e-9)
    hk = torch.where(k_first, pk, torch.where(j_first, pk * (1 - pj), pk * share))
    hj = torch.where(j_first, pj, torch.where(k_first, pj * (1 - pk), pj * share))
    hk = (hk * w).sum((-1, -2))                                              # [B, i, k]
    hj = (hj * w).sum((-1, -2))
    pj_alone = (pj * w).sum((-1, -2))                                        # [B, i, 1]
    on_k = both * hk
    # j keeps its hits outside k's span in full; inside, hj of its pj_alone
    keep_jk = 1 - both * (1 - hj / pj_alone.clamp(min=1e-9)) / lat_j.clamp(min=1e-9)
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
    sig = (torch.sqrt(u["cal_area"].clamp(min=0))[:, :, None] * dist / u["cal_dist"].clamp(min=1.0)[:, :, None]
           * torch.sqrt((1 - u["acc"] / 100).clamp(min=0))[:, :, None]).clamp(min=1e-3)
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


def clear_shot(u, pw, target, can_shoot, order_target, order_attack, params, with_catch=False):
    """Direct fire's line of fire (module docstring): (target [B, N], clear [B, N] share of the men
    that shoot) and, with_catch, the shots' catch by friends between ([B, N, N], None without missile.arc_los).
    A direct-fire unit blocked at its target beyond the obstructed ratio takes the next
    target in range; blocked there too, it holds fire (-1). Other shooters: as given, clear 1."""
    ratio = params.battle.get("unit_firing_line_of_sight_considered_obstructed_ratio", 0.75)
    direct = u["direct"]
    arc_on = params.sim["missile"].get("arc_los") is not None
    blocked, catch = _lines(u, pw, target, params, arc_on)
    bad = direct & (target >= 0) & (blocked >= ratio)
    exclude = torch.zeros_like(pw["enemy"]).scatter_(2, target.clamp(min=0)[:, :, None], True) & bad[:, :, None]
    per_man = per_man_mask(u, params)
    again = choose_target(u, pw, can_shoot & bad, order_target, order_attack, exclude=exclude, per_man=per_man)
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
        share = rank_share(u, pw).gather(2, target.clamp(min=0)[:, :, None]).squeeze(2)
        clear = torch.where(per_man, clear * share, clear)
    out = (target, torch.where(target >= 0, clear, torch.zeros_like(clear)))
    return out + (catch,) if with_catch else out


def volley(u, pw, target, dt, params, contact=None, clear=None, loaded=None, catch=None):
    """Shots, and HP taken per pair [B, N, N] (i shoots, f is hit) this step; per-hit damage
    [B, N, N]. clear [B, N]: share of the shooter's men with a clear line (clear_shot; None: all).
    loaded [B, N]: share of the men loaded, who all shoot now (None: men x dt / reload, the steady rate).
    contact [B, N, N]: which units touch in melee. Of the hits aimed at a unit in
    melee, the shooter's friendly_fire share lands on its own units in contact with the target
    (measured, docs/en/training/simulator.md)."""
    ms = params.sim["missile"]
    B = params.battle
    shooting = target >= 0
    per_man = dt / u["reload"].clamp(min=1e-6) if loaded is None else loaded
    shots = torch.where(shooting, u["men"] * per_man, torch.zeros_like(u["men"]))
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
        aimed = torch.where(lone_melee, aimed_old, aimed)
        # A lone man among them (a lord) is hit as a lone target is: single_entity_factor of his
        # share (without it a lord in melee with shot enemies took the whole friendly fire).
        # The target's own share: a lone man in melee takes single_entity_in_melee of it (measured 1:
        # config/nn/sim.json missile.single_in_melee_why); out of melee single_entity_factor.
        in_melee = (u["men0"] <= 1) & (engaged > 0)
        own = torch.where(in_melee, lone, own_out)
        if ms.get("single_entity_in_melee") is not None:
            own = torch.where(in_melee, torch.full_like(men, float(ms["single_entity_in_melee"])), own_out)
        landed = torch.bmm(aimed * ff, friends * lone[:, None, :]) + aimed * (1 - ff) * own[:, None, :]
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
    # Resistance: missile; with missile.physical_resist also physical (CA: physical resistance works against all
    # non-magical damage, the sum at most 90 %; config/nn/sim.json missile.physical_resist_why).
    resist = u["resist_missile"]
    if ms.get("physical_resist"):
        resist = (resist + u["resist_physical"]).clamp(max=0.9)
    # the same damage rule as a blow (melee.per_hit: armour roll, overkill; a lone man loses the mean)
    hit = melee.per_hit(u["m_damage"][:, :, None], u["m_ap"][:, :, None], u["armour"][:, None, :],
                        u["hp_man"][:, None, :], resist[:, None, :], single=(u["men0"] <= 1)[:, None, :])
    hp = landed * hit * (1 - shield)
    return shots, hp, hit
