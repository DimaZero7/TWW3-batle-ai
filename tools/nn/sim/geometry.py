"""Where units are relative to each other: formation size, pairwise distances, sides, contact.

A unit is a rectangle: `width` m of front ordered (fixed at the start), files = floor(width / h) men h apart
and as many ranks as its men fill, v apart (h, v: the unit's formation template in the game's database,
config/nn/units.json spacing; config/nn/sim.json formation.spacing_m where a unit has none); a lord is a
circle of his radius. Bearing b in degrees: facing (sin b, cos b) in (x, z).
"""
import math

import torch

DEG = math.pi / 180


def melee_depth(sim):
    """formation.melee_depth's ranks per lost share (config/nn/sim.json; 0 / absent: off) - dims' melee_depth."""
    return float(((sim.get("formation") or {}).get("melee_depth") or {}).get("ranks") or 0.0)


def dims(u, spacing, melee_depth=0.0):
    """(front, depth) [B, N], m: files = floor(ordered width / h) (at least 1, at most the men), front = files x h,
    depth = ceil(men / files) x v (an artillery unit: one rank, v) (the game's own formations, data/roster: front (files - 1) h, depth (ranks - 1)
    v, within 1 % / 10 % at 15-40 m; build/meleecore/spacing_check.py). spacing: h and v of a unit without its own.
    melee_depth (formation.melee_depth.ranks, c; 0: off): a formation in melee (its melee clock contact_s > 0) does
    not close up as its men fall - ranks = max(ceil(r), min(r + c (1 - men / men0), ceil(men0 / files0))), r = men /
    files, files0 = floor(width / h): the dead leave c ranks' worth of room at their whole loss (config/nn/sim.json
    formation.melee_depth.why); out of melee the box closes up as before (the missile probe P1)."""
    men = u["men"].clamp(min=0)
    single = u["men0"] <= 1
    if "sp_h" in u:
        h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
        v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    else:
        h = v = torch.full_like(men, float(spacing))
    files0 = torch.clamp(torch.floor(u["width"] / h + 1e-4), min=1)
    files = torch.minimum(files0, men.clamp(min=1))
    ranks = torch.ceil(men.clamp(min=1) / files)
    if melee_depth > 0 and "contact_s" in u:
        men0 = u["men0"].clamp(min=1)
        loose = torch.minimum(men.clamp(min=1) / files + melee_depth * (1 - men / men0).clamp(min=0),
                              torch.ceil(men0 / files0))
        ranks = torch.where(u["contact_s"] > 0, torch.maximum(ranks, loose), ranks)
    if "eng0" in u:
        # an artillery unit (artillery.py): one rank of its guns, eng0 files h apart, its crew round them - the box is
        # the guns' line, v deep (the database's artillery spacing templates; NOT MEASURED: the probe reads the places)
        ranks = torch.where(u["eng0"] > 0, torch.ones_like(ranks), ranks)
    front = files * h
    depth = ranks * v
    lord = 2 * u["radius"]
    return torch.where(single, lord, front), torch.where(single, lord, depth)


def wrap(angle):
    """Radians into (-pi, pi]."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def half_extent(front, depth, phi):
    """Distance from a rectangle's centre to its edge in direction phi (radians from the facing)."""
    s, c = torch.sin(phi).abs(), torch.cos(phi).abs()
    a = (front / 2) / s.clamp(min=1e-6)
    b = (depth / 2) / c.clamp(min=1e-6)
    return torch.minimum(a, b)


def face_length(front, depth, phi):
    """Length of the rectangle's side that faces direction phi: the front (or back) or the flank."""
    s, c = torch.sin(phi).abs(), torch.cos(phi).abs()
    through_front = s * depth <= c * front
    return torch.where(through_front, front, depth)


def _box_point_dist(px, pz, cx, cz, fx, fz, hf, hl):
    """Distance from points (px, pz) to a rectangle centred (cx, cz), facing (fx, fz), hf half its depth along the
    facing and hl half its front across it (0 inside). Shapes broadcast."""
    rx, rz = px - cx, pz - cz
    a = (rx * fx + rz * fz).abs() - hf                  # along the facing
    c = (rx * fz - rz * fx).abs() - hl                  # across it (the front's direction (cos b, -sin b))
    return torch.sqrt(a.clamp(min=0) ** 2 + c.clamp(min=0) ** 2)


def rect_gap(x, z, b, front, depth):
    """[B, N, N] the signed distance between unit i's and unit j's formation rectangles, m: the shortest distance
    between their edges when apart, minus the least depth of their overlap when they overlap (the separating-axis
    test over the four edge normals: exact for rectangles at any bearings and offsets). Head-on and square to each
    other, with their centres on one line, it equals the centre-line gap of pairwise (dist less both half extents)."""
    br = b * DEG
    fx, fz = torch.sin(br), torch.cos(br)               # facing; the front runs along (fz, -fx)
    hf, hl = depth / 2, front / 2
    dx = x[:, None, :] - x[:, :, None]                  # [B, i, j]: j's centre from i's
    dz = z[:, None, :] - z[:, :, None]
    fxi, fzi, fxj, fzj = fx[:, :, None], fz[:, :, None], fx[:, None, :], fz[:, None, :]
    hfi, hli, hfj, hlj = hf[:, :, None], hl[:, :, None], hf[:, None, :], hl[:, None, :]
    C = (fxi * fxj + fzi * fzj).abs()                   # |cos| and |sin| of the bearings' difference
    S = (fxi * fzj - fzi * fxj).abs()
    d_fi = (dx * fxi + dz * fzi).abs()
    d_li = (dx * fzi - dz * fxi).abs()
    d_fj = (dx * fxj + dz * fzj).abs()
    d_lj = (dx * fzj - dz * fxj).abs()
    over = torch.stack((hfi + hfj * C + hlj * S - d_fi,
                        hli + hfj * S + hlj * C - d_li,
                        hfj + hfi * C + hli * S - d_fj,
                        hlj + hfi * S + hli * C - d_lj), 0).min(0).values
    # apart: the shortest distance is from a corner of one to the other (convex polygons)
    sg = torch.tensor((1.0, 1.0, -1.0, -1.0), device=x.device, dtype=x.dtype)
    sl = torch.tensor((1.0, -1.0, 1.0, -1.0), device=x.device, dtype=x.dtype)
    # corners [B, N, 4]: centre + sg hf facing + sl hl front direction (fz, -fx)
    kx = x[..., None] + sg * hf[..., None] * fx[..., None] + sl * hl[..., None] * fz[..., None]
    kz = z[..., None] + sg * hf[..., None] * fz[..., None] - sl * hl[..., None] * fx[..., None]
    # j's corners to i's box [B, i, j, 4], i's corners to j's box
    d_ji = _box_point_dist(kx[:, None, :, :], kz[:, None, :, :], x[:, :, None, None], z[:, :, None, None],
                           fx[:, :, None, None], fz[:, :, None, None], hf[:, :, None, None], hl[:, :, None, None])
    d_ij = _box_point_dist(kx[:, :, None, :], kz[:, :, None, :], x[:, None, :, None], z[:, None, :, None],
                           fx[:, None, :, None], fz[:, None, :, None], hf[:, None, :, None], hl[:, None, :, None])
    apart = torch.minimum(d_ji.min(3).values, d_ij.min(3).values)
    return torch.where(over > 0, -over, apart)


def front_contact(x, z, b, front, depth):
    """[B, N, N] True where formation i meets formation j through i's front or back, False through a flank: of i's two
    axes, the one along which the two rectangles overlap less (or lie farther apart) is the contact's normal (the
    separating-axis test's least penetration, as rect_gap). Two fronts meeting with their centres offset - a unit whose
    front overlaps a neighbour of its enemy by a few metres - touch through the fronts, though the neighbour's centre
    lies off i's flank (the centre-line rule, pairwise rel_i, called that a flank)."""
    br = b * DEG
    fx, fz = torch.sin(br), torch.cos(br)
    hf, hl = depth / 2, front / 2
    dx = x[:, None, :] - x[:, :, None]
    dz = z[:, None, :] - z[:, :, None]
    fxi, fzi, fxj, fzj = fx[:, :, None], fz[:, :, None], fx[:, None, :], fz[:, None, :]
    C = (fxi * fxj + fzi * fzj).abs()
    S = (fxi * fzj - fzi * fxj).abs()
    o_f = hf[:, :, None] + hf[:, None, :] * C + hl[:, None, :] * S - (dx * fxi + dz * fzi).abs()
    o_l = hl[:, :, None] + hf[:, None, :] * S + hl[:, None, :] * C - (dx * fzi - dz * fxi).abs()
    return o_f <= o_l


def pairwise(u, spacing, rect=False, melee_depth=0.0):
    """Everything about unit pairs, each [B, N, N] (row i, column j):
    dist      centre distance, m
    theta     bearing from i to j, radians
    gap       distance between the two formations' edges, m (negative: overlap); rect (config/nn/sim.json
              contact.rect_gap): the rectangles' true signed distance (rect_gap) between two formations, else the
              centre-line gap (dist less each rectangle's half extent along the line between the centres: exact
              head-on, but it misses an overlap of two boxes at an angle or offset whose corners meet off that line);
              a lone man (a circle) keeps the centre-line gap
    reach_j   distance from i's centre to j's edge, m (missile range)
    rel_i     where j is seen from i, radians from i's facing
    rel_j     where i is seen from j, radians from j's facing
    face_i, face_j   length of i's side towards j, of j's side towards i
    enemy     different sides, both present
    melee_depth: dims' (formation.melee_depth: a formation in melee keeps room for its dead)
    """
    front, depth = dims(u, spacing, melee_depth)
    x, z = u["x"], u["z"]
    dx = x[:, None, :] - x[:, :, None]
    dz = z[:, None, :] - z[:, :, None]
    dist = torch.sqrt(dx * dx + dz * dz + 1e-9)
    theta = torch.atan2(dx, dz)
    b = u["b"] * DEG
    rel_i = wrap(theta - b[:, :, None])
    rel_j = wrap(theta + math.pi - b[:, None, :])
    fi, di = front[:, :, None], depth[:, :, None]
    fj, dj = front[:, None, :], depth[:, None, :]
    ext_j = half_extent(fj, dj, rel_j)
    ext = half_extent(fi, di, rel_i) + ext_j
    gap = dist - ext
    if rect:
        single = u["men0"] <= 1
        lone = single[:, :, None] | single[:, None, :]
        gap = torch.where(lone, gap, rect_gap(x, z, u["b"], front, depth))
    side = u["side"]
    enemy = (side[:, :, None] != side[:, None, :]) & (side[:, :, None] > 0) & (side[:, None, :] > 0)
    return {"dist": dist, "theta": theta, "gap": gap, "reach_j": dist - ext_j, "rel_i": rel_i, "rel_j": rel_j,
            "face_i": face_length(fi, di, rel_i), "face_j": face_length(fj, dj, rel_j), "enemy": enemy,
            "front": front, "depth": depth}


def engage_gap(u, pw, combat_m, spacing=1.5):
    """[B, N, N] the edge gap of two formations' boxes at which their outer men are within the game's formed combat
    distance (config/nn/sim.json contact.engage_why): a man strikes an enemy man whose centre is within combat_m (the
    database's entity_action_attack_formed_combat_distance) plus that man's radius; the outer men stand half a man's
    place inside the box's edge (sp_v through the front or back, sp_h through a flank, dims: depth = ranks x v), so the
    boxes' gap is combat_m + the larger radius - the two half places. The side each shows the other: the centre line's
    (face_length's through-front test)."""
    front, depth = pw["front"], pw["depth"]
    if "sp_h" in u:
        h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
        v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    else:
        h = v = torch.full_like(front, float(spacing))

    def half_place(f, d, hh, vv, rel):
        s, c = torch.sin(rel).abs(), torch.cos(rel).abs()
        return torch.where(s * d <= c * f, vv, hh) / 2

    hp_i = half_place(front[:, :, None], depth[:, :, None], h[:, :, None], v[:, :, None], pw["rel_i"])
    hp_j = half_place(front[:, None, :], depth[:, None, :], h[:, None, :], v[:, None, :], pw["rel_j"])
    r = torch.maximum(u["radius"][:, :, None], u["radius"][:, None, :])
    return combat_m + r - hp_i - hp_j


def contact_limit(u, pw, ccal, rules=None, spacing=1.5, engage=True):
    """[B, N, N] the largest edge gap (pw["gap"]) at which units i and j touch: contact.reach_m (a lone man:
    contact.lord_reach_m), + contact.hold_m when either is in melee already (u["m"]); with contact.engage on, two
    formations either of which is in melee touch while their outer men are within the formed combat distance
    (engage_gap; never less than the old hold). rules: the game's battle rules (params.battle); spacing: a man's place
    for a unit without its own (formation.spacing_m); engage False: the old hold whatever contact.engage says (the
    release of contact.engage_release, battle.py)."""
    reach = float(ccal["reach_m"])
    lone = u["men0"] <= 1
    pair_lone = lone[:, :, None] | lone[:, None, :]
    in_melee = u["m"][:, :, None] | u["m"][:, None, :]
    lim = torch.where(pair_lone, torch.full_like(pw["gap"], float(ccal.get("lord_reach_m", reach))),
                      torch.full_like(pw["gap"], reach)) + in_melee.float() * float(ccal["hold_m"])
    if engage and ccal.get("engage"):
        combat_m = float((rules or {}).get("entity_action_attack_formed_combat_distance", 2.5))
        eng = torch.maximum(lim, engage_gap(u, pw, combat_m, spacing))
        lim = torch.where(in_melee & ~pair_lone, eng, lim)
    return lim


def sector(rel, front_deg, rear_deg):
    """0 front, 1 flank, 2 rear, from an angle off the facing (radians)."""
    a = rel.abs()
    return torch.where(a <= front_deg * DEG, 0, torch.where(a >= rear_deg * DEG, 2, 1))


def side4(rel, front_deg, rear_deg):
    """0 front, 1 right flank, 2 rear, 3 left flank, from an angle off the facing (radians, + to the right)."""
    s = sector(rel, front_deg, rear_deg)
    return torch.where((s == 1) & (rel < 0), 3, s)


def blocked_share(u, pw, mask, gx, gz):
    """[B, N]: the largest share of unit i's front that a formation j with mask [B, N, N] (row i, column j) covers
    across i's way to (gx, gz): i's front (pw["front"], centred on its centre, square to its way) against j's box seen
    across that way (its front and depth at j's bearing), their overlap over i's front, 0..1 (0 where mask has none)."""
    wx, wz = gx - u["x"], gz - u["z"]
    n = torch.sqrt(wx * wx + wz * wz).clamp(min=1e-6)
    lx, lz = wz / n, -wx / n                                     # across the way
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]
    off = rx * lx[:, :, None] + rz * lz[:, :, None]              # j's centre across i's way
    b = u["b"] * DEG
    sj, cj = torch.sin(b)[:, None, :], torch.cos(b)[:, None, :]  # j's facing (sin, cos); its front (cos, -sin)
    l_x, l_z = lx[:, :, None], lz[:, :, None]
    ext = (pw["front"][:, None, :] / 2 * (cj * l_x - sj * l_z).abs()
           + pw["depth"][:, None, :] / 2 * (sj * l_x + cj * l_z).abs())
    half = (pw["front"] / 2).clamp(min=1e-6)[:, :, None]
    ov = (torch.minimum(half, off + ext) - torch.maximum(-half, off - ext)).clamp(min=0)
    share = torch.where(mask, (ov / (2 * half)).clamp(max=1.0), torch.zeros_like(ov))
    return share.max(2).values
