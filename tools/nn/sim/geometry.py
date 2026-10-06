"""Where units are relative to each other: formation size, pairwise distances, sides, contact.

A unit is a rectangle: `width` m of front ordered (fixed at the start), files = floor(width / h) men h apart
and as many ranks as its men fill, v apart (h, v: the unit's formation template in the game's database,
config/nn/units.json spacing; config/nn/sim.json formation.spacing_m where a unit has none); a lord is a
circle of his radius. Bearing b in degrees: facing (sin b, cos b) in (x, z).
"""
import math

import torch

DEG = math.pi / 180


def dims(u, spacing):
    """(front, depth) [B, N], m: files = floor(ordered width / h) (at least 1, at most the men), front = files x h,
    depth = ceil(men / files) x v (the game's own formations, data/roster: front (files - 1) h, depth (ranks - 1)
    v, within 1 % / 10 % at 15-40 m; build/meleecore/spacing_check.py). spacing: h and v of a unit without its own."""
    men = u["men"].clamp(min=0)
    single = u["men0"] <= 1
    if "sp_h" in u:
        h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
        v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    else:
        h = v = torch.full_like(men, float(spacing))
    files = torch.clamp(torch.floor(u["width"] / h + 1e-4), min=1)
    files = torch.minimum(files, men.clamp(min=1))
    ranks = torch.ceil(men.clamp(min=1) / files)
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


def pairwise(u, spacing):
    """Everything about unit pairs, each [B, N, N] (row i, column j):
    dist      centre distance, m
    theta     bearing from i to j, radians
    gap       distance between the two formations' edges, m (negative: overlap)
    reach_j   distance from i's centre to j's edge, m (missile range)
    rel_i     where j is seen from i, radians from i's facing
    rel_j     where i is seen from j, radians from j's facing
    face_i, face_j   length of i's side towards j, of j's side towards i
    enemy     different sides, both present
    """
    front, depth = dims(u, spacing)
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
    side = u["side"]
    enemy = (side[:, :, None] != side[:, None, :]) & (side[:, :, None] > 0) & (side[:, None, :] > 0)
    return {"dist": dist, "theta": theta, "gap": dist - ext, "reach_j": dist - ext_j, "rel_i": rel_i, "rel_j": rel_j,
            "face_i": face_length(fi, di, rel_i), "face_j": face_length(fj, dj, rel_j), "enemy": enemy,
            "front": front, "depth": depth}


def sector(rel, front_deg, rear_deg):
    """0 front, 1 flank, 2 rear, from an angle off the facing (radians)."""
    a = rel.abs()
    return torch.where(a <= front_deg * DEG, 0, torch.where(a >= rear_deg * DEG, 2, 1))
