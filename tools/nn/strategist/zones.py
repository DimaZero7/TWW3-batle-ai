"""The deployment zones and the strategist's axes (plan.md "Вход", "Перенос в игру").

A zone is a turned rectangle {x, z, b, front, depth} [B] - its centre, its bearing (where its "forward" looks: at the
enemy's zone), its width across (front) and its depth; the same shape as the game's (the battle XML's rectangles, as
build/maps/*/meta.json deployment_zones: x, z, width, depth, bearing_deg - of_meta()).

    axes(mine, theirs, raw, bounds)   frame.Axes: the field's centre; "up" the field's own axis (a multiple of 90 deg)
                                      nearest the bearing from my zone's centre to the enemy zone's - no unit's place
    cells_ok(axes, zone, front, depth, heading)   [B, C*C] bool: a group's zone front x depth turned heading centred
                                      on the cell lies wholly inside my deployment zone
    apart(cx, cz, front, depth, heading, placed, on)   [B, C*C] bool: a group's zone at the cells overlaps no zone
                                      placed already (turned rectangles, separating axes; touching is no overlap)
"""
import torch

from tools.nn.operative import frame as FR
from tools.nn.operative import step as SP
from tools.nn.strategist import cfg

KEYS = ("x", "z", "b", "front", "depth")


def of_meta(z, device="cpu"):
    """A zone of one battle from a build/maps meta.json deployment_zones entry."""
    t = lambda v: torch.tensor([float(v)], device=device)                       # noqa: E731
    return {"x": t(z["x"]), "z": t(z["z"]), "b": t(z["bearing_deg"]), "front": t(z["width"]), "depth": t(z["depth"])}


def take(zone, k):
    return {n: v[k] for n, v in zone.items()}


def bearing(x0, z0, x1, z1):
    return torch.remainder(torch.rad2deg(torch.atan2(x1 - x0, z1 - z0)), 360.0)


def axes(mine, theirs, raw, bounds=None):
    field = FR.field_of(raw, bounds)
    way = bearing(mine["x"], mine["z"], theirs["x"], theirs["z"])
    k = torch.round(torch.remainder(way - field["b"], 360.0) / 90.0)
    heading = torch.remainder(field["b"] + 90.0 * k, 360.0)
    return FR.Axes(field["x"], field["z"], heading, field)


def as_area(zone):
    """{x, z, b, front, depth, on} [B, 1] for tactic/windows.area_layer."""
    out = {k: zone[k][:, None] for k in KEYS}
    out["on"] = torch.ones_like(out["x"], dtype=torch.bool)
    return out


def cells_ok(cx, cz, zone, front, depth, heading):
    """cx, cz [B, M] the cells' centres; front, depth, heading [B]."""
    return SP.field_ok(cx, cz, front, depth, heading, zone)


def radius(front, depth):
    return 0.5 * torch.sqrt(front * front + depth * depth)


def apart(cx, cz, front, depth, heading, placed, on):
    """cx, cz [B, M] the cells; front, depth, heading [B] the zone wanted; placed {cx, cz, front, depth, heading} [B, G]
    the zones placed so far, on [B, G] which. Each zone grown by zone_gap_m / 2 a side."""
    from tools.nn.strategist import deploy as DP
    if not bool(on.any()):
        return torch.ones_like(cx, dtype=torch.bool)
    gap = float(cfg()["zone_gap_m"])
    B, M = cx.shape
    ex = lambda v: v[:, None].expand(B, M)                                       # noqa: E731
    ca = DP.corners(cx, cz, ex(heading), ex(front) + gap, ex(depth) + gap)        # [B, M, 4, 2]
    cb = DP.corners(placed["cx"], placed["cz"], placed["heading"], placed["front"] + gap, placed["depth"] + gap)
    hit = DP._overlap(ca[:, :, None], cb[:, None])                              # [B, M, G]
    return ~(hit & on[:, None, :]).any(2)


def local(zone, x, z):
    """(s, f) of world points [B, ...] in the zone's axes (right, forward)."""
    from tools.nn.unit import zmap as Z
    sh = (-1,) + (1,) * (x.dim() - 1)
    return Z.local(x, z, zone["x"].view(sh), zone["z"].view(sh), zone["b"].view(sh))


def world(zone, s, f):
    from tools.nn.unit import zmap as Z
    sh = (-1,) + (1,) * (s.dim() - 1)
    return Z.world(s, f, zone["x"].view(sh), zone["z"].view(sh), zone["b"].view(sh))
