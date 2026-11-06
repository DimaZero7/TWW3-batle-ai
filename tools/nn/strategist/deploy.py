"""The deployment by code (plan.md "Расстановка — кодом", "Перенос в игру"): the groups' zones -> every unit's place.

A group's zone is the tactician's tiles (tactic/zone.grid of n and the form, tile_m = 80 m) centred on the chosen cell,
turned by the chosen sector. The melee units fill the rows of tiles from the front, the shooters after them (each
kind by slot); a row's units are centred in it; a unit stands on its place's centre facing the zone's turn. The result
is per unit exactly the game's teleport_to_location(position, bearing, width_m): x, z, the bearing rounded to the
engine's 64 sectors (docs/ru/game/units/commands.md), the width - the unit's own frontage, at most a tile less
gap_m (the neighbours never overlap).

    places(members, shooter, cx, cz, heading, line, width0)   {x, z, b, width, on} [B, N]
    errors(st, own, zone, spacing)    [B] long: the deployment's errors on the built state - an own unit in no group or
                                      in two, a unit's formation (geometry.dims) not wholly inside my deployment zone,
                                      two own formations overlapping (separating axes)
"""
import torch

from tools.nn.operative.groups import grid
from tools.nn.tactic import cfg as tcfg
from tools.nn.unit import zmap as Z

SECTORS = 64              # the engine keeps a unit's bearing in 64 sectors
GAP_M = 6.0               # tools/nn/armies/place.py LATERAL_GAP_M: between neighbours in a row


def round_bearing(b):
    step = 360.0 / SECTORS
    return torch.remainder(torch.round(b / step) * step, 360.0)


def places(members, shooter, cx, cz, heading, line, width0):
    """members [B, G, N] bool; shooter [B, N]; cx, cz, heading [B, G]; line [B, G] bool; width0 [B, N] the units' own
    frontage (m)."""
    B, G, N = members.shape
    dev = members.device
    tile = float(tcfg()["tile_m"])
    n = members.sum(2)
    rows, cols = grid(n, line)
    slot = torch.arange(N, device=dev, dtype=torch.float32)
    key = torch.where(members, shooter[:, None, :].float() * 1000.0 + slot, torch.full((B, G, N), 1e9, device=dev))
    rank = key.argsort(2).argsort(2)                                     # [B, G, N] the place in the group's order
    r = torch.div(rank, cols[..., None], rounding_mode="floor")
    i = rank % cols[..., None]
    k = torch.minimum(cols[..., None], n[..., None] - r * cols[..., None])
    s = (i.float() + 0.5 - k.float() / 2) * tile
    f = (rows[..., None].float() / 2 - r.float() - 0.5) * tile
    x, z = Z.world(s, f, cx[..., None], cz[..., None], heading[..., None])
    on = members.any(1)
    g = members.float().argmax(1)                                        # [B, N] each unit's group
    pick = lambda v: v.gather(1, g[:, None, :])[:, 0]                    # noqa: E731
    b = round_bearing(heading.gather(1, g))
    return {"x": torch.where(on, pick(x), torch.zeros(B, N, device=dev)),
            "z": torch.where(on, pick(z), torch.zeros(B, N, device=dev)), "b": torch.where(on, b, torch.zeros_like(b)),
            "width": torch.where(on, width0.clamp(max=tile - GAP_M), width0), "on": on, "group": torch.where(on, g, -1)}


def corners(x, z, b, front, depth):
    """[..., 4, 2] a turned rectangle's corners."""
    out = []
    for ss, ff in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        cx, cz = Z.world(ss * front / 2, ff * depth / 2, x, z, b)
        out.append(torch.stack([cx, cz], -1))
    return torch.stack(out, -2)


def _overlap(ca, cb):
    """[...] bool: two convex quads [..., 4, 2] overlap (separating axes, a touch is no overlap)."""
    sep = torch.zeros(ca.shape[:-2], dtype=torch.bool, device=ca.device)
    for c in (ca, cb):
        for e in range(2):
            d = c[..., e + 1, :] - c[..., e, :]
            nrm = torch.stack([-d[..., 1], d[..., 0]], -1)
            pa = (ca * nrm[..., None, :]).sum(-1)
            pb = (cb * nrm[..., None, :]).sum(-1)
            sep = sep | (pa.amax(-1) <= pb.amin(-1) + 1e-3) | (pb.amax(-1) <= pa.amin(-1) + 1e-3)
    return ~sep


def errors(st, own, member, zone, spacing):
    """[B] long (module doc). member [B, N] long each slot's group (-1 none)."""
    from tools.nn.sim import geometry
    u = st.u
    front, depth = geometry.dims(u, spacing)
    B, N = own.shape
    err = (own & (member < 0)).sum(1)
    cs = corners(u["x"], u["z"], u["b"], front, depth)                  # [B, N, 4, 2]
    s, f = Z.local(cs[..., 0], cs[..., 1], zone["x"][:, None, None], zone["z"][:, None, None], zone["b"][:, None, None])
    out = (s.abs() > zone["front"][:, None, None] / 2 + 1e-3) | (f.abs() > zone["depth"][:, None, None] / 2 + 1e-3)
    err = err + (out.any(2) & own).sum(1)
    for i in range(N):
        for j in range(i + 1, N):
            both = own[:, i] & own[:, j]
            if bool(both.any()):
                err = err + (_overlap(cs[:, i], cs[:, j]) & both).long()
    return err
