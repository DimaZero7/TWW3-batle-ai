"""The general's map layers (map_input.py) for many battles at once, on the fly, with torch: for battles whose
deployment zones differ per battle (drills turn and move the whole battle on the map).

A zone is a Rect: (cx, cz, ux, uz, half_f, half_l) - centre, forward unit vector (towards the enemy), half depth
along forward, half width along lateral. The field is the simulator's square (map_input.arena_field), the grid
128 x 128 cells of 16 m centred on the map's centre (0, 0), as map_input. Every rule is exact for rectangles:
* zone flag: the cell's 4 corners lie in the rectangle (a convex shape holds the cell then);
* vanguard flag: the shortest distance between the cell and the other side's rectangle >= band (between two
  convex shapes apart it runs from a corner of one to the other; a cell and a rectangle wider than the cell's
  diagonal never cross without one holding a corner of the other);
* both flags only in cells wholly in the field. Side 2's layers are half-turned and swapped (map_input.unpack).

Zones of a battle (zones_of): an army description (tools/nn/train/drills: sides -> units with x, z) gets
* the arena's <deployment_area> squares (map_input.arena_rects) when its side's start points lie in one of
  them (generated battles, scenes, embedded drills, also half-turned);
* else (a drill turned and moved) the rectangle in the side's axes (origin its units' centroid, forward towards
  the enemy's centroid, as drills.axes) around its units' start points, ZONE_FRONT_M (40 m, as far as the
  arena's zone reaches in front of the front line) wider on every side.

    layers(rects [B, 2, 6], sides [B]) -> [B, 6, 128, 128]
"""
import functools
import math

import numpy as np
import torch

from tools.nn.general import map_input as mi

EPS = {torch.float32: 1e-3, torch.float64: 1e-6}   # m, tolerance on edges: float32 at ~1000 m keeps ~1e-4 m


def zone_rect(points, enemy_points, margin):
    """Rect around points (x, z) in the axes of their centroid towards the enemy's centroid, margin wider."""
    p = np.asarray(points, float)
    e = np.asarray(enemy_points, float)
    o, c = p.mean(0), e.mean(0)
    d = c - o
    n = max(1e-6, float(np.hypot(*d)))
    ux, uz = d / n
    f = (p[:, 0] - o[0]) * ux + (p[:, 1] - o[1]) * uz
    lat = (p[:, 0] - o[0]) * uz - (p[:, 1] - o[1]) * ux
    fm, lm = (f.max() + f.min()) / 2, (lat.max() + lat.min()) / 2
    cx = o[0] + fm * ux + lm * uz
    cz = o[1] + fm * uz - lm * ux
    return (float(cx), float(cz), float(ux), float(uz), float((f.max() - f.min()) / 2 + margin),
            float((lat.max() - lat.min()) / 2 + margin))


def _holds(rect, pts):
    cx, cz, ux, uz, hf, hl = rect
    dx, dz = pts[:, 0] - cx, pts[:, 1] - cz
    f, lat = dx * ux + dz * uz, dx * uz - dz * ux
    return bool(((np.abs(f) <= hf + 1e-6) & (np.abs(lat) <= hl + 1e-6)).all())


@functools.lru_cache(maxsize=1)
def _default_squares():
    from tools.nn import scenario
    return list(mi.arena_rects(scenario.load_arena()).values())


def zones_of(desc, arena=None):
    """{1: Rect, 2: Rect} of an army description (module doc). arena: an arena dict (tools.nn.scenario
    load_arena) whose squares are tried; default config/nn/arena.json."""
    from tools.nn.armies.place import ZONE_FRONT_M
    squares = list(mi.arena_rects(arena).values()) if arena else _default_squares()
    pts = {s: np.array([(u["x"], u["z"]) for u in desc["sides"][s]["units"]], float) for s in (1, 2)}
    out = {}
    for s in (1, 2):
        fit = [r for r in squares if _holds(r, pts[s])]
        out[s] = fit[0] if fit else zone_rect(pts[s], pts[3 - s], ZONE_FRONT_M)
    return out


def rects_tensor(zones, device="cpu", dtype=torch.float32):
    """[B, 2, 6] from a list of zones_of() dicts."""
    return torch.tensor([[z[1], z[2]] for z in zones], device=device, dtype=dtype)


def _point_rect_dist(px, pz, r):
    """Distance from points [B, ...] to rectangles r [B, 6] (0 inside)."""
    shape = (-1,) + (1,) * (px.dim() - 1)
    cx, cz, ux, uz, hf, hl = (r[:, k].reshape(shape) for k in range(6))
    dx, dz = px - cx, pz - cz
    f = (dx * ux + dz * uz).abs() - hf
    lat = (dx * uz - dz * ux).abs() - hl
    return torch.hypot(f.clamp(min=0), lat.clamp(min=0)), (f <= EPS[f.dtype]) & (lat <= EPS[f.dtype])


def _rect_corners(r):
    """[B, 4, 2] corners of rectangles [B, 6]."""
    cx, cz, ux, uz, hf, hl = r.unbind(-1)
    out = []
    for sf, sl in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append(torch.stack([cx + sf * hf * ux + sl * hl * uz, cz + sf * hf * uz - sl * hl * ux], -1))
    return torch.stack(out, 1)


def _side_flags(zone, other, xs, zs, band, full):
    """(zone flag, vanguard flag) [B, n, n] of one side: zone its rectangle, other the enemy's."""
    n = xs.numel() - 1
    PX, PZ = xs.view(1, 1, -1), zs.view(1, -1, 1)               # grid points [1, n+1, n+1] (rows z, cols x)
    _, inside = _point_rect_dist(PX.expand(zone.shape[0], n + 1, n + 1), PZ.expand(zone.shape[0], n + 1, n + 1), zone)
    whole = inside[:, :-1, :-1] & inside[:, 1:, :-1] & inside[:, :-1, 1:] & inside[:, 1:, 1:]
    d, _ = _point_rect_dist(PX.expand(other.shape[0], n + 1, n + 1), PZ.expand(other.shape[0], n + 1, n + 1), other)
    dc = torch.minimum(torch.minimum(d[:, :-1, :-1], d[:, 1:, :-1]), torch.minimum(d[:, :-1, 1:], d[:, 1:, 1:]))
    rc = _rect_corners(other)                                    # [B, 4, 2]
    xa, xb, za, zb = xs[:-1], xs[1:], zs[:-1], zs[1:]
    px, pz = rc[..., 0].unsqueeze(-1), rc[..., 1].unsqueeze(-1)  # [B, 4, 1]
    dx = torch.clamp(torch.maximum(xa - px, px - xb), min=0)     # [B, 4, n] along x
    dz = torch.clamp(torch.maximum(za - pz, pz - zb), min=0)     # [B, 4, n] along z
    dr = torch.hypot(dz.unsqueeze(-1), dx.unsqueeze(-2)).amin(1)  # [B, n, n]
    clear = torch.minimum(dc, dr) >= band - EPS[dc.dtype]
    return whole & full, clear & full


def layers(rects, sides, field=None, band=mi.BAND_M, chunk=256):
    """[B, 6, 128, 128] (dtype and device of rects [B, 2, 6]) for the side each battle's network plays
    (sides [B] of 1 / 2): field, outside, our zone, enemy zone, our vanguard, enemy vanguard."""
    assert float(torch.minimum(rects[..., 4], rects[..., 5]).min()) * 2 > mi.CELL_M * math.sqrt(2), \
        "a zone narrower than a cell's diagonal"
    dev, dt = rects.device, rects.dtype
    field = field or mi.arena_field()
    cx, cz = (field[0] + field[1]) / 2, (field[2] + field[3]) / 2
    e = torch.arange(mi.GRID + 1, device=dev, dtype=dt) * mi.CELL_M
    xs, zs = cx - mi.SIZE_M / 2 + e, cz - mi.SIZE_M / 2 + e
    ox = (torch.minimum(xs[1:], torch.tensor(field[1], device=dev, dtype=dt))
          - torch.maximum(xs[:-1], torch.tensor(field[0], device=dev, dtype=dt))).clamp(min=0) / mi.CELL_M
    oz = (torch.minimum(zs[1:], torch.tensor(field[3], device=dev, dtype=dt))
          - torch.maximum(zs[:-1], torch.tensor(field[2], device=dev, dtype=dt))).clamp(min=0) / mi.CELL_M
    share = oz.unsqueeze(-1) * ox.unsqueeze(0)                   # [n, n]
    full = share >= 1 - 1e-6
    sides = torch.as_tensor(sides, device=dev).view(-1)
    out = []
    for i in range(0, rects.shape[0], chunk):
        r = rects[i:i + chunk]
        z1, v1 = _side_flags(r[:, 0], r[:, 1], xs, zs, band, full)
        z2, v2 = _side_flags(r[:, 1], r[:, 0], xs, zs, band, full)
        s = share.expand(r.shape[0], -1, -1)
        lay = torch.stack([s, 1 - s, z1.to(dt), z2.to(dt), v1.to(dt), v2.to(dt)], 1)
        two = (sides[i:i + chunk] == 2).view(-1, 1, 1, 1)
        turned = lay.flip(-1, -2)[:, [0, 1, 3, 2, 5, 4]]
        out.append(torch.where(two, turned, lay))
    return torch.cat(out)


def of_battles(descs, sides, device="cpu", dtype=torch.float32, arena=None):
    """layers() of army descriptions (zones_of each)."""
    return layers(rects_tensor([zones_of(d, arena) for d in descs], device, dtype), sides)
