"""Cells of a square grid against a shape (a simple polygon in metres), exactly, no sampling.

A grid: `cells(x0, z0, n, cell_m)` -> arrays of every cell's bounds, row-major [iz, ix]
(ix along +x, iz along +z; cell (0, 0) has its low corner at (x0, z0)).

* share(poly)          the part of each cell's area inside the polygon (0..1): a cell the
                       boundary crosses is clipped by the polygon (Sutherland-Hodgman against
                       the cell's square), its area by the shoelace sum; other cells are 0 or 1.
* wholly_inside(poly)  the cell lies entirely in the polygon (its edges may touch the boundary).
* clear_of(poly, r)    the cell lies entirely farther than r from the polygon (and not in it):
                       the shortest cell-to-boundary distance is >= r.

Shapes: rect(), rounded_rect() (the game's deployment zones are rectangles with rounded corners).
Plain numpy.
"""
import math

import numpy as np

EPS = 1e-6


def rect(cx, cz, width, height, orient=0.0):
    """Corners [4, 2] of a width x height rectangle centred at (cx, cz), turned by orient radians
    (width along x at orient 0)."""
    return rounded_rect(cx, cz, width, height, 0.0, orient)


def rect_corners(cx, cz, ux, uz, half_f, half_l):
    """Corners [4, 2] (counter-clockwise) of a Rect: centre (cx, cz), forward unit (ux, uz), half depth along
    forward, half width along lateral (right of forward: (uz, -ux)). The Rect params are what
    map_batch.py takes per zone."""
    vx, vz = uz, -ux
    pts = np.array([[cx + sf * half_f * ux + sl * half_l * vx, cz + sf * half_f * uz + sl * half_l * vz]
                    for sf, sl in ((1, 1), (1, -1), (-1, -1), (-1, 1))], float)
    ang = np.arctan2(pts[:, 1] - cz, pts[:, 0] - cx)
    return pts[np.argsort(ang)]


def rounded_rect(cx, cz, width, height, radius, orient=0.0, arc_step_deg=5.0):
    """Vertices [M, 2] of a rectangle with corners rounded by `radius` (arcs as chords every arc_step_deg)."""
    w, h = width / 2.0, height / 2.0
    r = min(max(radius, 0.0), w, h)
    pts = []
    for (sx, sz, a0) in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        ccx, ccz = sx * (w - r), sz * (h - r)
        if r == 0.0:
            pts.append((ccx, ccz))
            continue
        steps = max(1, int(round(90.0 / arc_step_deg)))
        for k in range(steps + 1):
            a = math.radians(a0 + 90.0 * k / steps)
            pts.append((ccx + r * math.cos(a), ccz + r * math.sin(a)))
    p = np.asarray(pts, float)
    c, s = math.cos(orient), math.sin(orient)
    out = np.empty_like(p)
    out[:, 0] = cx + p[:, 0] * c - p[:, 1] * s
    out[:, 1] = cz + p[:, 0] * s + p[:, 1] * c
    return _dedupe(out)


def _dedupe(p):
    keep = np.ones(len(p), bool)
    keep[1:] = np.hypot(*(p[1:] - p[:-1]).T) > EPS
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) <= EPS:
        keep[-1] = False
    return p[keep]


def area(poly):
    """Signed-free area of a simple polygon [M, 2]."""
    x, z = poly[:, 0], poly[:, 1]
    return abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))) / 2.0


def cells(x0, z0, n, cell_m):
    """(xa, xb, za, zb) [n, n] bounds of each cell, row-major [iz, ix]."""
    e = np.arange(n, dtype=float) * cell_m
    xa = np.broadcast_to(x0 + e[None, :], (n, n))
    za = np.broadcast_to(z0 + e[:, None], (n, n))
    return xa, xa + cell_m, za, za + cell_m


def _segments(poly):
    a = np.asarray(poly, float)
    return a, np.roll(a, -1, axis=0)


def inside(poly, x, z):
    """Even-odd test: points (x, z) (any equal shapes) inside the polygon."""
    a, b = _segments(poly)
    x = np.asarray(x, float)[..., None]
    z = np.asarray(z, float)[..., None]
    ax, az, bx, bz = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    crosses = (az > z) != (bz > z)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = ax + (z - az) * (bx - ax) / (bz - az)
    return (crosses & (x < xi)).sum(-1) % 2 == 1


def _seg_hits_box(ax, az, bx, bz, xa, xb, za, zb):
    """Liang-Barsky: the segment meets the closed box. Arrays broadcast ([N, 1] boxes, [M] segments)."""
    dx, dz = bx - ax, bz - az
    t0 = np.zeros(np.broadcast(ax, xa).shape)
    t1 = np.ones_like(t0)
    ok = np.ones(t0.shape, bool)
    for p, q in ((-dx, ax - xa), (dx, xb - ax), (-dz, az - za), (dz, zb - az)):
        p = np.broadcast_to(p, t0.shape)
        q = np.broadcast_to(q, t0.shape)
        par = np.abs(p) < 1e-12
        ok &= ~(par & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = q / p
        t0 = np.where(~par & (p < 0), np.maximum(t0, t), t0)
        t1 = np.where(~par & (p > 0), np.minimum(t1, t), t1)
    return ok & (t0 <= t1)


def _point_box_dist(px, pz, xa, xb, za, zb):
    dx = np.maximum(np.maximum(xa - px, px - xb), 0.0)
    dz = np.maximum(np.maximum(za - pz, pz - zb), 0.0)
    return np.hypot(dx, dz)


def _point_seg_dist(px, pz, ax, az, bx, bz):
    dx, dz = bx - ax, bz - az
    ll = dx * dx + dz * dz
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ll > 0, ((px - ax) * dx + (pz - az) * dz) / ll, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), pz - (az + t * dz))


def _flat(grid):
    return [np.asarray(g, float).reshape(-1, 1) for g in grid]


def crossed(poly, grid):
    """[n, n] the polygon's boundary passes through the cell's open interior."""
    xa, xb, za, zb = _flat(grid)
    a, b = _segments(poly)
    hit = _seg_hits_box(a[:, 0], a[:, 1], b[:, 0], b[:, 1], xa + EPS, xb - EPS, za + EPS, zb - EPS)
    return hit.any(-1).reshape(np.shape(grid[0]))


def boundary_distance(poly, grid):
    """[n, n] the shortest distance from each cell (closed square) to the polygon's boundary."""
    xa, xb, za, zb = _flat(grid)
    a, b = _segments(poly)
    ax, az, bx, bz = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    d = np.minimum(_point_box_dist(ax, az, xa, xb, za, zb), _point_box_dist(bx, bz, xa, xb, za, zb))
    for cx, cz in ((xa, za), (xb, za), (xa, zb), (xb, zb)):
        d = np.minimum(d, _point_seg_dist(cx, cz, ax, az, bx, bz))
    d = np.where(_seg_hits_box(ax, az, bx, bz, xa, xb, za, zb), 0.0, d)
    return d.min(-1).reshape(np.shape(grid[0]))


def _centres(grid):
    xa, xb, za, zb = grid
    return (np.asarray(xa) + np.asarray(xb)) / 2, (np.asarray(za) + np.asarray(zb)) / 2


def wholly_inside(poly, grid):
    """[n, n] bool: the cell lies entirely in the polygon."""
    cx, cz = _centres(grid)
    return inside(poly, cx, cz) & ~crossed(poly, grid)


def clear_of(poly, r, grid):
    """[n, n] bool: every point of the cell is farther than r (or exactly r) from the polygon."""
    cx, cz = _centres(grid)
    return ~inside(poly, cx, cz) & (boundary_distance(poly, grid) >= r - EPS)


def _clip(poly, xa, xb, za, zb):
    """Sutherland-Hodgman: the polygon clipped by the box (a list of vertices, maybe empty)."""
    out = [tuple(p) for p in poly]
    for axis, lim, keep_ge in ((0, xa, True), (0, xb, False), (1, za, True), (1, zb, False)):
        if not out:
            break
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            cin = cur[axis] >= lim if keep_ge else cur[axis] <= lim
            pin = prev[axis] >= lim if keep_ge else prev[axis] <= lim
            if cin != pin:
                t = (lim - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
    return out


def share(poly, grid):
    """[n, n] float: the part of each cell's area inside the polygon."""
    xa, xb, za, zb = (np.asarray(g, float) for g in grid)
    cx, cz = _centres(grid)
    out = inside(poly, cx, cz).astype(float)
    cut = crossed(poly, grid)
    for iz, ix in zip(*np.nonzero(cut)):
        box = (xa[iz, ix], xb[iz, ix], za[iz, ix], zb[iz, ix])
        clipped = _clip(np.asarray(poly, float), *box)
        whole = (box[1] - box[0]) * (box[3] - box[2])
        out[iz, ix] = area(np.asarray(clipped)) / whole if len(clipped) >= 3 else 0.0
    return np.clip(out, 0.0, 1.0)


def share_box(box, grid):
    """[n, n] float: the part of each cell inside an axis-aligned box (xmin, xmax, zmin, zmax): the
    overlap of the intervals along x times along z."""
    xa, xb, za, zb = (np.asarray(g, float) for g in grid)
    bx0, bx1, bz0, bz1 = box
    ox = np.clip(np.minimum(xb, bx1) - np.maximum(xa, bx0), 0.0, None) / (xb - xa)
    oz = np.clip(np.minimum(zb, bz1) - np.maximum(za, bz0), 0.0, None) / (zb - za)
    return ox * oz
