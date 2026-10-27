"""The zone map: one builder for every level (build/diag_g49/plan_unit_net.md section 2; unit_base.md: the same format
on every level, a different scale).

A map is CELLS x CELLS cells of cell_m metres round a centre, turned so that the frame's heading points up: row 0 is
the farthest forward, column 0 the farthest left. A unit's map: the centre is the unit, the heading its front, so an
enemy straight ahead is at the top whatever way the unit faces; a general's map: the same builder with other numbers.
The cell size is an input of the network too (the shapes are the same at every scale, the patterns are not).

Layers (numbers fixed, new ones go last):
    0 field      share of the cell on the map (inside the square map's edge)
    1 off_map    share off it (1 - field)
    2 obstacle   share under an obstacle (the simulator has none yet: zeros)
    3 enemy      enemies' density 0..1: the share of the cell under a formation x the share of its places filled
                 (men / (files x ranks)); a few men clipping a corner give ~0.05, not 1
    4 enemy_cos  cos of the enemy's front relative to the frame's heading (0: the same way, 180: facing the frame),
                 x its density; two in one cell: their density-weighted mean x the density
    5 enemy_sin  sin of it (positive: turned clockwise, to the right)
    6 own        own units' density (not the frame's own unit)
    7 own_cos    their front's cos (as 4)
    8 own_sin    their front's sin
    9 self       the frame's own unit's density (its front is up: no direction layers)
   10 area       share of the cell inside my area (a turned rectangle; zeros without one)

A formation is a turned rectangle (front x depth, its bearing); every cell it covers is painted by the exact share of
the cell's area inside it (Green's theorem: the integral of x dy over the parts of the rectangle's edges inside the
cell and of the cell's edges inside the rectangle; no sampling, so a sliver is never lost and the share does not
depend on the cell size). Exact numbers (angle, front, depth) belong in the unit's row of the list, not on the map.
Bearings are the simulator's: degrees clockwise from +z, facing (sin b, cos b) in (x, z). Torch, batched over R maps:
run it on the GPU for many units.
"""
import math

import torch

CELLS = 30
LAYERS = ("field", "off_map", "obstacle", "enemy", "enemy_cos", "enemy_sin", "own", "own_cos", "own_sin", "self",
          "area")
L = {name: i for i, name in enumerate(LAYERS)}
NONE, ENEMY, OWN, SELF = 0, 1, 2, 3          # a unit's role on the map
DIRECTION = (L["enemy_cos"], L["enemy_sin"], L["own_cos"], L["own_sin"])
DEG = math.pi / 180
# a rectangle's centre is moved this far (m) before the area sums: an edge lying exactly on a cell's edge would count
# twice or not at all (the share's error is at most JITTER x its perimeter)
JITTER = (7.1e-4, 3.1e-4)


def local(x, z, cx, cz, heading):
    """World (x, z) -> the frame's (s right, f forward), m. heading: degrees. Shapes broadcast."""
    h = torch.as_tensor(heading, dtype=x.dtype, device=x.device) * DEG
    dx, dz = x - cx, z - cz
    c, s = torch.cos(h), torch.sin(h)
    return dx * c - dz * s, dx * s + dz * c


def world(s, f, cx, cz, heading):
    """The frame's (s right, f forward) -> world (x, z)."""
    h = torch.as_tensor(heading, dtype=s.dtype, device=s.device) * DEG
    c, sn = torch.cos(h), torch.sin(h)
    return cx + s * c + f * sn, cz - s * sn + f * c


def cell_centre(row, col, cell_m, cells=CELLS):
    """(s, f) of a cell's centre in the frame, m."""
    return (col + 0.5 - cells / 2) * cell_m, (cells / 2 - row - 0.5) * cell_m


def cell_of(s, f, cell_m, cells=CELLS):
    """(row, col) of the cell a frame point lies in (long; outside the map: < 0 or >= cells)."""
    col = torch.floor(s / cell_m + cells / 2).long()
    row = torch.floor(cells / 2 - f / cell_m).long()
    return row, col


def rects_of(u, spacing, melee_depth=0.0):
    """A simulator state's units (u: tools/nn/sim/state.py) as the map's rectangles: x, z, b, front, depth [B, N] and
    fill = men / (files x ranks), the share of the box's places filled (a lord: 1). Roles are the caller's."""
    from tools.nn.sim import geometry
    front, depth = geometry.dims(u, spacing, melee_depth)
    if "sp_h" in u:
        h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
        v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    else:
        h = v = torch.full_like(front, float(spacing))
    places = (front / h).round() * (depth / v).round()
    fill = (u["men"].clamp(min=0) / places.clamp(min=1)).clamp(max=1)
    fill = torch.where(u["men0"] <= 1, (u["men"] > 0).float(), fill)
    return {"x": u["x"], "z": u["z"], "b": u["b"], "front": front, "depth": depth, "fill": fill}


def _clip(t0, t1, p, q):
    """Liang-Barsky: the segment's parameter range [t0, t1] cut by p t <= q (an empty range: t1 < t0)."""
    par = p.abs() < 1e-9
    r = q / torch.where(par, torch.ones_like(p), p)
    t0 = torch.where(~par & (p < 0), torch.maximum(t0, r), t0)
    t1 = torch.where(~par & (p > 0), torch.minimum(t1, r), t1)
    return t0, torch.where(par & (q < 0), torch.full_like(t1, -1.0), t1)


def _cover(rs, rf, a, front, depth, cell_m, cells):
    """[R, K, C, C] the exact share of each cell inside each rectangle: centres (rs, rf) [R, K] in the frame (s right,
    f forward), a [R, K] its front's angle from the frame's heading (radians, clockwise), front x depth [R, K]."""
    R, K = rs.shape
    dev = rs.device
    rs, rf = rs + JITTER[0], rf + JITTER[1]
    k = torch.arange(cells + 1, device=dev, dtype=torch.float32)
    xe = (k - cells / 2)[None] * cell_m[:, None]              # [R, C+1] column edges, left to right
    ye = (cells / 2 - k)[None] * cell_m[:, None]              # [R, C+1] row edges, top (forward) down
    X0, X1 = xe[:, None, None, :-1], xe[:, None, None, 1:]    # [R, 1, 1, C]
    Yt, Yb = ye[:, None, :-1, None], ye[:, None, 1:, None]    # [R, 1, C, 1]
    Xc = (X0 + X1) / 2                                        # sums about each cell's centre (less cancellation)
    ux, uy = torch.sin(a), torch.cos(a)                        # the front's direction in (s, f)
    vx, vy = uy, -ux                                           # its right
    hd, hw = depth / 2, front / 2
    corner = [(rs + sd * hd * ux + sw * hw * vx, rf + sd * hd * uy + sw * hw * vy)
              for sd, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]     # counter-clockwise in (s, f)
    area = torch.zeros((R, K, cells, cells), device=dev)
    zeros = torch.zeros((R, K, cells, cells), device=dev)
    for (x0, y0), (x1, y1) in zip(corner, corner[1:] + corner[:1]):
        # the rectangle's edge, its part inside each cell
        x0_, y0_ = x0[..., None, None], y0[..., None, None]
        dx, dy = (x1 - x0)[..., None, None], (y1 - y0)[..., None, None]
        t0, t1 = zeros, zeros + 1.0
        t0, t1 = _clip(t0, t1, -dx, x0_ - X0)
        t0, t1 = _clip(t0, t1, dx, X1 - x0_)
        t0, t1 = _clip(t0, t1, -dy, y0_ - Yb)
        t0, t1 = _clip(t0, t1, dy, Yt - y0_)
        on = t1 > t0
        area = area + torch.where(on, dy * ((x0_ - Xc) * (t1 - t0) + dx * (t1 * t1 - t0 * t0) / 2), zeros)
    # the cells' vertical edges (the horizontal ones add nothing to x dy), their part inside the rectangle: an upward
    # segment at s = xe from Yb to Yt, in the rectangle's frame across = (q . v), along = (q . u)
    col = xe[:, None, None, :]                                 # [R, 1, 1, C+1]
    yb = ye[:, None, 1:, None]                                 # [R, 1, C, 1]
    L_ = cell_m[:, None, None, None]
    qx, qy = col - rs[..., None, None], yb - rf[..., None, None]
    vx_, vy_, ux_, uy_ = (t[..., None, None] for t in (vx, vy, ux, uy))
    acr0, acr_d = qx * vx_ + qy * vy_, L_ * vy_
    alo0, alo_d = qx * ux_ + qy * uy_, L_ * uy_
    hw_, hd_ = hw[..., None, None], hd[..., None, None]
    t0 = torch.zeros((R, K, cells, cells + 1), device=dev)
    t1 = t0 + 1.0
    t0, t1 = _clip(t0, t1, -acr_d.expand_as(t0), (acr0 + hw_).expand_as(t0))
    t0, t1 = _clip(t0, t1, acr_d.expand_as(t0), (hw_ - acr0).expand_as(t0))
    t0, t1 = _clip(t0, t1, -alo_d.expand_as(t0), (alo0 + hd_).expand_as(t0))
    t0, t1 = _clip(t0, t1, alo_d.expand_as(t0), (hd_ - alo0).expand_as(t0))
    frac = (t1 - t0).clamp(min=0)                              # [R, K, C, C+1]
    half = cell_m[:, None, None, None] / 2
    # the right edge goes up (+L), the left one down (-L): x - Xc = +half / -half
    area = area + half * L_ * frac[..., 1:] + half * L_ * frac[..., :-1]
    return (area / (cell_m[:, None, None, None] ** 2)).clamp(0.0, 1.0)


def build(cx, cz, heading, cell_m, units=None, area=None, bounds=None, cells=CELLS, chunk=None):
    """[R, len(LAYERS), cells, cells] float maps.

    cx, cz, heading (degrees), cell_m: [R] the frames. units: {x, z, b, front, depth, fill, role} [R, K] (role NONE:
    an empty slot). area: {x, z, b, front, depth} [R] my area (a turned rectangle; front across its bearing) and
    optionally on [R] bool, or None. bounds: the square map's half-size (a number or [R]; the map is centred on 0, as
    State.bounds), None: no edge. chunk: maps per pass (memory: ~20 x R x K x cells^2 numbers)."""
    R = cx.shape[0]
    dev = cx.device
    K = 0 if units is None else units["x"].shape[1]
    chunk = chunk or max(1, 16384 // max(K, 1))
    if R > chunk:
        def part(v, i):
            return v[i:i + chunk] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == R else v
        return torch.cat([build(part(cx, i), part(cz, i), part(heading, i), part(cell_m, i),
                                None if units is None else {k: part(v, i) for k, v in units.items()},
                                None if area is None else {k: part(v, i) for k, v in area.items()},
                                part(bounds, i), cells, chunk) for i in range(0, R, chunk)])
    out = torch.zeros((R, len(LAYERS), cells, cells), device=dev)
    if bounds is None:
        out[:, L["field"]] = 1.0
    else:
        bd = torch.as_tensor(bounds, dtype=torch.float32, device=dev).expand(R)
        zero = torch.zeros(R, device=dev)
        rs, rf = local(zero, zero, cx, cz, heading)
        out[:, L["field"]] = _cover(rs[:, None], rf[:, None], (-heading * DEG)[:, None], (2 * bd)[:, None],
                                    (2 * bd)[:, None], cell_m, cells)[:, 0]
    out[:, L["off_map"]] = 1.0 - out[:, L["field"]]
    if units is not None and K:
        rs, rf = local(units["x"], units["z"], cx[:, None], cz[:, None], heading[:, None])
        a = (units["b"] - heading[:, None]) * DEG
        dens = _cover(rs, rf, a, units["front"], units["depth"], cell_m, cells) * units["fill"][..., None, None]
        role = units["role"]
        for r, d_l, c_l, s_l in ((ENEMY, "enemy", "enemy_cos", "enemy_sin"), (OWN, "own", "own_cos", "own_sin"),
                                 (SELF, "self", None, None)):
            w = dens * (role == r).float()[..., None, None]
            tot = w.sum(1)
            out[:, L[d_l]] = tot.clamp(max=1.0)
            if c_l:
                scale = tot.clamp(max=1.0) / tot.clamp(min=1e-6)
                out[:, L[c_l]] = (w * torch.cos(a)[..., None, None]).sum(1) * scale
                out[:, L[s_l]] = (w * torch.sin(a)[..., None, None]).sum(1) * scale
    if area is not None:
        rs, rf = local(area["x"], area["z"], cx, cz, heading)
        a = (area["b"] - heading) * DEG
        cov = _cover(rs[:, None], rf[:, None], a[:, None], area["front"][:, None], area["depth"][:, None],
                     cell_m, cells)[:, 0]
        if "on" in area:
            cov = cov * area["on"].float()[:, None, None]
        out[:, L["area"]] = cov
    return out
