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
   11 reach      a shooter's range circle round the frame's centre (where it can fire if it turns: the circle the
                 player sees), the share of the cell within its range (the cell centre's distance, edge smoothed over
                 one cell); zeros without a missile weapon
   12 fire_arc   the part of that circle it fires in now without turning: within its fire arc now each side of the
                 frame's heading (view.fire_arc: standing 45 deg, on the move 90 or 0; the edge smoothed over a cell)
   13 height     the ground at the cell's centre less the ground under the frame's unit, / HEIGHT_M, within +-2
                 (tools/nn/sim/terrain.py; build/diag_g49/terrain/mechanic.md 6.2: the rules weigh only the difference,
                 a map's own level is not the network's business); zeros on a flat battle
   14 slope_fwd  the ground's rise per metre at the cell's centre along the frame's heading (forward), within +-1 (the
                 speed on a slope by the way's rise: the two components give it for any way)
   15 slope_right  the same to the frame's right

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
          "area", "reach", "fire_arc", "height", "slope_fwd", "slope_right")
LAYERS_BEFORE_FIRE = 11      # the layers before the shooter's (checkpoints made before them: train.restore pads them)
LAYERS_BEFORE_HEIGHT = 13    # the layers before the ground's (the same padding)
HEIGHT_M = 20.0              # the height layer's unit (m): +-2 = +-40 m, the missile rule's whole span
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


def _share(rs, rf, a, front, depth, xe, ye, cell):
    """[S..., Wr, Wc] the exact share of each cell of a grid inside each rectangle. Rectangles: centres (rs, rf) [S...] in
    the frame (s right, f forward), a [S...] its front's angle from the frame's heading (radians, clockwise), front x
    depth [S...]. The grid: column edges xe [S..., Wc+1] left to right, row edges ye [S..., Wr+1] top (forward) down,
    cell [S...] its size (leading dims broadcast with the rectangles')."""
    rs, rf = rs + JITTER[0], rf + JITTER[1]
    X0, X1 = xe[..., None, :-1], xe[..., None, 1:]            # [S, 1, Wc]
    Yt, Yb = ye[..., :-1, None], ye[..., 1:, None]            # [S, Wr, 1]
    Xc = (X0 + X1) / 2                                        # sums about each cell's centre (less cancellation)
    ux, uy = torch.sin(a), torch.cos(a)                        # the front's direction in (s, f)
    vx, vy = uy, -ux                                           # its right
    hd, hw = depth / 2, front / 2
    corner = [(rs + sd * hd * ux + sw * hw * vx, rf + sd * hd * uy + sw * hw * vy)
              for sd, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]     # counter-clockwise in (s, f)
    shape = torch.broadcast_shapes(rs.shape + (1, 1), X0.shape, Yt.shape)
    zeros = torch.zeros(shape, device=rs.device)
    area = zeros
    for (x0, y0), (x1, y1) in zip(corner, corner[1:] + corner[:1]):
        # the rectangle's edge, its part inside each cell
        x0_, y0_ = x0[..., None, None], y0[..., None, None]
        dx, dy = (x1 - x0)[..., None, None], (y1 - y0)[..., None, None]
        t0, t1 = zeros, zeros + 1.0
        t0, t1 = _clip(t0, t1, -dx, x0_ - X0)
        t0, t1 = _clip(t0, t1, dx, X1 - x0_)
        t0, t1 = _clip(t0, t1, -dy, y0_ - Yb)
        t0, t1 = _clip(t0, t1, dy, Yt - y0_)
        area = area + torch.where(t1 > t0, dy * ((x0_ - Xc) * (t1 - t0) + dx * (t1 * t1 - t0 * t0) / 2), zeros)
    # the cells' vertical edges (the horizontal ones add nothing to x dy), their part inside the rectangle: an upward
    # segment at s = xe from Yb to Yt, in the rectangle's frame across = (q . v), along = (q . u)
    L_ = cell[..., None, None]
    qx, qy = xe[..., None, :] - rs[..., None, None], Yb - rf[..., None, None]     # [S, Wr, Wc+1]
    vx_, vy_, ux_, uy_ = (t[..., None, None] for t in (vx, vy, ux, uy))
    acr0, acr_d = qx * vx_ + qy * vy_, L_ * vy_
    alo0, alo_d = qx * ux_ + qy * uy_, L_ * uy_
    hw_, hd_ = hw[..., None, None], hd[..., None, None]
    t0 = torch.zeros(torch.broadcast_shapes(acr0.shape, alo0.shape), device=rs.device)
    t1 = t0 + 1.0
    t0, t1 = _clip(t0, t1, -acr_d.expand_as(t0), (acr0 + hw_).expand_as(t0))
    t0, t1 = _clip(t0, t1, acr_d.expand_as(t0), (hw_ - acr0).expand_as(t0))
    t0, t1 = _clip(t0, t1, -alo_d.expand_as(t0), (alo0 + hd_).expand_as(t0))
    t0, t1 = _clip(t0, t1, alo_d.expand_as(t0), (hd_ - alo0).expand_as(t0))
    frac = (t1 - t0).clamp(min=0)                              # [S, Wr, Wc+1]
    # the right edge goes up (+L), the left one down (-L): x - Xc = +half / -half
    area = area + L_ * L_ / 2 * (frac[..., 1:] + frac[..., :-1])
    return (area / (L_ * L_)).clamp(0.0, 1.0)


_COMPILED = []


def _shared(dev):
    """_share as it runs on dev: on CUDA compiled (tools/nn/sim/compiled.py: the eager numbers; ~70 elementwise steps
    fused into a few kernels - unfused they are 90 % of the map's time), on the CPU (tests, the companion) eager."""
    if torch.device(dev).type != "cuda":
        return _share
    if not _COMPILED:
        from tools.nn.sim.compiled import compiled
        _COMPILED.append(compiled(_share, dynamic=True))
    return _COMPILED[0]


def _finish(units):
    """The unit layers' sums (enemy, its cos, sin, own, its cos, sin, self: [R, 7, C, C]) -> the layers: the density
    clamped to 1, the direction sums scaled to it (a density-weighted mean x the clamped density)."""
    out = []
    for d in (0, 3):
        tot = units[:, d]
        scale = tot.clamp(max=1.0) / tot.clamp(min=1e-6)
        out += [tot.clamp(max=1.0), units[:, d + 1] * scale, units[:, d + 2] * scale]
    return torch.stack(out + [units[:, 6].clamp(max=1.0)], 1)


_FINISHED = []


def _finished(dev):
    if torch.device(dev).type != "cuda":
        return _finish
    if not _FINISHED:
        from tools.nn.sim.compiled import compiled
        _FINISHED.append(compiled(_finish, dynamic=True))
    return _FINISHED[0]


def _cover(rs, rf, a, front, depth, cell_m, cells):
    """[R, K, C, C] the share of each cell of the whole map inside each rectangle (_share; rs.. [R, K])."""
    k = torch.arange(cells + 1, device=rs.device, dtype=torch.float32)
    xe = ((k - cells / 2)[None] * cell_m[:, None])[:, None]   # [R, 1, C+1]
    ye = ((cells / 2 - k)[None] * cell_m[:, None])[:, None]
    return _shared(rs.device)(rs, rf, a, front, depth, xe, ye, cell_m[:, None])


def _paint(flat, nlayers, mi, rs, rf, a, front, depth, value, layer, cell, cells):
    """Add value x share into flat (the maps [R, nlayers, C, C] as one row) for M rectangles, each only on the cells of
    its bounding box (no work on cells it cannot reach, none on empty slots). mi [M] the map of each, rs.. [M] as
    _share's, value / layer [M, V]: what it adds to which layers, cell [M] its map's cell size."""
    dev = rs.device
    ux, uy = torch.sin(a).abs(), torch.cos(a).abs()
    ext_s = front / 2 * uy + depth / 2 * ux + 1e-2           # the box's half extents across / along the frame
    ext_f = front / 2 * ux + depth / 2 * uy + 1e-2
    js, jf = rs + JITTER[0], rf + JITTER[1]
    c_lo = torch.floor((js - ext_s) / cell + cells / 2).long()
    c_hi = torch.floor((js + ext_s) / cell + cells / 2).long()
    r_lo = torch.floor(cells / 2 - (jf + ext_f) / cell).long()
    r_hi = torch.floor(cells / 2 - (jf - ext_f) / cell).long()
    Wc, Wr = (int(v) + 1 for v in torch.stack([(c_hi - c_lo).max(), (r_hi - r_lo).max()]).tolist())   # one sync
    kc = torch.arange(Wc + 1, device=dev)
    kr = torch.arange(Wr + 1, device=dev)
    cols = c_lo[:, None] + kc                                 # [M, Wc+1] the box's column edges' indices
    rows = r_lo[:, None] + kr                                 # [M, Wr+1]
    xe = (cols.float() - cells / 2) * cell[:, None]
    ye = (cells / 2 - rows.float()) * cell[:, None]
    share = _shared(dev)(rs, rf, a, front, depth, xe, ye, cell)     # [M, Wr, Wc]
    r_i, c_i = rows[:, :-1, None], cols[:, None, :-1]
    inside = (r_i >= 0) & (r_i < cells) & (c_i >= 0) & (c_i < cells)
    share = torch.where(inside, share, torch.zeros((), device=dev))
    base = mi[:, None, None] * (nlayers * cells * cells) + r_i.clamp(0, cells - 1) * cells + c_i.clamp(0, cells - 1)
    for v in range(value.shape[1]):
        idx = base + layer[:, v, None, None] * (cells * cells)
        flat.index_add_(0, idx.reshape(-1), (share * value[:, v, None, None]).reshape(-1))


def fire_layers(rng, arc, cell_m, cells=CELLS):
    """([R, C, C], [R, C, C]) the reach and fire_arc layers for ranges rng [R] (m, 0: none) and fire arcs now arc [R]
    (degrees each side of the heading): by the cell centre's distance d, the share min(1, max(0, (range - d) / cell +
    1/2)); the arc's by the angle off the heading, smoothed over the cell's angular size at d."""
    k = torch.arange(cells, device=rng.device, dtype=torch.float32)
    s, f = cell_centre(k[:, None], k[None, :], cell_m.view(-1, 1, 1), cells)
    s, f = s.expand(-1, cells, cells), f.expand(-1, cells, cells)
    d = torch.sqrt(s * s + f * f)
    cm = cell_m.view(-1, 1, 1)
    reach = ((rng.view(-1, 1, 1) - d) / cm + 0.5).clamp(0.0, 1.0) * (rng.view(-1, 1, 1) > 0).float()
    off = torch.atan2(s, f).abs()                                    # radians off the heading (forward: f > 0)
    span = (cm / d.clamp(min=1e-3)).clamp(max=math.pi)
    inarc = ((arc.view(-1, 1, 1) * DEG - off) / span + 0.5).clamp(0.0, 1.0)
    inarc = torch.where(arc.view(-1, 1, 1) >= 180.0, torch.ones_like(inarc), inarc)
    return reach, reach * inarc


def ground_layers(cx, cz, heading, cell_m, terr, h0, cells=CELLS, place=None):
    """([R, C, C] x 3) the height, slope_fwd and slope_right layers: the ground (tools/nn/sim/terrain.py, the map terr [R]
    of each frame's battle, -1 flat) at each cell's centre less h0 [R] (the ground under the frame's unit) / HEIGHT_M
    within +-2, and its rise per metre there along the frame's heading and to its right (central differences over 2 m),
    within +-1."""
    from tools.nn.sim import terrain
    k = torch.arange(cells, device=cx.device, dtype=torch.float32)
    s, f = cell_centre(k[:, None], k[None, :], cell_m.view(-1, 1, 1), cells)
    s, f = s.expand(-1, cells, cells), f.expand(-1, cells, cells)
    sh = (-1, 1, 1)
    x, z = world(s, f, cx.view(sh), cz.view(sh), heading.view(sh))
    u = {"terr": terr.long()[:, None]}
    if place is not None:
        # (place [3 x R]: each frame's battle laid on its map, terrain.set_map)
        u.update({k: v.float()[:, None] for k, v in zip(terrain.PLACE, place)})
    h = terrain.height(u, x, z)
    hd = heading.view(sh) * DEG
    fx, fz = torch.sin(hd), torch.cos(hd)
    rx, rz = torch.cos(hd), -torch.sin(hd)
    fwd = (terrain.height(u, x + fx, z + fz) - terrain.height(u, x - fx, z - fz)) / 2
    right = (terrain.height(u, x + rx, z + rz) - terrain.height(u, x - rx, z - rz)) / 2
    return ((h - h0.view(sh)) / HEIGHT_M).clamp(-2, 2), fwd.clamp(-1, 1), right.clamp(-1, 1)


def build(cx, cz, heading, cell_m, units=None, area=None, bounds=None, cells=CELLS, chunk=None, fire=None,
          ground=None):
    """[R, len(LAYERS), cells, cells] float maps.

    cx, cz, heading (degrees), cell_m: [R] the frames. units: {x, z, b, front, depth, fill, role} [R, K] (role NONE:
    an empty slot). area: {x, z, b, front, depth} [R] my area (a turned rectangle; front across its bearing) and
    optionally on [R] bool, or None. bounds: the square map's half-size (a number or [R]; the map is centred on 0, as
    State.bounds), None: no edge. chunk: maps per pass. A rectangle is painted only on the cells of its bounding
    box, an empty slot (role NONE) not at all (_paint). fire: {range, arc} [R] the frame's unit's missile range (m)
    and fire arc now (degrees each side) for the reach / fire_arc layers (fire_layers), None: zeros. ground: {terr, h0}
    [R] the frame's battle's height map (tools/nn/sim/terrain.py; -1 flat) and the ground under its unit for the
    height / slope layers (ground_layers), None: zeros."""
    R = cx.shape[0]
    dev = cx.device
    K = 0 if units is None else units["x"].shape[1]
    chunk = chunk or max(1, (1 << 20) // max(K, 1))
    if R > chunk:
        def part(v, i):
            return v[i:i + chunk] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == R else v
        return torch.cat([build(part(cx, i), part(cz, i), part(heading, i), part(cell_m, i),
                                None if units is None else {k: part(v, i) for k, v in units.items()},
                                None if area is None else {k: part(v, i) for k, v in area.items()},
                                part(bounds, i), cells, chunk,
                                None if fire is None else {k: part(v, i) for k, v in fire.items()},
                                None if ground is None else {k: part(v, i) for k, v in ground.items()})
                          for i in range(0, R, chunk)])
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
        role = units["role"]
        used = role != NONE
        dens_v = units["fill"] * used.float()
        # each rectangle adds its density to its role's layer and density x cos / sin of its front to the direction
        # layers (the frame's own unit: its density only); the sums are scaled to the clamped density below
        lay = torch.tensor([[0, 0, 0], [L["enemy"], L["enemy_cos"], L["enemy_sin"]],
                            [L["own"], L["own_cos"], L["own_sin"]], [L["self"], 0, 0]], device=dev)[role]
        dirs = ((role == ENEMY) | (role == OWN)).float()
        val = torch.stack([dens_v, dens_v * torch.cos(a) * dirs, dens_v * torch.sin(a) * dirs], -1)
        mi, ki = used.nonzero(as_tuple=True)
        if mi.numel():                  # the sums go straight into out's layers (zeros so far), then are scaled
            _paint(out.view(-1), len(LAYERS), mi, rs[mi, ki], rf[mi, ki], a[mi, ki], units["front"][mi, ki],
                   units["depth"][mi, ki], val[mi, ki], lay[mi, ki], cell_m[mi], cells)
        out[:, L["enemy"]:L["self"] + 1] = _finished(dev)(out[:, L["enemy"]:L["self"] + 1])
    if area is not None:
        rs, rf = local(area["x"], area["z"], cx, cz, heading)
        a = (area["b"] - heading) * DEG
        cov = _cover(rs[:, None], rf[:, None], a[:, None], area["front"][:, None], area["depth"][:, None],
                     cell_m, cells)[:, 0]
        if "on" in area:
            cov = cov * area["on"].float()[:, None, None]
        out[:, L["area"]] = cov
    if fire is not None:
        out[:, L["reach"]], out[:, L["fire_arc"]] = fire_layers(fire["range"], fire["arc"], cell_m, cells)
    if ground is not None:
        out[:, L["height"]], out[:, L["slope_fwd"]], out[:, L["slope_right"]] = ground_layers(
            cx, cz, heading, cell_m, ground["terr"], ground["h0"], cells, place=ground.get("place"))
    return out
