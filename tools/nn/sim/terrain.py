"""The ground's height (build/diag_g49/terrain/mechanic.md: the game's rules from the program; config/nn/sim.json
terrain with its sources). A battle without a height map is flat - every rule here is off and the simulator steps as
before, bit for bit.

The maps: a registry of height grids (register: float32 [rows, columns], row = z, column = x, x = x0 + column x step,
z = z0 + row x step - tools/nn/map_grid.py's heights.npz; the game's own ground is a regular grid read bilinearly, its
nodes 2 m apart on the captured Kislev map). A battle's map is u["terr"] [B, N] (its id, the same on every slot; -1 or
absent: flat), a per-battle tensor as every other, so a batch narrowed, compacted or cloned keeps its maps.

    height(u, x, z)              the ground under points [B, ...], bilinear (0 on a flat battle)
    slope(u, x, z, dx, dz, L)    (h(p + L d) - h(p)) / L along the unit direction d
    pair_dh(u)                   [B, i, j] the ground under j's centre less i's
    speed_factor(g, cap)         the game's speed on a slope: up x 1 / (1 + g), down x (1 + |g|) at most cap
    melee_factor(u, pw, params)  [B, i, j] the damage of i's blows on j by the height between their fighting men
    missile_factor(dh, params)   [B, i, j] the damage of i's projectiles on j by the height between them
    hill(u, pw, params)          [B, N] the morale's 'on the hill' (+10)
    ground_closed(...)           [B, N] the shooter's line of fire at its target closed by the ground
"""
import math

import numpy as np
import torch

_MAPS = []          # (grid float32 [R, C], x0, z0, step) on the host
FRAMES = {}         # map id -> (min_x, max_x, min_z, max_z): the playable area (the field), map coordinates
_DEV = {}           # device -> (flat grids [M * Rmax * Cmax], R [M], C [M], x0 [M], z0 [M], step [M], Rmax, Cmax)


def register(grid, x0, z0, step):
    """Add a height grid (numpy or tensor [rows, columns], m) to the registry; returns its id."""
    g = np.asarray(grid.cpu() if torch.is_tensor(grid) else grid, dtype=np.float32)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "a height grid is [rows >= 2, columns >= 2]"
    _MAPS.append((g, float(x0), float(z0), float(step)))
    _DEV.clear()
    mid = len(_MAPS) - 1
    FRAMES[mid] = (float(x0), float(x0) + (g.shape[1] - 1) * float(step), float(z0),
                   float(z0) + (g.shape[0] - 1) * float(step))
    return mid


def set_frame(map_id, min_x, max_x, min_z, max_z):
    """The map's field: the playable area, where move orders are allowed (the minimap's frame: the Assembly Kit's
    Playable Area sets the radar's extent - docs/ru/game/map/boundaries.md; Kislev's -512..512 x -512..512, the manual
    ride reached 3.7-20.7 m of each side). Without one: the height grid's extent (register)."""
    FRAMES[int(map_id)] = (float(min_x), float(max_x), float(min_z), float(max_z))


def frame_local(terr, place=None):
    """The field of each battle's map in the battle's own frame: (cx, cz, b, front, depth) [R] - a turned rectangle
    (the map's x span across its bearing, z span along it; a bearing as everywhere: degrees clockwise from +z), for
    terr [R] map ids (>= 0) and place (tpx, tpz, tpr) [R] each or None (the map's own frame). A battle laid on the map
    at (px, pz, a) (set_map) sees the map's point (X, Z) at x = dX cos a - dZ sin a, z = dX sin a + dZ cos a, dX = X - px,
    dZ = Z - pz; the map's bearing 0 is the battle's -a."""
    dev = terr.device
    ids = terr.long().clamp(min=0)
    tab = torch.tensor([FRAMES.get(k, (0.0, 0.0, 0.0, 0.0)) for k in range(max(len(_MAPS), 1))], device=dev)
    f = tab[ids]
    cx, cz = (f[:, 0] + f[:, 1]) / 2, (f[:, 2] + f[:, 3]) / 2
    front, depth = f[:, 1] - f[:, 0], f[:, 3] - f[:, 2]
    if place is None:
        return cx, cz, torch.zeros_like(cx), front, depth
    px, pz, a = (v.float() for v in place)
    t = a * (math.pi / 180)
    c, s = torch.cos(t), torch.sin(t)
    dx, dz = cx - px, cz - pz
    return dx * c - dz * s, dx * s + dz * c, torch.remainder(-a, 360.0), front, depth


def load_npz(path, layers=True):
    """Register a map_grid heights.npz (height [rows, columns], x0, z0, step: tools/nn/map_grid.py); returns its id.
    layers: with the map folder's ground types (heights.npz ground + meta.json names) and trunks (trees.npz) - ground.py;
    False: the heights alone. The map's field (set_frame): meta.json's frame when the folder has one."""
    from pathlib import Path
    import json
    d = np.load(path)
    mid = register(d["height"], float(d["x0"]), float(d["z0"]), float(d["step"]))
    meta = Path(path).parent / "meta.json"
    if meta.exists():
        fr = json.loads(meta.read_text(encoding="utf-8")).get("frame")
        if fr:
            set_frame(mid, fr["min_x"], fr["max_x"], fr["min_z"], fr["max_z"])
    if layers:
        from tools.nn.sim import ground
        ground.load_layers(mid, Path(path).parent)
    return mid


def set_map(st, map_id, battles=None, place=None):
    """Give battles (a list / bool mask; None: all) the map map_id (None or -1: flat). place (px, pz, deg) [B] each
    (tensors or numbers; None: none): the battle's own frame laid on the map - a point (x, z) of the battle stands on
    the map at (x cos a + z sin a + px, -x sin a + z cos a + pz), a = deg (a bearing b on the map: b + deg); the battle
    itself keeps its coordinates (to_map: only the map's layers are read through it)."""
    u = st.u
    mid = -1 if map_id is None else int(map_id)
    if "terr" not in u:
        if mid < 0:
            return
        u["terr"] = torch.full_like(u["side"], -1, dtype=torch.int64)
    sel = torch.ones(st.B, dtype=torch.bool, device=st.device) if battles is None else torch.as_tensor(
        battles, device=st.device)
    if sel.dtype != torch.bool:
        sel = torch.zeros(st.B, dtype=torch.bool, device=st.device).index_fill_(0, sel.long(), True)
    u["terr"] = torch.where(sel[:, None], torch.full_like(u["terr"], mid), u["terr"])
    if place is not None:
        for k, v in zip(PLACE, place):
            if k not in u:
                u[k] = torch.zeros_like(u["x"])
            v = torch.as_tensor(v, dtype=u["x"].dtype, device=st.device).expand(st.B)
            u[k] = torch.where(sel[:, None], v[:, None].expand_as(u[k]), u[k])


PLACE = ("tpx", "tpz", "tpr")          # a battle's frame on its map (set_map place; the same on every slot)


def to_map(u, x, z):
    """(x, z) on the map of points x, z [B, ...] of each battle (set_map place; without one: as they are)."""
    if "tpx" not in u:
        return x, z
    sh = (-1,) + (1,) * (x.dim() - 1)
    a = u["tpr"][:, 0].view(sh) * (math.pi / 180)
    c, s = torch.cos(a), torch.sin(a)
    return x * c + z * s + u["tpx"][:, 0].view(sh), -x * s + z * c + u["tpz"][:, 0].view(sh)


def on(u):
    """Any battle of the batch has a height map (the rules run; flat battles in it see height 0)."""
    return "terr" in u


def _device(dev):
    key = str(dev)
    if key not in _DEV:
        Rm = max(g.shape[0] for g, *_ in _MAPS)
        Cm = max(g.shape[1] for g, *_ in _MAPS)
        flat = np.zeros((len(_MAPS), Rm, Cm), np.float32)
        for k, (g, *_r) in enumerate(_MAPS):
            flat[k, :g.shape[0], :g.shape[1]] = g
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)       # noqa: E731
        _DEV[key] = (torch.from_numpy(flat.reshape(-1)).to(dev), t([g.shape[0] for g, *_ in _MAPS], torch.int64),
                     t([g.shape[1] for g, *_ in _MAPS], torch.int64), t([m[1] for m in _MAPS]),
                     t([m[2] for m in _MAPS]), t([m[3] for m in _MAPS]), Rm, Cm)
    return _DEV[key]


def height(u, x, z):
    """The ground (m) under points x, z [B, ...] of each battle's map (bilinear between the grid's nodes, held at its
    edge); 0 on a flat battle and without maps."""
    if "terr" not in u or not _MAPS:
        return torch.zeros_like(x)
    x, z = to_map(u, x, z)
    G, R, C, X0, Z0, S, Rm, Cm = _device(x.device)
    mid = u["terr"][:, 0]
    flat = mid < 0
    m = mid.clamp(min=0)
    sh = (-1,) + (1,) * (x.dim() - 1)
    fx = (x - X0[m].view(sh)) / S[m].view(sh)
    fz = (z - Z0[m].view(sh)) / S[m].view(sh)
    Cb, Rb = C[m].view(sh), R[m].view(sh)
    fx = torch.minimum(fx.clamp(min=0), (Cb - 1).float())
    fz = torch.minimum(fz.clamp(min=0), (Rb - 1).float())
    i0 = torch.minimum(fx.floor().long(), Cb - 2)
    j0 = torch.minimum(fz.floor().long(), Rb - 2)
    ax, az = fx - i0.float(), fz - j0.float()
    base = m.view(sh) * (Rm * Cm)

    def at(j, i):
        return G[(base + j * Cm + i).reshape(-1)].view(x.shape)
    h = (at(j0, i0) * (1 - ax) * (1 - az) + at(j0, i0 + 1) * ax * (1 - az) + at(j0 + 1, i0) * (1 - ax) * az
         + at(j0 + 1, i0 + 1) * ax * az)
    return torch.where(flat.view(sh), torch.zeros_like(h), h)


def slope(u, x, z, dx, dz, L):
    """The ground's rise per metre from x, z along the unit direction (dx, dz) over L m."""
    return (height(u, x + dx * L, z + dz * L) - height(u, x, z)) / L


def ground(u):
    """[B, N] the ground under each unit's centre."""
    return height(u, u["x"], u["z"])


def pair_dh(u):
    """[B, i, j] the ground under j's centre less the ground under i's."""
    h = ground(u)
    return h[:, None, :] - h[:, :, None]


def speed_factor(g, cap):
    """The game's speed on a slope g (rise per metre along the way; mechanic.md 2, the program): up x 1 / (1 + g), down
    x (1 + |g|) at most cap (the database: x1.5)."""
    return torch.where(g >= 0, 1 / (1 + g), (1 - g).clamp(max=float(cap)))


def melee_factor(u, pw, params):
    """[B, i, j] the multiplier of i's blows on j by the height between their fighting men (mechanic.md 1: the game's
    damage x (1 - clamp(k x dh / d_max, -k, k)) per pair of men, dh = the defender's feet less the striker's; the
    database's melee_height_damage_modifier_max_coefficient 0.3 / _max_difference 1 m). A unit is a rectangle here, its
    men in contact along the side of i that faces j: n_pairs pairs evenly along the part of that side j's box covers (its
    overlap; geometry as melee.front_overlap), each the striker's man pair_gap_m / 2 inside the contact line and the
    defender's as far outside it (the probe terrain: the fighting pairs 1.53-1.59 m apart); the multiplier is the
    pairs' mean (a single man: his centre against the other's nearest side)."""
    tc = params.sim["terrain"]
    k = float(params.battle.get("melee_height_damage_modifier_max_coefficient", 0.3))
    dmax = float(params.battle.get("melee_height_damage_modifier_max_difference", 1.0))
    n = int(tc.get("pairs", 5))
    gap = float(tc["pair_gap_m"])
    b = u["b"] * math.pi / 180
    sb, cb = torch.sin(b), torch.cos(b)
    front_i, depth_i = pw["front"][:, :, None], pw["depth"][:, :, None]
    s, c = torch.sin(pw["rel_i"]), torch.cos(pw["rel_i"])
    tf = s.abs() * depth_i <= c.abs() * front_i                 # i's front or back faces j (else a flank)
    sgn_f = torch.where(c >= 0, 1.0, -1.0)
    sgn_s = torch.where(s >= 0, 1.0, -1.0)
    # the normal from i towards j (its facing or a flank's) and the side's line
    nx = torch.where(tf, sb[:, :, None] * sgn_f, cb[:, :, None] * sgn_s)
    nz = torch.where(tf, cb[:, :, None] * sgn_f, -sb[:, :, None] * sgn_s)
    lx, lz = -nz, nx
    edge = torch.where(tf, depth_i, front_i) / 2                 # the contact line from i's centre
    half = torch.where(tf, front_i, depth_i) / 2
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]
    off = rx * lx + rz * lz
    bj = b[:, None, :]
    ext = (pw["front"][:, None, :] / 2 * (torch.cos(bj) * lx - torch.sin(bj) * lz).abs()
           + pw["depth"][:, None, :] / 2 * (torch.sin(bj) * lx + torch.cos(bj) * lz).abs())
    lo = torch.maximum(-half, off - ext)
    hi = torch.minimum(half, off + ext)
    hi = torch.maximum(hi, lo)
    xi, zi = u["x"][:, :, None], u["z"][:, :, None]
    acc = torch.zeros_like(off)
    for q in range(n):
        a = lo + (hi - lo) * (q + 0.5) / n
        px, pz = xi + lx * a + nx * edge, zi + lz * a + nz * edge
        h_s = height(u, px - nx * gap / 2, pz - nz * gap / 2)
        h_d = height(u, px + nx * gap / 2, pz + nz * gap / 2)
        acc = acc + (1 - (k * (h_d - h_s) / dmax).clamp(-k, k))
    return acc / n


def missile_factor(dh, params):
    """[B, i, j] the multiplier of i's projectiles on j (mechanic.md 3: the game's damage x (1 + clamp(k x (shooter's
    ground - target's) / d_max, -k, k)); the database's missile_height_damage_modifier_max_coefficient 0.3 /
    _max_difference 40 m); dh [B, i, j] the target's ground less the shooter's (pair_dh)."""
    k = float(params.battle.get("missile_height_damage_modifier_max_coefficient", 0.3))
    dmax = float(params.battle.get("missile_height_damage_modifier_max_difference", 40.0))
    return 1 + (k * (-dh) / dmax).clamp(-k, k)


def hill(u, pw, alive, params):
    """[B, N] the morale's 'on the hill' (mechanic.md 4, the program; the database's ume_encouraged_on_the_hill +10): an
    enemy within radius_m (centre to centre) and the ground under the unit's front rank (its centre + depth / 2 ahead)
    at least dh_m above the ground under every such enemy's centre."""
    tc = params.sim["terrain"]["hill"]
    b = u["b"] * math.pi / 180
    d = pw["depth"] / 2
    hf = height(u, u["x"] + torch.sin(b) * d, u["z"] + torch.cos(b) * d)
    hc = ground(u)
    near = pw["enemy"] & alive[:, None, :] & (pw["dist"] <= float(tc["radius_m"]))
    top = torch.where(near, hc[:, None, :], torch.full_like(pw["dist"], -1e9)).amax(2)
    return alive & near.any(2) & (hf >= top + float(tc["dh_m"]))


def ground_closed(u, pw, target, params, rays=(-0.5, 0.0, 0.5), samples=None):
    """[B, N] bool: shooter i's line of fire at its target closed by the ground (mechanic.md 3, the program: three rays
    from its front - the middle and the ends - along the low arc to the target's nearest man at aim_frac of his height;
    the ground in the way blocks a ray but in the last 0.1 m at the target; fewer than rays_clear clear: closed). The
    arc: the muzzle muzzle_m above the shooter's ground, the low launch angle through the aim point (missile._arc_tan);
    the ground sampled at `samples` points along each ray (config/nn/sim.json terrain.los_samples)."""
    from tools.nn.sim import missile
    ms = params.sim["missile"]
    fcfg = ms.get("friend_los") or {}
    arc = ms.get("arc_los")
    hm = float(arc["muzzle_m"]) if arc else float(fcfg.get("muzzle_m", 1.5))
    M = int(samples or params.sim["terrain"].get("los_samples", 24))
    t = target.clamp(min=0)[:, :, None]
    th = pw["theta"].gather(2, t)[..., 0]                    # [B, i] bearing to the target (rad, from +z)
    reach = pw["reach_j"].gather(2, t)[..., 0].clamp(min=1.0)
    dx, dz = torch.sin(th), torch.cos(th)
    b = u["b"] * math.pi / 180
    fx, fz = torch.sin(b), torch.cos(b)                         # i's facing
    sx, sz = torch.cos(b), -torch.sin(b)                        # i's front line, to the right
    front, depth = pw["front"], pw["depth"]
    ha = float(fcfg.get("aim_frac", 0.75)) * u["height"].gather(1, target.clamp(min=0))
    tx, tz = u["x"] + dx * reach, u["z"] + dz * reach           # the target's nearest man (its edge on the line)
    h_t = height(u, tx, tz) + ha
    v = u["muzzle_v"].clamp(min=1.0)
    clear = torch.zeros_like(u["men"])
    k = torch.arange(1, M + 1, device=u["x"].device, dtype=u["x"].dtype) / (M + 1)
    for r in rays:
        ox = u["x"] + fx * depth / 2 + sx * r * front
        oz = u["z"] + fz * depth / 2 + sz * r * front
        h0 = height(u, ox, oz) + hm
        L = torch.sqrt((tx - ox) ** 2 + (tz - oz) ** 2).clamp(min=1.0)
        dh = h_t - h0
        tan = missile._arc_tan(v, L, dh)
        a = ((tan * L - dh) / (L * L)).clamp(min=0)
        xs = L[..., None] * k                                  # [B, N, M]
        stop = xs < (L[..., None] - 0.1)
        px = ox[..., None] + (tx - ox)[..., None] * (xs / L[..., None])
        pz = oz[..., None] + (tz - oz)[..., None] * (xs / L[..., None])
        y = h0[..., None] + tan[..., None] * xs - a[..., None] * xs * xs
        hit = (height(u, px, pz) > y) & stop
        clear = clear + (~hit.any(-1)).float()
    return (clear < int(fcfg.get("rays_clear", 2))) & (target >= 0)
