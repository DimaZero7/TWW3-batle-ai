"""The map's ground types and tree trunks (build/diag_g49/terrain/ground.md, trees.md, stealth.md: the game's rules from
its program and database; config/nn/sim.json terrain.ground / terrain.trees with their sources). Rides on terrain.py's
map registry: a map's layers (register_layers) are on the same grid and id as its heights; a battle whose map has no
layers - or no map - has none of this, bit for bit.

Layers of a map, on its height grid's nodes (x = x0 + i step; a point's cell is the node at floor((x - x0) / step) - the
game's 2 m ground cells have their edges on even metres, so on Kislev's 1 m capture both nodes of a cell read it):
  cls      the ground class: 0 nothing, 1 forest, 2 shallow water, 3 deep water, 4 slow (rock, scrub) - from the captured
           ground names (GROUND_CLASS); the rest (grass, mud, snow, sand, road, sharp stones, sparse forest ...) nothing
  lam      the trunks' stopping density of a projectile path through the cell: sum of 2 r over the trunks whose centre
           is in it, per m2 (a path crossing the cell meets a trunk of radius r with chance 2 r / (the cell's width) per
           m: the expected share of the literal trunks over the projectiles' spread)
  top/base the highest trunk's top / the lowest trunk's base in the cell (world y): a path stops only between them
  s0/s1/s2 the trunks around a point for the men's way through them (count, sum r, sum r2 per m2 over a box of
           MOVE_BOX_M centred on the node)

  points(u, front, depth, nf, nd)        [B, N, P] x, z of nf x nd points spread over each unit's box (its men)
  classes(u, x, z)                       [B, ...] ground class under points (0 without layers)
  shares(u, front, depth, params)        [B, N, 5] the share of each unit's men on each class
  factors(u, share, params)              (speed, attack) [B, N] the database's ground multipliers, by the unit's men
  trunk_keep(u, pw, target, lob, params) [B, N] the share of shooter i's projectiles no trunk stops on the way
  trunk_speed(u, front, depth, params)   [B, N] the unit's speed through the trunks
"""
import math

import numpy as np
import torch

from tools.nn.sim import terrain

# the captured ground names (map_grid.py, bm:ground_type) -> class (ground.md 1: the database's
# ground_type_to_stat_effects rows; a name not here: nothing)
GROUND_CLASS = {"forest": 1, "shallow_water": 2, "deep_water": 3, "rock": 4, "scrub": 4}
NCLASS = 5
MOVE_BOX_M = 5              # the box the trunks' density for the men's way is taken over (a few men's widths)
LAM_BOX_M = 3               # the box the projectiles' stopping density is spread over (the paths' spread)

_LAYERS = {}                # map id -> host arrays (same shape as its height grid)
_DEV = {}                   # device -> stacked tensors


def classes_of(codes, names):
    """The ground class grid (int8) of a capture's ground codes (uint8) and their names (meta.json ground_names)."""
    lut = np.array([GROUND_CLASS.get(str(n), 0) for n in names] + [0] * (256 - len(names)), np.int8)
    return lut[np.asarray(codes, np.int64)]


def trunk_grids(trunks, shape, x0, z0, step):
    """(lam, top, base, s0, s1, s2) float32 [rows, columns] of a map's trunks (map_grid.tree_bodies: x, z, y, r, h) on
    its grid (module doc)."""
    R, C = shape
    lam = np.zeros(shape, np.float32)
    top = np.full(shape, -1e9, np.float32)
    base = np.full(shape, 1e9, np.float32)
    cnt = np.zeros(shape, np.float32)
    sr = np.zeros(shape, np.float32)
    sr2 = np.zeros(shape, np.float32)
    if trunks is not None and len(trunks["x"]):
        i = np.floor((np.asarray(trunks["x"]) - x0) / step).astype(np.int64)
        j = np.floor((np.asarray(trunks["z"]) - z0) / step).astype(np.int64)
        ok = (i >= 0) & (i < C) & (j >= 0) & (j < R)
        i, j = i[ok], j[ok]
        r = np.asarray(trunks["r"], np.float32)[ok]
        y = np.asarray(trunks["y"], np.float32)[ok]
        h = np.asarray(trunks["h"], np.float32)[ok]
        area = step * step
        np.add.at(lam, (j, i), 2 * r / area)
        np.maximum.at(top, (j, i), y + h)
        np.minimum.at(base, (j, i), y)
        np.add.at(cnt, (j, i), 1.0)
        np.add.at(sr, (j, i), r)
        np.add.at(sr2, (j, i), r * r)
    def box(a, m, mean=True):
        # the sum over an m x m box of cells centred on each node (mean: per m2 of the box)
        k = max(1, int(round(m / step)))
        p = np.pad(a, ((k // 2, k - 1 - k // 2), (k // 2, k - 1 - k // 2)))
        cs = np.cumsum(np.cumsum(p.astype(np.float64), 0), 1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        s = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
        return (s / (k * k) if mean else s).astype(np.float32)

    def spread(a, m, fn, fill):
        # the max / min over the same box (a path anywhere in the box may meet the trunk)
        k = max(1, int(round(m / step)))
        p = np.pad(a, ((k // 2, k - 1 - k // 2), (k // 2, k - 1 - k // 2)), constant_values=fill)
        out = np.full(a.shape, fill, np.float32)
        for dj in range(k):
            for di in range(k):
                out = fn(out, p[dj:dj + a.shape[0], di:di + a.shape[1]])
        return out
    # the projectiles' density over LAM_BOX_M (the expectation of the literal trunks over the paths' spread across a
    # few metres: a single path's cells otherwise hit or miss a trunk whole); the men's over MOVE_BOX_M
    lam_s = box(lam, LAM_BOX_M)
    top_s = spread(top, LAM_BOX_M, np.maximum, -1e9)
    base_s = spread(base, LAM_BOX_M, np.minimum, 1e9)
    per_m2 = 1.0 / (step * step)
    return (lam_s, top_s, base_s, box(cnt, MOVE_BOX_M) * per_m2, box(sr, MOVE_BOX_M) * per_m2,
            box(sr2, MOVE_BOX_M) * per_m2)


def register_layers(map_id, ground=None, names=None, trunks=None):
    """Give map map_id (terrain.register's id) its ground codes [rows, columns] with their names and / or its trunks
    (map_grid.tree_bodies' arrays); either may be None (no ground types / no trees)."""
    g, x0, z0, step = terrain._MAPS[map_id]
    cls = classes_of(ground, names) if ground is not None else np.zeros(g.shape, np.int8)
    assert cls.shape == g.shape, "the ground layer is on the height grid"
    _LAYERS[map_id] = {"cls": cls, "trees": trunks is not None and len(trunks["x"]) > 0,
                       "ground": ground is not None,
                       **dict(zip(("lam", "top", "base", "s0", "s1", "s2"), trunk_grids(trunks, g.shape, x0, z0, step)))}
    _DEV.clear()


def load_layers(map_id, folder):
    """The layers of a map folder (build/maps/<name>/): heights.npz's ground codes with meta.json's names, trees.npz
    when it is there. Returns (ground share, trunks) for the record."""
    import json
    from pathlib import Path
    folder = Path(folder)
    d = np.load(folder / "heights.npz")
    names = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["ground_names"] \
        if (folder / "meta.json").exists() and "ground" in d.files else None
    trunks = dict(np.load(folder / "trees.npz")) if (folder / "trees.npz").exists() else None
    if names is None and trunks is None:
        return None, 0                      # (heights alone: the map has no layers)
    register_layers(map_id, d["ground"] if names is not None else None, names, trunks)
    return names, 0 if trunks is None else len(trunks["x"])


def on(u):
    """Some battle of the batch has a map with layers."""
    return terrain.on(u) and bool(_LAYERS)


def _device(dev):
    key = str(dev)
    if key not in _DEV:
        Rm = max(g.shape[0] for g, *_ in terrain._MAPS)
        Cm = max(g.shape[1] for g, *_ in terrain._MAPS)
        M = len(terrain._MAPS)
        out = {}
        for name, fill, dt in (("cls", 0, np.int8), ("lam", 0, np.float32), ("top", -1e9, np.float32),
                               ("base", 1e9, np.float32), ("s0", 0, np.float32), ("s1", 0, np.float32),
                               ("s2", 0, np.float32)):
            a = np.full((M, Rm, Cm), fill, dt)
            for k in range(M):
                if k in _LAYERS:
                    L = _LAYERS[k][name]
                    a[k, :L.shape[0], :L.shape[1]] = L
            out[name] = torch.from_numpy(a.reshape(-1)).to(dev)
        out["has_trees"] = torch.tensor([bool(_LAYERS.get(k, {}).get("trees")) for k in range(M)], device=dev)
        out["has_ground"] = torch.tensor([bool(_LAYERS.get(k, {}).get("ground")) for k in range(M)], device=dev)
        _DEV[key] = (out, Rm, Cm)
    return _DEV[key]


def _at(u, name, x, z):
    """The layer `name` at points x, z [B, ...] of each battle's map (the node at floor; held at the grid's edge); the
    layer's empty value on a flat battle."""
    out, Rm, Cm = _device(x.device)
    G, R, C, X0, Z0, S, _, _ = terrain._device(x.device)
    x, z = terrain.to_map(u, x, z)                  # (a battle laid on the map in its own frame: terrain.set_map place)
    mid = u["terr"][:, 0]
    flat = mid < 0
    m = mid.clamp(min=0)
    sh = (-1,) + (1,) * (x.dim() - 1)
    i = torch.floor((x - X0[m].view(sh)) / S[m].view(sh)).long()
    j = torch.floor((z - Z0[m].view(sh)) / S[m].view(sh)).long()
    i = torch.minimum(i.clamp(min=0), C[m].view(sh) - 1)
    j = torch.minimum(j.clamp(min=0), R[m].view(sh) - 1)
    v = out[name][((m.view(sh) * Rm + j) * Cm + i).reshape(-1)].view(x.shape)
    empty = {"cls": 0, "lam": 0, "top": -1e9, "base": 1e9}.get(name, 0)
    return torch.where(flat.view(sh), torch.full_like(v, empty), v)


def points(u, front, depth, nf, nd):
    """(x, z) [B, N, nf x nd]: the centres of an nf x nd split of each unit's box (front across its facing, depth along
    it) - its men, evenly."""
    b = u["b"] * math.pi / 180
    fx, fz = torch.sin(b), torch.cos(b)
    sx, sz = torch.cos(b), -torch.sin(b)
    a = (torch.arange(nf, device=b.device, dtype=b.dtype) + 0.5) / nf - 0.5
    c = (torch.arange(nd, device=b.device, dtype=b.dtype) + 0.5) / nd - 0.5
    A = (a[:, None] * torch.ones_like(c)[None, :]).reshape(-1)
    D = (torch.ones_like(a)[:, None] * c[None, :]).reshape(-1)
    ox = A[None, None, :] * front[..., None]
    od = D[None, None, :] * depth[..., None]
    return (u["x"][..., None] + sx[..., None] * ox + fx[..., None] * od,
            u["z"][..., None] + sz[..., None] * ox + fz[..., None] * od)


def classes(u, x, z):
    """[B, ...] the ground class under points (0 without layers)."""
    if not on(u):
        return torch.zeros_like(x, dtype=torch.int64)
    return _at(u, "cls", x, z).long()


def shares(u, front, depth, params):
    """[B, N, NCLASS] the share of each unit's men (terrain.ground.points nf x nd over its box) on each ground class."""
    nf, nd = (int(v) for v in params.sim["terrain"]["ground"]["points"])
    px, pz = points(u, front, depth, nf, nd)
    c = classes(u, px, pz)
    return torch.stack([(c == k).float().mean(-1) for k in range(NCLASS)], -1)


def missile_cover(u, share, params):
    """[B, N] the share of the projectile hits a unit still takes for its men on forest ground: 1 - forest share x (1 -
    terrain.ground.forest_cover) - each man on a forest cell is hit at forest_cover of his chance (MEASURED, the probe
    cover: archers from ~40 m, a lone Warlock Engineer hit 36 % of the shots in the woods against 46 % on grass, x0.77,
    clanrats' damage a shot x0.74, the damage a hit the same; the program's reading - the database's
    missile_target_in_cover_penalty, -0.2 off the chance - would give ~26 %: config/nn/sim.json terrain.ground
    forest_cover_why). 1 without a ground layer or the key."""
    k = (params.sim["terrain"].get("ground") or {}).get("forest_cover")
    if k is None:
        return torch.ones_like(u["x"])
    return 1 - share[..., 1] * (1 - float(k))


def factors(u, share, params):
    """(speed, attack) [B, N]: the database's ground_type_to_stat_effects by the unit's men (each man his own cell: the
    mean of the multipliers over its men). A penalty (x < 1) does not touch a strider; ignore_trees is spared the forest's;
    the multipliers by the unit's group (small / large: u large; the database's ground_stat_effect_group)."""
    gc = params.sim["terrain"]["ground"]
    large = u["large"]
    strider = u["strider"] if "strider" in u else torch.zeros_like(large)
    ign = u["ign_trees"] if "ign_trees" in u else torch.zeros_like(large)
    one = torch.ones_like(u["x"])
    sp = [one, one, one, one, one]
    at = [one, one, one, one, one]
    f_large, w_small, deep, slow = (float(gc["forest_large"]), float(gc["shallow_small"]), float(gc["deep_all"]),
                                    float(gc["slow_all"]))
    sp[1] = torch.where(large & ~ign, one * f_large, one)
    at[1] = sp[1]
    sp[2] = torch.where(~large, one * w_small, one)
    at[2] = sp[2]
    sp[3] = one * deep
    sp[4] = one * slow
    spd = sum(share[..., k] * sp[k] for k in range(NCLASS))
    att = sum(share[..., k] * at[k] for k in range(NCLASS))
    spd = torch.where(strider, torch.maximum(spd, one), spd)
    att = torch.where(strider, torch.maximum(att, one), att)
    return spd, att


def trunk_speed(u, front, depth, params):
    """[B, N] the unit's speed through the trunks: each man meets a trunk of radius r on 2 (r + R) of every metre's
    width (R his body across: radius x radii_ratio) and loses detour x (r + R) of way to it, so his time a metre is 1 +
    2 detour sum (r + R)2 per m2 (the trunks around him: s0, s1, s2), the speed 1 / that, over the unit's men (its
    points); a strider and ignore_trees pass through (the game's bodies do not meet)."""
    tc = params.sim["terrain"]["trees"]
    nf, nd = (int(v) for v in params.sim["terrain"]["ground"]["points"])
    px, pz = points(u, front, depth, nf, nd)
    s0, s1, s2 = _at(u, "s0", px, pz), _at(u, "s1", px, pz), _at(u, "s2", px, pz)
    ratio = u["body_ratio"] if "body_ratio" in u else torch.ones_like(u["radius"])
    R = (u["radius"] * torch.where(ratio > 0, ratio, torch.ones_like(ratio)))[..., None]
    loss = 2 * float(tc["detour"]) * (s2 + 2 * R * s1 + R * R * s0)
    f = (1 / (1 + loss)).mean(-1)
    through = (u["strider"] if "strider" in u else torch.zeros_like(u["large"])) | \
        (u["ign_trees"] if "ign_trees" in u else torch.zeros_like(u["large"]))
    return torch.where(through, torch.ones_like(f), f)


def trunk_keep(u, pw, target, lob, params, rays=(-0.5, 0.0, 0.5)):
    """[B, N] the share of shooter i's projectiles at its target that no trunk stops (trees.md 1, 5: a projectile that
    meets a trunk stops, no damage; the trunks do not count over the projectile's first veg_m metres of flight -
    vegetation_ignore_time x its speed). Each ray (the shooter's front: the middle and the ends, to the target's nearest
    man) is its flight - the low arc as the line of fire's (missile._arc_tan; lob: 45 deg) - sampled at `samples` points:
    past veg_m of path and between a cell's trunk base and top, the path takes lam x the horizontal step; the ray keeps
    exp(-sum), the unit the rays' mean. 1 where the battle's map has no trees or there is no target."""
    if not on(u) or target is None:
        return torch.ones_like(u["x"])
    from tools.nn.sim import missile
    out, _, _ = _device(u["x"].device)
    tc = params.sim["terrain"]["trees"]
    M = int(tc["samples"])
    fcfg = params.sim["missile"].get("friend_los") or {}
    arc = params.sim["missile"].get("arc_los")
    hm = float(arc["muzzle_m"]) if arc else float(fcfg.get("muzzle_m", 1.5))
    t = target.clamp(min=0)[:, :, None]
    th = pw["theta"].gather(2, t)[..., 0]
    reach = pw["reach_j"].gather(2, t)[..., 0].clamp(min=1.0)
    dx, dz = torch.sin(th), torch.cos(th)
    b = u["b"] * math.pi / 180
    fx, fz = torch.sin(b), torch.cos(b)
    sx, sz = torch.cos(b), -torch.sin(b)
    ha = float(fcfg.get("aim_frac", 0.75)) * u["height"].gather(1, t[..., 0])
    tx, tz = u["x"] + dx * reach, u["z"] + dz * reach
    h_t = terrain.height(u, tx, tz) + ha
    v = u["muzzle_v"].clamp(min=1.0)
    veg = u["veg_m"] if "veg_m" in u else torch.zeros_like(v)
    k = (torch.arange(M, device=v.device, dtype=v.dtype) + 0.5) / M
    keep = torch.zeros_like(v)
    for r in rays:
        ox = u["x"] + fx * pw["depth"] / 2 + sx * r * pw["front"]
        oz = u["z"] + fz * pw["depth"] / 2 + sz * r * pw["front"]
        h0 = terrain.height(u, ox, oz) + hm
        L = torch.sqrt((tx - ox) ** 2 + (tz - oz) ** 2).clamp(min=1.0)
        dh = h_t - h0
        tan = missile._arc_tan(v, L, dh)
        if lob is not None:
            tan = torch.where(lob, torch.ones_like(tan), tan)
        a = ((tan * L - dh) / (L * L)).clamp(min=0)
        xs = L[..., None] * k
        y = h0[..., None] + tan[..., None] * xs - a[..., None] * xs * xs
        # the path flown to each sample (the arc's length: horizontal x sqrt(1 + slope2), the slope tan - 2 a x)
        sl = tan[..., None] - 2 * a[..., None] * xs
        step = (L / M)[..., None]
        flown = torch.cumsum(step * torch.sqrt(1 + sl * sl), -1)
        px = ox[..., None] + (tx - ox)[..., None] * k
        pz = oz[..., None] + (tz - oz)[..., None] * k
        lam = _at(u, "lam", px, pz)
        live = (flown > veg[..., None]) & (y <= _at(u, "top", px, pz)) & (y >= _at(u, "base", px, pz) - 0.5)
        keep = keep + torch.exp(-(lam * step * live.float()).sum(-1))
    keep = keep / len(rays)
    mid = u["terr"][:, 0].clamp(min=0)
    has = out["has_trees"][mid] & (u["terr"][:, 0] >= 0)
    # (no unit's attribute takes part in the game's check: the Wood Elves' arrows have the same 42 m - trees.md 1)
    return torch.where(has[:, None] & (target >= 0), keep, torch.ones_like(keep))
