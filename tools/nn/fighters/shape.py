"""The shape of a unit from its soldiers' places (numpy, no torch): the same numbers for the game's probe recordings
(charge_probe 'probe_men': every man's place once a second, decimetres) and for the fighters lab's own men.

Per second of a pair of units (a, its enemy e), each a [n, 2] array of the men's places (m) and the unit's bearing (deg):
  near(r)   men of a within r m of their nearest enemy man (r = 3.15: the database's formed combat distance 2.5 m + a
            man's radius 0.65 - the reach the step replay and the probes count damage by)
  gap       the front's gap: the median nearest-enemy distance of a's men whose nearest enemy is within 6 m (the men in
            the fight; the probes: 1.1-1.4 m at the contact, ~3 m after 20 s)
  front_gap the median nearest-enemy distance of the 15 men nearest the enemy (the front rank's standing distance)
  depth_m   the men's depth along the unit's facing, (p97 - p3) / 0.94 (build/formdepth: ranks = depth / v + 1)
  width_m   the same across it
  wrap      the share of a's men level with or behind the enemy's front men, seen from the enemy (in its frame along its
            facing towards a: closer to its centre than its front men's p90 less 1.5 m) and beside its body - men that
            flowed round the enemy (the game's 'drop' in a long melee)
  contact   any man of a within 3.15 m of a man of e
Routers: along / across the flight (p90 - p10 of the men's places along the router's own displacement over the next
seconds, and across it): the mob's stretch.

unit_features (torch; the rest of the module is numpy): what a unit's men look like - the same function for the lab's
men and for the game's recorded soldiers (soldiersdata.py; the companion from the bridge's state, --shape), every unit
of a batch at once, fixed shapes. Per unit, in its own frame (its bearing; centred on its men's mean):
  men_m2       men per square metre of their width x depth
  width_m, depth_m  the men's spread across / along the facing: sqrt(12) x their standard deviation (a full rectangle's
               side; a lone man 0)
  skew_fwd_m, skew_side_m  the men's mean off the middle of their extent (max + min) / 2, along / across: + the mass at
               the front (the rear thinned) / at the right
  contact      the share of the men within reach (REACH_M) of an enemy man
  holes        the share of the 4 x 4 cells of the box (centre +- width / 2, depth / 2) with no man in it
  front_holes  the same of its front row of 4 cells
  wounded      1 - health / (men x a man's health): the wounds the living carry (health as a share of the start)
  grid 4 x 4   the share of the men in each cell of that box, rows from the rear to the front, cells from left to right
               (a man beyond the box counts in its edge cell)"""
import math

import numpy as np

REACH_M = 3.15


def xy(flat):
    """[n, 2] metres from a probe's flat decimetre list."""
    a = np.asarray(flat, float)
    if a.size < 2:
        return np.zeros((0, 2))
    return a.reshape(-1, 2) / 10.0


def frame(p, cx, cz, bearing_deg):
    """(across, along) of points p [n, 2] in a frame at (cx, cz) facing bearing (deg: facing (sin b, cos b))."""
    b = math.radians(bearing_deg)
    fx, fz = math.sin(b), math.cos(b)
    rx, rz = p[:, 0] - cx, p[:, 1] - cz
    return rx * fz - rz * fx, rx * fx + rz * fz


def nearest(a, e):
    """[n] distance from each man of a to his nearest man of e (inf when e is empty)."""
    if len(a) == 0:
        return np.zeros(0)
    if len(e) == 0:
        return np.full(len(a), np.inf)
    d = np.sqrt(((a[:, None, :] - e[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def extent(v):
    if len(v) < 3:
        return np.nan
    lo, hi = np.percentile(v, [3, 97])
    return float((hi - lo) / 0.94)


def pair_second(a, e, b_a, b_e, radii=(2.5, REACH_M)):
    """One second's numbers of unit a against its enemy e (module docstring)."""
    out = {}
    if len(a) == 0 or len(e) == 0:
        return out
    d = nearest(a, e)
    for r in radii:
        out[f"near_{r:g}"] = int((d <= r).sum())
    out["contact"] = bool((d <= REACH_M).any())
    close = d[d <= 6.0]
    out["gap"] = float(np.median(close)) if len(close) else np.nan
    srt = np.sort(d)
    out["front_gap"] = float(np.median(srt[:15])) if len(srt) >= 15 else np.nan
    ca, cz = a.mean(0)
    acr, alo = frame(a, ca, cz, b_a)
    out["depth_m"] = extent(alo)
    out["width_m"] = extent(acr)
    # wrap: in e's frame facing towards a's centre
    ce = e.mean(0)
    to_a = math.degrees(math.atan2(ca - ce[0], cz - ce[1]))
    ea, el = frame(e, ce[0], ce[1], to_a)
    aa, al = frame(a, ce[0], ce[1], to_a)
    front = np.percentile(el, 90)
    half_w = np.percentile(np.abs(ea), 95)
    beside = (al < front - 1.5) & (np.abs(aa) <= half_w + 4.0)
    out["wrap"] = float(beside.mean())
    out["men"] = len(a)
    return out


def router_second(p_now, p_later):
    """The routing mob's stretch along its flight and across it (m, p90 - p10): the flight is the mob's centre's way from
    now to `p_later` (its places some seconds on)."""
    if len(p_now) < 5 or len(p_later) < 5:
        return {}
    c0, c1 = p_now.mean(0), p_later.mean(0)
    v = c1 - c0
    n = float(np.hypot(*v))
    if n < 1.0:
        return {}
    ux, uz = v / n
    rx, rz = p_now[:, 0] - c0[0], p_now[:, 1] - c0[1]
    along = rx * ux + rz * uz
    across = rx * uz - rz * ux
    q = lambda w: float(np.percentile(w, 90) - np.percentile(w, 10))
    return {"along": q(along), "across": q(across), "men": len(p_now)}


def lane_series(lane, who="a", other="tg", until_s=240.0):
    """Per second after the first contact: [(t - contact, pair_second)] of lane's unit `who` against `other` from its
    'men' rows (charge_probe.load_run's lane: men [(t, a flat, tg flat)]; samples give the bearings)."""
    c = (lane.get("contacts") or {}).get(1)
    if c is None or not lane.get("men"):
        return []
    s = lane["samples"]
    ts = np.array([x[0] for x in s])
    col = {"a": 1, "tg": 2}
    bear = {w: np.array([np.nan if (x[col[w]] or {}).get("b") is None else float(x[col[w]]["b"]) for x in s])
            for w in ("a", "tg")}
    out = []
    for t, fa, ft in lane["men"]:
        if t < c - 0.01 or t > c + until_s:
            continue
        k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        P = {"a": xy(fa), "tg": xy(ft)}
        ba, be = bear[who][k], bear[other][k]
        if not (np.isfinite(ba) and np.isfinite(be)):
            continue
        r = pair_second(P[who], P[other], ba, be)
        if r:
            out.append((t - c, r))
    return out


FEATURES = ("men_m2", "width_m", "depth_m", "skew_fwd_m", "skew_side_m", "contact", "holes", "front_holes", "wounded")
GRID = 4


def unit_features(px, pz, uid, live, side_m, bearing, men0, hp, N, reach=REACH_M, cell_m=3.6, cap=12, hash_size=8192,
                  use_triton=None):
    """[B, N, len(FEATURES) + GRID * GRID] (module docstring). Men [B, M]: px, pz (m), uid (the unit slot, long), live
    (bool), side_m (the man's side); units [B, N]: bearing (deg), men0 (men at the start), hp (health, a share of the
    start). A unit with no living man: zeros. use_triton (None: on CUDA when Triton is there): the men in reach of an
    enemy by the contact kernel (scans.contact_scan) straight from the cell table - the torch way builds the neighbour
    lists [B, M, 9 x cap] (a training batch of 1024 battles x 7168 places: 6 GB a list). Needs torch."""
    import torch
    from tools.nn.fighters import cells
    B, M = px.shape
    dev = px.device
    f = live.float()
    uid = torch.where(live, uid, torch.zeros_like(uid))
    zeros = torch.zeros((B, N), device=dev)
    usum = lambda v: zeros.scatter_add(1, uid, v * f)
    n = usum(torch.ones_like(px))
    nn_ = n.clamp(min=1)
    cx, cz = usum(px) / nn_, usum(pz) / nn_
    br = bearing * (math.pi / 180)
    sb, cb = torch.sin(br).gather(1, uid), torch.cos(br).gather(1, uid)
    rx, rz = px - cx.gather(1, uid), pz - cz.gather(1, uid)
    ac = rx * cb - rz * sb                     # across: (cos b, -sin b)
    al = rx * sb + rz * cb                     # along the facing: (sin b, cos b)
    width = torch.sqrt((usum(ac * ac) / nn_).clamp(min=0) * 12.0)
    depth = torch.sqrt((usum(al * al) / nn_).clamp(min=0) * 12.0)
    big = torch.full((B, N), 1e9, device=dev)
    ext = lambda v, how: (big if how == "amin" else -big).scatter_reduce(
        1, uid, torch.where(live, v, torch.full_like(v, 1e9 if how == "amin" else -1e9)), reduce=how)
    skew_f = -(ext(al, "amax") + ext(al, "amin")) / 2      # the mean (0 here) off the middle of the extent
    skew_s = -(ext(ac, "amax") + ext(ac, "amin")) / 2
    area = width.clamp(min=1.0) * depth.clamp(min=1.0)
    # in reach of an enemy man (the lab's neighbour cells)
    tab = cells.Table(px, pz, live, cell_m, hash_size)
    from tools.nn.fighters.kernels import _triton_ok
    if _triton_ok(px, use_triton):
        from types import SimpleNamespace
        from tools.nn.fighters import scans
        side_u = torch.zeros((B, N), dtype=torch.long, device=dev).scatter_reduce(
            1, uid, torch.where(live, side_m.long(), torch.zeros_like(uid)), reduce="amax")
        near = SimpleNamespace(table=tab, px=px, pz=pz, cap=cap)
        _, hi = scans.contact_scan(near, live, uid, side_u, zeros, reach, use_triton=True)
        touch = (hi >= 0) & live
    else:
        cand = tab.query(px, pz, cap)
        cxm, czm = cells.gather2(px, cand), cells.gather2(pz, cand)
        d2 = (cxm - px[:, :, None]) ** 2 + (czm - pz[:, :, None]) ** 2
        foe = (cand >= 0) & cells.gather2(live, cand) & (cells.gather2(side_m, cand) != side_m[:, :, None])
        touch = (foe & (d2 <= reach * reach)).any(2) & live
    contact = usum(touch.float()) / nn_
    # the 4 x 4 box
    w_m, d_m = width.gather(1, uid).clamp(min=1e-3), depth.gather(1, uid).clamp(min=1e-3)
    ci = torch.clamp(((ac / w_m + 0.5) * GRID).floor(), 0, GRID - 1).long()
    ri = torch.clamp(((al / d_m + 0.5) * GRID).floor(), 0, GRID - 1).long()
    cell = uid * GRID * GRID + ri * GRID + ci
    grid = torch.zeros((B, N * GRID * GRID), device=dev).scatter_add(1, cell, f).reshape(B, N, GRID * GRID)
    holes = (grid == 0).float().mean(2)
    front_holes = (grid[:, :, (GRID - 1) * GRID:] == 0).float().mean(2)
    grid = grid / nn_[:, :, None]
    wounded = (1 - hp * men0 / nn_).clamp(0, 1)
    out = torch.stack([n / area, width, depth, skew_f, skew_s, contact, holes, front_holes, wounded], 2)
    out = torch.cat([out, grid], 2)
    return torch.where((n > 0)[:, :, None], out, torch.zeros_like(out))


def features_from_rows(rows, names, sides, bearings, men0, hp, device="cpu"):
    """unit_features for one battle from places {unit name: [k, 2] m} (a recorded soldiers' row, soldiersdata.Places
    or the bridge's state): names [N] the units' order, sides / bearings / men0 / hp [N]. -> [N, F] (torch)."""
    import torch
    pts, uid = [], []
    for k, nm in enumerate(names):
        p = rows.get(nm)
        if p is not None and len(p):
            pts.append(np.asarray(p, float))
            uid.append(np.full(len(p), k))
    if not pts:
        return torch.zeros((len(names), len(FEATURES) + GRID * GRID))
    P = torch.as_tensor(np.concatenate(pts), dtype=torch.float32, device=device)[None]
    U = torch.as_tensor(np.concatenate(uid), dtype=torch.long, device=device)[None]
    t = lambda a: torch.as_tensor(np.nan_to_num(np.asarray(a, float)), dtype=torch.float32, device=device)[None]
    S = torch.as_tensor(np.asarray(sides), dtype=torch.long, device=device)[None]
    return unit_features(P[..., 0], P[..., 1], U, torch.ones_like(U, dtype=torch.bool), S.gather(1, U), t(bearings),
                         t(men0), t(hp), len(names))[0]
