"""The way round 'cannot stand' ground (build/diag_map/pathing.md: the game's path from the probes 31.10 and the program;
config/nn/sim.json terrain.passable.detour with its sources). Rides on ground.py's 'cannot stand' layer (blk, the map's
passable.npz): a battle on a map without it - or with the switch off - has none of this, bit for bit.

The game (pathing.md, rules 8-12): a unit's centre walks the shortest way of straight legs from corner to corner round
the closed areas grown by r = W / 2 + pad_m (W the formation's front: the probes' best r 14-16 m at W 25-28 m on Kislev,
10 / 16 / 20 m at W 18 / 28 / 38 m at the hamlet - C), holds its course on the next corner on each leg, its formation
neither narrows nor breaks; no limit on the way's length (x1.54 of the straight line walked).

The simulator, per map (once; numpy here - the battle builds the graph on its device, _graph_t):
  clearance(blocked, step)          [R, C] m: each node's distance to the nearest closed node (the grid's border closed)
  corners(F, x0, z0, step, r, ...)  the corners of the areas grown by r: the level r of the clearance traced (marching
                                    squares on a coarser grid), simplified (Douglas-Peucker), its convex vertices pushed
                                    out a little - [K, 2] map x, z
  graph(F, ..., corners, r, ...)    the legs between corners whose straight line keeps clear of r (sampled), and the
                                    shortest ways between every two corners (Floyd-Warshall): D [K, K] m, NX [K, K] the
                                    next corner on the way (-1 none)
The battle (torch, battle.py; the state's pa_* records, terrain.set_map):
  steer(u, params, gx, gz, follow, front, dt, clock)  a unit under a move / withdraw order or closing on its attack's
                                    target, out of melee (follow), heads for its next corner instead of the goal. No
                                    check where the line is surely clear by its ends' clearance; else a new way is
                                    planned (plan; every plan_every_s of the battle's time) when the goal moved off the
                                    old line's margin (a way of corners: by max(goal_m, replan_share x distance)), the
                                    unit had none or its way went bad (a step not made - the 'cannot stand' stop); a
                                    straight line clear of r - no corner: the old step bit for bit
  plan(...)                         (kept out of torch.compile: the units needing a way are picked by nonzero; its
                                    pieces compiled on CUDA, _kernel) the straight line first; else the corners seen
                                    from the unit and from the goal and the shortest way between them:
                                    start -> c_i -> ... -> c_j -> goal
"""
import math

import numpy as np

try:
    import torch
except ImportError:              # (the host's tests: the map's part is numpy alone)
    torch = None

P = 16                           # the corners a unit's way keeps (the probes' longest way: 6)


# --- the map (numpy) ---

def clearance(blocked, step, fmax=64.0):
    """[R, C] float32 m: each node's distance to the nearest closed node (blocked > 0; the nodes just outside the grid
    count as closed - the map's edge), exact (Euclidean between node centres) up to fmax, fmax beyond. A separable exact
    transform: per column the nearest closed node along it, then per row min_k (g_k^2 + (i - k)^2) over |i - k| <= fmax."""
    b = np.asarray(blocked) > 0
    R, C = b.shape
    bp = np.ones((R + 2, C + 2), bool)
    bp[1:-1, 1:-1] = b
    cap = int(math.ceil(fmax / step)) + 1
    idx = np.broadcast_to(np.arange(R + 2)[:, None], bp.shape)
    last = np.maximum.accumulate(np.where(bp, idx, -10 ** 6), axis=0)
    nxt = np.minimum.accumulate(np.where(bp, idx, 10 ** 6)[::-1], axis=0)[::-1]
    g = np.minimum(idx - last, nxt - idx).clip(max=cap).astype(np.float32)
    g2 = g * g
    best = g2.copy()
    for d in range(1, cap + 1):
        dd = np.float32(d * d)
        np.minimum(best[:, d:], g2[:, :-d] + dd, out=best[:, d:])
        np.minimum(best[:, :-d], g2[:, d:] + dd, out=best[:, :-d])
    return np.minimum(np.sqrt(best[1:-1, 1:-1]) * np.float32(step), np.float32(fmax)).astype(np.float32)


def at(F, x0, z0, step, x, z):
    """The clearance at points x, z (numpy, map coordinates): bilinear between the nodes (a distance field is smooth
    off its ridges: the nearest node would be up to 0.7 step off), held at the grid's edge."""
    R, C = F.shape
    fx = np.clip((np.asarray(x, np.float64) - x0) / step, 0, C - 1)
    fz = np.clip((np.asarray(z, np.float64) - z0) / step, 0, R - 1)
    i0 = np.minimum(np.floor(fx).astype(np.int64), C - 2)
    j0 = np.minimum(np.floor(fz).astype(np.int64), R - 2)
    ax, az = fx - i0, fz - j0
    return (F[j0, i0] * (1 - ax) * (1 - az) + F[j0, i0 + 1] * ax * (1 - az) + F[j0 + 1, i0] * (1 - ax) * az
            + F[j0 + 1, i0 + 1] * ax * az)


def contours(v, x0, z0, h):
    """The closed loops of the level 0 of v [R, C] (nodes x0 + i h, z0 + j h; v <= 0 the closed side, the grid's outside
    closed): a list of [n, 2] (x, z) arrays, each walked with the closed side on its left (cross(q - p, c - p) > 0 for
    a closed point c beside a step p -> q). Marching squares with the crossings interpolated; a saddle cell joins its
    closed corners when its centre's mean is closed."""
    R0, C0 = v.shape
    w = np.full((R0 + 2, C0 + 2), -1.0, np.float64)
    w[1:-1, 1:-1] = v
    R, C = w.shape
    a, b, c, d = w[:-1, :-1], w[:-1, 1:], w[1:, 1:], w[1:, :-1]         # (j, i) (j, i+1) (j+1, i+1) (j+1, i)
    case = (a <= 0) * 1 + (b <= 0) * 2 + (c <= 0) * 4 + (d <= 0) * 8
    centre = (a + b + c + d) / 4 <= 0
    jj, ii = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    HE = R * (C - 1)                                                     # horizontal edges first, then vertical

    def edge(e, j, i):
        """(key, x, z) of the crossing on cell (j, i)'s edge e: 0 bottom a-b, 1 right b-c, 2 top d-c, 3 left a-d."""
        if e == 0:
            p, q, key = w[j, i], w[j, i + 1], j * (C - 1) + i
            t = p / (p - q)
            return key, i + t, j.astype(np.float64)
        if e == 1:
            p, q, key = w[j, i + 1], w[j + 1, i + 1], HE + j * C + i + 1
            t = p / (p - q)
            return key, (i + 1).astype(np.float64), j + t
        if e == 2:
            p, q, key = w[j + 1, i], w[j + 1, i + 1], (j + 1) * (C - 1) + i
            t = p / (p - q)
            return key, i + t, (j + 1).astype(np.float64)
        p, q, key = w[j, i], w[j + 1, i], HE + j * C + i
        t = p / (p - q)
        return key, i.astype(np.float64), j + t

    # each case's segments: (edge, edge, the reference corner 0..3 = a b c d) - the reference's closed flag says the side
    simple = {1: [(3, 0, 0)], 14: [(3, 0, 0)], 2: [(0, 1, 1)], 13: [(0, 1, 1)], 4: [(1, 2, 2)], 11: [(1, 2, 2)],
              8: [(2, 3, 3)], 7: [(2, 3, 3)], 3: [(3, 1, 0)], 12: [(3, 1, 0)], 6: [(0, 2, 1)], 9: [(0, 2, 1)]}
    seg_s, seg_e, seg_sx, seg_sz = [], [], [], []

    def add(mask, e0, e1, ref):
        j, i = jj[mask], ii[mask]
        if not len(j):
            return
        k0, x0_, z0_ = edge(e0, j, i)
        k1, x1_, z1_ = edge(e1, j, i)
        cj = j + (ref in (2, 3))
        ci = i + (ref in (1, 2))
        closed = w[cj, ci] <= 0
        cr = (x1_ - x0_) * (cj - z0_) - (z1_ - z0_) * (ci - x0_)
        flip = (cr > 0) != closed                                       # the closed side must be on the left
        seg_s.append(np.where(flip, k1, k0))
        seg_e.append(np.where(flip, k0, k1))
        seg_sx.append(np.where(flip, x1_, x0_))
        seg_sz.append(np.where(flip, z1_, z0_))
    for k, segs in simple.items():
        m = case == k
        for e0, e1, ref in segs:
            add(m, e0, e1, ref)
    # saddles: 5 (a, c closed), 10 (b, d closed)
    for k, joined_cuts, apart_cuts in ((5, [(0, 1, 1), (2, 3, 3)], [(3, 0, 0), (1, 2, 2)]),
                                        (10, [(3, 0, 0), (1, 2, 2)], [(0, 1, 1), (2, 3, 3)])):
        for cen, segs in ((True, joined_cuts), (False, apart_cuts)):
            m = (case == k) & (centre == cen)
            for e0, e1, ref in segs:
                add(m, e0, e1, ref)
    if not seg_s:
        return []
    s, e = np.concatenate(seg_s), np.concatenate(seg_e)
    sx, sz = np.concatenate(seg_sx), np.concatenate(seg_sz)
    by_start = np.full(HE + R * C, -1, np.int64)
    by_start[s] = np.arange(len(s))
    nxt = by_start[e]
    seen = np.zeros(len(s), bool)
    loops = []
    for k0 in range(len(s)):
        if seen[k0]:
            continue
        idx = []
        k = k0
        while k >= 0 and not seen[k]:
            seen[k] = True
            idx.append(k)
            k = nxt[k]
        if len(idx) >= 3:
            idx = np.array(idx)
            loops.append(np.stack([x0 + (sx[idx] - 1) * h, z0 + (sz[idx] - 1) * h], 1))
    return loops


def _dp_open(pts, eps):
    """Douglas-Peucker on an open polyline [n, 2]: the kept points' mask (the ends kept)."""
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        p, q = pts[i0], pts[i1]
        seg = pts[i0 + 1:i1]
        dx, dz = q - p
        L = math.hypot(dx, dz)
        if L < 1e-9:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs((seg[:, 0] - p[0]) * dz - (seg[:, 1] - p[1]) * dx) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i0 + 1 + k
            keep[m] = True
            stack.append((i0, m))
            stack.append((m, i1))
    return keep


def simplify(loop, eps):
    """A closed loop [n, 2] simplified (Douglas-Peucker, eps m), split at its point 0 and the point farthest from it."""
    if len(loop) < 4:
        return loop
    far = int(np.argmax(np.hypot(loop[:, 0] - loop[0, 0], loop[:, 1] - loop[0, 1])))
    a = loop[:far + 1]
    b = np.concatenate([loop[far:], loop[:1]])
    ka, kb = _dp_open(a, eps), _dp_open(b, eps)
    return np.concatenate([a[ka], b[kb][1:-1]])


def corners(F, x0, z0, step, r, contour_m=2.0, eps_m=0.5, push_m=0.75, tol_m=0.5):
    """[K, 2] map x, z: the convex corners of the closed areas grown by r - the level r of the clearance F traced on
    a grid of contour_m, simplified to eps_m, the vertices where the walk (closed side on the left) turns left, each
    pushed out by push_m along its two edges' outward normals (the legs between neighbours then keep clear of r);
    corners whose clearance is under r - tol_m (a gap narrower than the growth) are dropped."""
    s = max(1, int(round(contour_m / step)))
    G = F[::s, ::s].astype(np.float64) - r
    out = []
    for loop in contours(G, x0, z0, step * s):
        poly = simplify(loop, eps_m)
        n = len(poly)
        if n < 3:
            continue
        prv, nxt = np.roll(poly, 1, 0), np.roll(poly, -1, 0)
        e1, e2 = poly - prv, nxt - poly
        cr = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        l1 = np.hypot(e1[:, 0], e1[:, 1]).clip(min=1e-9)[:, None]
        l2 = np.hypot(e2[:, 0], e2[:, 1]).clip(min=1e-9)[:, None]
        # outward (the free side: right of the walk) normals of the two edges, their sum's direction
        n1 = np.stack([e1[:, 1], -e1[:, 0]], 1) / l1
        n2 = np.stack([e2[:, 1], -e2[:, 0]], 1) / l2
        bis = n1 + n2
        bl = np.hypot(bis[:, 0], bis[:, 1])[:, None]
        cos_half = (bl / 2).clip(min=0.5)
        node = poly + bis / bl.clip(min=1e-9) * (push_m / cos_half)
        ok = cr > 1e-9
        out.append(node[ok])
    if not out:
        return np.zeros((0, 2), np.float32)
    c = np.concatenate(out)
    c = c[at(F, x0, z0, step, c[:, 0], c[:, 1]) > r - tol_m]
    return c.astype(np.float32)


def clear_np(F, x0, z0, step, ax, az, bx, bz, thr, h=2.0, chunk=1 << 22):
    """[n] bool: the straight lines a -> b (arrays [n]) keep the clearance above thr ([n] or a number) at every sample
    (h m apart at most, both ends included)."""
    ax, az, bx, bz = (np.asarray(v, np.float64) for v in (ax, az, bx, bz))
    thr = np.broadcast_to(np.asarray(thr, np.float64), ax.shape)
    L = np.hypot(bx - ax, bz - az)
    out = np.ones(len(ax), bool)
    order = np.argsort(L)
    Ls = L[order]
    k = 0
    while k < len(order):
        # (sorted by length: a chunk of lines at most twice the first one's length, ~chunk samples)
        S0 = int(math.ceil(Ls[k] / h)) + 2
        last = int(np.searchsorted(Ls, 2 * Ls[k] + 2 * h, side="right"))
        n = max(1, min(chunk // S0, last - k))
        sel = order[k:k + n]
        S = int(math.ceil(Ls[k + n - 1] / h)) + 2
        t = np.linspace(0.0, 1.0, S)[None, :]
        px = ax[sel, None] + (bx - ax)[sel, None] * t
        pz = az[sel, None] + (bz - az)[sel, None] * t
        out[sel] = (at(F, x0, z0, step, px, pz) > thr[sel, None]).all(1)
        k += n
    return out


def graph(F, x0, z0, step, c, r, tol_m=0.5, h=2.0):
    """The corners' graph: D [K, K] float32 the shortest way between corners i and j along legs clear of r - tol_m
    (inf: none), NX [K, K] int32 the corner after i on the way to j (j itself on a straight leg; -1 none; i on i)."""
    K = len(c)
    D = np.full((K, K), np.inf, np.float64)
    NX = np.full((K, K), -1, np.int64)
    if K:
        iu, ju = np.triu_indices(K, 1)
        vis = clear_np(F, x0, z0, step, c[iu, 0], c[iu, 1], c[ju, 0], c[ju, 1], r - tol_m, h)
        L = np.hypot(c[iu, 0] - c[ju, 0], c[iu, 1] - c[ju, 1])
        a, b, L = iu[vis], ju[vis], L[vis]
        D[a, b] = D[b, a] = L
        NX[a, b], NX[b, a] = b, a
        np.fill_diagonal(D, 0.0)
        NX[np.arange(K), np.arange(K)] = np.arange(K)
        for k in range(K):
            alt = D[:, k:k + 1] + D[k:k + 1, :]
            better = alt < D
            if better.any():
                D = np.where(better, alt, D)
                NX = np.where(better, NX[:, k:k + 1], NX)
    return D.astype(np.float32), NX.astype(np.int32)


def route_np(F, x0, z0, step, c, D, NX, s, g, r, tol_m=0.5, h=2.0):
    """The way (numpy; the checks' and the tests' twin of plan) from s to g (x, z) for growth r: [n, 2] the corners
    to walk through (none: the straight line is clear), or None (no way)."""
    fs, fg = (float(at(F, x0, z0, step, p[0], p[1])) for p in (s, g))
    if clear_np(F, x0, z0, step, [s[0]], [s[1]], [g[0]], [g[1]], min(r, fs, fg) - tol_m, h)[0]:
        return np.zeros((0, 2), np.float32)
    K = len(c)
    if not K:
        return None
    vs = clear_np(F, x0, z0, step, np.full(K, s[0]), np.full(K, s[1]), c[:, 0], c[:, 1], min(r, fs) - tol_m, h)
    vg = clear_np(F, x0, z0, step, c[:, 0], c[:, 1], np.full(K, g[0]), np.full(K, g[1]), min(r, fg) - tol_m, h)
    ds = np.where(vs, np.hypot(c[:, 0] - s[0], c[:, 1] - s[1]), np.inf)
    dg = np.where(vg, np.hypot(c[:, 0] - g[0], c[:, 1] - g[1]), np.inf)
    tot = ds[:, None] + D + dg[None, :]
    k = int(np.argmin(tot))
    i, j = divmod(k, K)
    if not np.isfinite(tot[i, j]):
        return None
    way = [i]
    while way[-1] != j and len(way) < K:
        way.append(int(NX[way[-1], j]))
    return c[way]


# --- the battle (torch) ---

DEFAULTS = {"on": True, "pad_m": 1.5, "r_step_m": 1.0, "r_min_m": 2.0, "contour_m": 2.0, "eps_m": 0.5, "push_m": 0.75,
            "tol_m": 0.5, "sample_m": 2.0, "fmax_m": 256.0, "reach_m": 2.0, "goal_m": 1.0, "replan_share": 0.1,
            "plan_every_s": 1.0,
            "narrow": "smallest"}
_F = {}                  # (map id, the layer's array id, fmax) -> (clearance numpy [R, C], x0, z0, step)
_TAB = {}                # (map id, the layer's id, fmax, q, settings) -> (corners [K, 2], D [K, K], NX [K, K]) numpy
_DEV = {}                # (device, ...) -> the same on the device
STATE = ("pa_x", "pa_z", "pa_r", "pa_n", "pa_i", "pa_gx", "pa_gz", "pa_sx", "pa_sz", "pa_m", "pa_ok", "pa_c", "pa_nf")


def cfg_of(params):
    """terrain.passable.detour (config/nn/sim.json) over DEFAULTS, or None when it is off or absent."""
    pc = params.sim["terrain"].get("passable") or {}
    d = pc.get("detour")
    if not d:
        return None
    out = dict(DEFAULTS)
    out.update({k: v for k, v in d.items() if not k.endswith("why")})
    return out if out.get("on") else None


def add_state(u):
    """The ways' records (terrain.set_map on a map with the 'cannot stand' layer): per unit up to P corners in the
    battle's frame (pa_x, pa_z [B, N x P]) with the way left from each to the goal (pa_r), their count pa_n and the next
    one pa_i, the goal the way was planned for (pa_gx, pa_gz) and the unit's place then (pa_sx, pa_sz), a straight
    line's margin over the growth (pa_m; -1: a way of corners), whether it holds (pa_ok); for the checks - no rule reads
    them - the ways planned round closed ground (pa_c) and the ones not found (pa_nf)."""
    if "pa_n" in u:
        return
    B, N = u["x"].shape
    for k in ("pa_x", "pa_z", "pa_r"):
        u[k] = torch.zeros((B, N * P), dtype=u["x"].dtype, device=u["x"].device)
    u["pa_n"] = torch.zeros((B, N), dtype=torch.int64, device=u["x"].device)
    u["pa_i"] = torch.zeros_like(u["pa_n"])
    for k in ("pa_gx", "pa_gz", "pa_sx", "pa_sz", "pa_c", "pa_nf"):
        u[k] = torch.zeros_like(u["x"])
    u["pa_m"] = torch.full_like(u["x"], -1.0)
    u["pa_ok"] = torch.zeros_like(u["x"], dtype=torch.bool)


def _map(mid, cfg):
    from tools.nn.sim import ground, terrain
    L = ground._LAYERS[mid]
    k = (mid, id(L["blk"]), float(cfg["fmax_m"]))
    if k not in _F:
        _g, x0, z0, step = terrain._MAPS[mid]
        _F[k] = (clearance(L["blk"], step, float(cfg["fmax_m"])), x0, z0, step)
    return k, _F[k]


def _graph_t(F, x0, z0, step, c, r, tol_m, h, dev):
    """graph() on the device (the same legs and ways; numpy's twin for the tests): built the first time a map and a
    growth are met in a battle."""
    K = len(c)
    if K == 0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.int32)
    Fd = torch.from_numpy(F.reshape(-1)).to(dev)
    geo = (x0, z0, step, F.shape[0], F.shape[1])
    cd = torch.from_numpy(c).to(dev).double()
    iu, ju = torch.triu_indices(K, K, 1, device=dev)
    vis = _clear_t(Fd, geo, cd[iu, 0].float(), cd[iu, 1].float(), cd[ju, 0].float(), cd[ju, 1].float(),
                   torch.full((len(iu),), r - tol_m, device=dev), h)
    L = torch.sqrt((cd[iu, 0] - cd[ju, 0]) ** 2 + (cd[iu, 1] - cd[ju, 1]) ** 2)
    D = torch.full((K, K), float("inf"), dtype=torch.float64, device=dev)
    NX = torch.full((K, K), -1, dtype=torch.int64, device=dev)
    a, b, L = iu[vis], ju[vis], L[vis]
    D[a, b] = L
    D[b, a] = L
    NX[a, b], NX[b, a] = b, a
    ar = torch.arange(K, device=dev)
    D[ar, ar] = 0.0
    NX[ar, ar] = ar
    for k in range(K):
        alt = D[:, k:k + 1] + D[k:k + 1, :]
        better = alt < D
        D = torch.where(better, alt, D)
        NX = torch.where(better, NX[:, k:k + 1].expand(K, K), NX)
    return D.float().cpu().numpy(), NX.int().cpu().numpy()


def _table(mid, q, cfg, dev=None):
    fk, (F, x0, z0, step) = _map(mid, cfg)
    r = q * float(cfg["r_step_m"])
    k = fk + (q,) + tuple(float(cfg[s]) for s in ("r_step_m", "contour_m", "eps_m", "push_m", "tol_m", "sample_m"))
    if k not in _TAB:
        c = corners(F, x0, z0, step, r, float(cfg["contour_m"]), float(cfg["eps_m"]), float(cfg["push_m"]),
                    float(cfg["tol_m"]))
        if dev is not None and torch is not None:
            D, NX = _graph_t(F, x0, z0, step, c, r, float(cfg["tol_m"]), float(cfg["sample_m"]), dev)
        else:
            D, NX = graph(F, x0, z0, step, c, r, float(cfg["tol_m"]), float(cfg["sample_m"]))
        _TAB[k] = (c, D, NX)
    return k, _TAB[k]


def _dev(key, make, dev):
    k = (str(dev),) + key
    if k not in _DEV:
        _DEV[k] = make()
    return _DEV[k]


def _stack(cfg, dev):
    """Every registered map's clearance stacked on the device (flat [M x Rm x Cm]) with its grids (x0, z0, step, R, C
    [M]): a map without the 'cannot stand' layer reads 1e9 (no closed ground) - the per-step test (clear_here)."""
    from tools.nn.sim import ground, terrain
    M = len(terrain._MAPS)
    key = ("stack", M, tuple(id(ground._LAYERS[k]["blk"]) if ground.has_passable(k) else 0 for k in range(M)),
           float(cfg["fmax_m"]))
    if (str(dev),) + key not in _DEV:
        Rm = max(g.shape[0] for g, *_ in terrain._MAPS)
        Cm = max(g.shape[1] for g, *_ in terrain._MAPS)
        a = np.full((M, Rm, Cm), 1e9, np.float32)
        for k in range(M):
            if ground.has_passable(k):
                F = _map(k, cfg)[1][0]
                a[k, :F.shape[0], :F.shape[1]] = F
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)       # noqa: E731
        _DEV[(str(dev),) + key] = (torch.from_numpy(a.reshape(-1)).to(dev), t([m[1] for m in terrain._MAPS]),
                                   t([m[2] for m in terrain._MAPS]), t([m[3] for m in terrain._MAPS]),
                                   t([g.shape[0] for g, *_ in terrain._MAPS], torch.int64),
                                   t([g.shape[1] for g, *_ in terrain._MAPS], torch.int64), Rm, Cm)
    return _DEV[(str(dev),) + key]


def clear_here(u, x, z, cfg):
    """[B, ...] the clearance (m) at points x, z [B, ...] of each battle's map (bilinear; 1e9 on a flat battle and on a
    map without the layer)."""
    from tools.nn.sim import terrain
    G, X0, Z0, S, R, C, Rm, Cm = _stack(cfg, x.device)
    mx, mz = terrain.to_map(u, x, z)
    mid = u["terr"][:, 0]
    m = mid.clamp(min=0)
    sh = (-1,) + (1,) * (x.dim() - 1)
    Rb, Cb = R[m].view(sh), C[m].view(sh)
    fx = torch.minimum(((mx - X0[m].view(sh)) / S[m].view(sh)).clamp(min=0), (Cb - 1).float())
    fz = torch.minimum(((mz - Z0[m].view(sh)) / S[m].view(sh)).clamp(min=0), (Rb - 1).float())
    i0 = torch.minimum(fx.floor().long(), Cb - 2)
    j0 = torch.minimum(fz.floor().long(), Rb - 2)
    ax, az = fx - i0, fz - j0
    k = m.view(sh) * (Rm * Cm) + j0 * Cm + i0

    def g(o):
        return G[(k + o).reshape(-1)].view(x.shape)
    f = g(0) * (1 - ax) * (1 - az) + g(1) * ax * (1 - az) + g(Cm) * (1 - ax) * az + g(Cm + 1) * ax * az
    return torch.where((mid < 0).view(sh), torch.full_like(f, 1e9), f)


_FN = {}                 # name -> the planner's pieces compiled (CUDA)


def _kernel(name, fn, dev):
    """fn compiled on CUDA (torch.compile, dynamic shapes, the eager numbers - compiled.py: the planner's many small
    operations in a few kernels; a plan call was ~1000 kernel launches, CPU-bound), as it is elsewhere."""
    if torch.device(dev).type != "cuda":
        return fn
    if name not in _FN:
        from tools.nn.sim.compiled import compiled
        _FN[name] = compiled(fn, dynamic=True)
    return _FN[name]


def _bil(Fd, x0, z0, step, R, C, x, z):
    fx = ((x - x0) / step).clamp(0, C - 1)
    fz = ((z - z0) / step).clamp(0, R - 1)
    i0 = fx.floor().long().clamp(max=C - 2)
    j0 = fz.floor().long().clamp(max=R - 2)
    ax, az = fx - i0, fz - j0
    k = j0 * C + i0
    return (Fd[k] * (1 - ax) * (1 - az) + Fd[k + 1] * ax * (1 - az) + Fd[k + C] * (1 - ax) * az
            + Fd[k + C + 1] * ax * az)


def _at_t(Fd, geo, x, z):
    """at() on the device: Fd the clearance flat [R x C], geo (x0, z0, step, R, C)."""
    return _kernel("bil", _bil, x.device)(Fd, *geo, x, z)


def _margin(Fd, x0, z0, step, R, C, ax, az, bx, bz, thr, t):
    px = ax[:, None] + (bx - ax)[:, None] * t
    pz = az[:, None] + (bz - az)[:, None] * t
    return (_bil(Fd, x0, z0, step, R, C, px, pz) - thr[:, None]).amin(1)


def _clear_t(Fd, geo, ax, az, bx, bz, thr, h, chunk=1 << 22, margin=False):
    """[n] bool (torch): the straight lines a -> b keep the clearance above thr at every sample (h m apart at most);
    margin: also [n] the least clearance over thr along each line."""
    n = ax.shape[0]
    if n == 0:
        out = torch.ones(0, dtype=torch.bool, device=ax.device)
        return (out, torch.zeros(0, dtype=ax.dtype, device=ax.device)) if margin else out
    # (as many samples as the map's diagonal needs, whatever the lines: no wait for the device to say the longest)
    S = int(math.ceil(math.hypot(geo[3], geo[4]) * geo[2] / h)) + 2
    t = torch.linspace(0.0, 1.0, S, device=ax.device, dtype=ax.dtype)
    rows = max(1, chunk // S)
    fn = _kernel("margin", _margin, ax.device)
    marg = torch.cat([fn(Fd, *geo, ax[k:k + rows], az[k:k + rows], bx[k:k + rows], bz[k:k + rows], thr[k:k + rows], t)
                      for k in range(0, n, rows)])
    out = marg > 0
    return (out, marg) if margin else out


def _grid(mid, cfg, dev):
    fk, (F, x0, z0, step) = _map(mid, cfg)
    return fk, _dev(fk, lambda: torch.from_numpy(F.reshape(-1)).to(dev), dev), (x0, z0, step, F.shape[0], F.shape[1])


def _thresholds(Fd, geo, sx, sz, tx, tz, r, tol):
    """(a unit or a goal nearer a closed area than r: its own leg keeps the clearance it has - pathing.md's proposal 2)
    the thresholds of the start's and the goal's legs."""
    return torch.minimum(r, _at_t(Fd, geo, sx, sz)) - tol, torch.minimum(r, _at_t(Fd, geo, tx, tz)) - tol


_STACK = {}              # (device, map id, the layer's id, settings) -> (growths built [list], the stacked tables)


def _tables(mid, qs, cfg, dev):
    """The corners' tables of every growth of map mid built so far (with qs, a list, among them) stacked on the device,
    padded to the largest: (growths [Q] sorted, corners [Q, K, 2] (padding: the first), their mask [Q, K], D [Q, K, K]
    (inf), NX [Q, K, K] (-1)); stacked anew only when a growth is new."""
    from tools.nn.sim import ground
    key = (str(dev), mid, id(ground._LAYERS[mid]["blk"])) + tuple(
        float(cfg[s]) for s in ("fmax_m", "r_step_m", "contour_m", "eps_m", "push_m", "tol_m", "sample_m"))
    have = _STACK.get(key)
    if have is None or not set(qs) <= set(have[0]):
        built = sorted(set(qs) | set(have[0] if have else []))
        tabs = [_table(mid, q, cfg, dev)[1] for q in built]
        K = max(1, max(len(c) for c, _, _ in tabs))
        Q = len(built)
        C = np.full((Q, K, 2), 1e7, np.float32)
        Mk = np.zeros((Q, K), bool)
        D = np.full((Q, K, K), np.inf, np.float32)
        NX = np.full((Q, K, K), -1, np.int64)
        for i, (c, d, nx) in enumerate(tabs):
            k = len(c)
            C[i, :k], Mk[i, :k], D[i, :k, :k], NX[i, :k, :k] = c, True, d, nx
            C[i, k:] = c[0] if k else 0.0             # (the padding on a real corner: its lines stay on the map)
        _STACK[key] = have = (built, torch.tensor(built, device=dev),
                              *(torch.from_numpy(v).to(dev) for v in (C, Mk, D, NX)))
    return have


def _route(mid, q, sx, sz, tx, tz, cfg, dev):
    """The ways (map frame) from s to t [m] whose straight line is not clear, each at its growth q [m] x r_step (one
    call for every growth of the map): (count [m], corners [m, P, 2], the way left from each to the goal [m, P], not
    found [m])."""
    _fk, Fd, geo = _grid(mid, cfg, dev)
    Mb = sx.shape[0]
    _built, qt, Ct, Mk, Dt, NXt = _tables(mid, q.unique().tolist(), cfg, dev)
    qi = torch.searchsorted(qt, q)
    tol, h = float(cfg["tol_m"]), float(cfg["sample_m"])
    K = Ct.shape[1]
    thr_s, thr_g = _thresholds(Fd, geo, sx, sz, tx, tz, q.to(sx.dtype) * float(cfg["r_step_m"]), tol)
    cd = Ct[qi].to(sx.dtype)                                           # [Mb, K, 2]
    mk = Mk[qi]

    def rep(v):
        return v[:, None].expand(Mb, K).reshape(-1)
    cx, cz = cd[:, :, 0].reshape(-1), cd[:, :, 1].reshape(-1)
    # (the padding's corners are not seen)
    vs = _clear_t(Fd, geo, rep(sx), rep(sz), cx, cz, rep(thr_s), h).view(Mb, K) & mk
    vg = _clear_t(Fd, geo, cx, cz, rep(tx), rep(tz), rep(thr_g), h).view(Mb, K) & mk
    rows = max(1, (1 << 22) // (K * K))
    pick = _kernel("pick", _pick, dev)
    parts = [pick(vs[k:k + rows], vg[k:k + rows], cd[k:k + rows], sx[k:k + rows], sz[k:k + rows], tx[k:k + rows],
                  tz[k:k + rows], Dt, qi[k:k + rows]) for k in range(0, Mb, rows)]
    best, arg = (torch.cat([p_[i] for p_ in parts]) for i in (0, 1))
    return _kernel("walk", _walk, dev)(NXt, qi, cd, arg, best, tx, tz)


def _pick(vs, vg, cd, sx, sz, tx, tz, Dt, qi):
    """The shortest start -> c_i -> ... -> c_j -> goal over the corners seen (vs from the start, vg from the goal):
    (its length [m] - inf: none, i x K + j [m])."""
    K = cd.shape[1]
    inf = torch.full_like(cd[:, :, 0], float("inf"))
    ds = torch.where(vs, torch.sqrt((cd[:, :, 0] - sx[:, None]) ** 2 + (cd[:, :, 1] - sz[:, None]) ** 2), inf)
    dg = torch.where(vg, torch.sqrt((cd[:, :, 0] - tx[:, None]) ** 2 + (cd[:, :, 1] - tz[:, None]) ** 2), inf)
    tot = (ds[:, :, None] + Dt[qi] + dg[:, None, :]).reshape(-1, K * K)
    return tot.min(1)


def _walk(NXt, qi, cd, arg, best, tx, tz):
    """The corners of the ways picked (_pick) by the next-corner tables: (count, corners [m, P, 2], the way left from
    each to the goal [m, P], not found [m])."""
    Mb, K = cd.shape[0], cd.shape[1]
    found = torch.isfinite(best)
    i0, j0 = arg // K, arg % K
    cols = [i0]
    cur = i0
    for _p in range(1, P):
        live_p = (cols[-1] >= 0) & (cols[-1] != j0)
        nx = NXt[qi, cur, j0]
        cols.append(torch.where(live_p, nx, torch.full_like(nx, -1)))
        cur = torch.where(live_p, nx, cur)
    way = torch.stack(cols, 1)
    way = torch.where(found[:, None], way, torch.full_like(way, -1))
    n_w = (way >= 0).sum(1)
    pts = cd.gather(1, way.clamp(min=0)[:, :, None].expand(Mb, P, 2))  # [Mb, P, 2]
    goal = torch.stack([tx, tz], 1)[:, None, :].expand(Mb, P, 2)
    idx = torch.arange(P, device=cd.device)[None, :]
    nxt = torch.where(((idx + 1) < n_w[:, None])[:, :, None], torch.roll(pts, -1, 1), goal)
    seg = torch.where(idx < n_w[:, None], torch.sqrt(((nxt - pts) ** 2).sum(2)), torch.zeros_like(pts[:, :, 0]))
    W = torch.where((idx < n_w[:, None])[:, :, None], pts, torch.zeros_like(pts))
    rem = torch.flip(torch.cumsum(torch.flip(seg, [1]), 1), [1])
    return n_w, W, rem, ~found


def _eager(fn):
    """fn kept out of torch.compile (battle.stepper): the units needing a way are picked by nonzero, the corners'
    tables are built on the host the first time a map and a growth are met (magic._eager's reason)."""
    return torch.compiler.disable(fn) if torch is not None else fn


@_eager
def plan(u, need, gx, gz, front, cfg):
    """New ways for the units need [B, N] to the goals gx, gz (the battle's frame) at their fronts' growth; in place
    (the state's pa_* records)."""
    if not bool(need.any()):
        return
    from tools.nn.sim import ground, terrain
    B, N = need.shape
    dev = need.device
    b, n = need.nonzero(as_tuple=True)
    M = b.shape[0]
    mid = u["terr"][b, n].long()
    X, Z = terrain.to_map(u, u["x"], u["z"])
    GX, GZ = terrain.to_map(u, gx, gz)
    sx, sz, tx, tz = X[b, n], Z[b, n], GX[b, n], GZ[b, n]
    rs = float(cfg["r_step_m"])
    qmin = max(1, int(round(float(cfg["r_min_m"]) / rs)))
    q = torch.round((front[b, n] / 2 + float(cfg["pad_m"])) / rs).long().clamp(min=qmin)
    cnt = torch.zeros(M, dtype=torch.int64, device=dev)
    W = torch.zeros((M, P, 2), dtype=sx.dtype, device=dev)
    rem = torch.zeros((M, P), dtype=sx.dtype, device=dev)
    nf = torch.zeros(M, dtype=torch.bool, device=dev)
    mg = torch.full((M,), -1.0, dtype=sx.dtype, device=dev)
    tol, h = float(cfg["tol_m"]), float(cfg["sample_m"])
    for m_ in torch.unique(mid).tolist():
        if m_ < 0 or not ground.has_passable(m_):
            continue
        sel = (mid == m_).nonzero(as_tuple=True)[0]
        # the straight lines first (all of the map's units at once, each at its own growth)
        _fk, Fd, geo = _grid(m_, cfg, dev)
        thr_s, thr_g = _thresholds(Fd, geo, sx[sel], sz[sel], tx[sel], tz[sel], q[sel].to(sx.dtype) * rs, tol)
        ok, mg_ = _clear_t(Fd, geo, sx[sel], sz[sel], tx[sel], tz[sel], torch.minimum(thr_s, thr_g), h, margin=True)
        # (the samples h apart: between two the clearance may be up to h / 2 lower)
        mg[sel] = torch.where(ok, (mg_ - h / 2).clamp(min=0), torch.full_like(mg_, -1.0))
        bl = sel[~ok]
        if len(bl) == 0:
            continue
        c_, w_, r_, f_ = _route(m_, q[bl], sx[bl], sz[bl], tx[bl], tz[bl], cfg, dev)
        if cfg.get("narrow") == "smallest" and bool((f_ & (q[bl] > qmin)).any()):
            # no way at the front's growth (a gap narrower than the formation): the narrowest growth's way
            # (terrain.passable.detour narrow_why: not seen in the game)
            s2 = (f_ & (q[bl] > qmin)).nonzero(as_tuple=True)[0]
            b2 = bl[s2]
            c2, w2, r2, f2 = _route(m_, torch.full_like(q[b2], qmin), sx[b2], sz[b2], tx[b2], tz[b2], cfg, dev)
            c_[s2], w_[s2], r_[s2], f_[s2] = c2, w2, r2, f2
        cnt[bl], W[bl], rem[bl], nf[bl] = c_, w_, r_, f_
    wx, wz = W[:, :, 0], W[:, :, 1]
    if "tpx" in u:
        # (the corners back to the battle's own frame: terrain.to_map's inverse)
        ang = u["tpr"][b, 0][:, None] * (math.pi / 180)
        co, si = torch.cos(ang), torch.sin(ang)
        dX, dZ = wx - u["tpx"][b, 0][:, None], wz - u["tpz"][b, 0][:, None]
        wx, wz = dX * co - dZ * si, dX * si + dZ * co
    for k, v in (("pa_x", wx), ("pa_z", wz), ("pa_r", rem)):
        t = u[k].view(B, N, P).clone()
        t[b, n] = v.to(t.dtype)
        u[k] = t.view(B, N * P)
    u["pa_n"] = u["pa_n"].index_put((b, n), cnt)
    u["pa_i"] = u["pa_i"].index_put((b, n), torch.zeros_like(cnt))
    u["pa_gx"] = u["pa_gx"].index_put((b, n), gx[b, n])
    u["pa_gz"] = u["pa_gz"].index_put((b, n), gz[b, n])
    u["pa_sx"] = u["pa_sx"].index_put((b, n), u["x"][b, n])
    u["pa_sz"] = u["pa_sz"].index_put((b, n), u["z"][b, n])
    u["pa_m"] = u["pa_m"].index_put((b, n), mg.to(u["pa_m"].dtype))
    u["pa_ok"] = u["pa_ok"].index_put((b, n), torch.ones_like(nf))
    u["pa_c"] = u["pa_c"].index_put((b, n), (cnt > 0).to(u["pa_c"].dtype), accumulate=True)
    u["pa_nf"] = u["pa_nf"].index_put((b, n), nf.to(u["pa_nf"].dtype), accumulate=True)


def steer(u, params, gx, gz, follow, front, dt, clock=None):
    """The goal of each unit this step: its next corner where it walks a way round closed ground, else gx, gz as they
    are. follow [B, N]: the units heading for a point (a move / withdraw order) or closing on an attack's target, out of
    melee; front [B, N] their formations' fronts (W); clock [B] the battles' clocks (None: a plan every step) - new ways are
    planned every plan_every_s of a battle's time (from its start), a unit given a new goal meanwhile walks straight to it
    or on its old way. Returns (gx, gz, has [B, N] on a way, the way left to the goal
    [B, N]: movement.velocity brakes on it, not on the corner)."""
    cfg = cfg_of(params)
    B, N = gx.shape
    ok = u["pa_ok"] & follow
    dx, dz = gx - u["x"], gz - u["z"]
    tol = torch.clamp(float(cfg["replan_share"]) * torch.sqrt(dx * dx + dz * dz), min=float(cfg["goal_m"]))
    dg2 = (gx - u["pa_gx"]) ** 2 + (gz - u["pa_gz"]) ** 2
    # a straight line found clear with a margin stays clear while the new line lies within it of the old one: the goal
    # moved less and the unit is off the old line by less (the new line is within the larger of the two of the old;
    # the clearance is 1-Lipschitz) - no new check needed (a moving attack target, a re-given order)
    ax, az = u["pa_sx"], u["pa_sz"]
    ex, ez = u["pa_gx"] - ax, u["pa_gz"] - az
    t = (((u["x"] - ax) * ex + (u["z"] - az) * ez) / (ex * ex + ez * ez).clamp(min=1e-9)).clamp(0, 1)
    off2 = (u["x"] - ax - t * ex) ** 2 + (u["z"] - az - t * ez) ** 2
    m2 = u["pa_m"].clamp(min=0) ** 2
    moved = torch.where(u["pa_m"] >= 0, (dg2 > m2) | (off2 > m2), dg2 > tol * tol)
    # a line surely clear (the clearance is 1-Lipschitz: every point of it has at least (F(start) + F(goal) - its
    # length) / 2): no check, no corner - the old step; such a unit's way (if any) is dropped
    L = torch.sqrt(dx * dx + dz * dz)
    r = front / 2 + float(cfg["pad_m"]) + float(cfg["r_step_m"])
    sure = follow & ((clear_here(u, u["x"], u["z"], cfg) + clear_here(u, gx, gz, cfg) - L) / 2 > r)
    u["pa_n"] = torch.where(sure, torch.zeros_like(u["pa_n"]), u["pa_n"])
    ok = ok & ~sure
    need = follow & ~sure & (~ok | moved)
    if clock is not None:
        k = max(1, int(round(float(cfg["plan_every_s"]) / float(dt))))
        need = need & ((torch.round(clock / float(dt)).long() % k) == 0)[:, None]
    u["pa_ok"] = ok
    plan(u, need, gx, gz, front, cfg)
    base = (torch.arange(N, device=gx.device) * P)[None, :]
    spd = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    reach = torch.clamp(spd * dt, min=float(cfg["reach_m"]))

    def corner():
        i = base + u["pa_i"].clamp(max=P - 1)
        wx, wz, wr = u["pa_x"].gather(1, i), u["pa_z"].gather(1, i), u["pa_r"].gather(1, i)
        return wx, wz, wr, torch.sqrt((wx - u["x"]) ** 2 + (wz - u["z"]) ** 2)
    # (a unit whose way went bad and is not planned anew yet walks straight: pa_ok)
    has = follow & u["pa_ok"] & (u["pa_i"] < u["pa_n"])
    wx, wz, wr, d = corner()
    u["pa_i"] = torch.where(has & (d <= reach), u["pa_i"] + 1, u["pa_i"])
    has = follow & u["pa_ok"] & (u["pa_i"] < u["pa_n"])
    wx, wz, wr, d = corner()
    return torch.where(has, wx, gx), torch.where(has, wz, gz), has, d + wr


def spoil(u, stopped):
    """The ways of the units whose step was not made (stopped [B, N]: the 'cannot stand' stop) go bad: new ones next
    step (the game's 'path invalid' -> a new path, pathing.md rule 5)."""
    u["pa_ok"] = u["pa_ok"] & ~stopped
