"""The group's zone by its formation (tmp/tactic_slice/stage0_results.md section 1; net/config/tactic.json
orders.zone_fit): the grid of tiles and the zone's centre such that the first frame the tactic level gives each unit
(48 x 48, the open cell whose nudged frame is nearest the unit: orders.frame_walk on uzone.Seq) lies within
reissue.MOVE_M of the unit's deployed place - so that 'keep' means 'hold where deployed'. Plain geometry, no torch but
the check; the zone's turn is the row's course (the units' facing), so every frame faces as its unit does.

    fit(places, course)         closed form: the fewest 80 m tiles (rows x cols >= n, <= zone.MAX_TILES) whose rectangle
                                holds every unit's 48 m frame (rows nearer than 48 m pushed apart, half each way); across
                                - the centre moved by less than a cell so the first unit stands on a cell centre, clean
                                edges; forward - the front row's frames flush with the zone's front edge
    candidates(places, course)  the zones the search tries: the grids of the 3 fewest tile counts x centre shifts
    search(places, course)      the candidates through the real first-frame walk: none moved over MOVE_M, the fewest
                                tiles; none - the least largest move
    zone_for(places, course)    fit if it passes the walk, else search; cached by the formation's shape (the places in
                                the course's axes about their middle, to 0.1 m): every battle of one size and layout is the
                                same geometry, so a group costs the search once
Each returns {cx, cz, heading, line, rows, cols} (zone_for adds 'worst': the largest first move, m).
"""
import math

import torch

from net.common import reissue
from net.tactic import cfg
from net.tactic import zone as ZN

TILE, MARGIN, C = 80.0, 80.0, 30        # (net/config/tactic.json tile_m, margin_m, cells)


def _frame():
    return float(cfg()["orders"]["frame_m"])


def _cell(rows, cols):
    return (max(rows, cols) * TILE + 2 * MARGIN) / C


def local(cx, cz, h_deg, x, z):
    """(s right, f forward) of (x, z) in axes centred (cx, cz) turned h (unit/zmap.local)."""
    h = math.radians(h_deg)
    dx, dz = x - cx, z - cz
    return dx * math.cos(h) - dz * math.sin(h), dx * math.sin(h) + dz * math.cos(h)


def world(cx, cz, h_deg, s, f):
    h = math.radians(h_deg)
    return cx + s * math.cos(h) + f * math.sin(h), cz - s * math.sin(h) + f * math.cos(h)


def targets_f(sf, tol=4.0):
    """[f] per unit: where its frame can stand forward. Rows (f within tol) whose frames would overlap a row in front
    (lateral overlap and a gap under a frame) are pushed back a frame behind it; then all shifted forward by half the
    largest push (two rows 40 m apart: +4 / -4 m)."""
    fr = _frame()
    n = len(sf)
    order = sorted(range(n), key=lambda i: -sf[i][1])
    t = [sf[i][1] for i in range(n)]
    for a, i in enumerate(order):
        for j in order[:a]:
            if abs(sf[i][0] - sf[j][0]) < fr - 1e-6 and t[j] - t[i] < fr and sf[j][1] - sf[i][1] > tol:
                t[i] = t[j] - fr
    push = max(sf[i][1] - t[i] for i in range(n))
    return [x + push / 2 for x in t]


def _box(sf):
    h = _frame() / 2
    tf = targets_f(sf)
    return tf, min(s for s, _ in sf) - h, max(s for s, _ in sf) + h, min(tf) - h, max(tf) + h


def _grids(n, s_lo, s_hi, f_lo, f_hi):
    return [(r * c, r, c) for r in range(1, ZN.MAX_TILES + 1) for c in range(1, ZN.MAX_TILES + 1)
            if n <= r * c <= ZN.MAX_TILES and c * TILE >= s_hi - s_lo - 1e-6 and r * TILE >= f_hi - f_lo - 1e-6]


def _align(c0, u0, cell, lo, hi, half):
    """The zone's centre along one axis, moved by less than a cell from c0 so that u0 - c lands on a cell centre; among
    those the one whose frames keep a cell from the zone's edge or sit flush on it."""
    best = None
    for k in range(-C, C + 1):
        c = u0 - (k + 0.5) * cell
        if abs(c - c0) >= cell or lo < c - half - 1e-6 or hi > c + half + 1e-6:
            continue
        gaps = (lo - (c - half), (c + half) - hi)
        clean = all(g < 1e-3 or g >= cell - 1e-3 for g in gaps)
        key = (not clean, abs(c - c0))
        if best is None or key < best[0]:
            best = (key, c)
    return c0 if best is None else best[1]


def _mean(places):
    n = len(places)
    return sum(p[0] for p in places) / n, sum(p[1] for p in places) / n


def fit(places, course):
    """The closed form (module doc); None: no grid of <= MAX_TILES tiles holds the frames."""
    mx, mz = _mean(places)
    sf = [local(mx, mz, course, x, z) for x, z in places]
    _, s_lo, s_hi, f_lo, f_hi = _box(sf)
    gs = _grids(len(places), s_lo, s_hi, f_lo, f_hi)
    if not gs:
        return None
    _, r, c = min(gs, key=lambda g: (g[0], max(g[1], g[2]), g[1]))
    cs = _align((s_lo + s_hi) / 2, sf[0][0], _cell(r, c), s_lo, s_hi, c * TILE / 2)
    cf = f_hi - r * TILE / 2
    cx, cz = world(mx, mz, course, cs, cf)
    return {"cx": cx, "cz": cz, "heading": course, "line": r == 1, "rows": r, "cols": c}


def candidates(places, course, grids=3):
    """[zones] the search tries (module doc): per grid the centre across - the box's middle, either edge flush, the first
    unit on a cell centre; forward - each row flush, the middle, a row's first unit on a cell centre; both axes swept in
    eighths of a cell, +- one cell."""
    mx, mz = _mean(places)
    sf = [local(mx, mz, course, x, z) for x, z in places]
    tf, s_lo, s_hi, f_lo, f_hi = _box(sf)
    gs = _grids(len(places), s_lo, s_hi, f_lo, f_hi)
    counts = sorted({g[0] for g in gs})[:grids]
    firsts = {}
    for i, (s, _) in enumerate(sf):
        firsts.setdefault(round(tf[i] / 4.0), (s, tf[i]))
    out = []
    for cnt, r, c in sorted(g for g in gs if g[0] in counts):
        cell = _cell(r, c)
        hs, hf = c * TILE / 2, r * TILE / 2
        c0s, c0f = (s_lo + s_hi) / 2, (f_lo + f_hi) / 2
        cs_l = {round(c0s, 3), round(s_lo + hs, 3), round(s_hi - hs, 3)}
        for k in range(-2 * C, 2 * C):
            v = sf[0][0] - (k + 0.5) * cell
            if abs(v - c0s) < cell:
                cs_l.add(round(v, 3))
        cf_l = {round(c0f, 3), round(f_hi - hf, 3), round(f_lo + hf, 3)}
        for _, fr in firsts.values():
            for k in range(-2 * C, 2 * C):
                v = fr - (k + 0.5) * cell
                if abs(v - c0f) < cell:
                    cf_l.add(round(v, 3))
        for k in range(-8, 9):
            cs_l.add(round(c0s + k * cell / 8, 3))
            cf_l.add(round(c0f + k * cell / 8, 3))
        for cs in sorted(cs_l):
            if s_lo < cs - hs - 1e-6 or s_hi > cs + hs + 1e-6:
                continue
            for cf in sorted(cf_l):
                if f_lo < cf - hf - 1e-6 or f_hi > cf + hf + 1e-6:
                    continue
                cx, cz = world(mx, mz, course, cs, cf)
                out.append({"cx": cx, "cz": cz, "heading": course, "line": r == 1, "rows": r, "cols": c, "tiles": cnt})
    return out


def first_moves(places, zones):
    """[R] the largest first move (m) of each zone's frames from the places (orders.frame_walk, every unit forced; inf: the
    zone gave the level's 'no room' error)."""
    from net.tactic import orders as OR
    from net.tactic import raw as TR
    from net.tactic.parcel import ParcelError
    R, G = len(zones), len(places)
    t = lambda k, dt=torch.float32: torch.tensor([z[k] for z in zones], dtype=dt)       # noqa: E731
    mr = torch.zeros((R, G, len(TR.RAW)))
    mr[..., TR.I["x"]] = torch.tensor([p[0] for p in places])[None]
    mr[..., TR.I["z"]] = torch.tensor([p[1] for p in places])[None]
    out = torch.full((R,), float("inf"))
    parts = [list(range(R))]
    while parts:
        ids = parts.pop()
        sub = torch.tensor(ids)
        tiles = ZN.Tiles(t("cx")[sub], t("cz")[sub], t("heading")[sub], t("rows", torch.long)[sub],
                         t("cols", torch.long)[sub])
        able = torch.ones((len(ids), G), dtype=torch.bool)
        try:
            w = OR.frame_walk(mr[sub], able, tiles, None, able.clone())
        except ParcelError:
            if len(ids) > 1:
                parts += [ids[:len(ids) // 2], ids[len(ids) // 2:]]
            continue
        fr = w["frames"]
        from net.unit import zmap as Z
        us, uf = Z.local(mr[sub][..., TR.I["x"]], mr[sub][..., TR.I["z"]], tiles.cx[:, None], tiles.cz[:, None],
                         tiles.heading[:, None])
        out[sub] = torch.sqrt((fr["s"] - us) ** 2 + (fr["f"] - uf) ** 2).amax(1)
    return out


def search(places, course):
    """(zone, the largest first move m) by the real walk over candidates() (module doc); (None, nan) when none."""
    cands = candidates(places, course)
    if not cands:
        return None, float("nan")
    d = first_moves(places, cands)
    best = None
    for i, z in enumerate(cands):
        v = float(d[i])
        ok = v <= reissue.MOVE_M
        key = (not ok, z["tiles"] if ok else 0, v)
        if best is None or key < best[0]:
            best = (key, i, v)
    if not math.isfinite(best[2]):
        return None, float("nan")
    z = dict(cands[best[1]])
    z.pop("tiles", None)
    return z, best[2]


_CACHE = {}


def zone_for(places, course):
    """{cx, cz, heading, line, rows, cols, worst} of a group deployed at places [(x, z)] (slot order) facing course
    (module doc); None when no zone of <= MAX_TILES tiles holds the frames."""
    mx, mz = _mean(places)
    sf = [local(mx, mz, course, x, z) for x, z in places]
    key = tuple((round(s, 1), round(f, 1)) for s, f in sf)
    if key not in _CACHE:
        pl = [(s, f) for s, f in sf]                         # (in the course's axes about the middle: course 0)
        z = fit(pl, 0.0)
        worst = float(first_moves(pl, [z])[0]) if z is not None else float("inf")
        if not worst <= reissue.MOVE_M:
            z, worst = search(pl, 0.0)
        _CACHE[key] = None if z is None else (z["cx"], z["cz"], z["line"], z["rows"], z["cols"], worst)
    hit = _CACHE[key]
    if hit is None:
        return None
    cs, cf, line, rows, cols, worst = hit
    cx, cz = world(mx, mz, course, cs, cf)
    return {"cx": cx, "cz": cz, "heading": course, "line": line, "rows": rows, "cols": cols, "worst": worst}
