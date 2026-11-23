"""S0-1 'the line on the crest': the candidate finder and the teacher's rule (tmp/drill_s1/design_merged.md sections 1-3).
Torch-free (numpy): one file for the simulator, the generator and the bridge. Numbers: net/config/strategist.json
"crest" (each with its _why); the line's width from net/config/stage0.json general (the stage's rows).

    ground(name)                 a captured map's grids (sim/maps/<name>: heights.npz height / ground, passable.npz)
    ground_of(height, x0, z0, ground=None, blocked=None, forest=(), water=())   any grid (a synthetic ridge)
    front_m(n), two_rows(n)      the line's width W for n units, a second row behind
    measure(G, x, z, b, W)       one line's measures (band, profile, legality shares)
    find(G, zone, n)             {"centre": token, "cands": [token] <= top, left to right, "sites": [...], "lines", "legal",
                                 "foot": the legal line of the lowest band - the exam's DUMB arm, or None}
    teach(found, attack)         the teacher: 0 = KEEP, k >= 1 = "on the crest" on cands[k - 1]
    places(token, n)             [(x, z, b)] n units on the line (the stage's rows: row_m apart, rows_of a row)
    layout_ok(G, pts)            every unit's footprint (centre, front ends, footprint_back_m behind) off blocked ground
    lord_place(pts, b, zone)     (x, z, b) the lord: lord_back_m behind the line's middle, kept lord_edge_m inside the
                                 zone's rear - ONE rule for the simulator (exam, input) and the bridge (LinePlace)
    slot_bands(G, token, n)      the band within slot_m of each front-row unit's centre (slot order)
    at(G, zone, n, geoms)        the finder's tokens at given (s, f, turn) - a twin keeping its cause scene's geometry

A zone is {x, z, b (or bearing), front, depth} (net/stage0/specs.zone). A line: centre (s, f) in the zone's axes (s
right, f forward - towards the enemy), turn db to the zone's bearing, width W. The finder decides nothing: no worth
threshold (METHOD: code only turns a pointer into places) - that is teach()'s.
"""
import json
import math
from functools import lru_cache

import numpy as np

from net import CONFIG, ROOT


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


@lru_cache(maxsize=1)
def cfg():
    return _strip(json.loads((CONFIG / "strategist.json").read_text(encoding="utf-8")))["crest"]


@lru_cache(maxsize=1)
def _rows():
    g = _strip(json.loads((CONFIG / "stage0.json").read_text(encoding="utf-8")))["general"]
    return float(g["enemy_row_m"]), int(g["enemy_rows_of"]), float(g["enemy_rank_m"])


# ---------------------------------------------------------------- the ground

def ground_of(height, x0, z0, ground=None, blocked=None, forest=(), water=(), step=1.0):
    """The finder's grids: height[iz, ix] at x = x0 + ix x step, z = z0 + iz x step; ground codes with the forest /
    water codes; blocked 1 = cannot stand (None: nowhere)."""
    h = np.asarray(height, dtype=np.float64)
    return {"h": h, "x0": float(x0), "z0": float(z0), "step": float(step),
            "g": None if ground is None else np.asarray(ground),
            "blk": np.zeros(h.shape, np.uint8) if blocked is None else np.asarray(blocked),
            "forest": tuple(forest), "water": tuple(water)}


@lru_cache(maxsize=8)
def ground(name):
    """A captured map's grids (sim/maps/<name>/heights.npz, passable.npz, meta.json ground_names)."""
    d = ROOT / "sim" / "maps" / name
    H = np.load(d / "heights.npz")
    P = np.load(d / "passable.npz")
    names = json.loads((d / "meta.json").read_text(encoding="utf-8"))["ground_names"]
    code = {n: i for i, n in enumerate(names)}
    return ground_of(H["height"], float(H["x0"]), float(H["z0"]), H["ground"], P["blocked"],
                     [code[n] for n in ("forest",) if n in code],
                     [code[n] for n in ("shallow_water", "deep_water") if n in code], float(H["step"]))


def _height(G, x, z):
    h = G["h"]
    st = G.get("step", 1.0)
    fx = np.clip((np.asarray(x) - G["x0"]) / st, 0, h.shape[1] - 1.001)
    fz = np.clip((np.asarray(z) - G["z0"]) / st, 0, h.shape[0] - 1.001)
    ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
    ax, az = fx - ix, fz - iz
    return (h[iz, ix] * (1 - ax) * (1 - az) + h[iz, ix + 1] * ax * (1 - az) + h[iz + 1, ix] * (1 - ax) * az
            + h[iz + 1, ix + 1] * ax * az)


def height(G, x, z):
    """Bilinear height at world (x, z)."""
    return _height(G, x, z)


def _node(G, a, x, z):
    grid = G[a]
    st = G.get("step", 1.0)
    ix = np.clip(np.floor((np.asarray(x) - G["x0"]) / st).astype(int), 0, grid.shape[1] - 1)
    iz = np.clip(np.floor((np.asarray(z) - G["z0"]) / st).astype(int), 0, grid.shape[0] - 1)
    return grid[iz, ix]


# ---------------------------------------------------------------- the line

def front_m(n):
    row_m, per, _ = _rows()
    return row_m * (min(int(n), per) - 1) + float(cfg()["unit_front_m"])


def two_rows(n):
    return int(n) > _rows()[1]


def _bearing(zone):
    return float(zone["b"] if "b" in zone else zone["bearing"])


def world(zone, s, f):
    """A point of the zone's axes (s right, f forward) in the world (net/stage0/specs.world)."""
    r = math.radians(_bearing(zone))
    return zone["x"] + s * math.cos(r) + f * math.sin(r), zone["z"] - s * math.sin(r) + f * math.cos(r)


def measure_many(G, xs, zs, bs, W, n=None):
    """The measures of L lines centred at (xs, zs) [L] facing bs [L] (degrees), W m across (points every point_m):
    {name: [L]} and "profile" [L, D]. n: the units of the line (the unit-slot column; None: front_m's k from W)."""
    c = cfg()
    d_ahead = np.asarray(c["ahead_m"], float)
    di = {int(d): i for i, d in enumerate(c["ahead_m"])}
    xs, zs, r = np.asarray(xs, float), np.asarray(zs, float), np.radians(np.asarray(bs, float))
    fx, fz, sx, sz = np.sin(r), np.cos(r), np.cos(r), -np.sin(r)
    s = np.arange(-W / 2, W / 2 + 0.1, float(c["point_m"]))
    X = xs[:, None, None] + sx[:, None, None] * s[None, :, None] + fx[:, None, None] * d_ahead[None, None, :]
    Z = zs[:, None, None] + sz[:, None, None] * s[None, :, None] + fz[:, None, None] * d_ahead[None, None, :]
    H = _height(G, X, Z)                                              # [L, points, d]
    near, far = (int(v) for v in c["band_m"])
    band = (H[..., di[near]] - H[..., di[far]]) / float(far - near)    # [L, points]
    h0 = H[..., di[0]]
    L = c["legal"]
    a, e = di[int(L["steep_d"][0])], di[int(L["steep_d"][1])]
    steps = (H[..., a:e] - H[..., a + 1:e + 1]) / np.diff(d_ahead[a:e + 1])
    jb = [di[int(d)] for d in L["blocked_d"]]
    blk = _node(G, "blk", X[..., jb], Z[..., jb]) > 0
    if G["g"] is not None:
        g = _node(G, "g", X[..., jb], Z[..., jb])
        forest = np.isin(g, G["forest"]) if G["forest"] else np.zeros_like(blk)
        water = np.isin(g, G["water"]) if G["water"] else np.zeros_like(blk)
    else:
        forest = water = np.zeros_like(blk)
    jf = [jb.index(di[int(d)]) for d in L["forest_d"]]
    lo, hi = (int(v) for v in c["rise_m"])
    # the unit-slot column: the band within slot_m of each front-row unit's centre (the stage's rows), the least of them
    row_m, per, _ = _rows()
    k = min(int(n), per) if n is not None else int(round((W - float(c["unit_front_m"])) / row_m)) + 1
    su = (np.arange(k) - (k - 1) / 2) * row_m
    near_u = np.abs(s[None, :] - su[:, None]) <= float(c["slot_m"])          # [k, points]
    slot = (band[:, None, :] * near_u[None]).sum(-1) / near_u.sum(-1).clip(min=1)[None]
    m = {"band": band.mean(1), "band_p25": np.percentile(band, 25, axis=1),
         "share05": (band >= float(c["share_band"])).mean(1),
         "band_slot_min": slot.min(1),
         "rise60": (h0 - H[..., di[lo]]).mean(1), "rise100": (h0 - H[..., di[hi]]).mean(1),
         "back20": (H[..., di[int(c["back_m"])]] - h0).mean(1),
         "steep": steps.reshape(len(xs), -1).max(1), "blocked": blk.reshape(len(xs), -1).mean(1),
         "forest": forest[..., jf].reshape(len(xs), -1).mean(1), "water": water.reshape(len(xs), -1).mean(1),
         "h0": h0.mean(1), "profile": H.mean(1) - h0.mean(1)[:, None]}
    m["slots"] = slot
    m["legal"] = ((m["blocked"] <= L["blocked"]) & (m["forest"] <= L["forest"]) & (m["water"] <= L["water"])
                  & (m["steep"] <= L["steep"]))
    return m


COLS = ("band", "band_p25", "share05", "band_slot_min", "rise60", "rise100", "back20", "steep", "blocked", "forest",
        "water", "h0")


def _rowdict(m, i):
    out = {k: float(m[k][i]) for k in COLS}
    out["profile"] = [float(v) for v in m["profile"][i]]
    out["slots"] = [float(v) for v in m["slots"][i]]
    out["legal"] = bool(m["legal"][i])
    return out


def measure(G, x, z, b, W, n=None):
    """The measures of the line centred at (x, z) facing b, W m across (measure_many of one)."""
    return _rowdict(measure_many(G, [x], [z], [b], W, n), 0)


def _token(G, zone, s, f, db, W, n):
    x, z = world(zone, s, f)
    b = (_bearing(zone) + db) % 360.0
    m = measure(G, x, z, b, W, n)
    m.update(s=float(s), f=float(f), turn=float(db), x=float(x), z=float(z), b=float(b), front=float(W), n=int(n))
    m["layout"] = layout_ok(G, places(m, n))
    m["valid"] = bool(m["legal"] and m["layout"])
    return m


def lines(G, zone, n):
    """Every candidate line of n units in the zone: centres on the grid_m lattice of the zone's axes about its centre,
    each turn of turns_deg; the ends (and a second row's rear) edge_m inside the zone."""
    c = cfg()
    W = front_m(n)
    rear = _rows()[2] if two_rows(n) else 0.0
    step, edge, reach = float(c["grid_m"]), float(c["edge_m"]), float(c["turn_reach_m"])
    front, depth = float(zone["front"]), float(zone["depth"])
    cells, cen = [], []
    for db in c["turns_deg"]:
        sa, ca = abs(math.sin(math.radians(db))), math.cos(math.radians(db))
        smax = front / 2 - edge - (W / 2 * ca + reach * sa)
        flo = -depth / 2 + edge + rear + W / 2 * sa
        fhi = depth / 2 - edge - W / 2 * sa
        if smax < 0 or fhi < flo:
            continue
        for i in range(math.ceil(-smax / step - 1e-9), math.floor(smax / step + 1e-9) + 1):
            for j in range(math.ceil(flo / step - 1e-9), math.floor(fhi / step + 1e-9) + 1):
                x, z = world(zone, i * step, j * step)
                cells.append((i, j, float(db)))
                cen.append((x, z, (_bearing(zone) + db) % 360.0))
    if not cen:
        return []
    a = np.asarray(cen)
    m = measure_many(G, a[:, 0], a[:, 1], a[:, 2], W, n)
    lay = layout_many(G, a[:, 0], a[:, 1], a[:, 2], n)
    out = []
    for k, (i, j, db) in enumerate(cells):
        t = _rowdict(m, k)
        t.update(s=i * step, f=j * step, turn=db, x=float(a[k, 0]), z=float(a[k, 1]), b=float(a[k, 2]), front=float(W),
                 n=int(n), cell=(i, j))
        t["layout"] = bool(lay[k])
        t["legal"] = t["valid"] = bool(t["legal"] and t["layout"])
        out.append(t)
    return out


def components(cells):
    """[[index]] of the 8-connected components of cells [(i, j)] (several lines may share a cell: other turns)."""
    by = {}
    for k, c in enumerate(cells):
        by.setdefault(c, []).append(k)
    seen, out = set(), []
    for c0 in by:
        if c0 in seen:
            continue
        seen.add(c0)
        stack, comp = [c0], []
        while stack:
            a = stack.pop()
            comp += by[a]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (a[0] + di, a[1] + dj)
                    if nb in by and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(comp)
    return out


def find(G, zone, n):
    """{"centre": the KEEP line's token (s = f = 0, turn 0), "cands": the best line of each of the top sites by band,
    left to right (s ascending), "sites": every site's best line (band descending) with its "lines" count, "lines",
    "legal": counts}. A site: a connected component of legal lines with band >= site_band. No site: cands empty (the
    skill masked)."""
    c = cfg()
    every = lines(G, zone, n)
    ok = [t for t in every if t["legal"] and t["band"] >= float(c["site_band"])]
    sites = []
    for comp in components([t["cell"] for t in ok]):
        best = max((ok[k] for k in comp), key=lambda t: (t["band"], -abs(t["turn"]), -abs(t["s"]), -abs(t["f"])))
        sites.append({**best, "lines": len(comp)})
    sites.sort(key=lambda t: -t["band"])
    cands = sorted(sites[:int(c["top"])], key=lambda t: t["s"])
    centre = _token(G, zone, 0.0, 0.0, 0.0, front_m(n), n)
    legal = [t for t in every if t["legal"]]
    foot = min(legal, key=lambda t: (t["band"], abs(t["turn"]), abs(t["s"]), abs(t["f"]))) if legal else None
    return {"centre": centre, "cands": cands, "sites": sites, "lines": len(every), "legal": len(legal), "foot": foot}


def at(G, zone, n, geoms):
    """The finder's output with its candidates at the given [(s, f, turn)] (left to right as given), measured on G: a
    twin keeping its cause scene's candidate geometry (offsets, count, order) and differing by the ground only
    (Astra s1_step2_review 3: presence and order of the candidates must not tell the cause). The centre and the foot
    as find()'s."""
    W = front_m(n)
    fd = find(G, zone, n)
    cands = [_token(G, zone, float(s), float(f), float(db), W, n) for s, f, db in geoms]
    return {**fd, "cands": cands, "sites": [], "pinned": True}


# ---------------------------------------------------------------- the teacher (one rule beside the finder)

def fires(token, centre):
    """The teacher's test of one candidate (config crest teacher; ASSUMPTIONS judged by the exam's bins)."""
    t = cfg()["teacher"]
    return (bool(token["valid"]) and token["band"] >= float(t["band"])
            and token["band"] - centre["band"] >= float(t["delta"]) and token["share05"] >= float(t["share"]))


def teach(found, attack):
    """0 = KEEP (the centre line), k >= 1 = "on the crest" on found["cands"][k - 1] (design_merged section 3): mission
    attack -> KEEP; no valid candidate -> KEEP; else the candidate of the largest band if it passes fires(), else KEEP."""
    if attack:
        return 0
    valid = [(k, t) for k, t in enumerate(found["cands"]) if t["valid"]]
    if not valid:
        return 0
    k, best = max(valid, key=lambda kt: kt[1]["band"])
    return k + 1 if fires(best, found["centre"]) else 0


# ---------------------------------------------------------------- the line -> the units' places

def places(token, n):
    """[(x, z, b)] of n units on the line (net/stage0/specs.rows' form: rows of rows_of, row_m apart, the rows
    rank_m apart backwards from the line), facing the line's bearing."""
    row_m, per, rank_m = _rows()
    zn = {"x": token["x"], "z": token["z"], "b": token["b"]}
    out = []
    for i in range(int(n)):
        r, c = divmod(i, per)
        k = min(per, int(n) - r * per)
        x, z = world(zn, (c - (k - 1) / 2) * row_m, -r * rank_m)
        out.append((x, z, float(token["b"])))
    return out


def _foot_offsets(n):
    """[P, 2] (s, f) of every footprint point of n units on a line (places' rows), relative to the line's centre."""
    c = cfg()
    row_m, per, rank_m = _rows()
    hw, back = float(c["unit_front_m"]) / 2, float(c["footprint_back_m"])
    out = []
    for i in range(int(n)):
        r, cc = divmod(i, per)
        k = min(per, int(n) - r * per)
        su, fu = (cc - (k - 1) / 2) * row_m, -r * rank_m
        out += [(su + s, fu + f) for s in (-hw, 0.0, hw) for f in (0.0, -back)]
    return np.asarray(out, float)


def layout_many(G, xs, zs, bs, n):
    """[L] bool: the layout of n units on each line (centre xs, zs facing bs) stands (layout_ok of each)."""
    o = _foot_offsets(n)
    r = np.radians(np.asarray(bs, float))[:, None]
    X = np.asarray(xs, float)[:, None] + o[None, :, 0] * np.cos(r) + o[None, :, 1] * np.sin(r)
    Z = np.asarray(zs, float)[:, None] - o[None, :, 0] * np.sin(r) + o[None, :, 1] * np.cos(r)
    return ~(_node(G, "blk", X, Z) > 0).any(1)


def layout_ok(G, pts):
    """Every unit's footprint off blocked ground (module doc; config crest footprint_back_m)."""
    c = cfg()
    hw, back = float(c["unit_front_m"]) / 2, float(c["footprint_back_m"])
    xs, zs = [], []
    for x, z, b in pts:
        r = math.radians(b)
        for s in (-hw, 0.0, hw):
            for f in (0.0, -back):
                xs.append(x + s * math.cos(r) + f * math.sin(r))
                zs.append(z - s * math.sin(r) + f * math.cos(r))
    if not xs:
        return True
    return not bool((_node(G, "blk", np.asarray(xs), np.asarray(zs)) > 0).any())


def lord_place(pts, b, zone, back=None):
    """(x, z, b) of the lord (module doc): back (None: net/config/stage0.json lord_back_m) behind the middle of pts
    [(x, z, ...)] along b, shortened so that he stays lord_edge_m inside zone's rear (the zone's own facing)."""
    from net.stage0 import cfg as scfg
    back = float(scfg()["lord_back_m"]) if back is None else float(back)
    mx = sum(p[0] for p in pts) / max(len(pts), 1)
    mz = sum(p[1] for p in pts) / max(len(pts), 1)
    a = math.radians(float(b))
    za = math.radians(_bearing(zone))
    f_of = lambda x, z: (x - zone["x"]) * math.sin(za) + (z - zone["z"]) * math.cos(za)       # noqa: E731
    f_min = -float(zone["depth"]) / 2 + float(cfg()["lord_edge_m"])
    cos_t = max(math.cos(a - za), 0.3)
    over = f_min - f_of(mx - back * math.sin(a), mz - back * math.cos(a))
    if over > 0:
        back = max(back - over / cos_t, 0.0)
    return mx - back * math.sin(a), mz - back * math.cos(a), float(b)


def slot_bands(G, token, n):
    """[k] the band within slot_m of each front-row unit's centre, slot order (measure_many's slot column)."""
    m = measure_many(G, [token["x"]], [token["z"]], [token["b"]], front_m(n), n)
    return [float(v) for v in m["slots"][0]]
