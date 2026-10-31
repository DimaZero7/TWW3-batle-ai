"""The unit drills on real maps (train.py / exam_*.py --map <name | a,b | all>): every skirmish of a batch laid on one
of the maps (build/maps/<name>/: heights, ground types, trunks, 'cannot stand' - tools/nn/sim/terrain.py, ground.py) at
a random place and turn. The skirmish keeps its own coordinates - the drill, its areas, the scripts unchanged (the
relative geometry as on the flat field); only the map's layers are read through the battle's frame (terrain.set_map
place, terrain.to_map).

The maps: --map takes one name, a comma list or 'all' (every folder of build/maps with heights.npz); each must have
REQUIRED (a map without its 'cannot stand' layer would let the drills stand where the game refuses). A skirmish's map:
with one map that one (no draw - the places as with a single map before); with several, in training a random one
(evenly), in the exams in turn (skirmish b on map b mod M: the same share of each, the same set every build).

A place: the skirmish's extent - its units, its areas (with their half diagonals) and goals, about the middle of their
box, + MARGIN_M for the approach - turned by a random angle and centred at a random point of the map that keeps it
within the map's field (terrain.FRAMES, less EDGE_M); a place is taken when every unit's centre and the corners of its
box, and the centres of the areas and goals, lie on free nodes (passable.npz: not 'cannot stand' - the game refuses to
deploy or move there; a map without the layer: heights.npz clear), else drawn again (at most TRIES; then the try with
the fewest closed points - recorded in map_place 'free'). A skirmish wider than the map stands at its middle.
The exam's places: a fixed seed for every build (the same set for every network measured).

Metrics (Meter, for the exams; rollout's stats "terrain"): the mean height of our network units' ground over the
nearest enemy's while in melee (m: + we stand higher, better), and the share of our men on forest ground while the
enemy's missiles hit us (more in the woods, better: the trunks stop the arrows, the woods hide).
"""
import math
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn.sim import geometry, ground, terrain

MAPS = project.ROOT / "build" / "maps"
REQUIRED = ("heights.npz", "meta.json", "passable.npz")     # a map's files for the drills (map_grid.py convert, layers)
MARGIN_M = 60.0          # beyond the skirmish's extent: the approach, the men's spread
EDGE_M = 10.0            # from the map's frame
TRIES = 200            # (beyond: the try with the fewest closed points; the exams found a free one for 360 of 360)
EXAM_SEED = 7310

_ACTIVE = {}             # maps [entries], exam; the first map's entry's keys too (one map: as before)
_LOADED = {}             # folder -> entry (a map registered once a process)


def check(folder):
    """Raise when a map folder lacks one of REQUIRED (says which and how to make it)."""
    folder = Path(folder)
    missing = [f for f in REQUIRED if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"map {folder.name} ({folder}): no {', '.join(missing)} - capture / convert it "
                                "(tools/nn/map_grid.py capture, convert, layers; build/diag_map/scan.md)")


def names(spec, root=MAPS):
    """The maps of --map: a name, 'a,b' or 'all' (every folder of root with heights.npz), or a list; each checked."""
    root = Path(root)
    if isinstance(spec, (list, tuple)):
        out = [str(n) for n in spec]
    elif str(spec).strip().lower() == "all":
        out = sorted(f.name for f in root.iterdir() if (f / "heights.npz").exists()) if root.exists() else []
    else:
        out = [n.strip() for n in str(spec).split(",") if n.strip()]
    assert out, f"no maps in {spec!r} (root {root})"
    for n in out:
        check(root / n)
    return out


def _entry(folder):
    """The map's registry id and its host grids (heights, clear, blocked, its field), once a process."""
    key = str(Path(folder).resolve())
    e = _LOADED.get(key)
    if e is not None and e["mid"] < len(terrain._MAPS) and (e["mid"] in ground._LAYERS or e["blocked"] is None):
        return e
    mid = terrain.load_npz(Path(folder) / "heights.npz")
    d = np.load(Path(folder) / "heights.npz")
    H = d["height"]
    clear = d["clear"] if "clear" in d.files else np.ones(H.shape, np.uint8)
    pf = Path(folder) / "passable.npz"
    blocked = np.load(pf)["blocked"] if pf.exists() else None
    # the map's field (terrain.FRAMES: meta.json frame - the playable area; without one the grid's extent; Kislev: both
    # -512..512)
    lo_x, hi_x, lo_z, hi_z = terrain.FRAMES[mid]
    e = dict(name=Path(folder).name, mid=mid, clear=clear, blocked=blocked, height=H, x0=float(d["x0"]),
             z0=float(d["z0"]), step=float(d["step"]), lo_x=lo_x, hi_x=hi_x, lo_z=lo_z, hi_z=hi_z)
    _LOADED[key] = e
    return e


def activate(name, exam=False, root=MAPS):
    """Lay every drill's skirmishes on the maps `name` (names: one, 'a,b', 'all') from now on (this process); exam: the
    fixed places. Returns the first map's id."""
    ms = [_entry(Path(root) / n) for n in names(name, root)]
    _ACTIVE.clear()
    _ACTIVE.update(ms[0])
    _ACTIVE.update(maps=ms, exam=exam, name=",".join(m["name"] for m in ms))
    return ms[0]["mid"]


def deactivate():
    _ACTIVE.clear()


def active():
    return bool(_ACTIVE)


def maps():
    """The active maps' entries (empty when none is active)."""
    return list(_ACTIVE.get("maps", []))


def pick(B, gen, exam=None):
    """[B] numpy: each skirmish's map (an index into maps()) - one map: 0 (no draw); several: in the exams in turn, else
    a random one evenly (gen)."""
    M = len(maps())
    exam = _ACTIVE.get("exam") if exam is None else exam
    if M <= 1:
        return np.zeros(B, np.int64)
    if exam or gen is None:
        return np.arange(B, dtype=np.int64) % M
    return torch.randint(M, (B,), generator=gen).numpy().astype(np.int64)


def height_at(x, z, m=None):
    """The map's (m: an entry; None: the first active one) ground (its nearest node) at world points x, z (numpy)."""
    a = m or _ACTIVE
    i = np.clip(np.round((np.asarray(x) - a["x0"]) / a["step"]).astype(int), 0, a["height"].shape[1] - 1)
    j = np.clip(np.round((np.asarray(z) - a["z0"]) / a["step"]).astype(int), 0, a["height"].shape[0] - 1)
    return a["height"][j, i]


def to_world(x, z, px, pz, deg):
    """A battle's local points onto the map (terrain.set_map place's rule), numpy."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, z = np.asarray(x, float), np.asarray(z, float)
    return x * c + z * s + px, -x * s + z * c + pz


def free_at(x, z, m=None):
    """[...] bool: world points x, z (numpy) of the map (m; None: the first active one) on free nodes - not 'cannot
    stand' (passable.npz; the node at floor, as ground.blocked); a map without the layer: heights.npz clear."""
    a = m or _ACTIVE
    has = a.get("blocked") is not None
    grid = a["blocked"] if has else a["clear"]
    i = np.clip(np.floor((np.asarray(x) - a["x0"]) / a["step"]).astype(int), 0, grid.shape[1] - 1)
    j = np.clip(np.floor((np.asarray(z) - a["z0"]) / a["step"]).astype(int), 0, grid.shape[0] - 1)
    v = grid[j, i] > 0
    return ~v if has else v


def extents(setup):
    """Per battle: (cx, cz, R, points that must be free [list of (x, z)]) - the middle of the skirmish's box and the
    radius about it that holds its units, its areas (+ their half diagonals) and its goals (+ theirs), + MARGIN_M; the
    points: every unit's centre and its box's corners, the areas' and goals' centres."""
    from tools.nn.sim.params import load
    st = setup["st"]
    u = st.u
    B = st.B
    pts = [[] for _ in range(B)]
    rad = [[] for _ in range(B)]
    start = [[] for _ in range(B)]
    pres = (u["side"] > 0).cpu().numpy()
    ux, uz = u["x"].cpu().numpy(), u["z"].cpu().numpy()
    front, depth = (v.cpu().numpy() for v in geometry.dims(u, load().sim["formation"]["spacing_m"]))
    bb = np.radians(u["b"].cpu().numpy())
    for b in range(B):
        for i in np.nonzero(pres[b])[0]:
            pts[b].append((ux[b, i], uz[b, i]))
            rad[b].append(0.0)
            start[b].append((ux[b, i], uz[b, i]))
            fx, fz, sx, sz = math.sin(bb[b, i]), math.cos(bb[b, i]), math.cos(bb[b, i]), -math.sin(bb[b, i])
            for ka in (-0.5, 0.5):
                for kd in (-0.5, 0.5):
                    start[b].append((ux[b, i] + sx * ka * front[b, i] + fx * kd * depth[b, i],
                                     uz[b, i] + sz * ka * front[b, i] + fz * kd * depth[b, i]))
    for key in ("area", "goal"):
        a = setup.get(key)
        if not isinstance(a, dict) or "x" not in a:
            continue
        on = a.get("on")
        ax, az = a["x"].cpu().numpy(), a["z"].cpu().numpy()
        half = (np.sqrt(a["front"].cpu().numpy() ** 2 + a["depth"].cpu().numpy() ** 2) / 2
                if "front" in a and "depth" in a else np.zeros_like(ax))
        if ax.ndim == 1:
            ax, az, half = ax[:, None], az[:, None], np.broadcast_to(np.asarray(half).reshape(B, -1), (B, 1))
        onn = on.cpu().numpy() if on is not None else np.ones_like(ax, bool)
        for b in range(B):
            for i in np.nonzero(onn[b])[0]:
                pts[b].append((ax[b, i], az[b, i]))
                rad[b].append(float(half[b, i]))
                start[b].append((ax[b, i], az[b, i]))
    out = []
    for b in range(B):
        p = np.array(pts[b]) if pts[b] else np.zeros((1, 2))
        r = np.array(rad[b]) if rad[b] else np.zeros(1)
        lo, hi = (p - r[:, None]).min(0), (p + r[:, None]).max(0)
        c = (lo + hi) / 2
        R = float((np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) + r).max()) + MARGIN_M
        out.append((float(c[0]), float(c[1]), R, start[b]))
    return out


def places(setup, gen, which=None):
    """(px, pz, deg, free) numpy [B]: the skirmishes' frames on their maps (which [B]: indices into maps(); None: the
    first) and whether every point that must be free is (module doc)."""
    ms = maps() or [_ACTIVE]
    ex = extents(setup)
    B = len(ex)
    which = np.zeros(B, np.int64) if which is None else which
    px, pz, deg, free = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B, bool)
    for b, (cx, cz, R, start) in enumerate(ex):
        a = ms[int(which[b])]
        mid_x, mid_z = (a["lo_x"] + a["hi_x"]) / 2, (a["lo_z"] + a["hi_z"]) / 2
        best = None
        for _ in range(TRIES):
            ang = float(torch.rand((), generator=gen)) * 360.0
            span_x = max(0.0, (a["hi_x"] - a["lo_x"]) / 2 - R - EDGE_M)
            span_z = max(0.0, (a["hi_z"] - a["lo_z"]) / 2 - R - EDGE_M)
            wx = mid_x + (float(torch.rand((), generator=gen)) * 2 - 1) * span_x
            wz = mid_z + (float(torch.rand((), generator=gen)) * 2 - 1) * span_z
            t = math.radians(ang)
            c, s = math.cos(t), math.sin(t)
            # the box's middle (cx, cz) onto (wx, wz): offset = w - rot(c)
            ox, oz = wx - (cx * c + cz * s), wz - (-cx * s + cz * c)
            sx = np.array([p[0] * c + p[1] * s + ox for p in start])
            sz = np.array([-p[0] * s + p[1] * c + oz for p in start])
            closed = int((~free_at(sx, sz, a)).sum()) if len(sx) else 0
            if best is None or closed < best[3]:
                best = (ox, oz, ang, closed)
            if closed == 0:
                break
        px[b], pz[b], deg[b], free[b] = best[0], best[1], best[2], best[3] == 0
    return px, pz, deg, free


def lay(st, which, px, pz, deg):
    """terrain.set_map for each active map on its skirmishes (which [B]) at their places (numpy [B])."""
    t = lambda v: torch.tensor(np.asarray(v, np.float64), dtype=torch.float32, device=st.device)       # noqa: E731
    for k, m in enumerate(maps() or [_ACTIVE]):
        sel = np.asarray(which) == k
        if sel.any():
            terrain.set_map(st, m["mid"], battles=torch.tensor(sel, device=st.device), place=(t(px), t(pz), t(deg)))


def apply(setup, gen=None):
    """Lay the setup's skirmishes on the active maps (in place; nothing when none is active). Returns the setup."""
    if not _ACTIVE or setup.get("map_place") is not None:      # (a drill that chose its own places: drill_height)
        return setup
    if _ACTIVE["exam"] or gen is None:
        gen = torch.Generator().manual_seed(EXAM_SEED)
    st = setup["st"]
    which = pick(st.B, gen)
    px, pz, deg, free = places(setup, gen, which)
    lay(st, which, px, pz, deg)
    setup["map_place"] = {"px": px, "pz": pz, "deg": deg, "map": which, "free": free}
    return setup


class Meter:
    """The exams' terrain readings over a rollout (module doc): per step, for our network units (bi, ii)."""

    def __init__(self, B, device):
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.dh_sum, self.melee_s, self.forest_sum, self.fire_s = z(), z(), z(), z()

    def step(self, st, bi, ii, P, was_done):
        u = st.u
        dt = float(P.dt)
        live = (~was_done[bi]) & (u["men"][bi, ii] > 0) & ~u["gone"][bi, ii]
        h = terrain.ground(u)                                         # [B, N] under each centre
        enemy = (u["side"][bi][:, :] > 0) & (u["side"][bi] != u["side"][bi, ii][:, None]) & (u["men"][bi] > 0) \
            & ~u["gone"][bi]
        d2 = (u["x"][bi] - u["x"][bi, ii][:, None]) ** 2 + (u["z"][bi] - u["z"][bi, ii][:, None]) ** 2
        d2 = torch.where(enemy, d2, torch.full_like(d2, 1e18))
        j = d2.argmin(1)
        melee = live & u["m"][bi, ii] & enemy.any(1)
        dh = h[bi, ii] - h[bi, j]
        self.dh_sum.index_add_(0, bi, torch.where(melee, dh, torch.zeros_like(dh)) * dt)
        self.melee_s.index_add_(0, bi, melee.float() * dt)
        shot = live & (u["under_fire_s"][bi, ii] < 0.5 * dt)
        if bool(shot.any()) and ground.on(u):
            front, depth = geometry.dims(u, P.sim["formation"]["spacing_m"])
            fs = ground.shares(u, front, depth, P)[..., 1][bi, ii]
        else:
            fs = torch.zeros_like(dh)
        self.forest_sum.index_add_(0, bi, torch.where(shot, fs, torch.zeros_like(fs)) * dt)
        self.fire_s.index_add_(0, bi, shot.float() * dt)

    def result(self):
        ms, fs = float(self.melee_s.sum()), float(self.fire_s.sum())
        return {"maps": [m["name"] for m in maps()],
                "dh_melee_m": round(float(self.dh_sum.sum()) / ms, 3) if ms else None, "melee_s": round(ms, 1),
                "forest_under_fire": round(float(self.forest_sum.sum()) / fs, 3) if fs else None,
                "under_fire_s": round(fs, 1)}
