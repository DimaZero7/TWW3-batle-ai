"""The unit drills on a real map (train.py / exam_*.py --map <name>): every skirmish of a batch laid at a random place of
the map (build/maps/<name>/: heights, ground types, trunks - tools/nn/sim/terrain.py, ground.py), at a random turn. The
skirmish keeps its own coordinates - the drill, its areas, the scripts unchanged (the relative geometry as on the flat
field); only the map's layers are read through the battle's frame (terrain.set_map place, terrain.to_map).

A place: the skirmish's extent - its units, its areas (with their half diagonals) and goals, about the middle of their
box, + MARGIN_M for the approach - turned by a random angle and centred at a random point of the map that keeps it
within the map's frame (less EDGE_M); a place whose units start on ground the game calls not clear (heights.npz clear 0:
water, cliffs) is drawn again (at most TRIES; then the last one). A skirmish wider than the map stands at its middle.
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
MARGIN_M = 60.0          # beyond the skirmish's extent: the approach, the men's spread
EDGE_M = 10.0            # from the map's frame
TRIES = 20
EXAM_SEED = 7310

_ACTIVE = {}             # name, map id, clear grid, frame, exam


def activate(name, exam=False, root=MAPS):
    """Lay every drill's skirmishes on the map `name` from now on (this process); exam: the fixed places."""
    folder = Path(root) / name
    mid = terrain.load_npz(folder / "heights.npz")
    d = np.load(folder / "heights.npz")
    H = d["height"]
    x0, z0, step = float(d["x0"]), float(d["z0"]), float(d["step"])
    clear = d["clear"] if "clear" in d.files else np.ones(H.shape, np.uint8)
    _ACTIVE.clear()
    _ACTIVE.update(name=name, mid=mid, clear=clear, height=H, x0=x0, z0=z0, step=step, exam=exam,
                   lo_x=x0, hi_x=x0 + (H.shape[1] - 1) * step, lo_z=z0, hi_z=z0 + (H.shape[0] - 1) * step)
    return mid


def deactivate():
    _ACTIVE.clear()


def active():
    return bool(_ACTIVE)


def height_at(x, z):
    """The active map's ground (its nearest node) at world points x, z (numpy)."""
    a = _ACTIVE
    i = np.clip(np.round((np.asarray(x) - a["x0"]) / a["step"]).astype(int), 0, a["height"].shape[1] - 1)
    j = np.clip(np.round((np.asarray(z) - a["z0"]) / a["step"]).astype(int), 0, a["height"].shape[0] - 1)
    return a["height"][j, i]


def to_world(x, z, px, pz, deg):
    """A battle's local points onto the map (terrain.set_map place's rule), numpy."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, z = np.asarray(x, float), np.asarray(z, float)
    return x * c + z * s + px, -x * s + z * c + pz


def _clear_at(x, z):
    a = _ACTIVE
    i = np.clip(np.floor((np.asarray(x) - a["x0"]) / a["step"]).astype(int), 0, a["clear"].shape[1] - 1)
    j = np.clip(np.floor((np.asarray(z) - a["z0"]) / a["step"]).astype(int), 0, a["clear"].shape[0] - 1)
    return a["clear"][j, i] > 0


def extents(setup):
    """Per battle: (cx, cz, R, unit points [list of (x, z)]) - the middle of the skirmish's box and the radius about it
    that holds its units, its areas (+ their half diagonals) and its goals (+ theirs), + MARGIN_M."""
    st = setup["st"]
    u = st.u
    B = st.B
    pts = [[] for _ in range(B)]
    rad = [[] for _ in range(B)]
    start = [[] for _ in range(B)]
    pres = (u["side"] > 0).cpu().numpy()
    ux, uz = u["x"].cpu().numpy(), u["z"].cpu().numpy()
    for b in range(B):
        for i in np.nonzero(pres[b])[0]:
            pts[b].append((ux[b, i], uz[b, i]))
            rad[b].append(0.0)
            start[b].append((ux[b, i], uz[b, i]))
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
    out = []
    for b in range(B):
        p = np.array(pts[b]) if pts[b] else np.zeros((1, 2))
        r = np.array(rad[b]) if rad[b] else np.zeros(1)
        lo, hi = (p - r[:, None]).min(0), (p + r[:, None]).max(0)
        c = (lo + hi) / 2
        R = float((np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) + r).max()) + MARGIN_M
        out.append((float(c[0]), float(c[1]), R, start[b]))
    return out


def places(setup, gen):
    """(px, pz, deg) numpy [B]: the skirmishes' frames on the active map (module doc)."""
    a = _ACTIVE
    ex = extents(setup)
    B = len(ex)
    px, pz, deg = np.zeros(B), np.zeros(B), np.zeros(B)
    mid_x, mid_z = (a["lo_x"] + a["hi_x"]) / 2, (a["lo_z"] + a["hi_z"]) / 2
    for b, (cx, cz, R, start) in enumerate(ex):
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
            px[b], pz[b], deg[b] = ox, oz, ang
            if len(sx) == 0 or bool(_clear_at(sx, sz).all()):
                break
    return px, pz, deg


def apply(setup, gen=None):
    """Lay the setup's skirmishes on the active map (in place; nothing when none is active). Returns the setup."""
    if not _ACTIVE or setup.get("map_place") is not None:      # (a drill that chose its own places: drill_height)
        return setup
    if _ACTIVE["exam"] or gen is None:
        gen = torch.Generator().manual_seed(EXAM_SEED)
    px, pz, deg = places(setup, gen)
    st = setup["st"]
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=st.device)       # noqa: E731
    terrain.set_map(st, _ACTIVE["mid"], place=(t(px), t(pz), t(deg)))
    setup["map_place"] = {"px": px, "pz": pz, "deg": deg}
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
        return {"dh_melee_m": round(float(self.dh_sum.sum()) / ms, 3) if ms else None, "melee_s": round(ms, 1),
                "forest_under_fire": round(float(self.forest_sum.sum()) / fs, 3) if fs else None,
                "under_fire_s": round(fs, 1)}
