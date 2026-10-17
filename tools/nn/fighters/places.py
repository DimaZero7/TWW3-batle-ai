"""The step replay from the game's own soldiers (docs/en/training/fighters_lab.md, "Replay from the game's places").

A battle recorded with the soldiers' places (soldiersdata.py) gives every unit's men where they stood at each window
start. There each living man of the lab is put on a recorded place (nearest pairs first - a greedy match of least way, so the wounds
the lab carries stay near where they were) and that place, in the unit's frame, is his place for the window
(F.fixed, F.fa, F.ff; installed() wraps state.slots for it, the lab's step itself unchanged) - the formation keeps
the game's shape and moves with its centre. A unit with no
record keeps the lab's formation. Needs torch."""
from pathlib import Path

import numpy as np
import torch

from tools.nn.fighters import state as S


_SLOTS = S.slots


def _slots(F, u):
    sa, sf = _SLOTS(F, u)
    fixed = getattr(F, "fixed", None)
    if fixed is None:
        return sa, sf
    lone = (u["men0"] <= 1).gather(1, F.uid)
    return torch.where(fixed & ~lone, F.fa, sa), torch.where(fixed & ~lone, F.ff, sf)


class installed:
    """with installed(): state.slots gives the recorded places of the men put by put() (move.py reads S.slots)."""

    def __enter__(self):
        S.slots = _slots
        return self

    def __exit__(self, *exc):
        S.slots = _SLOTS


def match(cost):
    """(rows, cols): pairs taken nearest first, each row and column once (greedy; min(rows, cols) pairs)."""
    order = np.argsort(cost, axis=None, kind="stable")
    r_used = np.zeros(cost.shape[0], bool)
    c_used = np.zeros(cost.shape[1], bool)
    rows, cols = [], []
    need = min(cost.shape)
    for k in order:
        i, j = divmod(int(k), cost.shape[1])
        if r_used[i] or c_used[j]:
            continue
        r_used[i] = c_used[j] = True
        rows.append(i)
        cols.append(j)
        if len(rows) == need:
            break
    return np.array(rows, int), np.array(cols, int)


def put(F, u, b, n, pts):
    """The living men of unit n of battle b onto the recorded places pts [k, 2] (world, m): positions and places for
    the window. Returns the men placed."""
    live = (S.alive(F)[b] & (F.uid[b] == n)).nonzero().squeeze(1)
    if not len(live) or not len(pts):
        return 0
    cx, cz = float(u["x"][b, n]), float(u["z"][b, n])
    px = (u["x"][b, n] + F.dx[b, live]).cpu().numpy()
    pz = (u["z"][b, n] + F.dz[b, live]).cpu().numpy()
    cost = (px[:, None] - pts[None, :, 0]) ** 2 + (pz[:, None] - pts[None, :, 1]) ** 2
    ri, ci = match(cost)
    men = live[torch.as_tensor(ri, device=live.device)]
    P = torch.as_tensor(pts[ci], dtype=F.dx.dtype, device=F.dx.device)
    ox, oz = P[:, 0] - cx, P[:, 1] - cz
    br = float(u["b"][b, n]) * np.pi / 180
    s, c = float(np.sin(br)), float(np.cos(br))
    F.dx[b, men], F.dz[b, men] = ox, oz
    F.fa[b, men] = ox * c - oz * s                  # across (cos b, -sin b)
    F.ff[b, men] = ox * s + oz * c                  # along the facing (sin b, cos b)
    F.fixed[b, men] = True
    F.tgt[b, men] = -1
    return len(ri)


class FromRecording:
    """The places provider of a step replay batch: battle row b -> its run (b // copies), the run's soldiers, its
    names' slots. Call at a window start (hook.Hook.on_reset)."""

    def __init__(self, run_dirs, copies):
        from tools.nn.fighters import soldiersdata
        from tools.nn.sim import check, scenario
        self.copies = copies
        hs = max(check.units_a_side(d) for d in run_dirs)
        self.runs = []
        for d in run_dirs:
            g = check.load(d)
            slot = scenario.slots(scenario.from_recording(g, d), hs)
            self.runs.append({"name": Path(d).name, "t": g.t, "slot": {nm: slot[nm] for nm in g.names},
                              "places": soldiersdata.load(d)})
        self.stats = {"windows": 0, "units": 0, "men": 0, "missing": 0}

    def __call__(self, F, u, row):
        """At the window start of recording row `row` (the step replay's st.t)."""
        B = F.B
        if getattr(F, "fixed", None) is None:
            F.fixed = torch.zeros_like(F.valid)
            F.fa = torch.zeros_like(F.dx)
            F.ff = torch.zeros_like(F.dx)
        F.fixed[:] = False
        for b in range(B):
            r = self.runs[b // self.copies]
            if row >= len(r["t"]):
                continue
            rec = r["places"].at(float(r["t"][row]))
            if rec is None:
                self.stats["missing"] += 1
                continue
            self.stats["windows"] += 1
            for nm, pts in rec.items():
                n = r["slot"].get(nm)
                if n is None or float(u["men0"][b, n]) <= 1:
                    continue
                k = put(F, u, b, n, pts)
                self.stats["units"] += 1
                self.stats["men"] += k
