"""Drill U4 'friends beside' (build/diag_g49/unit_base.md, the drills' plan): two of our melee infantry units, both
commanded by the network (one set of weights, two copies, each seeing its own zone - the other one among its 'own' rows
and on its map's 'own' layers), each with its own area as a tactical level would cut them, the order 'hold'; one or two
enemies of the other faction under `nearest`.

A skirmish (specs):
* our two units of one faction (any two of its roster, the same kind twice too), the same bearing (the line's front),
  each of a random frontage as U1; their areas on the line's grid (8 m cells, axes along the front): each w x h cells
  (drill_hold.MIN_CELLS..MAX_PAIR_CELLS), the layout
      side     side by side, a gap of GAPS cells between the areas
      echelon  side by side, the second shifted SHIFT cells forward or back
  each unit at a cell corner of its area, centred across it (where its formation turns in place, as U1);
* the enemies (ENEMIES 1 or 2, of the other faction): from the front, a flank, the front and a flank, or both ahead at an
  angle (bearings from the line's front, around the line's centre), GAP_M m off, staggered by 0-15 s;
* the end (Ender): both our units routed or dead - over; every enemy dead, gone or routed out of both our zones - over;
  at most LIMIT_S.
The reward: the side's (the trade, the end) shared by both units: one network for every unit and no extra term for 'do
not get in the way' (the project's rule) - jostling or uncovering the neighbour costs both through the trade.
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

MAX_PAIR_CELLS = 8
GAPS = (0, 1, 2)                 # cells between the two areas (side by side)
SHIFT = (1, 3)                   # cells forward / back (echelon)
LAYOUTS = ("side", "echelon")
ENEMY_SETS = {"front": (0.0,), "flank": (90.0,), "front_flank": (0.0, 90.0), "front_angle": (-35.0, 35.0)}
GAP_M = (90.0, 170.0)
LAG_S = (0.0, 15.0)
JITTER = 15.0
LIMIT_S = DH.LIMIT_S


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of (layout, enemy set) to cycle (the exam)."""
    R = DH.roster()
    pp = DH._passports()
    lo = DH.min_cells()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        layout, es = fixed[k % len(fixed)] if fixed else (LAYOUTS[ri(2)], list(ENEMY_SETS)[ri(len(ENEMY_SETS))])
        mine = []
        for _ in range(2):
            key = R[ours][ri(len(R[ours]))]
            f = u(*DH.WIDTH)
            front, depth = DH.box(pp[key], f)
            mine.append({"key": key, "factor": f, "front": front, "depth": depth,
                         "w": ri(MAX_PAIR_CELLS - lo + 1) + lo, "h": ri(MAX_PAIR_CELLS - lo + 1) + lo})
        gap = GAPS[ri(len(GAPS))] if layout == "side" else 0
        shift = (ri(SHIFT[1] - SHIFT[0] + 1) + SHIFT[0]) * (1 if float(torch.rand(1, generator=gen)) < 0.5 else -1) \
            if layout == "echelon" else 0
        side = 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1
        foes = []
        lag0 = 0.0
        for rel in ENEMY_SETS[es]:
            key = R[theirs][ri(len(R[theirs]))]
            foes.append({"key": key, "rel": (rel * (side if es in ("flank", "front_flank") and rel else 1)
                                             + u(-JITTER, JITTER)) % 360,
                         "gap": u(*GAP_M) + lag0 * pp[key]["speed"]["run"]})
            lag0 = u(*LAG_S)
        out.append({"ours": ours, "theirs": theirs, "b": u(0.0, 360.0), "units": mine, "layout": layout, "enemies": es,
                    "gap_cells": gap, "shift": shift, "foes": foes})
    return out


def exam_specs(seed=4907, per=30):
    """The fixed exam: per skirmishes for each layout x enemy set (2 x 4 x 30 = 240)."""
    gen = torch.Generator().manual_seed(seed)
    cells = [(l_, e_) for l_ in LAYOUTS for e_ in ENEMY_SETS]
    return specs(per * len(cells), gen, fixed=cells)


def _layout(s):
    """[(area centre right, forward, w, h, unit right, forward)] of our two, in the line's frame (m), on its grid."""
    C = DH.CELL
    a, b = s["units"]
    # the first area's left edge at 0, the second right of it after the gap; then everything centred on the pair
    x0 = 0.0
    x1 = (a["w"] + s["gap_cells"]) * C
    f0, f1 = 0.0, s["shift"] * C
    out = []
    for un, xl, fb in ((a, x0, f0), (b, x1, f1)):
        w, h = un["w"] * C, un["h"] * C
        # the unit at the cell corner nearest the area's middle (its formation turns there where the area allows)
        ux = xl + round(un["w"] / 2) * C
        uz = fb + round(un["h"] / 2) * C - h / 2
        out.append([xl + w / 2, fb, w, h, ux, uz])
    mid = (out[0][0] + out[1][0]) / 2
    for o in out:
        o[0] -= mid
        o[4] -= mid
    return out


def build(sp, P, device):
    """The state (ours: slots 0 and 1; the enemies from slot H), own [B, N] (slots 0 and 1) and the areas [B, N]."""
    armies, areas = [], []
    for s in sp:
        lay = _layout(s)
        mine = []
        for un, (ax, af, w, h, ux, uf) in zip(s["units"], lay):
            x, z = Z.world(torch.tensor(ux), torch.tensor(uf), 0.0, 0.0, s["b"])
            mine.append({"key": un["key"], "x": float(x), "z": float(z), "b": s["b"], "width": un["front"]})
        foes = []
        for f in s["foes"]:
            a = math.radians(s["b"] + f["rel"])
            ex, ez = f["gap"] * math.sin(a), f["gap"] * math.cos(a)
            foes.append({"key": f["key"], "x": ex, "z": ez, "b": math.degrees(math.atan2(-ex, -ez)) % 360})
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": mine},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": foes}}})
        areas.append(lay)
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, :2] = True
    area = {k: torch.zeros((B, N), device=device) for k in ("x", "z", "b", "front", "depth")}
    for b_, (s, lay) in enumerate(zip(sp, areas)):
        for i, (ax, af, w, h, _, _) in enumerate(lay):
            x, z = Z.world(torch.tensor(ax), torch.tensor(af), 0.0, 0.0, s["b"])
            area["x"][b_, i], area["z"][b_, i], area["b"][b_, i] = float(x), float(z), s["b"]
            area["front"][b_, i], area["depth"][b_, i] = w, h
    area["on"] = own.clone()
    return st, own, area


class Ender:
    """U4's end rule after every step: both our units out (routed, dead, gone) - over; every enemy dead, gone or routed
    out of both our units' zones - over; at most LIMIT_S."""

    def __init__(self, B, device, limit_s=LIMIT_S):
        self.over = torch.zeros(B, dtype=torch.bool, device=device)
        self.why = torch.zeros(B, dtype=torch.long, device=device)
        self.arrived = torch.ones(B, dtype=torch.bool, device=device)
        self.limit = limit_s

    def after_step(self, st, spacing, area=None):
        u = st.u
        ours = (u["side"] == 1) & (u["men0"] > 0)
        foe = (u["side"] == 2) & (u["men0"] > 0)
        out_u = (u["men"] <= 0) | u["gone"] | u["r"]
        we_out = (~ours | out_u).all(1)
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        in0, _ = V.zone(now, idx, torch.zeros_like(idx))
        in1, _ = V.zone(now, idx, torch.ones_like(idx))
        gone = (u["men"] <= 0) | u["gone"] | (u["r"] & ~(in0 | in1))
        foes_gone = (~foe | gone).all(1)
        late = st.t >= self.limit
        new = ~self.over & (we_out | foes_gone | late)
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(foes_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foes_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
