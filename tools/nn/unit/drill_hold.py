"""Drill U1 'hold the line' (build/diag_g49/unit_base.md, Drills): one melee infantry unit of the network against one
of the other faction under the script `nearest`, the order 'hold' in an area cut on the unit's own grid.

Roster: the melee infantry (caste melee_infantry) of the Empire and the Skaven from the passports - no shooters, no
lords, no war beasts; every Empire x Skaven pair, the network on either side.

A skirmish:
* the network's unit stands at a random bearing with a random frontage (WIDTH x its default, men0 / rank_depth files);
* its area: w x h cells of 8 m (MIN_CELLS..10 each) on the unit's grid at the order (its frame: the centre on a cells'
  corner, the axes its front), the unit's centre at a corner far enough from every edge for its formation to turn in
  place (half its diagonal) where the area allows; MIN_CELLS: the widest formation of the roster at the widest frontage
  turns in place (its diagonal, cells_needed());
* the enemy: on any side (a bearing from the unit's front 0..360, the exam: face / flank / rear), its centre GAP_M m
  from ours, facing us; under `nearest` it attacks at a run. 60 m at least: a formation's about-face on a facing order
  takes ~6.2 s (3.9 s re-form + 180 deg at 80 deg/s, config/nn/sim.json turn), the fastest enemy here runs 4.2 m/s
  (+ the charge's last 30 m at its charge speed): ~13 s to cover 60 m - time to turn;
* the end (Ender): our unit routs (or dies) - over (the rest is the tactical level's); the enemy routs - the skirmish
  goes on until it leaves our unit's zone (view.zone) or the map; at most LIMIT_S.
"""
import json
import math
from pathlib import Path

import torch

from tools.nn.sim import scenario
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

ROOT = Path(__file__).resolve().parents[3]
CELL = 8.0
MAX_CELLS = 10
WIDTH = (0.8, 1.25)          # the frontage's factor (a random formation)
GAP_M = (75.0, 160.0)        # the centres' distance (the edges ~60 m apart at least)
LIMIT_S = 180.0
STARTS = {"face": 0.0, "flank": 90.0, "rear": 180.0}     # the enemy's bearing from our front (the exam)
FACTIONS = {"emp": "wh_main_emp_empire", "skv": "wh2_main_skv_skaven"}


def _passports():
    return json.loads((ROOT / "config/nn/units.json").read_text(encoding="utf-8"))["units"]


# Kept out of the unit drills (31.10.2026): the Pusbags joined the pool (config/nn/units.json) with their own disease in
# melee and vortices at death; the drills' fixed exam (6 x 10 x 2 x 3 = 360 battles) stays as it was until it is decided
# whether the unit level should learn them; the Warp Grinders the same (31.10.2026: two active abilities on every enemy
# within 35 m - Warp-Quake's direct damage and Seismic Snare's no-move)
NOT_IN_DRILLS = ("wh3_dlc29_skv_inf_pusbags", "wh2_dlc14_skv_inf_warp_grinder_0")


def roster():
    """{"emp": [keys], "skv": [keys]}: the melee infantry of each faction (but NOT_IN_DRILLS)."""
    out = {"emp": [], "skv": []}
    for k, v in _passports().items():
        if v.get("caste") != "melee_infantry" or v.get("missile") or v["men"] <= 1 or k in NOT_IN_DRILLS:
            continue
        side = "skv" if "_skv_" in k else "emp" if "_emp_" in k else None
        if side:
            out[side].append(k)
    return out


def box(p, factor=1.0):
    """(front, depth) m of a passport's formation at factor x its default frontage (geometry.dims' rule)."""
    h, v = p["spacing"]["h"], p["spacing"]["v"]
    files0 = p["men"] // p["rank_depth"]
    files = max(1, math.floor(files0 * h * factor / h + 1e-4))
    return files * h, math.ceil(p["men"] / files) * v


def cells_needed():
    """(cells, metres, key): the area's least side - the widest formation's diagonal at the widest frontage."""
    pp = _passports()
    best = max(((math.hypot(*box(pp[k], WIDTH[1])), k) for ks in roster().values() for k in ks))
    return math.ceil(best[0] / CELL), round(best[0], 1), best[1]


MIN_CELLS = None


def min_cells():
    global MIN_CELLS
    if MIN_CELLS is None:
        MIN_CELLS = cells_needed()[0]
    return MIN_CELLS


def specs(n, gen, starts=None):
    """n random skirmishes (dicts): pair, our side, bearing, frontage, area cells, corner, the enemy's angle, gap."""
    R = roster()
    pp = _passports()
    out = []
    lo = min_cells()
    for k in range(n):
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        me = R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))]
        foe = R[theirs][int(torch.randint(len(R[theirs]), (1,), generator=gen))]
        out.append(_spec(me, foe, ours, gen, lo, pp, None if starts is None else starts[k % len(starts)]))
    return out


def _spec(me, foe, ours, gen, lo, pp, start):
    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
    factor = u(*WIDTH)
    w = int(torch.randint(lo, MAX_CELLS + 1, (1,), generator=gen))
    h = int(torch.randint(lo, MAX_CELLS + 1, (1,), generator=gen))
    front, depth = box(pp[me], factor)
    r = math.hypot(front, depth) / 2

    def corner(n):
        k_lo, k_hi = math.ceil(r / CELL), n - math.ceil(r / CELL)
        if k_lo > k_hi:
            k_lo = k_hi = n // 2
        return int(torch.randint(k_lo, k_hi + 1, (1,), generator=gen))
    if start is None:
        rel = u(0.0, 360.0)
    else:
        rel = (STARTS[start] * (1 if float(torch.rand(1, generator=gen)) < 0.5 else -1) + u(-10.0, 10.0)) % 360
    return {"me": me, "foe": foe, "ours": ours, "b": u(0.0, 360.0), "factor": factor, "front": front, "depth": depth,
            "w": w, "h": h, "kx": corner(w), "kz": corner(h), "rel": rel, "gap": u(*GAP_M), "start": start}


def exam_specs(seed=4901):
    """The fixed exam: every pair both ways x face / flank / rear, one random rest each (the same set every time)."""
    gen = torch.Generator().manual_seed(seed)
    R = roster()
    pp = _passports()
    lo = min_cells()
    out = []
    for ours, theirs in (("emp", "skv"), ("skv", "emp")):
        for me in R[ours]:
            for foe in R[theirs]:
                for start in STARTS:
                    out.append(_spec(me, foe, ours, gen, lo, pp, start))
    return out


def build(sp, P, device):
    """The state, own [B, N] (the network's slot), its area {x, z, b, front, depth, on} [B, N] and the specs."""
    armies = []
    for s in sp:
        a = math.radians(s["b"] + s["rel"])
        ex, ez = s["gap"] * math.sin(a), s["gap"] * math.cos(a)
        face = (math.degrees(math.atan2(-ex, -ez))) % 360
        mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
        theirs = {"key": s["foe"], "x": ex, "z": ez, "b": face}
        f_me, f_foe = FACTIONS[s["ours"]], FACTIONS["skv" if s["ours"] == "emp" else "emp"]
        armies.append({"attacker": 2, "sides": {1: {"faction": f_me, "units": [mine]},
                                                2: {"faction": f_foe, "units": [theirs]}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    # the area on the unit's grid: cells kx.. to its left / kz.. behind its centre (a cells' corner)
    cx, cz, ab, aw, ad = [], [], [], [], []
    for s in sp:
        sc = (s["w"] / 2 - s["kx"]) * CELL            # the area's centre right of / ahead of ours, in our frame
        fc = (s["h"] / 2 - s["kz"]) * CELL
        x, z = Z.world(torch.tensor(sc), torch.tensor(fc), 0.0, 0.0, s["b"])
        cx.append(float(x)), cz.append(float(z)), ab.append(s["b"]), aw.append(s["w"] * CELL), ad.append(s["h"] * CELL)
    f = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {"x": f(cx), "z": f(cz), "b": f(ab), "front": f(aw), "depth": f(ad), "on": own.clone()}
    return st, own, area


class Ender:
    """The drill's end rule, after every simulator step: st.done / st.winner set as the drill wants (the simulator's
    own end - a side without a standing unit - is undone while the enemy routs within our zone)."""

    def __init__(self, B, device, limit_s=LIMIT_S):
        self.over = torch.zeros(B, dtype=torch.bool, device=device)
        self.why = torch.zeros(B, dtype=torch.long, device=device)      # 1 we routed, 2 the enemy left, 3 the limit
        self.limit = limit_s
        self.arrived = torch.ones(B, dtype=torch.bool, device=device)      # U1: in place from the start

    def after_step(self, st, spacing, area=None):
        u = st.u
        H = st.N // 2
        we_out = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        inz, _ = V.zone(now, idx, torch.zeros_like(idx))
        foe_gone = (u["men"][:, H] <= 0) | u["gone"][:, H] | (u["r"][:, H] & ~inz[:, H])
        late = st.t >= self.limit
        new = ~self.over & (we_out | foe_gone | late)
        why = torch.where(we_out, 1, torch.where(foe_gone, 2, 3))
        self.why = torch.where(new, why, self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foe_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
