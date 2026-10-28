"""Drill U6 'withdraw from melee' (build/diag_g49/unit_base.md, the drills' plan): our melee infantry unit is in melee
with one enemy (the U1 pairs, both ways) and is ordered to take a new area behind or aside, 40-150 m away (the stage
'going'), at a random pace (the order's rule). The simulator's melee exit does the rest (the 24 s window, the hold in
contact, the enemy's blows in the back, its pursuit under `nearest` - as in the game); the network decides how to go
(which cells, its facing; it cannot hold or attack: its own cell and the enemy are out of the new area).

A skirmish (specs): our unit at a random bearing and frontage; the enemy touching its front (or a flank), attacking it;
WARM_S seconds of fighting before the order (both scripted: ours holds, the enemy attacks); the area w x h cells of 8 m
(drill_hold.MIN_CELLS..10) on our grid, its nearest edge EDGE_M away, its direction from our front 'behind' (180 +-
45 deg) or 'aside' (+-90 +- 30 deg).
The end (Ender): our rout - over; arrived and the enemy out of our zone (or dead, gone) - over; the limit ARRIVED_S after
the arrival, never arrived: the way at the pace x 1.5 + 60 s after the order.
"""
import math

import torch

from tools.nn.sim import battle, geometry, scenario
from tools.nn.sim import orders as O
from tools.nn.train import opponents
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

WARM_S = 5.0
EDGE_M = (40.0, 150.0)
DIRS = {"behind": (180.0, 45.0), "aside": (90.0, 30.0)}
CONTACT = ("front", "flank")
PACES = ("fast", "slow")
ARRIVED_S = 120.0


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of (direction, pace) to cycle (the exam)."""
    R = DH.roster()
    pp = DH._passports()
    lo = DH.min_cells()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        me, foe = R[ours][ri(len(R[ours]))], R[theirs][ri(len(R[theirs]))]
        way, pace = fixed[k % len(fixed)] if fixed else (list(DIRS)[ri(2)], PACES[ri(2)])
        mid, spread = DIRS[way]
        side = 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1
        rel = (mid * (side if way == "aside" else 1) + u(-spread, spread)) % 360
        factor = u(*DH.WIDTH)
        front, depth = DH.box(pp[me], factor)
        speed = pp[me]["speed"]["run" if pace == "fast" else "walk"]
        out.append({"me": me, "foe": foe, "ours": ours, "theirs": theirs, "b": u(0.0, 360.0), "factor": factor,
                    "front": front, "depth": depth, "way": way, "pace": pace, "rel": rel, "edge": u(*EDGE_M),
                    "w": ri(DH.MAX_CELLS - lo + 1) + lo, "h": ri(DH.MAX_CELLS - lo + 1) + lo,
                    "contact": CONTACT[ri(2)], "speed": speed})
    return out


def exam_specs(seed=4911, per=60):
    """The fixed exam: per skirmishes for each direction x pace (2 x 2 x 60 = 240)."""
    gen = torch.Generator().manual_seed(seed)
    return specs(per * 4, gen, fixed=[(w, p) for w in DIRS for p in PACES])


def build(sp, P, device, step=None):
    """The state after WARM_S of melee (ours slot 0, the enemy slot H), own [B, N], the areas [B, N], the pace [B]."""
    from tools.nn.unit import net as Nt
    armies, areas = [], []
    C = DH.CELL
    for s in sp:
        foe_b = (s["b"] + 180.0) % 360 if s["contact"] == "front" else (s["b"] + 90.0) % 360   # facing us
        mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
        theirs = {"key": s["foe"], "x": 0.0, "z": 100.0, "b": foe_b}
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": [theirs]}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    H = N // 2
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    touch = min(0.0, P.sim["contact"]["reach_m"])
    for b_, s in enumerate(sp):
        # the enemy's front against our front (contact 'front') or our left flank ('flank')
        if s["contact"] == "front":
            d = float(depth[b_, 0] / 2 + depth[b_, H] / 2) + touch
            a = math.radians(s["b"])
        else:
            d = float(front[b_, 0] / 2 + depth[b_, H] / 2) + touch
            a = math.radians(s["b"] - 90.0)
        st.u["x"][b_, H], st.u["z"][b_, H] = d * math.sin(a), d * math.cos(a)
    # WARM_S of fighting: ours holds, the enemy attacks it
    step = step or battle.stepper(device)
    for _ in range(int(round(WARM_S / P.dt))):
        o = opponents.nearest(st)
        o.kind[:, 0], o.target[:, 0] = O.HOLD, -1
        step(st, o, P)
    for b_, s in enumerate(sp):
        # the area on our grid at the order: its nearest edge s["edge"] away along s["rel"] from our front
        a = math.radians(s["rel"])
        half = abs(math.sin(a)) * s["w"] * C / 2 + abs(math.cos(a)) * s["h"] * C / 2
        sx, sf = (s["edge"] + half) * math.sin(a), (s["edge"] + half) * math.cos(a)
        sx = round((sx - (s["w"] % 2) * C / 2) / C) * C + (s["w"] % 2) * C / 2
        sf = round((sf - (s["h"] % 2) * C / 2) / C) * C + (s["h"] % 2) * C / 2
        b0 = float(st.u["b"][b_, 0])
        x, z = Z.world(torch.tensor(sx), torch.tensor(sf), float(st.u["x"][b_, 0]), float(st.u["z"][b_, 0]), b0)
        areas.append((float(x), float(z), b0, s["w"] * C, s["h"] * C))
        dx, dz = abs(sx) - s["w"] * C / 2, abs(sf) - s["h"] * C / 2
        s["edge_real"] = math.hypot(max(dx, 0.0), max(dz, 0.0))
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda i: torch.tensor([a_[i] for a_ in areas], device=device, dtype=torch.float32)[:, None].expand(B, N).clone()  # noqa: E731
    area = {"x": col(0), "z": col(1), "b": col(2), "front": col(3), "depth": col(4), "on": own.clone()}
    pace = torch.tensor([Nt.PACE_FAST if s["pace"] == "fast" else Nt.PACE_SLOW for s in sp], device=device)
    return st, own, area, pace


class Ender:
    """U6's end rule and stage: arrived [B] (our centre once in the new area), arrive_t [B], t0 [B] (the order); over:
    our rout or death; arrived and the enemy out of our zone (or dead, gone); ARRIVED_S after the arrival; never
    arrived - the way at the pace x 1.5 + 60 s after the order."""

    def __init__(self, sp, st):
        B, dev = st.B, st.device
        self.over = st.done.clone()                     # (a battle over in the warm-up: our rout there)
        self.why = torch.where(st.done, 1, 0)
        self.arrived = torch.zeros(B, dtype=torch.bool, device=dev)
        self.arrive_t = torch.full((B,), -1.0, device=dev)
        self.t0 = st.t.clone()
        self.never = self.t0 + torch.tensor([s["edge_real"] / s["speed"] * 1.5 + 60.0 for s in sp], device=dev)

    def after_step(self, st, spacing, area):
        u = st.u
        H = st.N // 2
        a0 = {k: v[:, 0] for k, v in area.items() if k != "on"}
        live = ~self.over
        new_in = live & (Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], a0) >= 0) & ~self.arrived
        self.arrive_t = torch.where(new_in, st.t, self.arrive_t)
        self.arrived = self.arrived | new_in
        we_out = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        inz, _ = V.zone(now, idx, torch.zeros_like(idx))
        foe_gone = (u["men"][:, H] <= 0) | u["gone"][:, H]
        clear = self.arrived & (~inz[:, H] | foe_gone)
        late = torch.where(self.arrived, st.t >= self.arrive_t + ARRIVED_S, st.t >= self.never)
        new = live & (we_out | clear | foe_gone | late)
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(clear | foe_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(clear | foe_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
