"""Drill U2 'reach and hold' (build/diag_g49/unit_base.md, Drills): the network's unit starts OUT of its area, the order
'hold' at the stage 'going'; once its centre is in the area the stage is 'in place' (then as U1). The same 16 melee
infantry, the 60 Empire x Skaven pairs, the network on either side.

A skirmish (specs):
* the unit at a random bearing and frontage (as U1); its area w x h cells of 8 m (drill_hold.MIN_CELLS..10) on its grid
  at the order (axes along its front), the area's nearest edge NEAR_M or FAR_M m away in a random direction (often far
  beyond the map's 240 m: the order carries the area's raw numbers, net.order_of);
* the pace of the order: 'fast' (run only) or 'slow' (walk only) at random - the order's rule, not the unit's choice;
* the enemy goes TO THE AREA (to_area below), from a random side, its start set by the race wanted: 'early' (we reach
  the area EARLY_S s before it), 'close' (within CLOSE_S s of it), 'held' (it starts in the area and holds it);
* the end (Ender): our rout - over; the enemy's rout - on until it leaves our zone; ARRIVED_S after our arrival; never
  arrived: the walk there at the pace x 1.5 + 60 s.
The enemy's script to_area: in melee, or with its centre within GRAB_M of the area, it attacks the nearest of us (if
within GRAB_M of the area or in melee with it), else it holds; otherwise it runs to the area's centre.
"""
import math

import torch

from tools.nn.sim import orders as O
from tools.nn.train import opponents
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

NEAR_M = (10.0, 120.0)        # the area's edge from our centre: near (on the map) ...
FAR_M = (120.0, 700.0)        # ... or far (mostly beyond it)
FAR_SHARE = 0.6
ARRIVED_S = 180.0
EARLY_S = (15.0, 45.0)
CLOSE_S = 8.0
GRAB_M = 20.0
RACES = ("early", "close", "held")
PACES = ("fast", "slow")


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of (race, pace) to cycle (the exam)."""
    R = DH.roster()
    pp = DH._passports()
    lo = DH.min_cells()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        me = R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))]
        foe = R[theirs][int(torch.randint(len(R[theirs]), (1,), generator=gen))]
        race, pace = fixed[k % len(fixed)] if fixed else (RACES[int(torch.randint(3, (1,), generator=gen))],
                                                           PACES[int(torch.randint(2, (1,), generator=gen))])
        out.append(_spec(me, foe, ours, race, pace, u, gen, lo, pp))
    return out


def _spec(me, foe, ours, race, pace, u, gen, lo, pp):
    factor = u(*DH.WIDTH)
    front, depth = DH.box(pp[me], factor)
    w = int(torch.randint(lo, DH.MAX_CELLS + 1, (1,), generator=gen))
    h = int(torch.randint(lo, DH.MAX_CELLS + 1, (1,), generator=gen))
    far = float(torch.rand(1, generator=gen)) < FAR_SHARE
    edge = u(*(FAR_M if far else NEAR_M))
    way = u(0.0, 360.0)                          # the area's direction from our front
    speed = pp[me]["speed"]["run" if pace == "fast" else "walk"]
    e_run = pp[foe]["speed"]["run"]
    # the area's centre: its edge `edge` m away along `way` (the half extent along that way added), snapped to our grid
    a = math.radians(way)
    half = abs(math.sin(a)) * w * DH.CELL / 2 + abs(math.cos(a)) * h * DH.CELL / 2
    dist = edge + half
    s, f = dist * math.sin(a), dist * math.cos(a)
    s = round((s - (w % 2) * DH.CELL / 2) / DH.CELL) * DH.CELL + (w % 2) * DH.CELL / 2
    f = round((f - (h % 2) * DH.CELL / 2) / DH.CELL) * DH.CELL + (h % 2) * DH.CELL / 2
    # the real distance to the area's nearest point after the snap (a corner may lie farther than the way's edge)
    dx, dz = abs(s) - w * DH.CELL / 2, abs(f) - h * DH.CELL / 2
    edge = math.hypot(max(dx, 0.0), max(dz, 0.0))
    if edge < 1.0:                               # (snapped onto the area: one cell farther along the way)
        s, f = s + DH.CELL * math.copysign(1, s) * (abs(s) >= abs(f)), f + DH.CELL * math.copysign(1, f) * (abs(f) > abs(s))
        dx, dz = abs(s) - w * DH.CELL / 2, abs(f) - h * DH.CELL / 2
        edge = math.hypot(max(dx, 0.0), max(dz, 0.0))
    t_us = edge / speed
    if race == "early":
        t_e = t_us + u(*EARLY_S)
    elif race == "close":
        t_e = max(t_us + u(-CLOSE_S, CLOSE_S), 2.0)
    else:
        t_e = 0.0
    side = u(0.0, 360.0)                         # the enemy's side of the area
    return {"me": me, "foe": foe, "ours": ours, "b": u(0.0, 360.0), "factor": factor, "front": front, "depth": depth,
            "w": w, "h": h, "s": s, "f": f, "edge": edge, "race": race, "pace": pace, "t_us": t_us,
            "e_dist": t_e * e_run, "e_side": side}


def exam_specs(seed=4903):
    """The fixed exam: 360 skirmishes - every pair once (a random side for the network) x race x pace."""
    gen = torch.Generator().manual_seed(seed)
    R = DH.roster()
    pp = DH._passports()
    lo = DH.min_cells()
    out = []
    for e in R["emp"]:
        for k in R["skv"]:
            ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
            me, foe = (e, k) if ours == "emp" else (k, e)
            for race in RACES:
                for pace in PACES:
                    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
                    out.append(_spec(me, foe, ours, race, pace, u, gen, lo, pp))
    return out


def build(sp, P, device):
    """The state, own [B, N], the areas {x, z, b, front, depth, on} [B, N], the pace [B] (net.PACE_*)."""
    from tools.nn.sim import scenario
    from tools.nn.unit import net as Nt
    armies, areas = [], []
    for s in sp:
        ax, az = Z.world(torch.tensor(s["s"]), torch.tensor(s["f"]), 0.0, 0.0, s["b"])
        ax, az = float(ax), float(az)
        hw, hd = s["w"] * DH.CELL / 2, s["h"] * DH.CELL / 2
        if s["race"] == "held":
            ex, ez = ax, az                                       # in the area's middle
        else:
            g = math.radians(s["e_side"])
            reach = abs(math.sin(g)) * hw + abs(math.cos(g)) * hd   # (its edge on that side, roughly)
            ex, ez = ax + (reach + s["e_dist"]) * math.sin(g + math.radians(s["b"])), \
                az + (reach + s["e_dist"]) * math.cos(g + math.radians(s["b"]))
        face = math.degrees(math.atan2(ax - ex, az - ez)) % 360 if s["race"] != "held" else \
            math.degrees(math.atan2(-ex, -ez)) % 360
        mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
        theirs = {"key": s["foe"], "x": ex, "z": ez, "b": face}
        f_me, f_foe = DH.FACTIONS[s["ours"]], DH.FACTIONS["skv" if s["ours"] == "emp" else "emp"]
        armies.append({"attacker": 2, "sides": {1: {"faction": f_me, "units": [mine]},
                                                2: {"faction": f_foe, "units": [theirs]}}})
        areas.append((ax, az, s["b"], 2 * hw, 2 * hd))
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda i: torch.tensor([a[i] for a in areas], device=device, dtype=torch.float32)[:, None].expand(B, N).clone()  # noqa: E731
    area = {"x": col(0), "z": col(1), "b": col(2), "front": col(3), "depth": col(4), "on": own.clone()}
    pace = torch.tensor([Nt.PACE_FAST if s["pace"] == "fast" else Nt.PACE_SLOW for s in sp], device=device)
    return st, own, area, pace


def to_area(st, area):
    """The enemy's script (its slots: side 2): run to the area's centre; there (its centre within GRAB_M of the area) or in
    melee attack the nearest of ours if that one is within GRAB_M of the area or fighting it, else hold."""
    u = st.u
    a0 = {k: v[:, 0] for k, v in area.items()}
    o = opponents.nearest(st)                                    # (ATTACK of the nearest, every unit)
    near, _ = opponents._nearest(st)
    sd = lambda x, z: Dc.signed_dist(x, z, {k: v[:, None] for k, v in a0.items()})     # noqa: E731
    me_in = sd(u["x"], u["z"]) >= -GRAB_M
    tx, tz = u["x"].gather(1, near), u["z"].gather(1, near)
    t_in = sd(tx, tz) >= -GRAB_M
    fight = (me_in & t_in) | u["m"]
    run = ~me_in & ~u["m"]
    go = O.hold(st.B, st.N, st.device)
    go.kind = torch.where(fight, o.kind, torch.where(run, O.MOVE, O.HOLD))
    go.target = torch.where(fight, o.target, torch.full_like(o.target, -1))
    go.x = torch.where(run, a0["x"][:, None].expand_as(u["x"]), u["x"])
    go.z = torch.where(run, a0["z"][:, None].expand_as(u["z"]), u["z"])
    go.run = fight | run
    return go


class Ender:
    """U2's end rule after every step, and the stage: arrived [B] (our centre once in the area: 'in place' from then on),
    arrive_t [B]; the limit ARRIVED_S after the arrival, else the walk there x 1.5 + 60 s."""

    def __init__(self, sp, device):
        B = len(sp)
        self.over = torch.zeros(B, dtype=torch.bool, device=device)
        self.why = torch.zeros(B, dtype=torch.long, device=device)
        self.arrived = torch.zeros(B, dtype=torch.bool, device=device)
        self.arrive_t = torch.full((B,), -1.0, device=device)
        self.enemy_t = torch.full((B,), -1.0, device=device)
        self.never = torch.tensor([s["t_us"] * 1.5 + 60.0 for s in sp], device=device)

    def after_step(self, st, spacing, area):
        u = st.u
        H = st.N // 2
        a0 = {k: v[:, 0] for k, v in area.items()}
        live = ~self.over
        inside = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], a0) >= 0
        new_in = live & inside & ~self.arrived
        self.arrive_t = torch.where(new_in, st.t, self.arrive_t)
        self.arrived = self.arrived | new_in
        e_in = live & (Dc.signed_dist(u["x"][:, H], u["z"][:, H], a0) >= 0) & (self.enemy_t < 0)
        self.enemy_t = torch.where(e_in, st.t, self.enemy_t)
        we_out = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        inz, _ = V.zone(now, idx, torch.zeros_like(idx))
        foe_gone = (u["men"][:, H] <= 0) | u["gone"][:, H] | (u["r"][:, H] & ~inz[:, H])
        late = torch.where(self.arrived, st.t >= self.arrive_t + ARRIVED_S, st.t >= self.never)
        new = live & (we_out | foe_gone | late)
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(foe_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foe_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
