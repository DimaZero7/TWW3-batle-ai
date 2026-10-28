"""Drill U7 'a routing enemy' (build/diag_g49/unit_base.md, the drills' plan): our melee infantry holds its area (as U1);
the enemy (the same pool, under `nearest`) is either breaking in melee with us or fleeing across our area. The
skill: cut the routers down while they are in the area (routers die cheap by the game's rules: the pursuit's blows,
no fight back), not chase past the edge (the leash), come back and stand.

A skirmish (specs): our unit at a random bearing and frontage, its area w x h cells of 8 m (drill_hold.MIN_CELLS..10)
on its grid, the unit at a cell corner near the middle (as U1); the enemy:
    breaking  touching our front, attacking us, already worn: LOST of its men and health gone over the last FIGHT_S
              seconds (the morale's casualty windows' history: u lost_hist), its morale points MORALE x its leadership -
              the simulator's morale rules (the losses, the losing fight) take it from there; it routs soon or holds
    crossing  routing already (the simulator's rout state: r, MoraleState 6) from a third unit - one of ours, not the
              network's, chasing it (script below) - beside our area: the router starts SIDE_M m to our left, the third
              unit THIRD_M m beyond it, so that its flight (away from every standing enemy, 1/d weighted:
              movement.flee_goal) runs across our front MISS_M m ahead of our centre - near, it crosses our area within
              the reach of our charge; far, past the area's front edge (not ours to chase). A third unit that only
              stood does not do: the router turns away from our unit long before our front (the trace,
              build/diag_g49/u7/trace.py).
              ('fleeing straight away from us' is out: a router runs at 0.985 of its run and our infantry never
              catches it within the area - the scripts' check, build/diag_g49/u7/scripts.py)
Neither 'just rallied': the post-rally no-rout timer is long over (rally_s set past it), as for any unit in a long
fight. The end (Ender): our rout - over; the enemy dead, gone or out of our zone - over; at most LIMIT_S.
"""
import math

import torch

from tools.nn.sim import battle, geometry, scenario
from tools.nn.sim import orders as O
from tools.nn.train import opponents
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

KINDS = ("breaking", "crossing")
LOST = (0.6, 0.85)             # the enemy's share of men / health already lost (the sim: 0.6 -> 38 % rout within 30 s, 0.7 -> 65 %, 0.8 -> 97 %)
MORALE = (0.05, 0.3)           # its morale points / leadership (breaking)
OUR_LOST = (0.0, 0.3)
SIDE_M = (15.0, 45.0)          # crossing: the router's start to our left (often within the area's half width)
THIRD_M = (8.0, 15.0)          # ... the third unit beyond it
MISS_M = (5.0, 35.0)           # ... its line across our front, ahead of our centre
FIGHT_S = (30.0, 60.0)        # the losses so far came over the last this many seconds (the casualty windows' history)
WARM_S = 1.0
LIMIT_S = 120.0


def specs(n, gen, fixed=None):
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
        kind = fixed[k % len(fixed)] if fixed else KINDS[ri(2)]
        factor = u(*DH.WIDTH)
        front, depth = DH.box(pp[me], factor)
        w, h = ri(DH.MAX_CELLS - lo + 1) + lo, ri(DH.MAX_CELLS - lo + 1) + lo
        out.append({"me": me, "foe": foe, "ours": ours, "theirs": theirs, "kind": kind, "b": u(0.0, 360.0),
                    "factor": factor, "front": front, "depth": depth, "w": w, "h": h, "lost": u(*LOST),
                    "morale": u(*MORALE), "our_lost": u(*OUR_LOST), "side_off": u(-10.0, 10.0),
                    "fight_s": u(*FIGHT_S), "side_m": u(*SIDE_M), "third_m": u(*THIRD_M), "miss_m": u(*MISS_M),
                    "third": R[ours][ri(len(R[ours]))]})
    return out


def exam_specs(seed=4915, n=240):
    return specs(n, torch.Generator().manual_seed(seed), fixed=list(KINDS))


def build(sp, P, device):
    """The state (ours slot 0, the third unit slot 1 when crossing, the enemy slot H), own [B, N] (slot 0) and our
    areas [B, N]."""
    armies = []
    for s in sp:
        mine = [{"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}]
        if s["kind"] == "crossing":
            # the third unit to our left beyond the router's start, facing the way the router will flee
            x, z = Z.world(torch.tensor(-(s["side_m"] + s["third_m"])), torch.tensor(s["miss_m"]), 0.0, 0.0, s["b"])
            mine.append({"key": s["third"], "x": float(x), "z": float(z), "b": (s["b"] + 90.0) % 360})
        theirs = {"key": s["foe"], "x": 0.0, "z": 300.0, "b": (s["b"] + 180.0) % 360}
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": mine},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": [theirs]}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    H = N // 2
    u = st.u
    touch = min(0.0, P.sim["contact"]["reach_m"])
    no_rout = float(P.morale["post_rally_no_rout_timer"]) + 1.0
    for b, s in enumerate(sp):
        for i, lost in ((0, s["our_lost"]), (H, s["lost"])):
            keep = 1.0 - lost
            u["men"][b, i] = torch.ceil(u["men0"][b, i] * keep)
            u["hp_abs"][b, i] = u["hp0"][b, i] * keep
            u["hp"][b, i] = keep
            u["rally_s"][b, i] = no_rout
            # the losses' history (the morale's sliding casualty windows, u lost_hist: HP lost each step, newest
            # first): lost evenly over the last s["fight_s"] seconds of the fight so far
            if "lost_hist" in u and u["lost_hist"].shape[1]:
                Kh = u["lost_hist"].shape[1] // N
                n = min(Kh, int(round(s["fight_s"] / P.dt)))
                hist = u["lost_hist"][b].view(Kh, N)
                hist[:, i] = 0.0
                hist[:n, i] = float(u["hp0"][b, i]) * lost / max(int(round(s["fight_s"] / P.dt)), 1)
    front, depth = geometry.dims(u, P.sim["formation"]["spacing_m"])        # (the boxes after the losses)
    for b, s in enumerate(sp):
        a = math.radians(s["b"])
        if s["kind"] == "breaking":
            d = float(depth[b, 0] / 2 + depth[b, H] / 2) + touch
            u["morale"][b, H] = s["morale"] * u["leadership"][b, H]
            side = math.radians(s["b"] + 90.0)
            u["x"][b, H] = d * math.sin(a) + s["side_off"] * math.sin(side)
            u["z"][b, H] = d * math.cos(a) + s["side_off"] * math.cos(side)
        else:
            x, z = Z.world(torch.tensor(-s["side_m"]), torch.tensor(s["miss_m"]), 0.0, 0.0, s["b"])
            u["x"][b, H], u["z"][b, H] = float(x), float(z)
            u["b"][b, H] = (s["b"] + 90.0) % 360                      # turned the way it flees
            u["r"][b, H] = True
            u["ms"][b, H] = 6.0
            u["morale"][b, H] = -0.1 * u["leadership"][b, H]
        u["mp"][b, H] = u["morale"][b, H] / u["leadership"][b, H].clamp(min=1)
    step = battle.stepper(device)
    for _ in range(int(round(WARM_S / P.dt))):
        o = script(st)
        o.kind[:, 0], o.target[:, 0] = O.HOLD, -1
        step(st, o, P)
        # (the simulator ends a battle once a side has no standing unit - a routing enemy; the drill's end rule decides
        # here: only our own rout ends it in the warm-up)
        we_out = st.u["r"][:, 0] | (st.u["men"][:, 0] <= 0)
        st.done = st.done & we_out
        st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
    C = DH.CELL
    cx, cz = [], []
    for b, s in enumerate(sp):
        x, z = Z.world(torch.tensor((s["w"] / 2 - round(s["w"] / 2)) * C), torch.tensor((s["h"] / 2 - round(s["h"] / 2)) * C),
                       0.0, 0.0, s["b"])
        cx.append(float(x)), cz.append(float(z))
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    f = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {"x": f(cx), "z": f(cz), "b": f([s["b"] for s in sp]), "front": f([s["w"] * C for s in sp]),
            "depth": f([s["h"] * C for s in sp]), "on": own.clone()}
    return st, own, area


def script(st):
    """The enemy under `nearest`; the third unit (slot 1, ours, not the network's) chases the router (an attack on it:
    the simulator's pursuit at its run, no sprint) - the flight's source has to stay nearer the router than our unit is,
    or the router's flight (away from every standing enemy, 1/d) turns away from our unit before it reaches our front."""
    o = opponents.nearest(st)
    H = st.N // 2
    third = (st.u["side"][:, 1] == 1) & (st.u["men"][:, 1] > 0)
    o.kind[:, 1] = torch.where(third, O.ATTACK, O.HOLD)
    o.target[:, 1] = torch.where(third, H, -1)
    o.run[:, 1] = third
    return o


class Ender:
    """U7's end rule: our rout or death - over; the enemy dead, gone or out of our zone - over; at most LIMIT_S."""

    def __init__(self, st, limit_s=LIMIT_S):
        self.over = st.done.clone()                         # (done after the warm-up: only our own rout)
        self.why = torch.where(st.done, 1, 0)
        self.arrived = torch.ones(st.B, dtype=torch.bool, device=st.device)
        self.t_end = st.t + limit_s

    def after_step(self, st, spacing, area=None):
        u = st.u
        H = st.N // 2
        we_out = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        inz, _ = V.zone(now, idx, torch.zeros_like(idx))
        foe_gone = (u["men"][:, H] <= 0) | u["gone"][:, H] | ~inz[:, H]
        late = st.t >= self.t_end
        new = ~self.over & (we_out | foe_gone | late)
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(foe_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foe_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
