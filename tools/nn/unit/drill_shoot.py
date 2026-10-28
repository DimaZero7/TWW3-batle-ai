"""Drill U5 'shooters', the first one: fire at one target (build/diag_g49/unit_base.md, U5 - the shooters' order decided
by the user 28.10: a position (a small area) + a fire sector (its direction and width) + a main target (optional) + the
pace). The shooter may target any enemy in its sector within its range, however far beyond its position (view.allowed);
it moves only within its position (the hard ban, as infantry); it fires and does not walk to its target (decode.Leash:
an attack on an enemy beyond range is a hold facing it). Hybrids (infantry with fire) are not here yet.

A skirmish (specs):
* our shooter: the missile infantry of the Empire and the Skaven from the passports, hybrids out (the database's melee
  combat potential at least HYBRID_RATIO x its missile one: free company militia, Nuln ironsides, huntsmen); a random
  bearing and frontage (drill_hold.WIDTH);
* its position: w x h cells of 8 m (POS_CELLS) on its grid at the order, the unit at a cell corner near the middle;
* its fire sector: the middle within SECTOR_OFF of its front, SECTOR_W wide;
* one enemy, melee infantry of the other faction, starting in the sector (within 0.8 of its half width) START_M away;
  it goes (drill_go.to_area with a GOAL_M x GOAL_M goal) 'at' the shooter (the goal: its position) or 'past' it (the
  goal PAST_M away across the sector, near its edge), and attacks us only if we come within GRAB_M of its goal or touch
  it;
* the end (Ender): our rout or death - over; the enemy dead, gone or routing out of our zone - over; at most LIMIT_S.
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

POS_CELLS = (4, 8)
SECTOR_W = (60.0, 120.0)
SECTOR_OFF = 30.0
START_M = (150.0, 350.0)
PAST_M = (60.0, 120.0)
GOAL_M = 32.0
GRAB_M = 30.0
MODES = ("at", "past")
HYBRID_RATIO = 0.5
LIMIT_S = DH.LIMIT_S


def shooters():
    """{"emp": [keys], "skv": [keys]}: the pure shooters (missile infantry, not hybrids)."""
    pp = DH._passports()
    out = {"emp": [], "skv": []}
    for k, v in pp.items():
        if v.get("caste") != "missile_infantry" or not v.get("missile"):
            continue
        cp = v["combat_potential"]
        if cp["melee"] >= HYBRID_RATIO * cp["missile"]:
            continue
        side = "skv" if "_skv_" in k else "emp" if "_emp_" in k else None
        if side:
            out[side].append(k)
    return out


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of modes to cycle (the exam)."""
    S, R = shooters(), DH.roster()
    pp = DH._passports()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        me, foe = S[ours][ri(len(S[ours]))], R[theirs][ri(len(R[theirs]))]
        mode = fixed[k % len(fixed)] if fixed else MODES[ri(2)]
        factor = u(*DH.WIDTH)
        front, depth = DH.box(pp[me], factor)
        b = u(0.0, 360.0)
        sw = u(*SECTOR_W)
        sb = (b + u(-SECTOR_OFF, SECTOR_OFF)) % 360
        side = 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1
        out.append({"me": me, "foe": foe, "ours": ours, "theirs": theirs, "mode": mode, "b": b, "factor": factor,
                    "front": front, "depth": depth, "w": ri(POS_CELLS[1] - POS_CELLS[0] + 1) + POS_CELLS[0],
                    "h": ri(POS_CELLS[1] - POS_CELLS[0] + 1) + POS_CELLS[0], "sb": sb, "sw": sw,
                    "start_rel": u(-0.8, 0.8) * sw / 2, "start": u(*START_M),
                    "past_rel": side * u(0.5, 0.9) * sw / 2, "past": u(*PAST_M)})
    return out


def exam_specs(seed=4909, n=240):
    """The fixed exam: n skirmishes, 'at' and 'past' in turn."""
    return specs(n, torch.Generator().manual_seed(seed), fixed=list(MODES))


def build(sp, P, device):
    """The state (ours slot 0, the enemy slot H), own [B, N], our position {x, z, b, front, depth, on, sb, sw} [B, N] and
    the enemy's goal {x, z, b, front, depth} [B, N] (drill_go.to_area reads slot 0's)."""
    armies, pos, goal = [], [], []
    C = DH.CELL
    for s in sp:
        # the position: the unit at the cell corner nearest its middle
        kx, kz = round(s["w"] / 2), round(s["h"] / 2)
        px, pz = Z.world(torch.tensor((s["w"] / 2 - kx) * C), torch.tensor((s["h"] / 2 - kz) * C), 0.0, 0.0, s["b"])
        a = math.radians(s["sb"] + s["start_rel"])
        ex, ez = s["start"] * math.sin(a), s["start"] * math.cos(a)
        if s["mode"] == "at":
            gx, gz = float(px), float(pz)
        else:
            g = math.radians(s["sb"] + s["past_rel"])
            gx, gz = s["past"] * math.sin(g), s["past"] * math.cos(g)
        mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
        theirs = {"key": s["foe"], "x": ex, "z": ez, "b": math.degrees(math.atan2(gx - ex, gz - ez)) % 360}
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": [theirs]}}})
        pos.append((float(px), float(pz), s["b"], s["w"] * C, s["h"] * C, s["sb"], s["sw"]))
        goal.append((gx, gz))
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {k: col([p_[i] for p_ in pos]) for i, k in enumerate(("x", "z", "b", "front", "depth", "sb", "sw"))}
    area["on"] = own.clone()
    gl = {"x": col([g_[0] for g_ in goal]), "z": col([g_[1] for g_ in goal]), "b": col([0.0] * B),
          "front": col([GOAL_M] * B), "depth": col([GOAL_M] * B)}
    return st, own, area, gl


class Ender:
    """U5's end rule after every step: our rout or death - over; the enemy dead, gone or routing out of our zone - over;
    at most LIMIT_S."""

    def __init__(self, B, device, limit_s=LIMIT_S):
        self.over = torch.zeros(B, dtype=torch.bool, device=device)
        self.why = torch.zeros(B, dtype=torch.long, device=device)
        self.arrived = torch.ones(B, dtype=torch.bool, device=device)
        self.limit = limit_s

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
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(foe_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foe_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
