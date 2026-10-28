"""Drill U8 'a priority target' (build/diag_g49/unit_base.md, the user's decision 28.10): the levels above may name the
unit's main target (the order's 'has_target' and the 'my_target' flag of that enemy's row - the same main target of the
order, not a second notion); while it is allowed (its row's target_ok: infantry - its centre in the area; a shooter -
in its fire sector and within its range) the unit attacks it and does not change it; none named - the unit chooses
itself. Not a hard mask: a small cost a second while the named target is allowed and the network's target is another
one (or nobody) - train.py --prio-cost.

Two layouts, half the skirmishes each:
    duel   our shooter (drill_shoot: its position and fire sector) against an enemy shooter (the same pure-shooter
           roster, the other faction) standing in our sector DUEL_GAP x our range away and under `nearest` (it closes to
           its own range and fires at us), and an enemy melee infantry coming at our position (drill_shoot 'at',
           drill_go.to_area); the priority: the enemy shooter or the infantry, at random
    two    infantry holding its area against two enemies A and B (drill_multi, U3: its layouts, timings and hard
           skirmishes), both under `nearest`; the priority: A or B, at random
A priority is given in PRIO_SHARE of the skirmishes, none in the rest. The slots: ours 0, the enemies H (the duel's
shooter, U3's A) and H + 1 (the duel's infantry, U3's B). The end (drill_multi.Ender): our rout - over; both enemies
dead, gone or routed out of our zone - over; at most LIMIT_S.
"""
import math

import torch

from tools.nn.sim import orders as O
from tools.nn.sim import scenario
from tools.nn.train import opponents
from tools.nn.unit import drill_go as DG
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import drill_multi as DM
from tools.nn.unit import drill_shoot as DS

LAYOUTS = ("duel", "two")
PRIOS = {"duel": ("shooter", "infantry"), "two": ("A", "B")}     # in slot order from H
PRIO_SHARE = 0.7
DUEL_GAP = (0.5, 0.95)          # the enemy shooter's distance, x our range (it starts within our reach)
GRAB_M = DS.GRAB_M
LIMIT_S = DH.LIMIT_S
EXAM_CELLS = [(lay, p) for lay in LAYOUTS for p in PRIOS[lay] + (None,)]


def _range(key):
    return float(DH._passports()[key]["missile"]["range_m"])


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of (layout, priority or None) to cycle (the exam)."""
    S = DS.shooters()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        if fixed:
            layout, prio = fixed[k % len(fixed)]
        else:
            layout = LAYOUTS[ri(2)]
            prio = PRIOS[layout][ri(2)] if float(torch.rand(1, generator=gen)) < PRIO_SHARE else None
        if layout == "duel":
            s = DS.specs(1, gen, fixed=["at"])[0]
            s.update({"foe_s": S[s["theirs"]][ri(len(S[s["theirs"]]))], "s_rel": u(-0.8, 0.8) * s["sw"] / 2,
                      "s_gap": u(*DUEL_GAP) * _range(s["me"])})
        else:
            s = DM.specs(1, gen)[0]
        s.update({"layout": layout, "prio": prio})
        out.append(s)
    return out


def exam_specs(seed=4915, per=40):
    """The fixed exam: per skirmishes of each layout x priority (duel: shooter / infantry / none; two: A / B / none)."""
    return specs(per * len(EXAM_CELLS), torch.Generator().manual_seed(seed), fixed=EXAM_CELLS)


def prio_slot(s, H):
    """The priority's slot (H or H + 1), -1 none."""
    return -1 if s["prio"] is None else H + PRIOS[s["layout"]].index(s["prio"])


def build(sp, P, device):
    """The state, own [B, N] (slot 0), the areas {x, z, b, front, depth, on, sb, sw} [B, N] (sw 0: infantry), the duel
    infantry's goal {x, z, b, front, depth} [B, N] (our position), prio [B, N] long (slot 0: the priority's slot, -1
    none) and duel [B] bool."""
    armies, pos, goal = [], [], []
    for s in sp:
        if s["layout"] == "duel":
            arm, p_, g_ = DS.army(s)
            a = math.radians(s["sb"] + s["s_rel"])
            ex, ez = s["s_gap"] * math.sin(a), s["s_gap"] * math.cos(a)
            shooter = {"key": s["foe_s"], "x": ex, "z": ez, "b": math.degrees(math.atan2(-ex, -ez)) % 360}
            arm["sides"][2]["units"].insert(0, shooter)                         # slot H: the shooter, H + 1: infantry
        else:
            arm = DM.army(s)
            x, z = DM.area_centre(s)
            p_ = (x, z, s["b"], s["w"] * DH.CELL, s["h"] * DH.CELL, 0.0, 0.0)
            g_ = (x, z)
        armies.append(arm)
        pos.append(p_)
        goal.append(g_)
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    H = N // 2
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {k: col([p_[i] for p_ in pos]) for i, k in enumerate(("x", "z", "b", "front", "depth", "sb", "sw"))}
    area["on"] = own.clone()
    gl = {"x": col([g_[0] for g_ in goal]), "z": col([g_[1] for g_ in goal]), "b": col([0.0] * B),
          "front": col([DS.GOAL_M] * B), "depth": col([DS.GOAL_M] * B)}
    prio = torch.full((B, N), -1, dtype=torch.long, device=device)
    prio[:, 0] = torch.tensor([prio_slot(s, H) for s in sp], dtype=torch.long, device=device)
    duel = torch.tensor([s["layout"] == "duel" for s in sp], dtype=torch.bool, device=device)
    return st, own, area, gl, prio, duel


def script(st, goal, duel):
    """The enemies' orders: `nearest` for all (a shooter closes to its range and fires), the duel's infantry (slot H + 1)
    drill_go.to_area to our position."""
    o = opponents.nearest(st)
    g = DG.to_area(st, goal, grab_m=GRAB_M)
    m = torch.zeros((st.B, st.N), dtype=torch.bool, device=st.device)
    m[:, st.N // 2 + 1] = duel
    return O.merge(o, g, m)
