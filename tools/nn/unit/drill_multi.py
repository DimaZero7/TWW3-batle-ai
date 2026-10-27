"""Drill U3 'several enemies' (build/diag_g49/unit_base.md, Drills): as U1 (hold in the area, the same 16 melee infantry),
but two enemies A and B of the other faction (any two of its pool, the same kind twice too), both under `nearest`.

A skirmish: our unit, its area and its random frontage and facing as U1 (drill_hold); the enemies from different sides,
the layout (the bearings from our front, each +-JITTER deg; the side of the flank at random):
    front_flank   A ahead, B on a flank (90 deg)
    front_rear    A ahead, B behind (180 deg)
    front_angle   both ahead, ANGLE deg either side
the timing: 'together' (both arrive at once) or 'staggered' (B LAG_S s after A): A's centre GAP_M m away (as U1), B's
(A's time at its run + the lag) x B's run speed. A is the first to arrive (the branches' A); its bearing is relative to our front,
so 'ahead' means where our unit faces at the start, which is random anyway.
Hard skirmishes (HARD_SHARE of them): a weak A first and a dangerous B 5-20 s later (danger(): the database's melee
combat potential, B >= HARD_RATIO x A, both of one faction), front + flank or both ahead - where 'take A, then turn to B'
can pay (the probe build/diag_g49/probe_gs); the enemies' faction at random against ours (Empire against Empire, Skaven
against Skaven too: the simulator takes any two factions - the probe ran greatswords against greatswords).
The end (Ender): our rout - over; both enemies dead, gone or routed out of our zone - over; at most LIMIT_S.
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

LAYOUTS = {"front_flank": (0.0, 90.0), "front_rear": (0.0, 180.0), "front_angle": (-35.0, 35.0)}
TIMINGS = ("together", "staggered")
JITTER = 15.0
LAG_S = (5.0, 20.0)
LIMIT_S = DH.LIMIT_S
HARD_SHARE = 0.5                                     # the share of hard skirmishes in training
HARD_RATIO = 1.6                                     # B's combat potential at least this x A's
HARD_LAYOUTS = ("front_flank", "front_angle")        # not from the opposite side


def _spec(me, a, b, ours, layout, timing, u, gen, lo, pp):
    base = DH._spec(me, a, ours, gen, lo, pp, None)
    ra, rb = LAYOUTS[layout]
    side = 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1
    if layout != "front_angle":
        rb = rb * side
    base.update({"foe_b": b, "layout": layout, "timing": timing, "rel": (ra + u(-JITTER, JITTER)) % 360,
                 "rel_b": (rb + u(-JITTER, JITTER)) % 360, "lag": u(*LAG_S) if timing == "staggered" else 0.0})
    # B arrives lag s after A (together: 0), each at its run: a faster B starts farther
    base["gap_b"] = (base["gap"] / pp[a]["speed"]["run"] + base["lag"]) * pp[b]["speed"]["run"]
    return base


def danger(key):
    """A unit's strength: the database's melee combat potential (main_units.melee_cp, the passport's
    combat_potential.melee: slaves 125, clanrats 300, shield spearmen 375, greatswords 850, stormvermin 1050)."""
    return float(DH._passports()[key]["combat_potential"]["melee"])


def hard_pairs():
    """{faction: [(A, B)]}: a weak first enemy and a dangerous second of the same faction (B's combat potential at least
    HARD_RATIO x A's)."""
    R = DH.roster()
    return {f: [(a, b) for a in ks for b in ks if danger(b) >= HARD_RATIO * danger(a)] for f, ks in R.items()}


def specs(n, gen, fixed=None, hard_share=None):
    """n random skirmishes; fixed: a list of (layout, timing) to cycle (the exam's ordinary ones); hard_share (default
    HARD_SHARE, the exam 0): the share of hard ones - a weak enemy first and a dangerous one 5-20 s later, not from the
    opposite side (front + flank or both ahead), the enemies of either faction against ours of either (a mirror too)."""
    R = DH.roster()
    pp = DH._passports()
    lo = DH.min_cells()
    hp = hard_pairs()
    hs = HARD_SHARE if hard_share is None else hard_share
    out = []
    for k in range(n):
        u = lambda x, y: x + (y - x) * float(torch.rand(1, generator=gen))       # noqa: E731
        if float(torch.rand(1, generator=gen)) < hs:
            ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
            theirs = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
            me = R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))]
            a, b = hp[theirs][int(torch.randint(len(hp[theirs]), (1,), generator=gen))]
            layout = HARD_LAYOUTS[int(torch.randint(len(HARD_LAYOUTS), (1,), generator=gen))]
            sp = _spec(me, a, b, ours, layout, "staggered", u, gen, lo, pp)
            sp.update({"theirs": theirs, "hard": True})
            out.append(sp)
            continue
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        pick = lambda: R[theirs][int(torch.randint(len(R[theirs]), (1,), generator=gen))]   # noqa: E731
        me = R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))]
        layout, timing = fixed[k % len(fixed)] if fixed else (
            list(LAYOUTS)[int(torch.randint(len(LAYOUTS), (1,), generator=gen))],
            TIMINGS[int(torch.randint(len(TIMINGS), (1,), generator=gen))])
        sp = _spec(me, pick(), pick(), ours, layout, timing, u, gen, lo, pp)
        sp.update({"theirs": theirs, "hard": False})
        out.append(sp)
    return out


def exam_specs(seed=4905, per=40, hard=120):
    """The fixed exam: per ordinary skirmishes for each layout x timing (6 x 40 = 240) and `hard` hard ones (360)."""
    gen = torch.Generator().manual_seed(seed)
    cells = [(l_, t_) for l_ in LAYOUTS for t_ in TIMINGS]
    return specs(per * len(cells), gen, fixed=cells, hard_share=0.0) + specs(hard, gen, hard_share=1.0)


def build(sp, P, device):
    """The state (ours slot 0; A slot H, B slot H + 1), own [B, N], the areas {x, z, b, front, depth, on} [B, N]."""
    armies = []
    for s in sp:
        foes = []
        for key, rel, gap in ((s["foe"], s["rel"], s["gap"]), (s["foe_b"], s["rel_b"], s["gap_b"])):
            a = math.radians(s["b"] + rel)
            ex, ez = gap * math.sin(a), gap * math.cos(a)
            foes.append({"key": key, "x": ex, "z": ez, "b": math.degrees(math.atan2(-ex, -ez)) % 360})
        mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
        f_me = DH.FACTIONS[s["ours"]]
        f_foe = DH.FACTIONS[s.get("theirs") or ("skv" if s["ours"] == "emp" else "emp")]
        armies.append({"attacker": 2, "sides": {1: {"faction": f_me, "units": [mine]},
                                                2: {"faction": f_foe, "units": foes}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    cx, cz = [], []
    for s in sp:
        x, z = Z.world(torch.tensor((s["w"] / 2 - s["kx"]) * DH.CELL), torch.tensor((s["h"] / 2 - s["kz"]) * DH.CELL),
                       0.0, 0.0, s["b"])
        cx.append(float(x)), cz.append(float(z))
    f = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {"x": f(cx), "z": f(cz), "b": f([s["b"] for s in sp]), "front": f([s["w"] * DH.CELL for s in sp]),
            "depth": f([s["h"] * DH.CELL for s in sp]), "on": own.clone()}
    return st, own, area


class Ender:
    """U3's end rule after every step (as drill_hold.Ender with two enemies)."""

    def __init__(self, B, device, limit_s=LIMIT_S):
        self.over = torch.zeros(B, dtype=torch.bool, device=device)
        self.why = torch.zeros(B, dtype=torch.long, device=device)
        self.arrived = torch.ones(B, dtype=torch.bool, device=device)
        self.limit = limit_s

    def after_step(self, st, spacing, area=None):
        u = st.u
        foe = (u["side"] == 2) & (u["men0"] > 0)
        we_out = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
        now = V.raw(st, spacing)
        idx = torch.arange(st.B, device=st.device)
        inz, _ = V.zone(now, idx, torch.zeros_like(idx))
        out = (u["men"] <= 0) | u["gone"] | (u["r"] & ~inz)
        foes_gone = (~foe | out).all(1)
        late = st.t >= self.limit
        new = ~self.over & (we_out | foes_gone | late)
        self.why = torch.where(new, torch.where(we_out, 1, torch.where(foes_gone, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we_out, 2, torch.where(foes_gone, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new
