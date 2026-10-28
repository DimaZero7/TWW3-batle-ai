"""Drill U9 'the width for the task' (build/diag_g49/unit_base.md; the simulator's width order: Orders.width,
tools/nn/sim/reform.py; the network's width head: net.py, decode.width_of): our melee infantry holds an area whose
front (the tactical level's choice) is AREA_FRONT_M wide, against one or two enemy infantry units of random widths -
wider than ours (its flanks overlapped) or narrower; our own line starts at a random width. The skill: re-form to the
width the area and the enemy call for before the contact (the re-form takes the game's time: reform.ready_s), not to
be taken in the flank by the second enemy of a line, not to stretch thin.

Roster: the melee infantry of the Empire and the Skaven (drill_hold.roster), our unit of either faction, the enemies of
the other one.

A skirmish (specs):
* the area: front AREA_FRONT_M across, depth AREA_DEPTH_M (the narrowest line of the roster, 120 men in 5 files, is
  ~38 m deep), its front at a random bearing; our unit at its centre (+- CENTRE_JITTER_M across), facing the area's
  front +- FACE_JITTER, its line WIDTH_M wide (within its own limits: reform.limits on its men);
* kinds:
    one   one enemy ahead (the area's front +- ENEMY_BEARING), GAP_M from our centre, WIDTH_M wide (its own limits)
    two   two enemies side by side ahead (a line): the second beside the first, their centres (wA + wB) / 2 + LINE_GAP_M
          apart across the way to us, to a random side, the line's middle +- LINE_OFF_M of our centre's line; both
          at the same distance (+- 5 m), so they come together
  all under `nearest` (a run at the nearest of ours).
* the end (drill_multi.Ender): our rout - over; every enemy dead, gone or routed out of our zone - over; at most LIMIT_S.
The enemies' slots: H (A) and H + 1 (B; 'one': empty).
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import zmap as Z

KINDS = ("one", "two")
AREA_FRONT_M = (20.0, 80.0)
AREA_DEPTH_M = (40.0, 64.0)
WIDTH_M = (12.0, 72.0)         # a line's width at the start (ours and the enemies'), within each unit's own limits
CENTRE_JITTER_M = 4.0
FACE_JITTER = 15.0
ENEMY_BEARING = 25.0
GAP_M = (90.0, 170.0)          # the enemy's centre from ours: >= ~14 s at the fastest run (4.2 m/s) less the charge's
#                                last 30 m - time to re-form (30 -> 60 m at a run: 15 / 2 / 3 = 5 s; at a walk 10 + 4 s)
LINE_GAP_M = (2.0, 10.0)
LINE_OFF_M = 15.0
LIMIT_S = DH.LIMIT_S
SHARE = 0.04                   # config/nn/sim.json formation.reform.share (infantry): the narrowest files = men x this
MIN_FILES = 2


def width_limits(key):
    """(narrowest, widest) m of a passport's line (reform.limits on its men; the widest within decode.WIDTH_MAX_M)."""
    from tools.nn.unit import decode as Dc
    p = DH._passports()[key]
    h, men = p["spacing"]["h"], p["men"]
    return max(MIN_FILES, round(men * SHARE)) * h, min(men * h, Dc.WIDTH_MAX_M)


def _width(key, u):
    lo, hi = width_limits(key)
    return min(max(u(*WIDTH_M), lo), hi)


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of kinds to cycle (the exam)."""
    R = DH.roster()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        me = R[ours][ri(len(R[ours]))]
        kind = fixed[k % len(fixed)] if fixed else KINDS[ri(len(KINDS))]
        foes = [R[theirs][ri(len(R[theirs]))] for _ in range(1 if kind == "one" else 2)]
        b = u(0.0, 360.0)
        s = {"me": me, "ours": ours, "theirs": theirs, "kind": kind, "foes": foes, "b": b,
             "area_front": u(*AREA_FRONT_M), "area_depth": u(*AREA_DEPTH_M), "jitter": u(-CENTRE_JITTER_M, CENTRE_JITTER_M),
             "face": (b + u(-FACE_JITTER, FACE_JITTER)) % 360, "width": _width(me, u),
             "foe_widths": [_width(f, u) for f in foes], "bearing": u(-ENEMY_BEARING, ENEMY_BEARING), "gap": u(*GAP_M),
             "line_gap": u(*LINE_GAP_M), "line_side": 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1,
             "line_off": u(-LINE_OFF_M, LINE_OFF_M), "gap_b": u(-5.0, 5.0)}
        out.append(s)
    return out


def exam_specs(seed=4925, n=240):
    """The fixed exam: n skirmishes, 'one' and 'two' in turn."""
    return specs(n, torch.Generator().manual_seed(seed), fixed=list(KINDS))


def enemy_span(s):
    """The enemies' line across the way to us at the start, m (one: its width; two: both widths and the gap between)."""
    w = s["foe_widths"]
    return w[0] if len(w) == 1 else w[0] + w[1] + s["line_gap"]


def foes_of(s):
    """[(x, z, b, width)] the enemies' start: our centre at the origin."""
    a = math.radians(s["b"] + s["bearing"])
    way = (math.sin(a), math.cos(a))
    side = (math.cos(a), -math.sin(a))                     # across the way, to the right
    w = s["foe_widths"]
    if len(w) == 1:
        cx, cz = s["gap"] * way[0], s["gap"] * way[1]
        return [(cx, cz, math.degrees(math.atan2(-cx, -cz)) % 360, w[0])]
    d = (w[0] + w[1]) / 2 + s["line_gap"]
    off = [s["line_off"] - s["line_side"] * d / 2, s["line_off"] + s["line_side"] * d / 2]
    face = (s["b"] + s["bearing"] + 180.0) % 360          # the line faces back along the way
    out = []
    for k in range(2):
        g = s["gap"] + (s["gap_b"] if k == 1 else 0.0)
        out.append((g * way[0] + off[k] * side[0], g * way[1] + off[k] * side[1], face, w[k]))
    return out


def army(s):
    jx, jz = Z.world(torch.tensor(s["jitter"]), torch.tensor(0.0), 0.0, 0.0, s["b"])
    mine = {"key": s["me"], "x": float(jx), "z": float(jz), "b": s["face"], "width": s["width"]}
    foes = [{"key": k, "x": x, "z": z, "b": b, "width": w} for k, (x, z, b, w) in zip(s["foes"], foes_of(s))]
    return {"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                     2: {"faction": DH.FACTIONS[s["theirs"]], "units": foes}}}


def build(sp, P, device):
    """The state (ours slot 0; A slot H, B slot H + 1), own [B, N], the areas {x, z, b, front, depth, on} [B, N] (the
    area centred on the origin, its front at s["b"])."""
    st = scenario.build([army(s) for s in sp], P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    f = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {"x": f([0.0] * B), "z": f([0.0] * B), "b": f([s["b"] for s in sp]), "front": f([s["area_front"] for s in sp]),
            "depth": f([s["area_depth"] for s in sp]), "on": own.clone()}
    return st, own, area
