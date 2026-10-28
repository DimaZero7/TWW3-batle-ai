"""Drill U5b 'a clean line of fire' (build/diag_g49/unit_base.md, U5): two of our units under the network, the enemy's
infantry (1-2, under drill_go.to_area) coming at us. Half the skirmishes each layout:
    shooters   two shooters in neighbouring positions as a tactical level would cut them - side by side ('side') or one
               behind the other ('file') - with overlapping fire sectors
    screen     a shooter behind our melee infantry: the infantry holds its area in front (targets in its area), the
               shooter its position behind it (targets in its sector, U5's rule)
The simulator's line of fire past friends (missile.friend_los, the game's rules): the men of another own unit in the
way block a shooter's line; direct fire (bullets: handgunners, long rifles, jezzails, the ratling gun) holds fire at
that target (the next one in range, else none); arcing fire (arrows, bolts, slings) lobs at 45 deg over friends beyond
10 m; any man whose own path passes below a friend's head loses his shot (the bookkeeping fc_cap / fc_skip); the shots
of the clear lines that pass low over friends hit them (missile.arc_los catch, full damage: can_damage_allies).
The reward: the side's, shared (as U4); the order changes cost as everywhere.
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import drill_shoot as DS
from tools.nn.unit import zmap as Z

LAYOUTS = ("shooters", "screen")
SUBS = {"shooters": ("side", "file"), "screen": ("screen",)}
SECTOR_W = (60.0, 120.0)
START_M = (150.0, 300.0)
LAG_S = (0.0, 15.0)
GRAB_M = 30.0
LIMIT_S = DH.LIMIT_S
C = DH.CELL


def _unit_at(xl, back, w, h):
    """A unit's place at the cell corner nearest its area's middle: (right, forward) in the line's frame."""
    return xl + round(w / 2) * C, back + round(h / 2) * C


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of layouts to cycle (the exam)."""
    S, R = DS.shooters(), DH.roster()
    pp = DH._passports()
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        layout = fixed[k % len(fixed)] if fixed else LAYOUTS[ri(2)]
        sub = SUBS[layout][ri(len(SUBS[layout]))]
        if layout == "shooters":
            keys = [S[ours][ri(len(S[ours]))], S[ours][ri(len(S[ours]))]]
            sizes = [(ri(5) + 4, ri(5) + 4), (ri(5) + 4, ri(5) + 4)]                # positions 4-8 cells
        else:
            keys = [R[ours][ri(len(R[ours]))], S[ours][ri(len(S[ours]))]]
            lo = DH.min_cells()
            sizes = [(ri(9 - lo) + lo, ri(9 - lo) + lo), (ri(5) + 4, ri(5) + 4)]    # the infantry's area, the position
        units = []
        for key, (w, h) in zip(keys, sizes):
            f = u(*DH.WIDTH)
            front, depth = DH.box(pp[key], f)
            units.append({"key": key, "front": front, "depth": depth, "w": w, "h": h})
        b = u(0.0, 360.0)
        sb = (b + u(-15.0, 15.0)) % 360
        sw = u(*SECTOR_W)
        foes = []
        lag = 0.0
        for _ in range(1 + ri(2)):
            key = R[theirs][ri(len(R[theirs]))]
            foes.append({"key": key, "rel": u(-0.8, 0.8) * sw / 2, "gap": u(*START_M) + lag * pp[key]["speed"]["run"]})
            lag = u(*LAG_S)
        out.append({"ours": ours, "theirs": theirs, "layout": layout, "sub": sub, "b": b, "sb": sb, "sw": sw,
                    "sb2": (sb + u(-15.0, 15.0)) % 360, "units": units, "gap_cells": ri(2), "foes": foes})
    return out


def exam_specs(seed=4913, n=240):
    """The fixed exam: n skirmishes, the two layouts in turn."""
    return specs(n, torch.Generator().manual_seed(seed), fixed=list(LAYOUTS))


def _layout(s):
    """[(area right, forward, w m, h m, unit right, forward)] of our two in the line's frame, centred on the pair."""
    a, b = s["units"]
    g = s["gap_cells"] * C
    if s["sub"] == "side":
        xa, xb = 0.0, a["w"] * C + g
        ba, bb = -a["h"] * C / 2, -b["h"] * C / 2
    else:                    # 'file' / 'screen': the second unit's area right behind the first's
        xa = 0.0
        xb = (a["w"] - b["w"]) * C / 2
        ba = 0.0
        bb = -(b["h"] * C + g)
    out = []
    for un, xl, back in ((a, xa, ba), (b, xb, bb)):
        w, h = un["w"] * C, un["h"] * C
        ux, uz = _unit_at(xl, back, un["w"], un["h"])
        out.append([xl + w / 2, back + h / 2, w, h, ux, uz])
    mx = sum(o[0] for o in out) / 2
    mz = sum(o[1] for o in out) / 2
    for o in out:
        o[0] -= mx
        o[4] -= mx
        o[1] -= mz
        o[5] -= mz
    return out


def build(sp, P, device):
    """The state (ours: slots 0 and 1; the enemies from H), own [B, N], our areas {x, z, b, front, depth, on, sb, sw}
    [B, N] (sw 0: no sector - the infantry), the enemies' goal (drill_go.to_area reads slot 0's: our front unit's area)."""
    armies, lays = [], []
    for s in sp:
        lay = _layout(s)
        mine = []
        for un, o in zip(s["units"], lay):
            x, z = Z.world(torch.tensor(o[4]), torch.tensor(o[5]), 0.0, 0.0, s["b"])
            mine.append({"key": un["key"], "x": float(x), "z": float(z), "b": s["b"], "width": un["front"]})
        foes = []
        for f in s["foes"]:
            a = math.radians(s["sb"] + f["rel"])
            ex, ez = f["gap"] * math.sin(a), f["gap"] * math.cos(a)
            foes.append({"key": f["key"], "x": ex, "z": ez, "b": math.degrees(math.atan2(-ex, -ez)) % 360})
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": mine},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": foes}}})
        lays.append(lay)
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, :2] = True
    area = {k: torch.zeros((B, N), device=device) for k in ("x", "z", "b", "front", "depth", "sb", "sw")}
    goal = {k: torch.zeros((B, N), device=device) for k in ("x", "z", "b", "front", "depth")}
    for b_, (s, lay) in enumerate(zip(sp, lays)):
        for i, o in enumerate(lay):
            x, z = Z.world(torch.tensor(o[0]), torch.tensor(o[1]), 0.0, 0.0, s["b"])
            area["x"][b_, i], area["z"][b_, i], area["b"][b_, i] = float(x), float(z), s["b"]
            area["front"][b_, i], area["depth"][b_, i] = o[2], o[3]
            shooter = s["layout"] == "shooters" or i == 1
            area["sb"][b_, i] = (s["sb"] if i == 0 else s["sb2"]) if shooter else 0.0
            area["sw"][b_, i] = s["sw"] if shooter else 0.0
        o = lay[0]                                   # the enemies go at our front unit's area
        x, z = Z.world(torch.tensor(o[0]), torch.tensor(o[1]), 0.0, 0.0, s["b"])
        goal["x"][b_], goal["z"][b_], goal["b"][b_] = float(x), float(z), s["b"]
        goal["front"][b_], goal["depth"][b_] = o[2], o[3]
    area["on"] = own.clone()
    return st, own, area, goal
