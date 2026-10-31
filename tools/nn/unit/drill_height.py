"""Drill U10 'the height in my area' (a real map only: --map, Kislev): the network's unit holds an area with a rise in it
(the ground's range over the area at least RELIEF_M), an enemy's infantry comes at it from above, from below or across
the slope; a shooter's variant: a position on a slope, the enemy's infantry coming at it from above / below. The skirmish
is drill_hold's (infantry: its area, the enemy at a random gap and angle, the script `nearest`) or drill_shoot's 'at'
(a shooter's position and sector, the enemy going at it: drill_go.to_area); its place on the map is the drill's own (not
realmap's random one): a point of the map whose ground over the area rises RELIEF_M or more, turned so that the enemy
starts uphill ('up': it comes down at us), downhill ('down') or across the slope ('side') of the area.

No hint (no branches): whether the network finds the high ground is the question. The exam: exam_height.py.
"""
import math

import numpy as np
import torch

from tools.nn.train import opponents
from tools.nn.unit import drill_go as DG
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import drill_shoot as DS
from tools.nn.unit import realmap
from tools.nn.unit import zmap as Z
from tools.nn.sim import scenario

KINDS = ("hold", "shoot")
FROM = ("up", "down", "side")
RELIEF_M = 4.0             # the ground's rise over the area at least (the user's 3-5 m)
GRID = 7                   # the area's sample points a side (its ground's range, its slope)
TRIES = 60
MORE = 10                  # tries x MORE at most while none has its points free (the river map's closed wings)
EDGE_M = 10.0
MARGIN_M = 40.0
LIMIT_S = DH.LIMIT_S


def specs(n, gen, fixed=None):
    """n random skirmishes; fixed: a list of (kind, from) to cycle (the exam)."""
    R, pp = DH.roster(), DH._passports()
    lo = DH.min_cells()
    out = []
    for k in range(n):
        if fixed:
            kind, frm = fixed[k % len(fixed)]
        else:
            kind = KINDS[int(torch.randint(len(KINDS), (1,), generator=gen))]
            frm = FROM[int(torch.randint(len(FROM), (1,), generator=gen))]
        if kind == "hold":
            ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
            theirs = "skv" if ours == "emp" else "emp"
            s = DH._spec(R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))],
                         R[theirs][int(torch.randint(len(R[theirs]), (1,), generator=gen))], ours, gen, lo, pp, None)
        else:
            s = DS.specs(1, gen, fixed=["at"])[0]
        s.update(kind=kind, frm=frm, side=1 if float(torch.rand(1, generator=gen)) < 0.5 else -1)
        out.append(s)
    return out


def exam_specs(seed=4940, n=240):
    """The fixed exam: n skirmishes, (hold, shoot) x (up, down, side) in turn."""
    return specs(n, torch.Generator().manual_seed(seed), fixed=[(k, f) for k in KINDS for f in FROM])


def _local(s):
    """(army, area (x, z, b, front, depth, sb, sw), goal (x, z) or None, the enemy's start (x, z)) in the local frame."""
    if s["kind"] == "shoot":
        arm, pos, goal = DS.army(s)
        foe = arm["sides"][2]["units"][0]
        return arm, pos, goal, (foe["x"], foe["z"])
    a = math.radians(s["b"] + s["rel"])
    ex, ez = s["gap"] * math.sin(a), s["gap"] * math.cos(a)
    mine = {"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}
    theirs = {"key": s["foe"], "x": ex, "z": ez, "b": (math.degrees(math.atan2(-ex, -ez))) % 360}
    arm = {"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                    2: {"faction": DH.FACTIONS["skv" if s["ours"] == "emp" else "emp"],
                                        "units": [theirs]}}}
    sc, fc = (s["w"] / 2 - s["kx"]) * DH.CELL, (s["h"] / 2 - s["kz"]) * DH.CELL
    x, z = Z.world(torch.tensor(sc), torch.tensor(fc), 0.0, 0.0, s["b"])
    return arm, (float(x), float(z), s["b"], s["w"] * DH.CELL, s["h"] * DH.CELL, 0.0, 0.0), None, (ex, ez)


def area_points(pos, n=GRID):
    """(x, z) numpy [n * n] of the area's sample points (its box, edges included), local."""
    x, z, b, front, depth = pos[:5]
    a = np.linspace(-0.5, 0.5, n)
    S, F = np.meshgrid(a * front, a * depth)
    t = math.radians(b)
    return x + S.ravel() * math.cos(t) + F.ravel() * math.sin(t), z - S.ravel() * math.sin(t) + F.ravel() * math.cos(t)


def place(s, pos, foe_xz, gen, m=None):
    """(px, pz, deg, relief m) of one skirmish on the map m (realmap's entry; None: the first active one): a point
    whose ground over the area rises RELIEF_M or more, turned so that the enemy's start lies uphill / downhill / across
    the slope of the area (s['frm']); our unit's and the enemy's start on free nodes, 90 % of the area too
    (realmap.free_at: not 'cannot stand')."""
    a = m or realmap._ACTIVE
    ax, az = area_points(pos)
    cx, cz = float(ax.mean()), float(az.mean())
    beta = math.degrees(math.atan2(foe_xz[0] - cx, foe_xz[1] - cz))           # the enemy's start seen from the area
    R = max(math.hypot(foe_xz[0] - cx, foe_xz[1] - cz), math.hypot(pos[3], pos[4]) / 2) + MARGIN_M
    span_x = max(0.0, (a["hi_x"] - a["lo_x"]) / 2 - R - EDGE_M)
    span_z = max(0.0, (a["hi_z"] - a["lo_z"]) / 2 - R - EDGE_M)
    mid_x, mid_z = (a["lo_x"] + a["hi_x"]) / 2, (a["lo_z"] + a["hi_z"]) / 2
    best = None
    for k in range(TRIES * MORE):
        wx = mid_x + (float(torch.rand((), generator=gen)) * 2 - 1) * span_x
        wz = mid_z + (float(torch.rand((), generator=gen)) * 2 - 1) * span_z
        # the slope about the point (a plane through the area's ground, unturned): uphill's bearing
        dx, dz = ax - cx, az - cz
        h = realmap.height_at(wx + dx, wz + dz, a).astype(float)
        A = np.stack([dx, dz, np.ones_like(dx)], 1)
        gx, gz, _ = np.linalg.lstsq(A, h, rcond=None)[0]
        if math.hypot(gx, gz) < 1e-4:
            continue
        up = math.degrees(math.atan2(gx, gz))
        want = {"up": up, "down": up + 180.0, "side": up + 90.0 * s["side"]}[s["frm"]]
        deg = (want - beta) % 360.0
        t = math.radians(deg)
        px = wx - (cx * math.cos(t) + cz * math.sin(t))
        pz = wz - (-cx * math.sin(t) + cz * math.cos(t))
        wxa, wza = realmap.to_world(ax, az, px, pz, deg)
        hh = realmap.height_at(wxa, wza, a)
        relief = float(hh.max() - hh.min())
        ux, uz = realmap.to_world(np.array([0.0, foe_xz[0]]), np.array([0.0, foe_xz[1]]), px, pz, deg)
        ok = bool(realmap.free_at(ux, uz, a).all()) and bool(realmap.free_at(wxa, wza, a).mean() > 0.9)
        if ok and (best is None or relief > best[3]):
            best = (px, pz, deg, relief)
        if ok and relief >= RELIEF_M:
            break
        if k >= TRIES - 1 and best is not None:
            break                   # (past TRIES: only while no try is free - a map with wide closed parts)
    return best if best is not None else (mid_x, mid_z, 0.0, 0.0)


def build(sp, P, device, gen=None):
    """The state (ours slot 0, the enemy slot H) laid on the active maps at the drill's places (each skirmish's map:
    realmap.pick - one map: it; several: in turn in the exam, random in training), own [B, N], our area
    {x, z, b, front, depth, on, sb, sw} [B, N] (sw 0: infantry), the enemies' goal {x, z, b, front, depth} [B, N] (the
    shooters' position) and the places {px, pz, deg, relief} (numpy [B])."""
    assert realmap.active(), "drill_height runs on a real map only (--map)"
    loc = [_local(s) for s in sp]
    st = scenario.build([x[0] for x in loc], P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {k: col([x[1][i] for x in loc]) for i, k in enumerate(("x", "z", "b", "front", "depth", "sb", "sw"))}
    area["on"] = own.clone()
    gl = {"x": col([(x[2] or x[1])[0] for x in loc]), "z": col([(x[2] or x[1])[1] for x in loc]),
          "b": col([0.0] * B), "front": col([DS.GOAL_M] * B), "depth": col([DS.GOAL_M] * B)}
    gen = gen if gen is not None and not realmap._ACTIVE.get("exam") else torch.Generator().manual_seed(realmap.EXAM_SEED)
    which = realmap.pick(B, gen)
    ms = realmap.maps() or [realmap._ACTIVE]
    pl = np.array([place(s, x[1], x[3], gen, ms[int(which[i])]) for i, (s, x) in enumerate(zip(sp, loc))])
    realmap.lay(st, which, pl[:, 0], pl[:, 1], pl[:, 2])
    places = {"px": pl[:, 0], "pz": pl[:, 1], "deg": pl[:, 2], "relief": pl[:, 3], "map": which}
    return st, own, area, gl, places


def script(st, goal, shoot):
    """The enemy's orders: at a shooter's position (drill_go.to_area, shoot [B] the shooters' skirmishes), else attack
    the nearest (drill_hold's)."""
    a = opponents.nearest(st)
    b = DG.to_area(st, goal, grab_m=DS.GRAB_M)
    from tools.nn.sim import orders as O
    m = shoot[:, None].expand_as(a.kind)
    return O.merge(a, b, m)
