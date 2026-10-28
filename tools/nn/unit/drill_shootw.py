"""Drill U5c 'the shooter's width' (build/diag_g49/unit_base.md, 'Ширина у стрелков', the user 28.10): for a shooter a
change of width pays - narrow into a gap between our own units, so that they do not close its direct fire; wide in the
open, so that more of its men reach the target. One of our shooters under the network (slot 0, its position + fire
sector as U5), our melee infantry (slots 1, 2, under no order: they hold where they stand), 1-2 enemies.

Layouts (half the gap layouts each of 'gap' and 'angle' in the exam, a third each):
    gap     two of our infantry units side by side ahead of the shooter's position (their rear edges FRIEND_AHEAD_M
            beyond its front edge), a gap of GAP_M between their inner edges straight ahead of it; the enemies come
            from ahead (the sector within SECTOR_OFF of its front)
    angle   the same, the enemies from the side: the sector's middle ANGLE_DEG off the front (left or right) - the gap
            seen at a slant
    open    no friends, the position OPEN_FRONT_M wide; the enemies from ahead
The shooter: the missile infantry of drill_shoot.shooters(), its start's width a random START_WIDTH x its default
frontage (within the simulator's limits); its position: the gap's width + 2 x MARGIN_M (at least POS_MIN_M) wide, a
gap layout, 3-5 cells deep - wide enough for the network to choose a line wider than the gap.
The enemies (1-2): melee infantry going at our position (drill_go.to_area, its goal the position, at a run) from
START_M in the sector, or a shooter standing at SHOOTER_RANGE x its range from the position in the sector (it fires at
will at what is in its range: a hold).
The simulator's line of fire past friends (missile.friend_los): the men of another own unit in the way block a
shooter's line (direct fire holds, arcing fire lobs at 45 deg over friends beyond 10 m; a man whose path passes below
a friend's head loses his shot: u fc_cap / fc_skip). NOTE: the men of the shooter's own unit do not block its fire in
the simulator (the rule's text: 'the men of i's own unit never block, but in melee') - only its ranks' range does
(missile.rank_share, direct fire at the edge of its range).
The end (drill_multi.Ender): our shooter routs or dies - over; every enemy dead, gone or routing out of our zone -
over; at most LIMIT_S.
"""
import math

import torch

from tools.nn.sim import orders as O
from tools.nn.sim import scenario
from tools.nn.unit import drill_go as DG
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import drill_shoot as DS
from tools.nn.unit import zmap as Z

LAYOUTS = ("gap", "angle", "open")
GAP_M = (10.0, 45.0)
MARGIN_M = 20.0
POS_MIN_M = 48.0
OPEN_FRONT_M = (50.0, 80.0)
POS_DEPTH_CELLS = (3, 5)
FRIEND_AHEAD_M = (3.0, 12.0)
ANGLE_DEG = (30.0, 55.0)
SECTOR_OFF = 10.0
SECTOR_W = (50.0, 90.0)
START_WIDTH = (0.5, 1.6)
START_M = (150.0, 280.0)
SHOOTER_RANGE = (0.75, 0.95)
SHOOTER_SHARE = 0.4
LIMIT_S = DH.LIMIT_S
C = DH.CELL


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
        layout = fixed[k % len(fixed)] if fixed else LAYOUTS[ri(len(LAYOUTS))]
        me = S[ours][ri(len(S[ours]))]
        front, depth = DH.box(pp[me], u(*START_WIDTH))
        gap = u(*GAP_M)
        friends = []
        if layout != "open":
            for _ in range(2):
                key = R[ours][ri(len(R[ours]))]
                ff, fd = DH.box(pp[key], u(*DH.WIDTH))
                friends.append({"key": key, "front": ff, "depth": fd})
        pos_front = u(*OPEN_FRONT_M) if layout == "open" else max(POS_MIN_M, gap + 2 * MARGIN_M)
        b = u(0.0, 360.0)
        off = (1 if float(torch.rand(1, generator=gen)) < 0.5 else -1) * u(*ANGLE_DEG) if layout == "angle" \
            else u(-SECTOR_OFF, SECTOR_OFF)
        sw = u(*SECTOR_W)
        foes = []
        for j in range(1 + ri(2)):
            shooter = float(torch.rand(1, generator=gen)) < SHOOTER_SHARE
            key = S[theirs][ri(len(S[theirs]))] if shooter else R[theirs][ri(len(R[theirs]))]
            gap_m = u(*SHOOTER_RANGE) * pp[key]["missile"]["range_m"] if shooter else u(*START_M)
            foes.append({"key": key, "shooter": shooter, "rel": u(-0.7, 0.7) * sw / 2, "gap": gap_m})
        out.append({"ours": ours, "theirs": theirs, "layout": layout, "me": me, "front": front, "depth": depth,
                    "gap": gap, "friends": friends, "pos_front": pos_front,
                    "pos_depth": (ri(POS_DEPTH_CELLS[1] - POS_DEPTH_CELLS[0] + 1) + POS_DEPTH_CELLS[0]) * C,
                    "ahead": u(*FRIEND_AHEAD_M), "b": b, "sb": (b + off) % 360, "sw": sw, "foes": foes})
    return out


def exam_specs(seed=4931, n=240):
    """The fixed exam: n skirmishes, the layouts in turn."""
    return specs(n, torch.Generator().manual_seed(seed), fixed=list(LAYOUTS))


def friend_places(s):
    """[(right, forward)] of the two friends' centres in the shooter's frame (its centre at 0): side by side, the gap
    between their inner edges straight ahead, their rear edges `ahead` m beyond the shooter's front edge."""
    out = []
    for side, f in zip((-1, 1), s["friends"]):
        x = side * (s["gap"] / 2 + f["front"] / 2)
        z = s["depth"] / 2 + s["ahead"] + f["depth"] / 2
        out.append((x, z))
    return out


def build(sp, P, device):
    """The state (ours: the shooter slot 0, the friends 1-2; the enemies from H), own [B, N] (the shooter), its position
    {x, z, b, front, depth, on, sb, sw} [B, N] and the enemies' goal {x, z, b, front, depth} [B, N] (the position)."""
    armies = []
    for s in sp:
        mine = [{"key": s["me"], "x": 0.0, "z": 0.0, "b": s["b"], "width": s["front"]}]
        for f, (fx, fz) in zip(s["friends"], friend_places(s)):
            x, z = Z.world(torch.tensor(fx), torch.tensor(fz), 0.0, 0.0, s["b"])
            mine.append({"key": f["key"], "x": float(x), "z": float(z), "b": s["b"], "width": f["front"]})
        foes = []
        for f in s["foes"]:
            a = math.radians(s["sb"] + f["rel"])
            ex, ez = f["gap"] * math.sin(a), f["gap"] * math.cos(a)
            foes.append({"key": f["key"], "x": ex, "z": ez, "b": math.degrees(math.atan2(-ex, -ez)) % 360})
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": mine},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": foes}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda v: torch.tensor(v, device=device, dtype=torch.float32)[:, None].expand(B, N).clone()   # noqa: E731
    area = {"x": col([0.0] * B), "z": col([0.0] * B), "b": col([s["b"] for s in sp]),
            "front": col([s["pos_front"] for s in sp]), "depth": col([s["pos_depth"] for s in sp]),
            "sb": col([s["sb"] for s in sp]), "sw": col([s["sw"] for s in sp])}
    area["on"] = own.clone()
    goal = {"x": col([0.0] * B), "z": col([0.0] * B), "b": col([s["b"] for s in sp]),
            "front": col([s["pos_front"] for s in sp]), "depth": col([s["pos_depth"] for s in sp])}
    return st, own, area, goal


def shooter_slots(sp, st):
    """[B, N] bool: the enemies that are shooters (they stand: script)."""
    out = torch.zeros((st.B, st.N), dtype=torch.bool, device=st.device)
    H = st.N // 2
    for b, s in enumerate(sp):
        for j, f in enumerate(s["foes"]):
            out[b, H + j] = bool(f["shooter"])
    return out


def script(st, goal, stand):
    """The others' orders: the enemy's infantry to our position (drill_go.to_area); the enemy's shooters (stand [B, N])
    and our friends (side 1, not slot 0) hold where they stand (a shooter fires at will)."""
    o = DG.to_area(st, goal, grab_m=DS.GRAB_M)
    u = st.u
    slot = torch.arange(st.N, device=st.device)[None, :]
    hold = stand | ((u["side"] == 1) & (slot > 0))
    o.kind = torch.where(hold, torch.full_like(o.kind, O.HOLD), o.kind)
    o.target = torch.where(hold, torch.full_like(o.target, -1), o.target)
    o.x = torch.where(hold, u["x"], o.x)
    o.z = torch.where(hold, u["z"], o.z)
    o.run = torch.where(hold, torch.zeros_like(o.run), o.run)
    return o
