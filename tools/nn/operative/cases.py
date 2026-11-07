"""The operative's first cases (plan.md "Случаи"; config/nn/operative.json cases). All on a flat field, the whole layout
turned at random and shifted up to +-spread_m; the units of the tactician's first drill (its roster); the enemy a script
of the simulator. My groups in every case are 2 infantry + 1 lobbing shooters - the tactician's drill.

    gather  my three scattered on a circle (80-140 m from their centre, 120 deg +- 30 apart); three enemy infantry in a
            line 380-440 m ahead walk at the nearest. Right: one group of all, the zone on my centre, facing them
    split   two clumps 2 + 1 (the drill's start) 700-900 m apart across; each its own three enemy 300-380 m ahead,
            walking at the nearest. Right: two groups = the clumps, each its own enemy, the zone where it stands
    go      one clump 2 + 1; two enemy infantry + one shooters standing 350-420 m ahead (they hold and shoot). The
            command 'destroy' (half the battles): one group, the zone stepping to the enemy until the contact; 'hold':
            the zone where it stands

gather_a, split_a, go_a   the same three with armies (config cases.army, as the strategist's cases): each side a lord of
            its faction (listed first: the scenario's general) + roster units, 3-8 in all; my shooters 0 to half of the
            rest; the enemy's in the go case at least one. A lord is a melee unit of its group (as everywhere: range 0).
            gather_a: my units on the circle 360 / n apart (+-15 deg), the enemy army in rows (infantry, shooters 40 m
            behind, the lord 40 m behind them) walking at the nearest; split_a: my army in two clumps (the lord in the
            first, ceil(n / 2) and the rest), the enemy's likewise, one before each clump; go_a: my army in one clump, the
            enemy standing as the strategist's (shooters in front, infantry 40 m behind, the lord 40 m behind them).
            A clump of an army: its infantry in rows of up to 4 across (70 m apart, rows 50 m), its shooters a row 60 m
            behind - or in front (the drill's two starts)

specs(n, gen, fixed) -> dicts; build(sp, P, device) -> Setup-like tensors (the state, my / the enemy's masks, the
shooters, the clumps (the readings' groups: which of my units started together), the command, the limits).
"""
import math

import torch

from tools.nn.operative import cfg
from tools.nn.sim import scenario
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import drill1 as DR
from tools.nn.tactic.decide import order_vec
from tools.nn.unit import drill_hold as DH

CASES = ("gather", "split", "go", "gather_a", "split_a", "go_a")
OLD = CASES[:3]
ARMY = ("split_a", "go_a")   # (gather_a is built but out of the exam and the training: its measure - within 150 m of the
                             # first contact - is a group of 3's; an army's line is wider. build/diag_operative/army_check.md)
H = 8                      # slots a side (an army of 8)


def base(case):
    """'gather' of 'gather' and 'gather_a'."""
    return case.split("_")[0]


def _army_keys(gen, roster, lord, n, shooters):
    """{inf: [lord, infantry...], sh: [shooters...]}: n units in all (as strategist/cases._army)."""
    ri = lambda m: int(torch.randint(m, (1,), generator=gen))                    # noqa: E731
    pick = lambda names: names[ri(len(names))]                                   # noqa: E731
    return {"inf": [lord] + [pick(roster["infantry"]) for _ in range(n - 1 - shooters)],
            "sh": [pick(roster["shooters"]) for _ in range(shooters)]}


def _halves(a):
    """Two clumps of an army {inf, sh}: the lord and ceil(n / 2) - 1 more in the first, alternating."""
    units = [(k, False) for k in a["inf"]] + [(k, True) for k in a["sh"]]
    rest = units[1:]
    first = [units[0]] + rest[1::2]
    second = rest[0::2]
    n1 = (len(units) + 1) // 2
    while len(first) > n1:
        second.append(first.pop())
    while len(first) < n1:
        first.append(second.pop())

    def pack(us):
        return {"inf": [k for k, sh in us if not sh], "sh": [k for k, sh in us if sh]}
    return pack(first), pack(second)


def specs(n, gen, fixed=None, cases=CASES):
    """n random battles; fixed: cycle the cases (and the go's commands) in order (the exam)."""
    out = []
    ca = cfg()["cases"]["army"]
    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))         # noqa: E731
    ri = lambda m: int(torch.randint(m, (1,), generator=gen))                    # noqa: E731
    roster = tcfg()["drill1"]["roster"]
    sp = float(cfg()["cases"]["spread_m"])
    for k in range(n):
        case = cases[k % len(cases)] if fixed else cases[ri(len(cases))]
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        pick = lambda names: names[ri(len(names))]                               # noqa: E731
        clump = lambda: {"infantry": [pick(roster[ours]["infantry"]), pick(roster[ours]["infantry"])],     # noqa: E731
                         "shooters": pick(roster[ours]["shooters"]), "start": DR.STARTS[ri(2)],
                         "shooter_slot": ri(3)}
        s = {"case": case, "ours": ours, "theirs": theirs, "rot": u(0.0, 360.0), "sx": u(-sp, sp), "sz": u(-sp, sp)}
        if case in ARMY:
            lo, hi = (int(v) for v in ca["units"])
            nm, ne = lo + ri(hi - lo + 1), lo + ri(hi - lo + 1)
            mine = _army_keys(gen, roster[ours], ca["lords"][ours], nm, ri((nm - 1) // 2 + 1))
            foe_sh = 1 + ri(max((ne - 1) // 2, 1)) if case == "go_a" else ri((ne - 1) // 2 + 1)
            foes = _army_keys(gen, roster[theirs], ca["lords"][theirs], ne, foe_sh)
            s["army"] = True
            s["start"] = [DR.STARTS[ri(2)], DR.STARTS[ri(2)]]
            c = cfg()["cases"][base(case)]
            if case == "gather_a":
                s["clumps"] = [mine]
                s["a0"] = u(0.0, 360.0)
                s["r"] = [u(*c["radius_m"]) for _ in range(nm)]
                s["da"] = [u(-15.0, 15.0) for _ in range(nm)]
                s["face"] = [u(-45.0, 45.0) for _ in range(nm)]
                s["dist"] = [u(*c["dist_m"])]
                s["foes"] = [foes]
                s["command"] = "hold"
            elif case == "split_a":
                s["clumps"] = list(_halves(mine))
                s["apart"] = u(*c["apart_m"])
                s["dist"] = [u(*c["dist_m"]), u(*c["dist_m"])]
                s["foes"] = list(_halves(foes))
                s["command"] = "hold"
            else:
                s["clumps"] = [mine]
                s["dist"] = [u(*c["dist_m"])]
                s["foes"] = [foes]
                if fixed:
                    s["command"] = "destroy" if (k // len(cases)) % 2 == 0 else "hold"
                else:
                    s["command"] = "destroy" if float(torch.rand(1, generator=gen)) < 0.5 else "hold"
            out.append(s)
            continue
        if case == "gather":
            c = cfg()["cases"]["gather"]
            s["clumps"] = [clump()]
            s["a0"] = u(0.0, 360.0)
            s["r"] = [u(*c["radius_m"]) for _ in range(3)]
            s["da"] = [u(-30.0, 30.0) for _ in range(3)]
            s["face"] = [u(-45.0, 45.0) for _ in range(3)]
            s["dist"] = [u(*c["dist_m"])]
            s["foes"] = [[pick(roster[theirs]["infantry"]) for _ in range(3)]]
            s["command"] = "hold"
        elif case == "split":
            c = cfg()["cases"]["split"]
            s["clumps"] = [clump(), clump()]
            s["apart"] = u(*c["apart_m"])
            s["dist"] = [u(*c["dist_m"]), u(*c["dist_m"])]
            s["foes"] = [[pick(roster[theirs]["infantry"]) for _ in range(3)] for _ in range(2)]
            s["command"] = "hold"
        else:
            c = cfg()["cases"]["go"]
            s["clumps"] = [clump()]
            s["dist"] = [u(*c["dist_m"])]
            s["foes"] = [[pick(roster[theirs]["infantry"]), pick(roster[theirs]["infantry"]),
                          pick(roster[theirs]["shooters"])]]
            s["command"] = ("destroy" if (k // len(cases)) % 2 == 0 else "hold") if fixed else \
                ("destroy" if float(torch.rand(1, generator=gen)) < 0.5 else "hold")
        out.append(s)
    return out


def exam_specs(n=None, seed=None, cases=OLD):
    """The fixed exam: the old cases (OLD) or the armies' (ARMY: config exam.army_seed)."""
    c = cfg()["exam"]
    if seed is None:
        seed = c["army_seed"] if tuple(cases) == ARMY else c["seed"]
    return specs(int(c["battles"] if n is None else n), torch.Generator().manual_seed(int(seed)), fixed=True,
                 cases=tuple(cases))


def _block(a, start, x0, z0):
    """[(key, x, z, 0)] of an army's clump {inf, sh} facing +z (module doc): infantry in rows of up to 4 (70 m), the
    shooters a row 60 m behind (start 'shooters_front': 60 m in front)."""
    gap, row = 70.0, 50.0
    inf, sh = a["inf"], a["sh"]
    front = start == "shooters_front" and bool(sh)
    out = []
    for i, key in enumerate(inf):
        r, c = divmod(i, 4)
        n_row = min(4, len(inf) - 4 * r)
        z = (-30.0 - row * r) if front else (30.0 - row * r)
        out.append((key, x0 + (c - (n_row - 1) / 2) * gap, z0 + z, 0.0))
    zs = 30.0 if front else (30.0 - row * ((len(inf) - 1) // 4) - 60.0)
    for i, key in enumerate(sh):
        out.append((key, x0 + (i - (len(sh) - 1) / 2) * gap, z0 + zs, 0.0))
    return out


def _foe_rows(a, x0, z0, standing):
    """[(key, x, z, 180)] of an enemy army {inf, sh} ahead at z0 facing back, the lord first: standing (the go case)
    shooters in front, infantry 40 m behind, the lord 40 m behind them (strategist/cases.foes_local); walking:
    infantry in front, the shooters 40 m behind, the lord 40 m behind them. A clump without infantry (the lord in the
    other half): its shooters in one row."""
    gap, back = 70.0, 40.0
    inf, sh = a["inf"], a["sh"]
    out = []
    lord = inf[0] if inf else None
    others = inf[1:] if inf else []
    rows = ((sh, 0.0), (others, back)) if standing else ((others, 0.0), (sh, back))
    rows = [r for r in rows if r[0]]
    if lord is not None:
        out.append((lord, x0, z0 + (max(dz for _, dz in rows) + back if rows else 0.0), 180.0))
    for keys, dz in rows:
        for i, key in enumerate(keys):
            out.append((key, x0 + (i - (len(keys) - 1) / 2) * gap, z0 + dz, 180.0))
    return out


def _clump_units(cl, x0, z0, b0):
    """[(key, x, z, b)] of a clump laid out as the drill's start (drill1.layout) at x0, z0 facing b0 (local frame)."""
    out = []
    for key, r, f in DR.layout(cl):
        a = math.radians(b0)
        out.append((key, x0 + r * math.cos(a) + f * math.sin(a), z0 - r * math.sin(a) + f * math.cos(a), b0))
    return out


def local_layout(s):
    """(mine [(key, x, z, b)], foes [(key, x, z, b)], clump of each of mine [int], foe group of each foe) in the case's
    own frame (my centre at the origin, the enemy ahead: +z)."""
    gap = float(cfg()["cases"]["gather"]["foe_gap_m"])
    mine, clumps, foes, fg = [], [], [], []
    if s.get("army"):
        if s["case"] == "gather_a":
            a = s["clumps"][0]
            keys = a["inf"] + a["sh"]
            n = len(keys)
            for i, key in enumerate(keys):
                ang = math.radians(s["a0"] + 360.0 / n * i + s["da"][i])
                mine.append((key, s["r"][i] * math.sin(ang), s["r"][i] * math.cos(ang), s["face"][i] % 360))
                clumps.append(0)
            for f in _foe_rows(s["foes"][0], 0.0, s["dist"][0], False):
                foes.append(f)
                fg.append(0)
        elif s["case"] == "split_a":
            for c, a in enumerate(s["clumps"]):
                x0 = (c - 0.5) * s["apart"]
                for u_ in _block(a, s["start"][c], x0, 0.0):
                    mine.append(u_)
                    clumps.append(c)
            for c, a in enumerate(s["foes"]):
                x0 = (c - 0.5) * s["apart"]
                for f in _foe_rows(a, x0, s["dist"][c], False):
                    foes.append(f)
                    fg.append(c)
        else:
            for u_ in _block(s["clumps"][0], s["start"][0], 0.0, 0.0):
                mine.append(u_)
                clumps.append(0)
            for f in _foe_rows(s["foes"][0], 0.0, s["dist"][0], True):
                foes.append(f)
                fg.append(0)
        return mine, foes, clumps, fg
    if s["case"] == "gather":
        cl = s["clumps"][0]
        keys = [k for k, _, _ in DR.layout(cl)]
        for i, key in enumerate(keys):
            a = math.radians(s["a0"] + 120.0 * i + s["da"][i])
            mine.append((key, s["r"][i] * math.sin(a), s["r"][i] * math.cos(a), s["face"][i] % 360))
            clumps.append(0)
        for i, key in enumerate(s["foes"][0]):
            foes.append((key, (i - 1) * gap, s["dist"][0], 180.0))
            fg.append(0)
    elif s["case"] == "split":
        for c, cl in enumerate(s["clumps"]):
            x0 = (c - 0.5) * s["apart"]
            for key, x, z, b in _clump_units(cl, x0, 0.0, 0.0):
                mine.append((key, x, z, b))
                clumps.append(c)
            for i, key in enumerate(s["foes"][c]):
                foes.append((key, x0 + (i - 1) * gap, s["dist"][c], 180.0))
                fg.append(c)
    else:
        for key, x, z, b in _clump_units(s["clumps"][0], 0.0, 0.0, 0.0):
            mine.append((key, x, z, b))
            clumps.append(0)
        d = s["dist"][0]
        inf1, inf2, sh = s["foes"][0]
        foes += [(inf1, -gap / 2, d, 180.0), (inf2, gap / 2, d, 180.0), (sh, 0.0, d + 40.0, 180.0)]
        fg += [0, 0, 0]
    return mine, foes, clumps, fg


def _turn(x, z, b, s):
    a = math.radians(s["rot"])
    return (x * math.cos(a) + z * math.sin(a) + s["sx"], -x * math.sin(a) + z * math.cos(a) + s["sz"],
            (b + s["rot"]) % 360)


def army(s):
    mine, foes, _, _ = local_layout(s)
    side = lambda units: [{"key": k, **dict(zip(("x", "z", "b"), _turn(x, z, b, s)))} for k, x, z, b in units]  # noqa
    s1, s2 = side(mine), side(foes)
    if s.get("army"):                         # the lords: listed first, the scenario's generals (their auras, their slot)
        s1[0]["general"] = s2[0]["general"] = True
    return {"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": s1},
                                     2: {"faction": DH.FACTIONS[s["theirs"]], "units": s2}}}


class Setup:
    """A batch of case battles: st, own, shooter [B, N] bool, clump [B, N] long (-1 not mine), case [B] long,
    command [B, 7], destroy [B] bool (the command), stands [B] bool (the enemy holds), limit [B], side [B]."""

    def __init__(self, sp, P, device):
        c = cfg()["cases"]
        self.sp, self.P, self.device = sp, P, device
        self.st = scenario.build([army(s) for s in sp], P, device=device, per_side=H)
        B, N = self.st.B, self.st.N
        self.own = torch.zeros((B, N), dtype=torch.bool, device=device)
        self.clump = torch.full((B, N), -1, dtype=torch.long, device=device)
        for b, s in enumerate(sp):
            _, _, clumps, _ = local_layout(s)
            for i, k in enumerate(clumps):
                self.own[b, i] = True
                self.clump[b, i] = k
        self.shooter = (self.st.u["range"] > 0) & self.own
        self.case = torch.tensor([CASES.index(s["case"]) for s in sp], device=device)
        self.destroy = torch.tensor([s["command"] == "destroy" for s in sp], device=device)
        self.command = torch.cat([order_vec(1, s["command"], "any", device) for s in sp])
        self.stands = torch.tensor([base(s["case"]) == "go" for s in sp], device=device)
        self.limit = torch.tensor([float(c["go"]["hold_limit_s"]) if base(s["case"]) == "go" and s["command"] == "hold"
                                   else float(c[base(s["case"])]["limit_s"]) for s in sp], device=device)
        self.side = torch.ones(B, dtype=torch.long, device=device)
        from tools.nn.unit import train as UT
        self.card = UT.cards(self.st, device).float()


class Script:
    """The enemy: drill1.Script (at the nearest, at a walk, looked at again every 5 s); in the go case it holds."""

    def __init__(self, stands):
        self.inner = DR.Script()
        self.stands = stands

    def __call__(self, st):
        from tools.nn.sim import orders as O
        o = self.inner(st)
        if not bool(self.stands.any()):
            return o
        h = O.hold(st.B, st.N, st.device)
        return O.merge(o, h, self.stands[:, None].expand(st.B, st.N))
