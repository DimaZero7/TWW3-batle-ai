"""The strategist's cases (plan.md "Случаи"; config/nn/strategist.json cases). A flat field; two deployment zones (turned
rectangles zone_m, their front edges gap_m apart); the whole layout turned by k x 90 deg +- turn_jitter_deg and shifted
up to +-shift_m; the units of the tactician's first drill; the enemy set by the case (the strategist never sees where)
and played by a script of the simulator.

    wide  mine 2 x (2 infantry + 1 shooters); the enemy two threes of infantry at its zone's front edge, +-(340-460) m
          aside, walking at the nearest. The task 'hold'. Right: two groups, one facing each three
    attack, hold   random armies (config cases.army): each side a lord (its faction's, config lords) + 2-7 units of the
          roster, my shooters 0 to half, the enemy's at least one; the enemy's shooters in a line at its front edge (up
          to +-50 m aside), its infantry 40 m behind, its lord 40 m behind them; they stand and shoot. The task 'attack'
          / 'hold'. Right: 'attack' - the command 'destroy' (from the front edge); 'hold' - the command 'hold' at the back
          edge, out of reach. arena=True: both armies the arena's (config/nn/arena.json: Empire, a lord, 4 spearmen, 2
          archers), the zones the same as the cases' (the arena's deploy_zone)
    wide  (no lord; out of the training and the exam: the blob does as well - build/diag_strategist/formula_check.md)

A lord is listed first (the scenario puts a side's general in its first slot).

specs(n, gen, fixed) -> dicts; Pre(sp, P, device): the state before the deployment (my units anywhere: the input is
blind to it), the zones, the task, the enemy's threes; army(s, mine) -> the scenario description with my places.
"""
import math

import torch

from tools.nn.sim import scenario
from tools.nn.strategist import cfg
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import drill1 as DR
from tools.nn.tactic import raw as TR
from tools.nn.tactic.decide import order_vec
from tools.nn.unit import drill_hold as DH

CASES = ("wide", "attack", "hold")
H = 8


def _arena_side():
    import json
    from tools import config as project
    a = json.loads((project.ROOT / "config" / "nn" / "arena.json").read_text(encoding="utf-8"))
    units = sorted(a["units"], key=lambda u: not u.get("general"))
    return [u["key"] for u in units]


def _army(gen, keys, lord, n, shooters):
    """[lord, infantry..., shooters...]: n units in all, shooters of them (keys: the roster of a faction)."""
    ri = lambda m: int(torch.randint(m, (1,), generator=gen))                    # noqa: E731
    pick = lambda names: names[ri(len(names))]                                   # noqa: E731
    return [lord] + [pick(keys["infantry"]) for _ in range(n - 1 - shooters)] +         [pick(keys["shooters"]) for _ in range(shooters)]


def specs(n, gen, fixed=None, cases=None, arena=False):
    cases = tuple(cases or cfg()["train"]["cases"])
    out = []
    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))         # noqa: E731
    ri = lambda m: int(torch.randint(m, (1,), generator=gen))                    # noqa: E731
    roster = tcfg()["drill1"]["roster"]
    c = cfg()["cases"]
    j, sp = float(c["turn_jitter_deg"]), float(c["shift_m"])
    for k in range(n):
        case = cases[k % len(cases)] if fixed else cases[ri(len(cases))]
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        pick = lambda names: names[ri(len(names))]                               # noqa: E731
        clump = lambda: [pick(roster[ours]["infantry"]), pick(roster[ours]["infantry"]),        # noqa: E731
                         pick(roster[ours]["shooters"])]
        s = {"case": case, "ours": ours, "theirs": theirs, "rot": 90.0 * ri(4) + u(-j, j), "sx": u(-sp, sp),
             "sz": u(-sp, sp)}
        if case == "wide":
            s["mine"] = clump() + clump()
            s["side"] = [u(*c["wide"]["side_m"]), u(*c["wide"]["side_m"])]
            s["foes"] = [pick(roster[theirs]["infantry"]) for _ in range(6)]
            s["task"] = "hold"
        else:                                                                   # attack / hold: the same field
            s["side"] = [u(-float(c["go"]["side_m"]), float(c["go"]["side_m"]))]
            s["task"] = "destroy" if case == "attack" else "hold"
            if arena:
                s["ours"] = s["theirs"] = "emp"
                s["mine"] = _arena_side()
                s["foes"] = _arena_side()
            else:
                lo, hi = (int(v) for v in c["army"]["units"])
                lords = c["army"]["lords"]
                nm, ne = lo + ri(hi - lo + 1), lo + ri(hi - lo + 1)
                s["mine"] = _army(gen, roster[ours], lords[ours], nm, ri((nm - 1) // 2 + 1))
                s["foes"] = _army(gen, roster[theirs], lords[theirs], ne, 1 + ri(max((ne - 1) // 2, 1)))
        out.append(s)
    return out


def exam_specs(n=None, seed=None, arena=False):
    c = cfg()["exam"]
    return specs(int(c["battles"] if n is None else n),
                 torch.Generator().manual_seed(int(c["seed"] if seed is None else seed)), fixed=True,
                 cases=tuple(c["cases"]), arena=arena)


def zones_local():
    """((x, z, b) mine, (x, z, b) the enemy's) in the case's frame: my zone below facing +z."""
    c = cfg()["cases"]
    W_, D = c["zone_m"]
    y = float(c["gap_m"]) / 2 + float(D) / 2
    return (0.0, -y, 0.0), (0.0, y, 180.0)


def foes_local(s):
    """[(key, x, z, b)], [three index] of the enemy in the case's frame."""
    c = cfg()["cases"]
    z0 = float(c["gap_m"]) / 2
    out, three = [], []
    if s["case"] == "wide":
        w = c["wide"]
        z = z0 + float(w["edge_m"])
        gap = float(w["foe_gap_m"])
        for t, sign in enumerate((-1.0, 1.0)):
            cx = sign * s["side"][t]
            for i in range(3):
                out.append((s["foes"][3 * t + i], cx + (i - 1) * gap, z, 180.0))
                three.append(t)
    else:
        from tools.nn.sim.params import load
        g = c["go"]
        z = z0 + float(g["edge_m"])
        x0 = s["side"][0]
        gap = float(tcfg()["drill1"]["foe_gap_m"])
        P = load()
        fac = DH.FACTIONS[s["theirs"]]
        lord, rest = s["foes"][0], s["foes"][1:]
        sh = [k for k in rest if float(P.static(k, fac)["range"]) > 0]
        inf = [k for k in rest if float(P.static(k, fac)["range"]) <= 0]
        back = float(g["behind_m"])
        out.append((lord, x0, z + (2 * back if inf else back), 180.0))
        for row, (keys, dz) in enumerate(((sh, 0.0), (inf, back))):
            for i, key in enumerate(keys):
                out.append((key, x0 + (i - (len(keys) - 1) / 2) * gap, z + dz, 180.0))
        three += [0] * len(out)
    return out, three


def turn(x, z, b, s):
    a = math.radians(s["rot"])
    return (x * math.cos(a) + z * math.sin(a) + s["sx"], -x * math.sin(a) + z * math.cos(a) + s["sz"],
            (b + s["rot"]) % 360)


def army(s, mine=None):
    """The scenario description; mine: [{x, z, b, width}] my units' places (None: a row across my zone's centre)."""
    (mx, mz, mb), _ = zones_local()
    if mine is None:
        k = len(s["mine"])
        mine = []
        for i in range(k):
            x, z, b = turn(mx + (i - (k - 1) / 2) * 80.0, mz, mb, s)
            mine.append({"x": x, "z": z, "b": b})
    foes, _ = foes_local(s)
    side2 = [{"key": key, **dict(zip(("x", "z", "b"), turn(x, z, b, s)))} for key, x, z, b in foes]
    side1 = [{"key": key, **p} for key, p in zip(s["mine"], mine)]
    if s["case"] != "wide":                     # the lord: listed first, the scenario's general (its auras, its slot)
        side1[0]["general"] = side2[0]["general"] = True
    return {"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": side1},
                                     2: {"faction": DH.FACTIONS[s["theirs"]], "units": side2}}}


def zone_tensors(sp, device):
    c = cfg()["cases"]
    W_, D = (float(v) for v in c["zone_m"])
    out = []
    for which in (0, 1):
        rows = []
        for s in sp:
            x, z, b = turn(*zones_local()[which], s)
            rows.append((x, z, b))
        t = torch.tensor(rows, device=device)
        out.append({"x": t[:, 0], "z": t[:, 1], "b": t[:, 2], "front": torch.full((len(sp),), W_, device=device),
                    "depth": torch.full((len(sp),), D, device=device)})
    return out


class Pre:
    """A batch before the deployment: st0 (my units in a row across my zone's centre), raw0, card, own, shooter [B, N],
    three [B, N] long (the enemy's threes, -1 else), mine / enemy zones, command [B, 7] the task, destroy [B] bool the
    task 'attack', case [B], limit [B], side [B]."""

    def __init__(self, sp, P, device):
        from tools.nn.unit import train as UT
        c = cfg()["cases"]
        self.sp, self.P, self.device = sp, P, device
        self.st0 = scenario.build([army(s) for s in sp], P, device=device, per_side=H)
        B, N = self.st0.B, self.st0.N
        self.raw0 = TR.Seer(P).raw(self.st0)
        self.card = UT.cards(self.st0, device).float()
        self.own = (self.st0.u["side"] == 1)
        self.shooter = (self.st0.u["range"] > 0) & self.own
        self.three = torch.full((B, N), -1, dtype=torch.long, device=device)
        for b, s in enumerate(sp):
            _, three = foes_local(s)
            for i, t in enumerate(three):
                self.three[b, H + i] = t
        self.mine_zone, self.enemy_zone = zone_tensors(sp, device)
        self.command = torch.cat([order_vec(1, s["task"], "any", device) for s in sp])
        self.destroy = torch.tensor([s["task"] == "destroy" for s in sp], device=device)
        self.case = torch.tensor([CASES.index(s["case"]) for s in sp], device=device)
        self.limit = torch.tensor([float(c["go"]["hold_limit_s"]) if s["case"] == "hold" else
                                   float(c["wide" if s["case"] == "wide" else "go"]["limit_s"]) for s in sp],
                                  device=device)
        self.stands = self.case != CASES.index("wide")
        self.side = torch.ones(B, dtype=torch.long, device=device)
        self.width0 = self.st0.u["width"].clone()


class Script:
    """The enemy: drill1.Script (at the nearest, at a walk); in the attack / hold cases it holds (and shoots)."""

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
