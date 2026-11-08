"""Drill S-1 "snipers under cover" (build/diag_drills/catalog.md, S-1): the strategist's one decision - the deployment
of a small army on the Kislev plains against an enemy with fast units - and the first 150 s of the battle under the
chain (the operative's formula from the strategist's groups, the tacticians' formula, the frozen unit network), the
enemy a script of the simulator (ai_game, nearest, hold_shoot). The stupidity it guards against: the snipers at the
ends of the first row, ahead of the infantry, where the enemy's fast units reach them first (the chain's 12 game
battles: the snipers rout first in 10).

    specs(n, rnd)          the drill's battles: my army 9-12 units of one matchup's roster (the lord, 1-3 snipers, the
                           line, 0-4 halberds / spears, 0-1 shooters, fast, guns), the enemy's 8-12 of the same
                           matchup's enemy (the lord, 0-4 fast units on one flank or both, infantry and shooters at the
                           game AI's own places); the whole battle mirrored across the front, turned 180 deg (the other
                           zone, the other ground), shifted along the front, my zone moved back; the enemy's script
    full_specs(names, n, rnd)   the whole matchups M1-M3 (matchup/armies.augment) with the strategist deploying my side
    Batch(sp, P, device)   the battles before the deployment: my units in a row across my zone's centre and the enemy's
                           across its zone (the companion's StrategistSeer lays out the deployment document so), the
                           strategist's input (inputs.build - the same as the game's), the zones
    decide(batch, who)     'teacher' (the formula, greedy), 'exposed' (the bad one: the snipers at the front row's ends,
                           15 m ahead), 'deep' (the other bad one: behind the middle, 150 m behind the front row), 'sample' (the teacher's walk drawn round its wishes - the search's candidates;
                           no nudge: what a network can do), a StrategistNet (greedy, or sampled)
    play(batch, out, ...)  the places (deploy.places) -> the battle (matchup/duel.play: the chain from the strategist's
                           groups) -> the readings and the measures:
        sniper_kept   my snipers' men at 150 s / their men at the start, a routed unit 0 (the drill's measure)
        sn_first      the first of my units to rout (or die) is a sniper (by the battle's end)
        fast60_s      the first second an enemy fast unit comes within 60 m of one of my snipers (none: the limit)
        trade         the enemy's share of gold lost less mine
        useful        sniper_kept x min(1, their shots by 150 s / the teacher's on the same battle) - useful(m, ref)
        score         useful + trade (the search's mark; config reward: measure + trade)
"""
import copy
import math
import random

import numpy as np
import torch

from tools.nn.companion import strategist_input as SI
from tools.nn.matchup import armies as AR
from tools.nn.matchup import duel as DU
from tools.nn.model import passport as PP
from tools.nn.operative import army as AM
from tools.nn.operative import groups as GR
from tools.nn.operative import cfg as ocfg
from tools.nn.sim import orders as O
from tools.nn.strategist import army as SA
from tools.nn.strategist import battle as BT
from tools.nn.strategist import cfg
from tools.nn.strategist import deploy as DP
from tools.nn.strategist import inputs as IN
from tools.nn.strategist import net as NN
from tools.nn.strategist import teacher as TC
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import decide as DC
from tools.nn.tactic import raw as TR
from tools.nn.train import opponents

SOURCES = ("m1", "m2", "m3")
DC1 = cfg()["drill_s1"]                    # config/nn/strategist.json drill_s1 (each number with its reason)
ENEMIES = tuple(DC1["enemies"].items())
LIMIT_S = int(DC1["limit_s"])
DEEP_M = 150.0                             # the bad 'deep': the snipers this far behind the front row
PER_SIDE = 12                              # (the drill's armies are at most 12 a side: a smaller state, a faster step)
_ROLE = {}


def role(key):
    if key not in _ROLE:
        _ROLE[key] = AM.roles(np.asarray(PP.table([key])))[0]
    return _ROLE[key]


def _by_role(units):
    out = {}
    for u in units:
        out.setdefault(role(u["key"]), []).append(u)
    return out


def _zones():
    z = AR.zones()
    return ({"x": z[0]["x"], "z": z[0]["z"], "b": z[0]["bearing_deg"], "front": z[0]["width"], "depth": z[0]["depth"]},
            {"x": z[1]["x"], "z": z[1]["z"], "b": z[1]["bearing_deg"], "front": z[1]["width"], "depth": z[1]["depth"]})


def _unit(src, u, **kw):
    return {"key": u["key"], "x": u["x"], "z": u["z"], "b": u["b"], "general": bool(u["general"]),
            "width": u.get("width", 0.0) or None, **kw}


def _transform(sp, mirror, rot, dz, back):
    """The whole battle (the enemy's places, both zones) mirrored across the front (z -> -z), turned 180 deg about the
    map's centre (my zone the east one), shifted dz along the front; my zone moved back m along its facing."""
    def tr(x, z, b):
        if mirror:
            z, b = -z, (180.0 - b) % 360.0
        if rot:
            x, z, b = -x, -z, (b + 180.0) % 360.0
        return x, z + dz, b
    for u in sp["battle"][2]:
        u["x"], u["z"], u["b"] = tr(u["x"], u["z"], u["b"])
    for k in ("mine", "theirs"):
        zz = sp[k]
        zz["x"], zz["z"], zz["b"] = tr(zz["x"], zz["z"], zz["b"])
    m = sp["mine"]
    a = math.radians(m["b"])
    m["x"] -= back * math.sin(a)
    m["z"] -= back * math.cos(a)


def _enemy_script(rnd):
    names, w = zip(*ENEMIES)
    return rnd.choices(names, weights=w)[0]


def spec(rnd, src=None):
    """One drill battle (module doc)."""
    src = src or rnd.choice(SOURCES)
    base = AR.base(src)
    r1, r2 = _by_role(base[1]), _by_role(base[2])
    pick = lambda rs, k: [copy.deepcopy(rnd.choice(rs)) for _ in range(k)]                 # noqa: E731
    total = rnd.randint(*DC1["own_units"])
    n_sn = rnd.randint(*DC1["snipers"])
    n_anti = rnd.choice((0, 2, 2, 4)) if r1.get("anti") else 0
    n_sh = rnd.choice((0, 0, 1)) if r1.get("shooter") else 0
    n_fast = rnd.choice((0, 0, 1)) if r1.get("fast") else 0
    n_art = rnd.choice((0, 0, 1)) if r1.get("art") else 0
    n_line = max(2, total - 1 - n_sn - n_anti - n_sh - n_fast - n_art)
    rest = (pick(r1["line"], n_line) + pick(r1.get("anti", []), n_anti) + pick(r1["sniper"], n_sn)
            + pick(r1.get("shooter", []), n_sh) + pick(r1.get("fast", []), n_fast) + pick(r1.get("art", []), n_art))
    rnd.shuffle(rest)
    mine = [_unit(src, r1["lord"][0])] + [_unit(src, u, general=False) for u in rest]
    # the enemy: its lord, 0-4 fast units on a flank or both, 3-6 infantry and 1-3 shooters at the game AI's places
    n_f = rnd.choices(range(len(DC1["enemy_fast_weights"])), weights=DC1["enemy_fast_weights"])[0]
    flank = rnd.choice(("left", "right", "both"))
    foes = [_unit(src, r2["lord"][0])]
    inf = r2.get("line", []) + r2.get("anti", [])
    n_inf, n_sh2 = rnd.randint(3, 6), rnd.randint(1, 3)
    for k in range(n_inf):
        u = inf[k % len(inf)]
        foes.append(_unit(src, u, general=False, x=u["x"] + 25.0 * (k // len(inf))))
    shs = r2.get("shooter", []) + r2.get("sniper", [])
    for k in range(n_sh2):
        u = shs[k % len(shs)]
        foes.append(_unit(src, u, general=False, x=u["x"] + 25.0 * (k // len(shs))))
    fk = r2["fast"]
    for k in range(n_f):
        u = fk[k % len(fk)]
        sign = {"left": 1.0, "right": -1.0}.get(flank, 1.0 if k % 2 == 0 else -1.0)
        foes.append(_unit(src, u, general=False, x=rnd.uniform(130.0, 230.0), z=sign * rnd.uniform(90.0, 190.0),
                          b=270.0))
    mz, ez = _zones()
    sp = {"src": src, "battle": {1: mine, 2: foes,
                                 "factions": {1: AR.faction(src, 1), 2: AR.faction(src, 2)}},
          "mine": mz, "theirs": ez, "enemy": _enemy_script(rnd), "n_fast": n_f, "flank": flank, "n_sn": n_sn,
          "mirror": rnd.random() < 0.5, "rot": rnd.random() < 0.5,
          "dz": rnd.uniform(-float(DC1["dz_m"]), float(DC1["dz_m"])), "back": rnd.uniform(0.0, float(DC1["back_m"])),
          "kind": "drill"}
    _transform(sp, sp["mirror"], sp["rot"], sp["dz"], sp["back"])
    return sp


def specs(n, rnd):
    return [spec(rnd) for _ in range(n)]


def full_specs(names, n, rnd, enemy="ai_game"):
    """The whole matchups (matchup/armies.augment: the enemy's places jittered, mirror, 0-2 units out a side), the
    game's zones; my side's places are the strategist's."""
    out = []
    mz, ez = _zones()
    for k in range(n):
        name = names[k % len(names)]
        b = AR.augment(AR.base(name), rnd)
        sp = {"src": name, "battle": {1: [_unit(name, u) for u in b[1]], 2: [_unit(name, u) for u in b[2]],
                                      "factions": {1: AR.faction(name, 1), 2: AR.faction(name, 2)}},
              "mine": dict(mz), "theirs": dict(ez), "enemy": enemy, "kind": "full",
              "n_sn": sum(role(u["key"]) == "sniper" for u in b[1])}
        out.append(sp)
    return out


def _zone_t(sp, k, device):
    return {n: torch.tensor([float(s[k][m]) for s in sp], device=device)
            for n, m in (("x", "x"), ("z", "z"), ("b", "b"), ("front", "front"), ("depth", "depth"))}


class Batch:
    """The battles before the deployment (module doc)."""

    def __init__(self, sp, P, device, per_side=None):
        self.sp, self.P, self.device = sp, P, device
        self.per_side = per_side or max(PER_SIDE, max(max(len(s["battle"][1]), len(s["battle"][2])) for s in sp))
        self.mine_zone, self.enemy_zone = _zone_t(sp, "mine", device), _zone_t(sp, "theirs", device)
        rows = []
        for b, s in enumerate(sp):
            bt = {"factions": s["battle"]["factions"]}
            for side, zk in ((1, "mine"), (2, "theirs")):
                z = {k: torch.tensor([float(s[zk][k])]) for k in ("x", "z", "b", "front", "depth")}
                us = s["battle"][side]
                pl = SI.row_places(z, len(us))
                bt[side] = [{"key": u["key"], "x": x, "z": zz, "b": float(s[zk]["b"]), "general": u["general"],
                             "width": None} for u, (x, zz) in zip(us, pl)]
            rows.append(bt)
        self.names = [s["src"] for s in sp]
        self.st0, self.card = DU.build(self.names, rows, device, P, self.per_side)
        u = self.st0.u
        self.raw0 = TR.Seer(P).raw(self.st0)
        B = self.st0.B
        self.side = torch.ones(B, dtype=torch.long, device=device)
        self.command = DC.order_vec(B, "hold", "any", device)
        bounds = P.map_half
        axes = ZS.axes(self.mine_zone, self.enemy_zone, self.raw0, bounds)
        self.inp = IN.build(self.raw0, self.card, self.side, axes, self.mine_zone, self.enemy_zone, self.command,
                            bounds=bounds)
        self.ctx = NN.Ctx(axes, self.mine_zone, self.enemy_zone)
        self.own = u["side"] == 1
        self.shooter = (u["range"] > 0) & self.own
        self.width0 = u["width"].clone()

    def repeat(self, k):
        """(inp, ctx) of the batch k times over (battle b's copies at b, b + B, ...)."""
        B = self.st0.B
        idx = torch.arange(B, device=self.device).repeat(k)
        return take(self.inp, self.ctx, idx)


def take(inp, ctx, idx):
    i2 = {k: v[idx] for k, v in inp.items()}
    return i2, NN.Ctx(ctx.axes.take(idx), ZS.take(ctx.mine, idx), ZS.take(ctx.enemy, idx))


def sniper_groups(inp, comp):
    """[B, G] bool: the groups whose members are all snipers (their passports: operative/army.roles)."""
    cards = inp["rows"][..., :69].cpu().numpy()
    own = inp["own"].cpu().numpy()
    B, K = own.shape
    G = int(cfg()["groups"])
    out = torch.zeros((B, G), dtype=torch.bool)
    for b in range(B):
        rows = [j for j in range(K) if own[b, j]]
        rl = AM.roles(cards[b][rows])
        by = {}
        for j, r in zip(rows, rl):
            by.setdefault(int(comp[b, j]), set()).add(r)
        for g, rs in by.items():
            out[b, g] = rs == {"sniper"}
    return out.to(inp["own"].device)


@torch.no_grad()
def decide(inp, ctx, who, temp=None, greedy=True):
    """The strategist's choice net.choose output (module doc). who: 'teacher', 'exposed', 'sample' (temp [B] the cells'
    spread), a StrategistNet."""
    B, K = inp["own"].shape
    G = int(cfg()["groups"])
    dev = inp["own"].device
    if who == "teacher":
        return TC.play_formula(inp, ctx)
    if isinstance(who, str):
        bias = TC.wishes(inp, ctx)
        if who == "teacher0":
            # the formula's cells without its nudge (what a network's walk can do: its candidate in the search)
            bias["nudge"] = torch.zeros_like(bias["nudge"])
            return NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, dev), greedy=True, bias=bias)
        if who == "deep":
            # the bad one the other way: the snipers' groups behind the middle, DEEP_M behind the front row's cells
            # (out of the zone's depth 176 m: a deployment error by design - hidden, out of everyone's range)
            comp = bias["comp"].argmax(2)
            sg = sniper_groups(inp, comp)
            f_back = (-ctx.mine["depth"][:, None] / 2 + 32.0).expand(B, G)
            k = torch.cumsum(sg.long(), 1)
            s = (k.float() - (sg.sum(1, keepdim=True).float() + 1) / 2) * 40.0
            gx, gz = ZS.world(ctx.mine, s, f_back)
            bias["goal_x"] = torch.where(sg, gx, bias["goal_x"])
            bias["goal_z"] = torch.where(sg, gz, bias["goal_z"])
            back = torch.full((B, G), DEEP_M - float(cfg()["cell_m"]), device=dev)   # (the rows: a cell apart)
            bias["nudge"] = torch.where(sg, -back, bias["nudge"])
            return NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, dev), greedy=True, bias=bias)
        if who == "exposed":
            # the snipers' groups at the front row's ends, 15 m ahead of their cells (the chain's game battles: 22-58 m
            # ahead of the infantry's middle at the line's ends)
            comp = bias["comp"].argmax(2)
            sg = sniper_groups(inp, comp)
            edge = float(cfg()["teacher"]["edge_m"]) + 32.0
            s_end = ctx.mine["front"][:, None] / 2 - edge
            f_front = (ctx.mine["depth"][:, None] / 2 - 32.0).expand(B, G)
            k = torch.cumsum(sg.long(), 1)
            s = torch.where(k % 2 == 1, s_end, -s_end)
            gx, gz = ZS.world(ctx.mine, s, f_front)
            bias["goal_x"] = torch.where(sg, gx, bias["goal_x"])
            bias["goal_z"] = torch.where(sg, gz, bias["goal_z"])
            bias["nudge"] = torch.where(sg, torch.full_like(bias["nudge"], 15.0), bias["nudge"])
            return NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, dev), greedy=True, bias=bias)
        assert who == "sample", who
        bias["nudge"] = torch.zeros_like(bias["nudge"])
        bias["cell_temp"] = temp if temp is not None else torch.ones(B, device=dev)
        return NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, dev), greedy=False, bias=bias)
    return who(inp, ctx, greedy=greedy)


class Mixed(DU.Script):
    """The enemy's script per battle (opponents.<name>)."""

    def __init__(self, names, device):
        self.side = 2
        self.masks = [(n, torch.tensor([m == n for m in names], device=device)) for n in dict.fromkeys(names)]

    def orders(self, st, j):
        out = None
        for name, m in self.masks:
            o = getattr(opponents, name)(st)
            out = o if out is None else O.merge(out, o, m[:, None].expand(st.B, st.N))
        return out


class Watch:
    """Every second: each unit's first rout (or death) second, my snipers' nearest enemy fast unit, the men at
    LIMIT_S."""

    def __init__(self, sniper, fast_e, at=LIMIT_S):
        self.sniper, self.fast_e, self.at = sniper, fast_e, at
        self.rout = None
        self.fast60 = None
        self.men_at = None
        self.shots_at = None

    def __call__(self, sec, st):
        u = st.u
        t = float(sec + 1)
        out = u["r"] | (u["men"] <= 0) | u["gone"]
        if self.rout is None:
            self.rout = torch.full(u["men"].shape, -1.0, device=st.device)
            self.fast60 = torch.full((st.B,), -1.0, device=st.device)
        self.rout = torch.where(out & (self.rout < 0), torch.full_like(self.rout, t), self.rout)
        live_f = self.fast_e & ~out
        live_s = self.sniper & ~out
        d = torch.sqrt((u["x"][:, :, None] - u["x"][:, None, :]) ** 2 + (u["z"][:, :, None] - u["z"][:, None, :]) ** 2)
        near = ((d < 60.0) & live_s[:, :, None] & live_f[:, None, :]).flatten(1).any(1)
        self.fast60 = torch.where(near & (self.fast60 < 0), torch.full_like(self.fast60, t), self.fast60)
        if sec + 1 == self.at:
            self.men_at = (u["men"].float() * (~out).float()).clone()
            self.shots_at = (u["ammo0"] - u["a"]).float().clone()


def placed(batch, out, inp=None):
    """(battles with my units at the strategist's places, groups0 [B, G], places)."""
    inp = batch.inp if inp is None else inp
    plan = out["plan"]
    N = batch.st0.N
    mem = BT.slots_of(plan, inp, N)
    pl = DP.places(mem, batch.shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], batch.width0,
                   DP.slot_of(plan))
    battles = []
    for b, s in enumerate(batch.sp):
        bt = {"factions": s["battle"]["factions"], 2: copy.deepcopy(s["battle"][2]), 1: []}
        for i, u in enumerate(s["battle"][1]):
            bt[1].append({"key": u["key"], "general": u["general"], "x": float(pl["x"][b, i]),
                          "z": float(pl["z"][b, i]), "b": float(pl["b"][b, i]), "width": float(pl["width"][b, i])})
        battles.append(bt)
    B = batch.st0.B
    G = mem.shape[1]
    dev = batch.device
    g0 = GR.Groups.empty(B, N, G, dev)
    g0.member = pl["group"]
    g0.on = plan["on"].clone()
    g0.cx, g0.cz, g0.heading = plan["cx"].clone(), plan["cz"].clone(), plan["heading"].clone()
    g0.line = plan["line"].clone()
    g0.verb = torch.full((B, G), DC.VERBS.index("hold"), dtype=torch.long, device=dev)
    g0.pace = torch.full((B, G), DC.PACES.index("any"), dtype=torch.long, device=dev)
    g0.n0 = plan["n"].clone()
    return battles, g0, pl


def roles_of_state(st):
    return [[role(k) if k else "" for k in keys] for keys in st.keys]


@torch.no_grad()
def play(batch, out, unit_net, P, device, limit=LIMIT_S, inp=None, enemies=None):
    """(measures {name: [B]}, stats) of the battles (module doc)."""
    battles, g0, pl = placed(batch, out, inp)
    names = batch.names
    chain = DU.Chain(1, "formula", command="hold", groups0=g0)
    enemy = Mixed(enemies or [s["enemy"] for s in batch.sp], device)
    st0 = batch.st0
    rl = roles_of_state(st0)
    side = st0.u["side"]
    sn = torch.tensor([[r == "sniper" for r in rr] for rr in rl], device=device) & (side == 1)
    fast_e = torch.tensor([[r == "fast" for r in rr] for rr in rl], device=device) & (side == 2)
    watch = Watch(sn, fast_e, at=min(LIMIT_S, limit))
    cap = {}
    _b = DU.build

    def build(*a, **k):
        st, c = _b(*a, **k)
        cap["st"] = st
        return st, c
    DU.build = build
    try:
        rd, stats = DU.play(names, battles, {1: chain, 2: enemy}, unit_net, P, device, limit_s=limit,
                            per_side=batch.per_side, watch=watch)
    finally:
        DU.build = _b
    st = cap["st"]
    u = st.u
    spacing = P.sim["formation"]["spacing_m"]
    s0, _ = _b(names, battles, device, P, batch.per_side)            # the built state at second 0
    errs = DP.errors(s0, s0.u["side"] == 1, g0.member, batch.mine_zone, spacing) + out["plan"]["no_cell"]
    men0 = u["men0"].float().clamp(min=1)
    men_at = watch.men_at if watch.men_at is not None else (u["men"].float() * (~(u["r"] | u["gone"])).float())
    n_sn = sn.float().sum(1)
    kept = ((men_at / men0) * sn.float()).sum(1) / n_sn.clamp(min=1)
    own = side == 1
    rt = torch.where(own & (watch.rout >= 0), watch.rout, torch.full_like(watch.rout, 1e9))
    first = rt.amin(1)
    sn_first = ((rt <= first[:, None]) & sn).any(1) & (first < 1e9)
    trade = rd["lost"][:, 1] - rd["lost"][:, 0]
    fast60 = torch.where(watch.fast60 >= 0, watch.fast60, torch.full_like(watch.fast60, float(limit)))
    m = {"sniper_kept": kept, "sn_first": sn_first.float(), "any_rout": (first < 1e9).float(),
         "fast60_s": fast60, "trade": trade, "score": kept + trade, "winner": rd["winner"].float(),
         "lost_own": rd["lost"][:, 0], "lost_enemy": rd["lost"][:, 1], "deploy_errors": errs.float(),
         "has_sn": (n_sn > 0).float(), "sniper_recent_dealt": ((u["shot_dealt"].float() * sn.float()).sum(1)
                                                               / n_sn.clamp(min=1))}
    # (shot_dealt is the recent damage, decaying - the morale's balance - not a total: the fire is the projectiles
    # spent, a total)
    shots = (u["ammo0"] - u["a"]).float()
    s150 = watch.shots_at if watch.shots_at is not None else shots
    m["sniper_shots150"] = (s150 * sn.float()).sum(1) / n_sn.clamp(min=1)
    m["sniper_shots"] = (shots * sn.float()).sum(1) / n_sn.clamp(min=1)
    return m, stats


def useful(m, ref):
    """[B] the drill's measure (08.11, the orchestrator: a sniper is of use only alive AND firing): my snipers' men kept
    at 150 s x their projectiles spent by 150 s over the teacher's on the same battle, at most 1 (ref [B] the
    teacher's sniper_shots150; none fired by the teacher: the men alone) - hiding them deep keeps the men and loses the
    fire, standing them ahead fires and loses the men; fire beyond the teacher's buys nothing (the drill is the
    cover)."""
    fire = torch.where(ref > 1.0, (m["sniper_shots150"] / ref.clamp(min=1.0)).clamp(max=1.0),
                       torch.ones_like(ref))
    return m["sniper_kept"] * fire


def summary(m, sel=None):
    def ci(x):
        x = x.float()
        n = x.numel()
        mean = float(x.mean()) if n else float("nan")
        sd = float(x.std()) if n > 1 else 0.0
        return round(mean, 3), round(1.96 * sd / max(n, 1) ** 0.5, 3)
    out = {}
    for k, v in m.items():
        v = v if sel is None else v[sel]
        if k == "winner":
            out["win"] = round(float((v == 1).float().mean()), 3)
            out["draw"] = round(float((v == 0).float().mean()), 3)
            out["loss"] = round(float((v == 2).float().mean()), 3)
        else:
            out[k] = ci(v)
    out["n"] = int(next(iter(m.values())).numel() if sel is None else int(sel.sum()))
    return out


def cat_out(outs):
    """The walk's outputs of consecutive slices of a batch, as one."""
    def cat(vs):
        if isinstance(vs[0], dict):
            return {k: cat([v[k] for v in vs]) for k in vs[0]}
        if torch.is_tensor(vs[0]) and vs[0].dim() > 0:
            return torch.cat(vs, 0)
        return vs[0]
    return {k: cat([o[k] for o in outs]) for k in ("act", "plan")}


def take_out(out, idx):
    def tk(v):
        if isinstance(v, dict):
            return {k: tk(x) for k, x in v.items()}
        return v[idx] if torch.is_tensor(v) and v.dim() > 0 else v
    return {k: tk(out[k]) for k in ("act", "plan")}


def seeded(seed):
    return random.Random(seed)
