"""A batch of stage 0 specs (specs.py) played to its end (matchup/duel.play): my side the chain of our levels with a
behaviour on top (behave.py), the enemy a script or the formula chain, the readings of watch.py.

    Prep(specs, P, device, behaviour)   the battles before the first second: my side deployed by the strategist's
                  formula (a spec without its own places: its input as the game gives it - both sides in a row across
                  their zones' centres) with the behaviour's change of the plan, else at the spec's places; the enemy at
                  its places; each side's lord (the general first: slot 0 of its side) lord_back_m behind its units'
                  middle, away from the enemy, out of every group; the named reserve [B, N]
    StageChain    matchup/duel.Chain under the behaviour: its lord hidden from the chain (its row of the raw state not
                  alive: no group, no tile) and holding; the behaviour's orders over the chain's
    play(specs, behaviour, unit_net, P, device) -> (measures {name: [B]}, stats): the specs split by the enemy's kind
                  (scripts in one batch, the formula chain in another), each played, the measures put back in order
"""
import copy
import math

import torch

from net.strategist import zones as SI
from net.matchup import duel as DU
from net.operative import groups as GR
from sim import orders as O
from net.stage0 import behave as BH
from net.stage0 import cfg
from net.stage0 import watch as W
from net.strategist import battle as BT
from net.strategist import deploy as DP
from net.strategist import inputs as IN
from net.strategist import net as NN
from net.strategist import teacher as STC
from net.strategist import zones as ZS
from net.tactic import decide as DC
from net.tactic import raw as TR
from sim import scripts as opponents

PER_SIDE = 9                           # 8 units + the lord at most (config general sizes)
NAME = "m1"                            # (a matchup's name for duel.build: the factions are the spec's own)


def _zone_t(sp, k, device):
    return {n: torch.tensor([float(s[k][n]) for s in sp], device=device) for n in ("x", "z", "b", "front", "depth")}


def _lord_place(zone, pts):
    """(x, z, b) lord_back_m behind the middle of pts [(x, z)] along the zone's facing."""
    back = float(cfg()["lord_back_m"])
    mx = sum(p[0] for p in pts) / max(len(pts), 1)
    mz = sum(p[1] for p in pts) / max(len(pts), 1)
    a = math.radians(zone["b"])
    return mx - back * math.sin(a), mz - back * math.cos(a), zone["b"]


def _unit(key, x, z, b, general=False, width=None):
    return {"key": key, "x": float(x), "z": float(z), "b": float(b), "general": general, "width": width}


class Prep:
    def __init__(self, specs, P, device, behaviour=None, per_side=PER_SIDE):
        self.specs, self.P, self.device = specs, P, device
        self.behaviour = behaviour or BH.Behaviour()
        self.per_side = per_side
        self.maps = [s["map"] for s in specs]
        self.mine_zone, self.enemy_zone = _zone_t(specs, "mine_zone", device), _zone_t(specs, "enemy_zone", device)
        B, H = len(specs), per_side
        self.groups0 = None
        self.deploy_errors = torch.zeros(B, device=device)
        strat = specs[0]["places_mine"] is None
        assert all((s["places_mine"] is None) == strat for s in specs), "a batch is deployed one way"
        if strat:
            mine_places = self._strategist()
        else:
            mine_places = [list(s["places_mine"]) for s in specs]
        reserve = [list(s.get("reserve") or []) for s in specs]
        if strat and self.behaviour.wants_reserve():
            reserve = self._named_reserve(mine_places)
        self.battles = []
        for b, s in enumerate(specs):
            mz, ez = s["mine_zone"], s["enemy_zone"]
            mine = [_unit(k, *p) for k, p in zip(s["mine"], mine_places[b])]
            theirs = [_unit(k, *p) for k, p in zip(s["theirs"], s["places_theirs"])]
            l1 = _unit(cfg()["lords"][s["fac"][1]], *_lord_place(mz, [(u["x"], u["z"]) for u in mine]), general=True)
            l2 = _unit(cfg()["lords"][s["fac"][2]], *_lord_place(ez, [(u["x"], u["z"]) for u in theirs]), general=True)
            self.battles.append({1: [l1] + mine, 2: [l2] + theirs, "factions": dict(s["factions"])})
        self.reserve = torch.zeros((B, 2 * H), dtype=torch.bool, device=device)
        for b, r in enumerate(reserve):
            for i in r:
                self.reserve[b, 1 + i] = True                                 # (slot 0: the lord)

    def _strategist(self):
        """My units' places by the strategist's formula and the behaviour (drill_s1.Batch's way); the groups."""
        sp, P, dev = self.specs, self.P, self.device
        rows = []
        for s in sp:
            bt = {"factions": s["factions"]}
            for side, zk, keys in ((1, "mine_zone", s["mine"]), (2, "enemy_zone", s["theirs"])):
                z = {k: torch.tensor([float(s[zk][k])]) for k in ("x", "z", "b", "front", "depth")}
                lord = cfg()["lords"][s["fac"][side]]
                us = [lord] + list(keys)
                pl = SI.row_places(z, len(us))
                bt[side] = [_unit(k, x, zz, s[zk]["b"], general=(i == 0)) for i, (k, (x, zz)) in
                            enumerate(zip(us, pl))]
            rows.append(bt)
        self.st0, self.card = DU.build(NAME, rows, dev, P, self.per_side, maps=self.maps)
        u = self.st0.u
        raw0 = TR.Seer(P).raw(self.st0)
        B = self.st0.B
        side = torch.ones(B, dtype=torch.long, device=dev)
        command = DC.order_vec(B, cfg()["general"]["command"], "any", dev)
        bounds = P.map_half
        axes = ZS.axes(self.mine_zone, self.enemy_zone, raw0, bounds)
        inp = IN.build(raw0, self.card, side, axes, self.mine_zone, self.enemy_zone, command, bounds=bounds)
        ctx = NN.Ctx(axes, self.mine_zone, self.enemy_zone)
        with torch.no_grad():
            out = STC.play_formula(inp, ctx)
        self.behaviour.place(self, out)
        plan = out["plan"]
        N = self.st0.N
        mem = BT.slots_of(plan, inp, N)
        own = u["side"] == 1
        shooter = (u["range"] > 0) & own
        pl = DP.places(mem, shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], u["width"].clone(),
                       DP.slot_of(plan))
        self.pl = pl
        out_places = []
        for b, s in enumerate(sp):
            out_places.append([(float(pl["x"][b, 1 + i]), float(pl["z"][b, 1 + i]), float(pl["b"][b, 1 + i]))
                               for i in range(len(s["mine"]))])
        G = mem.shape[1]
        g0 = GR.Groups.empty(B, N, G, dev)
        member = pl["group"].clone()
        member[:, 0] = -1                                                     # (the lord: out of every group)
        g0.member = member
        g0.on = plan["on"].clone() & g0.members().any(2)
        g0.cx, g0.cz, g0.heading = plan["cx"].clone(), plan["cz"].clone(), plan["heading"].clone()
        g0.line = plan["line"].clone()
        g0.verb = torch.full((B, G), DC.VERBS.index("hold"), dtype=torch.long, device=dev)
        g0.pace = torch.full((B, G), DC.PACES.index("any"), dtype=torch.long, device=dev)
        g0.n0 = g0.members().sum(2).long()
        self.groups0 = g0
        self.no_cell = plan["no_cell"].float()
        return out_places

    def _named_reserve(self, places):
        """In a general battle: the rearmost reserve_share units moved reserve_back_m behind the front one (module
        doc of behave.py); places changed in place; returns the reserve's indices."""
        c = cfg()["behaviours"]
        out = []
        for b, s in enumerate(self.specs):
            n = len(s["mine"])
            k = 0
            for lo, kk in c["reserve_share"]:
                if n >= int(lo):
                    k = int(kk)
            if k == 0:
                out.append([])
                continue
            zn = s["mine_zone"]
            from net.stage0 import specs as SP
            loc = [SP.local(zn, x, z) for x, z, _ in places[b]]
            f_front = max(f for _, f in loc)
            order = sorted(range(n), key=lambda i: loc[i][1])[:k]
            f_new = max(f_front - float(c["reserve_back_m"]), -zn["depth"] / 2 + 15.0)
            for i in order:
                x, z = SP.world(zn, loc[i][0], min(loc[i][1], f_new))
                places[b][i] = (x, z, places[b][i][2])
            out.append(sorted(order))
        return out


class StageChain(DU.Chain):
    def __init__(self, side, behaviour=None, reserve=None, command="hold", groups0=None):
        super().__init__(side, "formula", command=command, groups0=groups0)
        self.behaviour = behaviour or BH.Behaviour()
        self.reserve0 = reserve
        self.over = None

    def start(self, st, P, device, card, unit_net):
        super().start(st, P, device, card, unit_net)
        u = st.u
        self.hide = u["lord"] & (u["side"] == self.side)
        self.reserve = None if self.reserve0 is None else (self.reserve0 & (u["side"] == self.side))
        self.behaviour.start(self, st)

    def _raw(self, raw):
        r = raw.clone()
        a = TR.I["alive"]
        r[..., a] = torch.where(self.hide, torch.zeros_like(r[..., a]), r[..., a])
        return r

    def decide(self, st, raw, K):
        return super().decide(st, self._raw(raw), K)

    def tick(self, st, raw):
        super().tick(st, self._raw(raw))

    def second(self, st):
        super().second(st)
        self.over = self.behaviour.second(self, st)

    def orders(self, st, j):
        o = super().orders(st, j)
        if self.over is not None:
            o = O.merge(o, self.over[0], self.over[1])
        return O.merge(o, O.hold(st.B, st.N, st.device), self.hide)


class Scripts(DU.Script):
    """The enemy's script per battle (opponents.<name>), its lord holding."""

    def __init__(self, names, device, side=2):
        self.side = side
        self.masks = [(n, torch.tensor([m == n for m in names], device=device)) for n in dict.fromkeys(names)]

    def orders(self, st, j):
        out = None
        for name, m in self.masks:
            o = getattr(opponents, name)(st)
            out = o if out is None else O.merge(out, o, m[:, None].expand(st.B, st.N))
        lord = st.u["lord"] & (st.u["side"] == self.side)
        return O.merge(out, O.hold(st.B, st.N, st.device), lord)


def _play_one(specs, behaviour, unit_net, P, device, duty=None):
    prep = Prep(specs, P, device, behaviour)
    limit = max(int(s["limit"]) for s in specs)
    chain = StageChain(1, behaviour, reserve=prep.reserve, command=cfg()["general"]["command"], groups0=prep.groups0)
    kinds = {s["enemy"] for s in specs}
    if kinds == {"chain"}:
        enemy = StageChain(2, BH.Behaviour(), command="destroy")
    else:
        assert "chain" not in kinds
        enemy = Scripts([s["enemy"] for s in specs], device)
    d = cfg()["drills"]
    t02, o03 = d["t02"], d["o03"]
    watch = W.Watch(P, cfg()["general"], chain=chain, reserve=prep.reserve, window_s=t02["flank_window_s"],
                    after_s=t02["trade_after_s"], early_s=o03["early_s"])
    rd, stats = DU.play([NAME] * len(specs), prep.battles, {1: chain, 2: enemy}, unit_net, P, device, limit_s=limit,
                        per_side=prep.per_side, watch=watch, maps=prep.maps, duty=duty)
    m = watch.measures(limit, rd["winner"])
    m["deploy_errors"] = prep.deploy_errors if prep.groups0 is None else prep.no_cell
    s1 = stats.get("side1", {})
    stats = {"battles": stats["battles"], "seconds": stats["seconds"], "wall_s": stats["wall_s"],
             "parcels": s1.get("parcels", 0), "parcel_errors": s1.get("parcel_errors", 0),
             "errors": s1.get("errors", [])[:3]}
    if isinstance(enemy, StageChain):
        stats["enemy_parcel_errors"] = enemy.stats["parcel_errors"]
    return m, stats


def play(specs, behaviour, unit_net, P, device, chunk=512, duty=None):
    """(measures {name: [B]} in the specs' order, stats summed) (module doc)."""
    groups = {}
    for i, s in enumerate(specs):
        groups.setdefault("chain" if s["enemy"] == "chain" else "scripts", []).append(i)
    parts = {}
    total = {"battles": 0, "wall_s": 0.0, "parcels": 0, "parcel_errors": 0, "enemy_parcel_errors": 0, "errors": []}
    for kind, idx in groups.items():
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            m, st = _play_one([specs[i] for i in ii], copy.deepcopy(behaviour), unit_net, P, device, duty)
            for k, v in m.items():
                parts.setdefault(k, {}).update({i: float(x) for i, x in zip(ii, v.tolist())})
            for k in ("battles", "wall_s", "parcels", "parcel_errors", "enemy_parcel_errors"):
                total[k] += st.get(k, 0)
            total["errors"] += st["errors"]
    m = {k: torch.tensor([v[i] for i in range(len(specs))]) for k, v in parts.items()}
    total["errors"] = total["errors"][:5]
    total["wall_s"] = round(total["wall_s"], 1)
    return m, total
