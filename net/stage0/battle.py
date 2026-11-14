"""A batch of stage 0 specs (specs.py) played to its end (matchup/duel.play): my side the chain of our levels with a
behaviour on top (behave.py), the enemy a script or the formula chain, the readings of watch.py.

    Prep(specs, P, device, behaviour)   the battles before the first second: my side deployed by the strategist's
                  formula (a spec without its own places: its input as the game gives it - both sides in a row across
                  their zones' centres) with the behaviour's change of the plan, else at the spec's places; the enemy at
                  its places; each side's lord (the general first: slot 0 of its side) lord_back_m behind its units'
                  middle, away from the enemy, out of every group; the named reserve [B, N]
    StageChain    matchup/duel.Chain under the behaviour: its lord hidden from the chain (its row of the raw state not
                  alive: no group, no tile) and holding; the behaviour's orders over the chain's
    play(specs, behaviour, unit_net, P, device, unit_level=None, gate=False, unit_greedy=True) -> (measures {name: [B]},
                  stats): the specs in batches of every enemy kind together (Mixed: the chain plays all, the scripts'
                  orders where theirs), each padded to a multiple of PAD (few shapes of the compiled step), the
                  measures put back in order; unit_level - my side's unit level instead of unit_net (a
                  network or a scripted player, net/unit/scripts.py; the enemy's chain keeps unit_net), gate - my
                  orders to the game re-issued only when they change (net/common/reissue.py: watch's game_orders_min)
    play_units / play_tactic   the general battles of the unit level (UnitSide: each frame its slot of the deployment)
                  / through the tactic level (TacticSide: one group, its zone by formation, a player of tactic/pin.py)
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
    def __init__(self, side, behaviour=None, reserve=None, command="hold", groups0=None, unit_level=None, gate=False,
                 unit_greedy=True):
        super().__init__(side, "formula", command=command, groups0=groups0, unit_level=unit_level, gate=gate,
                         unit_greedy=unit_greedy)
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


class Mixed(StageChain):
    """The enemy of a batch with both kinds: the formula chain plays every battle, the scripts' orders replace its own
    where the battle's enemy is a script - one batch, so one shape of the compiled step (module doc of play)."""

    def __init__(self, names, device):
        super().__init__(2, BH.Behaviour(), command="destroy")
        self.script = torch.tensor([n != "chain" for n in names], device=device)
        self.scripts = Scripts([n if n != "chain" else "hold" for n in names], device)

    def orders(self, st, j):
        return O.merge(super().orders(st, j), self.scripts.orders(st, j), self.script[:, None].expand(st.B, st.N))


def _play_one(specs, behaviour, unit_net, P, device, duty=None, unit_level=None, gate=False, unit_greedy=True,
              compile=None):
    prep = Prep(specs, P, device, behaviour)
    limit = max(int(s["limit"]) for s in specs)
    chain = StageChain(1, behaviour, reserve=prep.reserve, command=cfg()["general"]["command"], groups0=prep.groups0,
                       unit_level=unit_level, gate=gate, unit_greedy=unit_greedy)
    kinds = {s["enemy"] for s in specs}
    if kinds == {"chain"}:
        enemy = StageChain(2, BH.Behaviour(), command="destroy")
    elif "chain" in kinds:
        enemy = Mixed([s["enemy"] for s in specs], device)
    else:
        enemy = Scripts([s["enemy"] for s in specs], device)
    d = cfg()["drills"]
    t02, o03 = d["t02"], d["o03"]
    watch = W.Watch(P, cfg()["general"], chain=chain, reserve=prep.reserve, window_s=t02["flank_window_s"],
                    after_s=t02["trade_after_s"], early_s=o03["early_s"])
    rd, stats = DU.play([NAME] * len(specs), prep.battles, {1: chain, 2: enemy}, unit_net, P, device, limit_s=limit,
                        per_side=prep.per_side, watch=watch, maps=prep.maps, duty=duty, compile=compile)
    m = watch.measures(limit, rd["winner"])
    m["deploy_errors"] = prep.deploy_errors if prep.groups0 is None else prep.no_cell
    s1 = stats.get("side1", {})
    stats = {"battles": stats["battles"], "seconds": stats["seconds"], "wall_s": stats["wall_s"],
             "parcels": s1.get("parcels", 0), "parcel_errors": s1.get("parcel_errors", 0),
             "errors": s1.get("errors", [])[:3]}
    if isinstance(enemy, StageChain):
        stats["enemy_parcel_errors"] = enemy.stats["parcel_errors"]
    return m, stats


PAD = 64          # a batch is padded to a multiple of this many battles (copies, their results dropped)


def _pad(n, device, compile):
    """The copies a batch of n gets: up to a multiple of PAD when the step is compiled (CUDA), none else."""
    compiled = torch.device(device).type == "cuda" if compile is None else bool(compile)
    return (-n) % PAD if compiled else 0


def play(specs, behaviour, unit_net, P, device, chunk=512, duty=None, unit_level=None, gate=False, unit_greedy=True,
         compile=None):
    """(measures {name: [B]} in the specs' order, stats summed) (module doc). The specs go in batches of chunk, every
    enemy kind together (Mixed), each batch padded to a multiple of PAD on CUDA: the compiled step (sim/battle.stepper,
    torch.compile without dynamic shapes) is compiled anew - ~18 min on the RTX - for every new batch shape and is kept
    in the container's cache for the next runs, so few shapes; compile False: the plain step (no warm-up, ~5x slower a
    step)."""
    parts = {}
    total = {"battles": 0, "wall_s": 0.0, "parcels": 0, "parcel_errors": 0, "enemy_parcel_errors": 0, "errors": []}
    for a in range(0, len(specs), chunk):
        ii = list(range(a, min(a + chunk, len(specs))))
        pad = _pad(len(ii), device, compile)
        batch = [specs[i] for i in ii] + [copy.deepcopy(specs[ii[k % len(ii)]]) for k in range(pad)]
        m, st = _play_one(batch, copy.deepcopy(behaviour), unit_net, P, device, duty, unit_level=unit_level,
                          gate=gate, unit_greedy=unit_greedy, compile=compile)
        for k, v in m.items():
            parts.setdefault(k, {}).update({i: float(x) for i, x in zip(ii, v.tolist()[:len(ii)])})
        for k in ("battles", "wall_s", "parcels", "parcel_errors", "enemy_parcel_errors"):
            total[k] += st.get(k, 0)
        total["errors"] += st["errors"]
    m = {k: torch.tensor([v[i] for i in range(len(specs))]) for k, v in parts.items()}
    total["errors"] = total["errors"][:5]
    total["wall_s"] = round(total["wall_s"], 1)
    return m, total


FRAME_ROW = ("enemy_row_m", "enemy_rank_m")    # the unit level's frame in a general battle: its slot of the rows


class UnitSide(DU.Chain):
    """My side of a general battle of the unit level (play_units): no level above - each unit's frame is its slot of
    the deployment (the environment's rule, the same for every player and the network: the centre where it stands at
    the start, the zone's course, enemy_row_m across x enemy_rank_m deep - the rows' spacing); the frames never move;
    the lord holds."""

    def __init__(self, unit_level=None, gate=True, unit_greedy=False):
        super().__init__(1, "formula", unit_level=unit_level, gate=gate, unit_greedy=unit_greedy)

    def start(self, st, P, device, card, unit_net):
        from net.operative import lower as LW
        B, N = st.B, st.N
        u = st.u
        self.setup = DU._SideSetup(st, P, device, 1, card)
        level = unit_net if self.unit_level is None else self.unit_level
        self.units = LW.Units(self.setup, level, unit_greedy=self.unit_greedy, targets=False, still=True, aim=True,
                              gate=self.gate)
        g = cfg()["general"]
        own = self.setup.own
        self.units.area = {"x": u["x"].clone(), "z": u["z"].clone(), "b": u["b"].clone(),
                           "front": torch.full((B, N), float(g[FRAME_ROW[0]]), device=device),
                           "depth": torch.full((B, N), float(g[FRAME_ROW[1]]), device=device),
                           "sb": u["b"].clone(), "sw": torch.zeros((B, N), device=device), "on": own.clone()}
        self.hide = u["lord"] & own
        self.stats = {"parcels": 0, "parcel_errors": 0, "errors": [], "groups": 0.0, "decisions": 0}

    def decide(self, st, raw, K):
        pass

    def tick(self, st, raw):
        pass

    def second(self, st):
        self.lands = self.units.second(st)

    def orders(self, st, j):
        o = self.units.orders(st, j, self.lands, lambda s_: O.hold(st.B, st.N, st.device))
        return O.merge(o, O.hold(st.B, st.N, st.device), self.hide)


class TacticSide(UnitSide):
    """My side of a general battle through the tactic level (tmp/tactic_slice/design_flank_v2.md section 4): one group
    = my units but the lord (his row hidden, holding), its zone by formation (tactic/zfit.zone_for; config
    tactic.json orders.zone_fit off - a line 1 x n on the units' middle, as the pilot) handed once at second 0 through
    lower.Tacticians.apply, no operative; every 5 s the level's tick with the player (tactic/pin.py) and
    lower.Units.write, every second the unit's script. The level's own per-tick measures (self.acc [B]):
    unpinned_attacks (attack orders on an enemy not pinned by pin.pinned at the order), two_on_side (enemies with more
    strikers than the cap, or two on one side, after a tick)."""

    def __init__(self, player, hygiene=True, unit_level=None):
        from net.unit import scripts as SC
        super().__init__(SC.Unit() if unit_level is None else unit_level, gate=True, unit_greedy=False)
        self.player, self.hygiene = player, hygiene

    def start(self, st, P, device, card, unit_net):
        from net.operative import lower as LW
        from net.tactic import cfg as tcfg
        from net.tactic import zfit
        super().start(st, P, device, card, unit_net)
        u = st.u
        B, N = st.B, st.N
        self.mine = (u["side"] == 1) & ~u["lord"] & (u["men"] > 0)
        self.tac = LW.Tacticians(B, N, 1, device, player=self.player, hygiene=self.hygiene)
        g = GR.Groups.empty(B, N, 1, device)
        g.member = torch.where(self.mine, torch.zeros_like(g.member), g.member)
        g.on[:, 0] = self.mine.any(1)
        g.n0[:, 0] = self.mine.sum(1)
        g.verb[:, 0] = DC.VERBS.index("hold")
        rows, cols = [], []
        self.worst = []
        fit = bool(tcfg()["orders"]["zone_fit"])
        x, z, b = u["x"].cpu(), u["z"].cpu(), u["b"].cpu()
        for k in range(B):
            sl = self.mine[k].nonzero()[:, 0].tolist()
            pl = [(float(x[k, i]), float(z[k, i])) for i in sl]
            course = float(b[k, sl[0]]) if sl else 0.0
            zn = zfit.zone_for(pl, course) if (fit and pl) else None
            if zn is None:
                n = max(len(pl), 1)
                mx = sum(p[0] for p in pl) / n if pl else 0.0
                mz = sum(p[1] for p in pl) / n if pl else 0.0
                zn = {"cx": mx, "cz": mz, "heading": course, "line": True, "rows": 1, "cols": n, "worst": float("nan")}
            g.cx[k, 0], g.cz[k, 0], g.heading[k, 0], g.line[k, 0] = zn["cx"], zn["cz"], zn["heading"], bool(zn["line"])
            rows.append(zn["rows"])
            cols.append(zn["cols"])
            self.worst.append(zn["worst"])
        self.groups0 = g
        self.shape0 = (torch.tensor(rows), torch.tensor(cols))
        self.applied = False
        self.acc = {k: torch.zeros(B, device=device) for k in ("unpinned_attacks", "two_on_side")}

    def _raw(self, raw):
        r = raw.clone()
        a = TR.I["alive"]
        r[..., a] = torch.where(self.hide, torch.zeros_like(r[..., a]), r[..., a])
        return r

    def tick(self, st, raw):
        raw = self._raw(raw)
        if not self.applied:
            self.applied = True
            on = self.groups0.on
            self.tac.apply(self.groups0, on, torch.zeros_like(on), raw)
            self.tac.shape(*self.shape0)
        self.tac.tick(raw, self.setup.card, None, now=float(st.t.float().max()))
        self.units.write(self.tac)
        self._measure()

    APPROACH_M = 60.0            # (the drill's split of attack orders: 'approaching my line within 60 m')

    def _measure(self):
        """The level's per-tick measures (class doc) and the split of the new attack orders by what the target was
        doing at the order (from the level's facts): pinned (pin.pinned); else chasing a router (moving at
        scripts.APPROACH_MS or more, the nearest unit of mine by the edge gap routing); else approaching (closing on the
        nearest standing unit of mine at APPROACH_MS or more, within APPROACH_M edge to edge); else standing (under
        APPROACH_MS, not in melee); else other."""
        from net.tactic import orders as OR
        from net.tactic import pin as PN
        from net.unit import scripts as SC
        v = self.tac.view
        if v is None:
            return
        fx, bi = v["view"].fx, v["bi"]
        pin = PN.pinned(fx)
        k = fx.row_of_slot(v["tgt"])
        kc = k.clamp(min=0)
        at_pin = (k >= 0) & pin.gather(1, kc)
        new = v["new_attack"]
        self.acc["unpinned_attacks"].index_add_(0, bi, (new & ~at_pin).float().sum(1))
        # what the target was doing
        g = lambda t: t.gather(1, kc)                                                        # noqa: E731
        moving = g(fx.speed) >= SC.APPROACH_MS
        mine_any = fx.mine[:, None, :]
        gap_m = torch.where(mine_any, fx.gap, torch.full_like(fx.gap, OR.INF))              # [R, e, a]
        near_a = gap_m.argmin(2)                                                             # [R, K] my nearest unit
        router = fx.r.gather(1, near_a)
        stand = fx.mine & ~fx.r
        gap_s = torch.where(stand[:, None, :], fx.gap, torch.full_like(fx.gap, OR.INF))
        a_s = gap_s.argmin(2)
        dxs = fx.s.gather(1, a_s) - fx.s
        dfs = fx.f.gather(1, a_s) - fx.f
        nrm = torch.sqrt(dxs ** 2 + dfs ** 2).clamp(min=1e-6)
        closing = (fx.vs * dxs + fx.vf * dfs) / nrm
        appr = (closing >= SC.APPROACH_MS) & (gap_s.amin(2) <= self.APPROACH_M)
        chase = moving & g(router)
        cats = {"ord_pinned": at_pin, "ord_chasing": ~at_pin & chase,
                "ord_approaching": ~at_pin & ~chase & g(appr),
                "ord_standing": ~at_pin & ~chase & ~g(appr) & ~moving & ~g(fx.m)}
        cats["ord_other"] = ~(cats["ord_pinned"] | cats["ord_chasing"] | cats["ord_approaching"] | cats["ord_standing"])
        for name, c in cats.items():
            if name not in self.acc:
                self.acc[name] = torch.zeros_like(self.acc["unpinned_attacks"])
            self.acc[name].index_add_(0, bi, (new & (k >= 0) & c).float().sum(1))
        # every unit of mine holding an attack order on the enemy, routing or not
        att = (fx.my >= 0) & (v["skill"] == SC.ATTACK) & (k >= 0)
        hot = torch.nn.functional.one_hot(kc, fx.K).bool() & att[..., None]                # [r, G, K]
        side = v["side"]                                       # (the side written at each order)
        cap = int(OR.c_orders()["strikers"])
        over = (hot.sum(1) > cap).float().sum(1)
        for sd in (OR.LEFT, OR.RIGHT):
            over += ((hot & (side == sd)[..., None]).sum(1) > 1).float().sum(1)
        self.acc["two_on_side"].index_add_(0, bi, over)


def play_tactic(specs, player, P, device, hygiene=True, chunk=512, compile=None, watch_cls=None):
    """(measures {name: [B]}, stats) of stage 0's general battles of the unit level (play_units' battles and enemies)
    with my side through the tactic level (TacticSide) under player() - a factory of a fresh player of tactic/pin.py
    each batch (its memory starts anew); watch_cls - a watch.Watch subclass (watch.PinWatch for the drill's events)."""
    assert all(s["enemy"] != "chain" for s in specs), "the unit level's general battles: script enemies only"
    parts, total = {}, {"battles": 0, "wall_s": 0.0}
    for a in range(0, len(specs), chunk):
        ii = list(range(a, min(a + chunk, len(specs))))
        pad = _pad(len(ii), device, compile)
        batch = [specs[i] for i in ii] + [copy.deepcopy(specs[ii[k % len(ii)]]) for k in range(pad)]
        mine = TacticSide(player(), hygiene=hygiene)
        enemy = Scripts([s["enemy"] for s in batch], device)
        watch = (watch_cls or W.Watch)(P, cfg()["general"], chain=mine)
        limit = max(int(s["limit"]) for s in batch)
        rd, stt = DU.play([NAME] * len(batch), unit_battles(batch), {1: mine, 2: enemy}, None, P, device,
                          limit_s=limit, per_side=PER_SIDE, watch=watch, maps=[s["map"] for s in batch],
                          compile=compile)
        m = watch.measures(limit, rd["winner"])
        for name, v in mine.tac.counts().items():
            m[name] = v
        for name, v in mine.acc.items():
            m[name] = v
        m["zone_worst_m"] = torch.tensor(mine.worst, device=m["trade"].device)
        for k, v in m.items():
            v = v.to(m["trade"].device) if torch.is_tensor(v) else v
            if torch.is_tensor(v) and v.dim() == 1:
                parts.setdefault(k, {}).update({i: float(x) for i, x in zip(ii, v.tolist()[:len(ii)])})
            elif torch.is_tensor(v) and v.dim() == 2:
                parts.setdefault(k, {}).update({i: x for i, x in zip(ii, v.tolist()[:len(ii)])})
        total["battles"] += stt["battles"]
        total["wall_s"] += stt["wall_s"]
    m = {}
    for k, v in parts.items():
        vals = [v[i] for i in range(len(specs))]
        m[k] = torch.tensor(vals)
    total["wall_s"] = round(total["wall_s"], 1)
    return m, total


def unit_battles(specs):
    """The battles of specs for play_units: my units in rows across my zone (specs.rows, as the enemy's; centred), the
    enemy at its places, each lord lord_back_m behind its side."""
    from net.stage0 import specs as SP
    out = []
    for s in specs:
        mz, ez = s["mine_zone"], s["enemy_zone"]
        mine = [_unit(k, *p) for k, p in zip(s["mine"], SP.rows(mz, len(s["mine"])))]
        theirs = [_unit(k, *p) for k, p in zip(s["theirs"], s["places_theirs"])]
        l1 = _unit(cfg()["lords"][s["fac"][1]], *_lord_place(mz, [(u["x"], u["z"]) for u in mine]), general=True)
        l2 = _unit(cfg()["lords"][s["fac"][2]], *_lord_place(ez, [(u["x"], u["z"]) for u in theirs]), general=True)
        out.append({1: [l1] + mine, 2: [l2] + theirs, "factions": dict(s["factions"])})
    return out


def play_units(specs, unit_level, unit_net, P, device, gate=True, unit_greedy=False, chunk=512, compile=None, duty=None):
    """(measures {name: [B]}, stats) of stage 0's general battles of the unit level: my side UnitSide (unit_level - a
    scripted player or None: unit_net), the enemy its script (a spec's 'chain' enemy - a formula - not allowed here);
    batches padded as play's."""
    assert all(s["enemy"] != "chain" for s in specs), "the unit level's general battles: script enemies only"
    parts, total = {}, {"battles": 0, "wall_s": 0.0}
    for a in range(0, len(specs), chunk):
        ii = list(range(a, min(a + chunk, len(specs))))
        pad = _pad(len(ii), device, compile)
        batch = [specs[i] for i in ii] + [copy.deepcopy(specs[ii[k % len(ii)]]) for k in range(pad)]
        mine = UnitSide(unit_level, gate=gate, unit_greedy=unit_greedy)
        enemy = Scripts([s["enemy"] for s in batch], device)
        watch = W.Watch(P, cfg()["general"], chain=mine)
        limit = max(int(s["limit"]) for s in batch)
        rd, stt = DU.play([NAME] * len(batch), unit_battles(batch), {1: mine, 2: enemy}, unit_net, P, device,
                          limit_s=limit, per_side=PER_SIDE, watch=watch, maps=[s["map"] for s in batch],
                          compile=compile, duty=duty)
        m = watch.measures(limit, rd["winner"])
        for k, v in m.items():
            parts.setdefault(k, {}).update({i: float(x) for i, x in zip(ii, v.tolist()[:len(ii)])})
        total["battles"] += stt["battles"]
        total["wall_s"] += stt["wall_s"]
    m = {k: torch.tensor([v[i] for i in range(len(specs))]) for k, v in parts.items()}
    total["wall_s"] = round(total["wall_s"], 1)
    return m, total
