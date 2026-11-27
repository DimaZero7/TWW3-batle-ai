"""A batch of stage 0 specs (specs.py) played to its end (matchup/duel.play): my side the chain of our levels with a
behaviour on top (behave.py), the enemy a script or the formula chain, the readings of watch.py.

    Prep(specs, P, device, behaviour)   the battles before the first second: my side deployed by the strategist's
                  formula (a spec without its own places: its input as the game gives it - both sides in a row across
                  their zones' centres) with the behaviour's change of the plan, else at the spec's places; the enemy at
                  its places; each side's lord (the general first: slot 0 of its side) lord_back_m behind its units'
                  middle, away from the enemy, out of every group; the named reserve [B, N]
    StageChain    matchup/duel.Chain under the behaviour: its lord hidden from the chain (its row of the raw state not
                  alive: no group, no tile) and holding; the behaviour's orders over the chain's (the lord keys below
                  are not built here: on, it refuses)
    play(specs, behaviour, unit_net, P, device, unit_level=None, gate=False, unit_greedy=True) -> (measures {name: [B]},
                  stats): the specs in batches of every enemy kind together (Mixed: the chain plays all, the scripts'
                  orders where theirs), each padded to a multiple of PAD (few shapes of the compiled step), the
                  measures put back in order; unit_level - my side's unit level instead of unit_net (a
                  network or a scripted player, net/unit/scripts.py; the enemy's chain keeps unit_net), gate - my
                  orders to the game re-issued only when they change (net/common/reissue.py: watch's game_orders_min)
    play_units / play_tactic   the general battles of the unit level (UnitSide: each frame its slot of the deployment)
                  / through the tactic level (TacticSide: one group, its zone by formation, a player of tactic/pin.py)

The lords (config stage0.json; both off by default - the lords stand lord_back_m behind their side holding, the path bit
for bit as before): lord_script - my lord (UnitSide, TacticSide) played by the T2 lord script at the unit level
(net/unit/lord.T2: his orders land on the units' own landing step, his press on that step only), deployed
lord_script.POST_BACK_M behind my units (unit_battles); enemy_lord_fights - the enemy's lord by its script's own lord
rules (Scripts: sim/scripts.py ai_game / human / nearest ... each moves its lord; SceneEnemy: ai_game's lord rule),
deployed the same (unit_battles, Prep). The tactic level still does not see my lord (his row hidden).
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


def _lord_place(zone, pts, script=False):
    """(x, z, b) lord_back_m behind the middle of pts [(x, z)] along the zone's facing (script: the lord plays - the
    T2 post's distance, lord_script.POST_BACK_M)."""
    back = float(cfg()["lord_back_m"])
    if script:
        from bridge.companion import lord_script as LS
        back = float(LS.POST_BACK_M)
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
            l2 = _unit(cfg()["lords"][s["fac"][2]], *_lord_place(ez, [(u["x"], u["z"]) for u in theirs],
                                                                 bool(cfg().get("enemy_lord_fights"))), general=True)
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
        if (self.side == 1 and cfg().get("lord_script")) or (self.side == 2 and cfg().get("enemy_lord_fights")):
            raise NotImplementedError("stage0.json lord_script / enemy_lord_fights: the chain of play() keeps its lord "
                                      "hidden and holding - the lord keys run in UnitSide / TacticSide and Scripts")
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
    """The enemy's script per battle (opponents.<name>), its lord holding (lord_fights - config enemy_lord_fights when
    None: the lord by the script's own lord rules)."""

    def __init__(self, names, device, side=2, lord_fights=None):
        self.side = side
        self.masks = [(n, torch.tensor([m == n for m in names], device=device)) for n in dict.fromkeys(names)]
        self.lord_fights = bool(cfg().get("enemy_lord_fights")) if lord_fights is None else bool(lord_fights)

    def orders(self, st, j):
        out = None
        for name, m in self.masks:
            o = getattr(opponents, name)(st)
            out = o if out is None else O.merge(out, o, m[:, None].expand(st.B, st.N))
        if self.lord_fights:
            return out
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

    def __init__(self, unit_level=None, gate=True, unit_greedy=False, lord_script=None):
        super().__init__(1, "formula", unit_level=unit_level, gate=gate, unit_greedy=unit_greedy)
        self.lord_script = bool(cfg().get("lord_script")) if lord_script is None else bool(lord_script)
        self.lexe = None

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
        if self.lord_script:
            self._lord_start(st, device)

    def _lord_start(self, st, device):
        """The T2 lord script on my lord (module doc 'The lords'): one lord a battle, his order in force kept."""
        from net.unit import lord as LD
        B, N = st.B, st.N
        assert bool((self.hide.sum(1) == 1).all()), "lord_script: one lord of mine in every battle"
        li = self.hide.float().argmax(1)
        keys = [st.keys[b][int(li[b])] for b in range(B)]
        rank = torch.arange(N, dtype=torch.float64, device=device)[None, :].expand(B, N)
        self.lexe = LD.T2(li, keys, rank, device)
        self.l_int = O.hold(B, N, device)
        self.l_int.kind[:] = O.KEEP
        self.l_force = self.l_int.clone()
        self.l_ab = torch.full((B,), -1, dtype=torch.long, device=device)

    def decide(self, st, raw, K):
        pass

    def tick(self, st, raw):
        pass

    def second(self, st):
        self.lands = self.units.second(st)
        if self.lexe is not None:
            self._lord_second(st)

    def _lord_second(self, st):
        """The lord's decision of this second on the input's columns (lord.lord_inputs_sim) -> self.l_int, self.l_ab
        (a KEEP: his order in force goes on, as the bridge's)."""
        from net.unit import lord as LD
        raw, tgt, a_on, a_cd = LD.lord_inputs_sim(st, self.setup.P, self.units.ubuf[-1])
        d = self.lexe.decide(raw, tgt, a_on, a_cd, self.units.intended.kind, st.t.float(), ~st.done)
        lm, f = self.hide, self.l_force
        new = lm & (d["kind"] != O.KEEP)[:, None]
        col = lambda v: v[:, None].expand_as(lm)                                         # noqa: E731
        self.l_int = O.Orders(kind=torch.where(new, col(d["kind"]), f.kind),
                              x=torch.where(new, col(d["x"]).to(f.x.dtype), f.x),
                              z=torch.where(new, col(d["z"]).to(f.z.dtype), f.z),
                              target=torch.where(new, col(d["target"]), f.target),
                              run=torch.where(new, col(d["run"]), f.run),
                              face=torch.where(new, col(d["face"]).to(f.face.dtype), f.face))
        self.l_ab = d["press"]

    def orders(self, st, j):
        o = self.units.orders(st, j, self.lands, lambda s_: O.hold(st.B, st.N, st.device))
        if self.lexe is None:
            return O.merge(o, O.hold(st.B, st.N, st.device), self.hide)
        # the lord: his decision on the units' landing step, the order in force before it; a press on that step only
        land = (self.lands <= j)[:, None]
        mine = O.Orders(*(torch.where(land, getattr(self.l_int, k), getattr(self.l_force, k)) for k in O.FIELDS))
        mine.ability = torch.where((self.lands == j)[:, None] & self.hide, self.l_ab[:, None].expand_as(self.hide),
                                   torch.full_like(mine.ability, -1))
        return O.merge(o, mine, self.hide)

    def end_second(self):
        super().end_second()
        if self.lexe is not None:
            self.l_force = self.l_int.clone()


class TacticSide(UnitSide):
    """My side of a general battle through the tactic level (tmp/tactic_slice/design_flank_v2.md section 4): one group
    = my units but the lord (his row hidden, holding), its zone by formation (tactic/zfit.zone_for; config
    tactic.json orders.zone_fit off - a line 1 x n on the units' middle, as the pilot) handed once at second 0 through
    lower.Tacticians.apply, no operative - all of it the one side wrapper tactic/side.Side (self.wrap; this class its
    adapter to the simulator's state); every 5 s the level's tick with the player (tactic/pin.py) and
    lower.Units.write, every second the unit's script. The level's own per-tick measures (self.acc [B]):
    unpinned_attacks (attack orders on an enemy not pinned by pin.pinned at the order), two_on_side (enemies with more
    strikers than the cap, or two on one side, after a tick). zones [B] - each battle's zone from its spec ({cx, cz,
    heading, line, rows, cols, worst}: specs.drill_general's, zfit.zone_for on the spec's slot order; None - found here
    as before); seed - the order latency's own generator (lower.Units.gen; None: torch's default one); worn [B] - each
    battle's worn starts {mine, theirs: [share | None]} (scene_battles' units carry men = ceil(men0 x share); here each
    worn unit's hp_abs = hp0 x share and hp = share, its men asserted - review_attack_v23 G.2; morale and fatigue stay
    fresh). fearless [B] - each battle's {mine, theirs: [bool]} (tactic/lanes' copies of the game's lanes): those units
    get leadership and morale + 1e4 at start, as charge_probe.sim_lanes makes its twin fearless. At start: the men of
    each battle, both sides with lords, at most config step.men_max (the 100 ms step)."""

    def __init__(self, player, hygiene=True, unit_level=None, zones=None, seed=None, worn=None, fearless=None,
                 lord_script=None):
        from net.unit import scripts as SC
        super().__init__(SC.Unit() if unit_level is None else unit_level, gate=True, unit_greedy=False,
                         lord_script=lord_script)
        self.player, self.hygiene = player, hygiene
        self.zones, self.seed, self.worn = zones, seed, worn
        self.fearless = fearless

    def start(self, st, P, device, card, unit_net):
        from net.tactic import side as TS
        super().start(st, P, device, card, unit_net)
        if self.seed is not None:
            self.units.gen = torch.Generator(device=device).manual_seed(int(self.seed))
        self.bounds = P.map_half
        u = st.u
        B, N = st.B, st.N
        men_max = int(cfg()["step"]["men_max"])
        assert float(u["men"].sum(1).max()) <= men_max, f"a battle over {men_max} men: the engine's step is 200 ms there"
        if self.worn is not None:
            self._wear(u)
        if self.fearless is not None:
            for b, fl in enumerate(self.fearless):
                for side, base in (("mine", 1), ("theirs", PER_SIDE + 1)):
                    for j, on in enumerate((fl or {}).get(side) or []):
                        if on:
                            u["leadership"][b, base + j] += 1e4
                            u["morale"][b, base + j] += 1e4
        self.mine = (u["side"] == 1) & ~u["lord"] & (u["men"] > 0)
        # the level through the one side wrapper (tactic/side.py), shared with the bridge's companion
        self.wrap = TS.Side(self.player, self.hygiene, device).start(u["x"], u["z"], u["b"], self.mine, self.hide,
                                                                     self.zones)
        self.tac, self.worst = self.wrap.tac, self.wrap.worst
        self.acc = {k: torch.zeros(B, device=device) for k in ("unpinned_attacks", "two_on_side")}

    def _wear(self, u):
        """The worn starts (class doc): my units at slots 1.., theirs at PER_SIDE + 1.. (scene_battles' order)."""
        for b, w in enumerate(self.worn):
            if w is None:
                continue
            for side, base in (("mine", 1), ("theirs", PER_SIDE + 1)):
                for j, share in enumerate(w.get(side) or []):
                    if share is None:
                        continue
                    i = base + j
                    want = math.ceil(float(u["men0"][b, i]) * share)
                    assert int(u["men"][b, i]) == want, (b, i, float(u["men"][b, i]), want)
                    u["hp_abs"][b, i] = u["hp0"][b, i] * share
                    u["hp"][b, i] = share

    def tick(self, st, raw):
        self.wrap.tick(raw, self.setup.card, self.bounds, now=float(st.t.float().max()), sim=st)
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


def play_tactic(specs, player, P, device, hygiene=True, chunk=512, compile=None, watch_cls=None, seeds=None,
                full=False, by_limit=False, on_chunk=None, chunk_args=False, battles=None, enemy=None):
    """(measures {name: [B]}, stats) of stage 0's general battles of the unit level (play_units' battles and enemies)
    with my side through the tactic level (TacticSide) under player() - a factory of a fresh player of tactic/pin.py
    each batch (its memory starts anew); watch_cls - a watch.Watch subclass (watch.PinWatch for the drill's events).
    seeds - an int (chunk k: seed + k) or a list (chunk k: seeds[k]): torch.manual_seed and the order latency's own
    generator (TacticSide seed) at the start of each chunk; None - neither (the caller's global state, as before).
    full - every batch padded to a full chunk on a compiled step (one shape; else to a multiple of PAD); by_limit -
    the specs chunked in the order of their limits (a batch runs to the longest limit in it); a spec's end_s - its
    battle ends there (duel.play end_s: the watch's reads stop, no winner). on_chunk(ii, watch, side) after each chunk:
    ii the specs' indices of its battles 0..len(ii)-1 (the padding after them), its watch and TacticSide - their logs
    and the player. chunk_args: player(ii) and watch_cls(P, general, chain=, ii=) get the chunk's indices. A spec's zone
    (specs.drill_general, tactic/scenes) and worn starts go to TacticSide. battles(batch) - the battles of a batch
    (None: unit_battles; scene_battles for tactic/scenes' specs), enemy(batch, device) - its enemy side (None: Scripts
    of the specs' scripts; SceneEnemy for scenes)."""
    assert all(s["enemy"] != "chain" for s in specs), "the unit level's general battles: script enemies only"
    assert all(len(s["mine"]) <= PER_SIDE - 1 and len(s["theirs"]) <= PER_SIDE - 1 for s in specs), "<= 8 units a side"
    parts, total = {}, {"battles": 0, "wall_s": 0.0}
    order = list(range(len(specs)))
    if by_limit:
        order.sort(key=lambda i: int(specs[i]["limit"]))
    compiled = torch.device(device).type == "cuda" if compile is None else bool(compile)
    for ci, a in enumerate(range(0, len(specs), chunk)):
        ii = order[a:a + chunk]
        pad = (chunk - len(ii)) if (full and compiled) else _pad(len(ii), device, compile)
        batch = [specs[i] for i in ii] + [copy.deepcopy(specs[ii[k % len(ii)]]) for k in range(pad)]
        seed = None if seeds is None else (int(seeds) + ci if isinstance(seeds, int) else int(seeds[ci]))
        if seed is not None:
            torch.manual_seed(seed)
        zones = [s.get("zone") for s in batch]
        worn = [s.get("worn") for s in batch]
        fear = [s.get("fearless") for s in batch]
        mine = TacticSide(player(ii) if chunk_args else player(), hygiene=hygiene,
                          zones=zones if any(z is not None for z in zones) else None, seed=seed,
                          worn=worn if any(w is not None for w in worn) else None,
                          fearless=fear if any(f is not None for f in fear) else None)
        foe = Scripts([s["enemy"] for s in batch], device) if enemy is None else enemy(batch, device)
        wk = {"ii": ii} if chunk_args else {}
        watch = (watch_cls or W.Watch)(P, cfg()["general"], chain=mine, **wk)
        limit = max(int(s["limit"]) for s in batch)
        ends = [s.get("end_s") for s in batch]
        end_s = None if all(e is None for e in ends) else [float(limit if e is None else min(e, limit)) for e in ends]
        rd, stt = DU.play([NAME] * len(batch), (battles or unit_battles)(batch), {1: mine, 2: foe}, None, P, device,
                          limit_s=limit, per_side=PER_SIDE, watch=watch, maps=[s["map"] for s in batch],
                          compile=compile, end_s=end_s)
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
        if on_chunk is not None:
            on_chunk(ii, watch, mine)
    m = {}
    for k, v in parts.items():
        vals = [v[i] for i in range(len(specs))]
        if vals and isinstance(vals[0], list):           # (series of batches of other limits: the last value kept)
            T = max(len(x) for x in vals)
            vals = [x + x[-1:] * (T - len(x)) for x in vals]
        m[k] = torch.tensor(vals)
    total["wall_s"] = round(total["wall_s"], 1)
    return m, total


def scene_battles(specs):
    """The battles of tactic/scenes' specs: every unit at the spec's place (places_mine, places_theirs), a worn unit
    with men = ceil(men0 x share) (sim/scenario: hp_abs follows; TacticSide sets hp and checks), the lords at the
    spec's lords; an enemy's frontage width_theirs[j] m where the spec gives one (drill T0-2's target, sim/scenario's
    width)."""
    from net.tactic import scenes as SC
    out = []
    for s in specs:
        side = {}
        for k, (keys, places, worn) in ((1, (s["mine"], s["places_mine"], s["worn"]["mine"])),
                                         (2, (s["theirs"], s["places_theirs"], s["worn"]["theirs"]))):
            us = []
            wd = s.get("width_theirs") if k == 2 else None
            for j, (key, p, w) in enumerate(zip(keys, places, worn)):
                u = _unit(key, *p, width=None if not wd else wd[j])
                if w is not None:
                    u["men"] = math.ceil(SC._men0(key) * w)
                us.append(u)
            lord = _unit(cfg()["lords"][s["fac"][k]], *s["lords"][k], general=True)
            side[k] = [lord] + us
        out.append({1: side[1], 2: side[2], "factions": dict(s["factions"])})
    return out


class SceneEnemy(DU.Script):
    """The enemy of tactic/scenes' specs, each unit by its plan (plan_theirs; slots PER_SIDE + 1.., my unit k at slot
    1 + k): at - attack my unit k (it routs, dies or leaves: nearest; with stay - hold where it is); nearest; point - move to (x, z) and stand there
    (with a waypoint (wx, wz): there first, until within WAYPOINT_M);
    halt - attack k until the edge gap to the nearest standing unit of the other side (its lord out) is gap m or less,
    then stand (for the rest of the battle: the halt must not end next to a unit it passes); from_s - stand
    until that second first; script - the battle script name (sim/scripts) gives its orders; run - at a run, else a
    walk. The lord and units without a plan hold."""
    KINDS = ("hold", "at", "nearest", "point", "halt", "script")
    WAYPOINT_M = 15.0          # (a waypoint is passed within this, m: a unit's half depth and the step's slack)

    def __init__(self, specs, device):
        from sim.params import load
        self.side = 2
        self.spacing = load().sim["formation"]["spacing_m"]
        B, N = len(specs), 2 * PER_SIDE
        f = lambda v, dt=torch.float32: torch.full((B, N), v, dtype=dt, device=device)      # noqa: E731
        self.kind, self.k, self.run = f(0, torch.long), f(-1, torch.long), f(False, torch.bool)
        self.x, self.z, self.gap, self.from_s = f(0.0), f(0.0), f(0.0), f(0.0)
        self.wx, self.wz, self.wp = f(0.0), f(0.0), f(False, torch.bool)
        for b, s in enumerate(specs):
            for j, p in enumerate(s["plan_theirs"]):
                i = PER_SIDE + 1 + j
                self.kind[b, i] = self.KINDS.index(p["kind"])
                self.k[b, i] = 1 + int(p["k"]) if p.get("k") is not None else -1
                self.run[b, i] = bool(p.get("run", True))
                self.x[b, i], self.z[b, i] = float(p.get("x", 0.0)), float(p.get("z", 0.0))
                self.gap[b, i], self.from_s[b, i] = float(p.get("gap", 0.0)), float(p.get("from_s", 0.0))
                if p.get("wx") is not None:
                    self.wx[b, i], self.wz[b, i], self.wp[b, i] = float(p["wx"]), float(p["wz"]), True
        self.halted = f(False, torch.bool)
        self.stay = f(False, torch.bool)
        for b, s in enumerate(specs):
            for j, p in enumerate(s["plan_theirs"]):
                self.stay[b, PER_SIDE + 1 + j] = bool(p.get("stay", False))
        self.lord_fights = bool(cfg().get("enemy_lord_fights"))
        self.scripts = {}
        for b, s in enumerate(specs):
            for j, p in enumerate(s["plan_theirs"]):
                if p["kind"] == "script":
                    m = self.scripts.setdefault(p["name"], f(False, torch.bool))
                    m[b, PER_SIDE + 1 + j] = True

    def orders(self, st, j):
        from sim import geometry
        from sim import scripts as opponents
        u = st.u
        tk = self.k.clamp(min=0)
        gone = u["r"] | (u["men"] <= 0) | u["gone"]
        t_ok = (self.k >= 0) & ~gone.gather(1, tk)
        if bool((self.kind == 4).any()):
            front, depth = geometry.dims(u, self.spacing)
            other = (u["side"] > 0) & (u["side"] != self.side) & ~u["lord"] & ~gone              # [B, N]
            gp = geometry.rect_gap(u["x"], u["z"], u["b"], front, depth)
            g = torch.where(other[:, None, :], gp, torch.full_like(gp, float("inf"))).amin(2)
            self.halted |= (self.kind == 4) & t_ok & (g <= self.gap)
        att = (self.kind == 1) | ((self.kind == 4) & ~self.halted)
        o = O.hold(st.B, st.N, st.device)
        a = O.hold(st.B, st.N, st.device)
        a.kind[:] = O.ATTACK
        a.target = tk
        o = O.merge(o, a, att & t_ok)
        o = O.merge(o, opponents.nearest(st), (self.kind == 2) | (att & ~t_ok & ~self.stay))
        mv = O.hold(st.B, st.N, st.device)
        mv.kind[:] = O.MOVE
        self.wp &= torch.sqrt((u["x"] - self.wx) ** 2 + (u["z"] - self.wz) ** 2) > self.WAYPOINT_M
        mv.x, mv.z = torch.where(self.wp, self.wx, self.x), torch.where(self.wp, self.wz, self.z)
        o = O.merge(o, mv, self.kind == 3)
        for name, m in self.scripts.items():
            o = O.merge(o, getattr(opponents, name)(st), m)
        o.run = torch.where(self.kind > 0, self.run, o.run)
        t = st.t.float()
        wait = (t[:, None] if t.dim() else t) < self.from_s
        o = O.merge(o, O.hold(st.B, st.N, st.device), wait | (self.kind == 0) | (self.halted & (self.kind == 4)))
        lord = u["lord"] & (u["side"] == self.side)
        if self.lord_fights:                      # (config enemy_lord_fights: the lord by ai_game's lord rule)
            return O.merge(o, opponents.ai_game(st), lord)
        return O.merge(o, O.hold(st.B, st.N, st.device), lord)


def unit_battles(specs):
    """The battles of specs for play_units: my units in rows across my zone (specs.rows, as the enemy's; centred; a
    spec's perm - slot i stands at the rows' place perm[i], specs.drill_general's shuffled slots), the enemy at its
    places, each lord lord_back_m behind its side (a lord that plays - config lord_script, enemy_lord_fights - the T2
    post's distance)."""
    from net.stage0 import specs as SP
    out = []
    ls, ef = bool(cfg().get("lord_script")), bool(cfg().get("enemy_lord_fights"))
    for s in specs:
        mz, ez = s["mine_zone"], s["enemy_zone"]
        pl = SP.rows(mz, len(s["mine"]))
        if s.get("perm") is not None:
            pl = [pl[p] for p in s["perm"]]
        mine = [_unit(k, *p) for k, p in zip(s["mine"], pl)]
        theirs = [_unit(k, *p) for k, p in zip(s["theirs"], s["places_theirs"])]
        l1 = _unit(cfg()["lords"][s["fac"][1]], *_lord_place(mz, [(u["x"], u["z"]) for u in mine], ls), general=True)
        l2 = _unit(cfg()["lords"][s["fac"][2]], *_lord_place(ez, [(u["x"], u["z"]) for u in theirs], ef), general=True)
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
