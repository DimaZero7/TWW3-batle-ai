"""The chain in the game (companion --chain): the operative -> the tacticians -> the unit network command our side after
the strategist's deployment, by the same code and on the same clocks as the simulator's chain (tools/nn/strategist/
battle.py play, tools/nn/operative/lower.py; CLAUDE.md "Вход сети один в один", build/diag_g49/bridge_parity.md section
"цепочка").

Every state document (one a second of battle, the bridge's decide_ms 1000):

    raw        the tactician's raw state of the document (tactic_input.TacticSeer: the unit level's columns + the effects,
               the buttons) - one call a document, the operative and the tacticians read the same snapshot, as the
               simulator's TR.Seer
    second 0   the strategist's groups (deploy.Strategist.plans of this batch: members, zone centre, turn, form, n0, the
               starting verb and pace) become the operative's last groups and the tacticians' first parcels, as
               strategist/battle.placed + play's second 0; without a strategist plan: no groups, the command from the
               task (the operative's own battle, operative/env.play)
    every 15 s the operative (its network run8 by default, or its formula): drop_dead, the input (operative/inputs.build),
               the choice, the parcels to the tacticians (each through tactic/parcel.check)
    every 5 s  the tacticians (the formula tactic/teacher.drive by default, or a TacticNet): the frames -> the units' areas
    every 1 s  the unit network (GameUnits, the simulator's lower.Units on the raw state): the view of each own unit,
               its choice -> the simulator's orders (unit/decode.to_orders) -> the leash (decode.Leash, once a second
               here, every simulator step there) -> the bridge's orders (exchange.orders_list: move / attack / hold with
               the facing and the width, the run)

The clocks are the battle's seconds (the document's t, rounded): a level is due when its next second has come (a
document missed by the companion does not skip a decision, it comes on the next one). In one second the operative goes
first, then the tacticians, then the units - the simulator's order.

Not as in the simulator (bridge_parity.md "цепочка"): the leash is checked once a second (the simulator: every step,
0.1 s); an order lands when the game reads it (the simulator: the unit level's cadence); the velocity of the unit level
is the centre's shift (the known difference of unit_input).
"""
import time
from types import SimpleNamespace

import torch

from tools.nn.companion import exchange
from tools.nn.companion import tactic_input as TI
from tools.nn.companion import unit_input as UI
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import env as OEN
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as OIN
from tools.nn.operative import lower as LW
from tools.nn.operative import net as ONN
from tools.nn.operative import teacher as OTC
from tools.nn.sim import orders as O
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import train as UT
from tools.nn.unit import view as V

STATIC_U = ("men0", "wmin_k", "sp_h", "sp_v", "arc")


def static_u(keys, params, device="cpu"):
    """{men0, wmin_k, sp_h, sp_v, arc} [1, N] of the units by key (params.static; unknown: zeros) - what lower.Units
    reads of the simulator's state for the width limits (decode.width_limits), the rule 'a shooter shoots its
    target' (its own fire arc) and the rule 'stay in the area' (eng0: a gun's engines, 0 not artillery)."""
    out = {k: torch.zeros((1, len(keys)), device=device) for k in STATIC_U + ("eng0", "direct")}
    for i, key in enumerate(keys):
        if key in params.units:
            s = params.static(key)
            for k in STATIC_U:
                out[k][0, i] = float(s.get(k, 0.0))
            # an artillery unit's guns (sim/scenario.build: only a unit with engines has them)
            if params.units[key].get("engines"):
                out["eng0"][0, i] = float(s.get("eng0", 0.0))
            out["direct"][0, i] = float(bool(s.get("direct", False)))
    return out


def leash_state(raw_u):
    """The fields of the simulator's state the leash reads (decode.Leash.apply), from the unit level's raw [1, N, RAW]."""
    I = V.I
    g = lambda k: raw_u[..., I[k]]                     # noqa: E731
    return SimpleNamespace(u={"m": g("m") > 0, "r": g("r") > 0, "men": g("men"), "x": g("x"), "z": g("z"),
                              "vx": g("vx"), "vz": g("vz"), "range": g("range")})


class GameUnits(LW.Units):
    """lower.Units on the bridge's raw state: the same area, arrival, pace, memory, views, choice and decoder; the
    simulator's state only gave it V.raw (here the seer's columns) and who still decides (alive, not routing). The
    translator's rules of the simulator's matchup chain (lower.Units targets / still / aim; matchup/duel.Chain) read
    the state the game gives each second (refresh): places, men, alive, seen, routing, range, in melee, the target a
    shooter aims at (the document's 't'), its fire arc (the key's)."""

    def __init__(self, raw_u, keys, card, params, unit_net, unit_greedy=False, targets=False, still=False, aim=False,
                 lords=None):
        I = V.I
        own = raw_u[..., I["side"]] == 1
        u = {"x": raw_u[..., I["x"]].clone(), "z": raw_u[..., I["z"]].clone(), "b": raw_u[..., I["b"]].clone(),
             **static_u(keys, params, raw_u.device),
             "lord": torch.zeros_like(own) if lords is None else torch.as_tensor(lords, device=raw_u.device).bool()}
        st = SimpleNamespace(B=1, N=raw_u.shape[1], u=u, device=raw_u.device)
        setup = SimpleNamespace(st=st, P=params, device=raw_u.device, own=own,
                                shooter=(raw_u[..., I["range"]] > 0) & own, card=card)
        super().__init__(setup, unit_net, unit_greedy, targets=targets, still=still, aim=aim)
        # the order in force of each unit (what the bridge was given last: the leashed orders, a KEEP keeps it) - the
        # simulator's u order_kind / order_target, read by the rule 'a shooter keeps its target' (sticky_rule)
        self.order_kind = torch.full(own.shape, O.HOLD, dtype=torch.long, device=raw_u.device)
        self.order_target = torch.full(own.shape, -1, dtype=torch.long, device=raw_u.device)
        self.refresh(raw_u)

    def refresh(self, raw_u, aim_tgt=None, contact=None, order=None):
        """The state the rules read (class doc) from the unit level's raw [1, N, RAW]; aim_tgt [1, N] long each unit's
        target slot (-1 none), contact [1, N] bool in melee contact (None: the raw's melee flag); order (kind,
        target) [1, N] the orders in force (None: the ones this chain gave - the bridge does not give an equal one
        again)."""
        I = V.I
        u = self.setup.st.u
        # (vx, vz: the game's centre shift over the last second - the simulator's velocity; run: the passport's
        # running speed - the rule 'no attack on a fast side-stepping target', lower.Units.fast_rule)
        for k in ("x", "z", "b", "range", "front", "vx", "vz", "run"):
            u[k] = raw_u[..., I[k]]
        u["side"] = raw_u[..., I["side"]].long()
        if order is not None:
            self.order_kind, self.order_target = order[0].clone(), order[1].clone()
        u["order_kind"], u["order_target"] = self.order_kind, self.order_target
        u["men"] = raw_u[..., I["men"]]
        u["gone"] = ~(raw_u[..., I["alive"]] > 0)
        u["vis"] = raw_u[..., I["vis"]] > 0
        u["r"] = raw_u[..., I["r"]] > 0
        m = (raw_u[..., I["m"]] > 0) if contact is None else contact
        u["contact_s"] = m.float()
        none = torch.full(raw_u.shape[:2], -1, dtype=torch.long, device=raw_u.device)
        u["aim_tgt"] = none if aim_tgt is None else aim_tgt

    def second_raw(self, raw_u):
        """lower.Units.second with raw_u for V.raw(st) (module doc): the intended orders of this second."""
        I = V.I
        self.ubuf = torch.cat([self.ubuf[1:], raw_u[None]])
        own = self.setup.own
        ar = {k: self.area[k] for k in ("x", "z", "b", "front", "depth")}
        inside = Dc.signed_dist(raw_u[..., I["x"]], raw_u[..., I["z"]], ar) >= 0
        self.arrived = self.arrived | (inside & own)
        B, N = raw_u.shape[:2]
        bi, ii = self.bi, self.ii
        live = (raw_u[bi, ii, I["alive"]] > 0) & ~(raw_u[bi, ii, I["r"]] > 0)
        bl, il = bi[live], ii[live]
        intended = O.hold(B, N, raw_u.device)
        intended.kind[:] = O.KEEP
        if bl.numel():
            idx = torch.full_like(bl, V.T)
            prio = self.prio if self.targets else None
            v = UT.views(self.ubuf, idx, bl, il, self.setup.card, self.area, LW.UNIT_CFG, wlim=self.wlim, prio=prio)
            ord_u = Nt.order_of(raw_u[bl, il], {k: a[bl, il] for k, a in self.area.items() if k != "on"},
                                self.pace[bl, il], self.arrived[bl, il],
                                has_target=None if prio is None else v["prio"] >= 0)
            act = Nt.sample(self.net(v, ord_u, v["cell_m"]), greedy=self.greedy)
            intended = Dc.to_orders(act, v, raw_u, v["cell_m"], B, N, wlim=self.wlim)
            st = self.setup.st
            if self.still:
                intended = self.stand_still(intended, st, inside)
            if self.aim:
                intended = self.stay_rule(intended, st, inside)
                intended = self.aim_rule(intended, st, inside)
                intended = self.fast_rule(intended, st)
                intended = self.sticky_rule(intended, st, inside)
                intended = self.fast_rule(intended, st)
            self.n += int(bl.numel())
        self.intended = intended
        return intended, inside

    def leashed(self, raw_u):
        """The intended orders with the leash's replacements (decode.Leash.apply on the raw state; updates leash.on)."""
        out = self.leash.apply(leash_state(raw_u), self.intended, self.area, self.setup.own)
        self.in_force = self.intended
        given = self.setup.own & (out.kind != O.KEEP)
        self.order_kind = torch.where(given, out.kind, self.order_kind)
        self.order_target = torch.where(given, out.target, self.order_target)
        return out

    def line_closed(self, st, t):
        """lower.Units.line_closed with each unit's formation front from the raw state (its width and men there: the
        simulator's geometry.dims)."""
        u = st.u
        tc = t.clamp(min=0)
        vx, vz = u["x"].gather(1, tc) - u["x"], u["z"].gather(1, tc) - u["z"]
        L = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-6)
        ex, ez = vx / L, vz / L
        dx = u["x"][:, None, :] - u["x"][:, :, None]
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        along = dx * ex[:, :, None] + dz * ez[:, :, None]
        across = (dx * ez[:, :, None] - dz * ex[:, :, None]).abs()
        side = u["side"]
        mine = (side[:, :, None] == side[:, None, :]) & (u["men"] > 0)[:, None, :] & ~u["r"][:, None, :]
        eye = torch.eye(st.N, dtype=torch.bool, device=u["x"].device)[None]
        band = (along > 10.0) & (along < L[:, :, None] - 10.0) & (across < u["front"][:, None, :] / 2)
        return (band & mine & ~eye).any(2) & (t >= 0)


def groups_of_plan(plan, names, N, G, device="cpu"):
    """The strategist's groups as the operative's last groups (strategist/battle.placed): plan {names: {unit: group},
    on, cx, cz, heading, line, n [Gp]} -> GR.Groups [1, G] (verb hold, pace any, age 0; the plan's Gp groups - the
    strategist's 6 - in the first places, the rest empty: the operative's config groups since 07.11)."""
    G, Gp = int(G), min(int(G), len(plan["on"]))
    g = GR.Groups.empty(1, N, Gp, device)
    for i, n in enumerate(names):
        k = plan["group"].get(n, -1)
        if k is not None and k >= 0:
            g.member[0, i] = int(k)
    t = lambda k, dt=torch.float32: torch.as_tensor(plan[k], dtype=dt, device=device)[None, :Gp]  # noqa: E731
    g.on = t("on", torch.bool).clone()
    g.cx, g.cz, g.heading = t("cx").clone(), t("cz").clone(), t("heading").clone()
    g.line = t("line", torch.bool).clone()
    g.n0 = t("n", torch.long).clone()
    g.verb = torch.full((1, Gp), DC.VERBS.index("hold"), dtype=torch.long, device=device)
    g.pace = torch.full((1, Gp), DC.PACES.index("any"), dtype=torch.long, device=device)
    return g.padded(G)


def command_of(verb, pace, device="cpu"):
    """[1, VERBS + PACES] the operative's command (strategist/battle.play's order)."""
    o = torch.zeros((1, len(DC.VERBS) + len(DC.PACES)), device=device)
    o[0, DC.VERBS.index(verb)] = 1.0
    o[0, len(DC.VERBS) + DC.PACES.index(pace)] = 1.0
    return o


class Chain:
    """Our side in the game under the chain (module doc). plans: {batch: strategist plan} filled by the strategist's
    deployment (deploy.Strategist.plans), None or no plan of the batch: no starting groups; task: the command without a
    strategist ('hold' | 'destroy')."""
    MODE = "chain"            # the answer's 'mode' line (exchange.orders_text): the bridge's chain rules

    def __init__(self, unit_net, op_net=None, tactic_net=None, op_greedy=False, tactic_greedy=False, unit_greedy=False,
                 plans=None, task="hold", params=None, device="cpu", map_root=UI.MAPS, allow_flat=True, own=1,
                 check=True, op_mode="formula", targets=True, still=True, aim=True):
        assert own == 1, "the chain commands side 1"
        self.seer = TI.TacticSeer(map_root, device, own, params, allow_flat)
        self.P = self.seer.P
        self.bounds = self.P.map_half
        self.unit_net, self.op_net, self.tactic_net = unit_net, op_net, tactic_net
        self.op_greedy, self.tactic_greedy, self.unit_greedy = op_greedy, tactic_greedy, unit_greedy
        self.op_mode = op_mode
        # the translator's rules of the simulator's matchup chain (matchup/duel.Chain; lower.Units; 07-08.11,
        # build/diag_m1/night.md 6, 11, 18): the group's target to its units, a unit in its area stands, a shooter in
        # its area attacks its target within range, shooters / guns stay in their area (stay_rule, 08.11; False: the plain
        # chain of strategist/battle.play)
        self.targets, self.still, self.aim = targets, still, aim
        self.op_decide = None          # callable (inp, ctx, raw) -> the walk's output: the operative's decision by
        #                                someone else (the labelling over the game: the person, tools/label/game.py)
        self.plans = plans if plans is not None else {}
        self.task, self.device, self.own, self.check = task, device, own, check
        self.every, self.t_every = int(ocfg()["decide_s"]), int(tcfg()["decide_s"])
        self.G = int(ocfg()["groups"])
        self.batch = None
        self.plans_log = []            # this decision's operative records (the log)
        self.times = {}                # this decision's ms by level
        self.stats = {}

    # -- a new battle
    def start(self, batch, names, keys, card, r, lords=None):
        """A new battle: names, keys [N] in the raw state's order, card [1, N, F] (passport.table), r its first raw;
        lords [1, N] bool the sides' generals (None: by name, '<side>_lord' - the arena's script names)."""
        dev = self.device
        self.batch, self.names, self.keys, self.card = batch, list(names), list(keys), card
        N = r.shape[1]
        self.side = torch.full((1,), self.own, dtype=torch.long, device=dev)
        self.axes = FR.Axes.of(r, self.side, self.bounds)
        plan = self.plans.get(batch)
        if plan is not None:
            self.groups = groups_of_plan(plan, self.names, N, self.G, dev)
            verb, pace = plan["verb"], plan["pace"]
        else:
            self.groups = GR.Groups.empty(1, N, self.G, dev)
            verb, pace = self.task, "any"
        self.verb, self.pace_cmd = verb, pace
        self.order = command_of(verb, pace, dev)
        self.destroy = torch.tensor([verb == "destroy"], device=dev)
        self.K = OEN.rows_of(N)
        self.tac = LW.Tacticians(1, N, self.G, dev, net=self.tactic_net, greedy=self.tactic_greedy)
        base = r[..., :TR.BASE]
        if lords is None:
            lords = [[n.endswith("_lord") for n in self.names]]
        self.units = GameUnits(base, self.keys, card, self.P, self.unit_net, self.unit_greedy, targets=self.targets,
                               still=self.still, aim=self.aim, lords=lords)
        self.next_op = self.next_tac = None
        self.stats = {"parcels": 0, "parcel_errors": 0, "errors": [], "op_decisions": 0, "op_reforms": 0.0,
                      "tactic_ticks": 0, "unit_seconds": 0, "from_strategist": plan is not None}
        if plan is not None:
            parcels, idx = self.tac.apply(self.groups, self.groups.on, torch.zeros_like(self.groups.on), r)
            self._checked(parcels, idx, r)

    def _checked(self, parcels, idx, r):
        if self.check and idx.numel():
            e = PC.check(r, parcels, OEN.field_dict(self.axes, None))
            self.stats["parcels"] += len(e)
            bad = [x for x in e if x]
            self.stats["parcel_errors"] += len(bad)
            if bad and len(self.stats["errors"]) < 5:
                self.stats["errors"].append(bad[0])

    # -- the levels
    def operative(self, r):
        g = self.groups
        g.drop_dead(r)
        inp = OIN.build(r, self.card, self.side, self.axes, g, self.order, bounds=self.bounds, K=self.K)
        ctx = ONN.ctx_of(inp, r, self.axes, g)
        if self.op_decide is not None:
            o = self.op_decide(inp, ctx, r)          # a person in the operative's seat (tools/label/game.py)
        elif self.op_net is not None:
            o, _ = self.op_net(inp, ctx, greedy=self.op_greedy)
        else:
            o = OTC.play_formula(inp, ctx, self.destroy, self.op_mode)
        live = ctx.own.any(1)
        self.stats["op_decisions"] += int(live.sum())
        self.stats["op_reforms"] += float((o["reformed"] & live[:, None]).sum())
        new = o["new"]
        new.age = torch.where(new.on, new.age + self.every, new.age)
        task = o["task"] & live[:, None]
        parcels, idx = self.tac.apply(new, task, o["moved"], r)
        self._checked(parcels, idx, r)
        self.groups = new
        return self.describe_groups(new, task, o["moved"], o["reformed"])

    def describe_groups(self, g, task, moved, reformed):
        out = []
        for k in range(g.G):
            if not bool(g.on[0, k]):
                continue
            out.append({"g": k, "units": [n for i, n in enumerate(self.names) if int(g.member[0, i]) == k],
                        "target": [n for i, n in enumerate(self.names) if bool(g.target[0, k, i])],
                        "cx": round(float(g.cx[0, k]), 1), "cz": round(float(g.cz[0, k]), 1),
                        "heading": round(float(g.heading[0, k]), 1), "line": bool(g.line[0, k]),
                        "n0": int(g.n0[0, k]), "verb": DC.VERBS[int(g.verb[0, k])], "pace": DC.PACES[int(g.pace[0, k])],
                        "task": bool(task[0, k]), "moved": bool(moved[0, k]), "reformed": bool(reformed[0, k])})
        return out

    @torch.no_grad()
    def step(self, r, sec, raw_u=None, aim_tgt=None, contact=None, order=None):
        """One second of the chain on the raw state r [1, N, RAW] at battle second sec: the levels due (module doc) ->
        (the orders with the leash, the units' intended orders before it, inside [1, N] each unit in its area). raw_u:
        the unit level's raw (tests: the simulator's own, its leash column included); None - r's base columns with
        this chain's leash. aim_tgt [1, N] each unit's target slot (the document's 't'), contact [1, N] in melee
        contact (None: the raw's melee flag) - GameUnits.refresh."""
        t1 = time.perf_counter()
        if raw_u is None:
            raw_u = r[..., :TR.BASE].clone()
            raw_u[..., V.I["leash"]] = self.units.leash.on.float()
        self.units.refresh(raw_u, aim_tgt, contact, order)
        if self.next_op is None:
            self.next_op, self.next_tac = sec, sec
        self.plans_log = []
        if sec >= self.next_op:
            self.plans_log = self.operative(r)
            while self.next_op <= sec:
                self.next_op += self.every
        t2 = time.perf_counter()
        self.times["op_ms"] = (t2 - t1) * 1000
        if sec >= self.next_tac:
            self.tac.tick(r, self.card, self.bounds)
            self.units.write(self.tac)
            self.stats["tactic_ticks"] += 1
            while self.next_tac <= sec:
                self.next_tac += self.t_every
        t3 = time.perf_counter()
        self.times["tac_ms"] = (t3 - t2) * 1000
        intended, inside = self.units.second_raw(raw_u)
        o = self.units.leashed(raw_u)
        self.stats["unit_seconds"] += 1
        self.inside, self.sec, self.raw_u = inside, sec, raw_u
        self.times["unit_ms"] = (time.perf_counter() - t3) * 1000
        return o, intended, inside

    @torch.no_grad()
    def decide(self, doc):
        """-> (orders list, think ms, abilities []) for one state document (loop.Seat's decide)."""
        t0 = time.perf_counter()
        self.times = {}
        r = self.seer.raw(doc)
        if doc["batch"] != self.batch:
            self.start(doc["batch"], [u["n"] for u in doc["units"]], [u.get("key") or "" for u in doc["units"]],
                       self.seer.card, r)
        self.times["raw_ms"] = (time.perf_counter() - t0) * 1000
        sec = int(round(float(doc.get("t", 0)) / 1000.0))
        slot = {n: i for i, n in enumerate(self.names)}
        aim = torch.tensor([[slot.get(u.get("t"), -1) if u.get("t") else -1 for u in doc["units"]]],
                           dtype=torch.long, device=self.device)
        o, _, _ = self.step(r, sec, aim_tgt=aim)
        t4 = time.perf_counter()
        col = lambda f: getattr(o, f)[0].cpu().numpy()        # noqa: E731
        side = [int(u["side"]) for u in doc["units"]]
        out = exchange.orders_list(self.names, side, col("kind"), col("x"), col("z"), col("target"), col("run"),
                                   own=self.own, face=col("face"), width=col("width"))
        I = V.I
        raw_u = self.raw_u
        ctrl = {n for i, n in enumerate(self.names)
                if side[i] == self.own and float(raw_u[0, i, I["alive"]]) > 0 and not float(raw_u[0, i, I["r"]]) > 0}
        for x in out:
            if x["unit"] not in ctrl:
                x["out"] = True
        self.times["unit_ms"] += (time.perf_counter() - t4) * 1000
        return out, (time.perf_counter() - t0) * 1000, []

    def log_extra(self):
        """The decision's chain fields for the companion's JSON line: ms by level, the operative's groups when it decided,
        each own unit's area and whether it stands inside, the leash, the tacticians' fallback count."""
        a = self.units.area
        own = self.units.setup.own[0]
        areas = {n: [round(float(a[k][0, i]), 1) for k in ("x", "z", "b", "front", "depth")]
                 for i, n in enumerate(self.names) if bool(own[i])}
        inside = {n: bool(self.inside[0, i]) for i, n in enumerate(self.names) if bool(own[i])}
        leash = [n for i, n in enumerate(self.names) if bool(self.units.leash.on[0, i])]
        row = {"sec": self.sec, "ms": {k: round(v, 2) for k, v in self.times.items()}, "areas": areas,
               "inside": inside, "tactic_fell": self.tac.fell, "parcel_errors": self.stats["parcel_errors"]}
        if leash:
            row["leash"] = leash
        if self.plans_log:
            row["operative"] = self.plans_log
        return row


def describe(rec):
    """A line of an operative decision's groups."""
    return " | ".join(f"g{g['g']} {g['verb']} [{', '.join(n.removeprefix('own_') for n in g['units'])}]"
                      f" -> [{', '.join(n.removeprefix('enemy_') for n in g['target'])}] at ({g['cx']:.0f}, {g['cz']:.0f})"
                      f"{' task' if g['task'] else ''}{' moved' if g['moved'] else ''}{' reformed' if g['reformed'] else ''}"
                      for g in rec)
