"""The strategist's battle in the simulator (plan.md "Передача оперативнику"): the strategist's one decision -> every
unit's place (deploy.py) -> the built state -> the groups as the operative's last groups -> second 0: the tacticians get
the strategist's parcels (each through tactic/parcel.check), the operative its first decision; then the operative every
15 s (its network run5 by default, or its formula), the tacticians every 5 s (the formula, or a TacticNet), the frozen
unit network every second - the operative's own loop (operative/env.play) without its PPO bookkeeping.

    decide(pre, net=None, mode='formula', greedy=False)   (out, inp, ctx): the strategist's choice (net.choose)
    play(pre, out, inp, unit_net, op_net=None, ...)       (readings, stats) of the battle to its end
    measures(pre, rd)                                     {case: [B] float} plan.md "Мера"

The readings: wide - per enemy three its first contact with my army and 'hit' (my shooters touched by it within soon_s
of that contact); attack / hold - the contact, 'the infantry between' (tactic/drill1.Watch), 'advanced' (my centre went more than
advance_m towards the enemy's zone), the share of my army's health lost.
"""
import time
from types import SimpleNamespace

import torch

from tools.nn.operative import cfg as ocfg
from tools.nn.operative import env as OEN
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as OIN
from tools.nn.operative import lower as LW
from tools.nn.operative import net as ONN
from tools.nn.operative import teacher as OTC
from tools.nn.sim import battle, scenario
from tools.nn.strategist import cases as CS
from tools.nn.strategist import cfg
from tools.nn.strategist import deploy as DP
from tools.nn.strategist import inputs as IN
from tools.nn.strategist import net as NN
from tools.nn.strategist import teacher as TC
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import drill1 as DR
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.train import reward


def inputs_of(pre, raw=None):
    raw = pre.raw0 if raw is None else raw
    axes = ZS.axes(pre.mine_zone, pre.enemy_zone, raw, pre.st0.bounds)
    inp = IN.build(raw, pre.card, pre.side, axes, pre.mine_zone, pre.enemy_zone, pre.command, bounds=pre.st0.bounds)
    return inp, NN.Ctx(axes, pre.mine_zone, pre.enemy_zone)


@torch.no_grad()
def decide(pre, net=None, mode="formula", greedy=False):
    inp, ctx = inputs_of(pre)
    if net is None:
        out = TC.play_formula(inp, ctx, mode)
    else:
        out = net(inp, ctx, greedy=greedy)
    return out, inp, ctx


def slots_of(plan, inp, N):
    """[B, G, N] bool the groups' members by slot."""
    mem = plan["members"]
    B, G, K = mem.shape
    out = torch.zeros((B, G, N + 1), dtype=torch.bool, device=mem.device)
    sl = torch.where(inp["slot"] >= 0, inp["slot"], torch.full_like(inp["slot"], N))
    out.scatter_(2, sl[:, None, :].expand(B, G, K), mem)
    return out[..., :N]


def placed(pre, out, inp):
    """(st, groups0, places, members [B, G, N]): the built state with my units where the strategist put them."""
    plan = out["plan"]
    N = pre.st0.N
    mem = slots_of(plan, inp, N)
    pl = DP.places(mem, pre.shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], pre.width0)
    armies = []
    for b, s in enumerate(pre.sp):
        mine = []
        for i in range(len(s["mine"])):
            mine.append({"x": float(pl["x"][b, i]), "z": float(pl["z"][b, i]), "b": float(pl["b"][b, i]),
                         "width": float(pl["width"][b, i])})
        armies.append(CS.army(s, mine))
    st = scenario.build(armies, pre.P, device=pre.device, per_side=CS.H)
    G = mem.shape[1]
    B = st.B
    g0 = GR.Groups.empty(B, N, G, pre.device)
    g0.member = pl["group"]
    g0.on = plan["on"].clone()
    g0.cx, g0.cz, g0.heading = plan["cx"].clone(), plan["cz"].clone(), plan["heading"].clone()
    g0.line = plan["line"].clone()
    g0.verb = torch.full((B, G), DC.VERBS.index("hold"), dtype=torch.long, device=pre.device)
    g0.pace = torch.full((B, G), DC.PACES.index("any"), dtype=torch.long, device=pre.device)
    g0.n0 = plan["n"].clone()
    return st, g0, pl, mem


class Readings:
    def __init__(self, pre, st, spacing):
        dev = pre.device
        B, N = st.B, st.N
        self.pre = pre
        z = lambda: torch.zeros(B, dtype=torch.bool, device=dev)                  # noqa: E731
        self.t3 = torch.full((B, 2), -1.0, device=dev)
        self.hit3 = torch.zeros((B, 2), dtype=torch.bool, device=dev)
        self.has3 = torch.stack([(pre.three == k).any(1) for k in (0, 1)], 1)
        self.watch = DR.Watch(st, pre.own, pre.shooter, spacing, dev)
        self.advanced = z()
        u = st.u
        own = pre.own.float()
        k = own.sum(1).clamp(min=1)
        self.x0, self.z0 = (u["x"] * own).sum(1) / k, (u["z"] * own).sum(1) / k
        ex = pre.enemy_zone["x"] - pre.mine_zone["x"]
        ez = pre.enemy_zone["z"] - pre.mine_zone["z"]
        d = torch.sqrt(ex * ex + ez * ez).clamp(min=1e-6)
        self.ux, self.uz = ex / d, ez / d
        self.advance_m = float(ocfg()["reward"]["advance_m"])
        self.soon = float(tcfg()["reward"]["soon_s"])
        self.hp0 = reward.measure(st)[:, 0, 0]
        self.flank3 = torch.zeros((B, 2), dtype=torch.bool, device=dev)
        self.lost_c = torch.full((B,), float("nan"), device=dev)
        self.front_deg = float(cfg()["cases"]["wide"]["front_deg"])

    def step(self, st, was, dt, area=None):
        self.watch.step(st, was, dt, area)
        u = st.u
        B, N = st.B, st.N
        live = ~was
        own, sh = self.pre.own, self.pre.shooter
        hs = u["hit_sec"].view(B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)                         # [B, i, j]
        t = st.t.float()
        for k in (0, 1):
            th = self.pre.three == k
            tk = touch & own[:, :, None] & th[:, None, :]
            new = tk.flatten(1).any(1) & (self.t3[:, k] < 0) & live
            self.t3[:, k] = torch.where(new, t, self.t3[:, k])
            on_sh = (tk & sh[:, :, None]).flatten(1).any(1) & live
            soon = (self.t3[:, k] >= 0) & (t <= self.t3[:, k] + self.soon)
            self.hit3[:, k] = self.hit3[:, k] | (on_sh & soon)
            # met from the flank: a touching pair whose enemy stands more than front_deg off my unit's facing
            if bool((tk.flatten(1).any(1) & soon & live).any()):
                way = torch.rad2deg(torch.atan2(u["x"][:, None, :] - u["x"][:, :, None],
                                                u["z"][:, None, :] - u["z"][:, :, None]))
                off = (torch.remainder(way - u["b"][:, :, None] + 180.0, 360.0) - 180.0).abs()
                side = (tk & (off > self.front_deg)).flatten(1).any(1) & soon & live
                self.flank3[:, k] = self.flank3[:, k] | side
        first = self.watch.contact & torch.isnan(self.lost_c)
        if bool(first.any()):
            self.lost_c = torch.where(first, (self.hp0 - reward.measure(st)[:, 0, 0]).clamp(min=0), self.lost_c)
        alive = own & (u["men"] > 0) & ~u["gone"]
        w = alive.float()
        k = w.sum(1).clamp(min=1)
        mx, mz = (u["x"] * w).sum(1) / k - self.x0, (u["z"] * w).sum(1) / k - self.z0
        self.advanced = self.advanced | (((mx * self.ux + mz * self.uz) > self.advance_m) & live)

    @property
    def contact(self):
        return self.watch.contact

    def all_contact(self):
        return ((self.t3 >= 0) | ~self.has3).all(1)


class Ender:
    def __init__(self, pre, st):
        self.pre = pre
        self.over = torch.zeros(st.B, dtype=torch.bool, device=st.device)
        self.why = torch.zeros(st.B, dtype=torch.long, device=st.device)
        self.fight = float(cfg()["cases"]["fight_s"])
        self.wide = pre.case == CS.CASES.index("wide")

    def after_step(self, st, rd):
        u = st.u
        own = self.pre.own
        out = u["r"] | (u["men"] <= 0) | u["gone"] | (u["side"] <= 0)
        we = (out | ~own).all(1)
        they = (out | own).all(1)
        late = st.t >= self.pre.limit
        last = torch.where(self.wide, rd.t3.amax(1), rd.watch.t_contact)
        got = torch.where(self.wide, rd.all_contact(), rd.contact)
        fought = got & (st.t >= last + self.fight)
        new = ~self.over & (we | they | late | fought)
        self.why = torch.where(new, torch.where(we, 1, torch.where(they, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we, 2, torch.where(they, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new


@torch.no_grad()
def play(pre, out, inp, unit_net, op_net=None, op_mode="formula", tactic_net=None, op_greedy=False, check=True):
    """The battle (module doc). op_net None: the operative's formula (op_mode, operative/teacher.MODES)."""
    P, dev = pre.P, pre.device
    st, groups, pl, mem = placed(pre, out, inp)
    B, N = st.B, st.N
    G = groups.G
    oc = ocfg()
    every, t_every = int(oc["decide_s"]), int(tcfg()["decide_s"])
    limit = int(float(pre.limit.max()))
    steps = round(1.0 / P.dt)
    spacing = P.sim["formation"]["spacing_m"]
    errs = DP.errors(st, pre.own, groups.member, pre.mine_zone, spacing) + out["plan"]["no_cell"]
    from tools.nn.unit import train as UT
    order = torch.zeros((B, len(DC.VERBS) + len(DC.PACES)), device=dev)
    order[torch.arange(B, device=dev), out["plan"]["verb"]] = 1.0
    order[torch.arange(B, device=dev), len(DC.VERBS) + out["plan"]["pace"]] = 1.0
    setup = SimpleNamespace(st=st, P=P, device=dev, own=pre.own, shooter=pre.shooter, card=UT.cards(st, dev).float(),
                            side=pre.side, command=order, destroy=out["plan"]["verb"] == DC.VERBS.index("destroy"))
    seer = TR.Seer(P)
    tac = LW.Tacticians(B, N, G, dev, net=tactic_net)
    units = LW.Units(setup, unit_net)
    rd = Readings(pre, st, spacing)
    ender = Ender(pre, st)
    script = CS.Script(pre.stands)
    sim_step = battle.stepper(dev)
    m0 = reward.measure(st)
    K = OEN.rows_of(N)
    axes = None
    stats = {"parcels": 0, "parcel_errors": 0, "errors": [], "deploy_errors": int(errs.sum()),
             "deploy_error_battles": int((errs > 0).sum()), "op_reforms": 0.0, "op_decisions": 0}
    sec = 0
    t0 = time.time()

    def checked(parcels, idx, raw_t):
        if check and idx.numel():
            e = PC.check(raw_t, parcels, OEN.field_dict(axes, None))
            stats["parcels"] += len(e)
            bad = [x for x in e if x]
            stats["parcel_errors"] += len(bad)
            if bad and len(stats["errors"]) < 5:
                stats["errors"].append(bad[0])

    for sec in range(limit + 1):
        if bool(st.done.all()):
            break
        raw_t = seer.raw(st)
        if sec == 0:
            axes = FR.Axes.of(raw_t, pre.side, st.bounds)
            parcels, idx = tac.apply(groups, groups.on, torch.zeros_like(groups.on), raw_t)
            checked(parcels, idx, raw_t)
        if sec % every == 0:
            groups.drop_dead(raw_t)
            oinp = OIN.build(raw_t, setup.card, pre.side, axes, groups, order, bounds=st.bounds, K=K)
            octx = ONN.ctx_of(oinp, raw_t, axes, groups)
            if op_net is not None:
                o, _ = op_net(oinp, octx, greedy=op_greedy)
            else:
                o = OTC.play_formula(oinp, octx, setup.destroy, op_mode)
            live = ~st.done & octx.own.any(1)
            stats["op_decisions"] += int(live.sum())
            stats["op_reforms"] += float((o["reformed"] & live[:, None]).sum())
            new = o["new"]
            new.age = torch.where(new.on, new.age + every, new.age)
            parcels, idx = tac.apply(new, o["task"] & live[:, None], o["moved"], raw_t)
            checked(parcels, idx, raw_t)
            groups = new
        if sec % t_every == 0:
            tac.tick(raw_t, setup.card, st.bounds)
            units.write(tac)
        lands = units.second(st)
        for j in range(steps):
            was = st.done.clone()
            o_ = units.orders(st, j, lands, script)
            sim_step(st, o_, P)
            st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            rd.step(st, was, P.dt, units.area)
            ender.after_step(st, rd)
        units.end_second()
    m1 = reward.measure(st)
    trade = (m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])
    lost = (m0[:, 0, 0] - m1[:, 0, 0]).clamp(min=0)
    stats.update({"battles": B, "seconds": sec, "s": round(time.time() - t0, 1), "trade": round(float(trade.mean()), 4),
                  "op_reforms_per_battle": round(stats["op_reforms"] / B, 3),
                  "groups_start": round(float(groups_start(out).mean()), 3), "tactic_fell": tac.fell})
    readings = {"contact": rd.contact, "between": rd.watch.between, "hit": rd.watch.hit, "t3": rd.t3, "hit3": rd.hit3,
                "flank3": rd.flank3, "t_contact": rd.watch.t_contact, "lost_contact": rd.lost_c,
                "has3": rd.has3, "advanced": rd.advanced, "lost": lost, "trade": trade, "why": ender.why,
                "deploy_errors": errs, "places": pl}
    return readings, stats


def groups_start(out):
    return out["plan"]["on"].sum(1).float()


def measures(pre, rd):
    """{case: [B] float}: wide - the share of the enemy's threes whose first contact did not touch my shooters within
    soon_s; attack - a contact by the battle's end; hold - no contact and at most loss_share of my health lost (how far
    the operative walks meanwhile is its own measure: plan.md "Мера")."""
    ok3 = (rd["t3"] >= 0) & ~rd["hit3"] & rd["has3"]
    wide = ok3.float().sum(1) / rd["has3"].float().sum(1).clamp(min=1)
    keep = float(cfg()["cases"]["loss_share"])
    return {"wide": wide, "attack": rd["contact"].float(), "hold": (~rd["contact"] & (rd["lost"] <= keep)).float()}


def measure_of(pre, rd):
    """[B] each battle's own case's measure."""
    m = measures(pre, rd)
    out = torch.zeros_like(m["wide"])
    for i, cs in enumerate(CS.CASES):
        out = torch.where(pre.case == i, m[cs], out)
    return out
