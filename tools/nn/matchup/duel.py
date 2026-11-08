"""A batch of matchup battles played to their end: each side a chain of our levels or a script of the simulator.

    Chain(side, op, tactic, command)   the operative (op: 'formula' / a teacher.MODES mode / an OperativeNet) every 15 s,
                the tacticians (tactic: None - the formula, or a TacticNet) every 5 s, the frozen unit network every
                second (operative/lower.py) - the same loop as operative/env.play, for either side
    Script(side, name)    opponents.<name>(st) (ai_like, nearest, hold_shoot, human): its orders for that side's units
    play(name, battles, sides, unit_net, ...)   -> readings {winner [B] (1, 2, 0 none), lost [B, 2] (the share of the
                side's gold lost), lord [B, 2] (alive), t_end [B], contact_t [B]} and stats
"""
import time

import torch

from tools import config as project
from tools.nn.matchup import armies as AR
from tools.nn.matchup import cfg
from tools.nn.operative import env as OEN
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as OIN
from tools.nn.operative import lower as LW
from tools.nn.operative import net as ONN
from tools.nn.operative import teacher as OTC
from tools.nn.operative import cfg as ocfg
from tools.nn.sim import battle, scenario, terrain
from tools.nn.sim import orders as O
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.train import opponents, reward
from tools.nn.unit import train as UT

_MAPS = {}


def map_id(name):
    if name not in _MAPS:
        _MAPS[name] = terrain.load_npz(project.ROOT / "build" / "maps" / name / "heights.npz")
    return _MAPS[name]


class _SideSetup:
    """What lower.Units needs of a side (as cases.Setup)."""

    def __init__(self, st, P, device, side, card):
        self.st, self.P, self.device = st, P, device
        u = st.u
        self.own = u["side"] == side
        self.shooter = (u["range"] > 0) & self.own
        self.card = card


class Chain:
    def __init__(self, side, op="formula", tactic=None, command="hold", op_greedy=True, targets=True, unit_greedy=True, still=True,
                 aim=True, groups0=None):
        self.aim = aim
        self.groups0 = groups0          # the strategist's groups (strategist/battle.placed): the operative's last groups
        self.side, self.op, self.tactic, self.command_word, self.op_greedy = side, op, tactic, command, op_greedy
        self.targets = targets
        self.unit_greedy = unit_greedy
        self.still = still

    def start(self, st, P, device, card, unit_net):
        B, N = st.B, st.N
        self.setup = _SideSetup(st, P, device, self.side, card)
        self.G = int(ocfg()["groups"])
        self.groups = GR.Groups.empty(B, N, self.G, device) if self.groups0 is None else self.groups0.padded(self.G)
        self._parcels0 = self.groups0 is not None
        self.tac = LW.Tacticians(B, N, self.G, device, net=self.tactic)
        self.units = LW.Units(self.setup, unit_net, unit_greedy=self.unit_greedy, targets=self.targets,
                              still=self.still, aim=self.aim)
        self.sidev = torch.full((B,), self.side, dtype=torch.long, device=device)
        self.command = DC.order_vec(B, self.command_word, "any", device)
        self.destroy = torch.full((B,), self.command_word == "destroy", dtype=torch.bool, device=device)
        self.axes = None
        self.stats = {"parcels": 0, "parcel_errors": 0, "errors": [], "groups": 0.0, "decisions": 0, "reforms": 0.0,
                      "moves": 0.0, "plan_changes": 0.0, "windows": 0.0}
        self._useed = None
        self.by_window = []                     # plan changes (all battles) of each window from the second on

    def _plan_changes(self, new, live):
        """Windows of a real change of plan by the bridge's (raised) threshold: >= 3 own units in another group than
        the window before, or a group's target changed and held for the next window too (counted then)."""
        own = self.setup.own
        mem = new.member
        us = torch.where(mem >= 0, new.seed.gather(1, mem.clamp(min=0)), torch.full_like(mem, -1))
        if self._useed is not None:
            m0, u1, u2 = self._useed
            alive = own & (mem >= 0) & (m0 >= 0)
            regroup = ((mem != m0) & alive).sum(1) >= 3
            held = (alive & (u1 >= 0) & (u2 >= 0) & (u1 != u2) & (us == u1)).any(1)
            ch = (regroup | held) & live
            self.stats["plan_changes"] += float(ch.sum())
            self.by_window.append(float(ch.sum()))
            self.stats["windows"] += float(live.sum())
            u2 = u1
        else:
            u2 = us
        self._useed = (mem.clone(), us, u2)

    def decide(self, st, raw, K):
        if self.axes is None:
            self.axes = FR.Axes.of(raw, self.sidev, None)
        if self._parcels0:
            # second 0 with the strategist's groups: their parcels first, as the game's chain (companion/chain.start)
            self._parcels0 = False
            self.tac.apply(self.groups, self.groups.on, torch.zeros_like(self.groups.on), raw)
        self.groups.drop_dead(raw)
        inp = OIN.build(raw, self.setup.card, self.sidev, self.axes, self.groups, self.command, K=K)
        ctx = ONN.ctx_of(inp, raw, self.axes, self.groups)
        if isinstance(self.op, str):
            out = OTC.play_formula(inp, ctx, self.destroy, self.op)
        else:
            with torch.no_grad():
                out, _ = self.op(inp, ctx, greedy=self.op_greedy)
        live = ~st.done
        new = out["new"]
        new.age = torch.where(new.on, new.age + int(ocfg()["decide_s"]), new.age)
        parcels, idx = self.tac.apply(new, out["task"] & live[:, None], out["moved"], raw)
        if idx.numel():
            errs = PC.check(raw, parcels, OEN.field_dict(self.axes, None))
            self.stats["parcels"] += len(errs)
            bad = [e for e in errs if e]
            self.stats["parcel_errors"] += len(bad)
            if bad and len(self.stats["errors"]) < 5:
                self.stats["errors"].append(bad[0])
        self.stats["groups"] += float(new.on.sum())
        self.stats["reforms"] += float((out["reformed"] & live[:, None]).sum())
        self.stats["moves"] += float((out["task"] & out["moved"] & live[:, None]).sum())
        self.stats["decisions"] += int(live.sum())
        self._plan_changes(new, live)
        self.groups = new
        return inp, ctx, out

    def tick(self, st, raw):
        self.tac.tick(raw, self.setup.card, None)
        self.units.write(self.tac)

    def second(self, st):
        self.lands = self.units.second(st)

    def orders(self, st, j):
        hold = lambda s: O.hold(st.B, st.N, st.device)                       # noqa: E731
        return self.units.orders(st, j, self.lands, hold)

    def end_second(self):
        self.units.end_second()


class Script:
    def __init__(self, side, name="ai_like"):
        self.side, self.name = side, name
        self.fn = getattr(opponents, name)

    def start(self, *a):
        pass

    def orders(self, st, j):
        return self.fn(st)


def build(name, battles, device, P, per_side=20):
    """(st, card) of the battles (each {1: units, 2: units}: armies.base / augment) on the matchup's map; name - a
    matchup, or one per battle (its factions); a battle with its own 'factions' {1: .., 2: ..} takes those."""
    names = name if isinstance(name, (list, tuple)) else [name] * len(battles)
    st = scenario.build([AR.description(n, b) for n, b in zip(names, battles)], P, device=device, per_side=per_side)
    terrain.set_map(st, map_id(cfg()["map"]))
    return st, UT.cards(st, device).float()


@torch.no_grad()
def play(name, battles, sides, unit_net, P=None, device="cpu", limit_s=None, duty=None, per_side=20, watch=None):
    """sides {1: Chain | Script, 2: ...}; battles [{1: units, 2: units}]; watch(sec, st) after every second (a
    reading of its own). Returns (readings, stats)."""
    from tools.nn.sim.params import load
    P = P or load()
    st, card = build(name, battles, device, P, per_side)
    B, N = st.B, st.N
    limit = int(limit_s or cfg()["limit_s"])
    every, t_every = int(ocfg()["decide_s"]), int(tcfg()["decide_s"])
    steps = round(1.0 / P.dt)
    K = OEN.rows_of(N)
    for s, sd in sides.items():
        if isinstance(sd, Chain):
            sd.start(st, P, device, card, unit_net)
    seer = TR.Seer(P)
    sim_step = battle.stepper(device)
    m0 = reward.measure(st)
    side = st.u["side"]
    over = torch.zeros(B, dtype=torch.bool, device=device)
    winner = torch.zeros(B, dtype=torch.long, device=device)
    t_end = torch.full((B,), float(limit), device=device)
    contact = torch.full((B,), -1.0, device=device)
    t0 = time.time()
    sec = 0
    # the chain's habits (the bridge agent's base in the game, 07.11): in its area, the way walked, the shooters firing,
    # the shooters / artillery / lord in melee, the orders
    u = st.u
    caste = card[..., 36:49].argmax(-1)
    is_shooter = (u["range"] > 0) & (side == 1)
    is_lord = u["lord"] & (side == 1)
    acc = {k: torch.zeros(B, device=device) for k in ("inside", "alive_s", "walk", "fire_s", "shoot_s", "melee_shoot",
                                                     "melee_lord", "order_changes")}
    px, pz = u["x"].clone(), u["z"].clone()
    last_kind = None
    for sec in range(limit + 1):
        if bool(over.all()):
            break
        raw = seer.raw(st)
        for s, sd in sides.items():
            if isinstance(sd, Chain):
                if sec % every == 0:
                    sd.decide(st, raw, K)
                if sec % t_every == 0:
                    sd.tick(st, raw)
                sd.second(st)
        for j in range(steps):
            o = {s: sd.orders(st, j) for s, sd in sides.items()}
            orders = O.merge(o[1], o[2], side == 2)
            sim_step(st, orders, P)
            u = st.u
            out = u["r"] | (u["men"] <= 0) | u["gone"] | (u["side"] <= 0)
            gone1 = (out | (side != 1)).all(1)
            gone2 = (out | (side != 2)).all(1)
            new = ~over & (gone1 | gone2)
            winner = torch.where(new, torch.where(gone2 & ~gone1, 1, torch.where(gone1 & ~gone2, 2, 0)), winner)
            t_end = torch.where(new, st.t.float(), t_end)
            over = over | new
            fight = (u["m"] & (side > 0)).any(1)
            contact = torch.where((contact < 0) & fight, st.t.float(), contact)
            st.done = over.clone()
        u = st.u
        live1 = (side == 1) & (u["men"] > 0) & ~u["gone"] & ~over[:, None]
        acc["walk"] += (torch.sqrt((u["x"] - px) ** 2 + (u["z"] - pz) ** 2) * live1).sum(1)
        px, pz = u["x"].clone(), u["z"].clone()
        acc["alive_s"] += live1.float().sum(1)
        sh = is_shooter & live1 & ~u["r"]
        acc["shoot_s"] += sh.float().sum(1)
        acc["fire_s"] += (sh & u["fire"]).float().sum(1)
        acc["melee_shoot"] += (sh & u["m"]).float().sum(1)
        acc["melee_lord"] += (is_lord & live1 & u["m"]).float().sum(1)
        c1 = sides.get(1)
        if isinstance(c1, Chain):
            from tools.nn.unit import decode as Dc
            a = c1.units.area
            ins = Dc.signed_dist(u["x"], u["z"], {k: a[k] for k in ("x", "z", "b", "front", "depth")}) >= 0
            acc["inside"] += (ins & live1).float().sum(1)
            k = c1.units.intended.kind
            if last_kind is not None:
                acc["order_changes"] += ((k != last_kind) & (k != O.KEEP) & live1).float().sum(1)
            last_kind = k.clone()
        for sd in sides.values():
            if isinstance(sd, Chain):
                sd.end_second()
        if watch is not None:
            watch(sec, st)
        if duty is not None:
            duty.end()
    m1 = reward.measure(st)
    lost = torch.stack([m0[:, 0, 3] - m1[:, 0, 3], m0[:, 1, 3] - m1[:, 1, 3]], 1)
    lord = m1[:, :, 2]
    # no side out by the limit: the one that lost less gold (as the game's timeout to the defender: here a measure)
    stats = {"battles": B, "seconds": sec, "wall_s": round(time.time() - t0, 1)}
    for s, sd in sides.items():
        if isinstance(sd, Chain):
            stats[f"side{s}"] = {k: v for k, v in sd.stats.items()}
            stats[f"side{s}"]["groups_mean"] = round(sd.stats["groups"] / max(sd.stats["decisions"], 1), 2)
    n_own = (side == 1).float().sum(1).clamp(min=1)
    habits = {"inside": acc["inside"] / acc["alive_s"].clamp(min=1), "walk_m": acc["walk"] / n_own,
              "fire_share": acc["fire_s"] / acc["shoot_s"].clamp(min=1),
              "shooter_melee_share": acc["melee_shoot"] / acc["shoot_s"].clamp(min=1),
              "lord_melee_s": acc["melee_lord"], "order_changes_per_unit": acc["order_changes"] / n_own}
    return {"winner": winner, "lost": lost, "lord": lord, "t_end": t_end, "contact_t": contact, **habits}, stats


def summary(rd):
    w = rd["winner"]
    return {"win": round(float((w == 1).float().mean()), 3), "loss": round(float((w == 2).float().mean()), 3),
            "draw": round(float((w == 0).float().mean()), 3),
            "lost_own": round(float(rd["lost"][:, 0].mean()), 3), "lost_enemy": round(float(rd["lost"][:, 1].mean()), 3),
            "lord_alive": round(float(rd["lord"][:, 0].mean()), 3),
            "t_end": round(float(rd["t_end"].mean()), 1), "contact_t": round(float(rd["contact_t"].mean()), 1),
            **{k: round(float(rd[k].mean()), 3) for k in ("inside", "walk_m", "fire_share", "shooter_melee_share",
                                                         "lord_melee_s", "order_changes_per_unit") if k in rd}}


def enemy(name):
    """The enemy side (2) by name: a script of the simulator (opponents.<name>), or 'chain' - our formula chain on
    the enemy's side under the command 'destroy' (the army's formula, the target to the units, the stand-still rule)."""
    if name == "chain":
        return Chain(2, "formula", command="destroy")
    if name == "chain_hold":
        return Chain(2, "formula", command="hold")
    return Script(2, name)
