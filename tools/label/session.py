"""One labelling battle in the simulator (torch): the operative's battle of operative/env.play, second by second, with
the person in place of one level; it stops on that level's decisions and goes on when the person has decided.

A second, in env.play's order:
    begin       the tactician's raw state (tactic/raw.Seer, once a second); the field's axes at the first second
    operative   every op_every s (config/nn/operative.json decide_s, 15): the dead leave their groups, the input
                (operative/inputs.build), the decision - the person's (pause), the formula (operative/teacher) or the
                operative network (best choices); the parcels to the tacticians (operative/lower.Tacticians.apply),
                each checked by tactic/parcel.check
    tactic      every t_every s (config/nn/tactic.json decide_s, 5): the tacticians' frames - the formula, the
                tactician's network, or the person (pause; human.HumanTactic in the network's seat: the network's input
                is built by Tacticians.tick itself); the frames to the units
    sim         the unit network's second (operative/lower.Units: the latency, the leash), the enemy's script, the
                simulator's steps; the end: a side has nothing standing (dead, routing, gone) or the limit
The person's level decides with its own period (config) or the one the person set at the start (session.json says it).
The strategist (strategist.py) decides once, before second 0: the battle is built again with my units on its places,
its groups go to the tacticians and to the operative (the formula or run5) at second 0, the rest the battle plays.

    Session(matchup, level, period, upper, lower, nets)
    advance(seconds)   runs until the person's decision, the end, or that many seconds; returns the snapshots
    pending            None, or {"level", "sec", ...}: the decision waited for (what the level sees, the state in
                       force as an edit, the formula's hint)
    preview(edit)      the person's edit checked and played by the walk, nothing changed (errors or the result)
    decide(edit)       the same, then written to the record and put in force; the second goes on
    stop()             the end now (the person ends it)
"""
import time

import torch

from tools import config as project
from tools.label import cfg
from tools.label import human as HU
from tools.label import matchup as MU
from tools.label import record as RC
from tools.label import view as VW
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as IN
from tools.nn.operative import lower as LW
from tools.nn.operative import net as NN
from tools.nn.operative import teacher as OTC
from tools.nn.sim import battle, scenario, terrain
from tools.nn.sim.params import load
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import inputs as TIN
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.tactic import teacher as TTC
from tools.nn.tactic import uzone as U
from tools.nn.train import reward
from tools.nn.unit import train as UT

LEVELS = ("operative", "tactic", "strategist")
_MAP_IDS = {}


def map_id(name):
    """The simulator's id of a captured map (build/maps/<name>/heights.npz with its layers; registered once)."""
    if name not in _MAP_IDS:
        _MAP_IDS[name] = terrain.load_npz(project.ROOT / "build" / "maps" / name / "heights.npz")
    return _MAP_IDS[name]


class Battle:
    """What operative/lower.Units needs of a battle (as cases.Setup): st, P, device, own, shooter, card; and the side,
    the command from above, the enemy's script, the limit."""

    def __init__(self, m, P, device):
        self.P, self.device = P, device
        army = MU.army(m)
        self.slots, H = MU.order(m)
        self.st = scenario.build([army], P, device=device, per_side=H)
        if m.get("map", "flat") != "flat":
            terrain.set_map(self.st, map_id(m["map"]))
        u = self.st.u
        self.own = u["side"] == 1
        self.shooter = (u["range"] > 0) & self.own
        self.card = UT.cards(self.st, device).float()
        self.side = torch.ones(1, dtype=torch.long, device=device)
        cmd = m.get("command") or {}
        self.verb, self.pace = cmd.get("verb", "hold"), cmd.get("pace", "any")
        self.command = DC.order_vec(1, self.verb, self.pace, device)
        self.destroy = torch.tensor([self.verb == "destroy"], device=device)
        self.enemy = m.get("enemy", "attack")
        self.stands = torch.tensor([self.enemy == "hold"], device=device)
        self.limit = float(m.get("limit_s", cfg()["limit_s"]))


class Script:
    """The enemy's script (config enemy_modes): attack / attack_run - every standing unit at the nearest standing
    enemy (sim/replay.nearest_attack), at a walk / at a run, looked at again every look_s (tactic drill1's 5 s); hold -
    in place (it shoots what comes into range)."""

    def __init__(self, mode):
        from tools.nn.tactic import cfg as tc
        self.mode = mode
        self.look = float(tc()["drill1"]["look_s"])
        self.next = -1.0
        self.o = None

    def __call__(self, st):
        from tools.nn.sim import orders as O
        from tools.nn.sim import replay
        if self.mode == "hold":
            return O.hold(st.B, st.N, st.device)
        t = float(st.t.max())
        if self.o is None or t >= self.next - 1e-6:
            self.o = replay.nearest_attack(st, run=self.mode == "attack_run")
            self.next = t + self.look
        return self.o


class Session:
    def __init__(self, m, level="operative", period=None, upper="formula", lower="formula", nets=None, device="cpu",
                 name=None, out=True, out_root=None):
        errs = MU.check(m)
        if errs:
            raise HU.Refused(errs)
        if level not in LEVELS:
            raise HU.Refused([f"уровень {level!r}: один из {LEVELS}"])
        nets = nets or {}
        self.m, self.level, self.device = m, level, device
        self.upper, self.lower = upper, lower
        self.nets = nets
        self.P = load()
        self.G = int(ocfg()["groups"])
        self.op_every = int(period) if level == "operative" and period else int(ocfg()["decide_s"])
        self.t_every = int(period) if level == "tactic" and period else int(tcfg()["decide_s"])
        self.period = {"operative": self.op_every, "tactic": self.t_every, "strategist": 0}[level]
        self.op_net = None
        if (level == "tactic" and upper == "net") or (level == "strategist" and lower == "net"):
            self.op_net = nets.get("operative")
            if self.op_net is None:
                raise HU.Refused(["сеть оперативника не загружена"])
        self.htac = HU.HumanTactic() if level == "tactic" else None
        self.tnet = self.htac if level == "tactic" else (nets.get("tactic") if level == "operative" and lower == "net"
                                                         else None)
        if level == "operative" and lower == "net" and self.tnet is None:
            raise HU.Refused(["сеть тактика не загружена"])
        self.sec = 0
        self.stage = "begin"
        self.pending = None
        self.over = False
        self.result = None
        self.why = None
        self.raw = None
        self.op = None
        self.start_groups = None
        self.n_decisions = 0
        self.parcels = 0
        self.parcel_errors = []
        self.replay = []
        self._build(m)
        st = self.b.st
        info = {"level": level, "period_s": self.period,
                "period_config_s": {"operative": int(ocfg()["decide_s"]), "tactic": int(tcfg()["decide_s"]),
                                    "strategist": 0}[level], "operative_every_s": self.op_every,
                "tactic_every_s": self.t_every,
                "upper": upper if level == "tactic" else ("none" if level == "strategist" else "matchup command"),
                "lower": "person" if level == "tactic" else lower, "unit": "frozen unit network (greedy)",
                "nets": {k: cfg()["nets"][k] for k in ("operative", "tactic", "unit")}, "slots": self.b.slots,
                "keys": list(st.keys[0]), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.info = info
        self.rec = None
        if out:
            self.folder = RC.new_folder(name or m.get("name") or "battle", out_root)
            self.rec = RC.Recorder(self.folder, m, info)
        else:
            self.folder = None
        if level == "strategist":
            from tools.label import strategist as ST
            self.strat = ST.Level(self)
            self.pending = self.strat.pending()
            self.stage = "strategist"

    def _build(self, m):
        """The battle's pieces from a matchup (again after the strategist's deployment)."""
        P, device = self.P, self.device
        self.b = Battle(m, P, device)
        st = self.b.st
        N = st.N
        self.tac = LW.Tacticians(1, N, self.G, device, net=self.tnet, greedy=True)
        self.units = LW.Units(self.b, self.nets["unit"], unit_greedy=True)
        self.script = Script(self.b.enemy)
        self.seer = TR.Seer(P)
        self.groups = GR.Groups.empty(1, N, self.G, device)
        self.axes = None
        self.sim_step = battle.stepper(device)
        self.m0 = reward.measure(st)
        self.K = min(int(ocfg()["rows"]), N)
        self.Kt = min(int(tcfg()["rows"]), N)
        self.last_snap = VW.snapshot(self)
        self.replay = [self.last_snap]

    # ------------------------------------------------------------ the loop

    def advance(self, seconds=None):
        snaps = []
        while not self.over and self.pending is None:
            if seconds is not None and len(snaps) >= seconds and self.stage == "begin":
                break
            s = self._stage()
            if s is not None:
                snaps.append(s)
        return snaps

    def _stage(self):
        st = self.b.st
        if self.stage == "begin":
            if bool(st.done.all()) or self.sec > self.b.limit:
                self._finish("time" if self.sec > self.b.limit else self.why or "end")
                return None
            self.raw = self.seer.raw(st)
            if self.axes is None:
                self.axes = FR.Axes.of(self.raw, self.b.side, st.bounds)
            self.stage = "operative"
            return None
        if self.stage == "operative":
            if self.start_groups is not None:
                # the strategist's groups: the tacticians get their parcels at second 0, the operative sees them as its
                # last groups (tools/nn/strategist/battle.play's order)
                g0, self.start_groups = self.start_groups, None
                parcels, idx = self.tac.apply(g0, g0.on, torch.zeros_like(g0.on), self.raw)
                self._check_parcels(parcels, idx)
                self.groups = g0
            if self.sec % self.op_every == 0:
                self.groups.drop_dead(self.raw)
                inp = IN.build(self.raw, self.b.card, self.b.side, self.axes, self.groups, self.b.command,
                               bounds=st.bounds, K=self.K)
                ctx = NN.ctx_of(inp, self.raw, self.axes, self.groups)
                self.op = (inp, ctx)
                live = bool((~st.done & ctx.own.any(1))[0])
                if self.level == "operative" and live:
                    self.pending = self._op_pending(inp, ctx)
                    return None
                if self.op_net is not None:
                    out, _ = self.op_net(inp, ctx, greedy=True)
                else:
                    out = OTC.play_formula(inp, ctx, self.b.destroy)
                self._op_apply(out, live)
            self.stage = "tactic"
            return None
        if self.stage == "tactic":
            if self.sec % self.t_every == 0:
                if self.level == "tactic" and not bool(st.done[0]):
                    p = self._tac_pending()
                    if p is not None:
                        self.pending = p
                        return None
                self.tac.tick(self.raw, self.b.card, st.bounds)
                self.units.write(self.tac)
            self.stage = "sim"
            return None
        # the simulator's second
        P = self.P
        lands = self.units.second(st)
        for j in range(round(1.0 / P.dt)):
            o = self.units.orders(st, j, lands, self.script)
            self.sim_step(st, o, P)
            st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            self._ender(st)
        self.units.end_second()
        self.sec += 1
        self.stage = "begin"
        self.last_snap = VW.snapshot(self)
        self.replay.append(self.last_snap)
        return self.last_snap

    def _ender(self, st):
        u = st.u
        own = self.b.own
        out = u["r"] | (u["men"] <= 0) | u["gone"] | (u["side"] <= 0)
        we = bool((out | ~own).all(1)[0])
        they = bool((out | own).all(1)[0])
        late = bool(st.t[0] >= self.b.limit)
        if not bool(st.done[0]) and (we or they or late):
            self.why = "we are out" if we else ("the enemy is out" if they else "time")
            st.winner = torch.tensor([2 if we else (1 if they else 0)], device=st.device)
            st.done = torch.ones(1, dtype=torch.bool, device=st.device)

    def _op_apply(self, out, live):
        st = self.b.st
        new = out["new"]
        lv = torch.tensor([live], device=self.device)
        new.age = torch.where(new.on, new.age + self.op_every, new.age)
        parcels, idx = self.tac.apply(new, out["task"] & lv[:, None], out["moved"], self.raw)
        self._check_parcels(parcels, idx)
        self.groups = new

    def _check_parcels(self, parcels, idx):
        if idx.numel():
            f = self.axes.field
            errs = PC.check(self.raw, parcels, {"cx": f["x"], "cz": f["z"], "b": f["b"], "front": f["front"],
                                                "depth": f["depth"]})
            self.parcels += len(errs)
            self.parcel_errors += [e for e in errs if e]

    # ------------------------------------------------------------ the operative's decision

    def _seen(self, ctx):
        return [s for s, o, e in zip(*HU.op_rows(ctx)) if o or e]

    def _op_pending(self, inp, ctx):
        hint = OTC.play_formula(inp, ctx, self.b.destroy)
        hr = HU.op_result(hint, ctx, self.axes)
        now = HU.op_state(self.groups, ctx)
        first = not bool(self.groups.on.any())
        return {"level": "operative", "sec": self.sec, "seen": self._seen(ctx),
                "command": {"verb": self.b.verb, "pace": self.b.pace},
                "state": now, "edit": hr["edit"] if first else now, "edit_from": "hint" if first else "keep",
                "hint": hr["edit"], "hint_groups": hr["groups"], "can_keep": not first,
                "cells": {"cell_m": float(ocfg()["cell_m"]), "C": int(ocfg()["cells"]),
                          "axes": [float(self.axes.cx[0]), float(self.axes.cz[0]), float(self.axes.heading[0])]},
                "sectors": int(ocfg()["sectors"]), "verbs_open": ocfg()["verbs_open"],
                "paces_open": ocfg()["paces_open"], "max_group": int(tcfg()["max_group"]), "_hint_act": hint["act"]}

    def _op_try(self, edit):
        inp, ctx = self.op
        act, plan, filled = HU.op_act(edit, ctx, self.axes)
        out, errs = HU.op_walk(inp, ctx, act, plan)
        _, _, perrs = HU.op_parcels(out, self.raw, self.axes)
        errs += perrs
        return act, out, filled, errs

    # ------------------------------------------------------------ the tactician's decision

    def _tac_rows(self):
        idx = (self.tac.active & self.tac.mine.any(1)).nonzero()[:, 0]
        if idx.numel() == 0:
            return None
        tac = self.tac
        tiles = tac._keep(self.raw, idx)          # (idempotent within a second: tick calls it again)
        p = tac.parcel(idx)
        my = DC.my_slots(p)
        r = idx.shape[0]
        k = torch.arange(r, device=self.device)[:, None]
        s0 = my.clamp(min=0)
        prev = {n: tac.fr[n][idx][k, s0] for n in LW.FR_KEYS}
        prev["on"] = tac.fr_on[idx][k, s0] & (my >= 0)
        has_prev = bool(prev["on"].any())
        prev = prev if has_prev else None
        areas = U.world(prev, tiles) if prev is not None else None
        inp = TIN.build(self.raw, self.b.card, p, tiles, areas, bounds=self.b.st.bounds, rows=self.Kt)
        ctx = DC.ctx_of(inp, p, tiles, prev)
        rows_g = [int(g) for g in p.group.tolist()]
        return {"idx": idx, "tiles": tiles, "parcel": p, "my": my, "prev": prev, "inp": inp, "ctx": ctx,
                "rows_g": rows_g, "order": tac.order[idx]}

    def _tac_pending(self):
        T = self._tac_rows()
        if T is None:
            return None
        ctx, tiles, my, prev, rows_g = T["ctx"], T["tiles"], T["my"], T["prev"], T["rows_g"]
        hint = TTC.drive(ctx.mine_raw, ctx.able, tiles)
        hf = HU.tac_frames(hint["frames"], tiles, my, rows_g)
        cur = HU.tac_frames(prev, tiles, my, rows_g) if prev is not None else {g: {} for g in rows_g}
        must = U.Seq(tiles, my.shape[1]).must_change(prev, ctx.able)
        rows, edit, hint_edit = [], [], []
        R, Gm = my.shape
        seen = set()
        for r in range(R):
            g = rows_g[r]
            seen.update(s for s in T["inp"]["slot"][r].tolist() if s >= 0)
            units, eu, hu = [], [], []
            for j in range(Gm):
                s = int(my[r, j])
                if s < 0:
                    continue
                able = bool(ctx.able[r, j])
                mu = bool(must[r, j])
                units.append({"slot": s, "able": able, "must": mu, "frame": cur[g].get(s), "hint": hf[g].get(s)})
                if not able:
                    continue
                h = hf[g].get(s)
                if h is not None:
                    hu.append({"slot": s, "keep": False, "w": h["w"], "d": h["d"], "x": h["x"], "z": h["z"]})
                if mu or cur[g].get(s) is None:
                    if h is not None:
                        eu.append(dict(hu[-1]))
                else:
                    f = cur[g][s]
                    eu.append({"slot": s, "keep": True, "w": f["w"], "d": f["d"], "x": f["x"], "z": f["z"]})
            rect = tiles.rects()
            tl = [[round(float(rect["x"][r, t]), 1), round(float(rect["z"][r, t]), 1)]
                  for t in range(rect["x"].shape[1]) if float(rect["front"][r, t]) > 0]
            rows.append({"g": g, "cx": float(tiles.cx[r]), "cz": float(tiles.cz[r]), "heading": float(tiles.heading[r]),
                         "front": float(tiles.front[r]), "depth": float(tiles.depth[r]), "tiles": tl,
                         "near": float(tiles.near_size()[r]), "units": units,
                         "target": T["parcel"].target[r].nonzero()[:, 0].tolist()})
            edit.append({"g": g, "units": eu})
            hint_edit.append({"g": g, "units": hu})
        first = prev is None
        return {"level": "tactic", "sec": self.sec, "seen": sorted(seen), "rows": rows, "edit": {"rows": edit},
                "edit_from": "hint" if first else "keep", "hint": {"rows": hint_edit}, "can_keep": not first,
                "steps": [float(v) for v in tcfg()["size_steps_m"]], "cells": int(tcfg()["cells"]),
                "_hint": hint, "_T": T}

    def _tac_try(self, edit):
        T = self.pending["_T"]
        act = HU.tac_act(edit, T["idx"], T["rows_g"], T["ctx"], T["tiles"], T["prev"])
        out, errs = HU.tac_walk(T["inp"], T["ctx"], act, T["rows_g"])
        return act, out, errs

    # ------------------------------------------------------------ the person's calls

    def preview(self, edit):
        if self.pending is None:
            return {"ok": False, "errors": ["решения сейчас не ждут"]}
        try:
            if self.pending["level"] == "strategist":
                act = self.strat.act(edit)
                out, errs = self.strat.walk(act)
                res = self.strat.result(out)
                return {"ok": not errs, "errors": errs, "groups": res["groups"], "places": res["places"],
                        "edit": res["edit"]}
            if self.pending["level"] == "operative":
                act, out, filled, errs = self._op_try(edit)
                res = HU.op_result(out, self.op[1], self.axes)
                return {"ok": not errs, "errors": errs, "groups": res["groups"], "edit": filled}
            act, out, errs = self._tac_try(edit)
            if out is None:
                return {"ok": False, "errors": errs}
            T = self.pending["_T"]
            fr = HU.tac_frames(out["frames"], T["tiles"], T["my"], T["rows_g"])
            warn = [f"группа {T['rows_g'][r] + 1}: размер не влез — взят наименьший" for r in
                    out["fell"].nonzero()[:, 0].tolist()]
            return {"ok": not errs, "errors": errs, "warnings": warn,
                    "frames": {str(g): v for g, v in fr.items()}}
        except HU.Refused as e:
            return {"ok": False, "errors": e.errors}

    def _source(self, edit):
        if edit.get("keep"):
            return "keep"
        strip = lambda e: {k: v for k, v in e.items() if not k.startswith("_")}            # noqa: E731
        return "hint" if strip(edit) == strip(self.pending["hint"]) else "human"

    def decide(self, edit):
        """The person's decision: checked (errors: nothing changes, {"ok": False}); else written and put in force."""
        if self.pending is None:
            return {"ok": False, "errors": ["решения сейчас не ждут"]}
        pend = self.pending
        try:
            if pend["level"] == "strategist":
                act = self.strat.act(edit)
                out, errs = self.strat.walk(act)
                filled = edit
            elif pend["level"] == "operative":
                act, out, filled, errs = self._op_try(edit)
            else:
                act, out, errs = self._tac_try(edit)
                filled = edit
        except HU.Refused as e:
            return {"ok": False, "errors": e.errors}
        if errs:
            return {"ok": False, "errors": errs}
        source = self._source(edit)
        if pend["level"] == "strategist":
            res = self.strat.result(out)
            sample = self.strat.apply(act, out)
            sample.update({"formula_act": pend["_hint_act"], "source": source})
            text = {"sec": 0, "level": "strategist", "source": source, "edit": filled, "groups": res["groups"],
                    "places": res["places"]}
            if self.folder is not None:
                (self.folder / "matchup_deployed.json").write_text(
                    __import__("json").dumps(self.m_deployed, ensure_ascii=False, indent=1), encoding="utf-8")
            self.stage = "begin"
        elif pend["level"] == "operative":
            inp, ctx = self.op
            sample = {"level": "operative", "sec": self.sec, "raw": self.raw, "groups": RC.groups_dict(self.groups.clone()),
                      "card": self.b.card, "side": self.b.side, "command": self.b.command,
                      "axes": RC.axes_dict(self.axes), "bounds": self.b.st.bounds, "K": self.K, "act": act,
                      "formula_act": pend["_hint_act"], "source": source}
            res = HU.op_result(out, ctx, self.axes)
            text = {"sec": self.sec, "level": "operative", "source": source, "edit": filled, "groups": res["groups"]}
            self._op_apply(out, True)
            self.stage = "tactic"
        else:
            T = pend["_T"]
            self.htac.act = act
            self.htac.rows_g = T["rows_g"]
            self.tac.tick(self.raw, self.b.card, self.b.st.bounds)
            self.units.write(self.tac)
            last = self.htac.last
            hint = pend["_hint"]
            sample = {"level": "tactic", "sec": self.sec, "raw": self.raw, "card": self.b.card,
                      "bounds": self.b.st.bounds, "K": self.Kt, "parcel": RC.parcel_dict(last["parcel"]),
                      "tiles": RC.tiles_dict(last["tiles"]), "prev": last["prev"], "order": last["order"],
                      "act": act, "groups_of_rows": T["rows_g"], "source": source,
                      "formula_act": {k: hint[k] for k in ("width", "depth", "cell")}}
            fr = HU.tac_frames(last["out"]["frames"], last["tiles"], T["my"], T["rows_g"])
            text = {"sec": self.sec, "level": "tactic", "source": source, "edit": filled,
                    "frames": {str(g): v for g, v in fr.items()}}
            self.stage = "sim"
        if self.rec is not None:
            self.rec.add(sample, text)
        self.n_decisions += 1
        self.pending = None
        return {"ok": True, "source": source, "decisions": self.n_decisions}

    def stop(self):
        if not self.over:
            self._finish("stopped by the person")
        return self.result

    def _finish(self, why):
        st = self.b.st
        self.over = True
        self.pending = None
        m1 = reward.measure(st)
        trade = float((self.m0[0, 1, 3] - m1[0, 1, 3]) - (self.m0[0, 0, 3] - m1[0, 0, 3]))
        u = st.u
        left = {}
        for s, name in ((1, "mine"), (2, "enemy")):
            sel = (u["side"][0] == s)
            stand = sel & (u["men"][0] > 0) & ~u["gone"][0] & ~u["r"][0]
            left[name] = {"units": int(sel.sum()), "standing": int(stand.sum()),
                          "hp_share": round(float(m1[0, s - 1, 0]), 3), "gold_share": round(float(m1[0, s - 1, 3]), 3)}
        self.result = {"why": why, "seconds": self.sec, "winner": int(st.winner[0]), "trade_gold": round(trade, 4),
                       "left": left, "decisions": self.n_decisions, "parcels": self.parcels,
                       "parcel_errors": len(self.parcel_errors), "level": self.level, "period_s": self.period}
        if self.rec is not None:
            self.rec.finish(self.result, {"slots": self.b.slots, "keys": list(st.keys[0]), "snaps": self.replay})

    # ------------------------------------------------------------ the page's view

    @staticmethod
    def _rel(f):
        if f is None:
            return None
        try:
            return f.resolve().relative_to(project.ROOT.resolve()).as_posix()
        except ValueError:
            return str(f)

    def field(self):
        if self.axes is None:
            return None
        f = self.axes.field
        return {k: round(float(f[n][0]), 2) for k, n in (("x", "x"), ("z", "z"), ("b", "b"), ("front", "front"),
                                                          ("depth", "depth"))}

    def page_pending(self):
        if self.pending is None:
            return None
        return {k: v for k, v in self.pending.items() if not k.startswith("_")}

    def state(self):
        return {"sec": self.sec, "over": self.over, "result": self.result, "pending": self.page_pending(),
                "level": self.level, "period": self.period, "folder": self._rel(self.folder),
                "decisions": self.n_decisions, "snap": VW.for_page(self.last_snap), "keys": self.info["keys"],
                "limit": self.b.limit, "map": self.m.get("map", "flat"), "field": self.field(),
                "command": {"verb": self.b.verb, "pace": self.b.pace}}
