"""The chain 'tactic_v2 only' in the game (companion --chain --chain-tactic <a tactic_v2 checkpoint>,
bridge/docs/launch/watch.md "Цепочка «только тактик»"): the trained tactic level commands our side through the one side wrapper
tactic/side.Side with the network as its player (tactic/pin.NetPlayer, greedy - the exams' player), the unit level by the
unit's script (net/unit/scripts.Unit, no unit network), the strategist and the operative 'default' (no network: one
group for the whole army but the lord, no re-parcel). The same path as the simulator's stage0/battle.TacticSide.

Every state document (one a battle second):

    second 0   side.of_raw on the seer's raw (my units the alive rows of side 1, the lord by the passport's caste or by
               the name '<side>_lord'), Side.start: one group, its zone by formation (tactic/zfit), the lord's row hidden
    every 5 s  Side.tick(raw, card, bounds, now = the battle second) -> lower.Units.write: each unit's frame, skill and
               main target from the level (GameUnits, the unit level's raw)
    every 1 s  the unit's script on the unit level's raw (its skill, frame, main target, pace any, its own last order -
               the gate's output, and drill T0-2's approach, age = the second - the order's birth, and the flank walk's
               time guard walk_s, the target's facing at its birth and the last MOVE's second - lower.Units.script_order
               / note_sent, one code), no leash; the order gate reissue.Gate (a
               changed order only); the lord: hold once, then keep (the simulator's merge of hold over the hidden lord);
               with the state's lords = 'park' (bridge.build --park-lords) the lord gets no order at all (marked out,
               no op MOVE): the game's script holds him in a corner; with BAI_LORD_SCRIPT=T (the lord line's lane
               instrument, off by default) the lord's line and his `ability` lines come from lord_script.LordScript
               (the levels' input unchanged: he stays hidden from them) in the mode BAI_LORD_MODE (t2 - unset -,
               post, tactic); tactic: the level commands him too (tactic/side.Side with a lord player,
               BAI_LORD_PLAYER - tactic/lordtac.PLAYERS, unset LT; pass 1 still gets his row hidden), the level's
               lord store's target is the script's command, the script's drops go back into the store on the same
               second; the script's line is the last writer of the lord's order (after the operative's MOVE)
A tactic_v3 checkpoint (heads.TacticNet3, drill T0-2 'side of the strike') runs the same chain: the network's approach
goes through the level's masks into the unit's order; the record adds own() of 19, the S2 row columns, the side head's
columns and the approach.

bounds: P.map_half (the drill's flat battle), the chain's clock: the document's t rounded to the second (bridge/
companion/chain.py). Per tick the log gets the second, the engine step (tick_ms), the head-free units, each new attack
(unit -> target by the game's name) and the orders in force; the first dump_ticks ticks also the level's input (the
185-number rows, D2's columns, own, the pair numbers, the masks) and the document (game/battle/tactic_read.py).
"""
import os
import time

import torch

from bridge.companion import exchange
from bridge.companion import tactic_input as TI
from bridge.companion import unit_input as UI
from bridge.companion.chain import GameUnits
from net.common import reissue
from net.operative import opside as OPS       # (plan `holder`: the scripted operative, BAI_OP_SCRIPT)
from net.tactic import cfg as tcfg
from net.tactic import raw as TR
from net.tactic import side as TS
from net.unit import decode as Dc
from net.unit import view as V
from sim import orders as O

KIND = "tactic_v2"
KINDS = ("tactic_v2", "tactic_v3")
DUMP_TICKS = 12          # the level's input dumped for the first 12 ticks (60 s: the approach and the first contact)


def tick_record(tac):
    """{own, mt, cols, pair, slot, head, open, skill, tgt, side, my, new, rows, mask; own3, s2, sx, approach} of the
    level's last tick (lower.Tacticians.view): own() of the orders in force, ticks in melee, D2's columns, the walk's
    pair numbers and open masks, the head-free mask, the outputs (tgt a unit index), my slots (unit indexes), the new
    attacks, the input rows and their mask; drill T0-2's own() of 19, the S2 row columns, the side head's columns
    (heads.side_feats) and the approach after the masks. The simulator's side and the companion read it alike (the
    contract case)."""
    from net.tactic import cols as CL
    from net.tactic import heads as HD
    from net.tactic import orders as OR
    v = tac.view
    if v is None:
        return None
    view = v["view"]
    fx = view.fx
    given = torch.where(v["new_attack"], fx.row_of_slot(v["tgt"]), torch.full_like(v["tgt"], -1))
    x = HD.net_inputs(view)
    wi = HD.walk_inputs(fx, given)
    rec = {"own": OR.own(view.order, view.ctx.prev, fx, view.now), "mt": fx.mt, "cols": x["cols"],
           "pair": wi["pair"], "slot": fx.slot, "head": HD.head_free(fx), "open": wi["open"], "skill": v["skill"],
           "tgt": v["tgt"], "side": v["side"], "my": v["my"], "new": v["new_attack"],
           "own3": OR.own(view.order, view.ctx.prev, fx, view.now, approach=True), "s2": CL.side_rows(fx),
           "sx": HD.side_feats(fx), "approach": v.get("approach", torch.zeros_like(v["skill"]))}
    inp = getattr(view, "inp", None)
    if isinstance(inp, dict) and "rows" in inp:
        rec["rows"], rec["mask"] = inp["rows"], inp["mask"]
    return rec


def _frame0(units, own):
    """The unit level's frames before the first tick: each unit's slot of the rows where it stands (stage0/battle.UnitSide:
    config stage0 general enemy_row_m x enemy_rank_m)."""
    from net.stage0 import battle as SB
    from net.stage0 import cfg as scfg
    g = scfg()["general"]
    u = units.setup.st.u
    B, N = own.shape
    dev = own.device
    units.area = {"x": u["x"].clone(), "z": u["z"].clone(), "b": u["b"].clone(),
                  "front": torch.full((B, N), float(g[SB.FRAME_ROW[0]]), device=dev),
                  "depth": torch.full((B, N), float(g[SB.FRAME_ROW[1]]), device=dev),
                  "sb": u["b"].clone(), "sw": torch.zeros((B, N), device=dev), "on": own.clone()}


class TacticChain:
    """Our side in the game under the trained tactic level (module doc). net: a heads.TacticNet2 (ckpt.restore of a
    tactic_v2 checkpoint); bounds 'map_half' (default) - the simulator's P.map_half, or a float / None."""
    MODE = "chain"             # the answer's 'mode chain' (exchange.orders_text): the bridge's chain rules

    def __init__(self, net, greedy=True, params=None, device="cpu", map_root=UI.MAPS, allow_flat=True, own=1,
                 hygiene=True, bounds="map_half", dump_ticks=DUMP_TICKS, tick_ms=None):
        assert own == 1, "the chain commands side 1"
        self.seer = TI.TacticSeer(map_root, device, own, params, allow_flat)
        self.P = self.seer.P
        self.bounds = self.P.map_half if bounds == "map_half" else bounds
        self.net, self.greedy, self.device, self.own, self.hygiene = net, greedy, device, own, hygiene
        self.t_every = int(tcfg()["decide_s"])
        self.dump_ticks = int(dump_ticks)
        # the engine's step the launcher asked for (BAI_TICK_MS: launch.ps1 -TickMs; None - the engine's own)
        env = tick_ms if tick_ms is not None else os.environ.get("BAI_TICK_MS")
        self.tick_ms = int(env) if env not in (None, "") else None
        self.op_arm, self.op = OPS.parse(os.environ.get("BAI_OP_SCRIPT")), None     # (off: None - no hook runs)
        # the lord line's lane instrument (lord_script.py; BAI_LORD_SCRIPT=T, watch.ps1 -LordScript T): our lord by fixed
        # rules; unset or empty - off, and lord_script is never imported
        arm = os.environ.get("BAI_LORD_SCRIPT") or None
        if arm not in (None, "T"):
            raise ValueError(f"BAI_LORD_SCRIPT={arm!r}: 'T' or unset")
        self.lord_arm, self.lord_script = arm, None
        from bridge.companion import lord_script as LS
        self.lord_mode = os.environ.get("BAI_LORD_MODE") or "t2"
        if self.lord_mode not in LS.MODES:
            raise ValueError(f"BAI_LORD_MODE={self.lord_mode!r}: one of {LS.MODES} or unset")
        if self.lord_mode != "t2" and arm is None:
            raise ValueError("BAI_LORD_MODE without BAI_LORD_SCRIPT=T")
        self.lord_player_name = os.environ.get("BAI_LORD_PLAYER") or "LT"
        self.batch, self.parked = None, False
        self.times, self.extra = {}, None

    # -- a new battle
    def start(self, doc, r):
        from net.tactic import pin as PN
        from net.unit import scripts as SC
        dev = self.device
        self.batch = doc["batch"]
        self.names = [u["n"] for u in doc["units"]]
        self.keys = [u.get("key") or "" for u in doc["units"]]
        self.card = self.seer.card
        x, z, b, mine, hide = TS.of_raw(r, self.card, self.own)
        side = r[..., TR.I["side"]].round().long() == self.own
        by_name = torch.tensor([[n.endswith("_lord") for n in self.names]], device=r.device) & side
        hide = hide | by_name
        mine = mine & ~hide
        lp = None
        if self.lord_arm is not None and self.lord_mode == "tactic":
            from net.tactic import lordtac as LT
            lp = LT.player(self.lord_player_name)
        self.side = TS.Side(PN.NetPlayer(self.net, greedy=self.greedy), self.hygiene, dev,
                            lord_player=lp).start(x, z, b, mine, hide)
        N = r.shape[1]
        self.units = GameUnits(r[..., :TR.BASE], self.keys, self.card, self.P, SC.Unit(), targets=False, still=True,
                               aim=True, lords=hide)
        self.units.gate = reissue.Gate(1, N, dev)
        _frame0(self.units, self.units.setup.own)
        self.hide, self.lord_held = hide, torch.zeros_like(hide)
        # (bridge.build --park-lords: the lords held by the game's script; 'park_enemy' - only the enemy's, ours in play)
        self.parked = doc.get("lords") == "park"
        self.next_tac = None
        self.n_ticks = 0
        men = float((r[..., TR.I["men"]] * (r[..., TR.I["alive"]] > 0)).sum())
        self.start_info = {"men": round(men), "mine": [n for i, n in enumerate(self.names) if bool(mine[0, i])],
                           "lord": [n for i, n in enumerate(self.names) if bool(hide[0, i])], "tick_ms": self.tick_ms}
        self.op = None if self.op_arm is None else OPS.OpScript(self.op_arm, self.names, blocked=OPS.blocked_of(doc.get("map")))
        if self.op is not None:
            self.start_info["op"] = "holder:" + self.op_arm
        if doc.get("lords"):
            self.start_info["lords"] = doc.get("lords")
        self.lord_script = None
        if self.lord_arm is not None:
            # (lane T: our lord in play by the script; both lords parked contradicts it - refused loudly)
            if self.parked:
                raise ValueError("BAI_LORD_SCRIPT with both lords parked (bridge.build --park-lords): our lord is the "
                                 "script's - build with --park-lords enemy or none")
            from bridge.companion import lord_script as LS
            self.lord_script = LS.LordScript(self.names, self.keys, [int(u["side"]) for u in doc["units"]],
                                             self.own, self.lord_arm, mode=self.lord_mode)
            self.start_info["lord_script"] = {"arm": self.lord_arm, "lord": self.lord_script.lord,
                                              "press": self.lord_script.press, "mode": self.lord_mode}

    def script_second(self, raw_u, sec=0.0):
        """lower.Units.second's script branch on the unit level's raw [1, N, RAW] at battle second sec -> (the gated
        orders, inside)."""
        I = V.I
        u = self.units
        own = u.setup.own
        ar = {k: u.area[k] for k in ("x", "z", "b", "front", "depth")}
        inside = Dc.signed_dist(raw_u[..., I["x"]], raw_u[..., I["z"]], ar) >= 0
        u.arrived = u.arrived | (inside & own)
        live = own & (raw_u[..., I["alive"]] > 0) & ~(raw_u[..., I["r"]] > 0)
        # (lower.Units.second's script order and MOVE clock - one code: Units.script_order / note_sent)
        now = torch.full_like(u.born, float(sec))
        intended = u.net.intend(raw_u, u.script_order(raw_u, ar, now), live)
        u.note_sent(intended, live, now)
        intended = u.gate.apply(intended, live)
        u.intended = intended
        return intended, inside

    @torch.no_grad()
    def step(self, r, sec):
        """One second on the raw state r [1, N, RAW] at battle second sec -> (the orders to the game, the script's gated
        orders, the tick's record or None)."""
        t2 = time.perf_counter()
        rec = None
        if self.next_tac is None:
            self.next_tac = sec
        if sec >= self.next_tac:
            tac = self.side.tick(r, self.card, self.bounds, now=float(sec))
            self.units.write(tac)
            rec = tick_record(tac)
            self.n_ticks += 1
            while self.next_tac <= sec:
                self.next_tac += self.t_every
        t3 = time.perf_counter()
        self.times["tac_ms"] = (t3 - t2) * 1000
        lord = self.op.step(self.side, r, self.card, float(sec), self.units) if self.op is not None else {}
        raw_u = r[..., :TR.BASE].clone()
        raw_u[..., V.I["leash"]] = 0.0                  # (no leash under the script)
        intended, inside = self.script_second(raw_u, float(sec))
        # the lord: hold once, then keep (the simulator merges a hold over the hidden lord every step)
        first = self.hide & ~self.lord_held
        out = O.merge(intended, O.hold(1, r.shape[1], r.device), first)
        later = self.hide & self.lord_held
        out.kind = torch.where(later, torch.full_like(out.kind, O.KEEP), out.kind)
        self.lord_held = self.lord_held | self.hide
        if lord and not self.parked:                    # (plan `holder`: the lord's MOVE behind the moved zone)
            out = self.op.lord_orders(out, self.hide, lord)
        self.inside, self.sec, self.raw_u = inside, sec, raw_u
        self.times["unit_ms"] = (time.perf_counter() - t3) * 1000
        return out, intended, rec

    @torch.no_grad()
    def decide(self, doc):
        """-> (orders list, think ms, abilities []) for one state document (loop.Seat's decide)."""
        t0 = time.perf_counter()
        self.times = {}
        r = self.seer.raw(doc)
        new = doc["batch"] != self.batch
        if new:
            self.start(doc, r)
        self.times["raw_ms"] = (time.perf_counter() - t0) * 1000
        sec = int(round(float(doc.get("t", 0)) / 1000.0))
        o, _, rec = self.step(r, sec)
        col = lambda f: getattr(o, f)[0].cpu().numpy()        # noqa: E731
        side = [int(u["side"]) for u in doc["units"]]
        out = exchange.orders_list(self.names, side, col("kind"), col("x"), col("z"), col("target"), col("run"),
                                   own=self.own, face=col("face"), width=col("width"))
        I = V.I
        ctrl = {n for i, n in enumerate(self.names)
                if side[i] == self.own and float(self.raw_u[0, i, I["alive"]]) > 0
                and not float(self.raw_u[0, i, I["r"]]) > 0 and not (self.parked and bool(self.hide[0, i]))}
        uses = []
        if self.lord_script is not None:
            # (lord_script.py) the lord's line replaced by the script's (after the Gate); its button as an `ability`
            # line; what the reply does not carry (a lord marked out) is not cached as sent (commit)
            cmd, eng = self._lord_command(r)
            lo, uses = self.lord_script.decide(doc, [x for x in out if x["unit"] in ctrl], cmd=cmd,
                                               engaged=eng)                                   # (the run-in's onset)
            self._lord_mirror()
            out = [dict(lo) if x["unit"] == self.lord_script.lord else x for x in out]
            delivered = self.lord_script.lord in ctrl
            self.lord_script.commit(delivered)
            if not delivered:
                uses = []
        for x in out:
            if x["unit"] not in ctrl:
                x["out"] = True
        self.extra = self._extra(doc, rec, new)
        if self.lord_script is not None:
            self.extra["lord"] = self.lord_script.note
        return out, (time.perf_counter() - t0) * 1000, uses

    def _lord_command(self, r):
        """Mode tactic: (the level's lord target by name or None, net/unit/lord.engaged_of on this raw) - else (None,
        None)."""
        tac = self.side.tac
        if self.lord_mode != "tactic" or getattr(tac, "lord_player", None) is None:
            return None, None
        from net.unit import lord as LD
        li = torch.tensor([self.names.index(self.lord_script.lord)], device=r.device)
        eng = bool(LD.engaged_of(r[..., :TR.BASE], li, self.own)[0])
        t = int(tac.l_tgt[0])
        return (self.names[t] if 0 <= t < len(self.names) else None), eng

    def _lord_mirror(self):
        """Mode tactic: the script's drop of the command into the level's store on the same second (BUILD_NOTES 8),
        and his step-out (not the level's while he steps out)."""
        tac = self.side.tac
        if self.lord_mode != "tactic" or getattr(tac, "lord_player", None) is None:
            return
        from net.unit import lord as LD
        ls = self.lord_script
        dev = tac.l_tgt.device
        if ls.drop is not None:
            tac.lord_drop(torch.ones(1, dtype=torch.bool, device=dev), LD.DROP_NAMES.index(ls.drop))
        tac.l_busy = torch.tensor([ls.step is not None or bool(ls.out)], device=dev)

    # -- the log
    def _name(self, i):
        i = int(i)
        return self.names[i] if 0 <= i < len(self.names) else None

    def _extra(self, doc, rec, new):
        row = {"sec": self.sec, "tick_ms": self.tick_ms, "ms": {k: round(v, 2) for k, v in self.times.items()}}
        if new:
            row["start"] = self.start_info
        if getattr(self, "op", None) is not None and self.op.notes:
            row["op"] = self.op.pop_notes()
        if self.n_ticks <= self.dump_ticks:
            row["doc"] = doc               # (every second's: the replay's velocities need the documents between ticks)
        if rec is None:
            return row
        my = rec["my"][0].tolist()
        head = rec["head"][0].tolist()
        o = self.side.tac.orders_of()
        own = self.units.setup.own[0]
        tac = {"tick": self.n_ticks - 1,
               "head_free": [self._name(i) for i, h in zip(my, head) if h and i >= 0],
               "attacks": [[self._name(i), self._name(t)] for i, t, n in
                           zip(my, rec["tgt"][0].tolist(), rec["new"][0].tolist()) if n and i >= 0],
               "in_force": {self.names[i]: [int(o["skill"][0, i]), self._name(o["tgt"][0, i])]
                            for i in range(len(self.names)) if bool(o["on"][0, i]) and bool(own[i])}}
        row["tactic"] = tac
        if self.n_ticks <= self.dump_ticks:
            row["dump"] = {k: v[0].tolist() if v.dtype != torch.bool else v[0].int().tolist()
                           for k, v in rec.items()}
        return row

    def log_extra(self):
        return self.extra

    @staticmethod
    def describe_extra(row):
        """The console line of a decision's tactic fields (loop.answer)."""
        ms = row["ms"]
        line = (f"  tactic sec {row['sec']}: raw {ms.get('raw_ms', 0):.0f} tac {ms.get('tac_ms', 0):.0f} "
                f"unit {ms.get('unit_ms', 0):.0f} ms, tick_ms {row['tick_ms'] or 'engine'}")
        t = row.get("tactic")
        if t is not None:
            line += (f" | tick {t['tick']} head-free {len(t['head_free'])}, attacks "
                     + (", ".join(f"{a} -> {b}" for a, b in t["attacks"]) or "none"))
        return line
