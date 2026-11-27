"""Drill O0-1 'advance on the holder' - the exam before training (METHOD.md recipe step 4; tmp/op1/design_merged.md
FINAL, tmp/op1/design_opus.md 6): three players of the operative level on the SAME battles, paired by battle -
    DEFAULT  the stack without the operative (today: the zone never moves - KEEP always)
    DUMB     march onto his line once (opside's arm M: Hold(LM) at the first tick with att 1)
    TEACH    the drill's teacher (advance.teach: ADVANCE -> Hold(LS) / KEEP; LS - the stand-off line)
under the same tactic level (pin player --tactic, default TEACH: T0-1's teacher, which its network matches - net/drills.json
T0-1 reexam) and the unit script. Every player is asked through the level's hygiene mask (no zone change while a member
is in melee) and the teacher is asked beside it on the same facts (the shadow label: the probe's data).
Sets (scenes_o1): taught, held, G (general battles, att 1, the sally holder), G_hold (a holder that never comes -
reported, not gated). The judge: the whole-field trade by health shares of each side's start (enemy lost - own lost),
lords IN, from the decision tick o1.start_s over o1.window_s (600 s), paired by battle; the same without the lords'
health, the 300 s read and the battle's result printed beside it. The battles where a member of ours is caught on its walk
(its first melee while its last order to the game is a MOVE and it moves: an enemy came at it after its walk order -
sim/evidence/melee_core/spec.md 'Known gross error') are reported apart: every group also '[-comer]'.
LINES (written before the run, tmp/o1 brief; do not move):
    1  cause scenes: contact before the timer - TEACH >= 0.95, DEFAULT <= 0.05
    2  no-cause scenes: TEACH changes the order in <= 3 % of battles, its trade within +-2 errors of DEFAULT
    3  TEACH - DUMB on cause scenes: reported, not gated
    4  orders to the game a unit-minute (the lord's MOVEs counted) <= 1.2 (TEACH)
    5  held - taught, contact share of TEACH on cause scenes: not below -2 errors
    6  G (att 1 against a holder): TEACH's stalls (no contact before the timer) <= 0.05
    7  the label probe - net/operative/probe_o1.py
    python -m net.operative.exam_o1 run --sets taught,held,G,G_hold --players DEFAULT,DUMB,TEACH --out tmp/o1/exam [--n 16]
    python -m net.operative.exam_o1 table --out tmp/o1/exam
    python -m net.operative.exam_o1 preflight [--n 16]        (every start and twin of the sets built; counts)
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from net.matchup import duel as DU
from net.operative import advance as AD
from net.operative import lines as LN
from net.operative import sally as SA
from net.operative import scenes_o1 as SO
from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.stage0 import watch as W
from sim import orders as O

PLAYERS = ("DEFAULT", "DUMB", "TEACH")
PAIRS = (("TEACH", "DEFAULT"), ("TEACH", "DUMB"), ("DUMB", "DEFAULT"))
# (the exam after: the network NET read beside them when its files are there - lines(out, 'NET'); absent: as before)
TABLE_PLAYERS = PLAYERS + ("NET",)
TABLE_PAIRS = PAIRS + (("NET", "DEFAULT"), ("NET", "TEACH"), ("NET", "DUMB"))
SETS = ("taught", "held", "G", "G_hold", "G_def")
# (G_def - net/operative/attack.py: our mission hold, the enemy attacks from the start with the stage's scripts; card
# O0-1's no-harm piece Gq reads it; appended last: the torch starts of the four earlier sets stay)
TICK_COLS = ("b", "sec", "label", "act", "reason", "masked", "gap", "e_close_max") + LN.INPUT


class O1Side(BA.TacticSide):
    """My side: the tactic level (TacticSide) + the operative level of this drill (lines.Level under an advance player)
    after the tactic's tick every decide_s from start_s (the bridge's order: tactic, then the operative); the hidden lord
    follows after the zone's first move (lord HOLD once, then KEEP; a MOVE at the second of a new point - as the bridge)."""

    def __init__(self, tactic_player, op_player, specs, seed=None, zones=None, recorder=None, shadow=None, visible=None):
        super().__init__(tactic_player, hygiene=True, zones=zones, seed=seed)
        # (visible: the level's facts on the SEEN foes only - the network's input is built from them (Astra's o1 review
        # 1: the teacher must see what the network sees); None - o1.visible (False: the legacy exams' teacher))
        self.visible = visible
        self.op, self.specs = op_player, specs
        # (the drill engine's decision record, tmp/engine/design_merged.md 2: recorder.decision(...) at every decision
        # of a battle with facts, BEFORE its zone reaches the level below; None - the legacy side, nothing computed)
        self.recorder = recorder
        self.view = recorder is not None or bool(getattr(op_player, "wants_view", False))
        self.axes_b = {}
        # (shadow: a list - at every decision the level's facts again on the SEEN foes only, compared, nothing acted on)
        self.shadow = shadow

    def start(self, st, P, device, card, unit_net):
        from net.tactic import scenes as TSN
        super().start(st, P, device, card, unit_net)
        B = st.B
        self.c = LN.conf()
        self.level = LN.Level(B, [s["o1"]["att"] for s in self.specs], [TSN._blocked(s["map"]) for s in self.specs],
                              visible=self.visible)
        self.ticks = []
        self.lord_new = [None] * B
        z = lambda: torch.zeros(B, device=device)                      # noqa: E731
        self.n_change, self.n_masked, self.n_lord = z(), z(), z()
        self.moved_t = torch.zeros(B, dtype=torch.bool, device=device)

    def tick(self, st, raw):
        super().tick(st, raw)
        live = ~st.done
        if not bool(live.any()):
            return
        sec = int(round(float(st.t.float()[live].max())))
        c = self.c
        if sec < int(c["start_s"]) or (sec - int(c["start_s"])) % int(c["decide_s"]) or sec >= int(c["limit_s"]):
            return                                         # (no decision at the timer's last tick: the review 7)
        card = self.setup.card
        for b in range(st.B):
            if bool(st.done[b]):
                continue
            f = self.level.facts(self.wrap, raw, card, b, sec)
            if f is None:
                continue
            order = self.level.order[b]
            lab, why = AD.teach(f, c)
            if self.shadow is not None:
                self._shadow(raw, card, b, sec, f, lab, why)
            obs = self._observe(raw, card, b, f, order, sec) if self.view else None
            ptr = self.op(f, order, c, obs=obs, b=b) if getattr(self.op, "wants_view", False) else self.op(f, order, c)
            proposed = ptr
            masked = ptr is not None and f["melee"]
            if masked:
                ptr = None
                self.n_masked[b] += 1
            if self.recorder is not None:
                self.recorder.decision(b=b, sec=sec, f=f, order=order, obs=obs, teacher=lab, why=why,
                                       proposed=proposed, executed=ptr, raw=raw[b], card=card[b],
                                       axes=self.axes_b[b])
            if ptr is not None:
                self.level.apply(self.wrap, raw, b, ptr, f, self.units, sec)
                self.n_change[b] += 1
                self.moved_t[b] = True
            lp = self.level.lord(self.wrap, raw, b, f)
            if lp is not None:
                self.lord_new[b] = lp
            self.ticks.append([b, sec, int(lab is not None), -1 if ptr is None else LN.POINTERS.index(ptr),
                               AD.REASONS.index(why), int(masked), round(f["gap"], 2), round(f["e_close_max"], 3)]
                              + [round(float(x), 4) for x in f["row"]])

    def _shadow(self, raw, card, b, sec, f, lab, why):
        """The visible facts beside the level's (lines.Level.facts visible=True), appended to self.shadow as {b, sec,
        all_seen (every standing foe seen), unseen (their count), legacy / visible: teacher pointer, reason, gap,
        e_close_max, comer, melee, E size; none - no facts on the seen foes}."""
        from net.tactic import raw as TRW
        I = TRW.I
        S = raw[b]
        own = 1
        ok = (S[:, I["alive"]] > 0) & ~(S[:, I["r"]] > 0)
        foe = ok & (S[:, I["side"]].round().long() != own) & (S[:, I["side"]] > 0)
        unseen = int((foe & ~(S[:, I["vis"]] > 0)).sum())
        fv = self.level.facts(self.wrap, raw, card, b, sec, visible=True)

        def of(x, lb, wy):
            if x is None:
                return None
            return {"teacher": lb, "reason": wy, "gap": round(x["gap"], 3), "e_close_max": round(x["e_close_max"], 4),
                    "comer": x["comer"], "melee": x["melee"], "E": int(x["fc"]["E"].sum())}
        lv, wv = AD.teach(fv, self.c) if fv is not None else (None, None)
        self.shadow.append({"b": b, "sec": sec, "all_seen": unseen == 0, "unseen": unseen, "legacy": of(f, lab, why),
                            "visible": of(fv, lv, wv)})

    def _observe(self, raw, card, b, f, order, sec):
        """heads.observe of battle b; the field's axes fixed at the battle's first decision (frame.Axes)."""
        from net.operative import heads as HD
        if b not in self.axes_b:
            self.axes_b[b] = HD.axes_of(raw[b:b + 1], self.bounds)
        return HD.observe(raw[b:b + 1], card[b:b + 1], f, order, sec, self.axes_b[b])

    def orders(self, st, j):
        o = super().orders(st, j)
        if not bool(self.moved_t.any()) or self.lexe is not None:
            return o                                       # (stage0.json lord_script: the T2 lord script has him)
        hide = self.hide & self.moved_t[:, None]
        o.kind = torch.where(hide, torch.full_like(o.kind, O.KEEP), o.kind)
        if j == 0 and any(p is not None for p in self.lord_new):
            m = torch.zeros_like(self.hide)
            hold = torch.zeros_like(self.hide)
            x, z, f = o.x.clone(), o.z.clone(), o.face.clone()
            for b, q in enumerate(self.lord_new):
                if q is None:
                    continue
                m[b] = self.hide[b]
                self.n_lord[b] += 1
                self.lord_new[b] = None
                if q[0] == "hold":
                    hold[b] = self.hide[b]
                    continue
                p = q[1:]
                x[b] = torch.where(self.hide[b], torch.full_like(x[b], p[0]), x[b])
                z[b] = torch.where(self.hide[b], torch.full_like(z[b], p[1]), z[b])
                f[b] = torch.where(self.hide[b], torch.full_like(f[b], p[2]), f[b])
            o.kind = torch.where(m & ~hold, torch.full_like(o.kind, O.MOVE),
                                 torch.where(m & hold, torch.full_like(o.kind, O.HOLD), o.kind))
            o.target = torch.where(m, torch.full_like(o.target, -1), o.target)
            o.run = torch.where(m, torch.ones_like(o.run), o.run)
            o.x, o.z, o.face = x, z, f
        return o


class O1Watch(W.Watch):
    """Watch + the drill's reads (module doc). duel.play calls the watch AFTER the second sec is played: the state is
    that of the real second sec + 1 (the review 7). Series every SERIES_S real seconds from the deployment (index i =
    real second 5 i; index 0 the start, trade 0) of the health-share trade with and without the lords, a battle's value
    frozen at its terminal state (the second it ended included); measures() pads a short series with the terminal
    value. Each own member's first melee: caught on its walk (its last order to the game a MOVE and it moved >=
    still_ms the second before) and received from a stand (>= stand_s consecutive seconds under still_ms within
    stand_look_s before it). My lord's standing seconds (alive, not routing: the orders' denominator). The come twins'
    quality at the decision tick (real second t_dec): each enemy unit's miss to its cause pose (m) and its speed - per
    battle the max miss and the min speed; the enemy infantry's mean fatigue state (u fat) there, every battle."""
    SERIES_S = 5

    def __init__(self, P, cfg_general, chain=None, specs=None, per_side=None):
        super().__init__(P, cfg_general, chain=chain)
        self.specs, self.per_side = specs, per_side

    def _init(self, st):
        super()._init(st)
        u = st.u
        B, N = self.B, self.N
        side = u["side"]
        hp = u["hp_abs"]
        self.hp_start = {(s, lords): (hp * ((side == s) & (lords | ~u["lord"]))).sum(1).clamp(min=1e-6)
                         for s in (1, 2) for lords in (True, False)}
        z0 = torch.zeros(B, device=self.dev)
        self.ser_l, self.ser_nl = [z0.clone()], [z0.clone()]
        self.last_l, self.last_nl = z0.clone(), z0.clone()
        self.done_prev = torch.zeros(B, dtype=torch.bool, device=self.dev)
        c = LN.conf()
        self.still, self.stand_s, self.look = float(c["still_ms"]), int(c["stand_s"]), int(c["stand_look_s"])
        self.t_dec = int(c["start_s"])
        self.slow = torch.zeros((self.look, B, N), dtype=torch.bool, device=self.dev)
        self.first_m = torch.zeros((B, N), dtype=torch.bool, device=self.dev)
        self.v_prev = torch.zeros((B, N), device=self.dev)
        self.caught = torch.zeros(B, device=self.dev)
        self.recv, self.recv_n = torch.zeros(B, device=self.dev), torch.zeros(B, device=self.dev)
        self.lord_s = torch.zeros(B, device=self.dev)
        nan = torch.full((B,), float("nan"), device=self.dev)
        self.miss, self.vmin, self.efat = nan.clone(), nan.clone(), nan.clone()
        self.pose = None
        if self.specs is not None:
            px, pz = torch.zeros((B, N), device=self.dev), torch.zeros((B, N), device=self.dev)
            has = torch.zeros((B, N), dtype=torch.bool, device=self.dev)
            for b, sp in enumerate(self.specs[:B]):
                if sp["o1"].get("twin") == "come":
                    for j, (x, z) in enumerate(sp["o1"]["pose"]):
                        i = self.per_side + 1 + j
                        px[b, i], pz[b, i], has[b, i] = float(x), float(z), True
            self.pose = (px, pz, has)

    def _trade(self, u, lords):
        side = u["side"]
        keep = (lords | ~u["lord"])
        lost = [1.0 - (u["hp_abs"] * ((side == s) & keep)).sum(1) / self.hp_start[(s, lords)] for s in (1, 2)]
        return lost[1] - lost[0]

    def __call__(self, sec, st):
        super().__call__(sec, st)
        u = st.u
        t = sec + 1                                         # (the real second of this state)
        upd = ~self.done_prev                               # (live at this second's start: its terminal blow kept)
        self.last_l = torch.where(upd, self._trade(u, True), self.last_l)
        self.last_nl = torch.where(upd, self._trade(u, False), self.last_nl)
        self.done_prev = st.done.clone()
        live = ~st.done
        if t % self.SERIES_S == 0:
            self.ser_l.append(self.last_l.clone())
            self.ser_nl.append(self.last_nl.clone())
        my_lord = u["lord"] & (u["side"] == 1) & (u["men"] > 0) & ~u["r"] & ~u["gone"]
        self.lord_s += (my_lord.any(1) & upd).float()
        v = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
        if t == self.t_dec:
            foe = self.foe & (u["men"] > 0) & ~u["gone"]
            self.efat = (u["fat"].float() * foe).sum(1) / foe.float().sum(1).clamp(min=1)
            if self.pose is not None:
                px, pz, has = self.pose
                d = torch.sqrt((u["x"] - px) ** 2 + (u["z"] - pz) ** 2)
                big = torch.full_like(d, float("inf"))
                anyp = has.any(1)
                self.miss = torch.where(anyp, torch.where(has, d, -big).amax(1), self.miss)
                self.vmin = torch.where(anyp, torch.where(has, v, big).amin(1), self.vmin)
        self.slow = torch.cat([self.slow[1:], (v < self.still)[None]])
        new = self.mine & u["m"] & ~self.first_m & live[:, None]
        if bool(new.any()):
            gate = getattr(getattr(self.chain, "units", None), "gate", None)
            if gate is not None:
                walk = (gate.last.kind == O.MOVE) & (self.v_prev >= self.still)
                self.caught += (new & walk).float().sum(1)
            run = torch.zeros_like(self.first_m)
            k = torch.zeros(self.first_m.shape, dtype=torch.long, device=self.dev)
            for i in range(self.look - 1):                  # (stand_s consecutive slow seconds before the contact second)
                k = torch.where(self.slow[i], k + 1, torch.zeros_like(k))
                run = run | (k >= self.stand_s)
            self.recv += (new & run).float().sum(1)
            self.recv_n += new.float().sum(1)
            self.first_m |= new
        self.v_prev = v

    def measures(self, limit, winner=None):
        m = super().measures(limit, winner)
        T = int(limit) // self.SERIES_S + 1
        for k, ser, last in (("ser_l", self.ser_l, self.last_l), ("ser_nl", self.ser_nl, self.last_nl)):
            ser = ser + [last] * max(T - len(ser), 0)       # (a play that ended early: its terminal value to the limit)
            m[k] = torch.stack(ser[:T], 1)
        m["caught"] = self.caught
        m["recv_stand"] = torch.where(self.recv_n > 0, self.recv / self.recv_n.clamp(min=1),
                                      torch.full_like(self.recv, float("nan")))
        m["lord_s"] = self.lord_s
        m["twin_miss_m"], m["twin_v_min"], m["e_fat"] = self.miss, self.vmin, self.efat
        return m


def _per_side(batch):
    return max(BA.PER_SIDE, max(max(len(s["mine"]), len(s["theirs"])) for s in batch) + 1)


def play(name, specs, seeds, chunk, device, tactic="TEACH", compile=None, make_player=None, recorder=None, shadow=None,
         visible=None):
    """{per_battle {key: [B]}, ser_l, ser_nl [B][T], ticks [rows TICK_COLS, b the set's index]} of the player name on
    specs (chunk k under torch seed seeds[k]; on CUDA every batch padded to a full chunk: one compiled shape a
    per-side size). make_player(ci, ii) - the chunk's operative player instead of AD.PLAYERS[name]() (a network built
    by the caller BEFORE any chunk's seed); recorder - the drill engine's (recorder.chunk(ci, ii) before each chunk,
    then O1Side's decision calls; None: legacy)."""
    from net.tactic import exam_pin as EP
    from sim.params import load
    P = load()
    gen = scfg()["general"]
    compiled = torch.device(device).type == "cuda" if compile is None else bool(compile)
    out = {"per_battle": {}, "ser_l": [None] * len(specs), "ser_nl": [None] * len(specs), "ticks": []}
    t0 = time.time()
    for ci, a in enumerate(range(0, len(specs), chunk)):
        ii = list(range(a, min(a + chunk, len(specs))))
        pad = (chunk - len(ii)) if compiled else 0
        import copy
        batch = [specs[i] for i in ii] + [copy.deepcopy(specs[ii[k % len(ii)]]) for k in range(pad)]
        seed = int(seeds[ci])
        torch.manual_seed(seed)
        ps = _per_side(batch)
        zones = [s.get("zone") for s in batch]
        op = AD.PLAYERS[name]() if make_player is None else make_player(ci, ii)
        if recorder is not None:
            recorder.chunk(ci, ii)
        sh = [] if shadow is not None else None
        mine = O1Side(EP.build_under(tactic), op, batch, seed=seed, zones=zones, recorder=recorder, shadow=sh,
                      visible=visible)
        att = [s.get("o1", {}).get("enemy") == "attack" for s in batch]
        assert all(att) or not any(att), "a batch is all G_def battles or none"
        if any(att):
            from net.operative import attack as AT
            foe = AT.Attacker(batch, device, ps)
        else:
            foe = SA.Enemy(batch, device, ps)
        watch = O1Watch(P, gen, chain=mine, specs=batch, per_side=ps)
        limit = max(int(s["limit"]) for s in batch)
        scene = batch[0].get("kind") == "o1"
        assert all((s.get("kind") == "o1") == scene for s in batch), "a batch is all scenes or all general battles"
        battles = BA.scene_battles(batch) if scene else BA.unit_battles(batch)
        rd, stt = DU.play([BA.NAME] * len(batch), battles, {1: mine, 2: foe}, None, P, device, limit_s=limit,
                          per_side=ps, watch=watch, maps=[s["map"] for s in batch], compile=compiled)
        m = watch.measures(limit, rd["winner"])
        m["changes"], m["masked"], m["lord_orders"] = mine.n_change, mine.n_masked, mine.n_lord
        g = mine.units.gate
        own = watch.mine
        # (orders to the game a unit-minute: the infantry alone, and with the lord - his first HOLD and his orders over
        # the infantry's live seconds plus his standing ones: the review 6)
        m["orders_inf_min"] = 60.0 * (g.sent * own).sum(1) / (g.live_s * own).sum(1).clamp(min=1e-6)
        m["orders_with_lord_min"] = 60.0 * ((g.sent * own).sum(1) + mine.n_lord + 1.0) /             ((g.live_s * own).sum(1) + m["lord_s"]).clamp(min=1e-6)
        m["sally_s"] = foe.sally.fired_s
        for k, v in m.items():
            v = v.detach().cpu()
            if k in ("ser_l", "ser_nl"):
                for r, i in enumerate(ii):
                    out[k][i] = [round(float(x), 4) for x in v[r].tolist()]
            elif v.dim() == 1:
                pb = out["per_battle"].setdefault(k, [None] * len(specs))
                for r, i in enumerate(ii):
                    pb[i] = round(float(v[r]), 4)
        out["ticks"] += [[ii[r[0]]] + r[1:] for r in mine.ticks if r[0] < len(ii)]
        if shadow is not None:
            shadow += [{**r, "b": ii[r["b"]]} for r in sh if r["b"] < len(ii)]
        print(json.dumps({"player": name, "chunk": ci, "battles": len(ii), "wall_s": stt["wall_s"],
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out["tick_cols"] = list(TICK_COLS)
    return out


def sets_of(names, n=None, fair=False):
    """{name: (specs, the set's first torch seed)}; the seed by the set's place in SETS (tseed + 100 k) - the same in a
    run of all four sets (as every run so far) and in one of a part of them (chain_gpu.sh: a batch shape a process)."""
    S = SO.cs()
    out = {}
    from net.tactic import zfit
    for nm in names:
        zfit._CACHE.clear()       # (o1_train_review3 6: a set's starts from its own seeds only, not the build order)
        k = SETS.index(nm) if nm in SETS else len(SETS)
        if nm == "taught":
            sp = SO.taught_set(n)
        elif nm == "held":
            sp = SO.held_set(None if n is None else max(n // len(SO.HELD_AXES), 2))
        elif nm == "G":
            sp = SO.general_set(None if n is None else 2 * n)
        elif nm == "G_hold":
            sp = SO.general_set(None if n is None else n // 2, hold=True)
        elif nm == "G_def":
            from net.operative import attack as AT
            sp = AT.general_def(None if n is None else n)
        else:
            raise ValueError(nm)
        if fair and nm in ("taught", "held"):          # (the engine's fair starts: scenes_fair, the same family draws)
            from net.operative import scenes_fair as SF
            sp = [SF.fair(x) for x in sp]
        out[nm] = (sp, int(S["tseed"]) + 100 * k)
    return out


def meta(specs):
    return [{"cause": s["o1"]["cause"], "twin": s["o1"]["twin"], "att": s["o1"]["att"], "held": s.get("held"),
             "binds": SO.binds(s), "size": len(s["mine"]), "enemy": s["o1"]["enemy"], "comer": s["o1"].get("comer"),
             "sally_m": s["o1"].get("sally_m"), "axes": s.get("axes"), "seed": s.get("seed"), "map": s["map"],
             "seed_used": s.get("seed_used"),
             "pose": s["o1"].get("pose"), "key": battle_keys([s])[0]} for s in specs]


DRILL_FILES = ("lines.py", "advance.py", "sally.py", "scenes_o1.py")


def _h(x):
    import hashlib
    return hashlib.sha256(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_key(specs, chunk, tactic, seeds):
    """{part: hash} a player's file is reused only on (the review 4): the o1 config key, the drill's four files, the
    set's starts, the chunk and its torch seeds, the tactic player, the simulator's files (exam_pin.sim_key)."""
    from net import ROOT
    from net.tactic import exam_pin as EP
    return {"config_o1": _h(LN.conf()), **{f: _h((ROOT / "net" / "operative" / f).read_bytes()) for f in DRILL_FILES},
            "starts": _h(specs), "chunk": int(chunk), "seeds": list(seeds), "tactic": tactic, "sim_key": EP.sim_key()}


def _write(path, obj):
    """Atomic: a temporary file next to it, then replace."""
    import os
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def preflight(a):
    """Every start and twin of the sets at the chain's n built (a raise fails it); the count per set, per kind and per
    held axis, the substituted seeds."""
    from collections import Counter
    # (22.11 exam3: the starts the run plays - --fair as the run's; each set built and printed in turn, not all first;
    # one torch thread - the general sets' zone fits are tiny torch ops, 8 threads beside other containers crawled)
    torch.set_num_threads(1)
    out = {}
    for nm in a.sets.split(","):
        sp, _ = sets_of([nm], a.n or None, fair=bool(a.fair))[nm]
        kind = Counter("cause" if s["o1"]["cause"] else s["o1"]["twin"] for s in sp)
        held = Counter(f"{s.get('held')}, {'cause' if s['o1']['cause'] else s['o1']['twin']}" for s in sp if s.get("held"))
        sub = [(s["seed"], s["seed_used"]) for s in sp if s.get("seed_used") not in (None, s.get("seed"))]
        out[nm] = {"n": len(sp), "kinds": dict(kind), "held": dict(held), "substituted": sorted(set(sub)),
                   "binds": SO.binds_count(sp) if nm == "held" else None,
                   "per_side": max(len(s["mine"]) for s in sp) + 1}
        print(json.dumps({nm: out[nm]}), flush=True)
    print("PREFLIGHT OK", flush=True)
    return out


def run(a):
    from net.tactic import exam_pin as EP
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    for nm, (sp, t0) in sets_of(a.sets.split(","), a.n or None, fair=bool(a.fair)).items():
        mf = D / f"{nm}_meta.json"
        if mf.exists() and json.loads(mf.read_text(encoding="utf-8")) != json.loads(json.dumps(meta(sp))):
            raise SystemExit(f"{mf}: the kept set's metadata differs from these starts - another exam; use another --out")
        _write(mf, meta(sp))
        ch = a.chunk
        seeds = [t0 + k for k in range((len(sp) + ch - 1) // ch)]
        key = run_key(sp, ch, a.tactic, seeds)
        if a.visible:              # (the level's facts on the seen foes only: in the key only when asked)
            key["visible"] = True
        if a.fair:                 # (the engine's fair starts - scenes_fair: in the key only when asked)
            from net import ROOT
            key["fair"] = _h((ROOT / "net" / "operative" / "scenes_fair.py").read_bytes())
        if a.pin_device:           # (Astra's second o1 review: CUDA pads a chunk, the CPU does not - other RNG lanes)
            key["device"] = torch.device(a.device).type
        net_sha = None
        if "NET" in a.players.split(","):      # (Astra's o1 review 3: the NET arm bound to its checkpoint's sha)
            import hashlib
            net_sha = "fresh" if a.net in ("", "fresh") else hashlib.sha256(Path(a.net).read_bytes()).hexdigest()
        if nm == "G_def":          # (its enemy's file in its key only: the earlier sets' kept files keep their keys)
            from net import ROOT
            key["attack.py"] = _h((ROOT / "net" / "operative" / "attack.py").read_bytes())
        for name in a.players.split(","):
            f = D / f"{nm}_{name}.json"
            key_p = dict(key, net_sha=net_sha) if name == "NET" else key
            if f.exists():
                old = json.loads(f.read_text(encoding="utf-8")).get("key") or {}
                bad = sorted(k for k in key_p if old.get(k) != key_p[k])
                if bad:
                    raise SystemExit(f"{f}: kept file of another run - differs in {bad}; move it away or use another --out")
                print(json.dumps({"set": nm, "player": name, "kept": str(f)}), flush=True)
                continue
            t1 = time.time()
            mk = None
            if name == "NET":                              # (heads.OpNet1: built here, before any chunk's seed)
                from net.operative import ckpt as CK
                from net.operative import heads as HD
                net = HD.OpNet1(pick=True).to(a.device) if a.net in ("", "fresh") else CK.restore_o1(a.net, a.device)[0]
                mk = lambda ci, ii, net=net: HD.NetPlayer(net)          # noqa: E731
            res = play(name, sp, seeds, ch, a.device, tactic=a.tactic, make_player=mk, visible=bool(a.visible) or None)
            res.update(sim_key=EP.sim_key(), sim=a.sim, tactic=a.tactic, seeds=seeds, key=key_p,
                       battle_keys=battle_keys(sp))
            _write(f, res)
            print(json.dumps({"set": nm, "player": name, "battles": len(sp), "min": round((time.time() - t1) / 60, 1)}),
                  flush=True)
    return table(D)


# ------------------------------------------------------------------ the reads
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return round(sum(v) / len(v), 4) if v else None


def paired(a, b, sel):
    d = [a[i] - b[i] for i in sel if a[i] is not None and b[i] is not None and math.isfinite(a[i]) and math.isfinite(b[i])]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def share(flags):
    n = len(flags)
    if not n:
        return None
    p = sum(flags) / n
    q = (sum(flags) + 1) / (n + 2)                     # (the error from (k + 1) / (n + 2): 2 of 2 is not 'no error')
    return {"p": round(p, 4), "two_errors": round(2 * math.sqrt(q * (1 - q) / n), 4), "n": n}


def judge(d, c=None):
    """{j600, j600_nl, j300} [B]: the trade at the decision tick + window less at the decision tick (lords in / out)."""
    c = c or LN.conf()
    s = O1Watch.SERIES_S
    i0 = int(c["start_s"]) // s
    out = {}
    for key, ser in (("j600", d["ser_l"]), ("j600_nl", d["ser_nl"])):
        i1 = i0 + int(c["window_s"]) // s
        out[key] = [None if r is None else r[min(i1, len(r) - 1)] - r[min(i0, len(r) - 1)] for r in ser]
    i3 = i0 + 300 // s
    out["j300"] = [None if r is None else r[min(i3, len(r) - 1)] - r[min(i0, len(r) - 1)] for r in d["ser_l"]]
    return out


def reads(d, limit):
    pb = d["per_battle"]
    B = len(d["ser_l"])
    J = judge(d)
    contact = [1 if pb["t_contact"][b] is not None and pb["t_contact"][b] < limit else 0 for b in range(B)]
    ch = [0] * B
    for r in d["ticks"]:
        if r[3] >= 0:
            ch[r[0]] = 1
    return {**J, "contact": contact, "win": pb.get("win", [0] * B), "changed": ch, "caught": pb["caught"],
            "orders": pb["orders_with_lord_min"], "orders_units": pb.get("orders_inf_min", pb["game_orders_min"]),
            "recv": pb["recv_stand"], "t_contact": pb["t_contact"], "lord_orders": pb["lord_orders"],
            "twin_miss": pb.get("twin_miss_m", [None] * B), "twin_v": pb.get("twin_v_min", [None] * B),
            "e_fat": pb.get("e_fat", [None] * B)}


def groups_of(mt, nm, comer):
    B = len(mt)
    g = [("all", list(range(B)))]
    if nm in ("taught", "held"):
        g += [("cause", [b for b in range(B) if mt[b]["cause"]]), ("no cause", [b for b in range(B) if not mt[b]["cause"]]),
              ("no cause, come", [b for b in range(B) if mt[b]["twin"] == "come"]),
              ("no cause, role", [b for b in range(B) if mt[b]["twin"] == "role"])]
        if nm == "held":
            for ax in SO.HELD_AXES:
                g.append((f"held {ax}, cause", [b for b in range(B) if mt[b]["held"] == ax and mt[b]["cause"]]))
        for key in ("our_form", "enemy_form", "map", "comer"):
            for v in sorted({str((x["axes"] or {}).get(key)) for x in mt}):
                g.append((f"cause, {key} {v}", [b for b in range(B) if mt[b]["cause"] and str(mt[b]["axes"].get(key)) == v]))
    g = [(k, v) for k, v in g if v]
    return g + [(f"{k} [-comer]", [b for b in v if b not in comer]) for k, v in g] + [("comer", sorted(comer))]


def battle_keys(specs):
    """[a battle's key] in the set's order: its seed, the seed used, the twin, the held axis, the enemy kind."""
    return [[s.get("seed"), s.get("seed_used"), s["o1"].get("twin"), s.get("held"), s["o1"].get("enemy")] for s in specs]


def _same_battles(nm, pl, mt):
    """Astra's o1 review 3: the arms of a set are paired by battle, so every arm must have played the same battles -
    the same starts, torch chunk seeds, chunk and simulator (its run key), and the same battle keys where the file
    carries them (files before 22.11 do not: their key's 'starts' hash covers the specs in order); else a halt."""
    ref = None
    for p, d in pl.items():
        k = d.get("key") or {}
        sig = (k.get("starts"), tuple(k.get("seeds") or ()), k.get("chunk"), k.get("sim_key"), k.get("tactic"),
               k.get("visible"), k.get("fair"), k.get("device"), json.dumps(d.get("battle_keys")))
        if None in (sig[0], sig[2], sig[3]):
            raise SystemExit(f"set {nm}: arm {p} has no run key - cannot pair it by battle")
        if ref is None:
            ref = (p, sig)
        elif sig != ref[1]:
            raise SystemExit(f"set {nm}: arm {p} played other battles than {ref[0]} ({sig} vs {ref[1]}) - not paired")
        if "battle_keys" in d and len(d["battle_keys"]) != len(mt):
            raise SystemExit(f"set {nm}: arm {p} has {len(d['battle_keys'])} battle keys, the set {len(mt)}")
        if "battle_keys" in d and [x["key"] for x in mt if "key" in x] not in ([], d["battle_keys"]):
            raise SystemExit(f"set {nm}: arm {p}'s ordered battle keys differ from the set's metadata")


def table(D, names=SETS):
    D = Path(D)
    c = LN.conf()
    limit = int(c["limit_s"])
    out = {}
    for nm in names:
        files = {p: D / f"{nm}_{p}.json" for p in TABLE_PLAYERS if (D / f"{nm}_{p}.json").exists()}
        if not files:
            continue
        pl = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in files.items()}
        mt = json.loads((D / f"{nm}_meta.json").read_text(encoding="utf-8"))
        _same_battles(nm, pl, mt)
        R = {p: reads(d, limit) for p, d in pl.items()}
        comer = {b for p in ("TEACH", "DUMB", "NET") if p in R for b, x in enumerate(R[p]["caught"]) if x and x > 0}
        tab = {"battles": len(mt), "comer_battles": len(comer), "groups": {}, "pairs": {}}
        for lab, sel in groups_of(mt, nm, comer):
            row = {}
            for p, r in R.items():
                row[p] = {"contact": share([r["contact"][b] for b in sel]), "changed": share([r["changed"][b] for b in sel]),
                          "j600": _mean([r["j600"][b] for b in sel]), "j600_nl": _mean([r["j600_nl"][b] for b in sel]),
                          "win": _mean([r["win"][b] for b in sel]), "orders": _mean([r["orders"][b] for b in sel]),
                          "orders_units": _mean([r["orders_units"][b] for b in sel]),
                          "lord_orders": _mean([r["lord_orders"][b] for b in sel]),
                          "recv_stand": _mean([r["recv"][b] for b in sel]),
                          "t_contact": _mean([r["t_contact"][b] for b in sel if r["contact"][b]])}
            tab["groups"][lab] = row
            for x, y in TABLE_PAIRS:
                if x in R and y in R:
                    tab["pairs"].setdefault(f"{x} - {y}", {})[lab] = {
                        k: paired(R[x][k], R[y][k], sel) for k in ("j600", "j600_nl", "j300", "win")}
        if nm == "held":
            tab["binds"] = held_binds(mt, pl.get("DEFAULT") or next(iter(pl.values())))
        tab["twin_pose"] = twin_pose(mt, pl)
        tab["twin_quality"] = twin_quality(mt, R)
        out[nm] = tab
    out["lines"] = lines(out)
    if any("NET" in (out[nm].get("groups", {}).get("all") or {}) for nm in out if nm != "lines"):
        out["lines_NET"] = lines(out, "NET")
    (D / "exam_o1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_table(out)
    return out


def held_binds(mt, d):
    """{held axis: {scenes, binds}} - the spec's binding (scenes_o1.binds), the gap's from play: the level's first-tick
    front gap of the cause scene within +-20 m of the held value."""
    first = {}
    for r in d["ticks"]:
        first.setdefault(r[0], r[6])
    out = {}
    for b, x in enumerate(mt):
        if x["held"] is None or not x["cause"]:
            continue
        ok = bool(x["binds"])
        if x["held"] == "gap_m":
            v = SO.cs()["axes"]["gap_m"]["held"][0][0]
            ok = ok and b in first and abs(first[b] - v) <= 20.0
        r = out.setdefault(x["held"], {"scenes": 0, "binds": 0})
        r["scenes"] += 1
        r["binds"] += int(ok)
    gp = [first[b] for b, x in enumerate(mt) if x["held"] == "gap_m" and x["cause"] and b in first]
    if gp:
        out["gap_m"]["first_tick_gap_m"] = [round(min(gp), 1), round(sorted(gp)[len(gp) // 2], 1), round(max(gp), 1)]
    return out


def twin_pose(mt, pl):
    """The come twins against their cause scenes at the decision tick: the level's input row (non-motion columns) the
    same? - the largest |difference| of the gap (m) and of the E position columns, by pair (the first tick's rows of
    DEFAULT: both scenes still untouched by any player)."""
    d = pl.get("DEFAULT")
    if d is None:
        return None
    t0 = int(LN.conf()["start_s"])
    first = {r[0]: r for r in d["ticks"] if r[1] == t0}
    key = {}
    for b, x in enumerate(mt):
        if x["seed"] is not None and (x["cause"] or x["twin"] == "come"):
            key.setdefault((x["seed"], x["held"]), {})["come" if x["twin"] == "come" else "cause"] = b
    cols = TICK_COLS
    gi, er, ef, ec = cols.index("gap"), cols.index("e_right"), cols.index("e_fwd"), cols.index("e_close_max")
    dg, dp, closing = [], [], []
    for k, v in key.items():
        if "come" in v and "cause" in v and v["come"] in first and v["cause"] in first:
            a, b = first[v["cause"]], first[v["come"]]
            dg.append(abs(a[gi] - b[gi]))
            dp.append(max(abs(a[er] - b[er]), abs(a[ef] - b[ef])) * LN.PLACE_M)
            closing.append(b[ec])
    if not dg:
        return None
    q = lambda v, p: round(sorted(v)[min(len(v) - 1, int(p * len(v)))], 2)     # noqa: E731
    return {"pairs": len(dg), "gap_diff_m": [q(dg, 0.5), q(dg, 0.9), round(max(dg), 2)],
            "e_centroid_diff_m": [q(dp, 0.5), q(dp, 0.9), round(max(dp), 2)],
            "come_e_close_max_ms": [round(min(closing), 2), q(closing, 0.5)],
            "come_closing_share": round(sum(x >= float(LN.conf()["comes_ms"]) for x in closing) / len(closing), 4)}


def _g(out, nm, lab, p, k):
    return ((out.get(nm, {}).get("groups", {}).get(lab) or {}).get(p) or {}).get(k)


def _pool_share(parts):
    parts = [x for x in parts if x]
    if not parts:
        return None
    n = sum(x["n"] for x in parts)
    p = sum(x["p"] * x["n"] for x in parts) / n
    return {"p": round(p, 4), "two_errors": round(2 * math.sqrt(max(p * (1 - p), 1e-12) / n), 4), "n": n}


def _pool_pair(parts):
    parts = [x for x in parts if x]
    if not parts:
        return None
    n = sum(x["n"] for x in parts)
    m = sum(x["diff"] * x["n"] for x in parts) / n
    ss = sum(((x["two_errors"] / 2) ** 2 * x["n"]) * (x["n"] - 1) + x["n"] * (x["diff"] - m) ** 2 for x in parts)
    sd = math.sqrt(ss / max(n - 1, 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def _pp(out, nm, lab, pair="TEACH - DEFAULT", k="j600"):
    return ((out.get(nm, {}).get("pairs", {}).get(pair) or {}).get(lab) or {}).get(k)


def twin_quality(mt, R):
    """The come twins in play at the decision tick (DEFAULT's: the level has not moved anything): each battle's largest
    enemy miss to its cause pose (m) and slowest enemy speed (m/s); the enemy infantry's mean fatigue state, come
    twins against their cause scenes (the review 2: position and fatigue may give the label away)."""
    r = R.get("DEFAULT")
    if r is None:
        return None
    come = [b for b, x in enumerate(mt) if x["twin"] == "come" and r["twin_miss"][b] is not None
            and math.isfinite(r["twin_miss"][b])]
    cause = [b for b, x in enumerate(mt) if x["cause"]]
    if not come:
        return None
    q = lambda v, p: round(sorted(v)[min(len(v) - 1, int(p * len(v)))], 2)     # noqa: E731
    mi, vm = [r["twin_miss"][b] for b in come], [r["twin_v"][b] for b in come]
    return {"n": len(come), "miss_max_m": [q(mi, 0.5), q(mi, 0.9), round(max(mi), 2)],
            "v_min_ms": [round(min(vm), 2), q(vm, 0.5)],
            "e_fat_come": _mean([r["e_fat"][b] for b in come]), "e_fat_cause": _mean([r["e_fat"][b] for b in cause])}


PAIRS_MIN = 100           # (line 2: valid no-cause pairs needed to judge; fewer - 'n too small, not judged')
FULL = {"taught": 2 * 192, "held": 2 * 6 * 32, "G": 384}     # (the chain's set sizes: lines 1, 4-6 need the full sets)


def _ax_gap(t, h):
    gap = round(h["p"] - t["p"], 4)
    e2 = round(math.sqrt(t["two_errors"] ** 2 + h["two_errors"] ** 2), 4)
    return {"held": h, "gap": gap, "two_errors": e2, "passed": gap >= -e2}


def lines(out, arm="TEACH"):
    """The lines written before the run (module doc); each with its n; passed None = not judged (n too small or a set
    not at the chain's size). arm: the player in TEACH's place (the exam after: NET on the same lines)."""
    res = {}
    X = " [-comer]"
    ST = ("taught", "held")
    full = {nm: (out.get(nm) or {}).get("battles") == n for nm, n in FULL.items()}
    note = None if all(full.values()) else "sets not at the chain's size %s: %s - not judged" % (
        FULL, {nm: (out.get(nm) or {}).get("battles") for nm in FULL})
    tc = _pool_share([_g(out, nm, "cause", arm, "contact") for nm in ST])
    dc = _pool_share([_g(out, nm, "cause", "DEFAULT", "contact") for nm in ST])
    ok1 = bool(tc and dc and tc["p"] >= 0.95 and dc["p"] <= 0.05)
    res["1"] = {"rule": "cause scenes: contact before the timer TEACH >= 0.95, DEFAULT <= 0.05", arm: tc, "DEFAULT": dc,
                f"{arm}_-comer": _pool_share([_g(out, nm, "cause" + X, arm, "contact") for nm in ST]),
                "n": None if tc is None else tc["n"],
                "passed": ok1 if full["taught"] and full["held"] else None, "note": note}
    chg = _pool_share([_g(out, nm, "no cause", arm, "changed") for nm in ST])
    tr = {k: _pool_pair([_pp(out, nm, "no cause" + X, f"{arm} - DEFAULT", k=k) for nm in ST]) for k in ("j600", "j600_nl")}
    tr_all = _pool_pair([_pp(out, nm, "no cause", f"{arm} - DEFAULT") for nm in ST])
    npair = 0 if tr["j600"] is None else tr["j600"]["n"]
    ok_tr = tr["j600"] is not None and (abs(tr["j600"]["diff"]) <= tr["j600"]["two_errors"] or tr["j600"]["diff"] == 0)
    res["2"] = {"rule": "no-cause scenes: TEACH changes the order in <= 3 % of battles; trade (lords in) TEACH - DEFAULT "
                        "within +-2 errors on >= 100 valid pairs, the comer battles out (all, and without the lords beside)",
                "changed": chg, "trade_-comer": tr, "trade_all": tr_all, "n_pairs": npair,
                "by_twin": {t: {"changed": _pool_share([_g(out, nm, "no cause, " + t, arm, "changed") for nm in ST]),
                                "trade": _pool_pair([_pp(out, nm, "no cause, " + t, f"{arm} - DEFAULT") for nm in ST])} for t in ("come", "role")},
                "passed": (bool(chg and chg["p"] <= 0.03 and ok_tr) if npair >= PAIRS_MIN else None),
                "note": None if npair >= PAIRS_MIN else "n too small (%d < %d pairs) - not judged" % (npair, PAIRS_MIN)}
    d3 = _pool_pair([_pp(out, nm, "cause", f"{arm} - DUMB") for nm in ST])
    res["3"] = {"rule": "TEACH - DUMB on cause scenes: reported, not gated",
                "all": {k: _pool_pair([_pp(out, nm, "cause", f"{arm} - DUMB", k) for nm in ST])
                        for k in ("j600", "j600_nl", "j300", "win")},
                "-comer": {k: _pool_pair([_pp(out, nm, "cause" + X, f"{arm} - DUMB", k) for nm in ST])
                           for k in ("j600", "j600_nl", "j300", "win")},
                f"{arm} - DEFAULT, cause": {k: _pool_pair([_pp(out, nm, "cause", f"{arm} - DEFAULT", k) for nm in ST])
                                           for k in ("j600", "j600_nl", "win")},
                "recv_stand": {p: [_g(out, nm, "cause", p, "recv_stand") for nm in ST] for p in (arm, "DUMB")},
                "n": None if d3 is None else d3["n"]}
    om = {nm: _g(out, nm, "all", arm, "orders") for nm in ("taught", "held", "G") if nm in out}
    res["4"] = {"rule": "TEACH's orders to the game a unit-minute WITH the lord (his first HOLD and MOVEs over the "
                        "infantry's and his standing minutes) <= 1.2; the infantry alone beside",
                "with_lord": om, "infantry_alone": {nm: _g(out, nm, "all", arm, "orders_units") for nm in om},
                "lord_orders_a_battle": {nm: _g(out, nm, "all", arm, "lord_orders") for nm in om},
                "n": {nm: (out.get(nm) or {}).get("battles") for nm in om},
                "passed": (bool(om) and all(v is not None and v <= 1.2 for v in om.values())) if all(full.values())
                else None, "note": note}
    t = _g(out, "taught", "cause", arm, "contact")
    h = _g(out, "held", "cause", arm, "contact")
    if t and h:
        axes = {}
        for ax in SO.HELD_AXES:
            ha = _g(out, "held", "held %s, cause" % ax, arm, "contact")
            if ha:
                axes[ax] = _ax_gap(t, ha)
        pooled = _ax_gap(t, h)
        res["5"] = {"rule": "held - taught, contact share of TEACH on cause scenes: not below -2 errors, pooled AND on "
                            "each held axis", "taught": t, "pooled": pooled, "by_axis": axes,
                    "n": {"taught": t["n"], "held": h["n"], **{ax: v["held"]["n"] for ax, v in axes.items()}},
                    "passed": (pooled["passed"] and len(axes) == len(SO.HELD_AXES) and all(v["passed"] for v in axes.values()))
                    if full["taught"] and full["held"] else None, "note": note}
    g = _g(out, "G", "all", arm, "contact")
    gd = _g(out, "G", "all", "DEFAULT", "contact")
    gh = _g(out, "G_hold", "all", arm, "contact")
    res["6"] = {"rule": "G (att 1 against a holder): TEACH's stalls (no contact before the timer) <= 0.05",
                f"{arm}_stall": None if g is None else round(1 - g["p"], 4), "n": None if g is None else g["n"],
                "DEFAULT_stall": None if gd is None else round(1 - gd["p"], 4),
                f"G_hold_{arm}_stall (reported)": None if gh is None else round(1 - gh["p"], 4),
                "G_hold_n": None if gh is None else gh["n"], f"{arm} - DEFAULT j600 (G)": _pp(out, "G", "all", f"{arm} - DEFAULT"),
                "passed": bool(g and 1 - g["p"] <= 0.05) if full["G"] else None, "note": note}
    bd = (out.get("held") or {}).get("binds") or {}
    res["binds"] = {"rule": "every held axis binds in all its cause scenes", "by_axis": bd,
                    "n": sum(v["scenes"] for v in bd.values()) if bd else 0,
                    "passed": bool(bd) and len(bd) == len(SO.HELD_AXES)
                    and all(v["binds"] == v["scenes"] for v in bd.values())}
    return res


def _f(p):
    if p is None:
        return "-"
    if "p" in p:
        return f"{p['p']:.3f} ± {p['two_errors']:.3f} ({p['n']})"
    return f"{p['diff']:+.4f} ± {p['two_errors']:.4f} ({p['n']})"


def print_table(out):
    for nm, tab in out.items():
        if nm.startswith("lines"):           # (lines and lines_NET)
            continue
        print(f"\n## set {nm} ({tab['battles']} battles, comer battles {tab['comer_battles']}); twin poses {tab.get('twin_pose')}")
        print(f"  twin quality (in play): {tab.get('twin_quality')}")
        for lab, row in tab["groups"].items():
            print(f"  {lab}: " + " | ".join(f"{p} contact {_f(r['contact'])} changed {_f(r['changed'])} j600 {r['j600']} "
                                            f"nl {r['j600_nl']} win {r['win']} ord {r['orders']}" for p, r in row.items()))
        for pair, rows in tab["pairs"].items():
            for lab, r in rows.items():
                if lab in ("all", "cause", "no cause", "cause [-comer]", "no cause [-comer]"):
                    print(f"  {pair} {lab}: j600 {_f(r['j600'])} | nl {_f(r['j600_nl'])} | j300 {_f(r['j300'])} | win {_f(r['win'])}")
    print("\nlines:", json.dumps(out["lines"], indent=1))
    for k, v in out["lines"].items():
        print(f"LINE {k}: passed {v.get('passed', 'reported')} | n {json.dumps(v.get('n', v.get('n_pairs')))}"
              f"{' | ' + v['note'] if v.get('note') else ''}")


# ------------------------------------------------------------------ METHOD step 6: acceptance by episode (23.11)
TICK_S = 15               # (one operative tick: o1.decide_s)
ACCEPT = {"B1": 0.85, "B1_axis": 0.75, "B1_axis_gap": 0.10, "aim_minus": 0.10, "quiet_plus": 0.03,
          "orders_x_human": 1.5, "human_orders_min": 1.0, "neg_max": 0.5, "harm": -0.02}
_ACCEPT_WHY = ("METHOD.md step 6 (22.11) as the coordinator wrote it for O0-1 on 23.11 (tmp/engine/b/LINES_o1.md, "
               "written before the run); human_orders_min 1.0: game/vision/README.md 'orders a minute (human ~1)', "
               "net/common/reissue.py - an adopted norm, not a measurement of these sets")
# (the required support, Astra o1_train_review3 4: every set at its full size, every arm present; short = NOT JUDGED)
REQUIRED = {"taught": {"n": 384, "cause": 192, "nocause": 192}, "held": {"n": 384, "cause": 192, "nocause": 192,
                                                                         "axis": 32},
            "G": {"n": 384}, "G_hold": {"n": 64}, "G_def": {"n": 256}}
ARMS = ("DEFAULT", "TEACH", "NET")
FRESH_SETS = ("taught", "held")


class NotJudged(Exception):
    pass


def _ls():
    return LN.POINTERS.index("LS")


def _b1_own(d, cause_ids):
    """B1 on the arm's OWN path (o1_train_review3 5): per cause battle t0 = the teacher's first LS label on this
    path's tick rows (the whole path), tN = the arm's first executed LS; a hit iff both and |tN - t0| <= TICK_S; no
    teacher label on the path / no move: a miss (counted beside). Returns (hits [0|1] in cause_ids order, counts)."""
    t0, tN = {}, {}
    ls = _ls()
    for r in d["ticks"]:
        b, sec, lab, act = r[0], r[1], r[2], r[3]
        if lab and b not in t0:
            t0[b] = sec
        if act == ls and b not in tN:
            tN[b] = sec
    hits, c = [], {"no_label": 0, "no_move": 0, "early": 0, "late": 0}
    for b in cause_ids:
        a, n = t0.get(b), tN.get(b)
        h = a is not None and n is not None and abs(n - a) <= TICK_S
        if a is None:
            c["no_label"] += 1
        elif n is None:
            c["no_move"] += 1
        elif not h:
            c["early" if n < a else "late"] += 1
        hits.append(int(h))
    return hits, c


def _p(hits):
    """An unrounded share and its two errors (printing only)."""
    n = len(hits)
    p = sum(hits) / n
    return {"k": int(sum(hits)), "n": n, "p": p, "two_errors": 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)}


def _pair(a, b, sel, what):
    """Unrounded paired NET - ref over EVERY battle of sel; a missing / non-finite value: NOT JUDGED."""
    d = []
    for i in sel:
        x, y = a[i], b[i]
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            raise NotJudged(f"{what}: battle {i} has no finite value")
        d.append(x - y)
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(n - 1, 1))
    return {"diff": m, "two_errors": 2 * sd / math.sqrt(n), "n": n}


def _load_arms(D, F, run_sha):
    """{(set, arm)}: every required arm of every set, authenticated (o1_train_review3 3): the set's metadata at its
    full size; the arms paired by battle against it (_same_battles: starts, seeds, chunk, sim, tactic, visibility, fair,
    device, ordered battle keys) - FRESH against D's metadata too; every NET arm keyed net_sha = the run's final.pt
    sha, every FRESH arm net_sha 'fresh'; per battle arrays and the 600 s series complete."""
    c = LN.conf()
    i1 = int(c["start_s"]) // O1Watch.SERIES_S + int(c["window_s"]) // O1Watch.SERIES_S
    arms, mts = {}, {}
    for nm, req in REQUIRED.items():
        mf = D / f"{nm}_meta.json"
        if not mf.exists():
            raise NotJudged(f"{nm}: no metadata")
        mt = json.loads(mf.read_text(encoding="utf-8"))
        if len(mt) != req["n"]:
            raise NotJudged(f"{nm}: {len(mt)} battles, required {req['n']}")
        if "cause" in req and (sum(x["cause"] for x in mt) != req["cause"]
                               or sum(not x["cause"] for x in mt) != req["nocause"]):
            raise NotJudged(f"{nm}: cause / no-cause counts differ from {req}")
        if "axis" in req:
            for ax in SO.HELD_AXES:
                k = sum(1 for x in mt if x["cause"] and x["held"] == ax)
                if k != req["axis"]:
                    raise NotJudged(f"held: axis {ax} has {k} cause scenes, required {req['axis']}")
        mts[nm] = mt
        pl = {}
        for p in ARMS + (("FRESH",) if nm in FRESH_SETS else ()):
            f = (F / f"{nm}_NET.json") if p == "FRESH" else (D / f"{nm}_{p}.json")
            if not f.exists():
                raise NotJudged(f"{nm}: arm {p} missing ({f})")
            pl[p] = json.loads(f.read_text(encoding="utf-8"))
        if "FRESH" in pl:
            fm = F / f"{nm}_meta.json"
            if not fm.exists() or json.loads(fm.read_text(encoding="utf-8")) != mt:
                raise NotJudged(f"{nm}: the FRESH folder's metadata is not D's")
        try:
            _same_battles(nm, pl, mt)
        except SystemExit as e:
            raise NotJudged(str(e))
        for p, d in pl.items():
            sha = (d.get("key") or {}).get("net_sha")
            if p == "NET" and (run_sha is None or sha != run_sha):
                raise NotJudged(f"{nm}: NET arm net_sha {sha} != the run's final.pt {run_sha}")
            if p == "FRESH" and sha != "fresh":
                raise NotJudged(f"{nm}: FRESH arm net_sha {sha} != 'fresh'")
            if p in ("DEFAULT", "TEACH") and sha is not None:
                raise NotJudged(f"{nm}: arm {p} carries a network sha {sha}")
            B = len(mt)
            if len(d["ser_l"]) != B or any(len(v) != B for v in d["per_battle"].values() if isinstance(v, list)):
                raise NotJudged(f"{nm}: arm {p} has per-battle arrays not of {B}")
            if any(r is None or len(r) <= i1 for r in d["ser_l"]):
                raise NotJudged(f"{nm}: arm {p} has a series shorter than the 600 s window")
            arms[(nm, p)] = d
    return arms, mts


def accept(D, fresh=None, run=None):
    """METHOD step 6's lines for the NET arm of D (tmp/engine/b/LINES_o1.md): verdict PASSED / FAILED / NOT JUDGED
    (an arm missing, short, unauthenticated or unpaired: NOT JUDGED, never a pass); decisions on unrounded numbers."""
    D = Path(D)
    F = Path(fresh) if fresh else Path(str(D) + "_fresh")
    A = ACCEPT
    lim = int(LN.conf()["limit_s"])
    run_sha = None
    if run:
        rj = json.loads((Path(run) / "run.json").read_text(encoding="utf-8"))
        fp = Path(run) / "final.pt"
        if rj.get("passed") and rj.get("card") == "o0_1" and fp.exists():
            import hashlib
            s = hashlib.sha256(fp.read_bytes()).hexdigest()
            run_sha = s if s == rj.get("sha256") else None
    L, err = {}, None
    try:
        arms, mts = _load_arms(D, F, run_sha)
        R = {k: reads(v, lim) for k, v in arms.items()}
        cause = {nm: [b for b, x in enumerate(mts[nm]) if x["cause"]] for nm in FRESH_SETS}
        nocause = {nm: [b for b, x in enumerate(mts[nm]) if not x["cause"]] for nm in FRESH_SETS}
        b1 = {}
        for p in ("NET", "FRESH", "TEACH"):
            hits, cnt = [], {}
            for nm in FRESH_SETS:
                h, c = _b1_own(arms[(nm, p)], cause[nm])
                hits += h
                cnt = {k: cnt.get(k, 0) + v for k, v in c.items()}
            b1[p] = (hits, cnt)
        x = _p(b1["NET"][0])
        L["B1"] = {"NET": x, "counts": b1["NET"][1], "TEACH": _p(b1["TEACH"][0]), "passed": x["p"] >= A["B1"]}
        bt = _p(_b1_own(arms[("taught", "NET")], cause["taught"])[0])
        ax = {}
        for a_ in SO.HELD_AXES:
            sel = [b for b in cause["held"] if mts["held"][b]["held"] == a_]
            bx = _p(_b1_own(arms[("held", "NET")], sel)[0])
            ax[a_] = {"held": bx, "gap": bt["p"] - bx["p"],
                      "passed": bx["p"] >= A["B1_axis"] and bt["p"] - bx["p"] <= A["B1_axis_gap"]}
        L["B1_axes"] = {"taught": bt, "by_axis": ax, "passed": all(v["passed"] for v in ax.values())}
        sh = lambda p, key, sel_of: _p([R[(nm, p)][key][b] for nm in FRESH_SETS for b in sel_of[nm]])  # noqa: E731
        cN, cT = sh("NET", "contact", cause), sh("TEACH", "contact", cause)
        L["aim"] = {"NET": cN, "TEACH": cT, "passed": cN["p"] >= cT["p"] - A["aim_minus"]}
        qN, qT = sh("NET", "changed", nocause), sh("TEACH", "changed", nocause)
        gN, gT = _p(R[("G_def", "NET")]["changed"]), _p(R[("G_def", "TEACH")]["changed"])
        L["quiet"] = {"twins": {"NET": qN, "TEACH": qT}, "G_def": {"NET": gN, "TEACH": gT},
                      "passed": qN["p"] <= qT["p"] + A["quiet_plus"] and gN["p"] <= gT["p"] + A["quiet_plus"]}
        cap = A["orders_x_human"] * A["human_orders_min"]
        om = {}
        for nm in REQUIRED:
            v = R[(nm, "NET")]["orders"]
            if any(x is None or not math.isfinite(x) for x in v):
                raise NotJudged(f"{nm}: NET orders not finite in every battle")
            om[nm] = sum(v) / len(v)
        L["orders"] = {"NET": om, "cap": cap, "passed": all(v <= cap for v in om.values())}
        xf = _p(b1["FRESH"][0])
        L["NEG"] = {"FRESH": xf, "counts": b1["FRESH"][1], "passed": xf["p"] <= A["neg_max"]}
        harm = {}
        for k in ("j300", "j600"):
            harm[f"G_def {k}"] = _pair(R[("G_def", "NET")][k], R[("G_def", "DEFAULT")][k],
                                       range(len(mts["G_def"])), f"G_def {k}")
            dd = []
            for nm in FRESH_SETS:
                p_ = _pair(R[(nm, "NET")][k], R[(nm, "DEFAULT")][k], nocause[nm], f"{nm} twins {k}")
                dd.append(p_)
            n = sum(q["n"] for q in dd)
            m = sum(q["diff"] * q["n"] for q in dd) / n
            harm[f"twins {k}"] = {"diff": m, "n": n, "parts": dd}
        L["harm"] = {"cells": harm, "known_harm": [k for k, v in harm.items() if A["harm"] < v["diff"] < 0],
                     "passed": all(v["diff"] > A["harm"] for v in harm.values()),
                     "scope": "harm WITHOUT the cause only (G_def, the no-cause twins) - not the stage's safety; G "
                              "is the trick's price, printed (the coordinator's ruling 23.11)"}
        pr = {f"G NET - DEFAULT {k}": _pair(R[("G", "NET")][k], R[("G", "DEFAULT")][k], range(len(mts["G"])),
                                             f"G {k}") for k in ("j300", "j600")}
        pr.update({f"G TEACH - DEFAULT {k}": _pair(R[("G", "TEACH")][k], R[("G", "DEFAULT")][k],
                                                    range(len(mts["G"])), f"G {k}") for k in ("j300", "j600")})
        pr["cause NET - TEACH j600"] = {nm: _pair(R[(nm, "NET")]["j600"], R[(nm, "TEACH")]["j600"], cause[nm],
                                                   f"{nm} cause") for nm in FRESH_SETS}
        L["printed"] = pr
        gated = [k for k in L if k != "printed"]
        verdict = "PASSED" if all(L[k]["passed"] for k in gated) else "FAILED"
        failed = [k for k in gated if not L[k]["passed"]]
    except NotJudged as e:
        verdict, failed, err = "NOT JUDGED", [], str(e)
    out = {"verdict": verdict, "failed": failed, "not_judged": err, "lines": L, "accept": A, "required": REQUIRED,
           "run_sha256": run_sha, "why": _ACCEPT_WHY,
           "scope": "numeric lines only: the three battles by eye and the manual's traps are not part of this verdict"}
    tmp = D / "accept_o1.json.part"
    tmp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    import os
    os.replace(tmp, D / "accept_o1.json")
    for k, v in L.items():
        print(k, json.dumps(v, default=str)[:700], flush=True)
    print("ACCEPT", verdict, failed, err or "", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "table", "preflight", "accept"))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--players", default=",".join(PLAYERS))
    ap.add_argument("--out", default="tmp/o1/exam")
    ap.add_argument("--n", type=int, default=0, help="a smaller set: n cause scenes (held: n / 6 an axis; G: 2n)")
    ap.add_argument("--chunk", type=int, default=384)
    ap.add_argument("--tactic", default="TEACH", help="the tactic level's player (tactic/pin.PLAYERS)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sim", default="")
    ap.add_argument("--visible", type=int, default=0, help="the level's facts on the seen foes only (the engine's)")
    ap.add_argument("--fair", type=int, default=0, help="taught / held through net/operative/scenes_fair (the engine's)")
    ap.add_argument("--pin-device", type=int, default=0, help="the device type in every arm's key")
    ap.add_argument("--run", default="", help="accept: the training's run folder (run.json + final.pt)")
    ap.add_argument("--fresh", default="", help="accept: the untrained network's exam folder (default <out>_fresh)")
    ap.add_argument("--net", default="fresh", help="player NET: 'fresh' (a new OpNet1 = DEFAULT) or a save_o1 checkpoint")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "preflight":
        return preflight(a)
    if a.cmd == "accept":                       # (METHOD step 6: exit 0 PASSED, 1 FAILED, 2 NOT JUDGED, 3 error)
        try:
            v = accept(a.out, fresh=a.fresh or None, run=a.run or None)["verdict"]
        except Exception as e:                  # (a crash is not a measurement: o1_train_review3 4)
            print(f"ACCEPT ERROR {type(e).__name__}: {e}", flush=True)
            raise SystemExit(3)
        raise SystemExit({"PASSED": 0, "FAILED": 1}.get(v, 2))
    out = table(a.out)
    bad = [k for k, v in (out.get("lines") or {}).items() if isinstance(v, dict) and v.get("passed") is False]
    if bad:                     # (Astra's second o1 review 4: a failed line exits non-zero)
        print(f"TABLE: lines failed {bad}", flush=True)
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main()
