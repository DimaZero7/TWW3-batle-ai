"""Drill O0-1 'advance on the holder' - the exam before training (METHOD.md recipe step 4; tmp/op1/design_merged.md
FINAL, tmp/op1/design_opus.md 6): three players of the operative level on the SAME battles, paired by battle -
    DEFAULT  the stack without the operative (today: the zone never moves - KEEP always)
    DUMB     march onto his line once (opside's arm M: Hold(LM) at the first tick with att 1)
    TEACH    the drill's teacher (advance.teach: ADVANCE -> Hold(L60) / KEEP)
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
SETS = ("taught", "held", "G", "G_hold")
TICK_COLS = ("b", "sec", "label", "act", "reason", "masked", "gap", "e_close_max") + LN.INPUT


class O1Side(BA.TacticSide):
    """My side: the tactic level (TacticSide) + the operative level of this drill (lines.Level under an advance player)
    after the tactic's tick every decide_s from start_s (the bridge's order: tactic, then the operative); the hidden lord
    follows after the zone's first move (lord HOLD once, then KEEP; a MOVE at the second of a new point - as the bridge)."""

    def __init__(self, tactic_player, op_player, specs, seed=None, zones=None):
        super().__init__(tactic_player, hygiene=True, zones=zones, seed=seed)
        self.op, self.specs = op_player, specs

    def start(self, st, P, device, card, unit_net):
        from net.tactic import scenes as TSN
        super().start(st, P, device, card, unit_net)
        B = st.B
        self.c = LN.conf()
        self.level = LN.Level(B, [s["o1"]["att"] for s in self.specs], [TSN._blocked(s["map"]) for s in self.specs])
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
            ptr = self.op(f, order, c)
            masked = ptr is not None and f["melee"]
            if masked:
                ptr = None
                self.n_masked[b] += 1
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

    def orders(self, st, j):
        o = super().orders(st, j)
        if not bool(self.moved_t.any()):
            return o
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


def play(name, specs, seeds, chunk, device, tactic="TEACH", compile=None):
    """{per_battle {key: [B]}, ser_l, ser_nl [B][T], ticks [rows TICK_COLS, b the set's index]} of the player name on
    specs (chunk k under torch seed seeds[k]; on CUDA every batch padded to a full chunk: one compiled shape a
    per-side size)."""
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
        mine = O1Side(EP.build_under(tactic), AD.PLAYERS[name](), batch, seed=seed, zones=zones)
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
        print(json.dumps({"player": name, "chunk": ci, "battles": len(ii), "wall_s": stt["wall_s"],
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out["tick_cols"] = list(TICK_COLS)
    return out


def sets_of(names, n=None):
    S = SO.cs()
    out = {}
    for k, nm in enumerate(names):
        if nm == "taught":
            sp = SO.taught_set(n)
        elif nm == "held":
            sp = SO.held_set(None if n is None else max(n // len(SO.HELD_AXES), 2))
        elif nm == "G":
            sp = SO.general_set(None if n is None else 2 * n)
        elif nm == "G_hold":
            sp = SO.general_set(None if n is None else n // 2, hold=True)
        else:
            raise ValueError(nm)
        out[nm] = (sp, int(S["tseed"]) + 100 * k)
    return out


def meta(specs):
    return [{"cause": s["o1"]["cause"], "twin": s["o1"]["twin"], "att": s["o1"]["att"], "held": s.get("held"),
             "binds": SO.binds(s), "size": len(s["mine"]), "enemy": s["o1"]["enemy"], "comer": s["o1"].get("comer"),
             "sally_m": s["o1"].get("sally_m"), "axes": s.get("axes"), "seed": s.get("seed"), "map": s["map"],
             "seed_used": s.get("seed_used"),
             "pose": s["o1"].get("pose")} for s in specs]


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
    out = {}
    for nm, (sp, _) in sets_of(a.sets.split(","), a.n or None).items():
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
    for nm, (sp, t0) in sets_of(a.sets.split(","), a.n or None).items():
        _write(D / f"{nm}_meta.json", meta(sp))
        ch = a.chunk
        seeds = [t0 + k for k in range((len(sp) + ch - 1) // ch)]
        key = run_key(sp, ch, a.tactic, seeds)
        for name in a.players.split(","):
            f = D / f"{nm}_{name}.json"
            if f.exists():
                old = json.loads(f.read_text(encoding="utf-8")).get("key") or {}
                bad = sorted(k for k in key if old.get(k) != key[k])
                if bad:
                    raise SystemExit(f"{f}: kept file of another run - differs in {bad}; move it away or use another --out")
                print(json.dumps({"set": nm, "player": name, "kept": str(f)}), flush=True)
                continue
            t1 = time.time()
            res = play(name, sp, seeds, ch, a.device, tactic=a.tactic)
            res.update(sim_key=EP.sim_key(), sim=a.sim, tactic=a.tactic, seeds=seeds, key=key)
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


def table(D, names=SETS):
    D = Path(D)
    c = LN.conf()
    limit = int(c["limit_s"])
    out = {}
    for nm in names:
        files = {p: D / f"{nm}_{p}.json" for p in PLAYERS if (D / f"{nm}_{p}.json").exists()}
        if not files:
            continue
        pl = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in files.items()}
        mt = json.loads((D / f"{nm}_meta.json").read_text(encoding="utf-8"))
        R = {p: reads(d, limit) for p, d in pl.items()}
        comer = {b for p in ("TEACH", "DUMB") if p in R for b, x in enumerate(R[p]["caught"]) if x and x > 0}
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
            for x, y in PAIRS:
                if x in R and y in R:
                    tab["pairs"].setdefault(f"{x} - {y}", {})[lab] = {
                        k: paired(R[x][k], R[y][k], sel) for k in ("j600", "j600_nl", "j300", "win")}
        if nm == "held":
            tab["binds"] = held_binds(mt, pl.get("DEFAULT") or next(iter(pl.values())))
        tab["twin_pose"] = twin_pose(mt, pl)
        tab["twin_quality"] = twin_quality(mt, R)
        out[nm] = tab
    out["lines"] = lines(out)
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


def lines(out):
    """The lines written before the run (module doc); each with its n; passed None = not judged (n too small or a set
    not at the chain's size)."""
    res = {}
    X = " [-comer]"
    ST = ("taught", "held")
    full = {nm: (out.get(nm) or {}).get("battles") == n for nm, n in FULL.items()}
    note = None if all(full.values()) else "sets not at the chain's size %s: %s - not judged" % (
        FULL, {nm: (out.get(nm) or {}).get("battles") for nm in FULL})
    tc = _pool_share([_g(out, nm, "cause", "TEACH", "contact") for nm in ST])
    dc = _pool_share([_g(out, nm, "cause", "DEFAULT", "contact") for nm in ST])
    ok1 = bool(tc and dc and tc["p"] >= 0.95 and dc["p"] <= 0.05)
    res["1"] = {"rule": "cause scenes: contact before the timer TEACH >= 0.95, DEFAULT <= 0.05", "TEACH": tc, "DEFAULT": dc,
                "TEACH_-comer": _pool_share([_g(out, nm, "cause" + X, "TEACH", "contact") for nm in ST]),
                "n": None if tc is None else tc["n"],
                "passed": ok1 if full["taught"] and full["held"] else None, "note": note}
    chg = _pool_share([_g(out, nm, "no cause", "TEACH", "changed") for nm in ST])
    tr = {k: _pool_pair([_pp(out, nm, "no cause" + X, k=k) for nm in ST]) for k in ("j600", "j600_nl")}
    tr_all = _pool_pair([_pp(out, nm, "no cause") for nm in ST])
    npair = 0 if tr["j600"] is None else tr["j600"]["n"]
    ok_tr = tr["j600"] is not None and (abs(tr["j600"]["diff"]) <= tr["j600"]["two_errors"] or tr["j600"]["diff"] == 0)
    res["2"] = {"rule": "no-cause scenes: TEACH changes the order in <= 3 % of battles; trade (lords in) TEACH - DEFAULT "
                        "within +-2 errors on >= 100 valid pairs, the comer battles out (all, and without the lords beside)",
                "changed": chg, "trade_-comer": tr, "trade_all": tr_all, "n_pairs": npair,
                "by_twin": {t: {"changed": _pool_share([_g(out, nm, "no cause, " + t, "TEACH", "changed") for nm in ST]),
                                "trade": _pool_pair([_pp(out, nm, "no cause, " + t) for nm in ST])} for t in ("come", "role")},
                "passed": (bool(chg and chg["p"] <= 0.03 and ok_tr) if npair >= PAIRS_MIN else None),
                "note": None if npair >= PAIRS_MIN else "n too small (%d < %d pairs) - not judged" % (npair, PAIRS_MIN)}
    d3 = _pool_pair([_pp(out, nm, "cause", "TEACH - DUMB") for nm in ST])
    res["3"] = {"rule": "TEACH - DUMB on cause scenes: reported, not gated",
                "all": {k: _pool_pair([_pp(out, nm, "cause", "TEACH - DUMB", k) for nm in ST])
                        for k in ("j600", "j600_nl", "j300", "win")},
                "-comer": {k: _pool_pair([_pp(out, nm, "cause" + X, "TEACH - DUMB", k) for nm in ST])
                           for k in ("j600", "j600_nl", "j300", "win")},
                "TEACH - DEFAULT, cause": {k: _pool_pair([_pp(out, nm, "cause", "TEACH - DEFAULT", k) for nm in ST])
                                           for k in ("j600", "j600_nl", "win")},
                "recv_stand": {p: [_g(out, nm, "cause", p, "recv_stand") for nm in ST] for p in ("TEACH", "DUMB")},
                "n": None if d3 is None else d3["n"]}
    om = {nm: _g(out, nm, "all", "TEACH", "orders") for nm in ("taught", "held", "G") if nm in out}
    res["4"] = {"rule": "TEACH's orders to the game a unit-minute WITH the lord (his first HOLD and MOVEs over the "
                        "infantry's and his standing minutes) <= 1.2; the infantry alone beside",
                "with_lord": om, "infantry_alone": {nm: _g(out, nm, "all", "TEACH", "orders_units") for nm in om},
                "lord_orders_a_battle": {nm: _g(out, nm, "all", "TEACH", "lord_orders") for nm in om},
                "n": {nm: (out.get(nm) or {}).get("battles") for nm in om},
                "passed": (bool(om) and all(v is not None and v <= 1.2 for v in om.values())) if all(full.values())
                else None, "note": note}
    t = _g(out, "taught", "cause", "TEACH", "contact")
    h = _g(out, "held", "cause", "TEACH", "contact")
    if t and h:
        axes = {}
        for ax in SO.HELD_AXES:
            ha = _g(out, "held", "held %s, cause" % ax, "TEACH", "contact")
            if ha:
                axes[ax] = _ax_gap(t, ha)
        pooled = _ax_gap(t, h)
        res["5"] = {"rule": "held - taught, contact share of TEACH on cause scenes: not below -2 errors, pooled AND on "
                            "each held axis", "taught": t, "pooled": pooled, "by_axis": axes,
                    "n": {"taught": t["n"], "held": h["n"], **{ax: v["held"]["n"] for ax, v in axes.items()}},
                    "passed": (pooled["passed"] and len(axes) == len(SO.HELD_AXES) and all(v["passed"] for v in axes.values()))
                    if full["taught"] and full["held"] else None, "note": note}
    g = _g(out, "G", "all", "TEACH", "contact")
    gd = _g(out, "G", "all", "DEFAULT", "contact")
    gh = _g(out, "G_hold", "all", "TEACH", "contact")
    res["6"] = {"rule": "G (att 1 against a holder): TEACH's stalls (no contact before the timer) <= 0.05",
                "TEACH_stall": None if g is None else round(1 - g["p"], 4), "n": None if g is None else g["n"],
                "DEFAULT_stall": None if gd is None else round(1 - gd["p"], 4),
                "G_hold_TEACH_stall (reported)": None if gh is None else round(1 - gh["p"], 4),
                "G_hold_n": None if gh is None else gh["n"], "TEACH - DEFAULT j600 (G)": _pp(out, "G", "all"),
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
        if nm == "lines":
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "table", "preflight"))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--players", default=",".join(PLAYERS))
    ap.add_argument("--out", default="tmp/o1/exam")
    ap.add_argument("--n", type=int, default=0, help="a smaller set: n cause scenes (held: n / 6 an axis; G: 2n)")
    ap.add_argument("--chunk", type=int, default=384)
    ap.add_argument("--tactic", default="TEACH", help="the tactic level's player (tactic/pin.PLAYERS)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sim", default="")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "preflight":
        return preflight(a)
    return table(a.out)


if __name__ == "__main__":
    main()
