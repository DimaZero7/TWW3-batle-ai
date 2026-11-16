"""The branch test of drill T0-2 (METHOD 'парная игра'; tmp/tactic_slice/design_attack_v23.md section 2, review_attack_v23
D): states of a teacher's run, each replayed whole-batch from its fork with one branch held for WINDOW_S (tactic/replay.
Branch), read by the judge (plain health, watch.hp_trade) at the fork + 60 / 120 / 300 s and at the end, paired by state.

    record   the base (--base, a player of pin.PLAYERS) through Recorder (the base unchanged; every tick: the pinned enemies
             by the base's own pinned() with their ticks in melee mt, and my free units; with --kind-picks the kinds'
             candidates of tags.tick too) on a set of specs (--set pilot: exam_pin.pilot_set, its chunks' torch seeds;
             G: the first 768 battles of exam_pin.g_set - its trained sizes -, S / F: exam_pin.s_set's scenes; G, S, F in
             exam_pin.watched_run's full / by_limit batches); ASSERTS the file against --ref (exam_pin.check_ref);
             writes <out>/asis.json, rec.json and the states of the picks:
               1f    the battle's first cause (a) order (stage 1f's pick); fork = the first tick of the unbroken run of
                     'e pinned and i free' that ends at the order
               new   among cause (a) orders whose fork <= the battle's end - WINDOW_S (the judge does not see the orders
                     after that: the count is written): in even battles of the set the first order whose pin waited
                     (mt(e) >= 3 at the order: late), in odd ones the first order of any kind; none of the wanted kind -
                     the other
               n2w, n4, n5  (--kind-picks; review C, I.3 - stage 1f's near-miss pick by kind) the first head-free keep
                     of my unit i of that kind, not in a band (KindRecorder), fork <= the end - WINDOW_S; e the nearest
                     enemy making the kind (tags.tick kind_keys), d its centre distance, alone (no taught kind, not n1 /
                     n6 on that unit-tick: the table splits by it)
             --states-ref: ASSERTS the 1f pick equals that file's set A (stage 1f's states.json)
    branch   --pick 1f|new|n2w|n4|n5 --kinds keep,teach,now: one replay a kind (teach - 'attack e' at the teacher's tick, keep - no
             order to i, now - at the fork; masks: i closed to the base's orders and e to other strikers for WINDOW_S);
             each state's flag near2 (an unheld enemy - valid, mt 0, no attack order of mine on it, not e - comes at
             (orders.comes) a unit of mine in melee touching e (scripts.TOUCH_M, as stage 1f), or at i, edge gap within
             orders.near2_m) and near2_at_i (it comes at i itself), read at the fork tick; the fork's own measures
             (FWatch). ASSERTS every battle without a state, and every battle before its fork tick, equals asis;
             --branch-ref <dir>: ASSERTS the replay's series equal <dir>/A_<kind>.json's (stage 1f's files)
    table    the pairs teach - keep, now - keep, now - teach by state at 60 / 120 / 300 / 750 s, by near2, size, script,
             YES by script and near2_at_i, S's families (and A1 + A3 pooled, review L's pool A1 + A2 + A3 + A6 NO), n4's
             distance bands, the first contact side of i on e in the x branch, the widest hole in my line 30 s after the
             fork (x against y); beside, the gold read (rout credited) at 60 / 120 / 300 s, and STATED groups (review L:
             no pass mark reads them) by the keep arm's first intended attack of i (Track t_int / int_tgt): idle >= 120 s,
             pulled in < 120 s, onto e / another enemy
    all      record, then every branch not yet written, then the table

    DOCK_NAME=agent-t2 DOCK_CPUS=8 DOCK_MEM=6g bash ops/dock.sh net.tactic.branches all --base TIu_back_co --pick 1f \
        --out tmp/attack/s2/b1f
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from net.stage0 import watch as W
from net.tactic import exam_pin as EP
from net.tactic import orders as OR
from net.tactic import pin as PN
from net.tactic import replay as RP
from net.unit import scripts as SC

WINDOW_S = 300.0          # the branch holds its choice this long after the fork (the judge's window)
HOLE_M = 50.0             # (the fork's watch: a gap in my line wider than this)
READS = (60, 120, 300, 750)
GOLD_S = (60, 120, 300)   # (the gold read's seconds after the fork: Track, stated beside the plain reads)
IDLE_S = 120              # (stated groups, review L: the keep arm's i first intends an attack >= this after the fork)
STEP = W.PinWatch.SERIES_S


# ------------------------------------------------------------------ the recorder and the flag
class Recorder(RP.Branch):
    """The base unchanged (no plan); every tick, by the Tacticians' row (= the chunk's battle): pins {sec: {slot: mt}}
    the enemies pinned by the base's pinned() and their ticks in melee, free {sec: [slot]} my free units."""

    def __init__(self, base):
        super().__init__(base, {}, WINDOW_S, masks=False)
        self.pin, self.free = {}, {}

    def __call__(self, v):
        out = super().__call__(v)
        fx = v.fx
        now = int(round(float(v.now)))
        pin = PN.pinned(fx, touch=getattr(self.base, "touch", None))
        free = fx.free & fx.able & ~v.melee
        ps = torch.where(pin, fx.slot, torch.full_like(fx.slot, -1)).tolist()
        mts = fx.mt.tolist()
        fs = torch.where(free, fx.my, torch.full_like(fx.my, -1)).tolist()
        for r, b in enumerate(v.idx.tolist()):
            self.pin.setdefault(b, {})[now] = {x: int(m) for x, m in zip(ps[r], mts[r]) if x >= 0}
            self.free.setdefault(b, {})[now] = [x for x in fs[r] if x >= 0]
        return out


KIND_PICKS = ("n2w", "n4", "n5")
ALONE_NOT = ("n1", "n6")  # (a kind state 'alone': no taught kind, not n6 - masked in the loss -, not n1 - the cause early)


class KindRecorder(Recorder):
    """Recorder + by battle and kind of KIND_PICKS the tick's best candidate {sec: (i, e, d, alone)}: a head-free keep of
    mine of that kind, not in a band (tags.tick on the base's own output); e the nearest enemy of the kind's keys (d its
    centre distance); alone - the unit-tick has no taught kind (tags.TAUGHT) and none of ALONE_NOT (a candidate alone
    is taken before one that is not, at the same tick)."""

    def __init__(self, base):
        super().__init__(base)
        self.cand = {}

    def __call__(self, v):
        from net.tactic import tags as TG
        out = super().__call__(v)
        T = TG.tick(v, out, self.base)
        fx = v.fx
        now = int(round(float(v.now)))
        rows = v.idx.tolist()
        ks = T["kinds"]
        for n in KIND_PICKS:
            other = torch.stack([ks[m] for m in TG.TAUGHT + ALONE_NOT], 0).any(0)
            ok = ks[n] & ~T["dropped"]
            if not bool(ok.any()):
                continue
            key = T["kind_keys"][n] & fx.valid[:, None, :]
            d = torch.where(key, fx.g_dist, torch.full_like(fx.g_dist, OR.INF))
            dm, kk = d.min(2)
            for r, g in (ok & torch.isfinite(dm)).nonzero().tolist():
                b = rows[r]
                c = self.cand.setdefault(b, {}).setdefault(n, {})
                x = (int(fx.my[r, g]), int(fx.slot[r, int(kk[r, g])]), round(float(dm[r, g]), 2),
                     not bool(other[r, g]))
                if now not in c or (x[3], -x[2]) > (c[now][3], -c[now][2]):
                    c[now] = x
        return out


class FlagBranch(RP.Branch):
    """Branch + each planned battle's flags near2 and near2_at_i at its fork tick (module doc)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.flag = {}

    def __call__(self, v):
        fx = v.fx
        now = int(round(float(v.now)))
        rows = [(r, b) for r, b in enumerate(v.idx.tolist()) if b in self.plan and int(self.plan[b]["t"]) == now]
        if rows:
            cm = OR.comes(fx)                                                         # [R, i, j] j comes at i
            w = OR.Walk(fx)
            reach = float(OR.c_orders()["near2_m"])
            for r, b in rows:
                st = self.plan[b]
                ke = int(fx.row_of_slot(torch.full((fx.R, 1), int(st["e"]), dtype=torch.long,
                                                   device=fx.s.device))[r, 0])
                mine_m = fx.mine[r] & fx.m[r] & ~fx.r[r]
                fight = mine_m & (fx.gap[r, :, ke] <= SC.TOUCH_M) if (ke >= 0 and bool(fx.m[r, ke])) else mine_m.clone()
                ki = (fx.slot[r] == int(st["i"])) & fx.mask[r]
                foe = fx.valid[r] & ~fx.m[r] & (w.count[r] == 0)
                if ke >= 0:
                    foe[ke] = False
                hit = cm[r] & (fight | ki)[:, None] & foe[None, :] & (fx.gap[r] <= reach)  # [mine row, enemy row]
                self.flag[b] = {"near2": bool(hit.any()), "near2_at_i": bool((hit & ki[:, None]).any())}
        return super().__call__(v)


# ------------------------------------------------------------------ the fork's own measures
class Track:
    """The fork's measures of one plan {b: {t, i, e, u}} (the chunk's battles; FWatch doc)."""

    def __init__(self, plan, B, dev):
        f = lambda k, dv=-1: torch.tensor([int(plan[b][k]) if b in plan and k in plan[b] else dv for b in range(B)],  # noqa
                                          dtype=torch.long, device=dev)
        self.fT, self.fI, self.fE, self.fU = f("t", 10 ** 6), f("i"), f("e"), f("u")
        self.has = self.fI >= 0
        nan = lambda: torch.full((B,), float("nan"), device=dev)                    # noqa: E731
        self.fm = {k: nan() for k in ("hpi0", "hpe0", "hpi60", "hpe60", "hpi120", "hpe120", "t_eu", "t_ie",
                                      "t_att", "att_tgt", "gap30", "holes30", "side_ie", "t_int", "int_tgt")
                   + tuple(f"gold{w}" for w in GOLD_S)}
        self.fm["att_e"] = torch.zeros(B, device=dev)

    def step(self, w, sec, st):
        u = st.u
        B, N = w.B, w.N
        ar = torch.arange(B, device=w.dev)
        I, E, U = self.fI.clamp(min=0), self.fE.clamp(min=0), self.fU.clamp(min=0)
        hp = u["hp_abs"] / u["hp0"].clamp(min=1e-6)
        rel = sec - self.fT
        for dt, ki, ke in ((0, "hpi0", "hpe0"), (60, "hpi60", "hpe60"), (120, "hpi120", "hpe120")):
            due = self.has & (rel == dt)
            if bool(due.any()):
                self.fm[ki] = torch.where(due, hp[ar, I], self.fm[ki])
                self.fm[ke] = torch.where(due, hp[ar, E], self.fm[ke])
        on = self.has & (rel >= 0)
        if not bool(on.any()):
            return
        hs = u["hit_sec"].view(B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)
        t = torch.full((B,), float(sec), device=w.dev)
        teu = on & (self.fU >= 0) & touch[ar, E, U] & torch.isnan(self.fm["t_eu"])
        self.fm["t_eu"] = torch.where(teu, t, self.fm["t_eu"])
        tie = on & touch[ar, I, E] & torch.isnan(self.fm["t_ie"])
        self.fm["t_ie"] = torch.where(tie, t, self.fm["t_ie"])
        self.fm["side_ie"] = torch.where(tie, hs[ar, E, I].float(), self.fm["side_ie"])   # (exam_pin.SceneWatch's read)
        o = w.chain.tac.orders_of()
        att = on & o["on"][ar, I] & (o["skill"][ar, I] == SC.ATTACK) & (o["tgt"][ar, I] >= 0)
        first = att & torch.isnan(self.fm["t_att"])
        self.fm["t_att"] = torch.where(first, t, self.fm["t_att"])
        self.fm["att_tgt"] = torch.where(first, o["tgt"][ar, I].float(), self.fm["att_tgt"])
        inwin = att & (rel < WINDOW_S) & (o["tgt"][ar, I] == E)
        self.fm["att_e"] = torch.where(inwin, torch.ones_like(self.fm["att_e"]), self.fm["att_e"])
        # STATED only (review L): the first second from the fork the unit script of i intends an attack (its own counter
        # in the keep arm), and that attack's target (a battle unit index: e or another enemy)
        from sim import orders as O
        it = w.chain.units.intended
        ia = on & (it.kind[ar, I] == O.ATTACK) & torch.isnan(self.fm["t_int"])
        self.fm["t_int"] = torch.where(ia, t, self.fm["t_int"])
        self.fm["int_tgt"] = torch.where(ia, it.target[ar, I].float(), self.fm["int_tgt"])
        # the gold read (rout credited, watch._gold_trade) at min(fork + w, the play's last step) beside the plain one
        gd = W._gold_trade(u, w.lord)
        for gw in GOLD_S:
            k = f"gold{gw}"
            self.fm[k] = torch.where(on & (rel <= gw), gd, self.fm[k])
        due = self.has & (rel == 30)
        if bool(due.any()):
            g, n = w.holes(st)
            self.fm["gap30"] = torch.where(due, g, self.fm["gap30"])
            self.fm["holes30"] = torch.where(due, n, self.fm["holes30"])


class FWatch(W.PinWatch):
    """PinWatch + per planned battle of each plan (plans {name: {b: {t, i, e, u}}}, the chunk's battles): health share
    of i and e at t, t + 60, t + 120; the first second (from t) e touches u and i touches e; i's first attack order from
    t and its target, and whether i held 'attack e' within the window; at t + 30 the widest lateral gap between my
    standing units (all depths, across the start's axis my centroid -> the enemy's) and the number of gaps over
    HOLE_M."""

    def __init__(self, *a, plans=None, **k):
        super().__init__(*a, **k)
        self.plans = plans or {}

    def _init(self, st):
        super()._init(st)
        self.tracks = {n: Track(p, self.B, self.dev) for n, p in self.plans.items()}

    def __call__(self, sec, st):
        super().__call__(sec, st)
        for tr in self.tracks.values():
            tr.step(self, sec, st)

    def holes(self, st):
        return EP.line_holes(self, st, HOLE_M)


# ------------------------------------------------------------------ one run
def _local(plan, ii):
    return {k: plan[b] for k, b in enumerate(ii) if b in plan}


def play(name, make, plans, specs, seeds, chunk, a, P, **kw):
    """(the player file of exam_pin.run_player, fork measures {plan name: {b: dict}}, the wrappers by chunk
    [(ii, player)]): make(local plan) the wrapper of a chunk; plans {name: global plan} the plans FWatch measures."""
    fm, wraps = {n: {} for n in plans}, []

    def look(ii, watch, side):
        wraps.append((ii, side.player))
        for n, tr in watch.tracks.items():
            f = {k: v.tolist() for k, v in tr.fm.items()}
            for k, b in enumerate(ii):
                if b in plans[n]:
                    fm[n][b] = {kk: f[kk][k] for kk in f}
    res = EP.run_player(name, specs, seeds, chunk, a, P, make=lambda ii: make(_local(plans.get("plan", {}), ii)),
                        on_chunk=look, chunk_args=True,
                        watch_cls=lambda P_, g, chain, ii: FWatch(P_, g, chain=chain,
                                                                  plans={n: _local(p, ii) for n, p in plans.items()}),
                        **kw)
    return res, fm, wraps


def _glob(wraps, attr):
    out = {}
    for ii, w in wraps:
        out.update({ii[b]: x for b, x in getattr(w, attr, {}).items() if b < len(ii)})
    return out


# ------------------------------------------------------------------ the states' picks
def _fork(x, pins, frees):
    """The fork tick of an order {to, i, e}: back from the order while e pinned and i free at the tick before."""
    p, f = pins.get(x["b"], {}), frees.get(x["b"], {})
    t = x["to"]
    while x["e"] in p.get(t - 5, {}) and x["i"] in f.get(t - 5, []):
        t -= 5
    return t


def _orders(out, cause=1):
    ci = {k: i for i, k in enumerate(out["order_log_cols"])}
    by = {}
    for r in out["order_log"]:
        if int(r[ci["cause"]]) != cause:
            continue
        b = int(r[ci["b"]])
        by.setdefault(b, []).append({"b": b, "to": int(r[ci["sec"]]), "i": int(r[ci["slot"]]), "e": int(r[ci["target"]])})
    for v in by.values():
        v.sort(key=lambda x: (x["to"], x["i"]))
    return by


def _state(x, pins, frees):
    p, f = pins.get(x["b"], {}), frees.get(x["b"], {})
    t = _fork(x, pins, frees)
    mt = p.get(x["to"], {}).get(x["e"])
    return {"t": t, "i": x["i"], "e": x["e"], "u": -1, "to": x["to"], "lag": x["to"] - t,
            "ok_at_order": x["e"] in p.get(x["to"], {}) and x["i"] in f.get(x["to"], []),
            "pinned_before": x["e"] in p.get(x["to"] - 5, {}), "mt": mt, "late": mt is not None and mt >= 3}


def pick_1f(out, pins, frees):
    """Stage 1f's pick (module doc): {b: state}."""
    return {b: _state(v[0], pins, frees) for b, v in _orders(out).items()}


def pick_new(out, pins, frees, ends):
    """The new pick (module doc): ({b: state}, counts)."""
    S, cut = {}, 0
    for b, v in _orders(out).items():
        sts = [_state(x, pins, frees) for x in v]
        ok = [s for s in sts if s["t"] <= ends[b] - WINDOW_S]
        if not ok:
            cut += 1
            continue
        late = [s for s in ok if s["late"]]
        S[b] = (late[0] if late else ok[0]) if b % 2 == 0 else ok[0]
    return S, {"battles_with_orders": len(_orders(out)), "no_order_in_window": cut, "states": len(S),
               "late": sum(1 for s in S.values() if s["late"])}


# ------------------------------------------------------------------ the commands
G_N = 768                 # (set G's branch test: its first 768 battles - three chunks of the trained sizes)


def _set(a):
    """(specs, seeds, chunk, play kwargs) of --set (module doc)."""
    from net.stage0 import battle as BA
    if a.set == "pilot":
        return (*EP.pilot_set(a.n, range(a.seeds), a.limit), {})
    if a.set == "G":
        sp, seeds, ch = EP.g_set()
        held = EP.scfg()["drills"]["attack"]["sets"]["G"]["sizes_held"]
        assert G_N % ch == 0 and all(x["n"] not in held for x in EP.meta_of(sp[:G_N]))
        return sp[:G_N], seeds[:G_N // ch], ch, {"full": True, "by_limit": True}
    if a.set in ("S", "F"):
        sp, seeds, ch = EP.s_set(name=a.set)
        return sp, seeds, ch, {"full": True, "by_limit": True, "battles": BA.scene_battles, "enemy": BA.SceneEnemy}
    raise ValueError(a.set)


def _ints(d):
    return {int(b): x for b, x in d.items()}


def pick_kind(cand, ends, n):
    """{b: state} of kind n (module doc) from KindRecorder's candidates."""
    S = {}
    for b, c in cand.items():
        ts = sorted(t for t in c.get(n, {}) if t <= ends[b] - WINDOW_S)
        if ts:
            i, e, d, alone = c[n][ts[0]]
            S[b] = {"t": ts[0], "i": i, "e": e, "u": -1, "to": ts[0], "lag": 0, "d": d, "late": False, "alone": alone}
    return S


def record(a, P, D):
    specs, seeds, chunk, kw = _set(a)
    rec = KindRecorder if a.kind_picks else Recorder
    out, _, wraps = play(a.base, lambda pl: rec(a.base), {}, specs, seeds, chunk, a, P, **kw)
    refs = dict(x.split("=", 1) for x in a.ref.split(",") if x)
    out["selfcheck"] = EP.check_ref(a.base, out, refs[a.base], a.sim) if a.base in refs else {"compared": False,
                                                                                               "why": "no --ref"}
    pins = {b: {t: v for t, v in x.items()} for b, x in _glob(wraps, "pin").items()}
    frees = _glob(wraps, "free")
    ends = [min(float(s["limit"]), float(s.get("end_s") or s["limit"])) for s in specs]
    s1f = pick_1f(out, pins, frees)
    snew, cnt = pick_new(out, pins, frees, ends)
    out["base"] = a.base
    out["recipes"] = [s.get("recipe", a.set) for s in specs]
    sts = {"1f": s1f, "new": snew, "new_counts": cnt, "window_s": WINDOW_S}
    if a.kind_picks:
        cand = _glob(wraps, "cand")
        for n in KIND_PICKS:
            sts[n] = pick_kind(cand, ends, n)
        sts["kind_counts"] = {n: len(sts[n]) for n in KIND_PICKS}
        sts["kind_battles_any_tick"] = {n: sum(1 for c in cand.values() if c.get(n)) for n in KIND_PICKS}
    (D / "asis.json").write_text(json.dumps(out), encoding="utf-8")
    (D / "rec.json").write_text(json.dumps({"pins": pins, "free": frees}), encoding="utf-8")
    (D / "states.json").write_text(json.dumps(sts), encoding="utf-8")
    print(json.dumps({"states_1f": len(s1f), **{f"new_{k}": v for k, v in cnt.items()},
                      **sts.get("kind_counts", {}), "any_tick": sts.get("kind_battles_any_tick")}), flush=True)
    if a.states_ref:
        ref = _ints(json.loads(Path(a.states_ref).read_text(encoding="utf-8"))["A"])
        keys = ("t", "i", "e", "to", "lag", "ok_at_order", "pinned_before")
        bad = [b for b in set(ref) | set(s1f) if b not in ref or b not in s1f
               or any(ref[b][k] != s1f[b][k] for k in keys)]
        print(json.dumps({"states_ref": a.states_ref, "states": len(s1f), "differ": len(bad), "first": sorted(bad)[:5]}),
              flush=True)
        assert not bad, "the 1f pick does not reproduce the stage 1f states"


def branch(a, P, D, pick, kind):
    specs, seeds, chunk, kw = _set(a)
    S = _ints(json.loads((D / "states.json").read_text(encoding="utf-8"))[pick])
    if not S:
        print(json.dumps({"pick": pick, "kind": kind, "states": 0, "replay": "none"}), flush=True)
        return
    asis = json.loads((D / "asis.json").read_text(encoding="utf-8"))
    plan = {}
    for b, x in S.items():
        st = {"t": x["t"], "i": x["i"], "e": x["e"], "u": -1, "cause": 1 if pick in ("1f", "new") else 3}
        if kind == "teach":
            st.update(kind="at", at=x["to"])
        else:
            st["kind"] = kind
        plan[b] = st
    out, fm, wraps = play(asis["base"], lambda pl: FlagBranch(asis["base"], pl, WINDOW_S, masks=True),
                          {"plan": plan}, specs, seeds, chunk, a, P, **kw)
    nb = len(out["hp_series"])
    diff_free = [b for b in range(nb) if b not in plan and not EP.same(out["hp_series"][b], asis["hp_series"][b])]
    diff_pre = [b for b in plan if not EP.same(out["hp_series"][b][:plan[b]["t"] // STEP],
                                               asis["hp_series"][b][:plan[b]["t"] // STEP])]
    ci = {k: i for i, k in enumerate(out["order_log_cols"])}
    lo = lambda L: EP._log_key([r for r in L if int(r[ci["b"]]) not in plan])                  # noqa: E731
    log_ok = EP.same(lo(out["order_log"]), lo(asis["order_log"]))
    chk = {"pick": pick, "kind": kind, "states": len(plan), "differ_without_state": len(diff_free),
           "differ_before_fork": len(diff_pre), "order_log_without_state_same": log_ok}
    print(json.dumps(chk), flush=True)
    assert not diff_free and not diff_pre and log_ok, "the replay is not the asis battle where it must be"
    flags = _glob(wraps, "flag")
    logs = _glob(wraps, "log")
    res = {"pick": pick, "kind": kind, "states": len(plan), "wall_s": out["wall_s"], "hp_series": out["hp_series"],
           "per_battle": {k: out["per_battle"][k] for k in ("hp_trade", "t_contact", "game_orders_min", "routs_own",
                                                            "routs_enemy", "attack_orders")},
           "fm": fm["plan"], "log": logs, "order_log_cols": out["order_log_cols"],
           "order_log": [r for r in out["order_log"] if int(r[ci["b"]]) in plan], "selfcheck": chk,
           "flag": {b: flags.get(b) for b in plan}}
    if a.branch_ref and pick == "1f":
        p = Path(a.branch_ref) / f"A_{kind}.json"
        if p.exists():
            ref = json.loads(p.read_text(encoding="utf-8"))
            same = EP.same(ref["hp_series"], res["hp_series"])
            fl = {int(b): x for b, x in ref.get("flag", {}).items()}
            fl_ok = all(fl.get(b) == (res["flag"][b] or {}).get("near2") for b in plan if b in fl)
            res["selfcheck_ref"] = {"against": str(p), "hp_series": same, "near2": fl_ok}
            print(json.dumps({"branch_ref": str(p), "hp_series_bit_for_bit": same, "near2_same": fl_ok}), flush=True)
            assert same and fl_ok, f"the replay does not reproduce {p}"
        else:
            print(json.dumps({"branch_ref": str(p), "compared": False, "why": "absent"}), flush=True)
    (D / f"{pick}_{kind}.json").write_text(json.dumps(res), encoding="utf-8")


# ------------------------------------------------------------------ the table
def ms(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    n = len(v)
    if n < 2:
        return float("nan"), float("nan"), n
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, 2 * sd / math.sqrt(n), n


def f(m):
    return f"{m[0]:+.4f} ± {m[1]:.4f}" if math.isfinite(m[0]) else "-"


def J(d, b, t, w):
    if w == 750:
        return d["per_battle"]["hp_trade"][b]
    ser = d["hp_series"][b]
    return ser[min(int(round((t + w) / STEP)), len(ser) - 1)]


def pd(x, y, S, w):
    lx, ly = _ints(x.get("log", {})), _ints(y.get("log", {}))
    sel = [b for b in S if not lx.get(b, {}).get("why") and not ly.get(b, {}).get("why")]
    return ms([J(x, b, S[b]["t"], w) - J(y, b, S[b]["t"], w) for b in sel])


def pg(x, y, S, w):
    """pd's pair by the gold read at fork + w (Track's gold<w>; absent in older files: n 0)."""
    lx, ly = _ints(x.get("log", {})), _ints(y.get("log", {}))
    fx_, fy_ = _ints(x.get("fm", {})), _ints(y.get("fm", {}))
    k = f"gold{w}"
    sel = [b for b in S if not lx.get(b, {}).get("why") and not ly.get(b, {}).get("why")
           and k in fx_.get(b, {}) and k in fy_.get(b, {})]
    return ms([fx_[b][k] - fy_[b][k] for b in sel])


def table(D, picks=("1f", "new")):
    st = json.loads((D / "states.json").read_text(encoding="utf-8"))
    asis = json.loads((D / "asis.json").read_text(encoding="utf-8"))
    meta = asis["meta"]
    tc = asis["per_battle"]["t_contact"]
    lines, res = [], {}
    for pick in picks:
        S = _ints(st[pick])
        files = {p.stem[len(pick) + 1:]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(D.glob(f"{pick}_*.json"))}
        if not files:
            continue
        flag = {}
        for d in files.values():
            for b, x in d.get("flag", {}).items():
                if x is not None:
                    flag[int(b)] = x
        lines.append(f"\n## pick {pick}: {len(S)} states" + (f" ({st.get('new_counts')})" if pick == "new" else ""))
        rel = [S[b]["t"] - (tc[b] - 1) for b in S]
        lags = {}
        for b in S:
            lags[S[b]["lag"]] = lags.get(S[b]["lag"], 0) + 1
        lines.append(f"   fork - first contact mean {ms(rel)[0]:+.1f} s; lag (order - fork) {dict(sorted(lags.items()))}; "
                     f"late {sum(1 for b in S if S[b]['late'])}; near2 yes {sum(1 for b in S if flag.get(b, {}).get('near2'))}"
                     f", at i {sum(1 for b in S if flag.get(b, {}).get('near2_at_i'))}, missing {sum(1 for b in S if b not in flag)}")
        groups = ([("all", lambda b: True), ("NO (near2 off)", lambda b: flag.get(b, {}).get("near2") is False),
                   ("YES (near2 on)", lambda b: flag.get(b, {}).get("near2") is True),
                   ("  of them near2_at_i", lambda b: flag.get(b, {}).get("near2_at_i") is True),
                   ("late", lambda b: S[b]["late"]), ("2-4 units", lambda b: meta[b]["n"] <= 4),
                   ("5-8 units", lambda b: meta[b]["n"] > 4)]
                  + [(en, lambda b, en=en: meta[b]["enemy"] == en) for en in EP.SCRIPTS])
        yes = lambda b: flag.get(b, {}).get("near2") is True                                     # noqa: E731
        groups += [(f"  YES, {en}", lambda b, en=en: yes(b) and meta[b]["enemy"] == en) for en in EP.SCRIPTS]
        groups += [("  YES, not near2_at_i", lambda b: yes(b) and not flag.get(b, {}).get("near2_at_i"))]
        rc = asis.get("recipes") or []
        fams = [r for r in dict.fromkeys(rc) if r not in (None, "G", "pilot")]
        no = lambda b: flag.get(b, {}).get("near2") is False                                     # noqa: E731
        groups += [(f"family {r}", lambda b, r=r: rc[b] == r) for r in fams]
        groups += [(f"family {r}, NO", lambda b, r=r: rc[b] == r and no(b)) for r in fams]
        if "A1" in fams or "A3" in fams:
            groups += [("A1 + A3", lambda b: rc[b] in ("A1", "A3")), ("A1 + A3, NO", lambda b: rc[b] in ("A1", "A3")
                                                                       and no(b))]
        if any("alone" in v for v in S.values()):
            groups += [("kind alone", lambda b: S[b].get("alone") is True),
                       ("with a taught kind or n1 / n6", lambda b: S[b].get("alone") is False)]
        if any("d" in v for v in S.values()):
            n4 = [float(x) for x in EP.scfg()["drills"]["attack"]["kinds"]["n4_m"]]
            cuts = ((0, 170), (170, 200), (200, 240.01)) if n4 == [170.0, 240.0] else (
                (0, n4[0]), (n4[0], n4[0] + 30), (n4[0] + 30, n4[1] + 0.01))    # (n4 moved with the reach: kinds.n4_m)
            groups += [(f"d {lo:g}-{hi:g} m", lambda b, lo=lo, hi=hi: lo <= S[b].get("d", -1) < hi) for lo, hi in cuts]
        if fams:
            pool = ("A1", "A2", "A3", "A6")
            groups += [("L pool A1 + A2 + A3 + A6, NO", lambda b: rc[b] in pool and no(b))]
        # STATED groups (review L, last bullet; no pass mark reads them): by the keep arm's first intended attack of i
        fk = _ints(files["keep"].get("fm", {})) if "keep" in files else {}
        if any("t_int" in v for v in fk.values()):
            def t_in(b):
                x_ = fk.get(b, {}).get("t_int")
                return math.inf if x_ is None or x_ != x_ else x_ - S[b]["t"]

            def onto_e(b):
                x_ = fk.get(b, {}).get("int_tgt")
                return x_ is not None and x_ == x_ and int(x_) == int(S[b]["e"])
            idle, pulled = (lambda b: t_in(b) >= IDLE_S), (lambda b: t_in(b) < IDLE_S)
            base_g = [("", lambda b: True), (", NO", no)] + ([(", L pool NO", lambda b: rc[b] in pool and no(b))]
                                                             if fams else [])
            for sfx, gg in base_g:
                groups += [(f"STATED keep idle >= {IDLE_S} s{sfx}", lambda b, gg=gg: gg(b) and idle(b)),
                           (f"STATED keep pulled in < {IDLE_S} s{sfx}", lambda b, gg=gg: gg(b) and pulled(b)),
                           (f"STATED   pulled in onto e{sfx}", lambda b, gg=gg: gg(b) and pulled(b) and onto_e(b)),
                           (f"STATED   pulled in onto another{sfx}",
                            lambda b, gg=gg: gg(b) and pulled(b) and not onto_e(b))]
            groups += [(f"STATED family {r}, NO, keep idle", lambda b, r=r: rc[b] == r and no(b) and idle(b))
                       for r in fams]
            groups += [(f"STATED family {r}, NO, keep pulled in", lambda b, r=r: rc[b] == r and no(b) and pulled(b))
                       for r in fams]
        res[pick] = {}
        for x, y in (("teach", "keep"), ("now", "keep"), ("now", "teach")):
            if x not in files or y not in files:
                continue
            lines.append(f"   {x} - {y}: reads at fork + 60 / 120 / 300 s | 750 s")
            fx_ = _ints(files[x].get("fm", {}))
            fy_ = _ints(files[y].get("fm", {}))
            sides = {b: {1: "front", 2: "flank", 3: "rear"}.get(int(v["side_ie"]) if v.get("side_ie") == v.get("side_ie")
                                                                and v.get("side_ie") is not None else -1, "no contact")
                     for b, v in fx_.items()}
            g2 = groups + [(f"  {x}: i met e on its {sd}", lambda b, sd=sd: sides.get(b) == sd)
                           for sd in ("front", "flank", "rear", "no contact")]
            for lab, g in g2:
                SS = {b: v for b, v in S.items() if g(b)}
                cells = [pd(files[x], files[y], SS, w) for w in READS]
                res[pick][f"{x}-{y}:{lab.strip()}"] = [[round(c[0], 4), round(c[1], 4), c[2]] for c in cells]
                gc = [pg(files[x], files[y], SS, w) for w in GOLD_S]
                has_g = any(c[2] for c in gc)
                if has_g:
                    res[pick][f"{x}-{y}:{lab.strip()}:gold"] = [[round(c[0], 4), round(c[1], 4), c[2]] for c in gc]
                lines.append(f"     {lab} (n {len(SS)}): " + " | ".join(f(c) for c in cells)
                             + ("   || gold 60 / 120 / 300 s: " + " | ".join(f(c) for c in gc) if has_g else ""))
            hx = [fx_[b]["gap30"] for b in S if b in fx_ and b in fy_]
            hy = [fy_[b]["gap30"] for b in S if b in fx_ and b in fy_]
            hd = ms([p - q for p, q in zip(hx, hy)])
            res[pick][f"{x}-{y}:hole30"] = {"x": ms(hx)[0], "y": ms(hy)[0], "diff": [hd[0], hd[1], hd[2]]}
            lines.append(f"     widest hole in my line at fork + 30 s, m: {x} {ms(hx)[0]:.1f}, {y} {ms(hy)[0]:.1f}, "
                         f"{x} - {y} {f(hd)} ({hd[2]})")
    txt = "\n".join(lines)
    print(txt, flush=True)
    (D / "table.txt").write_text(txt, encoding="utf-8")
    (D / "table.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("record", "branch", "table", "all"))
    ap.add_argument("--base", default="TEACH")
    ap.add_argument("--set", default="pilot", choices=("pilot", "G", "S", "F"))
    ap.add_argument("--kind-picks", action="store_true", help="record: the kinds' picks n2w, n4, n5 too")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=750)
    ap.add_argument("--pick", default="1f,new")
    ap.add_argument("--kinds", default="keep,teach")
    ap.add_argument("--ref", default="", help="record's bit-for-bit reference: name=path[@sim commit] (exam_pin)")
    ap.add_argument("--sim", default="")
    ap.add_argument("--states-ref", default="")
    ap.add_argument("--branch-ref", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/attack/s2/branches")
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    picks = [x for x in a.pick.split(",") if x]
    kinds = [x for x in a.kinds.split(",") if x]
    if a.what in ("record", "all") and not (a.what == "all" and (D / "states.json").exists()):
        record(a, P, D)
    if a.what in ("branch", "all"):
        for pick in picks:
            for kind in kinds:
                if (pick in KIND_PICKS and kind == "teach") or (pick not in KIND_PICKS and kind == "now"
                                                                and a.what == "all"):
                    continue                    # (a kind's state has no teacher order; 'now' of 1f / new: branch only)
                if a.what == "branch" or not (D / f"{pick}_{kind}.json").exists():
                    t1 = time.time()
                    branch(a, P, D, pick, kind)
                    print(json.dumps({"done": f"{pick}_{kind}", "min": round((time.time() - t1) / 60, 1)}), flush=True)
    if a.what in ("table", "all"):
        table(D, picks)
    print(json.dumps({"minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
