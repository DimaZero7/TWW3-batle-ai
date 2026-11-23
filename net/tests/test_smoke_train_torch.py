"""One training step a level with a network (the smoke test of net): a short batch of the level's own battles with a
fresh network, one update that changes its weights, the checkpoint saved and read back through <level>/ckpt.py (the
bridge's way in) with the same weights. Needs torch (the container, CPU); the levels run the frozen unit network
(net/config/tactic.json train.unit_ckpt -> weights/unit/..., net/weights.json)."""
import pytest

torch = pytest.importorskip("torch")

from sim.params import load  # noqa: E402

P = load()


def _params(net):
    return [p.detach().clone() for p in net.parameters() if p.requires_grad]


def _changed(before, net):
    return any(not torch.equal(a, b) for a, b in zip(before, [p for p in net.parameters() if p.requires_grad]))


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture(scope="module")
def unit_net():
    from net.tactic import ckpt as TK
    return TK.unit_network("cpu")


def test_the_tactician_trains_a_step(tmp_path, monkeypatch, unit_net):
    from net.tactic import cfg
    from net.tactic import ckpt as TK
    from net.tactic import decide as DC
    from net.tactic import drill1 as DR
    from net.tactic import env as EN
    from net.tactic import train as TT
    monkeypatch.setitem(cfg()["drill1"], "limit_s", 20)
    torch.manual_seed(0)
    net, critic = DC.TacticNet(), DC.TacticCritic()
    setup = EN.Setup(DR.specs(2, torch.Generator().manual_seed(3)), P, "cpu")
    batch, rs, _ = EN.play(setup, unit_net, net=net, critic=critic)
    assert rs["decisions"] > 0
    pa, pc = TT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    tc = dict(cfg()["train"], minibatch=8, epochs=1)
    before = _params(net)
    us = TT.update(net, critic, opt, batch, tc, "cpu", None, 1.0)
    assert us["samples"] > 0 and _changed(before, net)
    TK.save(tmp_path / "t.pt", net, critic, opt, tc, 1)
    n2, c2, ck = TK.restore(tmp_path / "t.pt")
    assert ck["kind"] == TK.KIND and ck["updates"] == 1 and _same(net, n2) and _same(critic, c2)


def test_the_tactic_pointer_learns_a_step(tmp_path):
    """Drill T0-2's network (tactic/heads.TacticNet2 as pin.NetPlayer) drives two short pilot battles through the side
    wrapper, TEACH labels the same views (tags.Record); one update on those ticks with the walk fed with the
    labeller's choices changes the weights; the tactic_v2 checkpoint reads back the same."""
    from net.stage0 import battle as BA
    from net.tactic import ckpt as TK
    from net.tactic import exam_pin as EP
    from net.tactic import heads as HD
    from net.tactic import pin as PN
    from net.tactic import tags as TG
    torch.manual_seed(0)
    net = HD.TacticNet2()
    got, calls = [], []

    def hook(v, lab, T):
        calls.append(int(T["head_free"].sum()))
        if bool(T["head_free"].any()):
            got.append((v, T))

    def make():
        rec = TG.Record(PN.NetPlayer(net), PN.player("TEACH"), hook=hook)
        rec.needs_input = True
        return rec
    BA.play_tactic(EP.pilot_specs(2, 0, limit=160), make, P, "cpu", chunk=2, compile=False)
    assert got, calls
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    before = _params(net)
    total = 0.0
    for v, T in got:
        label = torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1))
        o = net.act(v, greedy=True, given=label)
        w = (T["head_free"] & ~T["dropped"] & ~T["dc_label"]).float()
        total = total + HD.loss(o["logits"], label, w, T["dc_keys"])[0]
    opt.zero_grad()
    total.backward()
    opt.step()
    assert _changed(before, net)
    TK.save_v2(tmp_path / "t2.pt", net, opt, ["T0-2"])
    n2, c2, ck = TK.restore(tmp_path / "t2.pt")
    assert ck["kind"] == TK.KIND2 and c2 is None and ck["row_width"] == 185 and _same(net, n2)


def test_the_operative_trains_a_step(tmp_path, unit_net):
    from net.operative import cases as CS
    from net.operative import cfg
    from net.operative import ckpt as OK
    from net.operative import env as EN
    from net.operative import net as NN
    from net.operative import train as OT
    torch.manual_seed(0)
    net, critic = NN.OperativeNet(), NN.OperativeCritic()
    s = CS.Setup(CS.specs(3, torch.Generator().manual_seed(0), fixed=True), P, "cpu")
    s.limit = torch.full_like(s.limit, 31.0)
    batch, rs, _ = EN.play(s, unit_net, net=net, critic=critic)
    assert rs["parcel_errors"] == 0 and rs["parcels"] >= 3
    pa, pc = OT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    tc = dict(cfg()["train"], minibatch=8, epochs=1)
    before = _params(net)
    us = OT.update(net, critic, opt, batch, tc, "cpu", None, 1.0)
    assert us["samples"] > 0 and _changed(before, net)
    OK.save(tmp_path / "o.pt", net, critic, opt, tc, 1)
    n2, c2, ck = OK.restore(tmp_path / "o.pt")
    assert ck["kind"] == OK.KIND and _same(net, n2) and _same(critic, c2)


def test_the_operatives_memory_trains_a_step():
    """net/operative/memory.py (a prototype, not in the network yet): a short battle seen every second, two decisions'
    records; the groups' sequences through the time attention and the units' summary through its block, one step."""
    from net.operative import frame as FR
    from net.operative import memory as MM
    from net.tactic import raw as TR
    from sim import battle, scenario
    from sim import scripts as opponents
    torch.manual_seed(0)

    def line(keys, z, b):
        return [{"key": k, "x": -30.0 + 30.0 * i, "z": z, "b": b} for i, k in enumerate(keys)]
    spec = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": line(["wh_main_emp_inf_spearmen_0"] * 2, -60.0, 0.0)},
        2: {"faction": "wh2_main_skv_skaven", "units": line(["wh2_main_skv_inf_clanrat_spearmen_0"] * 2, 60.0, 180.0)}}}
    st = scenario.build([spec], P)
    seer, store, axes = TR.Seer(P), MM.Store(1, st.N, 1, T=4), None
    for sec in range(1, 31):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
        raw = seer.raw(st)
        axes = axes or FR.Axes.of(raw, torch.ones(1, dtype=torch.long), st.bounds)
        store.see(raw, float(sec))
        if sec % 15 == 0:
            store.tick(float(sec), axes)
    members = torch.stack([store.enemy == 0, store.enemy], 1) & (raw[..., TR.I["side"]] > 0)[:, None]
    seq, mask, age = store.sequence(members)
    att, summ = MM.TimeAttention(), MM.Summary()
    G = members.shape[1]
    out = att(seq.flatten(0, 1), mask.flatten(0, 1), age.flatten(0, 1))
    out2 = summ(store.units()[0][:, :, 0].flatten(0, 1), store.summary(30.0, axes).flatten(0, 1))
    assert out.shape == (G, 64) and out2.shape == (st.N, 64) and bool(mask[:, :, :2].all())
    opt = torch.optim.Adam(list(att.parameters()) + list(summ.parameters()), lr=1e-3)
    before = _params(att) + _params(summ)
    (out.pow(2).mean() + out2.pow(2).mean()).backward()
    opt.step()
    assert _changed(before[:len(_params(att))], att) and _changed(before[len(_params(att)):], summ)


def test_the_strategist_trains_a_step(tmp_path, unit_net):
    from net.strategist import battle as BT
    from net.strategist import cases as CS
    from net.strategist import cfg
    from net.strategist import ckpt as SK
    from net.strategist import net as NN
    from net.strategist import train as ST
    torch.manual_seed(0)
    net, critic = NN.StrategistNet(), NN.StrategistCritic()
    tc = cfg()["train"]
    pre = CS.Pre(CS.specs(2, torch.Generator().manual_seed(31), cases=tuple(tc["cases"])), P, "cpu")
    pre.limit = torch.full_like(pre.limit, 20.0)
    inp, ctx = BT.inputs_of(pre)
    with torch.no_grad():
        o = net(inp, ctx)
    rd, rs = BT.play(pre, o, inp, unit_net, op_net=None)
    assert rs["parcel_errors"] == 0 and rs["deploy_errors"] == 0
    rw = cfg()["reward"]
    ret = float(rw["measure"]) * BT.measure_of(pre, rd) + float(rw["trade"]) * rd["trade"]
    params = [p for p in net.parameters() if p.requires_grad] + [p for p in critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(tc["lr"]))
    before = _params(net)
    ST.update(net, critic, opt, inp, ctx, o["act"], o["lp"].detach(), ret, tc, 1.0)
    assert _changed(before, net)
    SK.save(tmp_path / "s.pt", net, critic, opt, 1, 0)
    n2, c2, ck = SK.restore(tmp_path / "s.pt")
    assert ck["kind"] == SK.KIND and _same(net, n2) and _same(critic, c2)


# ------------------------------------------------------------------ the drill engine's canary (net/drill)
def _exam_files(D, n=256, drop=(), held_n=None, n_orders=1.0, offsets=None, flank=("N", "TEACH2"), shuffle=None,
                end_shift=None, swap=None):
    """Synthetic exam files of T0-2's after-exam (C, TEACH, TEACH2, N, N1 on taught / held / G): each arm the same base
    health series shifted by its offset (paired differences exact), the flank arms' strikes on the flank; every
    declared held axis bound; shuffle - the arm whose per-battle arrays are reversed (its battle keys too); end_shift
    {arm: x} - x added to the series' last point only (the battle's end, past + 450 s); swap - the arm whose battles 0
    and 1 trade places, every battle's arm meta alike (only the health series' first point keys them)."""
    import json
    import random
    from net.tactic import scenes as SN
    axes = list(SN.cs()["sets"]["held"]["axes"])
    off = offsets or {"C": -0.01, "TEACH": 0.0, "TEACH2": 0.04, "N": 0.03, "N1": 0.03}
    rnd = random.Random(5)
    D.mkdir(parents=True, exist_ok=True)
    for nm in ("taught", "held", "G"):
        m = held_n if (nm == "held" and held_n) else n
        mt = [{"cause": b % 2 == 0, "width": 30.0 + b % 20, "blocker": "none", "start": "beside", "line": "full",
               "heading": 0.0, "flip": None if b % 2 == 0 else "front", "worn_anvil": "fresh",
               "held": axes[(b // 2) % len(axes)] if nm == "held" else None, "binds": True if nm == "held" else None}
              for b in range(m)]
        (D / f"{nm}_meta.json").write_text(json.dumps(mt), encoding="utf-8")
        base = [[rnd.uniform(-0.1, 0.1) for _ in range(130)] for _ in range(m)]
        for p, o in off.items():
            if f"{nm}_{p}" in drop or nm in drop:
                continue
            hp = [[s[0]] + [x + o for x in s[1:]] for s in base]     # (the first point equal across arms: the key)
            hp = [s[:-1] + [s[-1] + (end_shift or {}).get(p, 0.0)] for s in hp]
            pb = {"t_contact": [20.0] * m, "hp_trade": [s[-1] for s in hp],
                  "game_orders_min": [n_orders if p == "N" else 1.0] * m, "attack_orders": [2.0] * m}
            st = [[b, 1, 3, 10, 2 if p in flank else 0, "arrived", 40, 2, 3, False, 60.0, 30.0] for b in range(m)]
            meta = [{"enemy": "scene", "n": b} if swap is None else {"enemy": "scene"} for b in range(m)]
            if p == swap:
                hp = [hp[1], hp[0]] + hp[2:]
                pb = {k: [v[1], v[0]] + v[2:] for k, v in pb.items()}
            if p == shuffle:
                hp, meta = hp[::-1], meta[::-1]
                pb = {k: v[::-1] for k, v in pb.items()}
            (D / f"{nm}_{p}.json").write_text(json.dumps({"per_battle": pb, "hp_series": hp, "strikes": st, "meta": meta,
                                                          "seeds": [1, 2], "sim_key": "k",
                                                          "ckpt_sha256": {"N": "c" * 64, "N1": "p" * 64}.get(p)}),
                                           encoding="utf-8")


def _fake_run(out, key, stops):
    """Checkpoints round_01..NN of one run key with their done-records (stop / passed as given)."""
    import json
    from net.drill import run as RN
    out.mkdir(parents=True, exist_ok=True)
    for i, stop in enumerate(stops, 1):
        p = out / f"round_{i:02d}.pt"
        torch.save({"run_key": key, "round": i}, p)
        (out / f"round_{i:02d}.done.json").write_text(json.dumps(
            {"round": i, "run_key": key, "sha256": RN._sha(p), "stop": stop, "passed": stop == "relative",
             "shards_manifest": None}), encoding="utf-8")


def _starved_rows(n=200):
    """Astra's census plant (tmp/astra/engine_code/out/repro/check.py): n LEFT labels at the side rule's band."""
    from net.tactic import cols
    ones = torch.ones((n, 1), dtype=torch.bool)
    zeros = torch.zeros_like(ones)
    x = torch.zeros(n, 1, 13)
    x[..., 10] = float(cols.c_side()["frontal_deg"]) / 180
    return dict(b=torch.zeros(n, dtype=torch.long), sec=torch.arange(n), tw=torch.ones(n), net=zeros[:, 0],
                label=zeros.long(), head=ones, open=ones[..., None], dc=zeros[..., None], command=zeros[..., None],
                kinds=zeros.int(), trivial=zeros, drop=zeros, drop_base=zeros, dc_thr=zeros, ap=ones.long(), x_k=x)


def _tactic_batch(G=2, K=3):
    """One stored group-tick (record.Batch) with two head-free slots: slot 0 a FRONT label through own, slot 1 a LEFT
    label through own whose target row the walk closed after slot 0's reservation."""
    from net.drill import record as RC
    from net.tactic import heads as HD
    opn = torch.ones((1, G, K), dtype=torch.bool)
    opn[0, 1, 2] = False                                       # (slot 1's label row 2 closed by slot 0's reservation)
    rows = {"b": torch.zeros(1, dtype=torch.long), "sec": torch.zeros(1, dtype=torch.int16), "tw": torch.ones(1),
            "net": torch.zeros(1, dtype=torch.bool), "command": torch.zeros((1, G, 4)), "open": opn,
            "dc": torch.zeros((1, G, K), dtype=torch.bool), "label": torch.tensor([[1, 2]]),
            "head": torch.ones((1, G), dtype=torch.bool), "ap": torch.tensor([[0, 1]]),
            "x_k": torch.zeros((1, G, HD.SIDE_X)) + 5.0,
            "dc_thr": torch.ones((1, G), dtype=torch.bool), "drop_base": torch.zeros((1, G), dtype=torch.bool),
            "drop": torch.tensor([[True, False]]), "cards": torch.zeros((1, 1))}
    return RC.from_netbuffer(rows)


def test_the_drill_engine_canary(tmp_path, monkeypatch):
    """net/drill's canary (tmp/engine/design_merged.md section 7; brief_A2): 8 short S2 / S2N battles collected through
    the engine's recorder - rebuild compared, census without halts, a weighted step changes the weights; and each
    planted defect gives its refusal, halt or BLOCKED: a twin in another split than its base; through-own FRONT / LEFT
    and a later slot's closed target; a foreign checkpoint; a restart after a relative stop under a larger cap
    (republished, never relabelled); a restart with a lowered cap; a child's V0; a starved head (labels all in the band;
    KEEP with every alternative closed); a registry path that does not exist; a training start inside an exam range; a
    shuffled exam arm; an absent NEG and a NEG passing a skill piece; an absent G; two held battles; a stale teacher
    cache; a FAILED verdict's exit code; a safety piece failing only at the battle's end; a failed label probe;
    post-finish rows in fit and V; a mark worse by tick but not by battle, and the reverse; the adopt record (a parent
    folder with neither it nor a passed run.json refused, other weights refused, a passed run refused)."""
    import json
    from types import SimpleNamespace
    from net.drill import census as CN
    from net.drill import exam as EX
    from net.drill import fit as FT
    from net.drill import masks as MK
    from net.drill import run as RN
    from net.drill import seeds as SD
    from net.drill import validate as VA
    from net.drill import verdict as VD
    from net.drill.cards import t0_2 as C2
    from net.drill.levels import tactic as LT
    from net.drill.levels import tactic_census as TS
    from net.drill.levels import tactic_collect as TC
    from net.drill.levels import tactic_v as TV
    from net.drill.store import Store
    from net.tactic import ckpt as TK
    from net.tactic import exam_pin as EP
    from net.tactic import imitate as IM
    from net.tactic import scenes as SN
    a = SimpleNamespace(device="cpu", sim="")

    def quiet(x):
        return None
    # a twin in another split than its base: halt
    with pytest.raises(SD.SeedHalt):
        SD.check_specs({"train": [SN.side_scene(7000000)], "V": [SN.side_scene(7000000, twin=True)]})
    # through own: FRONT dropped, LEFT kept; the later slot's closed target with weight > 0: excluded
    b = _tactic_batch()
    drop = MK.tactic_drop(b.aux["drop_base"], b.aux["dc_thr"], b.aux["ap"])
    assert bool(drop[0, 0]) and not bool(drop[0, 1])
    ma = MK.of("tactic_approach", b, torch.ones((1, 2)))
    assert not bool(ma.eligible[0, 1]) and int(ma.why[0, 1]) & MK.R_SIDE_CLOSED
    mp = MK.of("tactic_pointer", b)
    assert int(mp.why[0, 1]) & MK.R_OUT_OF_SET and not bool(mp.eligible[0, 1])
    # a foreign checkpoint in --out: refused
    _fake_run(tmp_path / "foreign", "someone-else", [None])
    with pytest.raises(SystemExit):
        RN.resume_state(SimpleNamespace(resume=""), tmp_path / "foreign", "this-run", {"n_max": 4})
    # a restart after a relative stop (round 2) under a larger cap (4): round 2 republished, its reason kept
    R = tmp_path / "rel"
    _fake_run(R, "k", [None, "relative"])
    rec, ck, fin = RN.resume_state(SimpleNamespace(resume=""), R, "k", {"n_max": 4})
    res = RN.publish(R, "k", fin, ck, quiet)
    assert res["round"] == 2 and res["stop"] == "relative" and res["passed"]
    assert RN._sha(R / "final.pt") == RN._sha(R / "round_02.pt")
    # a restart with a lowered cap: refused; the cap without the marks: no final.pt
    _fake_run(tmp_path / "long", "k", [None, None, None, None])
    with pytest.raises(SystemExit):
        RN.resume_state(SimpleNamespace(resume=""), tmp_path / "long", "k", {"n_max": 2})
    res = RN.publish(tmp_path / "long", "k", {"round": 4, "stop": RN.CAP_FAIL, "passed": False},
                     tmp_path / "long" / "round_04.pt", quiet)
    assert not res["passed"] and not (tmp_path / "long" / "final.pt").exists()
    # a FINISHED run (relative stop at round 3) restarted under a lowered cap (2): refused, never republished
    _fake_run(tmp_path / "fin3", "k", [None, None, "relative"])
    with pytest.raises(SystemExit):
        RN.resume_state(SimpleNamespace(resume=""), tmp_path / "fin3", "k", {"n_max": 2})
    # a stale own lock (same run key, its heartbeat silent > STALE_S) taken over and said; another key's refused
    import os
    import time as _t
    from net.drill import store as STO
    lk = tmp_path / "collect_r01.writer.lock"
    said = []
    for key_, ok_ in (("k", True), ("other", False)):
        lk.write_text(json.dumps({"pid": 999999, "host": "a-gone-container", "run_key": key_, "t": 0}), encoding="utf-8")
        os.utime(lk, (_t.time() - STO.STALE_S - 60,) * 2)
        if ok_:
            with STO.Lock(lk, "k", said.append):
                assert json.loads(lk.read_text(encoding="utf-8"))["run_key"] == "k"
            assert said and "lock_taken_over" in said[0] and not lk.exists()
        else:
            with pytest.raises(STO.StoreRefused):
                with STO.Lock(lk, "k", said.append):
                    pass
    # T0-1 with its reference: stops when equal to it; not when path T's recall is lower by > 2 errors
    from net.drill.cards import t0_1 as C1
    card1 = C1.card()

    def c_(p, n=2000):
        return {"share": p, "n": n, "two_errors": round(2 * (max(p * (1 - p), 1e-6) / n) ** 0.5, 4)}

    def tv1(fol_t):
        return {"followed_N": {"pooled": c_(0.80), "cells": {}}, "followed_T": {"pooled": c_(fol_t)},
                "false_N": {"tick": c_(0.001, 60000), "by_kind": {}}, "false_T": {"tick": c_(0.001, 60000)},
                "nocause_false": c_(0.02, 200), "uncaused": c_(0.05, 2500)}
    rep = []
    ok, why, jd = VA.relative_stop(card1, tv1(0.80), None, tv1(0.80), rep)
    assert ok and jd, why
    assert any("absolute" in x for x in rep) and any("minimum 50" in x for x in rep)
    ok, why, jd = VA.relative_stop(card1, tv1(0.74), None, tv1(0.80))
    assert not ok and jd and any("worse than the reference" in x for x in why), why
    ok, why, jd = VA.relative_stop(card1, tv1(0.80), None, None)
    assert not ok and not jd
    # T0-2: path N's S2N false-flank cell at n 44 is judged (its minimum 30); at 29 it is not
    card2 = C2.card()

    def tv2(n_s2n):
        grp = {c: c_(0.95, 400) for c in ("front", "left", "right")}
        side = {"all": {**grp, "missing": c_(0.02, 1200)}, "S2N": {"false_flank": c_(0.0, n_s2n)}}
        return {"side_engine": {"T": {"all": {**grp, "missing": c_(0.0, 1200)}, "S2N": {"false_flank": c_(0.0, 80)}},
                                "N": side},
                "parts": {"retention": {"followed_N": {"pooled": c_(0.8)}, "false_N": {"tick": c_(0.001, 60000)}}}}
    ok, why, jd = VA.relative_stop(card2, tv2(44), tv2(44))
    assert ok and jd, why
    ok, why, jd = VA.relative_stop(card2, tv2(29), tv2(29))
    assert not jd and any("S2N" in x and "< 30" in x for x in why), why
    # errors by battle: a false rate worse by tick but not paired by battle (one battle carries the excess), and the
    # reverse (a small excess in every other battle) - the cells as levels/tactic_v.ByBattle leaves them
    def bbc(x, n):
        c = c_(sum(x) / sum(n), sum(n))
        c.update(two_errors_tick=c["two_errors"], two_errors=VA.cluster_two_err(x, n), bb={"keys": "k", "x": x, "n": n})
        return c
    ref_x = [1] * 100
    c_one, r_one = bbc([61] + [1] * 99, [100] * 100), bbc(ref_x, [100] * 100)
    assert VA._worse("<=", {k: c_one[k] for k in ("share", "n")}, {k: r_one[k] for k in ("share", "n")})[0]
    w, e = VA._worse("<=", c_one, r_one)
    assert not w and VA.pair_of(c_one, r_one) == "paired by battle", e
    c_alt, r_alt = bbc([5 + b % 2 for b in range(50)], [10] * 50), bbc([5] * 50, [10] * 50)
    assert not VA._worse("<=", {k: c_alt[k] for k in ("share", "n")}, {k: r_alt[k] for k in ("share", "n")})[0]
    w, e = VA._worse("<=", c_alt, r_alt)
    assert w and e < 0.02, e
    # the adopt record: written by the command for a run that did not pass; parent_of takes the adopted reference
    # and refuses a folder with neither a passed run.json nor an adopt record, other weights, a passed run
    ref_w = C1.REFERENCE["path"]
    AD = tmp_path / "adopt_run"
    _fake_run(AD, "k", [None, RN.CAP_FAIL])
    (AD / "run.json").write_text(json.dumps({"round": 2, "stop": RN.CAP_FAIL, "passed": False}), encoding="utf-8")
    pa = SimpleNamespace(card="t0_2", init=ref_w, parent_run=str(tmp_path / "nothing"))
    (tmp_path / "nothing").mkdir()
    with pytest.raises(SystemExit, match="neither"):
        RN.parent_of(pa, card2, LT, "cpu", quiet)
    with pytest.raises(SystemExit, match="only the accepted"):
        RN.main(["adopt", "--card", "t0_1", "--out", str(AD), "--weights", str(AD / "round_01.pt"), "--reason", "x"])
    assert RN.main(["adopt", "--card", "t0_1", "--out", str(AD), "--weights", ref_w, "--reason", "canary"])["passed"]
    ad = json.loads((AD / RN.ADOPT).read_text(encoding="utf-8"))
    assert ad["sha256"] == C1.REFERENCE["sha256"] and [r["round"] for r in ad["rounds"]] == [1, 2]
    _, sha_p = RN.parent_of(SimpleNamespace(card="t0_2", init=ref_w, parent_run=str(AD)), card2, LT, "cpu", quiet)
    assert sha_p == C1.REFERENCE["sha256"]
    (AD / "run.json").write_text(json.dumps({"passed": True, "sha256": "x" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="passed true"):
        RN.main(["adopt", "--card", "t0_1", "--out", str(AD), "--weights", ref_w, "--reason", "canary"])
    # a child's V0 (another parent's sha in v_round_00.json): refused
    (tmp_path / "v0.json").write_text(json.dumps({"parent_sha": "child", "run_key": "k", "v": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        RN.v0_of(tmp_path / "v0.json", "parent", "k", lambda: {})
    # a starved head: all LEFT labels in the band; KEEP labels with every alternative closed
    card = SimpleNamespace(census={}, dont_care_max={}, band_share_max=0.15)
    rows = _starved_rows()
    meta = [{"kind": "scene", "recipe": "S2", "start": "second_row"}]
    w, _ = TS.weights(rows, meta)
    bt = TS.batch_of(rows)
    with pytest.raises(CN.CensusHalt, match="approach/left"):
        CN.check(TS.classes(bt, w, True), meta, bt.b, card)
    rows.update(label=-torch.ones((200, 1), dtype=torch.long), open=torch.zeros((200, 1, 1), dtype=torch.bool))
    w, _ = TS.weights(rows, meta)
    bt = TS.batch_of(rows)
    with pytest.raises(CN.CensusHalt, match="pointer/keep"):
        CN.check(TS.classes(bt, w, False), meta, bt.b, card)
    # a registry path that does not exist; a training start inside an exam range
    with pytest.raises(SD.SeedHalt):
        SD.load(tmp_path / "no_registry.json")
    entries = SD.load(RN.REGISTRY)
    taught = next(e for e in entries if e["name"] == "exam taught")
    s2 = next(e for e in entries if e["name"] == "S2 + S2N twins r1")
    assert SD.assert_starts(entries, {"t0_1", "t0_2"}, "train", [("tactic.side", None, s2["lo"])]) == 1
    with pytest.raises(SD.SeedHalt):
        SD.assert_starts(entries, {"t0_1", "t0_2"}, "train", [("tactic.side", None, taught["lo"] + 3)])
    # a twin's base redrawn at a retry t > 3 (realised): inside the registry
    los = {e["name"]: e["lo"] for e in entries}
    far = []
    for e in entries:
        if e["split"] == "train" and e["name"].endswith("(realised)"):
            lo = los[e["name"].split(" twin bases ")[0]]
            far += [(e["key"], x) for x, _ in [(e["lo"], 0)] + [tuple(s_) for s_ in e["subs"]]
                    if round((x - lo) / SD.RETRY_STEP) > 3]
    assert far, "no realised retry beyond t 3 in the training rounds"
    assert SD.assert_starts(entries, {"t0_1"}, "train", [("tactic.make", far[0][0], far[0][1])]) == 1
    # the exam's verdict: clean PASSED; a shuffled arm refused; absent NEG, a NEG passing a skill piece, absent G,
    # two held battles: BLOCKED; a FAILED verdict exits non-zero
    parent, cand = "p" * 64, "c" * 64
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"verdict": {"pass": True, "n_ok": True}}), encoding="utf-8")
    gate = {"kind": "half_the_gap"}
    t01 = {"passed": True, "sets": {s: {"passed_at": {"450": True, "end": True}, "marks_gate": {
        "followed_N": {"passed": True}, "false_tick": {"passed": True}}} for s in ("S", "G")}}

    def neg_dir(nm, **kw):
        D = tmp_path / nm
        _exam_files(D, **kw)
        (tmp_path / f"{nm}.log").write_text(json.dumps({"player": "N", "ckpt_sha256": parent}) + "\n", encoding="utf-8")
        return D
    neg = neg_dir("neg", offsets={"C": -0.01, "TEACH": 0.0, "TEACH2": 0.04, "N": 0.0, "N1": 0.0}, flank=("TEACH2",))
    neg_good = neg_dir("neg_good")

    def verdict(nm, neg_=neg, **kw):
        D = tmp_path / nm
        _exam_files(D, **kw)
        return VD.after(D, C2, t01, neg_, gate=gate, probe=str(probe), parent_sha=parent, cand_sha=cand)["m2"]
    v = verdict("clean")
    assert v["verdict"] == "PASSED", (v["blocked"], v["failed"])
    with pytest.raises(EX.ExamRefused):
        verdict("shuffled", shuffle="N")
    with pytest.raises(EX.ExamRefused, match="battle keys differ"):       # (Astra's swapped archived battles, V5)
        verdict("swapped", swap="N")
    for nm, kw in (("no_neg", {"neg_": None}), ("neg_passes", {"neg_": neg_good}), ("no_G", {"drop": ("G",)}),
                   ("two_held", {"held_n": 2})):
        v = verdict(nm, **kw)
        assert v["verdict"] == "BLOCKED", (nm, v["verdict"], v["failed"])
    assert verdict("fails", n_orders=2.0)["verdict"] == "FAILED"
    # a safety piece passing at + 450 s and failing at the battle's end: FAILED; a probe file that fails: BLOCKED
    v = verdict("end_only", end_shift={"N": -0.05})
    assert v["verdict"] == "FAILED" and "no_harm_G@end" in v["failed"] and "no_harm_G@450" not in v["failed"], v
    cells = {w: [{"diff": -0.03 if w == "end" else 0.01, "two_errors": 0.001, "n": 1100}] for w in C2._windows()}
    st_w = {x["window"]: x["status"] for x in C2._gated("no_harm_G", "safety", cells, -0.005)}   # (Astra's plant)
    assert st_w == {450: "pass", "end": "fail"}, st_w
    probe_bad = tmp_path / "probe_bad.json"
    probe_bad.write_text(json.dumps({"verdict": {"pass": False, "n_ok": True}}), encoding="utf-8")
    v = VD.after(tmp_path / "clean", C2, t01, neg, gate=gate, probe=str(probe_bad), parent_sha=parent, cand_sha=cand)["m2"]
    assert v["verdict"] == "BLOCKED" and any(b.startswith("probe ") for b in v["blocked"]), v["blocked"]
    probe_bad.write_text(json.dumps({"passed": True}), encoding="utf-8")       # (a legacy probe file: refused, V6)
    v = VD.after(tmp_path / "clean", C2, t01, neg, gate=gate, probe=str(probe_bad), parent_sha=parent, cand_sha=cand)["m2"]
    assert v["verdict"] == "BLOCKED" and "probe NOT JUDGED" in v["blocked"], v["blocked"]
    # the exam's N arms of other weights than the candidate's: BLOCKED (V4)
    v = VD.after(tmp_path / "clean", C2, t01, neg, gate=gate, probe=str(probe), parent_sha=parent, cand_sha="d" * 64)["m2"]
    assert v["verdict"] == "BLOCKED" and "weights NOT JUDGED" in v["blocked"], v["blocked"]
    monkeypatch.setattr(C2, "t01_compare", lambda x, y: t01)
    gf = tmp_path / "gate.json"
    gf.write_text(json.dumps(gate), encoding="utf-8")
    argv = ["after", "--card", "t0_2", "--gate", str(gf), "--probe", str(probe), "--neg", str(neg), "--parent-sha",
            parent, "--t01-new", "x", "--t01-ref", "y", "--run", str(tmp_path / "cand_run")]
    (tmp_path / "cand_run").mkdir()
    # (the candidate's final.pt carries its parent - verdict binds --parent-sha to its init_sha256; its N files are
    # played with its own sha)
    torch.save({"init_sha256": parent}, tmp_path / "cand_run" / "final.pt")
    cand_real = RN._sha(tmp_path / "cand_run" / "final.pt")
    for dn in ("fails", "clean"):
        for f in (tmp_path / dn).glob("*_N.json"):
            x = json.loads(f.read_text(encoding="utf-8"))
            f.write_text(json.dumps({**x, "ckpt_sha256": cand_real}), encoding="utf-8")
    (tmp_path / "cand_run" / "run.json").write_text(json.dumps({"passed": True, "sha256": cand_real}), encoding="utf-8")
    with pytest.raises(SystemExit, match="init_sha256"):
        VD.main([x if x != parent else "e" * 64 for x in argv] + ["--dir", str(tmp_path / "clean")])
    assert VD.main(argv + ["--dir", str(tmp_path / "fails")]) == 1
    assert VD.main(argv + ["--dir", str(tmp_path / "clean")]) == 0
    (tmp_path / "t01_after").mkdir()                                      # (one entry for T0-1 too, V1)
    (tmp_path / "t01_after" / "after.json").write_text("{}", encoding="utf-8")
    assert VD.main(["after", "--card", "t0_1", "--dir", str(tmp_path / "t01_after")]) == 1
    # a stale teacher cache: the replayed first chunk differs - halt; an interrupted build is played again
    vc = VA.TeacherCache(tmp_path / "vc")

    def play(x):
        def f(st_):
            st_.put("scene", {"b": torch.zeros(4, dtype=torch.long), "x": x}, [{"spec": 0}], None)
            return {"done_cnt": [0, 0]}
        return f
    Store(tmp_path / "vc" / "k1").put("scene", {"b": torch.zeros(1, dtype=torch.long)}, [{}], None)
    assert "interrupted" in vc.ensure("k1", play(torch.arange(4)), log=quiet)
    assert vc.ensure("k1", play(torch.arange(4)), play(torch.arange(4)), log=quiet).startswith("read (first chunk")
    with pytest.raises(VA.StaleCache):
        vc.ensure("k1", play(torch.arange(4)), play(torch.arange(4) + 1), log=quiet)
    # the clean collection: 8 battles, TEACH2 driving, the level's own input on the first chunk (rebuild compared)
    sp = IM._limit(IM.s2_specs(7000000, 4), 100)
    st = Store(tmp_path / "shards")
    buf = TC.ShardBuffer(st, side=True, check=16)
    buf.part = "scene_teacher"
    TC.play_chunks("TEACH2", sp, True, [5], 8, a, P, lambda ii: EP.build_under("TEACH2"), lambda d_: buf.hook(False),
                   lambda ii: buf.flush(ii, sp, "scene", False), labeller="TEACH2", needs_input_first=1,
                   on_driver=lambda s: setattr(buf, "seen", s))
    assert buf.checked > 0 and buf.check_max <= 1e-3
    rows, cards, meta = st.mixture(remap=LT.remap)
    assert "executed" in rows and "proposed" in rows
    w, _ = TS.weights(rows, meta)
    bt = TS.batch_of(rows)
    assert bool(((bt.proposed >= 0) == (bt.status == 1)).all())
    cen = CN.check(TS.classes(bt, w, True), meta, bt.b, card, smoke=8)
    assert not cen["halts"] and int((w > 0).sum()) > 0
    net, _ = TK.from_v2("weights/tactic/attack1/final.pt", "cpu")
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=1e-3)
    before = _params(net)
    FT.passes(LT, net, opt, rows, cards, w, "cpu", quiet, torch.Generator().manual_seed(1), 1, 1024, 1.0,
              batch_of=LT.batch_of)
    assert _changed(before, net)
    # post-finish rows: the legacy guard off (imitate.done_rows: none over) and battles 0-3 of the chunk marked over
    # from 20 s while the chunk goes on - the engine's own guard (st.done) takes them out of the fit's weights and of
    # V's teacher path
    from net.tactic import tags as TG
    monkeypatch.setattr(IM, "done_rows", lambda v: torch.zeros(v.idx.shape, dtype=torch.bool, device=v.fx.s.device))
    rec_call = TG.Record.__call__

    def ended(self, v):
        if float(v.now) >= 20 and v.sim is not None:
            v.sim.done[:4] = True
        return rec_call(self, v)
    monkeypatch.setattr(TG.Record, "__call__", ended)
    sp2 = IM._limit(IM.s2_specs(7000100, 4), 100)
    vc2 = VA.TeacherCache(tmp_path / "vcache")
    vc2.ensure("k", lambda st_: TV._play_T([("scene", sp2, True, [6], 8)], a, P, "TEACH2", True, st_, quiet), log=quiet)
    rows2, cards2, meta2, _ = vc2.read("k", remap=LT.remap)
    post = rows2["done_own"].bool()
    assert bool(post.any()), "no post-finish rows planted"
    w2, _ = TS.weights(rows2, meta2)
    assert float(w2[post].abs().sum()) == 0.0
    mT, _ = TV.ask_T(net, rows2, cards2, meta2)
    seen = set(zip(mT.rows[0]["b"].tolist(), mT.rows[0]["sec"].tolist()))
    gone = set(zip(rows2["b"][post].tolist(), rows2["v_sec32"][post].tolist()))
    assert not (seen & gone)
    monkeypatch.undo()                     # (the post-finish patches above off: the smokes below run the real code)
    # Astra engine_code3's plants (round 5). A one-battle cell, a roster missing one battle: not judged for a stop;
    # display rounding: the exact ratios of the counts
    assert VA.compare("<=", bbc([3, 0], [44, 0]), bbc([0, 0], [80, 0]))["unjudged"]
    x5 = [10, 80] * 50
    r5 = bbc(x5, [100] * 100)
    c5 = bbc([i + 1 for i in x5] + [46], [100] * 101)
    c5["bb"]["keys"] = "plus one"
    assert VA.compare("<=", c5, r5)["unjudged"] and not VA.compare("<=", c5, r5, stop=False)["unjudged"]
    n5 = [100] * 512
    x6 = [1] * 47 + [0] * 465
    assert not VA.compare("<=", bbc([x6[0] + 2] + x6[1:], n5), bbc(x6, n5))["worse"]
    # the battle digest: a changed chunk seed or recipe on the same geometry is another roster
    m5 = [{"kind": "scene", "spec": 0}]

    def parts5(recipe, seed):
        return [("scene", [{"mine": [1], "recipe": recipe, "seed": seed}], True, [seed], 1)]
    assert TV.battle_order(m5, parts5("S2", 100))[1] != TV.battle_order(m5, parts5("S2N", 101))[1]
    # a V0 of another V play key: refused
    (tmp_path / "v0k.json").write_text(json.dumps({"parent_sha": "p", "run_key": "k", "v_key": "old", "v": {}}),
                                       encoding="utf-8")
    with pytest.raises(SystemExit, match="V key"):
        RN.v0_of(tmp_path / "v0k.json", "p", "k", lambda: {}, v_key="new")
    # a hand-written adopt record: refused, of other weights and of the reference alike
    HW = tmp_path / "hand"
    HW.mkdir()
    for sha_hw in (RN._sha(AD / "round_01.pt"), C1.REFERENCE["sha256"]):
        (HW / RN.ADOPT).write_text(json.dumps({"card": "t0_1", "sha256": sha_hw}), encoding="utf-8")
        init_hw = str(AD / "round_01.pt") if sha_hw != C1.REFERENCE["sha256"] else ref_w
        with pytest.raises(SystemExit, match="refused"):
            RN.parent_of(SimpleNamespace(card="t0_2", init=init_hw, parent_run=str(HW)), card2, LT, "cpu", quiet)
    # the label probe before the training: none or a failed one refused, a passed one taken
    with pytest.raises(SystemExit, match="label probe"):
        RN.probe_of(SimpleNamespace(card="t0_2", smoke=0, probe=""))
    pb5 = tmp_path / "probe5.json"
    pb5.write_text(json.dumps({"card": "T0-2", "verdict": {"pass": False, "n_ok": True}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="passed probe"):
        RN.probe_of(SimpleNamespace(card="t0_2", smoke=0, probe=str(pb5)))
    pb5.write_text(json.dumps({"card": "T0-2", "verdict": {"pass": True, "n_ok": True}}), encoding="utf-8")
    assert RN.probe_of(SimpleNamespace(card="t0_2", smoke=0, probe=str(pb5)))
    # a same-key cache replayed under another process-local map handle: remapped before the compare, taken
    from unittest.mock import patch as _patch
    map_id = [1]

    def play5(parts, a_, P_, teacher, side, store, log, first_chunk=False):
        store.put("scene", {"b": torch.tensor([0]), "raw": torch.tensor([map_id[0]])}, [{}])
        return {}

    def remap5(rows, maps):
        return {**rows, "raw": torch.tensor([map_id[0]])}
    with _patch.object(TV, "_play_T", play5), _patch.object(LT, "remap", remap5):
        TV.teacher_path(tmp_path / "vc5", "same-key", (), None, None, "TEACH2", True, log=quiet)
        map_id[0] = 2
        TV.teacher_path(tmp_path / "vc5", "same-key", (), None, None, "TEACH2", True, log=quiet)
    # smoke parity (the coordinator's acceptance line 8): both cards' one-round CPU smokes bit-equal to the legacy
    # `learn train` smokes of the same tree and seeds (tmp/engine/a/legacy_att, legacy1 - cmp2 of round 2)
    S1 = json.dumps({"kind": "rounds", "passes": 1, "n_min": 1, "n_max": 1, "stop": "relative"})
    AS = tmp_path / "adopt_smoke"
    RN.main(["adopt", "--card", "t0_1", "--out", str(AS), "--weights", ref_w, "--reason", "canary smoke parent"])
    # (the legacy smokes and the engine's round-3 smokes were made with 6 CPU threads - DOCK_CPUS=6; another thread
    # count reorders the float reductions: 8 threads gave max |diff| 4e-6, tmp/engine/a/r5/parity2.py)
    th0 = torch.get_num_threads()
    torch.set_num_threads(6)
    for cd, leg, extra in (("t0_1", "tmp/engine/a/legacy_att", []),
                           ("t0_2", "tmp/engine/a/legacy1", ["--init", ref_w, "--parent-run", str(AS)])):
        o = tmp_path / f"parity_{cd}"
        RN.main(["train", "--card", cd, "--smoke", "8", "--no-v", "--device", "cpu", "--schedule", S1, "--out", str(o)]
                + extra)
        ne = torch.load(o / "round_01.pt", map_location="cpu", weights_only=False)["net"]
        nl = torch.load(f"{leg}/round_01.pt", map_location="cpu", weights_only=False)["net"]
        assert set(ne) == set(nl) and all(torch.equal(ne[k], nl[k]) for k in nl), f"{cd}: not bit-equal to {leg}"
    torch.set_num_threads(th0)
    # the exam's weights piece: all six set x arm files, a held-set log never authenticates G; T0-1's retention in the
    # exam judged by battle (after.json's battle counts), a legacy after.json NOT JUDGED
    W6 = tmp_path / "w6"
    _exam_files(W6)
    (W6 / "G_N1.json").unlink()
    assert VD.weights_piece(W6, ("taught", "held", "G"), "c" * 64, "p" * 64)["status"] == "NOT JUDGED"
    from net.tactic import learn as LN5

    def cell5(x, n):
        c = LN5.share(sum(x), sum(n))
        return TV.ByBattle(torch.arange(len(n)), len(n), list(range(len(n))), "k").put(c, torch.tensor(x),
                                                                                      torch.tensor(n), battle=True)
    for legacy in (False, True):
        for nm5, cc in (("child", cell5([61] + [1] * 99, [100] * 100)), ("parent", cell5([1] * 100, [100] * 100))):
            fo = tmp_path / f"t01x_{legacy}" / nm5
            fo.mkdir(parents=True)
            if legacy:
                cc = {k: v for k, v in cc.items() if k != "bb"}
            tab = {"followed_N": {"pooled": cell5([95] * 100, [100] * 100)}, "false_N": {"tick": cc}}
            (fo / "after.json").write_text(json.dumps({s: tab for s in ("S", "G")}), encoding="utf-8")
            for s in ("S", "G"):
                (fo / f"{s}_N.json").write_text(json.dumps({"hp_series": [[0]] * 100, "per_battle": {
                    "hp_trade": [0] * 100, "t_contact": [0] * 100}}), encoding="utf-8")
        mg = C2.t01_compare(tmp_path / f"t01x_{legacy}" / "child", tmp_path / f"t01x_{legacy}" / "parent")
        got = mg["sets"]["S"]["marks_gate"]["false_tick"]["passed"]
        assert got is (None if legacy else True), (legacy, mg["sets"]["S"]["marks_gate"])
