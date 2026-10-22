"""The kind head's dense signal (docs/en/training/general.md "The kind's own signal"): the outcome Q (kind_q.py) - the
outcome over the horizon from the battle itself (Runtime's window), its features, its fit, gate and shrink, its
advantage into the kind term, the agreement with the branches; the branches' centring and the "same" diagnosis
(branch.py); the kind term's sources (learn.py --general-kind-q). torch: the training container, on the CPU."""
import dataclasses
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import branch as gbranch                 # noqa: E402
from tools.nn.general import kind_q as kq                      # noqa: E402
from tools.nn.general import learn                             # noqa: E402
from tools.nn.general import runtime as gruntime               # noqa: E402
from tools.nn.train import ability_branch as ab                # noqa: E402
from tools.nn.train import rollout, run                        # noqa: E402

from tests.tools.test_general_branch_torch import GCFG, battles, collected, columns, grabbed, nets, same  # noqa: E402

NAN = float("nan")


# --- the outcome's window (Runtime) ---------------------------------------------------------------------------
def bare(B=2, h=10.0):
    rt = gruntime.Runtime.__new__(gruntime.Runtime)
    rt.B, rt.dev, rt.q_h, rt.q_wait = B, torch.device("cpu"), h, None
    rt.r = {"q_t0": torch.zeros(2 * B), "q_tr0": torch.zeros(2 * B), "q_trh": torch.full((2 * B,), NAN)}
    rt.b = {"epoch": torch.zeros(B, dtype=torch.long), "bid": torch.arange(100, 100 + B)}
    return rt


def at(*t):
    return SimpleNamespace(st=SimpleNamespace(t=torch.tensor(t)))


def test_an_ended_plans_outcome_is_the_trade_over_the_horizon_since_its_start():
    rt = bare()
    rt.r["q_t0"] = torch.tensor([0.0, 5.0, 0.0, 0.0])
    rt.r["q_tr0"] = torch.tensor([0.0, 0.1, -0.2, 0.0])
    rt.r["q_trh"] = torch.tensor([NAN, NAN, 0.3, NAN])        # row 2's plan was in force at its horizon
    rt._q_add(torch.tensor([0, 1, 2]), torch.tensor([7, 8, 9]))
    no = torch.tensor([False, False])
    rt._q_resolve(at(8.0, 8.0), torch.tensor([0.05, 0.2, 0.4, 0.0]), no)
    out = rt._q_take()
    assert out["qid"].tolist() == [9]                          # the trade at its horizon less the one at its start
    assert out["bid"].tolist() == [100]                        # row 2: side 2 of battle 0 - battle 0's number
    assert float(out["y"][0]) == pytest.approx(0.5) and float(out["span"][0]) == 10.0 and not bool(out["ended"][0])
    assert rt._q_take() is None                                # (taken once; rows 0 and 1 still wait: 8 s, 3 s)
    rt._q_resolve(at(10.0, 10.0), torch.tensor([0.07, 0.25, 0.4, 0.0]), no)
    out = rt._q_take()
    assert out["qid"].tolist() == [7] and float(out["y"][0]) == pytest.approx(0.07)      # 10 s since t0 = 0
    # battle 1 (row 1) ends at 12 s, 7 s into its window: the trade then is final, marked ended
    rt._q_resolve(at(11.0, 12.0), torch.tensor([0.0, 0.4, 0.0, 0.0]), torch.tensor([False, True]))
    out = rt._q_take()
    assert out["qid"].tolist() == [8] and float(out["y"][0]) == pytest.approx(0.3)
    assert float(out["span"][0]) == pytest.approx(7.0) and bool(out["ended"][0])
    assert len(rt.q_wait["qid"]) == 0
    # a battle restarted in the slot (its epoch moved): an older outcome there is dropped, never given
    rt._q_add(torch.tensor([3]), torch.tensor([10]))
    rt.b["epoch"][1] += 1
    rt._q_resolve(at(30.0, 30.0), torch.zeros(4), no)
    assert rt._q_take() is None and len(rt.q_wait["qid"]) == 0


def windowed(h, steps=3, branches=False):
    """collected()'s chunk with the outcomes' window h on, and the trade after every decision -> (env, actor, critic,
    batch, history [(t [B], trade [2B])], waits {qid: (row, t0, tr0)})."""
    torch.manual_seed(7)
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    rt = env._general(actor)                    # (made before the first step: its deployment's plans tracked too)
    rt.q_h = h
    hist, waits = [], {}
    cls = gruntime.Runtime
    after, add = cls.after, cls._q_add

    def after_(self, e, r_rows, finished):
        after(self, e, r_rows, finished)
        if self is rt:                          # (not a branch's copy)
            hist.append((e.st.t.float().clone(), rt.side_trade(e).clone()))

    def add_(self, rows, ids):
        for r_, i in zip(rows.tolist(), ids.tolist()):
            waits[i] = (r_, float(self.r["q_t0"][r_]), float(self.r["q_tr0"][r_]))
        add(self, rows, ids)
    cls.after, cls._q_add = after_, add_
    try:
        torch.manual_seed(8)
        env.step(actor, crit)
        rt.ret, rt.gamma = "critic", 0.9997
        if branches:
            rt.brancher = gbranch.Brancher(gbranch.Config(share=1.0, copies=2, horizon_s=3.0, cap=0, batch=4), "cpu")
        rt.r["t_plan"] -= 31.0
        torch.manual_seed(9)
        batch = rollout.collect(env, actor, crit, steps)
    finally:
        cls.after, cls._q_add = after, add
    return env, actor, crit, batch, hist, waits


@pytest.mark.parametrize("h", [1.0, 2.0])
def test_the_battles_outcomes_are_the_recorded_trade_and_leave_the_battles_bit_for_bit(h):
    env0, _, _, b0 = collected(False)
    env, actor, crit, batch, hist, waits = windowed(h)
    g = batch["general"]
    for k in ("obs", "critic_obs", "lp", "value", "reward", "done", "h0"):
        same(b0[k], batch[k], k)
    for k in env0.st.u:
        assert torch.equal(env0.st.u[k], env.st.u[k]), k                    # the main battles untouched
    for k in b0["general"]["ready"]:
        assert torch.equal(b0["general"]["ready"][k], g["ready"][k]), k
    ready, out = g["ready"], g["q_out"]
    assert "qid" in ready and len(set(ready["qid"].tolist())) == len(ready["qid"])
    assert out is not None and len(out["qid"]) and set(out["qid"].tolist()) <= set(ready["qid"].tolist())
    B = env.B
    bids = env.general_rt.b["bid"]
    assert len(set(bids.tolist())) == B and bool((bids >= 0).all())    # every battle its own number
    for qid, y, span, ended, bid in zip(out["qid"].tolist(), out["y"].tolist(), out["span"].tolist(),
                                        out["ended"].tolist(), out["bid"].tolist()):
        row, t0, tr0 = waits[qid]
        b = row % B
        assert bid == int(bids[b])                                     # both sides of a battle: its number
        # its start's trade: the one after the decision before (0 at the battle's start)
        before = [tr[row] for t, tr in hist if float(t[b]) == t0]
        assert (tr0 == pytest.approx(float(before[-1]), abs=1e-6)) if before else tr0 == 0.0
        # its end's: the first decision at least h seconds on
        end = next(tr[row] for t, tr in hist if float(t[b]) - t0 >= h - 1e-6)
        assert y == pytest.approx(float(end) - tr0, abs=1e-6) and span == h and not ended
    # the trade is the branches' (ability_branch.trade) of each side
    tr = env.general_rt.side_trade(env)
    for s in (1, 2):
        side = torch.full((B,), s)
        torch.testing.assert_close(tr[(s - 1) * B:s * B], ab.trade(env.st.u, side, env.weights.rout_share).float())


# --- the features -------------------------------------------------------------------------------------------
def test_the_qs_input_is_what_the_kind_head_reads_of_each_group():
    env, actor, crit, batch, _, _ = windowed(2.0)
    ready = batch["general"]["ready"]
    gen = actor.general
    G, K, D = gen.cfg.places, gen.cfg.kinds, gen.cfg.card
    X, lp = kq.features(gen, ready, minibatch=3)                      # (minibatches agree with one pass)
    X1, lp1 = kq.features(gen, ready)
    torch.testing.assert_close(X, X1)
    assert X.shape == (len(ready["kind"]), G, kq.width(D, K)) and lp.shape == (len(ready["kind"]), G, K)
    mine = ready["active"][:, :G]
    assert torch.count_nonzero(X[~mine]) == 0
    with torch.no_grad():
        c, d = gen(ready, act={k: ready[k] for k in learn.CHOICES})
        torch.testing.assert_close(X[..., :D][mine], d["query"].float()[mine])      # the kind head's own card
        from tools.nn.general import kind_critic as kc
        torch.testing.assert_close(X[..., D:D + kc.WIDTH], kc.features(gen, ready))
        torch.testing.assert_close(X[..., D + kc.WIDTH:D + kc.WIDTH + K][mine],
                                   gen.prev_kind_hot(ready, c, K).float()[mine])
        torch.testing.assert_close(X[..., D + kc.WIDTH + K:][mine], d["kind_ok"].float()[mine])   # the kinds allowed
        torch.testing.assert_close(lp, d["kind_logp"].float())
        # the inertia is kind_stick's: the kind head's bonus = B x the one-hot
        B = float(gen.cfg.kind_stick)
        bonus = gen.kind_bonus(ready, c, torch.zeros(len(ready["kind"]), G, K))
        torch.testing.assert_close(bonus, B * gen.prev_kind_hot(ready, c, K).float())


# --- the fit, the gate, the shrink --------------------------------------------------------------------------
CARD, K = 16, 6


def synthetic(n, kind_eff=0.03, state=0.03, noise=0.01, seed=0):
    """One group a plan: y = state x X[0] + kind_eff x the kind's number + noise (a uniform policy)."""
    g = torch.Generator().manual_seed(seed)
    W = kq.width(CARD, K)
    X = torch.randn(n, 1, W, generator=g)
    a = torch.randint(0, K, (n, 1), generator=g)
    eff = torch.linspace(-1, 1, K)
    y = state * X[:, 0, 0] + kind_eff * eff[a[:, 0]] + noise * torch.randn(n, generator=g)
    lp = torch.log(torch.full((n, 1, K), 1.0 / K))
    ready = {"kind": a, "active": torch.ones(n, 2, dtype=torch.bool), "qid": torch.arange(n)}
    out = {"qid": torch.arange(n), "y": y, "span": torch.full((n,), 60.0), "ended": torch.zeros(n, dtype=torch.bool),
           "bid": torch.arange(n) // 4}                              # 4 plans a battle
    return X, lp, ready, out, eff


def test_the_q_learns_a_known_outcome_its_kind_part_and_shrinks_noise():
    q = kq.OutcomeQ(K, CARD, warmup=2000, steps=600, seed=1)
    X, lp, ready, out, eff = synthetic(6000)
    q.hold(X, lp, ready)
    n, ended, span = q.join(out)
    assert n == 6000 and len(q) == 6000 and q.wait is not None and len(q.wait["qid"]) == 0 and span == 60.0
    st = q.fit()
    assert st["ev"] > 0.6 and q.ready() and st["coef"] > 0.5 and st["shrink"] > 0.5 and st["ev_kind"] > 0.3
    assert st["shrink"] == pytest.approx(min(1.0, max(0.0, st["coef"] - kq.SE_K * st["coef_se"])), abs=1e-6)
    # held out by battle: every 10th battle whole (its 4 plans), never a battle split
    val = kq.held_out(q.bid)
    assert st["held"] == int(val.sum()) and set(q.bid[val].tolist()).isdisjoint(q.bid[~val].tolist())
    assert int(val.sum()) == 4 * len(set(q.bid[val].tolist()))
    # its advantage: Q(chosen) - E_pi Q, the kinds' order kept
    qq = q.predict(torch.zeros(1, kq.width(CARD, K)), shrunk=False)[0]
    assert torch.equal(torch.argsort(qq), torch.argsort(eff))
    # only the outcomes: the same inputs with pure noise - not ready (ev below the floor), nothing to give
    pure = kq.OutcomeQ(K, CARD, warmup=2000, steps=300, seed=2)
    pure.hold(X, lp, ready)
    pure.join(dict(out, y=torch.randn(6000, generator=torch.Generator().manual_seed(3)) * 0.05))
    st = pure.fit()
    assert st["ev"] < 0.05 and not pure.ready() and st["shrink"] < 0.2
    # the warm-up: a good Q with fewer groups than it is not used
    few = kq.OutcomeQ(K, CARD, warmup=10_000, steps=300, seed=1)
    few.hold(X, lp, ready)
    few.join(out)
    few.fit()
    assert few.last["ev"] > 0.5 and not few.ready()
    # the state goes on; another width is dropped
    again = kq.OutcomeQ(K, CARD, warmup=2000)
    assert again.load(q.state()) and len(again) == len(q)
    torch.testing.assert_close(again.predict(X[:5, 0]), q.predict(X[:5, 0]), rtol=0, atol=1e-6)
    assert not kq.OutcomeQ(K, CARD + 1).load(q.state())


def test_the_outcomes_join_their_plans_by_id_and_old_features_stop_waiting():
    q = kq.OutcomeQ(K, CARD, warmup=10)
    X, lp, ready, out, _ = synthetic(10)
    ready = dict(ready, active=torch.tensor([[True, False]] * 10))
    X2 = torch.cat([X, torch.ones_like(X) * 7.0], 1)                 # a second place, not a group
    q.hold(X2, torch.cat([lp, lp], 1), ready)
    some = {k: v[[8, 2, 5]] for k, v in out.items()}                 # three known, in any order
    n, _, _ = q.join(some)
    assert n == 3 and len(q) == 3 and sorted(q.pid.tolist()) == [2, 5, 8]
    torch.testing.assert_close(q.y, out["y"][q.pid])
    torch.testing.assert_close(q.X, X[q.pid, 0])                       # each plan's own group only
    assert sorted(q.wait["qid"].tolist()) == [0, 1, 3, 4, 6, 7, 9]
    assert q.join({k: v[:0] for k, v in out.items()})[0] == 0          # nothing known: they wait
    for _ in range(kq.WAIT_CHUNKS):
        q.join(None)
    assert len(q.wait["qid"]) == 0 and q.lost == 7                     # too old: dropped, counted lost
    assert q.join(some)[0] == 0                                        # (an outcome without its plan: nothing)


# --- the advantage into the kind term, the agreement --------------------------------------------------------
def test_the_kind_parts_error_is_by_battle_and_the_wait_follows_the_horizon():
    g = torch.Generator().manual_seed(5)
    n = 2000
    v, a = torch.randn(n, generator=g), torch.randn(n, generator=g)
    y = 0.5 * v + 0.2 * a + torch.randn(n, generator=g)
    c, se = kq.kind_coef(y, v, a)
    assert abs(c - 0.2) < 4 * se and 0.01 < se < 0.05
    # every row twice in its own battle: the plain error would fall by sqrt 2, by battle it stays
    yy, vv, aa = (torch.cat([x, x]) for x in (y, v, a))
    c2, se2 = kq.kind_coef(yy, vv, aa, torch.cat([torch.arange(n), torch.arange(n)]))
    _, se_naive = kq.kind_coef(yy, vv, aa)
    assert c2 == pytest.approx(c, abs=1e-9) and se2 == pytest.approx(se, rel=0.05)
    assert se_naive == pytest.approx(se / math.sqrt(2), rel=0.05)
    assert kq.kind_coef(y[:5], v[:5], a[:5], torch.zeros(5, dtype=torch.long))[1] == float("inf")   # < 3 battles
    assert kq.wait_chunks(60, 1.0, 64) == 3 and kq.wait_chunks(60, 1.0, 16) == 6 and kq.wait_chunks(30, 1.0, 64) == 3


def test_the_qs_estimate_goes_into_the_kind_term_and_agrees_with_the_branches_log():
    env, actor, crit, batch, _, _ = windowed(2.0, branches=True)
    g = batch["general"]
    gen = actor.general
    Gp, Kk = gen.cfg.places, gen.cfg.kinds
    q = kq.OutcomeQ(Kk, gen.cfg.card, warmup=16)
    q.attach(batch, actor)
    st = q.stats
    for k in ("matured", "waiting", "samples", "on", "seconds", "ended_share", "lost", "orphans"):
        assert k in st, k
    assert "kind_q" not in g                                         # (too few groups to be ready)
    assert q.ready() is False
    # a fitted Q (its buffer filled with random groups) and ready: the estimate and the agreement
    W = kq.width(gen.cfg.card, Kk)
    gq = torch.Generator().manual_seed(4)
    q._add(torch.randn(400, W, generator=gq), torch.randint(0, Kk, (400,), generator=gq),
           torch.randn(400, generator=gq) * 0.01, torch.arange(400), torch.log(torch.full((400, Kk), 1 / Kk)),
           torch.arange(400) // 2)
    q.fit()
    q.ready = lambda: True
    q.use_shrink = False                                             # (the shrink 1: the advantage is visible)
    q.attach(batch, actor)
    assert float(q.model.shrink) == 1.0
    kqd = g["kind_q"]
    M = len(g["ready"]["kind"])
    assert kqd["q"].shape == (M, Gp, Kk) and kqd["scale"].shape == (Kk,) and bool((kqd["scale"] >= kq.FLOOR).all())
    assert "agree" in q.stats and q.stats["agree_n"] == len(g["branch"]["kind"])
    # the advantage the kind term takes: shrink x (Q(chosen) - E_pi0 Q) / its kind's scale
    X, lp = kq.features(gen, g["ready"])
    raw = q.predict(X, shrunk=False)
    want = kq.advantage(raw, g["ready"]["kind"], lp) / kqd["scale"][g["ready"]["kind"]]
    got = learn.critic_advantage(kqd, g["ready"]["kind"], lp, g["ready"]["active"][:, :Gp])
    mine = g["ready"]["active"][:, :Gp]
    torch.testing.assert_close(got[mine], want.clamp(-learn.ADV_CLIP, learn.ADV_CLIP)[mine], rtol=1e-4, atol=1e-5)
    # "outcome": the branches train nothing; "both": they do, beside the Q's groups
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    base = learn.LearnConfig(epochs=1, branches=1.0, minibatch=8, kind_own=True, kind_epochs=1)
    torch.manual_seed(11)
    s1 = learn.update(actor, opt, g["ready"], dataclasses.replace(base, kind_q="outcome"), branch=g["branch"],
                      kind_q=kqd)
    assert "general_branch_loss" not in s1 and "general_kind_critic_groups" in s1
    torch.manual_seed(11)
    s2 = learn.update(actor, opt, g["ready"], dataclasses.replace(base, kind_q="both"), branch=g["branch"],
                      kind_q=kqd)
    assert "general_branch_loss" in s2 and "general_kind_loss" in s2


def test_both_is_the_qs_mean_plus_the_branches_weight_times_theirs():
    env, actor, crit, batch, _, _ = windowed(2.0, branches=True)
    g = batch["general"]
    gen = actor.general
    ready, br = g["ready"], g["branch"]
    Gp, Kk = gen.cfg.places, gen.cfg.kinds
    torch.manual_seed(3)
    kqd = {"q": torch.randn(len(ready["kind"]), Gp, Kk) * 0.01, "scale": torch.full((Kk,), 0.01)}
    with torch.no_grad():
        d0 = gen(ready, act={k: ready[k] for k in learn.CHOICES})[1]
        adv_c = learn.critic_advantage(kqd, ready["kind"], d0["kind_logp"], ready["active"][:, :Gp])
        mine = ready["active"][:, :Gp]
        w = torch.ones(len(ready["kind"]), Gp)
        args = (d0["lp_kind"], w, mine)
        cfg = learn.LearnConfig(minibatch=10 ** 6, kind_own=True, branch_weight=0.3)
        _, lc = learn._kind_loss(gen, None, d0, adv_c, *args, dataclasses.replace(cfg, kind_q="both"), None)
        _, lb = learn._kind_loss(gen, br, None, None, *args, dataclasses.replace(cfg, kind_q="both"), None)
        _, both = learn._kind_loss(gen, br, d0, adv_c, *args, dataclasses.replace(cfg, kind_q="both"), None)
        _, old = learn._kind_loss(gen, br, d0, adv_c, *args, dataclasses.replace(cfg, kind_q="branch"), None)
        _, lb1 = learn._kind_loss(gen, br, None, None, *args, dataclasses.replace(cfg, kind_q="branch"), None)
    assert float(lb) == pytest.approx(0.3 * float(lb1), rel=1e-5)          # the branches alone: x their weight
    assert float(both) == pytest.approx(float(lc) + 0.3 * float(lb1), rel=1e-5)
    n_b, n_c = len(br["adv"]), float(mine.sum())
    assert float(old) == pytest.approx((float(lc) * n_c + float(lb1) * n_b) / (n_c + n_b), rel=1e-5)   # (before)
    with pytest.raises(ValueError):
        learn.LearnConfig(kind_q="critic")


# --- the branches: centring and "same" ----------------------------------------------------------------------
def test_the_branches_prices_are_centred_by_their_running_mean_and_the_raw_ones_logged():
    env, actor, crit, batch = collected(True)
    br, st = batch["general"]["branch"], env.general_rt.brancher.stats
    raw = br["price_raw"]
    assert "center" in st and st["center"] == pytest.approx(float(raw.mean()), abs=1e-6)   # the first update: its own
    torch.testing.assert_close(br["price"], raw - float(raw.mean()), rtol=0, atol=1e-6)
    for k in ("price_raw_mean", "better_raw", "worse_raw", "vs_teacher_price_raw"):
        assert k in st, k
    assert br["alt"].shape == (len(raw), 2)
    gb = env.general_rt.brancher
    assert "center" in gb.state()
    other = gbranch.Brancher(gb.cfg, "cpu")
    assert "center" in other.load(gb.state()) and other.center.mean() == pytest.approx(gb.center.mean())
    # off: as measured
    off = gbranch.Brancher(dataclasses.replace(gb.cfg, center=False), "cpu")
    assert off.center is None and "center" not in off.state()


def test_same_reprices_the_chosen_kind_with_another_seed_and_trains_nothing():
    env, actor, snap, cand = grabbed()
    b, side, group, kind = columns(cand)
    sub = ab.select(snap, b)
    with torch.no_grad():
        actor.task_in.weight.normal_(0, 0.5)
    same = kind[:, None].repeat(1, 2)
    kw = dict(seed=3, price="reward", openloop=True)
    d0, _ = gbranch.play_kinds(env, actor, sub, side, group, same, 2, 5, **kw)
    assert torch.count_nonzero(d0) == 0                            # the same seed: exactly 0
    d1, _ = gbranch.play_kinds(env, actor, sub, side, group, same, 2, 5, reseed=True, **kw)
    d2, _ = gbranch.play_kinds(env, actor, sub, side, group, same, 2, 5, reseed=True, **kw)
    assert torch.count_nonzero(d1) > 0 and torch.equal(d1, d2)      # another seed: the group plays differently
    # the brancher in "same": every copy's other kind is the chosen one, the prices logged, no rows, no critic's
    torch.manual_seed(7)
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    torch.manual_seed(8)
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.9997
    rt.brancher = gbranch.Brancher(gbranch.Config(share=1.0, copies=2, horizon_s=3.0, cap=0, batch=4, alt="same"),
                                   "cpu")
    rt.r["t_plan"] -= 31.0
    seen = []
    capture = rt.brancher.capture

    def capture_(e, info):
        capture(e, info)
        c = rt.brancher.chunk
        if c is not None and c.entries:
            seen.append(c.entries[-1])
    rt.brancher.capture = capture_
    torch.manual_seed(9)
    out = rollout.collect(env, actor, crit, 3)
    st = rt.brancher.stats
    assert "branch" not in out["general"] and st.get("same") == 1 and "price_raw_mean" in st
    assert seen and all(torch.equal(e["alt"], e["kind"][:, None].expand_as(e["alt"])) for e in seen)
    assert rt.brancher.critic is None or len(rt.brancher.critic) == 0
    assert "alt same" in gbranch.text(st)


# --- the flags ----------------------------------------------------------------------------------------------
def test_the_flags_defaults_outcome_the_horizon_and_the_centre():
    ap = run.parser()
    a = ap.parse_args(["--general", "--general-branches", "0.2"])
    assert a.general_kind_q == "outcome" and a.general_kind_q_horizon == 60.0 and a.general_branch_center == 1
    assert a.general_kind_q_warmup == 10000 and a.general_kind_q_buffer == 30000 and a.general_kind_q_shrink == 1
    assert a.general_kind_q_ev_min == 0.05
    cfg = learn.LearnConfig(branches=0.2, kind_q=a.general_kind_q)
    q = run.general_outcome_q(a, cfg, GCFG, "cpu")
    assert isinstance(q, kq.OutcomeQ) and q.warmup == 10000 and q.cap == 30000 and q.norm is not None
    assert q.wait_n == kq.wait_chunks(60.0, 1.0, a.steps)
    br = run.general_brancher(a, cfg, "cpu")
    assert br.cfg.center and not br.cfg.critic                         # the kind critic learns only under branch
    assert not run.general_brancher(a, dataclasses.replace(cfg, kind_q="both"), "cpu").cfg.critic
    assert run.general_brancher(a, dataclasses.replace(cfg, kind_q="branch"), "cpu").cfg.critic
    assert run.general_outcome_q(a, dataclasses.replace(cfg, kind_q="branch"), GCFG, "cpu") is None
    assert run.general_outcome_q(a, None, GCFG, "cpu") is None
    s = ap.parse_args(["--general", "--general-branches", "0.2", "--general-branch-alt", "same",
                       "--general-branch-center", "0", "--general-kind-q", "outcome"])
    br = run.general_brancher(s, cfg, "cpu")
    assert br.cfg.alt == "same" and not br.cfg.center and s.general_kind_q == "outcome"
    assert "kind Q" in kq.text({"samples": 3, "ev": 0.1, "agree": 0.2})
    assert math.isnan(kq.pearson(torch.zeros(2), torch.zeros(2)))


def test_a_run_logs_the_q_and_the_centre_and_saves_and_resumes_them(tmp_path, monkeypatch):
    import json
    from tools.nn.model import config, critic as mcritic, policy
    from tools.nn.train import checkpoint
    from tests.tools.test_general_branch_torch import TINY
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    torch.manual_seed(0)
    init = checkpoint.save(tmp_path / "init.pt", policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval(), "v2")
    common = ["--reward", "v2", "--battles", "4", "--steps", "4", "--minutes", "5", "--device",
              "cpu", "--no-eval", "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
              "--keep-every", "1e-6", "--general-card", "16", "--general-hidden", "16", "--general-heads", "2",
              "--general-return", "critic", "--general-branches", "1.0", "--general-branch-copies", "1",
              "--general-branch-horizon", "2", "--general-branch-batch", "4", "--general-branch-cap", "0",
              "--general-kind-q-horizon", "2", "--general-kind-q-warmup", "16", "--general-replan-s", "2"]
    args = run.parser().parse_args(["--name", "kq", "--init", str(init), "--general", "--updates", "2"] + common)
    _, _, out = run.train(args)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all("kind_q_samples" in r["general"] and "kind_q_on" in r["general"] for r in rows)
    assert any(r["general"].get("kind_q_matured", 0) > 0 for r in rows)
    assert any("branch_center" in r["general"] for r in rows)
    ts = checkpoint.train_state(out / "latest.pt")
    assert "general_kind_q" in ts and ts["general_kind"]["centered"] is True
    assert "critic" not in ts["general_kind"]                       # (outcome, the default: no kind critic)
    assert all("kind_q_lost" in r["general"] for r in rows)
    n = len(ts["general_kind_q"]["a"])
    assert n > 0 and n == rows[-1]["general"]["kind_q_samples"]
    # resumed (the default): the Q's buffer goes on
    again = run.parser().parse_args(["--name", "kq2", "--init", str(out / "latest.pt"), "--general", "--updates", "1"]
                                    + common)
    _, _, out2 = run.train(again)
    rows2 = [json.loads(x) for x in (out2 / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows2[0]["general"]["kind_q_samples"] >= n
