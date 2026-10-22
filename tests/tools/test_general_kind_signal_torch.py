"""The general's own signal for the mission kind (docs/en/training/general.md "The kind's own signal"): the branches'
partial open loop (only the swapped group re-decides), the kind critic (kind_critic.py), the kind head out of the plan's
PPO with its own term, lr and extra passes, the flags' defaults. torch: the training container, on the CPU."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import branch as gbranch                 # noqa: E402
from tools.nn.general import kind_critic as kc                  # noqa: E402
from tools.nn.general import learn                              # noqa: E402
from tools.nn.sim import orders as O                            # noqa: E402
from tools.nn.train import run                                  # noqa: E402

from tests.tools.test_general_branch_torch import collected, grabbed, columns   # noqa: E402
from tools.nn.train import ability_branch as ab                 # noqa: E402


def orders(R, N, v):
    f = (lambda dt: torch.full((R, N), v, dtype=dt))
    return O.Orders(*(f(torch.long if O.FIELDS[k][0] == "i" else torch.float32) for k in O.FIELDS))


class Fake:
    """An env whose _play records what it was given (the partial open loop's merge, no simulator)."""

    def __init__(self, R, N):
        self.rows_learn = torch.arange(R)
        self.rows_past = torch.arange(R, R + 1)
        self.got = None

    def _play(self, parts, scripted, delay, *a, **k):
        self.got = (parts, scripted, delay)
        return "played"


def test_the_open_loops_other_copy_plays_the_chosen_orders_but_for_the_groups_units():
    R, N = 3, 5
    env = Fake(R, N)
    mine = torch.zeros(R, N, dtype=torch.bool)
    mine[1, 2:4] = True                                              # the swapped group's units
    chosen = [(torch.arange(R), orders(R, N, 1)), (torch.arange(R, R + 1), orders(1, N, 7))]
    said = [(chosen, "chosen script", "chosen delay")]
    play = gbranch._openloop_play(env, said, mine)
    out = play([(env.rows_learn, orders(R, N, 2)), (env.rows_past, orders(1, N, 9))], "own script", "own delay")
    assert out == "played" and not said                              # this decision's record consumed
    parts, scripted, delay = env.got
    assert scripted == "chosen script" and delay == "chosen delay"   # the scripts and the landing: chosen's
    learn_o, past_o = parts[0][1], parts[1][1]
    for k in O.FIELDS:
        v = getattr(learn_o, k)
        assert bool((v[mine] == 2).all()) and bool((v[~mine] == 1).all()), k   # only the group's own orders
        assert bool((getattr(past_o, k) == 7).all()), k              # the enemy's (a past version): chosen's


def test_the_group_units_are_the_plans_members_on_the_learner_rows():
    env, actor, snap, cand = grabbed()
    b, side, group, kind = columns(cand)
    benv, _, _ = ab.branch_env(env, ab.select(snap, b))
    mine = gbranch.group_units(benv, side, group)
    n = len(cand)
    rt = benv.general_rt
    pos = torch.full((2 * n,), -1, dtype=torch.long)
    pos[benv.rows_learn] = torch.arange(len(benv.rows_learn))
    for j in range(n):
        p = int(pos[(side[j] - 1) * n + j])
        assert p >= 0
        want = rt.r["main"][(side[j] - 1) * n + j] == group[j]
        assert torch.equal(mine[p], want)
    assert int(mine.sum()) == sum(int((rt.r["main"][(side[j] - 1) * n + j] == group[j]).sum()) for j in range(n))
    none = gbranch.group_units(benv, side, torch.full_like(group, -1))
    assert not bool(none.any())


def synthetic(n, K=6, C=4, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, kc.WIDTH, generator=g)
    W = torch.randn(kc.WIDTH, K, generator=g) * 0.01
    q = X @ W
    a = torch.randint(0, K, (n,), generator=g)
    b = (a[:, None] + torch.randint(1, K, (n, C), generator=g)) % K
    d = kc.pair_diff(q, a, b) + noise * torch.randn(n, C, generator=g) * 0.01
    return X, a, b, d, W


def test_the_kind_critic_learns_a_known_price_and_shrinks_noise_to_zero():
    crit = kc.KindCritic(6, warmup=500, steps=400, seed=1)
    X, a, b, d, W = synthetic(3000)
    crit.add(X, a, b, d)
    st = crit.fit()
    assert st["samples"] == 3000 and crit.ready() and st["ev"] > 0.5 and 0.5 < st["shrink"] <= 1.0
    # its advantage is Q(chosen) - E_pi Q: 0 when every kind has the same Q, whatever the choice
    q = torch.ones(2, 3, 6)
    flat = kc.baseline_advantage(q, torch.zeros(2, 3, dtype=torch.long), torch.log_softmax(torch.randn(2, 3, 6), -1))
    assert float(flat.abs().max()) < 1e-6
    q = torch.zeros(1, 1, 6)
    q[0, 0, 2] = 1.0
    lp = torch.log(torch.tensor([[[0.5, 0.0, 0.5, 0.0, 0.0, 0.0]]]).clamp(min=1e-30))   # masked kinds: weight 0
    assert float(kc.baseline_advantage(q, torch.tensor([[2]]), lp)) == pytest.approx(0.5)
    # pure noise: the shrink falls towards 0 and it is not ready
    pure = kc.KindCritic(6, warmup=500, steps=200, seed=2)
    g = torch.Generator().manual_seed(5)
    pure.add(torch.randn(3000, kc.WIDTH, generator=g), a, b, torch.randn(3000, 4, generator=g) * 0.01)
    st = pure.fit()
    assert st["shrink"] < 0.5 and not pure.ready()
    # its state goes on
    again = kc.KindCritic(6, warmup=500)
    assert again.load(crit.state()) and len(again) == len(crit)
    torch.testing.assert_close(again.predict(X[:5]), crit.predict(X[:5]))
    assert not kc.KindCritic(7).load(crit.state())


def test_the_critics_features_are_our_groups_numbers_and_the_brancher_feeds_it():
    env, actor, crit, batch = collected(True)
    br = env.general_rt.brancher
    st = br.stats
    assert br.critic is not None and br.norm is not None               # the defaults: on
    assert len(br.critic) == st["branches"] and st["critic_samples"] == st["branches"]
    assert any(k.startswith("norm_rms_") for k in st)
    ready = batch["general"]["ready"]
    X = kc.features(actor.general, ready)
    G = actor.general.cfg.places
    assert X.shape == (len(ready["kind"]), G, kc.WIDTH) and bool(torch.isfinite(X).all())
    assert torch.count_nonzero(X[~ready["active"][:, :G]]) == 0
    assert br.kind_q(actor, ready) is None                            # not ready yet: no estimate
    br.attach(batch, actor)
    assert "kind_q" not in batch["general"]
    br.critic.ready = lambda: True                                    # (as if it were)
    br.attach(batch, actor)
    kq = batch["general"]["kind_q"]
    assert kq["q"].shape == (len(ready["kind"]), G, br.K) and kq["scale"].shape == (br.K,)
    assert bool((kq["scale"] > 0).all())
    # saved and loaded: the critic's buffer and the scales
    other = gbranch.Brancher(br.cfg, "cpu")
    assert set(other.load(br.state())) == {"critic", "norm"} and len(other.critic) == len(br.critic)


def forced(br, ready):
    """A kind critic estimate for the ended plans as if the critic were ready: Q random, the scales 0.01."""
    K = br.K
    torch.manual_seed(3)
    return {"q": torch.randn(len(ready["kind"]), ready["kind"].shape[1], K) * 0.01, "scale": torch.full((K,), 0.01)}


def general_update(actor, batch, cfg, kind_q=None, branch=True):
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    torch.manual_seed(11)
    return learn.update(actor, opt, batch["general"]["ready"], cfg,
                        branch=batch["general"]["branch"] if branch else None, kind_q=kind_q)


def snapshot(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def test_the_kind_head_out_of_the_plans_ppo_learns_only_from_its_own_term():
    env, actor, crit, batch = collected(True)
    base = learn.LearnConfig(epochs=1, branches=1.0, entropy_kind=0.0, minibatch=8)
    w0 = snapshot(actor.general)
    # without branches and the critic, under the own signal, the kind's query gets nothing from the plan's advantage
    general_update(actor, batch, dataclasses.replace(base, kind_own=True), branch=False)
    after = snapshot(actor.general)
    assert torch.equal(after["kind_q.weight"], w0["kind_q.weight"])
    assert not torch.equal(after["target_q.1.weight"], w0["target_q.1.weight"])     # the target still learns
    with torch.no_grad():
        for n, p in actor.general.named_parameters():
            p.copy_(w0[n])
    general_update(actor, batch, base, branch=False)                                  # the old way: it does
    assert not torch.equal(snapshot(actor.general)["kind_q.weight"], w0["kind_q.weight"])


def test_the_critics_groups_join_the_kind_term_and_the_extra_pass_moves_the_kind_head_alone():
    env, actor, crit, batch = collected(True)
    ready = batch["general"]["ready"]
    br = env.general_rt.brancher
    kq = forced(br, ready)
    w0 = snapshot(actor.general)
    cfg = learn.LearnConfig(epochs=1, branches=1.0, minibatch=8, kind_own=True, kind_epochs=1)
    st1 = general_update(actor, batch, cfg, kq)
    one = snapshot(actor.general)
    for k in ("general_kind_loss", "general_kind_kl", "general_kind_critic_groups", "general_kind_critic_adv_std",
              "general_branch_loss", "general_branch_kl"):
        assert k in st1 and st1[k] == st1[k], k
    assert st1["general_branch_kl"] >= 0.0 and st1["general_kind_kl"] >= 0.0          # k3
    assert st1["general_kind_critic_groups"] == float(ready["active"][:, :actor.general.cfg.places].sum())
    assert "general_kind_extra_steps" not in st1
    with torch.no_grad():
        for n, p in actor.general.named_parameters():
            p.copy_(w0[n])
    st2 = general_update(actor, batch, dataclasses.replace(cfg, kind_epochs=2), kq)
    two = snapshot(actor.general)
    assert st2["general_kind_extra_steps"] >= 1
    head = {n for n, _ in actor.general.named_parameters() if n.startswith("kind_q.") or n == "kinds"}
    for n in one:
        if n in head:
            continue
        assert torch.equal(one[n], two[n]), n                       # the extra pass: the kind head alone
    assert any(not torch.equal(one[n], two[n]) for n in head)


def test_the_critics_advantage_is_q_less_the_policys_mean_over_the_kinds_scale():
    q = torch.zeros(1, 2, 3)
    q[0, 0] = torch.tensor([0.02, 0.0, -0.02])
    kind = torch.tensor([[0, 1]])
    logp = torch.log(torch.full((1, 2, 3), 1 / 3))
    active = torch.tensor([[True, False]])
    a = learn.critic_advantage({"q": q, "scale": torch.tensor([0.01, 0.02, 0.01])}, kind, logp, active)
    assert float(a[0, 0]) == pytest.approx(2.0) and float(a[0, 1]) == 0.0


def test_the_flags_defaults_the_own_signal_and_the_kind_lr():
    ap = run.parser()
    a = ap.parse_args(["--general", "--general-branches", "0.2"])
    assert a.general_branch_cap == 128 and a.general_branch_openloop == 1 and a.general_kind_critic == 1
    assert a.general_kind_own_signal == 1 and a.general_kind_lr_mult == 2.0 and a.general_kind_epochs == 2
    assert a.general_kind_norm == "per-kind"
    cfg = learn.LearnConfig(branches=0.2)
    br = run.general_brancher(a, cfg, "cpu")
    assert br.cfg.openloop and br.cfg.critic and br.cfg.norm == "per-kind" and br.cfg.cap == 128
    off = run.general_brancher(ap.parse_args(["--general", "--general-branches", "0.2", "--general-branch-openloop",
                                              "0", "--general-kind-critic", "0", "--general-kind-norm", "chunk"]),
                               cfg, "cpu")
    assert not off.cfg.openloop and not off.cfg.critic and off.cfg.norm == "chunk"
    with pytest.raises(ValueError):
        gbranch.Config(norm="kind")
    with pytest.raises(ValueError):
        learn.LearnConfig(kind_epochs=0)
    # the kind head's own Adam group: lr x general x kind
    from tests.tools.test_general_branch_torch import nets
    actor, critic = nets()
    opt = run.optimizer(actor, critic, 1e-3, 1.0, 1.0, 0.5, 2.0)
    head = {id(p) for p in learn.kind_head_params(actor.general)}
    assert len(head) == 3                                            # kind_q weight and bias, kinds
    for g in opt.param_groups:
        for p in g["params"]:
            if id(p) in head:
                assert g["lr"] == pytest.approx(1e-3)                   # 1e-3 x 0.5 x 2
    assert {round(g["lr"], 8) for g in opt.param_groups} >= {5e-4, 1e-3}
