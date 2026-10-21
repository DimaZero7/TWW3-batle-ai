"""The mission branches (tools/nn/general/branch.py; run.py --general-branches P): a plan replayed with one group's
mission kind changed - the swap touches only that group's kind card, a swap to the same kind prices exactly 0, the
branches leave the batch and the battles bit for bit, P = 0 builds nothing, the term's old log-probability is the
plan's (docs/en/training/general.md "Mission branches"). torch: the training container, on the CPU."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import branch as gbranch                 # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import executor, learn                   # noqa: E402
from tools.nn.model import config, critic as mcritic, policy   # noqa: E402
from tools.nn.train import ability_branch as ab                # noqa: E402
from tools.nn.train import league, ppo, randomise, rollout, run, scenes  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, replan_min_s=1e9)
MIRROR = [("arena", "attack")]


def nets(seed=0):
    torch.manual_seed(seed)
    return executor.wrap(policy.Actor(TINY).eval(), GCFG).eval(), mcritic.Critic(TINY).eval()


def battles(B=4, mix=None, seed=1):
    params = rollout.params_with_limit(600.0)
    src = scenes.Generated(range(500, 500 + 2 * B), 5, params, "cpu", seed=seed)
    lay = league.layout(B, 1, mix or {"self": 1.0}, scene_attacker=[1])
    return rollout.Battles(lay, MIRROR, params=params, source=src, spread=randomise.NONE, compile=False)


class Grab:
    """An ability-branch hook that snapshots every battle at the first decision it sees."""

    def __init__(self):
        self.got = None

    def capture(self, env, info):
        if self.got is None:
            self.got = ab.snapshot(env, torch.arange(env.B), info)


def grabbed():
    """(env, actor, snapshot of every battle at the deployment's decision, [(battle, side, group, kind)] - one own
    group of every learner row with a plan)."""
    env = battles()
    actor, crit = nets()
    env.brancher = Grab()
    env.step(actor, crit)
    snap = env.brancher.got
    env.brancher = None
    rt = env.general_rt
    G = GCFG.places
    cand = []
    for row in range(2 * env.B):
        act = rt.r["active"][row, :G].nonzero().squeeze(1).tolist()
        if bool(rt.r["has"][row]) and act:
            cand.append((row % env.B, row // env.B + 1, act[0], int(rt.r["kind"][row, act[0]])))
    assert cand, "the deployment's plans have own groups"
    return env, actor, snap, cand


def columns(cand):
    return (torch.tensor([c[i] for c in cand]) for i in range(4))


@pytest.mark.parametrize("price", ["trade", "reward"])
def test_a_swap_to_the_same_kind_prices_exactly_zero_and_the_branches_repeat(price):
    env, actor, snap, cand = grabbed()
    b, side, group, kind = columns(cand)
    sub = ab.select(snap, b)
    with torch.no_grad():
        actor.task_in.weight.normal_(0, 0.5)                          # the executor listens to its task
    same = kind[:, None].repeat(1, 2)
    d0, _ = gbranch.play_kinds(env, actor, sub, side, group, same, 2, 5, seed=3, price=price)
    assert torch.count_nonzero(d0) == 0
    other = (kind[:, None].repeat(1, 2) + torch.tensor([1, 3])) % GCFG.kinds
    s = torch.get_rng_state()
    d1, bg = gbranch.play_kinds(env, actor, sub, side, group, other, 2, 5, seed=3, price=price)
    assert torch.equal(torch.get_rng_state(), s)                     # the global random state is the one before
    d2, _ = gbranch.play_kinds(env, actor, sub, side, group, other, 2, 5, seed=3, price=price)
    assert torch.equal(d1, d2) and d1.shape == (len(cand), 2) and bool((bg > 0).all())
    # an executor that does not listen to its task (the new input at zero): every swap prices exactly 0
    with torch.no_grad():
        actor.task_in.weight.zero_()
    d3, _ = gbranch.play_kinds(env, actor, sub, side, group, other, 2, 5, seed=3, price=price)
    assert torch.count_nonzero(d3) == 0


def test_the_swap_changes_only_the_groups_units_kind_card():
    env, actor, snap, cand = grabbed()
    b, side, group, kind = columns(cand)
    benv, _, _ = ab.branch_env(env, ab.select(snap, b))
    rt = benv.general_rt
    n = len(cand)
    rows = (side - 1) * n + torch.arange(n)
    before = rt.r["kc"].clone()
    alt = (kind + 1) % GCFG.kinds
    alt[0] = -1                                                       # none: nothing changes
    gbranch.swap_kind(benv, actor, side, group, alt)
    after = rt.r["kc"]
    hit = torch.zeros(2 * n, benv.N, dtype=torch.bool)
    hit[rows] = rt.r["main"][rows] == group[:, None]
    hit[rows[0]] = False
    assert bool(hit.any())
    assert torch.equal(after[~hit], before[~hit])
    j, u = hit.nonzero(as_tuple=True)
    want = actor.general.kinds.detach()[alt[(j % n)]]
    torch.testing.assert_close(after[j, u], want, rtol=0, atol=0)
    # the chosen kind's card is the kinds table's row (a swap to it changes no bit)
    k_now = actor.general.kinds.detach()[kind[(j % n)]]
    torch.testing.assert_close(before[j, u], k_now, rtol=0, atol=0)


def test_the_other_kinds_are_never_the_chosen_one_and_the_policys_weight_is_one_less_its_probability():
    br = gbranch.Brancher(gbranch.Config(share=1.0, copies=64), "cpu")
    logp = torch.log_softmax(torch.randn(5, 10), -1)
    kind = torch.tensor([0, 3, 9, 3, 5])
    alt, w = br.alternatives(logp, kind)
    assert alt.shape == (5, 64) and not bool((alt == kind[:, None]).any()) and torch.all(w == 1)
    assert set(alt.reshape(-1).tolist()) == set(range(10))
    br = gbranch.Brancher(gbranch.Config(share=1.0, copies=64, alt="policy"), "cpu")
    sure = logp.clone()
    sure[1] = torch.log_softmax(torch.tensor([0.0] * 3 + [50.0] + [0.0] * 6), -1)   # pi(chosen) ~ 1
    alt, w = br.alternatives(sure, kind)
    assert not bool((alt == kind[:, None]).any())
    torch.testing.assert_close(w, 1 - sure.exp().gather(1, kind[:, None])[:, 0])
    assert float(w[1]) < 1e-6
    with pytest.raises(ValueError):
        gbranch.Config(share=0.0)
    with pytest.raises(ValueError):
        gbranch.Config(alt="best")


def collected(branches, horizon=3.0):
    """A chunk of 3 decisions (every row re-plans at the first); branches: the mission branches on (all the new plans
    replayed) or not. -> (env, actor, critic, batch)."""
    torch.manual_seed(7)
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    torch.manual_seed(8)
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.9997
    if branches:
        rt.brancher = gbranch.Brancher(gbranch.Config(share=1.0, copies=2, horizon_s=horizon, cap=0, batch=4), "cpu")
    rt.r["t_plan"] -= 31.0
    torch.manual_seed(9)
    return env, actor, crit, rollout.collect(env, actor, crit, 3)


def updated(actor, crit, batch, branches, weight):
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    torch.manual_seed(10)
    return ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), minibatch=8),
                      general=learn.LearnConfig(epochs=1, branches=1.0 if branches else 0.0, branch_weight=weight))


def same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            same(a[k], b[k], f"{path}/{k}")
    elif torch.is_tensor(a):
        assert torch.equal(a, b), path
    elif hasattr(a, "_fields") or dataclasses.is_dataclass(a):
        for f in (a._fields if hasattr(a, "_fields") else [x.name for x in dataclasses.fields(a)]):
            same(getattr(a, f), getattr(b, f), f"{path}.{f}")


def test_the_branches_leave_the_batch_the_battles_and_the_update_bit_for_bit():
    env0, actor0, crit0, b0 = collected(False)
    env1, actor1, crit1, b1 = collected(True)
    br = b1["general"]["branch"]
    assert "branch" not in b0["general"]
    for k in ("obs", "critic_obs", "lp", "value", "reward", "done", "attacks", "h0", "last_value"):
        same(b0[k], b1[k], k)
    same(b0["general"]["table"], b1["general"]["table"], "table")
    same(b0["general"]["ready"], b1["general"]["ready"], "ready")
    for f in b0["action"].parts():
        assert torch.equal(getattr(b0["action"], f), getattr(b1["action"], f)), f
    for k in env0.st.u:
        assert torch.equal(env0.st.u[k], env1.st.u[k]), k                 # the main battles were not touched
    # the term at weight 0: the same weights after the update
    updated(actor0, crit0, b0, False, 0.0)
    updated(actor1, crit1, b1, True, 0.0)
    for (n, p), q in zip(actor0.named_parameters(), actor1.parameters()):
        assert torch.equal(p, q), n
    # what the branches gave
    E = len(br["adv"])
    st = env1.general_rt.brancher.stats
    assert E > 0 and st["branches"] == E and E <= st["plans"]
    assert br["rec"]["kind"].shape[0] == E and torch.isfinite(br["adv"]).all() and torch.isfinite(br["price"]).all()
    for k in ("price_mean", "price_std", "price_se", "better", "worse", "battles", "seconds"):
        assert k in st, k
    assert not bool(env1.general_rt.r["fresh"].any())                 # the marks are cleared every decision


def test_the_terms_old_log_probability_is_the_plans_and_its_gradient_reaches_the_general():
    env, actor, crit, batch = collected(True)
    br = batch["general"]["branch"]
    gen = actor.general
    actor.train()
    _, kl, _ = learn.branch_term(gen, br, 0.2)
    assert abs(kl) < 1e-5                                             # the ratio starts at 1
    br = dict(br, adv=torch.randn(len(br["adv"])))
    gen.zero_grad(set_to_none=True)
    loss, _, _ = learn.branch_term(gen, br, 0.2)
    loss.backward()
    for name in ("kinds", "kind_q.weight", "reader.", "council.", "groups."):
        grads = [p.grad for n, p in gen.named_parameters() if n.startswith(name)]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads), name
    for name in ("target_q.", "place_q.", "value."):                      # the other choices' heads: no gradient
        grads = [p.grad for n, p in gen.named_parameters() if n.startswith(name)]
        assert all(g is None or float(g.abs().sum()) == 0 for g in grads), name
    # the term on: the general's update logs it
    gen.zero_grad(set_to_none=True)
    actor.eval()
    st = updated(actor, crit, batch, True, 1.0)
    for k in ("general_branch_loss", "general_branch_kl", "general_branch_rows"):
        assert k in st and st[k] == st[k], k


def test_no_branches_at_zero_and_the_flags_reach_the_config():
    ap = run.parser()
    a0 = ap.parse_args(["--general"])
    assert a0.general_branches == 0.0
    assert run.general_brancher(a0, learn.LearnConfig(), "cpu") is None
    a1 = ap.parse_args(["--general", "--general-branches", "0.2", "--general-branch-copies", "2",
                        "--general-branch-horizon", "30", "--general-branch-alt", "policy", "--general-branch-greedy"])
    cfg = learn.LearnConfig(branches=a1.general_branches, branch_weight=a1.general_branch_weight)
    br = run.general_brancher(a1, cfg, "cpu")
    assert br.cfg.share == 0.2 and br.cfg.copies == 2 and br.cfg.horizon_s == 30.0 and br.cfg.alt == "policy"
    assert br.cfg.greedy and br.cfg.price == "trade"
    with pytest.raises(ValueError):
        learn.LearnConfig(branches=1.5)
    with pytest.raises(ValueError):
        learn.LearnConfig(branch_weight=-1.0)
