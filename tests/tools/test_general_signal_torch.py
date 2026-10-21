"""The general's learning signals in the battles (tools/nn/general/runtime.py, distinct.py, learn.py): the plan's target
and base by the unit critic (--general-return critic) at a re-plan, at the battle's end and for the deployment's plan;
the groups' behaviour and the distinct bonus in the reward (--general-distinct); the update's new stats
(docs/en/training/general.md "Training"). torch: the training container, on the CPU."""
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import distinct, executor, learn, runtime  # noqa: E402
from tools.nn.model import config, critic as mcritic, policy   # noqa: E402
from tools.nn.train import league, ppo, randomise, rollout, scenes  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
# no event re-plans (a group broken, a lord fallen): only the ones a test forces
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, replan_min_s=1e9)
MIRROR = [("arena", "attack")]
F = len(runtime.BEH)


def nets(seed=0):
    torch.manual_seed(seed)
    return executor.wrap(policy.Actor(TINY).eval(), GCFG).eval(), mcritic.Critic(TINY).eval()


def battles(B=4, mix=None, seed=1):
    params = rollout.params_with_limit(600.0)
    src = scenes.Generated(range(500, 500 + 2 * B), 5, params, "cpu", seed=seed)
    lay = league.layout(B, 1, mix or {"self": 1.0}, scene_attacker=[1])
    return rollout.Battles(lay, MIRROR, params=params, source=src, spread=randomise.NONE, compile=False)


def known_values(env):
    """The critic's value replaced by a known one: decision k (1, 2, ...) gives row i 0.1 k + 0.01 i."""
    seen = {"k": 0}

    def fake(critic, c):
        seen["k"] += 1
        return 0.1 * seen["k"] + 0.01 * torch.arange(env.R, dtype=torch.float32)
    env._values = fake
    return lambda k, i: 0.1 * k + 0.01 * i


def test_a_replanned_plans_target_is_the_reward_and_the_critic_at_the_replan():
    env = battles()
    actor, crit = nets()
    V = known_values(env)
    s1 = env.step(actor, crit)                                        # decision 1: the deployment's plans take V(1)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.9
    assert torch.equal(env.rows_learn, torch.arange(2 * env.B))       # (self battles: row i is learner row i)
    assert float(rt.r["v_start"][0]) == pytest.approx(V(1, 0))
    rt.begin_chunk(env)
    s2 = env.step(actor, crit)                                        # decision 2: nothing due
    assert not rt.ready
    rt.r["t_plan"][0] -= 31.0
    env.step(actor, crit)                                             # decision 3: row 0 re-plans
    assert len(rt.ready) == 1
    rec = rt.ready[0]
    r0, r1 = float(s1["reward"][0]), float(s2["reward"][0])
    assert float(rec["ret"][0]) == pytest.approx(r0 + 0.9 * r1 + 0.81 * V(3, 0), abs=1e-5)
    assert float(rec["base"][0]) == pytest.approx(V(1, 0), abs=1e-6)
    assert float(rec["done"][0]) == 0.0
    assert float(rt.r["v_start"][0]) == pytest.approx(V(3, 0))          # the new plan begins at the re-plan's value


def test_at_the_battles_end_the_critic_counts_zero_and_the_deployment_takes_the_first_decisions_value():
    env = battles()
    actor, crit = nets()
    V = known_values(env)
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.95
    rt.begin_chunk(env)
    env.step(actor, crit)
    finished = torch.zeros(env.B, dtype=torch.bool)
    finished[0] = True
    rows = [0, env.B]                                                 # both sides of battle 0
    rd, n = rt.r["r_disc"][rows].clone(), rt.r["n_dec"][rows].clone().float()
    base = rt.r["v_start"][rows].clone()
    rt.after(env, torch.full((env.R,), 0.5), finished)                # the last decision's reward, the battle ends
    rec = rt.ready[-1]
    want = rd + 0.95 ** n * 0.5
    torch.testing.assert_close(rec["ret"], want, atol=1e-6, rtol=0)  # no V(end): the end's reward only
    torch.testing.assert_close(rec["base"], base)
    assert torch.all(rec["done"] == 1.0) and not bool(rt.r["has"][rows].any())
    # the next battle's deployment: no value until its first decision, then that decision's
    env._reset(finished)
    env.cur = None
    assert bool(torch.isnan(rt.r["v_start"][rows]).all())
    env.step(actor, crit)                                             # decision 3
    assert [float(v) for v in rt.r["v_start"][rows]] == pytest.approx([V(3, 0), V(3, env.B)])


def test_the_classifier_starts_at_chance_and_learns_separable_behaviour():
    torch.manual_seed(0)
    K = 10
    L = distinct.Learner(F, K, "cpu", lr=1e-2, epochs=100, minibatch=128)
    L.weight, L.scale = 1.0, 1.0                                      # (scale: the executor's advantage std)
    M, G = 96, 3
    kind = torch.randint(0, K, (M, G))
    beh = torch.nn.functional.one_hot(kind, F).float()[..., :F] * 2.0 + 0.05 * torch.randn(M, G, F)
    ok = torch.ones(M, G, dtype=torch.bool)
    assert torch.all(L.bonus(beh, kind, ok) == 0)                     # uniform at the start: bonus 0
    st = L.fit({"beh": beh, "beh_ok": ok, "kind": kind})
    assert st["distinct_gain"] == pytest.approx(0.0, abs=1e-6) and st["distinct_chance"] == pytest.approx(0.1)
    st = L.fit({"beh": beh, "beh_ok": ok, "kind": kind})
    assert st["distinct_acc"] > 0.9 and st["distinct_gain"] > 1.0
    b = L.bonus(beh, kind, ok)
    assert float(b.mean()) > 1.0 and float(b.max()) <= math.log(K) + 1e-6
    L.scale = 0.1                                                     # in the advantage's units: x its std
    torch.testing.assert_close(L.bonus(beh, kind, ok), b * 0.1, rtol=1e-6, atol=1e-7)
    off = ok.clone()
    off[:, 1:] = False
    assert torch.all(L.bonus(beh, kind, off)[:, 1:] == 0)            # groups that do not count: 0
    L.weight = 0.0
    assert torch.all(L.bonus(beh, kind, ok) == 0)
    st2 = distinct.Learner(F, K, "cpu")
    assert st2.load(L.state()) and torch.equal(st2.net(beh), L.net(beh)) and st2.scale == 0.1


def test_the_groups_behaviour_is_their_kinds_shares_and_the_moves_from_the_sides_facing():
    env = battles(B=2)
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    rt.distinct = distinct.Learner(F, GCFG.kinds, "cpu")
    rows = torch.tensor([0])
    rt.plan(env, rows, deploy=False, obs=env.cur[0])                  # a fresh plan of row 0: empty sums
    r = rt.r
    u = env.st.u
    main = r["main"][0]
    g = int(main[(u["side"][0] == 1) & (main >= 0) & (main < GCFG.places)][0])     # an own group of row 0
    mem = (r["main"][0] == g) & (u["side"][0] == 1) & (u["men"][0] > 0) & ~u["gone"][0]
    R, N = env.R, env.N
    act = SimpleNamespace(kind=torch.full((R, N), 1, dtype=torch.long), run=torch.zeros(R, N, dtype=torch.bool))
    acting = torch.ones(R, N, dtype=torch.bool)
    rt.behave(env, act, acting)                                       # all "move", walking
    assert bool(r["b_fwd_ok"][0])
    f = r["b_fwd"][0]
    u["x"][0, mem] += 50.0 * f[0]
    u["z"][0, mem] += 50.0 * f[1]                                     # the group 50 m forward
    act.run[:] = True
    rt.behave(env, act, acting)                                       # running now
    beh, ok = rt.behaviour(rows)
    i = runtime.BEH.index
    assert bool(ok[0, g])
    assert float(beh[0, g, i("kind_move")]) == pytest.approx(1.0) and float(beh[0, g, :5].sum()) == pytest.approx(1.0)
    assert float(beh[0, g, i("run")]) == pytest.approx(0.5)
    assert float(beh[0, g, i("forward")]) == pytest.approx(0.5, abs=1e-4)
    assert float(beh[0, g, i("sideways")]) == pytest.approx(0.0, abs=1e-4)
    assert float(beh[0, g, i("path")]) == pytest.approx(0.5, abs=1e-4)
    assert torch.isfinite(beh).all()


def test_the_distinct_bonus_goes_into_the_rows_reward_and_not_into_the_generals_return():
    env = battles()
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    L = distinct.Learner(F, GCFG.kinds, "cpu")
    L.weight = 0.5
    L.bonus = lambda beh, kind, ok: ok.float()                        # 1 a group that counts
    rt.distinct = L
    rt.begin_chunk(env)
    env.step(actor, crit)
    rt.r["t_plan"][0] -= 31.0
    s = env.step(actor, crit)                                         # row 0 re-plans: its old plan's bonus is paid
    rec = rt.ready[-1]
    assert rec["beh"].shape == (1, GCFG.places, F) and torch.isfinite(rec["beh"]).all()
    groups = float(rec["beh_ok"].float().sum())
    assert groups > 0
    # the reward of the decision = the battle's (in the new plan's sum) + the bonus
    assert float(s["reward"][0]) == pytest.approx(float(rt.r["r_sum"][0]) + groups, abs=1e-5)
    assert rt.take_stats()["distinct_bonus"] == pytest.approx(1.0)


def test_a_tiny_update_with_the_critics_return_logs_its_numbers():
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.9997
    rt.distinct = distinct.Learner(F, GCFG.kinds, "cpu")
    rt.distinct.weight = 0.01
    rt.r["t_plan"] -= 28.0                                            # re-plans inside the chunk
    batch = rollout.collect(env, actor, crit, 4)
    ready = batch["general"]["ready"]
    assert ready is not None and "base" in ready and "beh" in ready
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), minibatch=8),
                    general=learn.LearnConfig(epochs=1, distinct=0.01))
    for k in ("general_critic_ev", "general_adv_mean", "general_adv_std", "general_task_in_w", "general_task_in_kind"):
        assert k in st and np.isfinite(st[k]), k
    assert st["general_task_in_w"] >= 0.0
    fit = rt.distinct.fit(ready)
    assert 0.0 <= fit["distinct_acc"] <= 1.0 and fit["distinct_groups"] > 0
