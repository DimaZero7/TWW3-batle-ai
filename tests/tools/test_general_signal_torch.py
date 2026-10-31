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
    # (seed 9 since 29.10.2026: the token grew by the cavalry's 4 columns and the random nets of seed 0 made no group that
    # counts - the bonus tests need one; build/diag_cav/seeds/try.py: seeds 0-9, only 9 gives all four; seed 0 again since
    # the artillery's 2 columns, 29.10.2026: build/diag_arty/seeds_try.py, seeds 0-15 - 0, 3 and 11 give all four, 9 one;
    # seed 2 since the Pusbags' 4 columns, 31.10.2026: build/diag_pusbags/seeds/try.log - 2, 3, 6, 7, 8 give all four)
    torch.manual_seed(2 if seed == 0 else seed)
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


def test_the_generals_update_comes_before_the_unit_networks_epochs():
    """ppo.update runs learn.update first: its numbers are those of learn.update alone on the same weights (the unit
    network's epochs, which move the general's weights through the task cards, come after it)."""
    import copy
    env = battles(B=4)
    actor, crit = nets()
    env.step(actor, crit)
    env.general_rt.ret = "critic"
    env.general_rt.r["t_plan"] -= 28.0
    batch = rollout.collect(env, actor, crit, 4)
    ready = batch["general"]["ready"]
    assert ready is not None
    L = learn.LearnConfig(epochs=1)
    a1, c1 = copy.deepcopy(actor), copy.deepcopy(crit)
    opt1 = torch.optim.Adam(list(a1.parameters()) + list(c1.parameters()), lr=1e-3)
    torch.manual_seed(3)
    st1 = learn.update(a1, opt1, ready, L)
    a2, c2 = copy.deepcopy(actor), copy.deepcopy(crit)
    opt2 = torch.optim.Adam(list(a2.parameters()) + list(c2.parameters()), lr=1e-3)
    torch.manual_seed(3)
    st2 = ppo.update(a2, c2, opt2, batch, dataclasses.replace(ppo.PPOConfig(), minibatch=8), general=L)
    for k in ("general_kl_plan", "general_clip", "general_kl", "general_pl"):
        assert st2[k] == st1[k], k
    assert st1["general_clip"] < 0.01                                 # (the plans' weights: no step between)


def test_the_clip_share_is_logged_apart_for_the_fresh_and_the_old_plans():
    env = battles(B=4)
    actor, crit = nets()
    env.step(actor, crit)                                             # the deployment's plans: before the chunk (old)
    rt = env.general_rt
    rt.begin_chunk(env)
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)                                             # every row re-plans: the old plans end
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)                                             # ... and again: the fresh plans end
    ready = rt.end_chunk()["ready"]
    assert bool(ready["new"].any()) and bool((~ready["new"]).any())
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    st = learn.update(actor, opt, ready, learn.LearnConfig(epochs=1, minibatch=10 ** 6))
    assert st["general_clip_fresh"] == 0.0 and st["general_clip_old"] == 0.0   # one step from the plans' own weights
    assert 0.0 < st["general_fresh_share"] < 1.0


def old_and_fresh(seed=0):
    """(actor, ready): a chunk's ended plans, some begun before it (old) and some in it (fresh)."""
    env = battles(B=4)
    actor, crit = nets(seed)
    env.step(actor, crit)
    rt = env.general_rt
    rt.begin_chunk(env)
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)
    ready = rt.end_chunk()["ready"]
    assert bool(ready["new"].any()) and bool((~ready["new"]).any())
    return actor, ready


def test_the_stale_weight_is_one_for_the_fresh_plans_and_truncated_for_the_old():
    lp_plan = torch.tensor([[-1.0, -2.0], [-1.0, -2.0], [-1.0, -2.0]])
    lp0 = torch.tensor([[-1.5, -1.0], [-1.5, -1.0], [-1.0, -2.0]])
    new = torch.tensor([True, False, False])
    ref, w = learn.stale(lp_plan, lp0, new)
    assert torch.equal(ref[0], lp_plan[0]) and torch.equal(w[0], torch.ones(2))       # fresh: as the plain PPO
    assert torch.equal(ref[1:], lp0[1:])                                                 # old: the update's start
    assert float(w[1, 0]) == pytest.approx(math.exp(-0.5)) and float(w[1, 1]) == 1.0     # min(1, exp(lp0 - lp_plan))
    assert torch.equal(w[2], torch.ones(2))
    ref, w = learn.stale(lp_plan[:, 0], lp0[:, 0], new)                                  # [E]: the branches' shape
    assert ref.shape == (3,) and float(w[1]) == pytest.approx(math.exp(-0.5))


def test_with_fresh_plans_only_the_update_is_the_plain_ppo_bit_for_bit():
    """All plans fresh: the same weights whether the records say so or carry no mark (the plain PPO on the plans'
    own log-probabilities, the code before the stale plans' weight)."""
    import copy
    actor, ready = old_and_fresh()
    fresh = {k: v[ready["new"]] for k, v in ready.items()}
    plain = {k: v for k, v in fresh.items() if k != "new"}
    outs = []
    for rec in (fresh, plain):
        a = copy.deepcopy(actor)
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        torch.manual_seed(5)
        st = learn.update(a, opt, rec, learn.LearnConfig(epochs=2, minibatch=4))
        outs.append((a, st))
    for (n, p), q in zip(outs[0][0].named_parameters(), outs[1][0].parameters()):
        assert torch.equal(p, q), n
    assert outs[0][1]["general_pl"] == outs[1][1]["general_pl"]
    assert "general_stale_w" not in outs[0][1]


def test_the_old_plans_ratio_starts_at_the_updates_policy_and_their_lag_is_a_weight():
    """The general's weights moved after the plans were made (as an update between chunks does): the old plans are
    not clipped at the first step (their ratio is against the update's start), their lag is the weight stale_w < 1;
    the fresh ones (ratio against their own log-probability) are."""
    actor, ready = old_and_fresh()
    gen = actor.general
    torch.manual_seed(1)
    with torch.no_grad():
        for p in gen.parameters():
            p.add_(0.3 * torch.randn_like(p) * p.abs().mean())
    G = gen.cfg.places
    with torch.no_grad():
        lp0 = gen(ready, act={k: ready[k] for k in learn.CHOICES})[1]["lp"]
    old = ready["active"][:, :G] & ~ready["new"][:, None]
    want = float(torch.exp((lp0 - ready["lp"]).clamp(max=0))[old].mean())
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    st = learn.update(actor, opt, ready, learn.LearnConfig(epochs=1, minibatch=10 ** 6))
    assert st["general_clip_old"] == 0.0 and st["general_clip_fresh"] > 0.0
    assert st["general_stale_w"] == pytest.approx(want, rel=1e-5) and 0.0 < st["general_stale_w"] < 1.0


def test_the_replans_are_counted_per_ended_battle_and_per_its_minute():
    env = battles(B=4)
    actor, crit = nets()
    env.step(actor, crit)                                             # the deployment
    rt = env.general_rt
    for _ in range(2):
        rt.r["t_plan"] -= 31.0
        env.step(actor, crit)                                         # every row re-plans
    rt.take_stats()
    finished = torch.zeros(env.B, dtype=torch.bool)
    finished[0] = True
    t = float(env.st.t[0])
    rt.after(env, torch.zeros(env.R), finished)
    s = rt.take_stats()
    assert s["replans_per_battle"] == 2.0                              # both sides of battle 0: 2 re-plans each
    assert s["battle_min"] == pytest.approx(t / 60, abs=0.01)
    assert s["replans_per_min"] == pytest.approx(2.0 / (t / 60), rel=1e-3)


def exec_chunk(weight, measure=True, teacher=0.0, T=4):
    """A chunk of T decisions (every row re-plans inside it) with the missions measured and the executor's bonus at
    weight (--general-exec), the teacher's share. -> (env, actor, critic, batch)."""
    torch.manual_seed(7)
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    torch.manual_seed(8)
    env.step(actor, crit)
    rt = env.general_rt
    rt.ret, rt.gamma = "critic", 0.9997
    rt.measure, rt.exec_w = measure, weight
    rt.set_teacher(teacher)
    rt.r["t_plan"] -= 28.0
    torch.manual_seed(9)
    return env, actor, crit, rollout.collect(env, actor, crit, T)


def test_the_mission_bonus_leaves_the_reward_the_generals_return_and_the_critic_bit_for_bit():
    """--general-exec: the bonus goes into the units' advantage only - the batch's reward and values, the general's
    plans' targets and bases, the critic's update are those without it to the bit; the executor's update is not."""
    _, a0, c0, b0 = exec_chunk(0.0, measure=False)
    env, a1, c1, b1 = exec_chunk(1.0)
    assert "exec" not in b0 and b1["exec"].shape == b1["reward"].shape + (env.N,)
    assert float(b1["exec"].abs().sum()) > 0
    for k in ("reward", "value", "done", "lp"):
        assert torch.equal(b0[k], b1[k]), k
    r0, r1 = b0["general"]["ready"], b1["general"]["ready"]
    for k in ("ret", "base", "lp", "kind", "target"):
        assert torch.equal(r0[k], r1[k]), k
    cfg = dataclasses.replace(ppo.PPOConfig(), minibatch=8, target_kl=0.0)
    outs = []
    for a, c, b in ((a0, c0, b0), (a1, c1, b1)):
        opt = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=1e-3)
        torch.manual_seed(10)
        outs.append(ppo.update(a, c, opt, b, cfg))
    for (n, p), q in zip(c0.named_parameters(), c1.parameters()):
        assert torch.equal(p, q), n                                    # the critic: not a bit moved by the bonus
    assert outs[0]["value_loss"] == outs[1]["value_loss"]
    assert any(not torch.equal(p, q) for p, q in zip(a0.parameters(), a1.parameters()))   # the executor: it is
    assert "general_exec_adv_std" in outs[1] and outs[1]["general_exec_adv_std"] > 0


def test_the_missions_are_measured_and_logged_by_kind():
    env, actor, crit, batch = exec_chunk(1.0)
    rt = env.general_rt
    st = rt.take_stats()
    names = [k.name for k in gconfig.MISSION_KINDS]
    seen = [n for n in names if f"exec_{n}" in st]
    assert seen and all(0.0 <= st[f"exec_{n}"] <= 1.0 for n in seen)
    assert st["exec_units"] > 0 and st["exec_bonus_abs"] > 0 and st["exec_weight"] == 1.0
    assert abs(st["exec_bonus"]) <= st["exec_bonus_abs"]
    # (each share rounded to 3 decimals: the sum within half a thousandth a kind)
    assert sum(st[f"kind_share_{n}"] for n in names) == pytest.approx(1.0, abs=5e-4 * len(names) + 1e-9)
    assert 0.0 < st["masked_pairs"] < 1.0
    assert all(math.isfinite(x) for x in st["exec_base"] if x == x)
    # the paid units are our units with a group in a plan of the learner's
    paid = batch["exec"] != 0
    assert bool(paid.any())


def test_the_teachers_plans_train_the_executor_and_not_the_general():
    env, actor, crit, batch = exec_chunk(1.0, teacher=1.0)
    rt = env.general_rt
    taught = rt.r["teach"] & (rt.r["owner"] == 0)
    assert bool(taught.any())                                          # the re-plans of the chunk are the teacher's
    # their missions are the rule's: every own group's kind takes its target's type, no record for the general
    for row in taught.nonzero().flatten().tolist():
        for g in rt.r["active"][row, :GCFG.places].nonzero().flatten().tolist():
            t = int(rt.r["target"][row, g])
            tt = gconfig.TARGET_TYPES[0 if t < GCFG.places else 1 if t < 2 * GCFG.places else 2]
            assert tt in gconfig.MISSION_KINDS[int(rt.r["kind"][row, g])].targets
    assert batch["general"]["table"] is not None                       # the executor's cards: the teacher's plans too
    st = rt.take_stats()
    assert st["teach_share"] > 0
    # the teacher's plans end: no record for the general's PPO
    learner = rt.r["has"] & (rt.r["owner"] == 0)
    n_free = int((learner & ~rt.r["teach"]).sum())                     # plans of the general's own still in force
    rt.begin_chunk(env)
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)                                              # every plan ends at a re-plan
    out = rt.end_chunk()
    n = 0 if out["ready"] is None else out["ready"]["lp"].shape[0]
    assert n == n_free < int(learner.sum()) and out["table"] is not None


def teach_and_ready():
    """(env, actor, ready, teach): a chunk's ended plans of the general (the deployment's) and of the teacher (the
    chunk's re-plans, ended at the next one) with the missions measured."""
    env, actor, crit, batch = exec_chunk(1.0, teacher=1.0)
    rt = env.general_rt
    taught = rt.r["has"] & (rt.r["owner"] == 0) & rt.r["teach"]
    want = {k: rt.r[k][taught].clone() for k in ("kind", "target", "main")}
    rt.begin_chunk(env)
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)
    teach = rt.end_chunk()["teach"]
    assert teach is not None and teach["lp"].shape[0] == int(taught.sum())
    for k, v in want.items():
        assert torch.equal(teach[k], v), k                           # the teacher's choices on the network's groups
    return env, actor, batch["general"]["ready"], teach


def test_the_teachers_plans_are_kept_for_the_imitation_with_the_groups_it_gave_a_mission():
    _, actor, ready, teach = teach_and_ready()
    G = GCFG.places
    use = teach["use"]
    assert bool(use.any()) and not bool((use & ~teach["active"][:, :G]).any())
    for k in ("num", "prev_main", "prev_kind", "has_prev", "main", "perm"):
        assert k in teach, k                                           # the states to compute the plan again
    loss, st = learn.imitate_term(actor.general, teach)
    assert loss.requires_grad and st["nll"] > 0 and 0 <= st["acc_kind"] <= 1 and 0 <= st["acc_target"] <= 1


def test_the_imitation_moves_only_the_general_and_at_zero_weight_changes_nothing():
    import copy
    _, actor, ready, teach = teach_and_ready()
    outs = []
    for w, t in ((0.0, None), (0.0, teach), (1.0, teach)):
        a = copy.deepcopy(actor)
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        torch.manual_seed(5)
        outs.append((a, learn.update(a, opt, ready, learn.LearnConfig(epochs=1, minibatch=8, imitate=w), teach=t)))
    for (n, p), q in zip(outs[0][0].named_parameters(), outs[1][0].parameters()):
        assert torch.equal(p, q), n                                     # W = 0: only logged
    st = outs[1][1]
    assert st["general_imitate_w"] == 0.0 and st["general_imitate_nll"] > 0 and "general_imitate_loss" not in st
    assert "general_imitate_acc_kind" in st and "general_imitate_acc_target" in st
    moved = {n for (n, p), q in zip(outs[1][0].named_parameters(), outs[2][0].parameters()) if not torch.equal(p, q)}
    assert moved and all(n.startswith("general.") for n in moved)      # the general's weights only
    assert outs[2][1]["general_imitate_loss"] > 0


def test_imitation_steps_bring_the_general_to_the_teachers_choices():
    _, actor, _, teach = teach_and_ready()
    gen = actor.general
    opt = torch.optim.Adam(gen.parameters(), lr=3e-3)
    _, st0 = learn.imitate_term(gen, teach)
    for _ in range(40):
        loss, _ = learn.imitate_term(gen, teach)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, st1 = learn.imitate_term(gen, teach)
    assert st1["nll"] < 0.7 * st0["nll"] and st1["acc_kind"] >= st0["acc_kind"]


def test_the_generals_execution_bonus_is_its_groups_centred_measure_and_zero_is_the_plain_ppo():
    import copy
    _, actor, crit, batch = exec_chunk(0.0)                          # (measured, the executor not paid)
    ready = batch["general"]["ready"]
    G = GCFG.places
    ok = ready["exec_ok"] & ready["active"][:, :G]
    assert bool(ok.any()) and bool(torch.isfinite(ready["exec_c"]).all())
    assert float(ready["exec_c"].abs().max()) <= 1.0 and bool((ready["exec_c"][~ready["exec_ok"]] == 0).all())
    plain = {k: v for k, v in ready.items() if k not in ("exec_c", "exec_ok")}
    outs = []
    for rec, s in ((plain, 0.0), (ready, 0.0), (ready, 0.5)):
        a = copy.deepcopy(actor)
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        torch.manual_seed(5)
        outs.append((a, learn.update(a, opt, rec, learn.LearnConfig(epochs=1, minibatch=8, exec_general=s))))
    for (n, p), q in zip(outs[0][0].named_parameters(), outs[1][0].parameters()):
        assert torch.equal(p, q), n                                     # S = 0: bit for bit
    assert "general_exec_general_term" not in outs[1][1]
    st = outs[2][1]
    want = float((0.5 * ready["exec_c"]).abs()[ok].mean())
    assert st["general_exec_general_term"] == pytest.approx(want, rel=1e-5) and want > 0
    assert any(not torch.equal(p, q) for p, q in zip(outs[1][0].parameters(), outs[2][0].parameters()))
