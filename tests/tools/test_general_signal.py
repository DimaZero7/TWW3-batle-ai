"""tools/nn/general/signal.py: the general's learning signals as formulas (no torch) - the plan's target and advantage
by the unit critic (run.py --general-return critic), the distinct bonus and its decay (--general-distinct), the
mission branches' other kind and advantage (--general-branches) (docs/en/training/general.md "Training")."""
import math

import numpy as np
import pytest

from tools.nn.general import signal


def test_the_plans_target_and_advantage_by_the_critic_on_toy_numbers():
    g = 0.99
    rewards = [0.1, -0.2, 0.3]                                     # the plan's 3 decisions
    r_disc = sum(r * g ** k for k, r in enumerate(rewards))
    # a plan cut by a re-plan: the critic's value at the re-plan's decision counts, discounted by 3 decisions
    t = signal.plan_target(np.array([r_disc]), np.array([3.0]), np.array([0.5]), 0.0, g)
    assert t[0] == pytest.approx(r_disc + g ** 3 * 0.5)
    a = signal.plan_advantage(np.array([r_disc]), np.array([3.0]), np.array([0.2]), np.array([0.5]), 0.0, g)
    assert a[0] == pytest.approx(r_disc + g ** 3 * 0.5 - 0.2)


def test_at_the_battles_end_the_target_is_the_reward_with_its_terminal_part():
    g = 0.9997
    rewards = [0.01, 0.0, 1.0]                                    # the last: the win
    r_disc = sum(r * g ** k for k, r in enumerate(rewards))
    t = signal.plan_target(np.array([r_disc]), np.array([3.0]), np.array([123.0]), 1.0, g)   # v_end not counted
    assert t[0] == pytest.approx(r_disc)
    done = np.array([0.0, 1.0])
    t2 = signal.plan_target(np.array([0.1, 0.1]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), done, 1.0)
    assert list(t2) == pytest.approx([1.1, 0.1])


def test_a_critic_that_is_right_gives_no_advantage():
    """V(start) = r + gamma^n V(end) exactly: the advantage is 0 (the plan did as the critic expected)."""
    g = 0.995
    v_end, n, r = 0.4, 30.0, 0.05
    v_start = r + g ** n * v_end
    assert signal.plan_advantage(r, n, v_start, v_end, 0.0, g) == pytest.approx(0.0)


def test_the_distinct_bonus_is_zero_at_chance_and_grows_with_the_guess():
    K = 10
    chance = np.log(np.full(4, 1.0 / K))
    assert np.allclose(signal.distinct_bonus(chance, K), 0.0)
    sure = np.log(np.array([0.2, 0.5, 0.9, 1.0]))
    b = signal.distinct_bonus(sure, K)
    assert np.all(np.diff(b) > 0) and b[-1] == pytest.approx(math.log(K))
    # a confident wrong guess is bounded below by -log K; the weight scales it
    assert signal.distinct_bonus(np.array([-50.0]), K)[0] == pytest.approx(-math.log(K))
    assert signal.distinct_bonus(np.log(np.array([0.5])), K, 0.01)[0] == pytest.approx(0.01 * math.log(5))


def test_the_distinct_weight_decays_linearly_to_zero():
    assert signal.distinct_weight(0.02, 100, 0) == pytest.approx(0.02)
    assert signal.distinct_weight(0.02, 100, 50) == pytest.approx(0.01)
    assert signal.distinct_weight(0.02, 100, 100) == 0.0
    assert signal.distinct_weight(0.02, 100, 300) == 0.0
    assert signal.distinct_weight(0.02, 0, 10 ** 6) == pytest.approx(0.02)          # no decay


def test_the_imitation_decays_to_its_floor_not_to_zero():
    assert signal.imitate_weight(0.3, 300, 0, 0.15) == pytest.approx(0.3)
    assert signal.imitate_weight(0.3, 300, 150, 0.15) == pytest.approx(0.225)
    assert signal.imitate_weight(0.3, 300, 300, 0.15) == pytest.approx(0.15)
    assert signal.imitate_weight(0.3, 300, 10 ** 6, 0.15) == pytest.approx(0.15)      # stays on the floor
    assert signal.imitate_weight(0.3, 300, 300, 0.0) == 0.0                           # floor 0: the old decay to 0
    assert signal.imitate_weight(0.3, 0, 10 ** 6, 0.15) == pytest.approx(0.3)         # no decay
    assert signal.imitate_weight(0.0, 300, 0, 0.15) == 0.0                            # off: the floor does not turn it on
    assert signal.imitate_weight(0.1, 300, 10 ** 6, 0.15) == pytest.approx(0.1)       # a floor above W: W stays


def test_the_distinct_scale_follows_the_executors_advantage_std():
    """The bonus is in the executor's advantage's units: its scale is a running mean of the advantage's std (ppo's
    adv_std_*), kept when the update logs none, never below the floor."""
    assert signal.SCALE0 == pytest.approx(0.06)
    s = signal.distinct_scale(0.06, 0.2)
    assert s == pytest.approx(0.7 * 0.06 + 0.3 * 0.2)
    for _ in range(60):
        s = signal.distinct_scale(s, 0.2)
    assert s == pytest.approx(0.2, abs=1e-6)
    assert signal.distinct_scale(0.1, None) == 0.1 and signal.distinct_scale(0.1, float("nan")) == 0.1
    assert signal.distinct_scale(0.001, 0.0001) == 0.005


def test_the_other_kind_skips_the_chosen_one_and_covers_the_rest_evenly():
    K = 10
    for a in range(K):
        got = [int(signal.other_kind(d, a)) for d in range(K - 1)]
        assert sorted(got) == [k for k in range(K) if k != a]
    draws = np.array([[0, 3, 8], [0, 3, 8]])
    assert signal.other_kind(draws, np.array([[0], [9]])).tolist() == [[1, 4, 9], [0, 3, 8]]


def test_the_branches_advantage_is_the_kinds_q_less_a_baseline_that_does_not_depend_on_it():
    # the true values of the 4 kinds of a group and the policy
    Q = np.array([0.30, -0.10, 0.05, 0.20])
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    K = len(Q)
    V = float(pi @ Q)
    for a in range(K):
        others = [b for b in range(K) if b != a]
        # uniform: the expected price is K / (K - 1) x (Q(a) - mean Q)
        e_uni = np.mean([Q[a] - Q[b] for b in others])
        assert e_uni == pytest.approx(K / (K - 1) * (Q[a] - Q.mean()))
        # by the policy without a, x (1 - pi(a)): exactly Q(a) - V
        q = pi[others] / (1 - pi[a])
        e_pol = sum(qb * (Q[a] - Q[b]) for qb, b in zip(q, others))
        assert signal.policy_weight(pi[a]) * e_pol == pytest.approx(Q[a] - V)
    # the policy gradient's direction: sum_a pi(a) grad log pi(a) x adv(a) with adv = Q - c for any c is the same
    # (the softmax's grad of log pi(a) by logit j: 1[a = j] - pi(j))
    grad = lambda adv: np.array([sum(pi[a] * ((a == j) - pi[j]) * adv[a] for a in range(K)) for j in range(K)])
    assert np.allclose(grad(Q - V), grad(K / (K - 1) * (Q - Q.mean())) * (K - 1) / K)


def test_the_branches_advantage_is_normalised_floored_and_clipped():
    price = np.array([0.01, -0.02, 0.0, 0.03])
    adv = signal.kind_advantage(price, np.ones(4), 0.005, 5.0)
    rms = math.sqrt(np.mean(price ** 2))
    assert np.allclose(adv, price / rms)
    tiny = signal.kind_advantage(np.array([1e-4, -1e-4]), np.ones(2), 0.005, 5.0)
    assert np.allclose(tiny, [0.02, -0.02])                                  # the floor: noise is not blown up
    w = np.array([1.0, 0.5, 1.0, 0.0])
    assert np.allclose(signal.kind_advantage(price, w, 0.0, 5.0)[3], 0.0)    # pi(a) = 1: no signal
    big = signal.kind_advantage(np.array([1.0] + [0.0] * 99), np.ones(100), 0.0, 5.0)
    assert big[0] == pytest.approx(5.0)                                      # clipped


def test_the_mission_bonus_pays_what_is_above_the_kinds_mean():
    m = np.array([0.9, 0.3, 0.5])
    np.testing.assert_allclose(signal.exec_bonus(m, 0.5, 2.0), [0.8, -0.4, 0.0])
    assert np.all(signal.exec_bonus(m, 0.5, 0.0) == 0)                       # W = 0: nothing paid


def test_the_mission_bonus_into_the_advantage_is_a_reward_to_go_cut_at_the_battles_end():
    gl = 0.9
    T, R, N = 200, 2, 3
    bonus = np.zeros((T, R, N))
    bonus[:, 0, 0] = 0.3                                                     # a steady bonus of one unit
    done = np.zeros((T, R))
    adv = signal.exec_advantage(bonus, done, gl)
    assert adv[0, 0, 0] == pytest.approx(0.3, rel=1e-6)                      # x (1 - gl): a steady bonus counts once
    assert np.all(adv[:, 1] == 0) and np.all(adv[:, 0, 1:] == 0)             # only that unit's
    assert adv[-1, 0, 0] == pytest.approx(0.3 * (1 - gl))                    # the chunk's end: no future
    one = np.zeros((T, R, N))
    one[5, 1, 2] = 1.0                                                       # a bonus at decision 5 ...
    a1 = signal.exec_advantage(one, done, gl)
    assert a1[5, 1, 2] == pytest.approx(1 - gl) and a1[3, 1, 2] == pytest.approx((1 - gl) * gl ** 2)
    assert np.all(a1[6:, 1, 2] == 0)                                         # ... credits the decisions up to it only
    cut = done.copy()
    cut[4, 1] = 1.0                                                          # the battle ended at 4: a new one from 5
    a2 = signal.exec_advantage(one, cut, gl)
    assert np.all(a2[:5, 1, 2] == 0) and a2[5, 1, 2] == pytest.approx(1 - gl)


def test_every_target_type_is_taken_by_two_kinds_at_least():
    """config.MISSION_KINDS: every kind takes some target types and says what it means; each target type is taken by
    two kinds at least (the mission branches swap the kind and keep the target)."""
    from tools.nn.general import config
    ks = config.MISSION_KINDS
    assert all(k.targets and set(k.targets) <= set(config.TARGET_TYPES) and k.meaning for k in ks)
    al = config.allowed(ks)
    assert len(al) == len(ks) and all(len(row) == len(config.TARGET_TYPES) for row in al)
    for j, t in enumerate(config.TARGET_TYPES):
        assert sum(row[j] for row in al) >= 2, t


def test_the_kind_advantage_takes_each_rows_scale_when_given():
    price, w = np.array([0.02, -0.01, 0.5]), np.ones(3)
    a = signal.kind_advantage(price, w, 0.005, 5.0, np.array([0.01, 0.001, 0.01]))
    assert list(a) == pytest.approx([2.0, -2.0, 5.0])               # the floor 0.005 under the second, the clip on the third
    chunk = signal.kind_advantage(price, w, 0.005, 5.0)
    assert list(chunk) == pytest.approx(list(price / np.sqrt(np.mean(price ** 2))))


def test_the_kind_norm_keeps_each_kinds_own_scale_over_updates():
    n = signal.KindNorm(3, floor=0.001)
    assert n.scale() == [0.001] * 3                                  # before any update: the floor
    big, small = [0.04] * 40, [0.004] * 40
    sc = n.update([0] * 40 + [1] * 40, big + small)
    assert sc[0] == pytest.approx(0.04) and sc[1] == pytest.approx(0.004)
    # a kind with few rows is filled with the update's mean square of all prices up to NORM_PRIOR rows
    ms = (40 * 0.04 ** 2 + 40 * 0.004 ** 2) / 80
    assert sc[2] == pytest.approx(math.sqrt(ms))
    # the memory: an update without kind 0 decays its sums but keeps its scale; kind 1's new rows move it
    sc2 = n.update([1] * 40, [0.008] * 40)
    assert sc2[0] == pytest.approx(0.04)
    want = math.sqrt((0.9 * 40 * 0.004 ** 2 + 40 * 0.008 ** 2) / (0.9 * 40 + 40))
    assert sc2[1] == pytest.approx(want)
    m = signal.KindNorm(3, floor=0.001)
    assert m.load(n.state()) and m.rms() == pytest.approx(n.rms()) and m.scale() == pytest.approx(n.scale())
    assert not signal.KindNorm(4).load(n.state()) and not signal.KindNorm(3).load(None)


def test_the_branches_running_mean_takes_the_updates_rows_and_forgets_in_about_ten_updates():
    c = signal.RunningMean()
    assert c.mean() == 0.0                                         # before any row
    assert c.update([-0.003, -0.001, -0.002]) == pytest.approx(-0.002)   # the first update: its own mean
    # a steady bias is learned whatever the pairs' spread; an old one fades (~10 updates' memory)
    for _ in range(60):
        m = c.update([0.001 + 0.01, 0.001 - 0.01])
    assert m == pytest.approx(0.001, abs=1e-6)
    c2 = signal.RunningMean()
    c2.update([1.0])
    after = c2.update([0.0])
    assert after == pytest.approx(signal.NORM_DECAY / (signal.NORM_DECAY + 1.0))
    assert c2.update([]) == pytest.approx(after)                   # an update without rows keeps the mean
    again = signal.RunningMean()
    assert again.load(c.state()) and again.mean() == pytest.approx(c.mean())
    assert not signal.RunningMean().load(None) and not signal.RunningMean().load({})


def test_an_outcome_is_known_at_the_horizon_or_at_the_battles_end():
    el = np.array([10.0, 59.0, 60.0, 61.0, 5.0])
    ended = np.array([False, False, False, False, True])
    assert signal.outcome_due(el, 60.0, ended).tolist() == [False, False, True, True, True]
