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
