"""tools/nn/general/signal.py: the general's learning signals as formulas (no torch) - the plan's target and advantage
by the unit critic (run.py --general-return critic), the distinct bonus and its decay (--general-distinct)
(docs/en/training/general.md "Training")."""
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
