"""The general's learning signals as plain formulas (no torch: they take numpy arrays and torch tensors alike, so the
host's tests check them; docs/en/training/general.md "Training").

* plan_target: a plan's target by the unit network's critic (run.py --general-return critic) - the reward over the
  plan's decisions, discounted, + gamma^n x the critic's value where the plan ended (0 when the battle ended there);
  the plan's advantage is that - the critic's value where it began (plan_advantage).
* distinct_bonus: the executor's bonus for missions that tell apart (run.py --general-distinct W) - a classifier
  guesses a group's mission kind from what its units did during the plan; the bonus is log q(kind | behaviour) - log(1/K)
  (0 when it guesses no better than chance, up to log K when it is sure and right), clipped to [-log K, log K], x W.
* distinct_weight: W down to 0 linearly over `decay` updates of the chain (0: no decay).
"""
import math


def plan_target(r_disc, n, v_end, done, gamma):
    """The critic's target of plans [M]: r_disc (the reward of the plan's decisions, the k-th x gamma^k) + gamma^n x
    v_end (the critic's value at the decision the plan ended on), v_end not counted where done (the battle ended: its
    last reward is in r_disc)."""
    return r_disc + (gamma ** n) * v_end * (1.0 - done)


def plan_advantage(r_disc, n, v_start, v_end, done, gamma):
    """The plan's advantage by the critic: plan_target - v_start (the critic's value at the plan's decision; a battle's
    first plan, the deployment: at the battle's first decision)."""
    return plan_target(r_disc, n, v_end, done, gamma) - v_start


def distinct_bonus(logq, kinds, weight=1.0):
    """The bonus of groups: logq - their log-probability of their own mission kind by the classifier (any shape), kinds -
    the number of kinds K. -> weight x clip(logq + log K, -log K, log K)."""
    lk = math.log(kinds)
    return weight * (logq + lk).clip(-lk, lk)


def distinct_weight(w0, decay, updates):
    """The bonus's weight after `updates` updates of the chain: w0 down to 0 linearly over `decay` (0: no decay)."""
    if not decay:
        return float(w0)
    return float(w0) * max(0.0, 1.0 - updates / float(decay))
