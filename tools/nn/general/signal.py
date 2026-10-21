"""The general's learning signals as plain formulas (no torch: they take numpy arrays and torch tensors alike, so the
host's tests check them; docs/en/training/general.md "Training").

* plan_target: a plan's target by the unit network's critic (run.py --general-return critic) - the reward over the
  plan's decisions, discounted, + gamma^n x the critic's value where the plan ended (0 when the battle ended there);
  the plan's advantage is that - the critic's value where it began (plan_advantage).
* distinct_bonus: the executor's bonus for missions that tell apart (run.py --general-distinct W) - a classifier
  guesses a group's mission kind from what its units did during the plan; the bonus is log q(kind | behaviour) - log(1/K)
  (0 when it guesses no better than chance, up to log K when it is sure and right), clipped to [-log K, log K], x W.
* distinct_weight: W down to 0 linearly over `decay` updates of the chain (0: no decay).
* distinct_scale: the bonus is in the executor's advantage's units: W x the guess's gain x the spread (std) of the
  executor's advantage before its normalisation (an average over the updates; ppo's adv_std_*) - W = 1 pays a group
  one advantage std for each nat the classifier guesses its kind better than chance.
* The mission branches (run.py --general-branches P; branch.py): a plan replayed with one group's kind a swapped for b.
  other_kind maps a uniform draw over the K - 1 other kinds; kind_advantage is the kind choice's advantage from the
  branches' prices (price = the chosen branch's trade - the swapped one's, the mean over the copies):
    uniform b (b != a, each 1 / (K - 1)): E[price] = Q(a) - mean_{b != a} Q(b) = K / (K - 1) x (Q(a) - mean Q): an
      advantage with an action-independent baseline (unbiased), weight 1;
    b by the policy without a (pi(b) / (1 - pi(a))): E[price] = Q(a) - E_{b != a} Q(b), and x (1 - pi(a)) it is
      exactly Q(a) - V (V = E_pi Q): weight 1 - pi(a).
  The weighted prices are divided by their root mean square over the chunk (at least `floor`) and clipped to +-clip.
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


SCALE0 = 0.06       # the executor's advantage std before the first update (line S27/g1-g8 logs: 0.058-0.068)


def distinct_scale(prev, adv_std, keep=0.7, floor=0.005):
    """The bonus's scale after an update: the running mean (keep x prev + (1 - keep) x now) of the executor's advantage
    std adv_std (None or not finite: prev kept), at least floor."""
    if adv_std is None or not math.isfinite(adv_std) or adv_std <= 0:
        return float(prev)
    return max(floor, keep * float(prev) + (1.0 - keep) * float(adv_std))


def other_kind(draw, chosen):
    """A uniform draw over the K - 1 other kinds: draw in 0..K-2 -> the kind (chosen skipped)."""
    return draw + (draw >= chosen)


def policy_weight(p_chosen):
    """The weight of a price whose other kind came from the policy without the chosen one: 1 - pi(chosen)."""
    return 1.0 - p_chosen


def kind_advantage(price, weight, floor, clip):
    """The kind choice's advantage of the branches (module doc): price x weight, divided by the root mean square of
    those over the chunk (at least floor), within +-clip. price, weight [E]."""
    x = price * weight
    rms = float((x * x).mean()) ** 0.5
    return (x / max(rms, floor)).clip(-clip, clip)
