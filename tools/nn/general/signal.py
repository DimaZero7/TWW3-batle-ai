"""The general's learning signals as plain formulas (no torch: they take numpy arrays and torch tensors alike, so the
host's tests check them; docs/en/training/general.md "Training").

* plan_target: a plan's target by the unit network's critic (run.py --general-return critic) - the reward over the
  plan's decisions, discounted, + gamma^n x the critic's value where the plan ended (0 when the battle ended there);
  the plan's advantage is that - the critic's value where it began (plan_advantage).
* distinct_bonus: the executor's bonus for missions that tell apart (run.py --general-distinct W) - a classifier
  guesses a group's mission kind from what its units did during the plan; the bonus is log q(kind | behaviour) - log(1/K)
  (0 when it guesses no better than chance, up to log K when it is sure and right), clipped to [-log K, log K], x W.
* distinct_weight: W down to 0 linearly over `decay` updates of the chain (0: no decay).
* imitate_weight: the teacher's imitation's W down to a floor M (run.py --general-imitate-min), not to 0, linearly over
  `decay` updates (W = 0: off, the floor does not turn it on; M above W: W stays).
* distinct_scale: the bonus is in the executor's advantage's units: W x the guess's gain x the spread (std) of the
  executor's advantage before its normalisation (an average over the updates; ppo's adv_std_*) - W = 1 pays a group
  one advantage std for each nat the classifier guesses its kind better than chance.
* exec_bonus / exec_advantage: the executor's bonus for carrying out its mission (run.py --general-exec W;
  mission.py measures it): per unit and decision W x (measure - its kind's running mean), into the unit's own advantage
  as a reward-to-go (never the reward: the critic and the general's return do not see it).
* The mission branches (run.py --general-branches P; branch.py): a plan replayed with one group's kind a swapped for b.
  other_kind maps a uniform draw over the K - 1 other kinds; kind_advantage is the kind choice's advantage from the
  branches' prices (price = the chosen branch's trade - the swapped one's, the mean over the copies):
    uniform b (b != a, each 1 / (K - 1)): E[price] = Q(a) - mean_{b != a} Q(b) = K / (K - 1) x (Q(a) - mean Q): an
      advantage with an action-independent baseline (unbiased), weight 1;
    b by the policy without a (pi(b) / (1 - pi(a))): E[price] = Q(a) - E_{b != a} Q(b), and x (1 - pi(a)) it is
      exactly Q(a) - V (V = E_pi Q): weight 1 - pi(a).
  The weighted prices are divided by their root mean square over the chunk (at least `floor`) and clipped to +-clip;
  with a scale per row (KindNorm, run.py --general-kind-norm per-kind, the default) by that instead.
* KindNorm: each mission kind's running scale of its prices (the ability branches' AbilityNorm, --ability-norm
  per-ability): per kind (the chosen one of the row) the decayed sums of its prices' squares and of its rows; an update
  multiplies both by NORM_DECAY (~10 updates' memory) and adds its rows; a kind's root mean square = sqrt((squares +
  fill x the update's mean square of all prices) / (rows + fill)), fill = NORM_PRIOR less its rows (at least 0), at
  least the floor. One chunk's 63-128 rows spread over 6 kinds give 10-20 rows a kind: a chunk's own mean square of
  them is mostly noise, and the kinds' prices differ in scale (a withdraw moves less gold than an attack).
* RunningMean: the branches' running mean price over all pairs of kinds (run.py --general-branch-center 1): decayed
  sums of the prices and of their rows (x NORM_DECAY an update, ~10 updates' memory, as KindNorm), the update's rows
  added before the mean is read. The partial open loop's "other" copy re-decides against a world that replays orders
  made for "chosen" and cannot answer it, so "chosen" wins by a steady -0.0027 of the budget whatever the pair
  (build/diag_kind_critic): the centred price = price - that mean.
* outcome_due: a plan's outcome for the outcome Q (run.py --general-kind-q outcome|both; kind_q.py) is known once
  `horizon` seconds of battle passed since the plan began, or earlier when the battle ended (its trade is then final;
  "ended" marks it).
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


def imitate_weight(w0, decay, updates, floor=0.0):
    """The imitation's weight after `updates` updates of the chain: w0 down to min(floor, w0) linearly over `decay` (0:
    no decay); w0 <= 0: 0 (off)."""
    w0 = float(w0)
    if w0 <= 0:
        return 0.0
    m = min(max(0.0, float(floor)), w0)
    return m + (w0 - m) * (distinct_weight(1.0, decay, updates))


SCALE0 = 0.06       # the executor's advantage std before the first update (line S27/g1-g8 logs: 0.058-0.068)


def distinct_scale(prev, adv_std, keep=0.7, floor=0.005):
    """The bonus's scale after an update: the running mean (keep x prev + (1 - keep) x now) of the executor's advantage
    std adv_std (None or not finite: prev kept), at least floor."""
    if adv_std is None or not math.isfinite(adv_std) or adv_std <= 0:
        return float(prev)
    return max(floor, keep * float(prev) + (1.0 - keep) * float(adv_std))


def exec_bonus(m, base, weight):
    """The executor's bonus of a decision for carrying out its mission (run.py --general-exec W; mission.py): units'
    measures m (0..1, any shape) against their kind's running mean base -> weight x (m - base): pays for doing it
    better than usual, not for easy missions."""
    return weight * (m - base)


def exec_advantage(bonus, done, gl):
    """The bonus's part of each unit's advantage (ppo.update): bonus [T, R, N] a decision, done [T, R] (float: the
    battle ended at t), gl = gamma x lambda of the executor's PPO -> (1 - gl) x sum_j gl^j bonus[t + j] within the
    battle and the chunk (a reward-to-go without a critic: the bonus never enters the reward, so neither the unit
    critic nor the general's return sees it). x (1 - gl): a bonus steady over the horizon counts once, W = 1 pays a
    unit one advantage std (the side's advantage is normalised) for a measure 1 above its kind's mean."""
    out = bonus * 0.0
    run = bonus[0] * 0.0
    for t in reversed(range(len(bonus))):
        run = bonus[t] + gl * (1.0 - done[t])[..., None] * run
        out[t] = run
    return (1.0 - gl) * out


def other_kind(draw, chosen):
    """A uniform draw over the K - 1 other kinds: draw in 0..K-2 -> the kind (chosen skipped)."""
    return draw + (draw >= chosen)


def policy_weight(p_chosen):
    """The weight of a price whose other kind came from the policy without the chosen one: 1 - pi(chosen)."""
    return 1.0 - p_chosen


def kind_advantage(price, weight, floor, clip, scale=None):
    """The kind choice's advantage of the branches (module doc): price x weight, divided by the root mean square of
    those over the chunk (at least floor) - or, given, by scale [E] (each row's kind's running scale, KindNorm; at
    least floor) -, within +-clip. price, weight [E]."""
    x = price * weight
    if scale is not None:
        return (x / scale.clip(floor, None)).clip(-clip, clip)
    rms = float((x * x).mean()) ** 0.5
    return (x / max(rms, floor)).clip(-clip, clip)


NORM_DECAY = 0.9    # KindNorm: the sums x this an update (~10 updates' memory)
NORM_PRIOR = 32.0   # ... a kind with fewer rows is filled up to this many with the update's mean square of all prices


class KindNorm:
    """Each mission kind's running scale of its prices (module doc). Plain lists: K is small, the host's tests run it."""

    def __init__(self, kinds, floor=0.005):
        self.K = int(kinds)
        self.floor = float(floor)
        self.sq = [0.0] * self.K
        self.n = [0.0] * self.K
        self.last = None

    def rms(self, shared_ms=0.0):
        """[K] each kind's root mean square now (budget share)."""
        out = []
        for s, n in zip(self.sq, self.n):
            fill = max(NORM_PRIOR - n, 0.0)
            out.append(max(math.sqrt((s + fill * shared_ms) / max(n + fill, 1e-12)), self.floor))
        return out

    def update(self, kinds, prices):
        """The update's priced rows: kinds [k] (ints 0..K-1) their chosen kind, prices [k] (budget share, either sign)
        -> [K] each kind's scale after adding them."""
        self.sq = [s * NORM_DECAY for s in self.sq]
        self.n = [n * NORM_DECAY for n in self.n]
        ms = 0.0
        for k, v in zip(kinds, prices):
            k, v = int(k), float(v)
            self.sq[k] += v * v
            self.n[k] += 1.0
            ms += v * v
        ms = ms / len(prices) if len(prices) else 0.0
        self.last = self.rms(ms)
        return self.last

    def scale(self):
        """[K] the scales of the last update (before any: the floor)."""
        return list(self.last) if self.last is not None else [self.floor] * self.K

    def state(self):
        return {"sq": list(self.sq), "n": list(self.n), "last": self.last}

    def load(self, d):
        """A saved state() (None, empty or of another K: fresh) -> loaded or not."""
        if not d or len(d.get("sq", ())) != self.K:
            return False
        self.sq, self.n = [float(x) for x in d["sq"]], [float(x) for x in d["n"]]
        self.last = None if d.get("last") is None else [float(x) for x in d["last"]]
        return True


class RunningMean:
    """The branches' running mean price (module doc). Plain floats: the host's tests run it."""

    def __init__(self, decay=NORM_DECAY):
        self.decay = float(decay)
        self.s = 0.0
        self.n = 0.0

    def mean(self):
        """The mean now (0 before any row)."""
        return self.s / self.n if self.n > 0 else 0.0

    def update(self, values):
        """The update's prices (any iterable of numbers) -> the mean after adding them (the sums decay first)."""
        self.s *= self.decay
        self.n *= self.decay
        for v in values:
            self.s += float(v)
            self.n += 1.0
        return self.mean()

    def state(self):
        return {"s": self.s, "n": self.n}

    def load(self, d):
        """A saved state() (None or empty: fresh) -> loaded or not."""
        if not d or "s" not in d:
            return False
        self.s, self.n = float(d["s"]), float(d["n"])
        return True


def outcome_due(elapsed, horizon, ended):
    """Whether a plan's outcome is known (module doc): elapsed seconds since the plan began, ended - the battle is over;
    numpy arrays and torch tensors alike (bool)."""
    return (elapsed >= horizon - 1e-6) | ended
