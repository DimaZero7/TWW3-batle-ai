"""How good each decision of a recorded side turned out - decided by the network's own critic, not by a rule of ours
(a rule would be a hidden teacher).

The advantage is the one PPO uses (tools/nn/train/ppo.py gae) on the recorded battle: the reward v2 of the side per
recorded second (the gold trade of the step, (enemy gold lost - own gold lost) / the armies' mean cost, the worst share
of a unit kept, a routing one ROUT_SHARE of what it has left; +-1 to the winner / loser at the end), the training's
gamma and lambda per 1 s decision (run.py: cadence.discount of 0.9997 / 0.95 per 0.5 s), up to the battle's end (no
bootstrap after it), minus V(s) of the critic on the side's own critic input. lam=1: the plain discounted return
minus V. The order change cost of v2 is left out: the changes of another player are read, not known (as the critic's
check, build/observe/critic/report.py).

window20 (a fallback when the critic is no good, run.py --observe-adv window20): the side's gold trade over the next
WINDOW_S seconds minus its mean over the battle's seconds; a decision counts as good when that is above 0 and the
trade itself too.
"""
import numpy as np

GAMMA = 0.9997 ** 2          # per 1 s decision (cadence.discount of PPOConfig.gamma per 0.5 s)
LAM = 0.95 ** 2
WINDOW_S = 20.0


def rewards(gold, winner, side):
    """[T] the reward of `side` for the step from second k to k + 1 (the last row 0: terminal). gold [T, 2] the gold lost
    so far by side 1 and 2 (share of the budget)."""
    T = len(gold)
    own, other = gold[:, side - 1], gold[:, 2 - side]
    r = np.zeros(T)
    if T < 2:
        return r
    r[:-1] = np.diff(other) - np.diff(own)
    r[-2] += 1.0 if winner == side else (-1.0 if winner in (1, 2) else 0.0)
    return r


def gae(r, v, gamma=GAMMA, lam=LAM):
    """(advantage [T], return [T]): the last row is terminal (its value and advantage 0)."""
    T = len(r)
    adv = np.zeros(T)
    run = 0.0
    for t in range(T - 2, -1, -1):
        nxt = v[t + 1] if t + 1 < T - 1 else 0.0
        delta = r[t] + gamma * nxt - v[t]
        run = delta + gamma * lam * run
        adv[t] = run
    ret = adv + np.asarray(v, float)
    ret[-1] = 0.0
    return adv, ret


def window(gold, t, side, seconds=WINDOW_S):
    """(trade [T]: the side's gold trade over [t, t + seconds), centred [T]: minus its mean over the battle)."""
    own, other = gold[:, side - 1], gold[:, 2 - side]
    T = len(t)
    end = np.searchsorted(t, np.asarray(t) + seconds, side="left").clip(max=T - 1)
    trade = (other[end] - other) - (own[end] - own)
    return trade, trade - trade.mean()
