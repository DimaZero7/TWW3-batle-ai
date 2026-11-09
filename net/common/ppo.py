"""PPO's three pieces the levels share (the old train/ppo.py's, unchanged): the advantages, the clipped loss, the
critic's quality."""
import torch


def gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages and returns [T, R]. dones[t] [R]: the battle ended at step t (no bootstrap past it)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        go_on = 1.0 - dones[t].float()
        nxt = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * go_on - values[t]
        running = delta + gamma * lam * go_on * running
        adv[t] = running
    return adv, adv + values


def policy_loss(lp_new, lp_old, adv, mask, clip):
    """Clipped surrogate per unit. lp_* [S, N], adv [S] (the side's) or [S, N] (each unit's: the side's + its own
    mission bonus, --general-exec), mask [S, N] -> (loss, clip share)."""
    ratio = torch.exp(lp_new - lp_old)
    a = adv[:, None] if adv.dim() == 1 else adv
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    m = mask.float()
    n = m.sum().clamp(min=1)
    clipped = (((ratio - 1).abs() > clip).float() * m).sum() / n
    return -(surrogate * m).sum() / n, clipped


def explained_variance(value, ret):
    """1 - Var(return - value) / Var(return): 1 a perfect critic, 0 no better than the mean."""
    v = ret.var()
    return float(1 - (ret - value).var() / v) if v > 1e-12 else 0.0
