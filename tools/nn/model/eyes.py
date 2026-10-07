"""The eyes (v2 with ModelConfig.eyes): auxiliary heads that learn what is coming from the simulator's truth, and
whose predictions go back into the network before its decision layers (a light concept bottleneck).

Heads (after the memory, before the last attention block and the order heads; policy.ActorV2._finish):
* own10, own30 - every own unit: the share of its health it will lose in the next 10 s / 30 s (0..1, sigmoid);
* threat30 - every enemy unit (seen now or remembered): the gold of our health it will destroy in the next 30 s, as a
  share of our army's starting cost x THREAT_SCALE (softplus);
* danger10 - every sector (sectors.py): our units standing in it now - the sum of the health shares they will lose in
  the next 10 s (softplus; from the sector token and the sums of the own and the enemy unit tokens standing there).
Back into the network: each prediction through a small linear layer, added to its own unit's / enemy unit's token
and to the sector token (the place head reads the sector tokens). The predictions are DETACHED there: the eyes learn
only from the truth, so PPO cannot bend what they mean into a private code; the trunk under them gets both
gradients (PPO through the decision, the eyes' loss through the heads).

Truth (tools/nn/train/rollout.py): at every decision the learner rows' unit health share (`hp`) and gold of enemy
health destroyed so far (`gold_out`, the simulator's bookkeeping) before the decision's simulator steps and after
them (before a finished battle restarts); targets() turns them into the changes over the next k decisions within the
rollout's chunk. A battle that ends within the window: its end state (nothing more after it); no end and the window
past the chunk's last decision: masked. The loss (loss()): squared error, a mean over the valid units (the danger:
over the sectors, divided by the row's own units), the four heads summed, x the PPO's eyes weight (ppo.PPOConfig.eyes).
"""
import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.model import sectors

HORIZONS_S = (10.0, 30.0)
THREAT_SCALE = 10.0
HEADS = ("own10", "own30", "threat30", "danger10")
PRED_KEYS = ("eyes_own", "eyes_threat", "eyes_danger")


class Eyes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, ds = cfg.d, cfg.sector_d
        self.cfg = cfg
        self.norm = nn.LayerNorm(d)
        self.own = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.threat = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.unit_proj = nn.Linear(d, 16)
        self.danger = nn.Sequential(nn.Linear(ds + 32, ds), nn.GELU(), nn.Linear(ds, 1))
        self.own_in = nn.Linear(2, d)
        self.threat_in = nn.Linear(1, d)
        self.danger_in = nn.Linear(1, ds)
        # fresh, every prediction starts near 0 (sigmoid / softplus of -6: 0.0025), as almost every target is: at 0.5 /
        # 0.69 the first updates' error (~23, mostly the 256 sectors) took the eyes' gradient 1000 x the policy's
        for head in (self.own, self.threat, self.danger):
            nn.init.constant_(head[-1].bias, -6.0)

    def forward(self, x, s, obs_t):
        """x [B, 1 + N, d], s [B, S, ds] -> (x, s with the predictions added, {eyes_own [B, N, 2], eyes_threat [B, N],
        eyes_danger [B, S]})."""
        u = self.norm(x[:, 1:])
        own_p = torch.sigmoid(self.own(u))
        threat_p = F.softplus(self.threat(u))[..., 0]
        own = (obs_t["own"] & obs_t["attend"])[..., None].to(u.dtype)
        enemy = (~obs_t["own"] & obs_t["attend"])[..., None].to(u.dtype)
        idx, on = sectors.where(self.cfg, obs_t)
        one = sectors.one_hot(self.cfg, idx, u.dtype) * on[..., None].to(u.dtype)       # [B, N, S]
        p = self.unit_proj(u)
        agg = torch.cat([torch.einsum("bns,bnk->bsk", one, p * own), torch.einsum("bns,bnk->bsk", one, p * enemy)], -1)
        danger_p = F.softplus(self.danger(torch.cat([s, agg], -1)))[..., 0]
        back = self.own_in(own_p.detach()) * own + self.threat_in(threat_p.detach()[..., None]) * enemy
        x = x + F.pad(back, (0, 0, 1, 0))
        s = s + self.danger_in(danger_p.detach()[..., None])
        return x, s, {"eyes_own": own_p, "eyes_threat": threat_p, "eyes_danger": danger_p}


def truth(u, rows):
    """The truth of the learner rows now: {"hp": [R, N] health share, "gold": [R, N] gold of enemy health destroyed so
    far} (u: the simulator's state; rows: each row's battle)."""
    return {"hp": u["hp"][rows].clone(), "gold": u["gold_out"][rows].clone()}


def own_cost(u, rows, sides):
    """[R] the row's own army's starting cost (gold)."""
    side = u["side"][rows]
    return (u["cost"][rows] * (side == sides[:, None]).float()).sum(1).clamp(min=1.0)


def _until(done):
    """[T, R] the first decision j >= t at which the battle ended (T: not within the chunk)."""
    T = done.shape[0]
    out = torch.empty(done.shape, dtype=torch.long, device=done.device)
    running = torch.full(done.shape[1:], T, dtype=torch.long, device=done.device)
    for t in reversed(range(T)):
        running = torch.where(done[t], torch.full_like(running, t), running)
        out[t] = running
    return out


def change(before, after, done, k):
    """(after - before over the next k decisions [T, R, N], valid [T, R]): before[t] at decision t's observation,
    after[t] after its simulator steps (before a restart). The change from t is after[min(t + k - 1, end)] -
    before[t]; valid when that index is inside the chunk."""
    T = done.shape[0]
    t = torch.arange(T, device=done.device)[:, None]
    j = torch.minimum(t + k - 1, _until(done))
    ok = j <= T - 1
    at = after.gather(0, j.clamp(max=T - 1)[..., None].expand_as(after))
    return at - before, ok


def targets(before, after, done, cost, decision_s):
    """The eyes' targets of a chunk: before / after: truth() dicts of [T, R, N], done [T, R], cost [T, R] (own_cost)
    -> {"own": [T, R, N, 2] health share lost in 10 / 30 s, "threat": [T, R, N], "ok": [T, R, 2] valid 10 / 30 s}."""
    k10, k30 = (max(1, round(h / decision_s)) for h in HORIZONS_S)
    hp10, ok10 = change(before["hp"], after["hp"], done, k10)
    hp30, ok30 = change(before["hp"], after["hp"], done, k30)
    gold30, _ = change(before["gold"], after["gold"], done, k30)
    return {"own": torch.stack([(-hp10).clamp(0, 1), (-hp30).clamp(0, 1)], -1),
            "threat": (gold30 / cost[..., None]).clamp(min=0) * THREAT_SCALE, "ok": torch.stack([ok10, ok30], -1)}


def _mean_var(err, y, m):
    m = m.to(err.dtype)
    n = m.sum().clamp(min=1)
    mse = (err ** 2 * m).sum() / n
    mu = (y * m).sum() / n
    var = (((y - mu) ** 2) * m).sum() / n
    return mse, var


def loss(cfg, logits, tg, obs):
    """(the eyes' loss: the four heads' squared errors summed, {head: (mse, reference)}) on flattened rows (reference:
    the target's variance; the danger's: the error of predicting 0):
    logits [M, ...], tg: targets() rows [M, ...], obs [M, ...]."""
    own = obs["own"] & obs["attend"]
    enemy = ~obs["own"] & obs["attend"]
    ok10, ok30 = tg["ok"][..., 0], tg["ok"][..., 1]
    p, y = logits["eyes_own"], tg["own"]
    out = {"own10": _mean_var(p[..., 0] - y[..., 0], y[..., 0], own & ok10[:, None]),
           "own30": _mean_var(p[..., 1] - y[..., 1], y[..., 1], own & ok30[:, None]),
           "threat30": _mean_var(logits["eyes_threat"] - tg["threat"], tg["threat"], enemy & ok30[:, None])}
    idx, on = sectors.where(cfg, obs)
    one = sectors.one_hot(cfg, idx, y.dtype) * (own & on)[..., None].to(y.dtype)          # [M, N, S]
    yd = torch.einsum("mns,mn->ms", one, y[..., 0])
    n_own = own.sum(-1).clamp(min=1).to(y.dtype)
    row = ok10.to(y.dtype)
    err = ((logits["eyes_danger"] - yd) ** 2).sum(-1) / n_own
    mse = (err * row).sum() / row.sum().clamp(min=1)
    # (the danger's reference: the error of predicting 0 everywhere, most sectors have none of our units)
    var = (((yd ** 2).sum(-1) / n_own) * row).sum() / row.sum().clamp(min=1)
    out["danger10"] = (mse, var)
    return sum(v[0] for v in out.values()), out
