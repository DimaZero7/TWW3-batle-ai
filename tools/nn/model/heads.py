"""Heads per own unit: order kind, move point, attack target (a pointer at an enemy unit), run.

* kind: hold, move, attack, withdraw, keep. Keep = no new order, the one in force goes on (a unit
  with no order yet holds): it lets the network leave a unit alone instead of re-issuing its order
  every decision. Units that take no orders (routing, dead, enemy) may only hold; attack needs a
  visible living enemy.
* point (move, withdraw): one of n_dir x n_dist bins — a direction in the side's frame (0 = towards the enemy)
  and a distance from the unit (geometric, dist_min..dist_max). Bins, not a Gaussian: the choice
  may have several peaks ("left flank or right flank"), stays exact under int8, and argmax is
  deterministic. The point is clipped to the map.
* target: a pointer, as in AlphaStar: the unit's query against every enemy's key; only enemies
  that are visible and alive now.
* run: run or walk (for move and attack).
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from tools.nn.model import observation as ob

from tools.nn.sim.orders import ATTACK, HOLD, KEEP, KINDS, MOVE, WITHDRAW  # noqa: F401  (the simulator's codes)
NEG = -1e9


@dataclass
class Action:
    kind: torch.Tensor     # [B, N] long
    point: torch.Tensor    # [B, N] long: bin
    target: torch.Tensor   # [B, N] long: unit index, -1 none
    run: torch.Tensor      # [B, N] bool


class Heads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        self.norm = nn.LayerNorm(d)
        self.kind = nn.Linear(d, len(KINDS))
        self.point = nn.Linear(d, cfg.points)
        self.run = nn.Linear(d, 1)
        self.q = nn.Linear(d, cfg.pointer)
        self.k = nn.Linear(d, cfg.pointer)

    def forward(self, x, obs_t):
        """x [B, 1 + N, d] -> masked logits: kind [B, N, 4], point [B, N, P], target [B, N, N], run [B, N]."""
        u = self.norm(x[:, 1:])
        ctrl, ok = obs_t["ctrl"], obs_t["target_ok"]
        kind = self.kind(u)
        # hold, move, attack, withdraw, keep (tools/nn/sim/orders.py KINDS order)
        allowed = torch.stack([torch.ones_like(ctrl), ctrl, ctrl & ok.any(-1, keepdim=True), ctrl, ctrl], -1)
        kind = kind.masked_fill(~allowed, NEG)
        target = self.q(u) @ self.k(u).transpose(1, 2) / math.sqrt(self.cfg.pointer)
        target = target.masked_fill(~ok[:, None, :], NEG)
        return {"kind": kind, "point": self.point(u), "target": target, "run": self.run(u)[..., 0]}


def _dists(logits, temperature=1.0):
    t = max(temperature, 1e-6)
    return (Categorical(logits=logits["kind"] / t), Categorical(logits=logits["point"] / t),
            Categorical(logits=logits["target"] / t), Bernoulli(logits=logits["run"] / t))


def sample(logits, greedy=False, temperature=1.0):
    """An Action from the logits (greedy: the most likely choice of each part)."""
    kind, point, target, run = _dists(logits, temperature)
    if greedy:
        a = Action(kind.probs.argmax(-1), point.probs.argmax(-1), target.probs.argmax(-1), logits["run"] > 0)
    else:
        a = Action(kind.sample(), point.sample(), target.sample(), run.sample() > 0.5)
    has_target = logits["target"].max(-1).values > NEG / 2
    a.target = torch.where((a.kind == ATTACK) & has_target, a.target, torch.full_like(a.target, -1))
    return a


def log_prob(logits, a, ctrl):
    """(log-probability [B, N], entropy [B, N]) of the parts of the action that matter.

    kind always (keep has nothing else); point for move and withdraw; target for attack; run for
    move and attack. Zero for units that take no orders (ctrl false).
    """
    kind, point, target, run = _dists(logits)
    move, attack = a.kind == MOVE, a.kind == ATTACK
    lp = kind.log_prob(a.kind)
    lp = lp + torch.where(move | (a.kind == WITHDRAW), point.log_prob(a.point), torch.zeros_like(lp))
    lp = lp + torch.where(attack, target.log_prob(a.target.clamp(min=0)), torch.zeros_like(lp))
    lp = lp + torch.where(move | attack, run.log_prob(a.run.float()), torch.zeros_like(lp))
    ent = kind.entropy() + point.entropy() * (kind.probs[..., MOVE] + kind.probs[..., WITHDRAW]) + run.entropy()
    zero = torch.zeros_like(lp)
    return torch.where(ctrl, lp, zero), torch.where(ctrl, ent, zero)


def point_offsets(cfg, device=None):
    """(forward, lateral) offset in metres of every point bin: [P, 2]."""
    ang = torch.arange(cfg.n_dir, device=device) * (2 * math.pi / cfg.n_dir)
    dist = torch.logspace(math.log10(cfg.dist_min), math.log10(cfg.dist_max), cfg.n_dist, device=device)
    f = torch.cos(ang)[:, None] * dist[None, :]
    l = torch.sin(ang)[:, None] * dist[None, :]
    return torch.stack([f.reshape(-1), l.reshape(-1)], -1)


def point_world(cfg, a, obs_t, frame, bounds):
    """World (x, z) [B, N, 2] of the move point: the unit's position + the bin's offset, inside the map."""
    off = point_offsets(cfg, a.point.device)[a.point]                     # [B, N, 2]
    pos = obs_t["pos"] * ob.POS + off
    x, z = frame.world(pos[..., 0], pos[..., 1])
    x = torch.minimum(torch.maximum(x, bounds[:, 0:1]), bounds[:, 1:2])
    z = torch.minimum(torch.maximum(z, bounds[:, 2:3]), bounds[:, 3:4])
    return torch.stack([x, z], -1)
