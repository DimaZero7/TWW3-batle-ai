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
* ability: per own unit, "none" or one of its ability slots (a pointer, as the target): the unit's
  query against a key made from each slot's state and passport (tools/nn/model/abilities.py), so
  the choice follows the ability's description, not its slot or name, and a new ability needs no
  new weights. Only abilities in obs abil_ok (owned, self-cast, ready, the unit takes orders);
  "none" always. Independent of the order kind: a lord may move and use an ability at once.
  Sampled only when asked (sample(abilities=True)): code that does not know the head (training
  before it is wired in) gets Action.ability None, and log_prob leaves it out.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from tools.nn.model import abilities as ab
from tools.nn.model import observation as ob

from tools.nn.sim.orders import ATTACK, HOLD, KEEP, KINDS, MOVE, WITHDRAW  # noqa: F401  (the simulator's codes)
NEG = -1e9


@dataclass
class Action:
    kind: torch.Tensor     # [B, N] long
    point: torch.Tensor    # [B, N] long: bin
    target: torch.Tensor   # [B, N] long: unit index, -1 none
    run: torch.Tensor      # [B, N] bool
    ability: torch.Tensor = None   # [B, N] long: ability slot to use, -1 none; None: not chosen


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
        self.ability_q = nn.Linear(d, cfg.pointer)
        self.ability_k = nn.Sequential(nn.Linear(ab.SIZE, d), nn.GELU(), nn.Linear(d, cfg.pointer))
        self.ability_none = nn.Linear(d, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Ability features appended since the checkpoint load with zero weights (encoder.pad_inputs)."""
        from tools.nn.model.encoder import pad_inputs
        pad_inputs(state_dict, prefix + "ability_k.0.weight", self.ability_k[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, obs_t):
        """x [B, 1 + N, d] -> masked logits: kind [B, N, 5], point [B, N, P], target [B, N, N], run [B, N],
        ability [B, N, 1 + SLOTS] (0: none, 1 + k: slot k; when obs_t has abilities)."""
        u = self.norm(x[:, 1:])
        ctrl, ok = obs_t["ctrl"], obs_t["target_ok"]
        kind = self.kind(u)
        # hold, move, attack, withdraw, keep (tools/nn/sim/orders.py KINDS order)
        allowed = torch.stack([torch.ones_like(ctrl), ctrl, ctrl & ok.any(-1, keepdim=True), ctrl, ctrl], -1)
        kind = kind.masked_fill(~allowed, NEG)
        target = self.q(u) @ self.k(u).transpose(1, 2) / math.sqrt(self.cfg.pointer)
        target = target.masked_fill(~ok[:, None, :], NEG)
        out = {"kind": kind, "point": self.point(u), "target": target, "run": self.run(u)[..., 0]}
        if obs_t.get("abil") is not None:
            usable = obs_t["abil_ok"] & ctrl[..., None]                                   # [B, N, SLOTS]
            if torch.compiler.is_compiling():
                # Compiled (as AbilityEncoder): every slot is scored, those that may not be used masked.
                abil = torch.where(usable[..., None], obs_t["abil"], torch.zeros_like(obs_t["abil"]))
                score = (self.ability_q(u)[:, :, None] * self.ability_k(abil)).sum(-1)
                slot = torch.where(usable, score / math.sqrt(self.cfg.pointer), torch.full_like(score, NEG))
            else:
                # Only the slots that may be used are scored (a few of B x N x SLOTS); the rest stay masked.
                b, n, k = usable.nonzero(as_tuple=True)
                score = (self.ability_q(u)[b, n] * self.ability_k(obs_t["abil"][b, n, k])).sum(-1)
                slot = torch.full(obs_t["abil_ok"].shape, NEG, device=u.device, dtype=score.dtype)
                slot = slot.index_put((b, n, k), score / math.sqrt(self.cfg.pointer))
            out["ability"] = torch.cat([self.ability_none(u), slot], -1)
        return out


def _dists(logits, temperature=1.0):
    # validate_args=False: the argument checks read the GPU's answer back (a sync per distribution,
    # ~10 a decision); the logits are finite by construction (masked with NEG, not -inf)
    t = max(temperature, 1e-6)
    return (Categorical(logits=logits["kind"] / t, validate_args=False),
            Categorical(logits=logits["point"] / t, validate_args=False),
            Categorical(logits=logits["target"] / t, validate_args=False),
            Bernoulli(logits=logits["run"] / t, validate_args=False))


def sample(logits, greedy=False, temperature=1.0, abilities=False):
    """An Action from the logits (greedy: the most likely choice of each part). abilities: also
    choose the ability (Action.ability; else None)."""
    kind, point, target, run = _dists(logits, temperature)
    if greedy:
        a = Action(kind.probs.argmax(-1), point.probs.argmax(-1), target.probs.argmax(-1), logits["run"] > 0)
    else:
        a = Action(kind.sample(), point.sample(), target.sample(), run.sample() > 0.5)
    has_target = logits["target"].max(-1).values > NEG / 2
    a.target = torch.where((a.kind == ATTACK) & has_target, a.target, torch.full_like(a.target, -1))
    if abilities and "ability" in logits:
        d = Categorical(logits=logits["ability"] / max(temperature, 1e-6), validate_args=False)
        a.ability = (d.probs.argmax(-1) if greedy else d.sample()) - 1
    return a


def log_prob(logits, a, ctrl):
    """(log-probability [B, N], entropy [B, N]) of the parts of the action that matter.

    kind always (keep has nothing else); point for move and withdraw; target for attack; run for
    move and attack; the ability choice when the action has one (Action.ability not None). Zero for
    units that take no orders (ctrl false).
    """
    kind, point, target, run = _dists(logits)
    move, attack = a.kind == MOVE, a.kind == ATTACK
    lp = kind.log_prob(a.kind)
    lp = lp + torch.where(move | (a.kind == WITHDRAW), point.log_prob(a.point), torch.zeros_like(lp))
    lp = lp + torch.where(attack, target.log_prob(a.target.clamp(min=0)), torch.zeros_like(lp))
    lp = lp + torch.where(move | attack, run.log_prob(a.run.float()), torch.zeros_like(lp))
    ent = kind.entropy() + point.entropy() * (kind.probs[..., MOVE] + kind.probs[..., WITHDRAW]) + run.entropy()
    if a.ability is not None and "ability" in logits:
        d = Categorical(logits=logits["ability"], validate_args=False)
        lp = lp + d.log_prob(a.ability.clamp(min=-1) + 1)
        ent = ent + d.entropy()
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
