"""The chained heads of v2 (ModelConfig.sectors > 0): each own unit's order is chosen part by part, every part knowing
the ones before it (autoregressive, as AlphaStar's action heads):

    kind (hold, move, attack, withdraw, keep)
      -> target: a pointer at a visible living enemy (for attack)
      -> place (for move and withdraw): a sector (the unit's query against the sector tokens, sectors.py), then one of
         its fine x fine cells (~25 m) given the sector chosen
      -> commitment: keep the new order 2 / 4 / 8 / 16 s (commit.py)
    run (move and attack) and the ability (as heads.py: independent of the order) beside them.

Every part after the kind reads the condition c = the unit's token + an embedding of the kind + the target's token
(an attack's; zero otherwise). With the kinds as they are only an attack has a target and only move / withdraw have a
place, so the place knows the kind (move or withdraw) and the target part is zero for it.

Masks: the kinds as heads.py's, and a unit held by its commitment (Obs "free" off, commit.py) may only keep; targets as
heads.py's; sectors and cells outside the map never (Obs "geo", sectors.inside). Fresh, the place prefers the sectors
near the unit (a learned per-unit weight x the distance in sectors, softplus 1 at the start, as the grid head).

Action.point is the cell of the (sectors x fine)^2 grid (sectors.point_index): its centre is the order's point
(heads.point_world). Sampling needs the parts in order, so the actor samples itself (ChainHeads.forward with no
action); given an action (training), the logits of every part are those under that action's earlier parts.
"""
import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from tools.nn.model import abilities as ab
from tools.nn.model import commit as cm
from tools.nn.model import sectors as sc
from tools.nn.model.heads import NEG, Action
from tools.nn.sim.orders import ATTACK, KEEP, KINDS, MOVE, WITHDRAW


def _cat(logits, temperature=1.0):
    # (a non-finite logit as masked: sampling never asserts on the GPU; it was never seen with finite inputs)
    logits = torch.nan_to_num(logits, nan=NEG, posinf=-NEG, neginf=NEG)
    return Categorical(logits=logits / max(temperature, 1e-6), validate_args=False)


def _pick(logits, greedy, temperature):
    return torch.nan_to_num(logits, nan=NEG).argmax(-1) if greedy else _cat(logits, temperature).sample()


class ChainHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, p, ds = cfg.d, cfg.pointer, cfg.sector_d
        self.cfg = cfg
        self.norm = nn.LayerNorm(d)
        self.kind = nn.Linear(d, len(KINDS))
        self.q, self.k = nn.Linear(d, p), nn.Linear(d, p)
        self.kind_emb = nn.Embedding(len(KINDS), d)
        self.target_emb = nn.Linear(d, d)
        self.cond = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.place_q = nn.Linear(d, ds)            # (the place's pointer: the sector tokens' width)
        self.place_k = nn.Linear(ds, ds)
        self.place_near = nn.Linear(d, 1)
        nn.init.zeros_(self.place_near.weight)
        nn.init.constant_(self.place_near.bias, math.log(math.e - 1))          # softplus = 1
        self.fine = nn.Sequential(nn.Linear(d + ds, d), nn.GELU(), nn.Linear(d, cfg.fine * cfg.fine))
        self.commit = nn.Linear(d, len(cm.DURATIONS))
        self.run = nn.Linear(d, 1)
        self.ability_q = nn.Linear(d, p)
        self.ability_k = nn.Sequential(nn.Linear(ab.SIZE, d), nn.GELU(), nn.Linear(d, p))
        self.ability_none = nn.Linear(d, 1)

    def forward(self, x, s, cells_in, obs_t, action=None, greedy=False, temperature=1.0, abilities=True):
        """x [B, 1 + N, d] unit tokens, s [B, S, ds] sector tokens, cells_in [B, S, k2] (sectors.inside) -> (logits,
        action). logits: kind [B, N, 5], target [B, N, N], sector [B, N, S], fine [B, N, k2], commit [B, N, 4],
        run [B, N], ability [B, N, 1 + SLOTS] - each under the action's earlier parts. action None: sampled."""
        cfg = self.cfg
        u = self.norm(x[:, 1:])
        ctrl, ok = obs_t["ctrl"], obs_t["target_ok"]
        free = obs_t.get("free", ctrl)
        # hold, move, attack, withdraw, keep (tools/nn/sim/orders.py KINDS order); a held unit only keeps
        allowed = torch.stack([~ctrl | free, free, free & ok.any(-1, keepdim=True), free, ctrl], -1)
        kind = self.kind(u).masked_fill(~allowed, NEG)
        target = (self.q(u) @ self.k(u).transpose(1, 2) / math.sqrt(cfg.pointer)).masked_fill(~ok[:, None, :], NEG)
        a_kind = _pick(kind, greedy, temperature) if action is None else action.kind
        has_target = target.max(-1).values > NEG / 2
        if action is None:
            a_target = torch.where((a_kind == ATTACK) & has_target, _pick(target, greedy, temperature),
                                   torch.full_like(a_kind, -1))
        else:
            a_target = action.target
        attack = (a_kind == ATTACK) & (a_target >= 0)
        tvec = u.gather(1, a_target.clamp(min=0)[..., None].expand(-1, -1, u.shape[-1])) * attack[..., None].to(u.dtype)
        c = self.cond(u + self.kind_emb(a_kind) + self.target_emb(tvec))
        # the place: a sector, then a fine cell of it
        sec_ok = cells_in.any(-1)                                                 # [B, S]
        sector = self.place_q(c) @ self.place_k(s).transpose(1, 2) / math.sqrt(cfg.sector_d)
        pos = obs_t["pos"].to(u.dtype) * sc.ob.POS
        dist = (pos[..., None, :] - sc.centres(cfg, u.device).to(u.dtype)).square().sum(-1).clamp(min=1e-6).sqrt()
        sector = sector - nn.functional.softplus(self.place_near(c)) * dist / sc.step(cfg)
        sector = sector.masked_fill(~sec_ok[:, None, :], NEG)
        if action is None:
            a_sector = _pick(sector, greedy, temperature)
        else:
            a_sector, a_fine_given = sc.split(cfg, action.point)
        idx = a_sector[..., None].expand(-1, -1, s.shape[-1])
        s_at = s.gather(1, idx)                                                   # [B, N, ds]
        fine = self.fine(torch.cat([c, s_at], -1))
        fine_ok = cells_in.gather(1, a_sector[..., None].expand(-1, -1, cells_in.shape[-1]))
        fine = fine.masked_fill(~fine_ok, NEG)
        a_fine = _pick(fine, greedy, temperature) if action is None else a_fine_given
        commit = self.commit(c)
        run = self.run(c)[..., 0]
        out = {"kind": kind, "target": target, "sector": sector, "fine": fine, "commit": commit, "run": run}
        if action is None:
            a_commit = _pick(commit, greedy, temperature)
            a_run = (run > 0) if greedy else Bernoulli(logits=run / max(temperature, 1e-6),
                                                       validate_args=False).sample() > 0.5
        else:
            a_commit, a_run = action.commit, action.run
        a_ability = None
        if obs_t.get("abil") is not None:
            usable = obs_t["abil_ok"] & ctrl[..., None]
            abil = torch.where(usable[..., None], obs_t["abil"], torch.zeros_like(obs_t["abil"]))
            score = (self.ability_q(u)[:, :, None] * self.ability_k(abil)).sum(-1) / math.sqrt(cfg.pointer)
            slot = torch.where(usable, score, torch.full_like(score, NEG))
            out["ability"] = torch.cat([self.ability_none(u), slot], -1)
            if action is None and abilities:
                a_ability = _pick(out["ability"], greedy, temperature) - 1
            elif action is not None:
                a_ability = action.ability
        point = sc.point_index(cfg, a_sector, a_fine)
        return out, Action(a_kind, point, a_target, a_run, a_ability, a_commit)


def log_prob(logits, a, ctrl):
    """(log-probability [B, N], entropy [B, N]) of the parts that matter (module doc): the kind always (a held unit's
    only choice, keep, gives 0); the target for an attack; the sector and the cell for move and withdraw; run for move
    and attack; the commitment with a new order (not keep); the ability when the action has one. Zero where ctrl is
    off. The entropy: the kind's + each later part's weighted by the chance of a kind that uses it (a held unit: 0)."""
    kind = Categorical(logits=logits["kind"], validate_args=False)
    move, attack = a.kind == MOVE, a.kind == ATTACK
    place = move | (a.kind == WITHDRAW)
    sector, fine = Categorical(logits=logits["sector"], validate_args=False), \
        Categorical(logits=logits["fine"], validate_args=False)
    target = Categorical(logits=logits["target"], validate_args=False)
    run = Bernoulli(logits=logits["run"], validate_args=False)
    commit = Categorical(logits=logits["commit"], validate_args=False)
    cfg_sector, cfg_fine = _split_like(logits, a.point)
    zero = torch.zeros_like(logits["run"])
    lp = kind.log_prob(a.kind)
    lp = lp + torch.where(attack, target.log_prob(a.target.clamp(min=0)), zero)
    lp = lp + torch.where(place, sector.log_prob(cfg_sector) + fine.log_prob(cfg_fine), zero)
    lp = lp + torch.where(move | attack, run.log_prob(a.run.float()), zero)
    if a.commit is not None:
        lp = lp + torch.where(a.kind != KEEP, commit.log_prob(a.commit), zero)
    p = kind.probs
    ent = (kind.entropy() + (sector.entropy() + fine.entropy()) * (p[..., MOVE] + p[..., WITHDRAW])
           + run.entropy() * (p[..., MOVE] + p[..., ATTACK]) + commit.entropy() * (1 - p[..., KEEP]))
    if a.ability is not None and "ability" in logits:
        d = Categorical(logits=logits["ability"], validate_args=False)
        lp = lp + d.log_prob(a.ability.clamp(min=-1) + 1)
        ent = ent + d.entropy()
    return torch.where(ctrl, lp, zero), torch.where(ctrl, ent, zero)


def _split_like(logits, point):
    """(sector, fine cell) of the action's point, from the logits' sizes (S sectors, k2 cells)."""
    S, k2 = logits["sector"].shape[-1], logits["fine"].shape[-1]
    n, k = math.isqrt(S), math.isqrt(k2)
    row, col = point // (n * k), point % (n * k)
    return (row // k) * n + col // k, (row % k) * k + col % k
