"""The chained heads of v2 (ModelConfig.sectors > 0): each own unit's order is chosen part by part, every part knowing
the ones before it (autoregressive, as AlphaStar's action heads):

    kind (hold, move, attack, withdraw, keep)
      -> target: a pointer at a visible living enemy (for attack)
      -> place (for move and withdraw): a sector (the unit's query against the sector tokens, sectors.py), then one of
         its fine x fine cells (~25 m) given the sector chosen
      -> commitment: keep the new order 2 / 4 / 8 / 16 s (commit.py)
    run (move and attack) and the ability (as heads.py: independent of the order) beside them.
The target's logits also get a learned number from the pair features of the unit and the enemy (cfg.pairs, pairs.py:
reach, closing speed, bearing; 0 at the start).

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


CAP = 30.0     # every head's logits are soft-capped: CAP x tanh(x / CAP) (no change while |x| < ~10)


def _cap(x, bad):
    """(x soft-capped, bad + its non-finite values per row [B]): a finite logit stays within +-CAP, so no softmax,
    logsumexp or sum of the choice overflows however far the weights drift; a non-finite one is counted (the log's
    nan_fixed) and later taken as masked (_clean)."""
    nonfinite = (~torch.isfinite(x)).reshape(x.shape[0], -1).sum(-1).to(torch.float32)
    return CAP * torch.tanh(x / CAP), bad + nonfinite


def _clean(logits, temperature=1.0):
    """The logits as a sampler takes them: a non-finite one as masked (NEG), / temperature."""
    return torch.nan_to_num(logits, nan=NEG, posinf=-NEG, neginf=NEG) / max(temperature, 1e-6)


def _cat(logits, temperature=1.0):
    return Categorical(logits=_clean(logits, temperature), validate_args=False)


def _gumbel(shape, like):
    """Gumbel(0, 1) noise: -log(-log(u)), u uniform in (0, 1) kept off its ends (finite whatever u is drawn)."""
    u = torch.rand(shape, device=like.device, dtype=like.dtype).clamp(1e-20, 1.0 - 1e-7)
    return -torch.log(-torch.log(u))


def _pick(logits, greedy, temperature):
    """A choice of softmax(logits / temperature) per row: the most likely (greedy) or a sample by the Gumbel-max
    trick, argmax(logits / T + Gumbel noise) - the same distribution as Categorical(logits).sample(), but no softmax
    and no torch.multinomial in the graph. Compiled whole (rollout.fast), the sector head's fused logsumexp + online
    softmax handed torch.multinomial a row it refused ('probability tensor contains either inf, nan or element < 0',
    a device assert that killed three v2 runs at 8-40 min, 08.10.2026; the same rows ran clean eagerly for 50 min).
    A masked choice (NEG) never wins against an allowed one: the noise is at most ~46. A row with nothing allowed
    (all masked, or all non-finite) takes its first choice (the kind's: hold)."""
    x = _clean(logits, 1.0 if greedy else temperature)
    pick = x.argmax(-1) if greedy else (x + _gumbel(x.shape, x)).argmax(-1)
    return torch.where(x.max(-1).values > NEG / 2, pick, torch.zeros_like(pick))


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
        if cfg.pairs:     # the pair features (pairs.py) -> + the logit of attacking j; 0 at the start
            from tools.nn.model import pairs as pr
            self.target_pair = nn.Linear(pr.SIZE, 1, bias=False)
            nn.init.zeros_(self.target_pair.weight)

    def forward(self, x, s, cells_in, obs_t, action=None, greedy=False, temperature=1.0, abilities=True, pairs=None):
        """x [B, 1 + N, d] unit tokens, s [B, S, ds] sector tokens, cells_in [B, S, k2] (sectors.inside) -> (logits,
        action). logits: kind [B, N, 5], target [B, N, N], sector [B, N, S], fine [B, N, k2], commit [B, N, 4],
        run [B, N], ability [B, N, 1 + SLOTS] - each under the action's earlier parts. action None: sampled.
        pairs [B, 1 + N, 1 + N, pairs.SIZE]: the pair features (cfg.pairs), added to the target's logits."""
        cfg = self.cfg
        u = self.norm(x[:, 1:])
        ctrl, ok = obs_t["ctrl"], obs_t["target_ok"]
        free = obs_t.get("free", ctrl)
        # hold, move, attack, withdraw, keep (tools/nn/sim/orders.py KINDS order); a held unit only keeps
        allowed = torch.stack([~ctrl | free, free, free & ok.any(-1, keepdim=True), free, ctrl], -1)
        bad = torch.zeros(u.shape[0], device=u.device)
        kind, bad = _cap(self.kind(u), bad)
        kind = kind.masked_fill(~allowed, NEG)
        target = self.q(u) @ self.k(u).transpose(1, 2) / math.sqrt(cfg.pointer)
        if pairs is not None and getattr(self, "target_pair", None) is not None:
            target = target + self.target_pair(pairs[:, 1:, 1:])[..., 0].to(target.dtype)
        target, bad = _cap(target, bad)
        target = target.masked_fill(~ok[:, None, :], NEG)
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
        sector, bad = _cap(sector - nn.functional.softplus(self.place_near(c)) * dist / sc.step(cfg), bad)
        sector = sector.masked_fill(~sec_ok[:, None, :], NEG)
        if action is None:
            a_sector = _pick(sector, greedy, temperature)
        else:
            a_sector, a_fine_given = sc.split(cfg, action.point)
        idx = a_sector[..., None].expand(-1, -1, s.shape[-1])
        s_at = s.gather(1, idx)                                                   # [B, N, ds]
        fine, bad = _cap(self.fine(torch.cat([c, s_at], -1)), bad)
        fine_ok = cells_in.gather(1, a_sector[..., None].expand(-1, -1, cells_in.shape[-1]))
        fine = fine.masked_fill(~fine_ok, NEG)
        a_fine = _pick(fine, greedy, temperature) if action is None else a_fine_given
        commit, bad = _cap(self.commit(c), bad)
        run, bad = _cap(self.run(c)[..., 0], bad)
        out = {"kind": kind, "target": target, "sector": sector, "fine": fine, "commit": commit, "run": run}
        if action is None:
            a_commit = _pick(commit, greedy, temperature)
            # (a uniform draw against the probability: Bernoulli's own sample, without its kernel's checks)
            p_run = torch.sigmoid(_clean(run, temperature))
            a_run = (run > 0) if greedy else torch.rand_like(p_run) < p_run
        else:
            a_commit, a_run = action.commit, action.run
        a_ability = None
        if obs_t.get("abil") is not None:
            usable = obs_t["abil_ok"] & ctrl[..., None]
            abil = torch.where(usable[..., None], obs_t["abil"], torch.zeros_like(obs_t["abil"]))
            score, bad = _cap((self.ability_q(u)[:, :, None] * self.ability_k(abil)).sum(-1) / math.sqrt(cfg.pointer), bad)
            slot = torch.where(usable, score, torch.full_like(score, NEG))
            none, bad = _cap(self.ability_none(u), bad)
            out["ability"] = torch.cat([none, slot], -1)
            if action is None and abilities:
                a_ability = _pick(out["ability"], greedy, temperature) - 1
            elif action is not None:
                a_ability = action.ability
        point = sc.point_index(cfg, a_sector, a_fine)
        out["nan_fixed"] = bad                     # [B] non-finite logits of the row (taken as masked)
        return out, Action(a_kind, point, a_target, a_run, a_ability, a_commit)


def log_prob(logits, a, ctrl):
    """(log-probability [B, N], entropy [B, N]) of the parts that matter (module doc): the kind always (a held unit's
    only choice, keep, gives 0); the target for an attack; the sector and the cell for move and withdraw; run for move
    and attack; the commitment with a new order (not keep); the ability when the action has one. Zero where ctrl is
    off. The entropy: the kind's + each later part's weighted by the chance of a kind that uses it (a held unit: 0)."""
    logits = {k: torch.nan_to_num(v, nan=NEG) for k, v in logits.items()}     # (a non-finite logit: masked)
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
