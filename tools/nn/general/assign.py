"""The learned group collection (GeneralConfig.groups "learned"; docs/ru/training/general.md "Сбор групп"): the
general is given only the number of groups K, the network puts every free unit of its side into one of K groups.

    a unit's key: its card after the council + its card before it (an MLP) + its role's embedding (groups.roles: the
      lord and heroes, fast, shooters, melee) + where it stands (pos); then one attention block among the side's free
      units (NEAR_LAYERS of them) with a learned bias a head by the pair's distance bin and whether they share a role
      (starts at - metres / 100: a unit hears its neighbours most - the rule's clusters are by place within a role,
      which a unit alone cannot see), layer-normed
    a group's query: its own learned card (places 0..G-1; a copy of the slot attention's starting cards at the start)
      + the embedding of K + (a re-plan) the last plan's card of that number (a layer from zero); one layer of listening
      to the side's free units (net.Listen: the queries see the army they divide), then `rounds` (3) rounds of slot
      attention among the K groups (the units split over them, softmax over the groups; each group takes the mean of
      its units' values): a clustering, the groups compete for the units
    logits [P, N, G] = query . key / sqrt(D) after the rounds; a re-plan: + a learned bonus (stick) on a unit's last
      group number

The membership is a sequential categorical choice with a capacity mask: the free units of the side in their slots'
order, each picks one of the K_eff = min(K, n) groups (n the side's free units) that is not full (at most
cap = ceil(hi x n / K_eff) units); when the units left are exactly as many as the groups still short of their minimum
(m = max(1, floor(lo x n / K_eff))) need, only those groups may be picked - so exactly K_eff groups are not empty and
every one is within [m, cap]. The logits do not depend on the earlier picks, only the mask does: the log-probability
of a membership given (the update) is computed for all units at once (the counts before each unit by a cumulative sum
of the one-hot membership), equal to the sum of the sequential steps'.

Only our side's free units take part; the enemy's groups stay the rule's (groups.py), as do the group cards
(net.SlotGroups.fixed) and everything after them: this module changes only which unit is in which group.
"""
import math

import torch
from torch import nn

from tools.nn.general.net import NEG, POS_FEATURES, POS_M, Listen, mlp
from tools.nn.model.encoder import Block

EPS = 1e-6
NEAR_BINS = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0)    # m: the near blocks' distance bins
NEAR_LAYERS = 2


def capacity(n, K, lo, hi):
    """(K_eff, m, cap) [P] long of n [P] free units and K [P] groups wanted: K_eff = min(K, n) (>= 0), the minimum
    m = max(1, floor(lo x n / K_eff)) and the maximum cap = ceil(hi x n / K_eff) units a group (0 where K_eff is 0)."""
    n = n.long()
    k_eff = torch.minimum(K.long().clamp(min=0), n)
    per = n.double() / k_eff.clamp(min=1).double()
    m = torch.floor(lo * per + EPS).long().clamp(min=1)
    cap = torch.ceil(hi * per - EPS).long().clamp(min=1)
    z = torch.zeros_like(n)
    on = k_eff > 0
    return k_eff, torch.where(on, m, z), torch.where(on, torch.maximum(cap, m), z)


def allowed(counts, left, k_eff, m, cap):
    """[.., G] bool: the groups a unit may pick with `counts` [.., G] units already in each group and `left` [..] units
    still to place (this one included): a group below K_eff that is not full; when left is no more than the units the
    groups short of m still need, only those. k_eff, m, cap: [..] (broadcast to counts' leading shape)."""
    G = counts.shape[-1]
    g = torch.arange(G, device=counts.device)
    live = g < k_eff[..., None]
    short = (m[..., None] - counts).clamp(min=0) * live
    need = short.sum(-1)
    ok = live & (counts < cap[..., None])
    force = left <= need
    return torch.where(force[..., None], ok & (short > 0), ok)


def sample(logits, who, k_eff, m, cap, greedy=False, gen=None):
    """The sequential choice (module doc): logits [P, N, G], who [P, N] the units that take part (in slot order),
    k_eff, m, cap [P] (capacity()). -> (a [P, N] each unit's group, -1 for the others; lp [P] the sum of the steps'
    log-probabilities; ent [P] the sum of their entropies). Sampled by Gumbel-max under the mask (one draw of noise
    [P, N, G] from torch's generator, or gen), or the most likely group each step (greedy)."""
    P, N, G = logits.shape
    dev = logits.device
    lg = logits.float()
    if greedy:
        noise = torch.zeros_like(lg)
    else:
        u = torch.rand(lg.shape, generator=gen, device=dev, dtype=lg.dtype)
        noise = -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-7)))
    counts = torch.zeros(P, G, dtype=torch.long, device=dev)
    left = who.long().sum(1)
    a = torch.full((P, N), -1, dtype=torch.long, device=dev)
    lp = torch.zeros(P, device=dev)
    ent = torch.zeros(P, device=dev)
    for n in range(N):
        on = who[:, n]
        ok = allowed(counts, left, k_eff, m, cap)
        ok = ok & on[:, None]
        anyok = ok.any(-1)
        x = torch.where(ok, lg[:, n], torch.full_like(lg[:, n], NEG))
        logp = torch.log_softmax(x, -1)
        pick = torch.where(ok, x + noise[:, n], torch.full_like(x, -math.inf)).argmax(-1)
        take = on & anyok
        a[:, n] = torch.where(take, pick, a[:, n])
        lp = lp + torch.where(take, logp.gather(1, pick[:, None])[:, 0], torch.zeros_like(lp))
        p = logp.exp() * ok
        ent = ent - torch.where(take, (p * torch.where(ok, logp, torch.zeros_like(logp))).sum(-1), torch.zeros_like(ent))
        counts = counts + (torch.nn.functional.one_hot(pick, G) * take[:, None]).long()
        left = left - on.long()
    return a, lp, ent


def log_prob(logits, who, k_eff, m, cap, a):
    """(lp [P], ent [P], ok [P, N, G]) of the membership a [P, N] given (sample()'s; -1 for the units not in `who`):
    the same steps' masks computed for all units at once (module doc), with gradients through logits."""
    P, N, G = logits.shape
    on = who & (a >= 0)
    oh = torch.nn.functional.one_hot(a.clamp(min=0), G).long() * on[..., None].long()        # [P, N, G]
    before = torch.cumsum(oh, 1) - oh
    w = who.long()
    left = torch.flip(torch.cumsum(torch.flip(w, [1]), 1), [1])                                # this one and after
    ok = allowed(before, left, k_eff[:, None].expand(P, N), m[:, None].expand(P, N), cap[:, None].expand(P, N))
    ok = ok & on[..., None]
    x = logits.float().masked_fill(~ok, NEG)
    logp = torch.log_softmax(x, -1)
    lp = (logp.gather(-1, a.clamp(min=0)[..., None])[..., 0] * on.float()).sum(1)
    p = logp.exp() * ok
    ent = -(p * torch.where(ok, logp, torch.zeros_like(logp))).sum(-1)
    return lp, (ent * on.float()).sum(1), ok


def groups_of(a, who, G):
    """[P, G] bool: the groups 0..G-1 that hold some unit of `who` [P, N] in the membership a [P, N] (-1 none)."""
    on = who & (a >= 0) & (a < G)
    return (torch.nn.functional.one_hot(a.clamp(0, G - 1), G).bool() & on[..., None]).any(1)


def keep_last(a, logits, who, prev_main, has_prev, keep, G):
    """GeneralConfig.groups_sticky: (a' [P, N], kept [P]) - the rows `keep` [P] (a re-plan by the timer) with a last
    plan (has_prev) and some unit of `who` [P, N] in one of its groups keep that plan's membership: each unit of who by
    its last group number (prev_main [P, N], 0..G-1; the units not in who - out of the fight - drop off, a group left
    empty goes); a unit of who without a last group (back in the fight) joins the kept group its logits [P, N, G] like
    most. The other rows keep a (the membership sampled). The numbers are the last plan's, so net.match_groups keeps
    them (and the kinds' inertia its groups)."""
    if prev_main is None or has_prev is None:
        return a, torch.zeros_like(keep)
    old = torch.where(who & (prev_main >= 0) & (prev_main < G), prev_main, torch.full_like(prev_main, -1))
    live = groups_of(old, who, G)                                                          # [P, G]
    kept = keep.bool() & has_prev.bool() & live.any(1)
    fill = logits.float().masked_fill(~live[:, None, :], NEG).argmax(-1)
    last = torch.where(old >= 0, old, torch.where(who, fill, torch.full_like(old, -1)))
    return torch.where(kept[:, None], last, a), kept


class Assign(nn.Module):
    """The learned membership's logits (module doc). d: the card's width; places: G; heads, ff: the listening layer's
    (the councils'); start [G, D]: the groups' starting query cards (the slot attention's first G); stick: the re-plan
    bonus's start (a learned value)."""

    def __init__(self, d, places, heads, ff, start=None, stick=1.0, roles=4, rounds=3):
        super().__init__()
        self.G = places
        self.unit = mlp(2 * d, 2 * d, d)
        self.role = nn.Embedding(roles, d)
        self.k_pos = nn.Linear(len(POS_FEATURES), d, bias=False)
        self.near = nn.ModuleList(Block(d, heads, ff) for _ in range(NEAR_LAYERS))
        # the near blocks' bias: a learned value a head, distance bin and "same role" (starts at - metres / 100)
        mid = torch.tensor([(a + b) / 2 for a, b in zip((0.0,) + NEAR_BINS, NEAR_BINS + (NEAR_BINS[-1] * 1.5,))])
        self.near_b = nn.Parameter((-(mid / 100.0))[None, None, :].repeat(heads, 2, 1).clone())
        self.norm_k = nn.LayerNorm(d)
        q0 = start.detach().clone() if start is not None else torch.randn(places, d) * 0.5
        self.query = nn.Parameter(q0)
        self.k_emb = nn.Embedding(places + 1, d)
        self.prev_in = nn.Linear(d, d)
        nn.init.zeros_(self.prev_in.weight)
        nn.init.zeros_(self.prev_in.bias)
        self.listen = Listen(d, heads, ff)
        self.rounds = rounds
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm_s = nn.LayerNorm(d)
        self.upd = mlp(2 * d, 2 * d, d)
        self.stick = nn.Parameter(torch.tensor(float(stick)))

    def forward(self, units, before, role, pos, K, who, prev=None, has_prev=None, prev_main=None):
        """logits [P, N, G] of units [P, N, D] after the council, before [P, N, D] (their cards before it), role [P, N]
        (groups.roles), pos [P, N, 7], K [P] (1..G), who [P, N] (the units that take part); a re-plan: prev [P, 2G, D]
        the last plan's cards in the groups' numbers, has_prev [P], prev_main [P, N] (-1 none)."""
        P, N, d = units.shape
        G = self.G
        key = self.unit(torch.cat([units, before], -1)) + self.role(role.clamp(0, self.role.num_embeddings - 1))
        key = key + self.k_pos(pos.to(key.dtype))
        bias = self.near_bias(pos, who, role).to(key.dtype)
        for block in self.near:
            key = block(key, bias)
        key = self.norm_k(key)
        q = self.query[None].expand(P, -1, -1) + self.k_emb(K.clamp(0, G))[:, None, :]
        if prev is not None and has_prev is not None:
            q = q + self.prev_in(prev[:, :G].to(q.dtype)) * has_prev.reshape(-1, 1, 1).to(q.dtype)
        ok = who | ~who.any(1, keepdim=True)                      # (a side without free units: any key, unused)
        q = self.listen(q.to(key.dtype), key, ok)
        # the slot attention's rounds (Locatello et al., 2020) among the K groups: every unit splits itself over the
        # groups (softmax over them: they compete), each group takes the weighted mean of its units' values
        live = torch.arange(G, device=units.device)[None] < K.clamp(1, G)[:, None]                 # [P, G]
        k_, v_ = self.k(key), self.v(key)
        w_on = who.to(key.dtype)[..., None]
        for _ in range(self.rounds):
            lg = torch.einsum("pgd,pnd->png", self.q(self.norm_s(q)), k_) / math.sqrt(d)
            attn = torch.softmax(lg.masked_fill(~live[:, None, :], NEG), -1) * w_on
            w = attn / (attn.sum(1, keepdim=True) + 1e-6)
            q = q + self.upd(torch.cat([q, torch.einsum("png,pnd->pgd", w, v_)], -1))
        logits = torch.einsum("pgd,pnd->png", self.q(self.norm_s(q)), k_) / math.sqrt(d)
        if prev_main is not None and has_prev is not None:
            last = (prev_main[..., None] == torch.arange(G, device=units.device)) & has_prev.reshape(-1, 1, 1)
            logits = logits + self.stick * last.to(logits.dtype)
        return logits

    def near_bias(self, pos, who, role):
        """[P, H, N, N] the units' attention bias among the side's free units: the learned near_b of the head, the pair's
        distance bin (NEAR_BINS, m: pos's forward and lateral x POS_M) and whether they share a role; -inf to the others
        (a unit always hears itself)."""
        xy = pos[..., :2].float() * POS_M
        d = (xy[:, :, None, :] - xy[:, None, :, :]).norm(dim=-1)                              # [P, N, N]
        bins = torch.bucketize(d, torch.tensor(NEAR_BINS, device=d.device))                    # 0..len(NEAR_BINS)
        same = (role[:, :, None] == role[:, None, :]).long()
        b = self.near_b[:, same, bins].permute(1, 0, 2, 3)                                    # [P, H, N, N]
        N = who.shape[1]
        ok = (who[:, None, :] & who[:, :, None]) | torch.eye(N, dtype=torch.bool, device=who.device)[None]
        return b.masked_fill(~ok[:, None], float("-inf"))
