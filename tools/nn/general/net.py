"""The general's network (S28 skeleton): one plan from the armies and the map (design/general/index.html, the blocks in
order; docs/en/training/general.md).

    army slots -> ArmyReader -> unit cards (+ effect cards: 0, simplification 2)          "Отряды с эффектами"
    tiles (sectors, 8 numbers) -> a small net + the tile's place                         "Участки карты" (simpl. 1)
    council: units + tiles, each asks each (transformer blocks, label own / enemy / tile) "Совещание"
      -> tiles after the council ("Карта после совещания"), units after it ("Армии после совещания")
    group collection: 10 places a side, slot attention, 3 rounds; a unit splits itself     "Сбор групп"
      100 % between its side's places (side mask); the dead and the routing take no part;
      the keys less a learned share of their side's mean (GeneralConfig.center)
      -> group cards and shares ("Группы"); a place is a group when it is some unit's main place
    group council: the groups listen to the tiles and to all groups (only groups change)   "Совещание групп"
    missions, step by step: kind (10 learned cards) -> target 1 -> target 2 or none        "Выбор заданий"
      targets: our groups (not itself), their groups, tiles; only our groups choose
    deployment place (the first plan only): a pointer to a tile of our zone                "Выбор мест расстановки"
    plan value: the critic of the plan, on the plan's cards (detached: its error does not  "Оценка замысла"
      move the rest)

Every choice is sampled (or given: the update's log-probabilities under the choices made). Per own group: its
log-probability (the sum of its steps') and entropy; groups that are not groups take no part.

Input dict (rows P; N the battle's slots, S tiles, G places a side):
    num [P, N, 45], lab [P, N, 3]   the army slots (units.py)
    on [P, N]                       the unit sits at the council (present, alive, not known dead)
    own [P, N]                      the general's side
    free [P, N]                     may join a group (on, not routing)
    tiles [P, S, 8], tile_ok [P, S] the tiles and those inside the map
    zone [P, S], deploy [P]         the deployment's tiles and whether the plan deploys
    prev [P, 2G, D], has_prev [P]   the places' cards of the last plan (the inertia of a re-plan)
    main [P, N] (optional)          the plan's membership when it is computed again (locked: the update, the refresh)
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.general.army_reader import ArmyReader
from tools.nn.model.encoder import Block

NEG = -1e9
INPUTS = ("num", "lab", "on", "own", "free", "tiles", "tile_ok", "zone", "deploy", "prev", "has_prev")
CHOICES = ("kind", "t1", "t2", "place")


def mlp(i, h, o):
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def key_bias(ok, heads):
    """[P, H, 1, L] attention bias: 0 where the key may be looked at, -inf elsewhere."""
    b = torch.zeros(ok.shape, dtype=torch.float32, device=ok.device).masked_fill(~ok, float("-inf"))
    return b[:, None, None, :].expand(-1, heads, -1, -1)


class SlotGroups(nn.Module):
    """"Сбор групп": slot attention (Locatello et al., 2020) over the units after the council. 2G places (ours first,
    then the enemy's), each with a learned starting card (one network for all places and both sides; the places differ
    by their starting cards) or, in a re-plan, its card of the last plan. Each round: every free unit splits itself over
    its side's places (softmax over the places: they compete), each place takes the weighted mean of its units' values,
    a GRU and a feed-forward update the place. -> (shares [P, N, 2G] of the last round, place cards [P, 2G, D])."""

    def __init__(self, d, places, rounds, center=0.0):
        super().__init__()
        self.G, self.rounds = places, rounds
        # the side's mean key taken off the keys, this learned share of it (GeneralConfig.center; None: plain)
        self.center = nn.Parameter(torch.tensor(float(center))) if center else None
        self.start = nn.Parameter(torch.randn(2 * places, d) * 0.5)
        self.norm_in, self.norm_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.k, self.v, self.q = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(d, d)
        self.ff = mlp(d, 2 * d, d)

    def side_ok(self, own):
        """[P, N, 2G]: an own unit may join only our places, an enemy unit only the enemy's (the side mask)."""
        G = self.G
        return torch.cat([own[..., None].expand(*own.shape, G), (~own)[..., None].expand(*own.shape, G)], -1)

    def forward(self, u, own, free, prev=None, has_prev=None):
        P, N, d = u.shape
        slots = self.start[None].expand(P, -1, -1)
        if prev is not None:
            slots = torch.where(has_prev.reshape(-1, 1, 1), prev.to(slots.dtype), slots)
        x = self.norm_in(u)
        k, v = self.k(x), self.v(x)
        if self.center is not None:
            # each side's mean key over its free units, a learned share of it taken off (GeneralConfig.center)
            mean = []
            for side in (own, ~own):
                w = (free & side).to(k.dtype)[..., None]
                mean.append((k * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0))
            k = k - self.center * torch.where(own[..., None], mean[0], mean[1])
        ok = self.side_ok(own)
        keep = free[..., None].to(u.dtype)
        attn = None
        for _ in range(self.rounds):
            q = self.q(self.norm_s(slots))
            logits = torch.einsum("pnd,pgd->png", k, q) / math.sqrt(d)
            attn = torch.softmax(logits.masked_fill(~ok, NEG), -1) * keep
            w = attn / (attn.sum(1, keepdim=True) + 1e-6)
            upd = torch.einsum("png,pnd->pgd", w, v)
            slots = self.gru(upd.reshape(-1, d), slots.reshape(-1, d)).reshape(P, -1, d)
            slots = slots + self.ff(slots)
        return attn, slots


class Listen(nn.Module):
    """"Совещание групп", one layer: the groups (queries) listen to the keys (tiles and all groups), pre-norm attention
    and feed-forward; the keys do not change."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.nq, self.nk, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.out = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, ff * d), nn.Linear(ff * d, d)

    def forward(self, g, keys, key_ok):
        P, Q, d = g.shape
        L = keys.shape[1]
        q = self.q(self.nq(g)).reshape(P, Q, self.h, -1).transpose(1, 2)
        k, v = self.kv(self.nk(keys)).reshape(P, L, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias(key_ok, self.h).to(q.dtype))
        g = g + self.out(a.transpose(1, 2).reshape(P, Q, d))
        return g + self.f2(F.gelu(self.f1(self.n2(g))))


def masked_mean(x, m):
    w = m.to(x.dtype)[..., None]
    return (x * w).sum(-2) / w.sum(-2).clamp(min=1.0)


def categorical(logits, ok, given=None, greedy=False):
    """(choice, log-probability, entropy) of one step: logits [.., C] masked by ok [.., C] (a row with nothing allowed
    gives choice 0, log-probability 0 - its group or plan does not use the step), the choice given or sampled."""
    any_ok = ok.any(-1)
    lg = logits.masked_fill(~ok, NEG)
    lg = torch.where(any_ok[..., None], lg, torch.zeros_like(lg))
    logp = torch.log_softmax(lg.float(), -1)
    if given is None:
        if greedy:
            choice = logp.argmax(-1)
        else:
            choice = torch.distributions.Categorical(logits=logp, validate_args=False).sample()
    else:
        choice = given.clamp(min=0)
    lp = logp.gather(-1, choice[..., None])[..., 0]
    p = logp.exp()
    ent = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
    zero = torch.zeros_like(lp)
    return choice, torch.where(any_ok, lp, zero), torch.where(any_ok, ent, zero)


class General(nn.Module):
    """The general's network (module doc). cfg: GeneralConfig; sectors: the unit network's sectors a side (tiles =
    sectors^2, simplification 1)."""

    def __init__(self, cfg, sectors=16):
        super().__init__()
        from tools.nn.general import army_input as ai
        from tools.nn.general.tiles import FEATURES
        self.cfg = cfg
        d, G, K = cfg.card, cfg.places, cfg.kinds
        self.S = sectors * sectors
        self.reader = ArmyReader(cfg.reader_hidden, d, numbers=ai.SIZE)
        self.label = nn.Embedding(3, d)                         # own unit, enemy unit, tile
        self.tile_in = nn.Sequential(nn.Linear(len(FEATURES), d), nn.GELU(), nn.Linear(d, d))
        self.tile_place = nn.Parameter(torch.randn(self.S, d) * 0.1)
        self.norm_in = nn.LayerNorm(d)
        self.council = nn.ModuleList(Block(d, cfg.council_heads, cfg.ff) for _ in range(cfg.council_layers))
        self.norm_out = nn.LayerNorm(d)
        self.groups = SlotGroups(d, G, cfg.rounds, cfg.center)
        self.listen = nn.ModuleList(Listen(d, cfg.council_heads, cfg.ff) for _ in range(cfg.group_layers))
        self.norm_g = nn.LayerNorm(d)
        self.kinds = nn.Parameter(torch.randn(K, d) / math.sqrt(d) * 4)
        self.kind_q = nn.Linear(d, d)
        self.t1_q = mlp(2 * d, 2 * d, d)
        self.t2_q = mlp(3 * d, 2 * d, d)
        self.cand_k = nn.Linear(d, d)
        self.none = nn.Linear(d, 1)
        self.place_q = mlp(2 * d, 2 * d, d)
        self.place_k = nn.Linear(d, d)
        self.value = mlp(4 * d + 1, d, 1)

    # --- the cards ---------------------------------------------------------------------------------------------
    def cards(self, inp):
        """Everything before the choices: {"units", "tiles" (after the council), "shares" [P, N, 2G], "places"
        [P, 2G, D] (the places' cards: the next plan's inertia), "groups" [P, 2G, D] (after the group council and its
        norm), "main" [P, N] (a free unit's main place, -1 none), "active" [P, 2G] (some unit's main place)}."""
        P, N = inp["on"].shape
        G = self.cfg.places
        on, own = inp["on"], inp["own"]
        u = self.reader(inp["num"], inp["lab"], ~on)
        u = u + self.label((~own).long()) * on[..., None].to(u.dtype)
        t = self.tile_in(inp["tiles"].to(u.dtype)) + self.tile_place + self.label.weight[2]
        x = self.norm_in(torch.cat([u, t], 1))
        ok = torch.cat([on, inp["tile_ok"]], 1)
        bias = key_bias(ok, self.cfg.council_heads).to(x.dtype)
        for block in self.council:
            x = block(x, bias)
        x = self.norm_out(x)
        units, tiles = x[:, :N], x[:, N:]
        shares, places = self.groups(units, own, inp["free"], inp.get("prev"), inp.get("has_prev"))
        free = inp["free"]
        if inp.get("main") is not None:
            # the plan's own membership, locked (a re-computation of a plan made: the update, the refresh); the
            # shares and cards are computed again (the gradient's path)
            main = inp["main"]
        else:
            main = torch.where(free, shares.argmax(-1), torch.full_like(free, -1, dtype=torch.long))
        hit = main[..., None] == torch.arange(2 * G, device=main.device)
        active = hit.any(1)
        keys = torch.cat([tiles, places], 1)
        key_ok = torch.cat([inp["tile_ok"], active], 1)
        g = places
        for layer in self.listen:
            g = layer(g, keys, key_ok)
        return {"units": units, "tiles": tiles, "shares": shares, "places": places, "groups": self.norm_g(g),
                "main": main, "active": active}

    # --- the choices -------------------------------------------------------------------------------------------
    def decide(self, inp, c, act=None, greedy=False):
        """The missions and places of our groups on the cards c (cards()): sampled, or under act {"kind", "t1", "t2",
        "place"} [P, G]. -> dict: the choices [P, G], "lp" [P, G] and "ent" [P, G] (the group's steps summed; 0 for a
        place that is not a group), "value" [P], "lp_kind" [P, G] (the kind step's alone), "kind_logp" [P, G, K]
        (every kind's log-probability; uniform for a place that is not a group). Targets: index 0..G-1 our groups, G..2G-1 theirs, 2G.. the tiles,
        2G + S "none" (target 2 only)."""
        G, S = self.cfg.places, self.S
        g, tiles, active = c["groups"], c["tiles"], c["active"]
        P, d = g.shape[0], g.shape[-1]
        mine = active[:, :G]                                                  # [P, G] our groups choose
        act = act or {}
        k_logits = self.kind_q(g[:, :G]) @ self.kinds.t() / math.sqrt(d)      # [P, G, K]
        k_ok = mine[..., None].expand(-1, -1, self.cfg.kinds)
        kind, lp_k, ent_k = categorical(k_logits, k_ok, act.get("kind"), greedy)
        # every kind's log-probability (the mission branches: the old log-probability, the alternatives' sampling)
        k_lg = k_logits.masked_fill(~k_ok, NEG)
        kind_logp = torch.log_softmax(torch.where(k_ok.any(-1, keepdim=True), k_lg, torch.zeros_like(k_lg)).float(), -1)
        kc = self.kinds[kind]                                                 # [P, G, D]
        cand = torch.cat([g, tiles], 1)                                       # [P, 2G + S, D]
        keys = self.cand_k(cand)
        cand_ok = torch.cat([active, inp["tile_ok"]], 1)                      # [P, C]
        self_ = torch.eye(G, 2 * G + S, dtype=torch.bool, device=g.device)    # a group is not its own target
        ok1 = cand_ok[:, None, :] & ~self_[None] & mine[..., None]
        q1 = self.t1_q(torch.cat([g[:, :G], kc], -1))
        t1, lp_1, ent_1 = categorical(q1 @ keys.transpose(1, 2) / math.sqrt(d), ok1, act.get("t1"), greedy)
        c1 = cand.gather(1, t1[..., None].expand(-1, -1, d))                  # [P, G, D]
        q2 = self.t2_q(torch.cat([g[:, :G], kc, c1], -1))
        l2 = torch.cat([q2 @ keys.transpose(1, 2) / math.sqrt(d), self.none(q2)], -1)
        not_t1 = t1[..., None] != torch.arange(2 * G + S + 1, device=g.device)
        ok2 = torch.cat([ok1, mine[..., None]], -1) & not_t1
        t2, lp_2, ent_2 = categorical(l2, ok2, act.get("t2"), greedy)
        deploy = inp["deploy"].reshape(-1, 1)
        okp = inp["zone"][:, None, :] & mine[..., None] & deploy[..., None]   # [P, G, S]
        qp = self.place_q(torch.cat([g[:, :G], kc], -1))
        place, lp_p, ent_p = categorical(qp @ self.place_k(tiles).transpose(1, 2) / math.sqrt(d), okp,
                                         act.get("place"), greedy)
        place = torch.where(okp.any(-1), place, torch.full_like(place, -1))
        m = mine.to(lp_k.dtype)
        lp = (lp_k + lp_1 + lp_2 + lp_p) * m
        ent = (ent_k + ent_1 + ent_2 + ent_p) * m
        # the plan's value on its cards, detached (its error trains only the value head)
        summary = torch.cat([masked_mean(g[:, :G].detach(), mine), masked_mean(g[:, G:].detach(), active[:, G:]),
                             masked_mean(tiles.detach(), inp["tile_ok"]), masked_mean(kc.detach(), mine),
                             inp["deploy"].to(g.dtype)[:, None]], -1)
        value = self.value(summary)[..., 0]
        return {"kind": kind, "t1": t1, "t2": t2, "place": place, "lp": lp, "ent": ent, "value": value,
                "ent_parts": torch.stack([ent_k, ent_1, ent_2, ent_p], -1) * m[..., None],
                "lp_kind": lp_k * m, "kind_logp": kind_logp}

    def forward(self, inp, act=None, greedy=False):
        c = self.cards(inp)
        return c, self.decide(inp, c, act, greedy)

    # --- the executor's cards ----------------------------------------------------------------------------------
    def unit_cards(self, c, kind, main):
        """The task's cards of every unit ("Задание отряда"): (its group card = the mix of our places' cards by its
        shares [P, N, D] - so the executor's gradient reaches the group collection -, its main group's mission kind
        card [P, N, D]; zeros for a unit without a group). kind [P, G], main [P, N] (the plan's, locked)."""
        G = self.cfg.places
        mix = torch.einsum("png,pgd->pnd", c["shares"][..., :G], c["groups"][:, :G])
        kc = self.kinds[kind]                                                 # [P, G, D]
        has = (main >= 0) & (main < G)
        idx = main.clamp(0, G - 1)
        kcu = kc.gather(1, idx[..., None].expand(-1, -1, kc.shape[-1]))
        z = has[..., None].to(mix.dtype)
        return mix * z, kcu * z
