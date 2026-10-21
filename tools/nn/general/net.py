"""The general's network (S28 skeleton): one plan from the armies and the map (design/general/index.html, the blocks in
order; docs/en/training/general.md).

    army slots -> ArmyReader -> unit cards (+ effect cards: 0, simplification 2)          "Отряды с эффектами"
      + where the unit stands and moves (pos: the side's frame, as the unit network's token)
    tiles (sectors, 8 numbers) -> a small net + the tile's place (+ its centre, the same    "Участки карты" (simpl. 1)
      projection as the units' positions)
    council: units + tiles, each asks each (transformer blocks, label own / enemy / tile) "Совещание"
      -> tiles after the council ("Карта после совещания"), units after it ("Армии после совещания")
    group collection: 10 places a side, slot attention, 3 rounds; a unit splits itself     "Сбор групп"
      100 % between its side's places (side mask); the dead and the routing take no part;
      the keys less a learned share of their side's mean (GeneralConfig.center), + the units' positions;
      a re-plan: a learned bonus for a unit's last group (GeneralConfig.stick: the groups' inertia)
      -> group cards and shares ("Группы"); a place is a group when it is some unit's main place
    the groups' numbers: a re-plan's groups are matched to the last plan's by their units (match_groups) and
      renumbered, so a group keeps its number (its prev card, its task) while it keeps its units
    group council: the groups listen to the tiles and to all groups (only groups change)   "Совещание групп"
    missions, step by step: kind (GeneralConfig.kinds learned cards) -> target              "Выбор заданий"
      a re-plan: + kind_stick on a group's logit of its last kind (the same number, some of its units kept)
      targets: our groups (not itself), their groups, the coarse tiles (target_grid^2, the council's tiles pooled);
      a kind takes only its target types (config.MISSION_KINDS: attack an enemy group, hold a tile or our group ...):
      the other pairs' logits are -inf, a kind with no target of its types there is not chosen; only our groups choose
    deployment place (the first plan only): a pointer to a tile of our zone                "Выбор мест расстановки"
    plan value: the critic of the plan, on the plan's cards (detached: its error does not  "Оценка замысла"
      move the rest)

Every choice is sampled (or given: the update's log-probabilities under the choices made). Per own group: its
log-probability (the sum of its steps') and entropy; groups that are not groups take no part.

Input dict (rows P; N the battle's slots, S tiles, G places a side):
    num [P, N, 45], lab [P, N, 3]   the army slots (units.py)
    pos [P, N, 7]                   where each unit stands (POS_FEATURES; zeros for an enemy never seen)
    on [P, N]                       the unit sits at the council (present, alive, not known dead)
    own [P, N]                      the general's side
    free [P, N]                     may join a group (on, not routing)
    tiles [P, S, 8], tile_ok [P, S] the tiles and those inside the map
    zone [P, S], deploy [P]         the deployment's tiles and whether the plan deploys
    prev [P, 2G, D], has_prev [P]   the places' cards of the last plan (the inertia of a re-plan)
    prev_main [P, N]                the last plan's group of every unit (-1 none): the groups' numbers (match_groups)
    prev_kind [P, G]                (optional) the last plan's mission kind of each of our group numbers (-1 none): the
                                    kind's inertia (GeneralConfig.kind_stick)
    main [P, N], perm [P, 2G]       (optional) the plan's membership and numbering when it is computed again (locked:
                                    the update, the refresh)
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from tools.nn.general.army_reader import ArmyReader
from tools.nn.model.encoder import Block

NEG = -1e9
INPUTS = ("num", "lab", "pos", "on", "own", "free", "tiles", "tile_ok", "zone", "deploy", "prev", "has_prev",
          "prev_main", "prev_kind")
CHOICES = ("kind", "target", "place")
# a unit's place in the general's input, as the unit network's token reads it (observation.py: the side's frame):
# forward and lateral / observation.POS, facing cos and sin, velocity forward and lateral / observation.VEL, known
# (own, or an enemy seen; else the row is zeros)
POS_FEATURES = ("fwd", "lat", "face_cos", "face_sin", "vel_fwd", "vel_lat", "known")
POS_M = 500.0                                                  # observation.POS (tools/nn/model is not imported here)


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
    a GRU and a feed-forward update the place. In a re-plan a unit's logit of its last plan's group (prev_main: the
    place that starts from that group's card) gets a learned bonus (GeneralConfig.stick): a unit stays unless the
    attention prefers another place clearly. -> (shares [P, N, 2G] of the last round, place cards [P, 2G, D])."""

    def __init__(self, d, places, rounds, center=0.0, stick=0.0, sharp=0.0):
        super().__init__()
        self.G, self.rounds = places, rounds
        # keys and queries layer-normed (no affine: each a vector of norm sqrt(d)), their dot product / sqrt(d) (= sqrt(d)
        # x their cosine) x a learned sharpness (GeneralConfig.sharp: its start; a log parameter, so it stays > 0);
        # None: the plain dot product / sqrt(d)
        self.log_sharp = nn.Parameter(torch.tensor(math.log(sharp))) if sharp else None
        # the side's mean key taken off the keys, this learned share of it (GeneralConfig.center; None: plain)
        self.center = nn.Parameter(torch.tensor(float(center))) if center else None
        # a re-plan's bonus on the logit of a unit's last group (GeneralConfig.stick; None: none)
        self.stick = nn.Parameter(torch.tensor(float(stick))) if stick else None
        self.start = nn.Parameter(torch.randn(2 * places, d) * 0.5)
        self.norm_in, self.norm_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.k, self.v, self.q = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.k_pos = nn.Linear(len(POS_FEATURES), d, bias=False)   # where a unit stands joins its key directly
        self.gru = nn.GRUCell(d, d)
        self.ff = mlp(d, 2 * d, d)

    def side_ok(self, own):
        """[P, N, 2G]: an own unit may join only our places, an enemy unit only the enemy's (the side mask)."""
        G = self.G
        return torch.cat([own[..., None].expand(*own.shape, G), (~own)[..., None].expand(*own.shape, G)], -1)

    def forward(self, u, own, free, prev=None, has_prev=None, pos=None, prev_main=None):
        P, N, d = u.shape
        slots = self.start[None].expand(P, -1, -1)
        if prev is not None:
            slots = torch.where(has_prev.reshape(-1, 1, 1), prev.to(slots.dtype), slots)
        x = self.norm_in(u)
        k, v = self.k(x), self.v(x)
        if pos is not None:
            k = k + self.k_pos(pos.to(k.dtype))
        if self.center is not None:
            # each side's mean key over its free units, a learned share of it taken off (GeneralConfig.center)
            mean = []
            for side in (own, ~own):
                w = (free & side).to(k.dtype)[..., None]
                mean.append((k * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0))
            k = k - self.center * torch.where(own[..., None], mean[0], mean[1])
        ok = self.side_ok(own)
        keep = free[..., None].to(u.dtype)
        stay = None
        if self.stick is not None and prev_main is not None and has_prev is not None:
            last = (prev_main[..., None] == torch.arange(2 * self.G, device=u.device)) & has_prev.reshape(-1, 1, 1)
            stay = self.stick * last.to(k.dtype)                             # [P, N, 2G]
        attn = None
        if self.log_sharp is not None:
            k = F.layer_norm(k, (d,))
        for _ in range(self.rounds):
            q = self.q(self.norm_s(slots))
            if self.log_sharp is not None:
                logits = torch.einsum("pnd,pgd->png", k, F.layer_norm(q, (d,))) * (self.log_sharp.exp() / math.sqrt(d))
            else:
                logits = torch.einsum("pnd,pgd->png", k, q) / math.sqrt(d)
            if stay is not None:
                logits = logits + stay
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


def match_groups(main, prev_main, G):
    """The groups' numbers of a re-plan: perm [P, 2G] - number s (ours 0..G-1, theirs G..2G-1) goes to the new place
    perm[s] - matching the new places (main [P, N]: each unit's new main place, -1 none) to the last plan's groups
    (prev_main [P, N], -1 none) by the units they share, greedily: the pair sharing the most units first, then the next
    of those left; a place keeps its own number when nothing tells otherwise (the new place j started from the last
    plan's card j). Only within a side (ours with ours). Without a last plan (prev_main all -1) the numbering is the
    identity."""
    P, N = main.shape
    dev = main.device
    ar = torch.arange(2 * G, device=dev)
    new = (main[..., None] == ar).to(torch.float32)                          # [P, N, 2G] new place j
    old = (prev_main[..., None] == ar).to(torch.float32)                     # [P, N, 2G] old number i
    share = torch.einsum("pnj,pni->pji", new, old)                           # [P, 2G (new j), 2G (old i)]
    side = ar // G
    same = side[:, None] == side[None, :]
    score = 2.0 * share + torch.eye(2 * G, device=dev)                       # (a unit shared outweighs the own number)
    score = score.masked_fill(~same[None], float("-inf"))
    perm = torch.full((P, 2 * G), -1, dtype=torch.long, device=dev)
    rows = torch.arange(P, device=dev)
    for _ in range(2 * G):
        flat = score.reshape(P, -1).argmax(-1)
        j, i = flat // (2 * G), flat % (2 * G)
        perm[rows, i] = j
        score[rows, j, :] = float("-inf")
        score[rows, :, i] = float("-inf")
    return perm


def renumber(perm, main):
    """main [P, N] (new places, -1 none) -> the units' group numbers under perm (match_groups)."""
    ar = torch.arange(perm.shape[1], device=perm.device).expand_as(perm)
    inv = torch.empty_like(perm).scatter_(1, perm, ar)
    return torch.where(main >= 0, inv.gather(1, main.clamp(min=0)), main)


def target_kind(target, G, C):
    """What mission target indices are (General.decide's): 0 our group, 1 their group, 2 a coarse tile, -1 other."""
    return torch.where(target < G, 0, torch.where(target < 2 * G, 1, torch.where(target < 2 * G + C, 2, -1)))


class General(nn.Module):
    """The general's network (module doc). cfg: GeneralConfig; sectors: the unit network's sectors a side (tiles =
    sectors^2, simplification 1); grid_half: their square's half side, m (ModelConfig.grid_half: the tiles' centres)."""

    def __init__(self, cfg, sectors=16, grid_half=800.0):
        super().__init__()
        from tools.nn.general import army_input as ai
        from tools.nn.general import tiles as gtiles
        self.cfg = cfg
        d, G, K = cfg.card, cfg.places, cfg.kinds
        self.S = sectors * sectors
        self.C = cfg.target_grid ** 2                           # the coarse tiles (the targets)
        self.reader = ArmyReader(cfg.reader_hidden, d, numbers=ai.SIZE)
        self.label = nn.Embedding(3, d)                         # own unit, enemy unit, tile
        self.tile_in = nn.Sequential(nn.Linear(len(gtiles.FEATURES), d), nn.GELU(), nn.Linear(d, d))
        self.tile_place = nn.Parameter(torch.randn(self.S, d) * 0.1)
        # where a unit stands and moves (POS_FEATURES); a tile's centre goes through the same forward / lateral columns
        self.pos_in = nn.Linear(len(POS_FEATURES), d)
        step = 2 * grid_half / sectors
        c = -grid_half + (torch.arange(sectors, dtype=torch.float32) + 0.5) * step
        f, l = torch.meshgrid(c, c, indexing="ij")
        tpos = torch.zeros(self.S, len(POS_FEATURES))
        tpos[:, 0], tpos[:, 1] = f.reshape(-1) / POS_M, l.reshape(-1) / POS_M
        self.register_buffer("tile_pos", tpos, persistent=False)
        self.register_buffer("coarse", gtiles.coarse_index(sectors, cfg.target_grid), persistent=False)
        # which (kind, target) pairs may be chosen: kind k with candidate c when its type (target_kind: our group, theirs,
        # a coarse tile) is one config.MISSION_KINDS[k] takes - [K, 2G + C]
        from tools.nn.general import config as gconfig
        allow = torch.tensor(gconfig.allowed(gconfig.mission_kinds(cfg)), dtype=torch.bool)       # [K, 3]
        types = target_kind(torch.arange(2 * G + self.C), G, self.C)
        self.register_buffer("pair_ok", allow[:, types], persistent=False)
        self.norm_in = nn.LayerNorm(d)
        self.council = nn.ModuleList(Block(d, cfg.council_heads, cfg.ff) for _ in range(cfg.council_layers))
        self.norm_out = nn.LayerNorm(d)
        self.groups = SlotGroups(d, G, cfg.rounds, cfg.center, cfg.stick, cfg.sharp)
        self.listen = nn.ModuleList(Listen(d, cfg.council_heads, cfg.ff) for _ in range(cfg.group_layers))
        self.norm_g = nn.LayerNorm(d)
        self.kinds = nn.Parameter(torch.randn(K, d) / math.sqrt(d) * 4)
        self.kind_q = nn.Linear(d, d)
        self.target_q = mlp(2 * d, 2 * d, d)
        self.cand_k = nn.Linear(d, d)
        self.place_q = mlp(2 * d, 2 * d, d)
        self.place_k = nn.Linear(d, d)
        self.value = mlp(4 * d + 1, d, 1)

    # --- the cards ---------------------------------------------------------------------------------------------
    def coarse_tiles(self, tiles, tile_ok):
        """(cards [P, C, D]: the mean of the council's tiles inside each coarse tile, ok [P, C]: some inside the map)."""
        P, S, d = tiles.shape
        w = tile_ok.to(tiles.dtype)
        idx = self.coarse.to(tiles.device)
        tot = tiles.new_zeros(P, self.C, d).index_add_(1, idx, tiles * w[..., None])
        n = tiles.new_zeros(P, self.C).index_add_(1, idx, w)
        return tot / n.clamp(min=1.0)[..., None], n > 0

    def cards(self, inp):
        """Everything before the choices: {"units", "tiles" (after the council), "coarse", "coarse_ok" (the targets'
        tiles), "shares" [P, N, 2G], "places" [P, 2G, D] (the places' cards: the next plan's inertia), "groups"
        [P, 2G, D] (after the group council and its norm), "main" [P, N] (a free unit's group, -1 none), "perm"
        [P, 2G] (the groups' numbering: match_groups), "active" [P, 2G] (some unit's group)}; every [.., 2G] is in the
        groups' numbers."""
        P, N = inp["on"].shape
        G = self.cfg.places
        on, own = inp["on"], inp["own"]
        u = self.reader(inp["num"], inp["lab"], ~on)
        pos = inp["pos"].to(u.dtype) * on[..., None].to(u.dtype)
        u = u + (self.label((~own).long()) + self.pos_in(pos)) * on[..., None].to(u.dtype)
        t = (self.tile_in(inp["tiles"].to(u.dtype)) + self.tile_place + self.label.weight[2]
             + self.pos_in(self.tile_pos.to(u.dtype)))
        x = self.norm_in(torch.cat([u, t], 1))
        ok = torch.cat([on, inp["tile_ok"]], 1)
        bias = key_bias(ok, self.cfg.council_heads).to(x.dtype)
        for block in self.council:
            x = block(x, bias)
        x = self.norm_out(x)
        units, tiles = x[:, :N], x[:, N:]
        shares, places = self.groups(units, own, inp["free"], inp.get("prev"), inp.get("has_prev"), pos,
                                     inp.get("prev_main"))
        free = inp["free"]
        if inp.get("main") is not None:
            # the plan's own membership and numbering, locked (a re-computation of a plan made: the update, the
            # refresh); the shares and cards are computed again (the gradient's path)
            main, perm = inp["main"], inp["perm"]
        else:
            raw = torch.where(free, shares.argmax(-1), torch.full_like(free, -1, dtype=torch.long))
            prev_main = inp.get("prev_main")
            perm = match_groups(raw, torch.full_like(raw, -1) if prev_main is None else prev_main, G)
            main = renumber(perm, raw)
        shares = shares.gather(2, perm[:, None, :].expand(-1, N, -1))
        places = places.gather(1, perm[..., None].expand(-1, -1, places.shape[-1]))
        hit = main[..., None] == torch.arange(2 * G, device=main.device)
        active = hit.any(1)
        keys = torch.cat([tiles, places], 1)
        key_ok = torch.cat([inp["tile_ok"], active], 1)
        g = places
        for layer in self.listen:
            g = layer(g, keys, key_ok)
        coarse, coarse_ok = self.coarse_tiles(tiles, inp["tile_ok"])
        return {"units": units, "tiles": tiles, "coarse": coarse, "coarse_ok": coarse_ok, "shares": shares,
                "places": places, "groups": self.norm_g(g), "main": main, "perm": perm, "active": active}

    # --- the choices -------------------------------------------------------------------------------------------
    def decide(self, inp, c, act=None, greedy=False):
        """The missions and places of our groups on the cards c (cards()): sampled, or under act {"kind", "target",
        "place"} [P, G]. -> dict: the choices [P, G], "lp" [P, G] and "ent" [P, G] (the group's steps summed; 0 for a
        place that is not a group), "ent_parts" [P, G, 3] (kind, target, place), "value" [P], "lp_kind" [P, G] (the
        kind step's alone), "kind_logp" [P, G, K] (every kind's log-probability; uniform for a place that is not a
        group), "kind_ok" [P, G, K] (the kinds that may be chosen: some target of their types), "lp_parts" [P, G, 3]
        (the kind's, target's and place's steps apart), "target_logits" / "target_ok" [P, G, 2G + C] (the target step's
        logits, masked, and the targets allowed for the group's kind: the teacher's imitation), "pairs" [P, G, 2] (the
        (kind, target) pairs allowed, all kinds x the targets there are: the log's masked share). Targets: 0..G-1 our
        groups, G..2G-1 theirs, 2G.. the coarse tiles (target_kind); a kind only with the target types it takes
        (config.MISSION_KINDS: the pair_ok buffer)."""
        G = self.cfg.places
        g, tiles, active = c["groups"], c["tiles"], c["active"]
        d = g.shape[-1]
        mine = active[:, :G]                                                  # [P, G] our groups choose
        act = act or {}
        cand_ok = torch.cat([active, c["coarse_ok"]], 1)                      # [P, 2G + C]
        self_ = torch.eye(G, 2 * G + self.C, dtype=torch.bool, device=g.device)   # a group is not its own target
        base_t = cand_ok[:, None, :] & ~self_[None] & mine[..., None]         # [P, G, 2G + C] the targets there are
        pair = self.pair_ok.to(g.device)                                      # [K, 2G + C] the kinds' target types
        # a kind may be chosen when some target there is of a type it takes (screen: another group of ours ...)
        k_ok = (base_t[:, :, None, :] & pair[None, None]).any(-1)            # [P, G, K]
        k_logits = self.kind_q(g[:, :G]) @ self.kinds.t() / math.sqrt(d)      # [P, G, K]
        k_logits = k_logits + self.kind_bonus(inp, c, k_logits)
        kind, lp_k, ent_k = categorical(k_logits, k_ok, act.get("kind"), greedy)
        # every kind's log-probability (the mission branches: the old log-probability, the alternatives' sampling)
        k_lg = k_logits.masked_fill(~k_ok, NEG)
        kind_logp = torch.log_softmax(torch.where(k_ok.any(-1, keepdim=True), k_lg, torch.zeros_like(k_lg)).float(), -1)
        kc = self.kinds[kind]                                                 # [P, G, D]
        cand = torch.cat([g, c["coarse"]], 1)                                 # [P, 2G + C, D]
        keys = self.cand_k(cand)
        ok_t = base_t & pair[kind]                                            # only the kind's target types
        qt = self.target_q(torch.cat([g[:, :G], kc], -1))
        t_logits = qt @ keys.transpose(1, 2) / math.sqrt(d)                  # [P, G, 2G + C]
        target, lp_t, ent_t = categorical(t_logits, ok_t, act.get("target"), greedy)
        deploy = inp["deploy"].reshape(-1, 1)
        okp = inp["zone"][:, None, :] & mine[..., None] & deploy[..., None]   # [P, G, S]
        qp = self.place_q(torch.cat([g[:, :G], kc], -1))
        place, lp_p, ent_p = categorical(qp @ self.place_k(tiles).transpose(1, 2) / math.sqrt(d), okp,
                                         act.get("place"), greedy)
        place = torch.where(okp.any(-1), place, torch.full_like(place, -1))
        m = mine.to(lp_k.dtype)
        lp = (lp_k + lp_t + lp_p) * m
        ent = (ent_k + ent_t + ent_p) * m
        # the plan's value on its cards, detached (its error trains only the value head)
        summary = torch.cat([masked_mean(g[:, :G].detach(), mine), masked_mean(g[:, G:].detach(), active[:, G:]),
                             masked_mean(tiles.detach(), inp["tile_ok"]), masked_mean(kc.detach(), mine),
                             inp["deploy"].to(g.dtype)[:, None]], -1)
        value = self.value(summary)[..., 0]
        # the masked share (the log): the (kind, target) pairs a group may choose of all kinds x the targets there are
        n_t = base_t.sum(-1).to(lp.dtype)
        pairs_ok = (base_t[:, :, None, :] & pair[None, None]).sum((-1, -2)).to(lp.dtype)
        return {"kind": kind, "target": target, "place": place, "lp": lp, "ent": ent, "value": value,
                "ent_parts": torch.stack([ent_k, ent_t, ent_p], -1) * m[..., None],
                "lp_kind": lp_k * m, "kind_logp": kind_logp, "kind_ok": k_ok,
                "lp_parts": torch.stack([lp_k, lp_t, lp_p], -1) * m[..., None],
                "target_logits": t_logits.masked_fill(~ok_t, NEG), "target_ok": ok_t,
                "pairs": torch.stack([pairs_ok, n_t * self.cfg.kinds], -1) * m[..., None]}

    def kind_bonus(self, inp, c, k_logits):
        """[P, G, K] the kind's inertia of a re-plan (GeneralConfig.kind_stick B): + B on a group's logit of the kind its
        number had in the last plan (inp prev_kind [P, G], -1 none), when the group kept some of that number's units
        (main and prev_main share a unit: a number given to a new place by default, match_groups' identity, is not the
        same group); zeros without prev_kind or with B = 0. (A kind with no target of its types now stays masked: the
        mask comes after.)"""
        B = float(getattr(self.cfg, "kind_stick", 0.0))
        pk = inp.get("prev_kind")
        if not B or pk is None or inp.get("prev_main") is None:
            return torch.zeros_like(k_logits)
        G, K = self.cfg.places, k_logits.shape[-1]
        ar = torch.arange(G, device=k_logits.device)
        kept = ((c["main"][..., None] == ar) & (inp["prev_main"][..., None] == ar)).any(1)       # [P, G]
        kept = kept & (pk >= 0) & inp["has_prev"].reshape(-1, 1)
        hot = (pk.clamp(0, K - 1)[..., None] == torch.arange(K, device=k_logits.device)) & kept[..., None]
        return B * hot.to(k_logits.dtype)

    def forward(self, inp, act=None, greedy=False):
        c = self.cards(inp)
        return c, self.decide(inp, c, act, greedy)

    # --- the executor's cards ----------------------------------------------------------------------------------
    def unit_cards(self, c, kind, main):
        """The task's cards of every unit ("Задание отряда"): (its group card = the mix of our places' cards by its
        shares [P, N, D] - so the executor's gradient reaches the group collection -, its group's mission kind
        card [P, N, D]; zeros for a unit without a group). kind [P, G], main [P, N] (the plan's, locked)."""
        G = self.cfg.places
        mix = torch.einsum("png,pgd->pnd", c["shares"][..., :G], c["groups"][:, :G])
        kc = self.kinds[kind]                                                 # [P, G, D]
        has = (main >= 0) & (main < G)
        idx = main.clamp(0, G - 1)
        kcu = kc.gather(1, idx[..., None].expand(-1, -1, kc.shape[-1]))
        z = has[..., None].to(mix.dtype)
        return mix * z, kcu * z
