"""The operative's network (plan.md; design/tactic "Зона оперативника").

    Encoder      'Анализ отряда' 185 -> 256 -> 128 and the squeeze 128 -> 8 (tactic/net.py, frozen from
                 build/diag_tactic/card_pretrain/card8.pt, as at the tactician); the cards painted into the field map,
                 'Анализ карты' 33 -> 48 -> 64 -> 64 (tactic/net.Stack: the vector 128 by mean and max, the map
                 64 x 32 x 32); 'Запись отряда': the card + place 3 + the last groups 14 -> 128; 'Токен группы': a
                 learned vector of the group's number (the empty ones differ by it) + (the mean of its members' records,
                 the mean of its target's records, its numbers 10) -> 128; the summary: the map's vector + the numbers
                 1 + the command 7 -> 128; the council: tactic/decide.Council (2 x unit.net.Block, 128, 4 heads) over
                 the rows (40), the tokens (6) and the summary
    Heads        per non-empty group 'keep / move / re-form' (3; zero weights, a bias on 'keep' at the start); per free
                 unit in slot order a pointer at the open groups (the re-formed and the empty ones; a full one, 10, is
                 closed); per group with a new task: form (square / line) -> target (a pointer: a seen enemy as the seed
                 or 'none'; the code grows it, target.py) -> turn (16 sectors from the bearing on the target; the
                 sectors breaking the target's rules closed) -> cell (1024: where the zone should go; the cells whose
                 zone leaves the field closed; the code makes the step, step.clamp) -> verb (4) -> pace (3) (the
                 closed ones: config verbs_open, paces_open)
    choose()     the walk: the one for the sampling, the update (act: the same choices again) and the teacher's term
                 (labels: the cross-entropies against the formula's choices, the formula's earlier choices in force)
    OperativeNet, OperativeCritic (its own encoder; the value of the summary's seat; not used in the game)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.operative import cfg
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as IN
from tools.nn.operative import step as SP
from tools.nn.operative import target as TG
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import net as NT
from tools.nn.tactic import raw as TR
from tools.nn.unit.net import NEG

FORMS = ("square", "line")
SQUARE, LINE = 0, 1
Q = 64                       # the pointers' key width


def _w():
    return int(tcfg()["council"]["width"])


class Encoder(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        W = _w()
        G = int(cfg()["groups"])
        k = int(tcfg()["card"])
        self.analysis = NT.UnitAnalysis()
        self.squeeze = NT.CardSqueeze()
        self.map = NT.Stack((len(IN.layers(k)), 48, 64, 64), 128)
        self.record = nn.Linear(int(tcfg()["analysis"]["out"]) + 3 + GR.PREV, W)
        self.number = nn.Parameter(torch.randn(G, W) * 0.1)
        self.token = nn.Linear(2 * W + GR.NUMBERS, W)
        self.summary = nn.Linear(128 + 1 + len(DC.VERBS) + len(DC.PACES), W)
        self.council = DC.Council()
        if card:
            ck = torch.load(DC.CARD_CKPT, map_location="cpu", weights_only=False)
            self.analysis.load_state_dict(ck["analysis"])
            self.squeeze.load_state_dict(ck["squeeze"])
            for p in list(self.analysis.parameters()) + list(self.squeeze.parameters()):
                p.requires_grad_(False)

    def forward(self, inp):
        mask = inp["mask"]
        zero = torch.zeros((), device=mask.device)
        cards = self.analysis(inp["rows"], mask)
        card8 = torch.where(mask[..., None], self.squeeze(cards), zero)
        m = inp["map"]
        k = card8.shape[-1]
        i, j = IN.LI["card_mine0"], IN.LI["card_enemy0"]
        m = torch.cat([m[:, :i], NT.paint_cards(card8, inp["cover"], inp["own"]),
                       NT.paint_cards(card8, inp["cover"], inp["enemy"]), m[:, j + k:]], 1)
        vec, fmap = self.map(m)
        rec = torch.where(mask[..., None], F.relu(self.record(torch.cat([cards, inp["place"], inp["prev"]], -1))), zero)
        tok = self.tokens(rec, inp["members"], inp["targets"], inp["tokens"])
        s = self.summary(torch.cat([vec, inp["numbers"], inp["command"]], 1))
        B, K = mask.shape
        x = torch.cat([rec, tok], 1)
        mm = torch.cat([mask, torch.ones(tok.shape[:2], dtype=torch.bool, device=mask.device)], 1)
        after, sm = self.council(x, s, mm)
        return {"units": after[:, :K], "tokens": after[:, K:], "summary": sm, "map": fmap, "mask": mask}

    def tokens(self, rec, members, targets, numbers):
        return F.relu(self.token(torch.cat([_mean(rec, members), _mean(rec, targets), numbers], -1))) + self.number[None]


def _mean(x, sel):
    """[B, G, W]: the mean of x [B, K, W] over the rows sel [B, G, K] (zeros where none)."""
    w = sel.float()
    return torch.einsum("bgk,bkw->bgw", w, x) / w.sum(2, keepdim=True).clamp(min=1)


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        c = cfg()
        W = _w()
        S = int(c["sectors"])
        C = int(c["cells"])
        h = int(c["council_extra"]["hidden"])
        self.keep_in = nn.Linear(W + GR.NUMBERS, h)
        self.keep = nn.Linear(h, 3)
        with torch.no_grad():
            self.keep.weight.zero_()
            self.keep.bias.zero_()
            self.keep.bias[GR.KEEP] = float(c["council_extra"]["keep_bias"])
        self.cq, self.ck = nn.Linear(W, Q), nn.Linear(W, Q)
        self.mix = nn.Linear(W, W)
        self.form = nn.Linear(W, len(FORMS))
        self.form_emb = nn.Embedding(len(FORMS), W)
        self.tq, self.tk, self.tnone = nn.Linear(W, Q), nn.Linear(W, Q), nn.Linear(W, 1)
        self.tmix = nn.Linear(W, W)
        self.turn = nn.Linear(W, S)
        self.turn_emb = nn.Embedding(S, W)
        self.where_core = nn.Linear(W, 64)
        self.where_a = nn.Conv2d(64 + 2 + 2, 32, 3, padding=1)
        self.join = nn.Linear(W, W)
        # the members' prior (a feature of the positions, learned scale): - distance to the group's centre so far / 100 m
        # for a group with members, a bias for an empty one; at the start 'join a group within join_m, else a new one'
        self.near_w = nn.Parameter(torch.tensor(1.0))
        self.empty_b = nn.Parameter(torch.tensor(-float(c["teacher"]["join_m"]) / 100.0))
        self.cmd = nn.Linear(len(DC.VERBS) + len(DC.PACES), W)
        self.where_b = nn.Conv2d(32, 1, 1)
        self.verb = nn.Linear(W, len(DC.VERBS))
        self.pace = nn.Linear(W, len(DC.PACES))
        k = (torch.arange(C, dtype=torch.float32) + 0.5) / C * 2 - 1
        self.register_buffer("coords", torch.stack([k[None, :].expand(C, C), -k[:, None].expand(C, C)])[None],
                             persistent=False)

    def keep_logits(self, tok, numbers):
        return self.keep(F.relu(self.keep_in(torch.cat([tok, numbers], -1))))

    def where(self, h, fmap, extra):
        """extra [R, 2, C, C]: the group's members' and its target's shares of the cells (where they stand now)."""
        y = F.relu(fmap + self.where_core(h)[:, :, None, None])
        y = torch.cat([y, self.coords.expand(y.shape[0], -1, -1, -1), extra], 1)
        return self.where_b(F.relu(self.where_a(y)))[:, 0].flatten(1)


@dataclass
class Ctx:
    """What the walk needs besides the network's output (net-free: the formula and the exam build it too)."""
    raw: torch.Tensor        # [B, N, RAW]
    axes: object             # frame.Axes
    groups: object           # groups.Groups before the decision
    slot: torch.Tensor       # [B, K]
    own: torch.Tensor        # [B, K] bool
    enemy: torch.Tensor      # [B, K] bool
    J: torch.Tensor          # [B, K, RAW]

    @property
    def row_group(self):
        g = self.groups.member.gather(1, self.slot.clamp(min=0))
        return torch.where(self.own, g, torch.full_like(g, -1))

    @property
    def cand(self):
        """[B, N] the seen alive enemies (the target's candidates)."""
        return to_slots(self.enemy, self.slot, self.raw.shape[1], False)

    def cells(self):
        return SP.cells(self.axes)


def to_slots(v, slot, N, fill):
    """[B, N]: the rows' values v [B, K] at their slots (slots without a row: fill)."""
    B = v.shape[0]
    out = torch.full((B, N + 1), fill, dtype=v.dtype, device=v.device)
    out.scatter_(1, torch.where(slot >= 0, slot, torch.full_like(slot, N)), v)
    return out[:, :N]


def ctx_of(inp, raw, axes, groups):
    return Ctx(raw, axes, groups, inp["slot"], inp["own"], inp["enemy"], inp["J"])


def open_mask(names, all_names, R, device):
    m = torch.tensor([n in names for n in all_names], device=device)
    return m[None].expand(R, -1)


def _pick(lg, given, greedy):
    if given is not None:
        return given
    return lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()


def _lp_ent(lg, x):
    d = torch.distributions.Categorical(logits=lg)
    p = d.probs
    return d.log_prob(x), -torch.where(p > 0, p * d.logits, torch.zeros_like(p)).sum(-1)


def _soft_ce(lg, q):
    """[R]: the cross-entropy of the logits lg against the distribution q (rows of zeros: 0)."""
    return -(torch.where(q > 0, q * F.log_softmax(lg, -1), torch.zeros_like(q))).sum(-1)


HEADS = ("keep", "comp", "form", "target", "turn", "cell", "verb", "pace")


def choose(enc, inp, ctx, heads, act=None, greedy=False, labels=None):
    """The walk (module doc). Returns {act: {keep [B, G], comp [B, K], form, target, turn, cell, verb, pace [B, G]},
    lp [B], ent {head: [B]}, new: the groups after (groups.Groups), task [B, G] (a new parcel), moved [B, G] (the same
    members: the tactician's tiles may stay), reformed [B, G] (the members changed by the network's choice), goal_x,
    goal_z [B, G] (the cell chosen), teach (with labels): the term and its agreement counts}."""
    c = cfg()
    G = int(c["groups"])
    raw, axes, old = ctx.raw, ctx.axes, ctx.groups
    B, K = ctx.slot.shape
    N = raw.shape[1]
    dev = raw.device
    idx = torch.arange(B, device=dev)
    units, tok = enc["units"], enc["tokens"]
    lp = torch.zeros(B, device=dev)
    ent = {h: torch.zeros(B, device=dev) for h in HEADS}
    teach = {"term": torch.zeros((), device=dev), "n": 0.0, **{f"{h}_ok": 0.0 for h in HEADS},
             **{f"{h}_n": 0.0 for h in HEADS}}
    A = {k: (act[k] if act is not None else None) for k in HEADS}
    out = {"keep": torch.zeros((B, G), dtype=torch.long, device=dev),
           "comp": torch.full((B, K), -1, dtype=torch.long, device=dev),
           **{k: torch.zeros((B, G), dtype=torch.long, device=dev) for k in ("form", "target", "turn", "cell", "verb",
                                                                              "pace")}}

    def score(name, lg, x, use, q=None, hard=None):
        nonlocal lp
        l, e = _lp_ent(lg, x)
        lp = lp + l * use
        ent[name] = ent[name] + e * use
        if labels is not None and bool(use.any()):
            if q is None:
                q = F.one_hot(hard.clamp(min=0), lg.shape[-1]).float()
            ce = _soft_ce(lg, q)
            teach["term"] = teach["term"] + (ce * use).sum()
            teach["n"] += float(use.sum())
            # agreement: the network's best choice is the label's (a soft label: within its support's upper half -
            # any seed of the piece, a cell or a sector next to the wanted one)
            best = lg.argmax(-1)
            if hard is None:
                hit = q.gather(1, best[:, None])[:, 0] >= 0.5 * q.amax(-1)
            else:
                hit = best == hard
            teach[f"{name}_ok"] += float((hit & use).sum())
            teach[f"{name}_n"] += float(use.sum())

    # 1. keep / move / re-form, per non-empty group
    on = old.on
    cmd = heads.cmd(inp["command"])                                       # [B, W]: the order from above, to every head
    lk = heads.keep_logits(tok + cmd[:, None], inp["tokens"])
    k = _pick(lk, A["keep"], greedy)
    k = torch.where(on, k, torch.full_like(k, GR.REFORM))
    for g in range(G):
        score("keep", lk[:, g], k[:, g], on[:, g], hard=None if labels is None else labels["keep"][:, g])
    out["keep"] = k
    # 2. the members: free units in slot order, pointing at the open groups
    rg = ctx.row_group
    free = ctx.own & ((rg < 0) | (k.gather(1, rg.clamp(min=0)) == GR.REFORM))
    opened = ~on | (k == GR.REFORM)
    new_rg = torch.where(free, torch.full_like(rg, -1), rg)
    count = torch.zeros((B, G), dtype=torch.long, device=dev)
    full = int(tcfg()["max_group"])
    tok_c = tok + cmd[:, None]
    for j in range(K):
        a = free[:, j]
        if not bool(a.any()):
            continue
        allowed = opened & (count < full)
        a = a & allowed.any(1)
        lg = torch.einsum("bq,bgq->bg", heads.cq(units[:, j]), heads.ck(tok_c)) / Q ** 0.5
        # the prior: the distance from this unit to each group's members chosen so far (kept groups: their members)
        w = ((new_rg[:, None, :] == torch.arange(G, device=dev)[None, :, None]) & ctx.own[:, None, :]).float()
        nw = w.sum(2)
        jx, jz = ctx.J[..., TR.I["x"]], ctx.J[..., TR.I["z"]]
        gx_ = (w * jx[:, None, :]).sum(2) / nw.clamp(min=1)
        gz_ = (w * jz[:, None, :]).sum(2) / nw.clamp(min=1)
        dist = torch.sqrt((gx_ - jx[:, j, None]) ** 2 + (gz_ - jz[:, j, None]) ** 2)
        lg = lg + torch.where(nw > 0, -heads.near_w * dist / 100.0, heads.empty_b * torch.ones_like(dist))
        lg = lg.masked_fill(~allowed, NEG)
        cj = _pick(lg, None if A["comp"] is None else A["comp"][:, j].clamp(min=0), greedy)
        score("comp", lg, cj, a, hard=None if labels is None else labels["comp"][:, j])
        new_rg[:, j] = torch.where(a, cj, new_rg[:, j])
        out["comp"][:, j] = torch.where(a, cj, torch.full_like(cj, -1))
        count.scatter_add_(1, cj[:, None], a.long()[:, None])
        # the walk is told who joined: the chosen group's token takes the unit in (the next units see it)
        tok_c = tok_c + F.one_hot(cj, G).float()[..., None] * (heads.join(units[:, j]) * a.float()[:, None])[:, None]
    mem_rows = new_rg[:, None, :] == torch.arange(G, device=dev)[None, :, None]           # [B, G, K]
    mem_rows = mem_rows & ctx.own[:, None, :]
    n_new = mem_rows.sum(2)
    task = (on & (k == GR.MOVE)) | (opened & (n_new > 0))
    new_on = (on & (k == GR.KEEP)) | task
    old_rows = old.members().gather(2, ctx.slot.clamp(min=0)[:, None, :].expand(B, G, K)) & ctx.own[:, None, :]
    same = (mem_rows == old_rows).all(2) & on
    reformed = on & (k == GR.REFORM) & ~same
    # the groups after: members (slot space)
    new = old.clone()
    # (an own unit without a row - impossible: every own alive unit has one - keeps -1)
    new.member = to_slots(torch.where(ctx.own, new_rg, torch.full_like(new_rg, -1)), ctx.slot, N, -1)
    new.on = new_on
    new.age = torch.where(task & ~same, torch.zeros_like(old.age), old.age)
    # 3. the tasks
    I = TR.I
    vopen = open_mask(c["verbs_open"], DC.VERBS, B, dev)
    popen = open_mask(c["paces_open"], DC.PACES, B, dev)
    cx_, cz_ = ctx.cells()
    S = int(c["sectors"])
    goal_x = new.cx.clone()
    goal_z = new.cz.clone()
    moved = torch.zeros((B, G), dtype=torch.bool, device=dev)
    for g in range(G):
        a = task[:, g]
        if not bool(a.any()):
            continue
        L = None if labels is None else {h: labels[h][:, g] for h in ("form", "target", "turn", "cell", "verb", "pace")}
        sel = mem_rows[:, g]
        h = tok[:, g] + heads.mix(_mean(units, sel[:, None])[:, 0]) + cmd
        lf = heads.form(h)
        fo = _pick(lf, None if A["form"] is None else A["form"][:, g], greedy)
        score("form", lf, fo, a, hard=None if L is None else L["form"])
        line = fo == LINE
        h = h + heads.form_emb(fo)
        n_al = n_new[:, g].clamp(min=1)
        near_m = SP.near(n_al, line)
        # the target: a seed or none
        keys = heads.tk(units)
        lt = torch.einsum("bq,bkq->bk", heads.tq(h), keys) / Q ** 0.5
        lt = torch.cat([lt.masked_fill(~ctx.enemy, NEG), heads.tnone(h)], 1)
        tt = _pick(lt, None if A["target"] is None else A["target"][:, g], greedy)
        score("target", lt, tt, a, q=None if L is None else L["target"])
        w = sel.float()
        nw = w.sum(1).clamp(min=1)
        gx = (ctx.J[..., I["x"]] * w).sum(1) / nw
        gz = (ctx.J[..., I["z"]] * w).sum(1) / nw
        seed = torch.where(tt < K, ctx.slot.gather(1, tt.clamp(max=K - 1)[:, None])[:, 0], torch.full_like(tt, -1))
        seed = torch.where(a, seed, torch.full_like(seed, -1))
        tgt = TG.grow(raw, idx, seed, ctx.cand, gx, gz, near_m)
        base = TG.base_heading(raw, idx, tgt, gx, gz, axes.heading)
        ok = TG.turn_ok(raw, idx, tgt, base, near_m, gx, gz)
        bad = ~ok[:, 0] & a
        if bool(bad.any()):
            only = torch.zeros_like(tgt)
            only[idx[bad], seed[bad].clamp(min=0)] = seed[bad] >= 0
            tgt = torch.where(bad[:, None], only, tgt)
            base = TG.base_heading(raw, idx, tgt, gx, gz, axes.heading)
            ok = TG.turn_ok(raw, idx, tgt, base, near_m, gx, gz)
        ok = ok | ~a[:, None]
        trows = tgt.gather(1, ctx.slot.clamp(min=0)) & ctx.enemy
        h = h + heads.tmix(_mean(units, trows[:, None])[:, 0])
        lr = heads.turn(h).masked_fill(~ok, NEG)
        tr = _pick(lr, None if A["turn"] is None else A["turn"][:, g], greedy)
        qt = None
        if L is not None:
            sg = float(cfg()["teacher"]["sigma_sectors"])
            dd = (torch.arange(S, device=dev)[None] - L["turn"][:, None]).abs()
            dd = torch.minimum(dd, S - dd).float()
            qt = torch.exp(-dd ** 2 / (2 * sg * sg)) * ok.float()
            qt = qt / qt.sum(1, keepdim=True).clamp(min=1e-12)
        score("turn", lr, tr, a, q=qt)
        heading = torch.remainder(base + tr.float() * (360.0 / S), 360.0)
        h = h + heads.turn_emb(tr)
        # the cell: the goal; the zone's size - the tiles kept on a move of the same form, else the members now
        keep_tiles = on[:, g] & same[:, g] & (line == old.line[:, g])
        n0 = torch.where(keep_tiles, old.n0[:, g], n_al)
        front, depth = SP.size(n0, line)
        fok = SP.field_ok(cx_, cz_, front, depth, heading, axes.field)
        fok = fok | ~fok.any(1, keepdim=True)
        cov = inp["cover"]
        extra = torch.stack([torch.einsum("bk,bkhw->bhw", sel.float(), cov),
                             torch.einsum("bk,bkhw->bhw", trows.float(), cov)], 1).clamp(max=1.0)
        lw = heads.where(h, enc["map"], extra).masked_fill(~fok, NEG)
        cl = _pick(lw, None if A["cell"] is None else A["cell"][:, g], greedy)
        qc = None
        if L is not None:
            C = int(c["cells"])
            sg = float(cfg()["teacher"]["sigma_cells"])
            r0, c0 = torch.div(L["cell"], C, rounding_mode="floor"), L["cell"] % C
            ar = torch.arange(C * C, device=dev)
            d2 = (torch.div(ar, C, rounding_mode="floor")[None] - r0[:, None]) ** 2 + ((ar % C)[None] - c0[:, None]) ** 2
            qc = torch.exp(-d2.float().view(B, -1) / (2 * sg * sg)) * fok.float()
            qc = qc / qc.sum(1, keepdim=True).clamp(min=1e-12)
        score("cell", lw, cl, a, q=qc)
        qx, qz = cx_.gather(1, cl[:, None])[:, 0], cz_.gather(1, cl[:, None])[:, 0]
        zx, zz = SP.clamp(gx, gz, qx, qz, near_m, front, depth, heading, axes.field)
        lv = heads.verb(h).masked_fill(~vopen, NEG)
        vb = _pick(lv, None if A["verb"] is None else A["verb"][:, g], greedy)
        score("verb", lv, vb, a & (vopen.sum(1) > 1), hard=None if L is None else L["verb"])
        lpc = heads.pace(h).masked_fill(~popen, NEG)
        pc = _pick(lpc, None if A["pace"] is None else A["pace"][:, g], greedy)
        score("pace", lpc, pc, a & (popen.sum(1) > 1), hard=None if L is None else L["pace"])
        for name, v in (("form", fo), ("target", tt), ("turn", tr), ("cell", cl), ("verb", vb), ("pace", pc)):
            out[name][:, g] = torch.where(a, v, out[name][:, g])
        new.target[:, g] = torch.where(a[:, None], tgt, new.target[:, g])
        new.seed[:, g] = torch.where(a, seed, new.seed[:, g])
        new.cx[:, g] = torch.where(a, zx, new.cx[:, g])
        new.cz[:, g] = torch.where(a, zz, new.cz[:, g])
        new.heading[:, g] = torch.where(a, heading, new.heading[:, g])
        new.line[:, g] = torch.where(a, line, new.line[:, g])
        new.verb[:, g] = torch.where(a, vb, new.verb[:, g])
        new.pace[:, g] = torch.where(a, pc, new.pace[:, g])
        new.n0[:, g] = torch.where(a, n0, new.n0[:, g])
        goal_x[:, g] = torch.where(a, qx, goal_x[:, g])
        goal_z[:, g] = torch.where(a, qz, goal_z[:, g])
        moved[:, g] = a & keep_tiles
    new.target = new.target & new.on[..., None]
    return {"act": out, "lp": lp, "ent": ent, "new": new, "task": task, "moved": moved, "reformed": reformed,
            "goal_x": goal_x, "goal_z": goal_z, "teach": teach}


class NullHeads:
    """Heads of zeros: the walk run with given choices and no network (the formula, the plain operators)."""

    def __init__(self, B, K, G, device):
        c = cfg()
        self.B, self.K, self.G, self.dev = B, K, G, device
        self.S, self.C = int(c["sectors"]), int(c["cells"])

    def _z(self, *shape):
        return torch.zeros(shape, device=self.dev)

    def keep_logits(self, tok, numbers):
        return self._z(tok.shape[0], tok.shape[1], 3)

    def cq(self, x):
        return self._z(x.shape[0], Q)

    def ck(self, x):
        return self._z(x.shape[0], x.shape[1], Q)

    tq = cq
    tk = ck

    def tnone(self, x):
        return self._z(x.shape[0], 1)

    def mix(self, x):
        return torch.zeros_like(x)

    tmix = mix

    def form(self, x):
        return self._z(x.shape[0], len(FORMS))

    def form_emb(self, i):
        return self._z(i.shape[0], _w())

    def turn(self, x):
        return self._z(x.shape[0], self.S)

    turn_emb = form_emb

    def where(self, h, fmap, extra=None):
        return self._z(h.shape[0], self.C * self.C)

    near_w = 0.0
    empty_b = 0.0

    def join(self, x):
        return torch.zeros_like(x)

    def cmd(self, x):
        return self._z(x.shape[0], _w())

    def verb(self, x):
        return self._z(x.shape[0], len(DC.VERBS))

    def pace(self, x):
        return self._z(x.shape[0], len(DC.PACES))


def null_enc(B, K, G, device):
    W = _w()
    return {"units": torch.zeros((B, K, W), device=device), "tokens": torch.zeros((B, G, W), device=device), "map": None}


class OperativeNet(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder(card)
        self.heads = Heads()

    def forward(self, inp, ctx, act=None, greedy=False, labels=None):
        enc = self.enc(inp)
        return choose(enc, inp, ctx, self.heads, act, greedy, labels), enc


class OperativeCritic(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder(card)
        self.value = nn.Linear(_w(), 1)

    def forward(self, inp):
        return self.value(self.enc(inp)["summary"])[:, 0]


def report(net):
    c = lambda m: sum(p.numel() for p in m.parameters())                    # noqa: E731
    e = net.enc
    out = {"analysis_frozen": c(e.analysis), "squeeze_frozen": c(e.squeeze), "map": c(e.map),
           "record": c(e.record), "tokens": e.number.numel() + c(e.token), "summary": c(e.summary),
           "council": c(e.council), "heads": c(net.heads)}
    out["total"] = c(net)
    out["to_train"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return out
