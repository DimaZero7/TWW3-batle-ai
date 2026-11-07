"""The strategist's network (plan.md "Выход и порядок голов").

    Encoder      'Анализ отряда' 185 -> 256 -> 128 (tactic/net.UnitAnalysis, frozen from the tactician's card8.pt, as at
                 the operative); 'Запись отряда' the card + own / enemy / shoots 3 -> 128; the deployment map 8 -> 32 ->
                 64 -> 64 (tactic/net.Stack: the vector 128, the map 64 x 32 x 32); 6 group tokens - a learned vector of
                 each number; the summary: the map's vector + the numbers 2 + the task 7 -> 128; the council
                 tactic/decide.Council over the rows (40), the tokens (6) and the summary
    Heads        per own unit in slot order a pointer at the 6 groups (the token learns who joined it: join; a full group,
                 10, closed); per group with members, in its number's order: form (square / line) -> turn (16 sectors
                 from the bearing my zone -> the enemy's zone) -> cell (1024: the zone's centre; closed: the zone not
                 wholly inside my deployment zone, or overlapping a zone placed already: separating axes); then the
                 battle's verb (4, open: config verbs_open) and pace (3, open: paces_open) - the operative's command
    choose()     the walk: sampling, the update (act: the same choices), the teacher's term (labels), the formula's walk
                 (bias: the formula's wishes as logits, greedy)
    StrategistNet, StrategistCritic (its own encoder, the value of the summary)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.operative import step as SP
from tools.nn.strategist import cfg
from tools.nn.strategist import inputs as IN
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import net as NT
from tools.nn.tactic import windows as W
from tools.nn.unit.net import NEG

FORMS = ("square", "line")
SQUARE, LINE = 0, 1
Q = 64
HEADS = ("comp", "form", "turn", "cell", "verb", "pace")
ORDER = len(DC.VERBS) + len(DC.PACES)


def _w():
    return int(tcfg()["council"]["width"])


class Encoder(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        W_ = _w()
        G = int(cfg()["groups"])
        self.analysis = NT.UnitAnalysis()
        self.record = nn.Linear(int(tcfg()["analysis"]["out"]) + 3, W_)
        self.number = nn.Parameter(torch.randn(G, W_) * 0.1)
        self.map = NT.Stack((len(IN.LAYERS), 32, 64, 64), 128)
        self.summary = nn.Linear(128 + 2 + ORDER, W_)
        self.council = DC.Council()
        if card:
            ck = torch.load(DC.CARD_CKPT, map_location="cpu", weights_only=False)
            self.analysis.load_state_dict(ck["analysis"])
            for p in self.analysis.parameters():
                p.requires_grad_(False)

    def forward(self, inp):
        mask = inp["mask"]
        zero = torch.zeros((), device=mask.device)
        cards = self.analysis(inp["rows"], mask)
        flags = torch.stack([inp["own"], inp["enemy"], inp["shooter"]], -1).float()
        rec = torch.where(mask[..., None], F.relu(self.record(torch.cat([cards, flags], -1))), zero)
        vec, fmap = self.map(inp["map"])
        s = self.summary(torch.cat([vec, inp["numbers"], inp["command"]], 1))
        B, K = mask.shape
        tok = self.number[None].expand(B, -1, -1)
        x = torch.cat([rec, tok], 1)
        mm = torch.cat([mask, torch.ones(tok.shape[:2], dtype=torch.bool, device=mask.device)], 1)
        after, sm = self.council(x, s, mm)
        return {"units": after[:, :K], "tokens": after[:, K:], "summary": sm, "map": fmap}


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        c = cfg()
        W_ = _w()
        S, C = int(c["sectors"]), int(c["cells"])
        self.cq, self.ck = nn.Linear(W_, Q), nn.Linear(W_, Q)
        self.join = nn.Linear(W_, W_)
        self.cmd = nn.Linear(ORDER, W_)
        self.mix = nn.Linear(W_, W_)
        self.form = nn.Linear(W_, len(FORMS))
        self.form_emb = nn.Embedding(len(FORMS), W_)
        self.turn = nn.Linear(W_, S)
        self.turn_emb = nn.Embedding(S, W_)
        self.where_core = nn.Linear(W_, 64)
        self.where_a = nn.Conv2d(64 + 2 + 1, 32, 3, padding=1)
        self.where_b = nn.Conv2d(32, 1, 1)
        self.verb = nn.Linear(W_, len(DC.VERBS))
        self.pace = nn.Linear(W_, len(DC.PACES))
        k = (torch.arange(C, dtype=torch.float32) + 0.5) / C * 2 - 1
        self.register_buffer("coords", torch.stack([k[None, :].expand(C, C), -k[:, None].expand(C, C)])[None],
                             persistent=False)

    def where(self, h, fmap, extra):
        y = F.relu(fmap + self.where_core(h)[:, :, None, None])
        y = torch.cat([y, self.coords.expand(y.shape[0], -1, -1, -1), extra], 1)
        return self.where_b(F.relu(self.where_a(y)))[:, 0].flatten(1)


class NullHeads:
    """Heads of zeros: the walk with given choices or the formula's bias, and no network."""

    def __init__(self, B, device):
        c = cfg()
        self.B, self.dev = B, device
        self.S, self.C = int(c["sectors"]), int(c["cells"])

    def _z(self, *shape):
        return torch.zeros(shape, device=self.dev)

    def cq(self, x):
        return self._z(x.shape[0], Q)

    def ck(self, x):
        return self._z(x.shape[0], x.shape[1], Q)

    def join(self, x):
        return torch.zeros_like(x)

    mix = join

    def cmd(self, x):
        return self._z(x.shape[0], _w())

    def form(self, x):
        return self._z(x.shape[0], len(FORMS))

    def form_emb(self, i):
        return self._z(i.shape[0], _w())

    def turn(self, x):
        return self._z(x.shape[0], self.S)

    turn_emb = form_emb

    def where(self, h, fmap, extra):
        return self._z(h.shape[0], self.C * self.C)

    def verb(self, x):
        return self._z(x.shape[0], len(DC.VERBS))

    def pace(self, x):
        return self._z(x.shape[0], len(DC.PACES))


def null_enc(B, K, G, device):
    W_ = _w()
    return {"units": torch.zeros((B, K, W_), device=device), "tokens": torch.zeros((B, G, W_), device=device),
            "summary": torch.zeros((B, W_), device=device), "map": None}


class Ctx:
    """What the walk needs besides the network: the axes, both zones, the cells' centres."""

    def __init__(self, axes, mine_zone, enemy_zone):
        self.axes, self.mine, self.enemy = axes, mine_zone, enemy_zone
        self.cx, self.cz = SP.cells(axes)
        self.base = ZS.bearing(mine_zone["x"], mine_zone["z"], enemy_zone["x"], enemy_zone["z"])


def _pick(lg, given, greedy):
    if given is not None:
        return given
    return lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()


def _lp_ent(lg, x):
    d = torch.distributions.Categorical(logits=lg)
    p = d.probs
    return d.log_prob(x), -torch.where(p > 0, p * d.logits, torch.zeros_like(p)).sum(-1)


def _soft_ce(lg, q):
    return -(torch.where(q > 0, q * F.log_softmax(lg, -1), torch.zeros_like(q))).sum(-1)


def _mean(x, sel):
    w = sel.float()
    return torch.einsum("bgk,bkw->bgw", w, x) / w.sum(2, keepdim=True).clamp(min=1)


def open_mask(names, all_names, R, device):
    return torch.tensor([n in names for n in all_names], device=device)[None].expand(R, -1)


def choose(enc, inp, ctx, heads, act=None, greedy=False, labels=None, bias=None):
    """The walk (module doc). Returns {act: {comp [B, K], form, turn, cell [B, G], verb, pace [B]}, lp [B], ent {head: [B]},
    plan: {members [B, G, K] (rows), on [B, G], n, cx, cz, heading, line [B, G], verb, pace [B]}, teach}. bias (the
    formula): comp [B, K, G], form [B, G, 2], turn [B, G, S], goal_x, goal_z [B, G], verb [B, 4] - added to the
    logits (the cell's: minus the squared distance of the cell from the goal)."""
    c = cfg()
    G = int(c["groups"])
    S, C = int(c["sectors"]), int(c["cells"])
    B, K = inp["mask"].shape
    dev = inp["mask"].device
    units, tok = enc["units"], enc["tokens"]
    lp = torch.zeros(B, device=dev)
    ent = {h: torch.zeros(B, device=dev) for h in HEADS}
    teach = {"term": torch.zeros((), device=dev), "n": 0.0, **{f"{h}_ok": 0.0 for h in HEADS},
             **{f"{h}_n": 0.0 for h in HEADS}}
    A = {k: (act[k] if act is not None else None) for k in HEADS}
    out = {"comp": torch.full((B, K), -1, dtype=torch.long, device=dev),
           **{k: torch.zeros((B, G), dtype=torch.long, device=dev) for k in ("form", "turn", "cell")}}

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
            best = lg.argmax(-1)
            hit = (q.gather(1, best[:, None])[:, 0] >= 0.5 * q.amax(-1)) if hard is None else best == hard
            teach[f"{name}_ok"] += float((hit & use).sum())
            teach[f"{name}_n"] += float(use.sum())

    cmd = heads.cmd(inp["command"])
    # 1. the members, slot order
    own = inp["own"]
    count = torch.zeros((B, G), dtype=torch.long, device=dev)
    full = int(tcfg()["max_group"])
    tok_c = tok + cmd[:, None]
    rg = torch.full((B, K), -1, dtype=torch.long, device=dev)
    for j in range(K):
        a = own[:, j]
        if not bool(a.any()):
            continue
        lg = torch.einsum("bq,bgq->bg", heads.cq(units[:, j]), heads.ck(tok_c)) / Q ** 0.5
        if bias is not None:
            lg = lg + bias["comp"][:, j]
        lg = lg.masked_fill(count >= full, NEG)
        cj = _pick(lg, None if A["comp"] is None else A["comp"][:, j].clamp(min=0), greedy)
        score("comp", lg, cj, a, hard=None if labels is None else labels["comp"][:, j])
        rg[:, j] = torch.where(a, cj, rg[:, j])
        count.scatter_add_(1, cj[:, None], a.long()[:, None])
        tok_c = tok_c + F.one_hot(cj, G).float()[..., None] * (heads.join(units[:, j]) * a.float()[:, None])[:, None]
    out["comp"] = rg
    mem = (rg[:, None, :] == torch.arange(G, device=dev)[None, :, None]) & own[:, None, :]       # [B, G, K]
    n = mem.sum(2)
    on = n > 0
    # 2. the groups' zones, in the numbers' order
    plan = {k: torch.zeros((B, G), device=dev) for k in ("cx", "cz", "heading", "r", "front", "depth")}
    plan["line"] = torch.zeros((B, G), dtype=torch.bool, device=dev)
    placed = torch.zeros((B, G), dtype=torch.bool, device=dev)
    cell_m = float(c["cell_m"])
    cellv = torch.full((B,), cell_m, device=dev)
    no_cell = torch.zeros(B, dtype=torch.long, device=dev)
    for g in range(G):
        a = on[:, g]
        if not bool(a.any()):
            continue
        L = None if labels is None else {h: labels[h][:, g] for h in ("form", "turn", "cell")}
        h = tok_c[:, g] + heads.mix(_mean(units, mem[:, g:g + 1])[:, 0])
        lf = heads.form(h)
        if bias is not None:
            lf = lf + bias["form"][:, g]
        fo = _pick(lf, None if A["form"] is None else A["form"][:, g], greedy)
        score("form", lf, fo, a, hard=None if L is None else L["form"])
        line = fo == LINE
        h = h + heads.form_emb(fo)
        lr = heads.turn(h)
        if bias is not None:
            lr = lr + bias["turn"][:, g]
        tr = _pick(lr, None if A["turn"] is None else A["turn"][:, g], greedy)
        qt = None
        if L is not None:
            sg = float(c["teacher"]["sigma_sectors"])
            dd = (torch.arange(S, device=dev)[None] - L["turn"][:, None]).abs()
            dd = torch.minimum(dd, S - dd).float()
            qt = torch.exp(-dd ** 2 / (2 * sg * sg))
            qt = qt / qt.sum(1, keepdim=True)
        score("turn", lr, tr, a, q=qt)
        heading = torch.remainder(ctx.base + tr.float() * (360.0 / S), 360.0)
        h = h + heads.turn_emb(tr)
        front, depth = SP.size(n[:, g].clamp(min=1), line)
        r = ZS.radius(front, depth)
        ok = ZS.cells_ok(ctx.cx, ctx.cz, ctx.mine, front, depth, heading)
        ok = ok & ZS.apart(ctx.cx, ctx.cz, front, depth, heading, plan, placed)
        none = ~ok.any(1) & a
        no_cell = no_cell + none.long()
        ok = ok | ~ok.any(1, keepdim=True)
        areas = {"x": plan["cx"], "z": plan["cz"], "b": plan["heading"], "front": plan["front"],
                 "depth": plan["depth"], "on": placed}
        lay = W.area_layer(areas, ctx.axes.cx, ctx.axes.cz, ctx.axes.heading, cellv, C)
        extra = lay[:, None]
        lw = heads.where(h, enc["map"], extra)
        if bias is not None:
            d2 = (ctx.cx - bias["goal_x"][:, g, None]) ** 2 + (ctx.cz - bias["goal_z"][:, g, None]) ** 2
            lw = lw - d2 / (cell_m * cell_m)
        lw = lw.masked_fill(~ok, NEG)
        cl = _pick(lw, None if A["cell"] is None else A["cell"][:, g], greedy)
        qc = None
        if L is not None:
            sg = float(c["teacher"]["sigma_cells"])
            r0, c0 = torch.div(L["cell"], C, rounding_mode="floor"), L["cell"] % C
            ar = torch.arange(C * C, device=dev)
            d2 = (torch.div(ar, C, rounding_mode="floor")[None] - r0[:, None]) ** 2 + ((ar % C)[None] - c0[:, None]) ** 2
            qc = torch.exp(-d2.float() / (2 * sg * sg)) * ok.float()
            qc = qc / qc.sum(1, keepdim=True).clamp(min=1e-12)
        score("cell", lw, cl, a, q=qc)
        for name, v in (("form", fo), ("turn", tr), ("cell", cl)):
            out[name][:, g] = torch.where(a, v, out[name][:, g])
        zx, zz = ctx.cx.gather(1, cl[:, None])[:, 0], ctx.cz.gather(1, cl[:, None])[:, 0]
        for k, v in (("cx", zx), ("cz", zz), ("heading", heading), ("r", r), ("front", front), ("depth", depth)):
            plan[k][:, g] = torch.where(a, v, plan[k][:, g])
        plan["line"][:, g] = a & line
        placed[:, g] = a
    # 3. the operative's command
    hb = enc["summary"] + cmd
    vopen = open_mask(c["verbs_open"], DC.VERBS, B, dev)
    popen = open_mask(c["paces_open"], DC.PACES, B, dev)
    lv = heads.verb(hb)
    if bias is not None:
        lv = lv + bias["verb"]
    lv = lv.masked_fill(~vopen, NEG)
    vb = _pick(lv, A["verb"], greedy)
    yes = torch.ones(B, dtype=torch.bool, device=dev)
    score("verb", lv, vb, yes & (vopen.sum(1) > 1), hard=None if labels is None else labels["verb"])
    lpc = heads.pace(hb).masked_fill(~popen, NEG)
    pc = _pick(lpc, A["pace"], greedy)
    score("pace", lpc, pc, yes & (popen.sum(1) > 1), hard=None if labels is None else labels["pace"])
    out["verb"], out["pace"] = vb, pc
    plan.update({"members": mem, "on": on, "n": n, "verb": vb, "pace": pc, "no_cell": no_cell})
    return {"act": out, "lp": lp, "ent": ent, "plan": plan, "teach": teach}


class StrategistNet(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder(card)
        self.heads = Heads()

    def forward(self, inp, ctx, act=None, greedy=False, labels=None):
        enc = self.enc(inp)
        return choose(enc, inp, ctx, self.heads, act, greedy, labels)


class StrategistCritic(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder(card)
        self.value = nn.Linear(_w(), 1)

    def forward(self, inp):
        return self.value(self.enc(inp)["summary"])[:, 0]


def report(net):
    c = lambda m: sum(p.numel() for p in m.parameters())                    # noqa: E731
    e = net.enc
    out = {"analysis_frozen": c(e.analysis), "record": c(e.record), "map": c(e.map), "tokens": e.number.numel(),
           "summary": c(e.summary), "council": c(e.council), "heads": c(net.heads), "total": c(net)}
    out["to_train"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return out
