"""The tactic level's order head of drill T0-2 'strike the pinned one' (tmp/tactic_slice/design_attack_v23.md section 6;
review_attack_v23 C, G): the pointer with a null key is the whole output - for each head-free unit of mine, in slot
order, one categorical over {null} + the rows open for it (null = keep the order in force, row k = 'attack k').

    head_free(fx)        [R, G] alive, not routing, not in melee, no attack order on a valid target, at least one open
                         row (a fresh walk) - the unit-ticks that carry a loss or count in a rate (as tags.tick)
    walk(fx, given)      the walk in slot order (orders.Walk: the cap from the orders in force, an earlier order counts at
                         once): yields g, its open rows [R, K] and pair numbers [R, K, 7] (orders.Facts.pair with the
                         walk's holders at g's turn); fed with given [R, G] rows (-1 none)
    walk_inputs(fx, given)  {open [R, G, K], pair [R, G, K, 7]} of the walk fed with given (the teacher's choices: later
                         slots see them) - what the buffer stores and the teacher-path read asks on
    OrderPointer         score(g, k) = MLP([card_g | own_g | card_k | pair(g, k)]) -> pair_hidden -> 1, score(g, null) =
                         Linear([card_g | own_g]); the last layers zero at the start, the null bias orders.net.null_bias:
                         a fresh network keeps. logits_all - every g at once on stored walk inputs
    choose_orders(enc, fx, pointer, given=None, greedy=True)  the live walk: {k [R, G] the pointer's pick (row, -1 null),
                         fed [R, G] what fed the walk (given, else the picks), logits [R, G, 1 + K], head [R, G]}; with
                         given the picks are still the pointer's own (the teacher-path read: the walk fed with the
                         teacher's earlier choices)
    loss(lg, label, w, dc)  cross-entropy at the label over {null} + the open keys less the don't-care keys dc [M, G, K];
                         weight w [M, G] (0: a dropped tick, a don't-care label, not head-free); a label outside its
                         allowed set gets weight 0 and is counted
    SmallEncoder         the label probe's encoder (section 6 item 3): card + extra columns -> one linear layer to width,
                         my units' own() through a layer from zero added to their rows, one council block of that width,
                         no maps
    ProbeNet             SmallEncoder + OrderPointer
    frozen_analysis()    'Анализ отряда' (185 -> 128) from weights/tactic/card8/card8.pt, frozen
NOT here (stage 5): EncoderV2, TacticNet2, NetPlayer, side.py, ckpt v2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.tactic import cfg
from net.tactic import orders as OR
from net.unit.net import NEG, Block

OWN = 16
PAIR = 7


def c_net():
    return cfg()["orders"]["net"]


def head_free(fx):
    """(module doc) [R, G]."""
    w = OR.Walk(fx)
    opn = torch.stack([w.open(g) for g in range(fx.G)], 1)
    return fx.able & ~fx.g_m & ~fx.striking & opn.any(2)


def walk(fx, given):
    """(module doc) yields (g, open [R, K], pair [R, K, 7]) in slot order; given[:, g] goes to the walk before g + 1."""
    w = OR.Walk(fx)
    for g in range(fx.G):
        op = w.open(g)
        yield g, op, fx.pair(g, w.count)
        k = given[:, g]
        on = (k >= 0) & fx.able[:, g]
        if bool(on.any()):
            w.give(g, k.clamp(min=0), on)


def walk_inputs(fx, given):
    """(module doc)."""
    ops, prs = [], []
    for g, op, pr in walk(fx, given):
        ops.append(op)
        prs.append(pr)
    return {"open": torch.stack(ops, 1), "pair": torch.stack(prs, 1)}


class OrderPointer(nn.Module):
    def __init__(self, width, pair=PAIR, own=OWN, hidden=None, null_bias=None):
        super().__init__()
        h = int(hidden or c_net()["pair_hidden"])
        self.l1 = nn.Linear(2 * width + own + pair, h)
        self.l2 = nn.Linear(h, 1)
        self.null = nn.Linear(width + own, 1)
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.zero_()
            self.null.weight.zero_()
            self.null.bias.fill_(float(c_net()["null_bias"] if null_bias is None else null_bias))

    def logits_all(self, cards, own, g_row, pair, opn):
        """[M, G, 1 + K]: cards [M, K, W] (after the council), own [M, G, OWN], g_row [M, G] my units' rows, pair
        [M, G, K, P], opn [M, G, K] the open keys (closed ones NEG); column 0 the null key."""
        M, K, W = cards.shape
        G = g_row.shape[1]
        cg = cards.gather(1, g_row[..., None].expand(-1, -1, W))                       # [M, G, W]
        x = torch.cat([cg[:, :, None].expand(-1, -1, K, -1), own[:, :, None].expand(-1, -1, K, -1),
                       cards[:, None].expand(-1, G, -1, -1), pair], -1)
        s = self.l2(F.relu(self.l1(x)))[..., 0].masked_fill(~opn, NEG)                 # [M, G, K]
        n = self.null(torch.cat([cg, own], -1))                                         # [M, G, 1]
        return torch.cat([n, s], -1)


def _pick(lg, greedy):
    return lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()


def choose_orders(enc, fx, pointer, given=None, greedy=True, extra_pair=None):
    """(module doc) enc {cards [R, K, W], own [R, G, OWN]}; extra_pair [R, G, K, e] more pair columns (the probe's X)."""
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    head = head_free(fx)
    k_out = torch.full((R, G), -1, dtype=torch.long, device=dev)
    lg_out = torch.full((R, G, 1 + K), NEG, device=dev)

    def one(g, op, pr):
        if extra_pair is not None:
            pr = torch.cat([pr, extra_pair[:, g]], -1)
        lg = pointer.logits_all(enc["cards"], enc["own"][:, g:g + 1], fx.g_row[:, g:g + 1], pr[:, None],
                                op[:, None])[:, 0]                                     # [R, 1 + K]
        lg = torch.where(head[:, g:g + 1], lg, torch.full_like(lg, NEG))
        lg[:, 0] = torch.where(head[:, g], lg[:, 0], torch.zeros_like(lg[:, 0]))       # (not head-free: null only)
        lg_out[:, g] = lg
        c = _pick(lg, greedy)
        k = torch.where(head[:, g] & (c > 0), c - 1, torch.full_like(c, -1))
        k_out[:, g] = k
        return k

    fed = torch.full((R, G), -1, dtype=torch.long, device=dev)
    w = OR.Walk(fx)
    for g in range(G):
        op = w.open(g)
        k = one(g, op, fx.pair(g, w.count))
        f = given[:, g] if given is not None else k
        fed[:, g] = f
        on = (f >= 0) & fx.able[:, g]
        if bool(on.any()):
            w.give(g, f.clamp(min=0), on)
    return {"k": k_out, "fed": fed, "logits": lg_out, "head": head}


def loss(lg, label, w, dc):
    """(module doc) -> (mean loss, {n, out_of_set}): lg [M, G, 1 + K], label [M, G] row (-1 null), w [M, G], dc
    [M, G, K]."""
    allowed = torch.cat([torch.ones_like(dc[..., :1]), ~dc], -1) & (lg > NEG / 2)
    lgm = lg.masked_fill(~allowed, NEG)
    tgt = (label + 1).clamp(min=0)
    ok = allowed.gather(-1, tgt[..., None])[..., 0]
    w = w * ok.float()
    ce = -F.log_softmax(lgm, -1).gather(-1, tgt[..., None])[..., 0]
    tot = w.sum().clamp(min=1e-9)
    return (w * ce).sum() / tot, {"n": int((w > 0).sum()), "out_of_set": int(((~ok) & (label >= 0)).sum())}


class SmallEncoder(nn.Module):
    def __init__(self, extra, width=64, heads=4, ff=128, card=128):
        super().__init__()
        self.inp = nn.Linear(card + extra, width)
        self.own = nn.Linear(OWN, width)
        with torch.no_grad():
            self.own.weight.zero_()
            self.own.bias.zero_()
        self.block = Block(width, heads, ff)

    def forward(self, card, cols, mask, own, g_row, g_on):
        """[M, K, W]: card [M, K, 128], cols [M, K, extra], mask [M, K], own [M, G, OWN], g_row [M, G], g_on [M, G]
        my unit there."""
        x = self.inp(torch.cat([card, cols], -1))
        add = self.own(own) * g_on[..., None].float()                                   # [M, G, W]
        x = x.scatter_add(1, g_row[..., None].expand(-1, -1, x.shape[-1]), add)
        zero = torch.zeros((), device=x.device)
        x = torch.where(mask[..., None], x, zero)
        return torch.where(mask[..., None], self.block(x, x, mask), zero)


class ProbeNet(nn.Module):
    """The label probe's model (module doc): extra row columns, extra pair columns."""

    def __init__(self, extra, extra_pair=0, width=64, heads=4, ff=128):
        super().__init__()
        self.enc = SmallEncoder(extra, width, heads, ff)
        self.ptr = OrderPointer(width, pair=PAIR + extra_pair)

    def forward(self, b):
        """[M, G, 1 + K] of a batch {card, cols, mask, own, g_row, g_on, pair, open}."""
        cards = self.enc(b["card"], b["cols"], b["mask"], b["own"], b["g_row"], b["g_on"])
        return self.ptr.logits_all(cards, b["own"], b["g_row"], b["pair"], b["open"])


def frozen_analysis(device="cpu", path=None):
    """(module doc) a UnitAnalysis in eval mode, no grad."""
    from net.tactic import decide as DC
    from net.tactic import net as NT
    a = NT.UnitAnalysis()
    ck = torch.load(path or DC.CARD_CKPT, map_location="cpu", weights_only=False)
    a.load_state_dict(ck["analysis"])
    for p in a.parameters():
        p.requires_grad_(False)
    return a.to(device).eval()
