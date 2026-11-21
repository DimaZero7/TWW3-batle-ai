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
    net_inputs(v)        the network's inputs of a View beyond inputs.build (v.inp): the extra row columns cols [R, K,
                         len(cols.D2)] (cols.row_cols, D2's list - review_attack_v23 M), own() [R, G, 16], g_row, g_on
                         [R, G] (my unit g has its own row)
    EncoderV2            decide.Encoder (the input, the summary, the council - untouched) + the extra columns beside each
                         card through a layer from zero, my units' own() through a layer from zero added to their cards,
                         both before the council
    TacticNet2           EncoderV2 + OrderPointer. Frozen: 'Анализ отряда' and the squeeze (card8.pt); from zero: the
                         maps' convolutions, the summary, the council, the new layers, the pointer - a fresh one keeps.
                         act(v, greedy, given) - the live walk on a View (choose_orders); the network's player:
                         pin.NetPlayer; the checkpoint: ckpt.save_v2 / restore (kind tactic_v2)
Drill T0-2 'side of the strike' (tmp/drill2/design_merged.md decision 2, design_opus.md section 2):
    side_feats(fx)       [R, G, K, 11] the side head's columns of the pair (g, k): cols.side_pairs (8: per flank of k the
                         way's clearance to enemies / own, the extra path, |gP|) and k's cols.SIDE_ROW (3: lateral
                         clearance left / right, the anvil's off-facing angle); walk_inputs(..., approach) adds held [R,
                         G, K, 2] - k's left / right flank held by a striker at g's turn (the walk's own state, as count)
    SideHead             a conditional 3-way head {front, left, right} of the target's flank for the pair the pointer
                         picked: MLP([card_g | own_g (19) | card_k | pair(g, k) | side_feats + held | side_thr of
                         them (the columns re-centred on the rule's thresholds)]) -> pair_hidden ->
                         3, the last layer zero and its bias (null_bias, 0, 0): a fresh head says 'front'
    side_loss(lg, label, w)  cross-entropy at the approach label [N] (0 front, 1 left, 2 right) weighted w [N]
    EncoderV3, PointerV3 EncoderV2 / OrderPointer with own() of 19 (the approach one-hot) and the 3 S2 row columns: the
                         new inputs through their own layers from zero ADDED to the old ones - a v2 network's weights in
                         the old layers give bit for bit the v2 network's numbers
    TacticNet3           EncoderV3 + PointerV3 + SideHead; act(v, greedy, given, given_ap) -> choose_orders' dict +
                         approach [R, G] (0 front where no attack); load_v2(state) - a tactic_v2 network's weights into
                         the old parts (the new ones stay at their start); the checkpoint: ckpt.save_v3 / restore
                         (kind tactic_v3)
    net_inputs3(v)       net_inputs + own() of 19 + s2 [R, K, 3] (cols.side_rows)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.tactic import cfg
from net.tactic import cols as CL
from net.tactic import decide as DC
from net.tactic import orders as OR
from net.unit.net import NEG, Block

OWN = 16
OWN2 = OWN + len(OR.APPROACHES)          # own() with the approach one-hot (orders.own(approach=True))
PAIR = 7
SIDE_X = len(CL.SIDE_PAIR) + len(CL.SIDE_ROW) + 2      # side_feats + held


def c_net():
    return cfg()["orders"]["net"]


def head_free(fx):
    """(module doc) [R, G]."""
    w = OR.Walk(fx)
    opn = torch.stack([w.open(g) for g in range(fx.G)], 1)
    return fx.able & ~fx.g_m & ~fx.striking & opn.any(2)


def held_of(w):
    """[R, K, 2] float: each row's left / right flank held by a striker of the walk w now (orders.Walk.held)."""
    return w.held[..., OR.LEFT:OR.RIGHT + 1].float()


def walk(fx, given, approach=None, with_held=False):
    """(module doc) yields (g, open [R, K], pair [R, K, 7]) in slot order (with_held: + held_of [R, K, 2]); given[:, g]
    goes to the walk before g + 1, with its approach[:, g] (drill T0-2; None: the side it stands on)."""
    w = OR.Walk(fx)
    for g in range(fx.G):
        op = w.open(g)
        yield (g, op, fx.pair(g, w.count)) + ((held_of(w),) if with_held else ())
        k = given[:, g]
        on = (k >= 0) & fx.able[:, g]
        if bool(on.any()):
            w.give(g, k.clamp(min=0), on, approach=None if approach is None else approach[:, g])


def walk_inputs(fx, given, approach=None):
    """(module doc); with approach [R, G] (the teacher's, drill T0-2) the walk is fed with it and held [R, G, K, 2] is
    returned too."""
    ops, prs, hs = [], [], []
    for x in walk(fx, given, approach, with_held=approach is not None):
        ops.append(x[1])
        prs.append(x[2])
        if approach is not None:
            hs.append(x[3])
    out = {"open": torch.stack(ops, 1), "pair": torch.stack(prs, 1)}
    if approach is not None:
        out["held"] = torch.stack(hs, 1)
    return out


def side_feats(fx):
    """(module doc) [R, G, K, 11]."""
    return torch.cat([CL.side_pairs(fx), CL.side_rows(fx)[:, None].expand(-1, fx.G, -1, -1)], -1)


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


def choose_orders(enc, fx, pointer, given=None, greedy=True, extra_pair=None, side=None, given_ap=None):
    """(module doc) enc {cards [R, K, W], own [R, G, OWN]}; extra_pair [R, G, K, e] more pair columns (the probe's X).
    side - a SideHead (drill T0-2): for each g the approach of the pair (g, its pick) - + approach [R, G] (0 where no
    pick), side_logits [R, G, 3]; the walk is fed with the approach too (given_ap [R, G] with given: the teacher's)."""
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    head = head_free(fx)
    k_out = torch.full((R, G), -1, dtype=torch.long, device=dev)
    lg_out = torch.full((R, G, 1 + K), NEG, device=dev)
    if side is not None:
        sx = side_feats(fx)
        ap_out = torch.zeros((R, G), dtype=torch.long, device=dev)
        sl_out = torch.zeros((R, G, len(OR.APPROACHES)), device=dev)

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

    def one_side(g, k, pr, held):
        kc = k.clamp(min=0)
        sl = side_logits(side, enc["cards"], enc["own"][:, g], fx.g_row[:, g], kc, pr, sx[:, g], held)
        a = _pick(sl, greedy)
        ap_out[:, g] = torch.where(k >= 0, a, torch.zeros_like(a))
        sl_out[:, g] = sl
        return ap_out[:, g]

    fed = torch.full((R, G), -1, dtype=torch.long, device=dev)
    w = OR.Walk(fx)
    for g in range(G):
        op = w.open(g)
        pr = fx.pair(g, w.count)
        k = one(g, op, pr)
        f = given[:, g] if given is not None else k
        fed[:, g] = f
        a = None
        if side is not None:
            ap = one_side(g, k if given is None else f, pr, held_of(w))
            a = given_ap[:, g] if (given is not None and given_ap is not None) else ap
        on = (f >= 0) & fx.able[:, g]
        if bool(on.any()):
            w.give(g, f.clamp(min=0), on, approach=a)
    out = {"k": k_out, "fed": fed, "logits": lg_out, "head": head}
    if side is not None:
        out.update(approach=ap_out, side_logits=sl_out)
    return out


def side_logits(head, cards, own_g, g_row, k, pair, sx, held):
    """[R, 3] the SideHead's logits of the pairs (g, k) [R] (one g): cards [R, K, W], own_g [R, OWN2], g_row [R], k [R],
    pair [R, K, 7], sx [R, K, 11], held [R, K, 2]."""
    W = cards.shape[-1]
    cg = cards.gather(1, g_row[:, None, None].expand(-1, 1, W))[:, 0]
    ck = cards.gather(1, k[:, None, None].expand(-1, 1, W))[:, 0]
    at = lambda t: t.gather(1, k[:, None, None].expand(-1, 1, t.shape[-1]))[:, 0]      # noqa: E731
    return head(cg, own_g, ck, at(pair), torch.cat([at(sx), at(held)], -1))


THR_X = 10        # side_thr's columns


def side_thr(x):
    """[.., THR_X] the side columns x [.., SIDE_X] re-centred on the rule's thresholds (config orders.side), 1 unit 5 m
    (5 deg), clamped to +-10: per flank l / r the way's clearance to enemies and to own units, the extra path; the
    lateral clearance l / r; the anvil's off-facing angle; and the extra path's left - right (1 unit = 1 m) - functions
    of the existing columns, no new fact (the label probe's recipe R3, tmp/t02/fix/b/probe_recipes.md: held-out followed
    0.959 / 0.937 / 0.929 against 0.935 / 0.867 / 0.861 without it)."""
    c = CL.c_side()
    pos = float(cfg()["pos_m"])
    cl = lambda t: t.clamp(-10.0, 10.0)                                              # noqa: E731
    out = []
    for o in (0, 4):
        out += [cl((x[..., o] - 1) * float(c["clear_enemy_m"]) / 5), cl((x[..., o + 1] - 1) * float(c["clear_own_m"]) / 5),
                cl((x[..., o + 2] - float(c["extra_cap_m"]) / pos) * pos / 5)]
    out += [cl((x[..., 8] - 1) * float(c["band_lat_m"]) / 5), cl((x[..., 9] - 1) * float(c["band_lat_m"]) / 5),
            cl((x[..., 10] - float(c["frontal_deg"]) / 180) * 180 / 5), cl((x[..., 2] - x[..., 6]) * pos / 1.0)]
    return torch.stack(out, -1)


class SideHead(nn.Module):
    """(module doc); its MLP reads x together with side_thr(x)."""

    def __init__(self, width, own=OWN2, pair=PAIR, extra=SIDE_X, hidden=None, front_bias=None):
        super().__init__()
        h = int(hidden or c_net()["pair_hidden"])
        self.l1 = nn.Linear(2 * width + own + pair + extra + THR_X, h)
        self.l2 = nn.Linear(h, len(OR.APPROACHES))
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.zero_()
            # (the null key's start, config orders.net.null_bias: e^4 ~ 55 against each flank - a fresh head 'front')
            self.l2.bias[OR.FRONT] = float(c_net()["null_bias"] if front_bias is None else front_bias)

    def forward(self, cg, own, ck, pair, x):
        return self.l2(F.relu(self.l1(torch.cat([cg, own, ck, pair, x, side_thr(x)], -1))))


def side_loss(lg, label, w):
    """(module doc) -> (mean loss, n): lg [N, 3], label [N] (0-2), w [N]."""
    ce = F.cross_entropy(lg, label.clamp(min=0), reduction="none")
    w = w * (label >= 0).float()
    return (w * ce).sum() / w.sum().clamp(min=1e-9), int((w > 0).sum())


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
    def __init__(self, extra, width=64, heads=4, ff=128, card=128, own=OWN):
        super().__init__()
        self.inp = nn.Linear(card + extra, width)
        self.own = nn.Linear(own, width)
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


class SideProbe(nn.Module):
    """Drill T0-2's label probe (METHOD step 4): SmallEncoder on the card + extra row columns, own() of 19, and a
    SideHead on the labelled pairs - the side the teacher gave its attack (g, k)."""

    def __init__(self, extra, width=64, heads=4, ff=128):
        super().__init__()
        self.enc = SmallEncoder(extra, width, heads, ff, own=OWN2)
        self.side = SideHead(width, hidden=2 * width)

    def forward(self, b):
        """[M, G, 3] of a batch {card, cols, mask, own, g_row, g_on, k [M, G] (clamped), pair_k [M, G, 7], x_k [M, G,
        SIDE_X]}."""
        cards = self.enc(b["card"], b["cols"], b["mask"], b["own"], b["g_row"], b["g_on"])
        W = cards.shape[-1]
        G = b["g_row"].shape[1]
        cg = cards.gather(1, b["g_row"][..., None].expand(-1, -1, W))
        ck = cards.gather(1, b["k"].clamp(min=0)[..., None].expand(-1, -1, W))
        return self.side(cg, b["own"][:, :G], ck, b["pair_k"], b["x_k"])


def d2_cols():
    """The names of the network's extra row columns (cols.D2: one list) and their indices into cols.COLS."""
    return CL.D2, tuple(CL.COLS.index(n) for n in CL.D2)


def g_on_of(fx):
    """[R, G] my unit g has its own row (the row g_row holds its slot) - the network's and the probe's one rule."""
    return (fx.my >= 0) & (fx.slot.gather(1, fx.g_row) == fx.my) & fx.mask.gather(1, fx.g_row)


def net_inputs(v):
    """(module doc) {cols, own, g_row, g_on} of a View."""
    fx = v.fx
    _, idx = d2_cols()
    cols = CL.row_cols(fx)[..., list(idx)]
    own = OR.own(v.order, v.ctx.prev, fx, v.now)
    return {"cols": cols, "own": own, "g_row": fx.g_row, "g_on": g_on_of(fx)}


class EncoderV2(DC.Encoder):
    """(module doc) forward(inp, order, cols, own, g_row, g_on) -> {cards, summary, mask, slot, map, own}."""

    def __init__(self):
        super().__init__()
        W = int(cfg()["council"]["width"])
        self.col_names = d2_cols()[0]
        self.cols = nn.Linear(len(self.col_names), W)
        self.own = nn.Linear(OWN, W)
        for layer in (self.cols, self.own):
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()

    def forward(self, inp, order, cols, own, g_row, g_on):
        o = self.inp(inp)
        s = self.summary(torch.cat([o["map_vec"], o["numbers"], order], 1))
        x = o["cards"] + self.cols(cols)
        add = self.own(own) * g_on[..., None].float()                                   # [R, G, W]
        x = x.scatter_add(1, g_row[..., None].expand(-1, -1, x.shape[-1]), add)
        x = torch.where(o["mask"][..., None], x, torch.zeros((), device=x.device))
        cards, sm = self.council(x, s, o["mask"])
        return {"cards": cards, "summary": sm, "mask": o["mask"], "slot": o["slot"], "map": o["map_near"], "own": own}


class TacticNet2(nn.Module):
    """(module doc)"""

    def __init__(self, card=True):
        super().__init__()
        self.enc = EncoderV2()
        self.ptr = OrderPointer(int(cfg()["council"]["width"]))
        if card:
            self.enc.load_card()

    def encode(self, v, x=None):
        """The encoder on a View (v.inp, v.command); x - net_inputs(v) if already built."""
        x = x or net_inputs(v)
        return self.enc(v.inp, v.command, x["cols"], x["own"], x["g_row"], x["g_on"])

    def act(self, v, greedy=True, given=None):
        """choose_orders on a View: {k, fed, logits, head} (module doc); given [R, G] rows - the walk fed with them (the
        teacher's choices: the update and the teacher-path read)."""
        return choose_orders(self.encode(v), v.fx, self.ptr, given=given, greedy=greedy)

    def frozen(self):
        return list(self.enc.inp.analysis.parameters()) + list(self.enc.inp.squeeze.parameters())


def net_inputs3(v):
    """(module doc) {cols, own [R, G, 19], g_row, g_on, s2 [R, K, 3]} of a View."""
    x = net_inputs(v)
    x["own"] = OR.own(v.order, v.ctx.prev, v.fx, v.now, approach=True)
    x["s2"] = CL.side_rows(v.fx)
    return x


class EncoderV3(EncoderV2):
    """(module doc) forward(inp, order, cols, own [.., 19], g_row, g_on, s2 [R, K, 3])."""

    def __init__(self):
        super().__init__()
        W = int(cfg()["council"]["width"])
        self.own_ap = nn.Linear(OWN2 - OWN, W)
        self.side_rows = nn.Linear(len(CL.SIDE_ROW), W)
        for layer in (self.own_ap, self.side_rows):
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()

    def forward(self, inp, order, cols, own, g_row, g_on, s2):
        o = self.inp(inp)
        s = self.summary(torch.cat([o["map_vec"], o["numbers"], order], 1))
        x = o["cards"] + self.cols(cols) + self.side_rows(s2)
        add = (self.own(own[..., :OWN]) + self.own_ap(own[..., OWN:])) * g_on[..., None].float()
        x = x.scatter_add(1, g_row[..., None].expand(-1, -1, x.shape[-1]), add)
        x = torch.where(o["mask"][..., None], x, torch.zeros((), device=x.device))
        cards, sm = self.council(x, s, o["mask"])
        return {"cards": cards, "summary": sm, "mask": o["mask"], "slot": o["slot"], "map": o["map_near"], "own": own}


class PointerV3(OrderPointer):
    """(module doc) OrderPointer on own() of 19: the approach one-hot through l1_ap / null_ap from zero."""

    def __init__(self, width):
        super().__init__(width)
        h = self.l1.out_features
        self.l1_ap = nn.Linear(OWN2 - OWN, h)
        self.null_ap = nn.Linear(OWN2 - OWN, 1)
        for layer in (self.l1_ap, self.null_ap):
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()

    def logits_all(self, cards, own, g_row, pair, opn):
        M, K, W = cards.shape
        G = g_row.shape[1]
        o16, oa = own[..., :OWN], own[..., OWN:]
        cg = cards.gather(1, g_row[..., None].expand(-1, -1, W))
        x = torch.cat([cg[:, :, None].expand(-1, -1, K, -1), o16[:, :, None].expand(-1, -1, K, -1),
                       cards[:, None].expand(-1, G, -1, -1), pair], -1)
        hdn = self.l1(x) + self.l1_ap(oa)[:, :, None]
        s = self.l2(F.relu(hdn))[..., 0].masked_fill(~opn, NEG)
        n = self.null(torch.cat([cg, o16], -1)) + self.null_ap(oa)
        return torch.cat([n, s], -1)


class TacticNet3(nn.Module):
    """(module doc)"""

    def __init__(self, card=True):
        super().__init__()
        W = int(cfg()["council"]["width"])
        self.enc = EncoderV3()
        self.ptr = PointerV3(W)
        self.side = SideHead(W)
        if card:
            self.enc.load_card()

    def encode(self, v, x=None):
        x = x or net_inputs3(v)
        return self.enc(v.inp, v.command, x["cols"], x["own"], x["g_row"], x["g_on"], x["s2"])

    def act(self, v, greedy=True, given=None, given_ap=None):
        return choose_orders(self.encode(v), v.fx, self.ptr, given=given, greedy=greedy, side=self.side,
                             given_ap=given_ap)

    def frozen(self):
        return list(self.enc.inp.analysis.parameters()) + list(self.enc.inp.squeeze.parameters())

    NEW = ("enc.own_ap.", "enc.side_rows.", "ptr.l1_ap.", "ptr.null_ap.", "side.")

    def load_v2(self, sd):
        """A tactic_v2 network's state into the old parts (module doc); every key of sd must land, the missing ones
        are exactly the new layers."""
        r = self.load_state_dict(sd, strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        bad = [k for k in r.missing_keys if not k.startswith(self.NEW)]
        assert not bad, bad
        return self


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
