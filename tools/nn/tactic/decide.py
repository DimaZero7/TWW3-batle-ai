"""The tactician's network after the input (design/tactic "Совещание", "Числа", "Приказы тактику", "Карточки после
совещания", "Выбор зоны отряда"; the user's decisions 01.11).

    Summary      the map's vector 128 + the numbers 4 + the order 7 -> 128 (one linear layer): one participant of the
                 council; the order: verb 4 + pace 3 (order_vec)
    Council      unit.net.Block (imported, not copied) x 2, width 128: each asks each, masked; up to 32 rows + the
                 summary speak; a row keeps its place, so a decision by row goes to that row's slot
    ZoneHeads    per unit: width and depth (a step of size_steps_m each, two small heads), then 'where': a cell of the
                 near window, as the unit's where head - the card (with the size just chosen, one-hot) is added to the
                 map 64 x 30 x 30, two coordinate layers (the cell's place in the zone's axes: a 3 x 3 stack sees only
                 7 cells, a zone is 15 - not an input of the game, it is the map's own grid), 3 x 3 conv, 1 x 1 conv.
                 Closed cells (uzone.Seq) get NEG, as at the unit
    TacticNet    TacticInput + the above. 'Анализ отряда' and the squeeze 128 -> 8 start from
                 build/diag_tactic/card_pretrain/card8.pt and stay frozen (the user's decision for the first drill)
    TacticCritic the same input and council with its own weights -> a value of the summary's seat; not used in the game

Choices run in slot order, each unit's frame closes cells for the next: choose() is the one walk for the sampling and
the update (the same choices again).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import config as project
from tools.nn.tactic import cfg
from tools.nn.tactic import net as NT
from tools.nn.tactic import raw as TR
from tools.nn.tactic import uzone as U
from tools.nn.unit.net import NEG, Block

VERBS = ("destroy", "hold", "tie", "avoid")
PACES = ("any", "slow", "fast")          # as unit.net PACE_ANY, PACE_SLOW, PACE_FAST
CARD_CKPT = project.ROOT / "build" / "diag_tactic" / "card_pretrain" / "card8.pt"


def order_vec(R, verb="hold", pace="any", device="cpu"):
    """[R, 7] the order: verb one-hot 4 + pace one-hot 3."""
    o = torch.zeros((R, len(VERBS) + len(PACES)), device=device)
    o[:, VERBS.index(verb)] = 1.0
    o[:, len(VERBS) + PACES.index(pace)] = 1.0
    return o


class Council(nn.Module):
    def __init__(self):
        super().__init__()
        c = cfg()["council"]
        self.blocks = nn.ModuleList(Block(int(c["width"]), int(c["heads"]), int(c["ff"])) for _ in range(int(c["layers"])))

    def forward(self, cards, summary, mask):
        """(cards after [R, K, W], the summary after [R, W]): cards [R, K, W], mask [R, K], summary [R, W]."""
        x = torch.cat([cards, summary[:, None]], 1)
        m = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        zero = torch.zeros((), device=x.device)
        for b in self.blocks:
            x = torch.where(m[..., None], b(x, x, m), zero)
        return x[:, :-1], x[:, -1]


class Encoder(nn.Module):
    """TacticInput -> the council's cards and summary, and the near map."""

    def __init__(self):
        super().__init__()
        c = cfg()["council"]
        self.inp = NT.TacticInput()
        self.summary = nn.Linear(int(cfg()["analysis"]["out"]) + 4 + len(VERBS) + len(PACES), int(c["width"]))
        self.council = Council()

    def forward(self, inp, order):
        o = self.inp(inp)
        s = self.summary(torch.cat([o["map_vec"], o["numbers"], order], 1))
        cards, sm = self.council(o["cards"], s, o["mask"])
        return {"cards": cards, "summary": sm, "mask": o["mask"], "slot": o["slot"], "map": o["map_near"]}

    def load_card(self, path=None):
        """The frozen part: 'Анализ отряда' + the squeeze from the card's pretraining."""
        ck = torch.load(path or CARD_CKPT, map_location="cpu", weights_only=False)
        self.inp.analysis.load_state_dict(ck["analysis"])
        self.inp.squeeze.load_state_dict(ck["squeeze"])
        for p in list(self.inp.analysis.parameters()) + list(self.inp.squeeze.parameters()):
            p.requires_grad_(False)
        return self


class ZoneHeads(nn.Module):
    def __init__(self):
        super().__init__()
        c = cfg()
        W = int(c["council"]["width"])
        S = len(c["size_steps_m"])
        ch = int(c["zone_head"]["map"])
        self.width = nn.Linear(W, S)
        self.depth = nn.Linear(W, S)
        self.where_core = nn.Linear(W + 2 * S, ch)
        self.where_a = nn.Conv2d(ch + 2, int(c["zone_head"]["conv"]), 3, padding=1)
        self.where_b = nn.Conv2d(int(c["zone_head"]["conv"]), 1, 1)
        C = int(c["cells"])
        k = (torch.arange(C, dtype=torch.float32) + 0.5) / C * 2 - 1
        self.register_buffer("coords", torch.stack([k[None, :].expand(C, C), -k[:, None].expand(C, C)])[None],
                             persistent=False)
        self.S = S

    def where(self, card, w, d, near):
        """[R, 900] raw logits; card [R, W], w, d [R] the size's steps, near [R, 64, 30, 30]."""
        size = torch.cat([F.one_hot(w, self.S), F.one_hot(d, self.S)], 1).float()
        y = F.relu(near + self.where_core(torch.cat([card, size], 1))[:, :, None, None])
        y = torch.cat([y, self.coords.expand(y.shape[0], -1, -1, -1)], 1)
        return self.where_b(F.relu(self.where_a(y)))[:, 0].flatten(1)


@dataclass
class Ctx:
    """What the choices need of the groups besides the network's output."""
    tiles: object
    my: torch.Tensor        # [R, G] long: my slots in order, -1 none
    able: torch.Tensor      # [R, G] bool: alive, not routing: the unit chooses
    rows: torch.Tensor      # [R, G] long: the row each unit's card is in (0 when it has none)
    mine_raw: torch.Tensor  # [R, G, RAW] my units' raw state (zeros when it has no row)

    def take(self, k, tiles):
        return Ctx(tiles, self.my[k], self.able[k], self.rows[k], self.mine_raw[k])


def my_slots(parcel):
    """[R, G] long: each group's slots in order, -1 past its last."""
    R = parcel.mine.shape[0]
    G = int(parcel.mine.sum(1).max())
    # (a slot's index sorted so that the mine come first, in slot order)
    N = parcel.mine.shape[1]
    key = torch.where(parcel.mine, torch.arange(N, device=parcel.mine.device)[None].expand(R, N), N)
    out = key.sort(1).values[:, :G]
    return torch.where(out < N, out, torch.full_like(out, -1))


def left_after(able):
    """[R, G] the units that still choose after each."""
    return able.flip(1).long().cumsum(1).flip(1) - able.long()


def ctx_of(inp, parcel, tiles):
    """Ctx of inputs.build's dict."""
    my = my_slots(parcel)
    pos = (inp["slot"][:, None] == my[:, :, None]) & (my[:, :, None] >= 0) & inp["mask"][:, None]      # [R, G, K]
    has = pos.any(2)
    rows = pos.float().argmax(2)
    J = inp["J"].gather(1, rows[..., None].expand(-1, -1, inp["J"].shape[-1]))
    able = has & (J[..., TR.I["alive"]] > 0) & ~(J[..., TR.I["r"]] > 0)
    return Ctx(tiles, my, able, rows, J)


def _pick(lg, given, greedy):
    if given is not None:
        return given
    return lg.argmax(1) if greedy else torch.distributions.Categorical(logits=lg).sample()


def choose(enc, ctx, heads, act=None, greedy=False):
    """The walk: {width, depth, cell} [R, G] long, frames (uzone.Seq.frames), lp [R], ent {width, depth, where} [R] (sums
    over the units), fell [R] (the last resort's uses). act: the same choices again (the update), else sampled."""
    tiles, able, rows = ctx.tiles, ctx.able, ctx.rows
    R, G = able.shape
    dev = able.device
    seq = U.Seq(tiles, G)
    st = U.steps(dev)
    lp = torch.zeros(R, device=dev)
    ent = {k: torch.zeros(R, device=dev) for k in ("width", "depth", "where")}
    out = {k: torch.zeros((R, G), dtype=torch.long, device=dev) for k in ("width", "depth", "cell")}
    fell = torch.zeros(R, device=dev)
    left = left_after(able)
    idx = torch.arange(R, device=dev)
    for j in range(G):
        card = enc["cards"][idx, rows[:, j]]
        a = able[:, j]
        lw, ld = heads.width(card), heads.depth(card)
        w = _pick(lw, None if act is None else act["width"][:, j], greedy)
        d = _pick(ld, None if act is None else act["depth"][:, j], greedy)
        ok, s, f, wm, dm, fb = seq.place_or_least(j, st[w], st[d], left[:, j], a)
        # the where head reads the size in force (the last resort's, if it fired)
        wi, di = (wm[:, None] - st).abs().argmin(1), (dm[:, None] - st).abs().argmin(1)
        lg = heads.where(card, wi, di, enc["map"]).masked_fill(~ok.flatten(1), NEG)
        c = _pick(lg, None if act is None else act["cell"][:, j], greedy)
        seq.commit(j, c, s, f, wm, dm, a)
        for k, l_, x in (("width", lw, w), ("depth", ld, d), ("where", lg, c)):
            dist = torch.distributions.Categorical(logits=l_)
            lp = lp + dist.log_prob(x) * a
            p = dist.probs
            ent[k] = ent[k] - torch.where(p > 0, p * dist.logits, torch.zeros_like(p)).sum(1) * a
        out["width"][:, j], out["depth"][:, j], out["cell"][:, j] = w, d, c
        fell = fell + fb.float()
    return {**out, "frames": seq.frames(), "lp": lp, "ent": ent, "fell": fell}


class TacticNet(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder()
        self.heads = ZoneHeads()
        if card:
            self.enc.load_card()

    def forward(self, inp, order, parcel, tiles, act=None, greedy=False):
        enc = self.enc(inp, order)
        ctx = ctx_of(inp, parcel, tiles)
        return choose(enc, ctx, self.heads, act, greedy), enc, ctx


class TacticCritic(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        self.enc = Encoder()
        self.value = nn.Linear(int(cfg()["council"]["width"]), 1)
        if card:
            self.enc.load_card()

    def forward(self, inp, order):
        return self.value(self.enc(inp, order)["summary"])[:, 0]


def report(net):
    """{block: weights, frozen and to train} of a TacticNet."""
    c = lambda m: sum(p.numel() for p in m.parameters())                    # noqa: E731
    i = net.enc.inp
    out = {"analysis_frozen": c(i.analysis), "squeeze_frozen": c(i.squeeze), "map": c(i.map),
           "summary": c(net.enc.summary), "council": c(net.enc.council),
           "size_heads": c(net.heads.width) + c(net.heads.depth),
           "where_head": c(net.heads.where_core) + c(net.heads.where_a) + c(net.heads.where_b)}
    out["total"] = c(net)
    out["to_train"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return out
