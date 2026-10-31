"""The tactician's analysis of its input (design/tactic "Анализ отряда", "Карточка на карте", "Анализ карты",
"Признаки карты"; the user's decisions 30-31.10). New modules, not trained yet; what comes after them (the rows'
council with the map, the outputs) is not designed yet.

    UnitAnalysis   one block for every row (the levels' common block): 185 -> 256 -> 128, ReLU after each layer;
                   row i -> card i (identity by slot: the block reads every row by itself, the rows never mix);
                   empty rows give zeros
    CardSqueeze    128 -> CARD (8; config/nn/tactic.json card) - one linear layer for every unit: the card on the map
    paint_cards    the squeezed cards into the cells under each formation: x the cell's share; two units in a cell -
                   their share-weighted mean x min(1, the shares' sum) (as zmap's direction layers)
    MapAnalysis    three 3x3 conv stacks, one per window: near 34 -> 48 -> 64 -> 64, far 28 -> 32 -> 32, target
                   17 -> 32 -> 32 (ReLU); each window's vector = mean and max over its cells -> a linear layer
                   (64, 32, 32: 128 'map features'), and the near window's whole last map 64 x 30 x 30 (for the output
                   'where to put the areas', as the unit's 'where' cell)
    TacticInput    all of it: inputs.build's dict -> {cards [R, K, 128], mask, slot, map_vec [R, 128],
                   map_near [R, 64, 30, 30], numbers [R, 4]}
weights() counts them (the convolutions ~111 thousand: near ~79, far ~17, target ~14).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.tactic import cfg
from tools.nn.tactic import rows as RW
from tools.nn.tactic import windows as W

ROW = len(RW.ROW)


class UnitAnalysis(nn.Module):
    def __init__(self, inp=ROW, hidden=None, out=None):
        super().__init__()
        a = cfg()["analysis"]
        self.l1 = nn.Linear(inp, int(hidden or a["hidden"]))
        self.l2 = nn.Linear(int(hidden or a["hidden"]), int(out or a["out"]))

    def forward(self, rows, mask=None):
        h = F.relu(self.l2(F.relu(self.l1(rows))))
        return h if mask is None else torch.where(mask[..., None], h, torch.zeros((), device=h.device))


class CardSqueeze(nn.Module):
    def __init__(self, inp=None, out=None):
        super().__init__()
        self.l = nn.Linear(int(inp or cfg()["analysis"]["out"]), int(out or cfg()["card"]))

    def forward(self, cards):
        return self.l(cards)


def paint_cards(card, cover, sel):
    """[R, k, C, C]: card [R, K, k] painted on cover [R, K, C, C] for the rows sel [R, K] (module doc)."""
    w = cover * sel[..., None, None].float()
    tot = w.sum(1)                                                     # [R, C, C]
    s = torch.einsum("rkhw,rkc->rchw", w, card)
    return s / tot.clamp(min=1e-6)[:, None] * tot.clamp(max=1.0)[:, None]


def fill_cards(inp, card8):
    """(near, far, target) with their card layers painted from the squeezed cards card8 [R, K, k]."""
    out = []
    for win in ("near", "far", "target"):
        m = inp[win]
        names = W.LAYERS[win](card8.shape[-1])
        parts, at = [], 0
        for first, key in W.CARDS[win]:
            i = names.index(first)
            parts += [m[:, at:i], paint_cards(card8, inp["cover"][win], inp["paint"][key])]
            at = i + card8.shape[-1]
        parts.append(m[:, at:])
        out.append(torch.cat(parts, 1))
    return tuple(out)


class Stack(nn.Module):
    """3x3 convs (padding 1, ReLU) over widths, then the vector of the last map: mean and max -> vec."""

    def __init__(self, widths, vec):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(widths, widths[1:]))
        self.vec = nn.Linear(2 * widths[-1], vec)

    def forward(self, m):
        for c in self.convs:
            m = F.relu(c(m))
        return F.relu(self.vec(torch.cat([m.mean((2, 3)), m.amax((2, 3))], 1))), m


class MapAnalysis(nn.Module):
    def __init__(self, card=None):
        super().__init__()
        k = int(card or cfg()["card"])
        self.near = Stack((len(W.near_layers(k)), 48, 64, 64), 64)
        self.far = Stack((len(W.far_layers(k)), 32, 32), 32)
        self.target = Stack((len(W.target_layers(k)), 32, 32), 32)

    def forward(self, near, far, target):
        vn, mn = self.near(near)
        vf, _ = self.far(far)
        vt, _ = self.target(target)
        return torch.cat([vn, vf, vt], 1), mn


class TacticInput(nn.Module):
    def __init__(self, card=None):
        super().__init__()
        self.analysis = UnitAnalysis()
        self.squeeze = CardSqueeze(out=card)
        self.map = MapAnalysis(card)

    def forward(self, inp):
        cards = self.analysis(inp["rows"], inp["mask"])
        card8 = torch.where(inp["mask"][..., None], self.squeeze(cards), torch.zeros((), device=cards.device))
        near, far, target = fill_cards(inp, card8)
        vec, near_map = self.map(near, far, target)
        return {"cards": cards, "card8": card8, "mask": inp["mask"], "slot": inp["slot"], "map_vec": vec,
                "map_near": near_map, "numbers": inp["numbers"], "near": near, "far": far, "target": target}


def count(m):
    return sum(p.numel() for p in m.parameters())


def weights(net=None):
    """{block: number of weights} of a TacticInput."""
    net = net or TacticInput()
    mp = net.map
    conv = lambda s: sum(count(c) for c in s.convs)              # noqa: E731
    return {"analysis": count(net.analysis), "squeeze": count(net.squeeze),
            "conv_near": conv(mp.near), "conv_far": conv(mp.far), "conv_target": conv(mp.target),
            "conv_all": conv(mp.near) + conv(mp.far) + conv(mp.target),
            "vectors": count(mp.near.vec) + count(mp.far.vec) + count(mp.target.vec), "total": count(net)}
