"""The unit network (~0.31 M weights; build/diag_g49/plan_unit_net.md section 4): one set of weights for every unit,
each copy sees only its own view (tools/nn/unit/view.py). Modules (each its own class, so that the row encoder can be
frozen and handed to the levels above):

    MapConv     the zone map 11 x 30 x 30: 4 x 3x3 convs (32; the last two stride 2), the mean over the field (32) and
                the 30 x 30 features of the second conv (for the 'where' head)
    RowEncoder  a row: the card (passport, 69) -> 64, the state (ROW, 25) -> 64, merged -> 96
    MemoryGRU   the row's T past snapshots (newest first; run oldest to newest) -> 64, merged with the row -> 96
    Block       attention: rows among themselves (self), then me -> the rows (the threat's attention), each + a feed-forward
    Core        [me 96, attention 96, map 32, order ORDER, cell size 1] -> 128 -> 128
    heads       where: a cell of the 30 x 30 map (the core added to the high-resolution map features; cells outside the
                area banned), face: 64 sectors of 5.625 deg (relative to my front), target: a pointer over the enemy rows
                + 'nobody' (enemies outside the area banned), pace: walk / run (slow bans the run, fast the walk)

A ban is a logit of NEG (-1e9): its probability is exactly 0 in fp32 and it is never sampled (-inf would make
the entropy's gradient NaN). The critic (Critic) is the same trunk
with its own weights over the view of the whole skirmish (view.build everyone=True), no bans, a value head.

The order (ORDER): pace any / slow / fast (one-hot), stage going / in place, a target given, an area given, and the
area as raw numbers in the unit's frame (U2: it may lie far beyond the map): its centre right / forward of the unit
(/ 100 m), its front and depth (/ 100 m), its bearing against the unit's front (cos, sin); a shooter's fire sector
(U5): given, its middle against the unit's front (cos, sin), its width / 180 (the area is then its position).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.model.passport import SIZE as CARD
from tools.nn.unit import view as V
from tools.nn.unit import zmap as Z

D = 96          # a row's width
CORE = 128
FACES = 64
ORDER = ("pace_any", "pace_slow", "pace_fast", "going", "in_place", "has_target", "has_area",
         "area_right", "area_forward", "area_front", "area_depth", "area_cos", "area_sin",
         "has_sector", "sector_cos", "sector_sin", "sector_width")
ORDER_BEFORE_SECTOR = 13   # the fields before the fire sector's (checkpoints made before it: train.restore pads them)
PACE_ANY, PACE_SLOW, PACE_FAST = 0, 1, 2
WALK, RUN = 0, 1
NEG = -1e9      # a banned choice: exp(NEG - max) is exactly 0 in fp32 (never drawn); -inf would give NaN gradients


class MapConv(nn.Module):
    def __init__(self, layers=len(Z.LAYERS), width=32):
        super().__init__()
        self.c1 = nn.Conv2d(layers, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.c3 = nn.Conv2d(width, width, 3, padding=1, stride=2)
        self.c4 = nn.Conv2d(width, width, 3, padding=1, stride=2)

    def forward(self, m):
        h2 = F.relu(self.c2(F.relu(self.c1(m))))
        h4 = F.relu(self.c4(F.relu(self.c3(h2))))
        return h4.mean((2, 3)), h2


class RowEncoder(nn.Module):
    def __init__(self, card=CARD, state=len(V.ROW), width=64, out=D):
        super().__init__()
        self.card = nn.Linear(card, width)
        self.state = nn.Linear(state, width)
        self.merge = nn.Linear(2 * width, out)

    def forward(self, card, state):
        return F.relu(self.merge(torch.cat([F.relu(self.card(card)), F.relu(self.state(state))], -1)))


class MemoryGRU(nn.Module):
    def __init__(self, state=len(V.ROW), hidden=64, out=D):
        super().__init__()
        self.gru = nn.GRU(state + 1, hidden, batch_first=True)
        self.merge = nn.Linear(out + hidden, out)

    def forward(self, row, mem, mem_mask):
        R, K, T, Fs = mem.shape
        x = torch.cat([mem, mem_mask[..., None].float()], -1).flip(2).reshape(R * K, T, Fs + 1)  # oldest first
        _, h = self.gru(x)
        return F.relu(self.merge(torch.cat([row, h[0].view(R, K, -1)], -1)))


class Block(nn.Module):
    """Attention (queries q over keys / values kv, masked) + a feed-forward, each with a residual and a layer norm."""

    def __init__(self, width=D, heads=4, ff=2 * D):
        super().__init__()
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.ReLU(), nn.Linear(ff, width))
        self.n2 = nn.LayerNorm(width)

    def forward(self, q, kv, mask):
        a, _ = self.att(q, kv, kv, key_padding_mask=~mask, need_weights=False)
        h = self.n1(q + a)
        return self.n2(h + self.ff(h))


class Core(nn.Module):
    def __init__(self, inp=D + D + 32 + len(ORDER) + 1, width=CORE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    """The view -> (core [R, 128], rows after attention [R, K, 96], the map's 30 x 30 features [R, 32, 30, 30])."""

    def __init__(self):
        super().__init__()
        self.map = MapConv()
        self.rows = RowEncoder()
        self.memory = MemoryGRU()
        self.among = Block()
        self.me = Block()
        self.core = Core()

    def forward(self, v, order, cell_m):
        pooled, hi = self.map(v["map"])
        r = self.memory(self.rows(v["card"], v["rows"]), v["mem"], v["mem_mask"])
        mask = v["mask"]
        r = torch.where(mask[..., None], r, torch.zeros((), device=r.device))   # empty rows: nothing, whatever they hold
        r = self.among(r, r, mask)
        r = torch.where(mask[..., None], r, torch.zeros((), device=r.device))
        me = r[:, :1]                                                     # row 0 is the unit itself
        att = self.me(me, r, mask)[:, 0]
        cm = cell_m.view(-1, 1) / 16.0
        return self.core(torch.cat([me[:, 0], att, pooled, order, cm], -1)), r, hi


class UnitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.where_core = nn.Linear(CORE, 32)
        self.where_a = nn.Conv2d(32, 16, 3, padding=1)
        self.where_b = nn.Conv2d(16, 1, 1)
        self.face = nn.Linear(CORE, FACES)
        self.t_query = nn.Linear(CORE, D)
        self.t_key = nn.Linear(D, D)
        self.t_none = nn.Linear(CORE, 1)
        self.pace = nn.Linear(CORE, 2)

    def forward(self, v, order, cell_m):
        """Masked logits: where [R, 900], face [R, 64], target [R, K + 1] (the last: nobody), pace [R, 2]."""
        core, r, hi = self.trunk(v, order, cell_m)
        y = F.relu(hi + self.where_core(core)[:, :, None, None])
        where = self.where_b(F.relu(self.where_a(y)))[:, 0].flatten(1)
        where = where.masked_fill(~v["cells_ok"].flatten(1), NEG)
        q = self.t_query(core)
        tl = (self.t_key(r) * q[:, None]).sum(-1) / math.sqrt(D)
        tl = tl.masked_fill(~v["target_ok"], NEG)
        target = torch.cat([tl, self.t_none(core)], 1)
        pace = self.pace(core)
        slow = order[:, ORDER.index("pace_slow")] > 0
        fast = order[:, ORDER.index("pace_fast")] > 0
        ban = torch.stack([fast, slow], 1)                                # fast bans the walk, slow the run
        pace = pace.masked_fill(ban, NEG)
        return {"where": where, "face": self.face(core), "target": target, "pace": pace}


class Critic(nn.Module):
    """The value of the unit's situation: the same trunk (own weights) over the whole skirmish's view, no bans."""

    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.value = nn.Linear(CORE, 1)

    def forward(self, v, order, cell_m):
        core, _, _ = self.trunk(v, order, cell_m)
        return self.value(core)[:, 0]


HEADS = ("where", "face", "target", "pace")


def sample(logits, greedy=False):
    """{head: [R] long} drawn from the masked logits (a banned choice: probability 0)."""
    out = {}
    for k in HEADS:
        lg = logits[k]
        out[k] = lg.argmax(1) if greedy else torch.distributions.Categorical(logits=lg).sample()
    return out


def log_prob(logits, act):
    """[R] the sum over the heads of the chosen actions' log-probabilities, and [R] the summed entropy."""
    lp, ent = 0.0, 0.0
    for k in HEADS:
        d = torch.distributions.Categorical(logits=logits[k])
        lp = lp + d.log_prob(act[k])
        p = d.probs
        ent = ent - torch.where(p > 0, p * d.logits, torch.zeros_like(p)).sum(1)
    return lp, ent


def order_vec(R, pace=PACE_ANY, going=False, has_target=False, has_area=False, device="cpu"):
    """[R, len(ORDER)] one order for R units (the tactical level fills it per unit later)."""
    o = torch.zeros((R, len(ORDER)), device=device)
    o[:, pace] = 1.0
    o[:, ORDER.index("going" if going else "in_place")] = 1.0
    o[:, ORDER.index("has_target")] = float(has_target)
    o[:, ORDER.index("has_area")] = float(has_area)
    return o


def order_of(me, area, pace, arrived, device=None, has_target=None):
    """[R, len(ORDER)] the orders of R units from their raw state me [R, view.RAW], their areas {x, z, b, front, depth} [R],
    pace [R] long (PACE_*), arrived [R] bool (in place; else going) and has_target [R] bool (the order names a main
    target - drill U8; its enemy's row carries the view's 'my_target' flag; None: no target)."""
    from tools.nn.unit import view as V
    R = me.shape[0]
    o = torch.zeros((R, len(ORDER)), device=me.device)
    o[torch.arange(R, device=me.device), pace] = 1.0
    o[:, ORDER.index("going")] = (~arrived).float()
    o[:, ORDER.index("in_place")] = arrived.float()
    if has_target is not None:
        o[:, ORDER.index("has_target")] = has_target.float()
    o[:, ORDER.index("has_area")] = 1.0
    s, f = Z.local(area["x"], area["z"], me[:, V.I["x"]], me[:, V.I["z"]], me[:, V.I["b"]])
    rel = (area["b"] - me[:, V.I["b"]]) * math.pi / 180
    o[:, ORDER.index("area_right")] = s / 100
    o[:, ORDER.index("area_forward")] = f / 100
    o[:, ORDER.index("area_front")] = area["front"] / 100
    o[:, ORDER.index("area_depth")] = area["depth"] / 100
    o[:, ORDER.index("area_cos")] = torch.cos(rel)
    o[:, ORDER.index("area_sin")] = torch.sin(rel)
    if "sw" in area:
        # the fire sector (shooters, drill U5): its middle's bearing against the unit's front (cos, sin), its width / 180
        on = area["sw"] > 0
        rs = (area["sb"] - me[:, V.I["b"]]) * math.pi / 180
        o[:, ORDER.index("has_sector")] = on.float()
        o[:, ORDER.index("sector_cos")] = torch.where(on, torch.cos(rs), torch.zeros_like(rs))
        o[:, ORDER.index("sector_sin")] = torch.where(on, torch.sin(rs), torch.zeros_like(rs))
        o[:, ORDER.index("sector_width")] = area["sw"] / 180
    return o


def weights(module):
    return sum(p.numel() for p in module.parameters())


def report(net):
    """{module: weights} of a UnitNet."""
    t = net.trunk
    out = {"map": weights(t.map), "rows": weights(t.rows), "memory": weights(t.memory), "among": weights(t.among),
           "me": weights(t.me), "core": weights(t.core),
           "where": weights(net.where_core) + weights(net.where_a) + weights(net.where_b), "face": weights(net.face),
           "target": weights(net.t_query) + weights(net.t_key) + weights(net.t_none), "pace": weights(net.pace)}
    out["total"] = weights(net)
    return out
