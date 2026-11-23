"""OpNet1 - the operative's network of drill O0-1 (tmp/engine/design_merged.md 5): one own group, KEEP or a line of
the code's short list, on the level's real input (not the engineered row lines.INPUT).

    observe(side, raw, card, b, f, order, sec, axes) -> obs   the network's input of battle b at a decision tick, one
                 row: inputs.build's rows [K, 185] (through the frozen card8 'Анализ отряда' in the network), mask,
                 place [K, 3], members / targets [K] (the own group; the target = E's rows), tokens [10] (the group's
                 numbers), gframe [K, 6]; order [8] (the order in force: pointer one-hot 4, the zone's distance to C and
                 to the stand-off line computed now, the goal's distance to C, age since birth / 60 s capped 5), mission
                 [2] (att, t / 600 capped 2); cand [C, 4] (type one-hot KEEP / LS, the candidate zone's distance to C, its
                 distance to the zone in force); legal [C] (masks: KEEP always; LS unless the order in force is that
                 line within same_line_m - lines' ls_same)
    stack(obs_list) -> batch     the rows stacked ([M, ...])
    OpNet1()     the logits [M, C] (illegal: NEG): records of the rows (card 128 + place 3 + gframe 6 -> 128), tokens of
                 the own group, E and the order, the mission as the council's summary seat (tactic/decide.Council
                 2 x 128); the candidate scorer (summary, the order's token after the council, the candidate's 4) ->
                 64 -> 1 with the last layer zero and a learned KEEP bias (operative.json council_extra.keep_bias): a
                 fresh network = DEFAULT (always KEEP)
    NetPlayer(net)   an advance player (exam_o1.O1Side) on the network, greedy; wants_view: the side passes obs
    (its checkpoint: net/operative/ckpt.py save_o1 / restore_o1, kind operative_o1_v1)
CANDIDATES = ("KEEP", "LS"); the pointer of candidate 1 is lines' "LS" (L0 closed - the teacher never emits it, LM out).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.operative import cfg
from net.operative import gframe as GF
from net.operative import groups as GR
from net.tactic import decide as DC
from net.tactic import net as NT
from net.unit.net import NEG

CANDIDATES = ("KEEP", "LS")
POINTERS = ("deploy", "L0", "LS", "LM")          # (= lines.POINTERS)
ORDER = 8
MISSION = 2
CAND = 4
PLACE_M = GF.PLACE_M


def _w():
    from net.tactic import cfg as tcfg
    return int(tcfg()["council"]["width"])


def order_feats(order, C, ls_zone, sec):
    """[8] (module doc) of the order in force (lines.Level.order[b]), the members' centroid C (x, z), the stand-off
    line's zone computed now (cx, cz, h)."""
    k = PLACE_M
    pt = [0.0] * 4
    pt[POINTERS.index(order["ptr"])] = 1.0
    zx, zz = order["zone"][0], order["zone"][1]
    return pt + [math.hypot(zx - C[0], zz - C[1]) / k, math.hypot(zx - ls_zone[0], zz - ls_zone[1]) / k,
                 math.hypot(C[0] - order["goal"][0], C[1] - order["goal"][1]) / k,
                 min((float(sec) - order["born"]) / 60.0, 5.0)]


def cand_feats(order, C, ls_zone):
    """[C, 4]: KEEP - the zone in force; LS - the stand-off line computed now."""
    k = PLACE_M
    zx, zz = order["zone"][0], order["zone"][1]
    return [[1.0, 0.0, math.hypot(zx - C[0], zz - C[1]) / k, 0.0],
            [0.0, 1.0, math.hypot(ls_zone[0] - C[0], ls_zone[1] - C[1]) / k, math.hypot(ls_zone[0] - zx, ls_zone[1] - zz) / k]]


def legal_of(f):
    """[C] bool (module doc): the operative's head mask on the level's facts."""
    return [True, not bool(f["ls_same"])]


def groups_of(N, mem, E, zone, sec, device):
    """The one own group of a battle as operative Groups (G = 1): members mem, target E (bool [N]), the zone in force."""
    g = GR.Groups.empty(1, N, 1, device)
    g.member[0] = torch.where(mem, torch.zeros_like(g.member[0]), g.member[0])
    g.on[0, 0] = True
    g.target[0, 0] = E
    g.cx[0, 0], g.cz[0, 0], g.heading[0, 0] = float(zone[0]), float(zone[1]), float(zone[2])
    g.n0[0, 0] = int(mem.sum())
    g.age[0, 0] = float(sec)
    return g


def axes_of(raw1, bounds):
    from net.operative import frame as FR
    return FR.Axes.of(raw1, torch.ones(1, dtype=torch.long, device=raw1.device), bounds)


def inputs_of(raw1, card1, mem, E, zone, sec, axes):
    """The raw-derived part of obs (rows .. gframe) from one battle's raw [1, N, RAW], card [1, N, 69]."""
    from net.operative import inputs as IN
    from net.tactic import raw as TR
    N = raw1.shape[1]
    E = E & (raw1[0, :, TR.I["vis"]] > 0)      # (only seen foes reach the input: the bridge hides the rest - an unseen E
    # unit would leak its place into gframe's frame; inputs.select's rows drop it anyway)
    g = groups_of(N, mem, E, zone, sec, raw1.device)
    side = torch.ones(1, dtype=torch.long, device=raw1.device)
    inp = IN.build(raw1, card1, side, axes, g, DC.order_vec(1, "hold", "any", device=raw1.device))
    gf = GF.gframe(raw1, inp["slot"], mem[None], E[None])
    return {"rows": inp["rows"][0], "mask": inp["mask"][0], "slot": inp["slot"][0], "place": inp["place"][0],
            "members": inp["members"][0, 0], "targets": inp["targets"][0, 0], "tokens": inp["tokens"][0, 0],
            "gframe": gf[0]}


def observe(raw1, card1, f, order, sec, axes):
    """(module doc) f: lines.Level.facts of the battle (its 'fc' the geometry's facts, 'ls' the stand-off zone)."""
    fc = f["fc"]
    C = (float(fc["C"][0]), float(fc["C"][1]))
    o = inputs_of(raw1, card1, fc["mem"], fc["E"], order["zone"], sec, axes)
    dev = raw1.device
    o["order"] = torch.tensor(order_feats(order, C, f["ls"], sec), device=dev)
    o["mission"] = torch.tensor([float(f["att"]), min(float(sec) / 600.0, 2.0)], device=dev)
    o["cand"] = torch.tensor(cand_feats(order, C, f["ls"]), device=dev)
    o["legal"] = torch.tensor(legal_of(f), device=dev)
    return o


NET_KEYS = ("rows", "mask", "place", "gframe", "members", "targets", "tokens", "order", "mission", "cand", "legal")


def stack(obs):
    return {k: torch.stack([o[k] for o in obs]) for k in NET_KEYS}


PICK_WHY = ("the candidates' scores share the state (summary, order seat) through the same s1 weights; the softmax "
            "over KEEP / LS is shift-invariant, so the shared state's gradient cancels between the two candidates except "
            "where an s1 ReLU gate differs between them (the 4 candidate numbers, s1 column norm 0.58 against 3.25 for "
            "the summary): from pass 2 the gradient reaching the council's output is ~1e-9 (round-off) against 0.04 at "
            "s1's output (tmp/engine/b/trace2.log) - the cause / come difference has to pass a second-order gate path, "
            "learnt or not by chance (probe_fit3 vs diag4, same seeds). pick: a zero-initialised linear readout of the "
            "state per candidate type (KEEP / LS: cand[..., :2]) added to the logit - a first-order path; a fresh "
            "network is still DEFAULT")


class OpNet1(nn.Module):
    def __init__(self, card=True, pick=None):
        super().__init__()
        from net.tactic import cfg as tcfg
        W = _w()
        c = cfg()["council_extra"]
        self.analysis = NT.UnitAnalysis()
        A = int(tcfg()["analysis"]["out"])
        self.record = nn.Linear(A + 3 + len(GF.COLS), W)
        self.group = nn.Linear(W + GR.NUMBERS, W)
        self.enemy = nn.Linear(W, W)
        self.order = nn.Linear(ORDER, W)
        self.mission = nn.Linear(MISSION, W)
        self.council = DC.Council()
        h = int(c["hidden"])
        self.s1 = nn.Linear(2 * W + CAND, h)
        self.s2 = nn.Linear(h, 1)
        with torch.no_grad():
            self.s2.weight.zero_()
            self.s2.bias.zero_()
        self.keep_bias = nn.Parameter(torch.tensor(float(c["keep_bias"])))
        self.use_pick = bool(c.get("pick", False) if pick is None else pick)
        if self.use_pick:
            self.pick = nn.Linear(2 * W, 2)              # (PICK_WHY) per candidate type: KEEP, LS
            with torch.no_grad():
                self.pick.weight.zero_()
                self.pick.bias.zero_()
        if card:
            ck = torch.load(DC.CARD_CKPT, map_location="cpu", weights_only=False)
            self.analysis.load_state_dict(ck["analysis"])
            for p in self.analysis.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        mask = x["mask"]
        zero = torch.zeros((), device=mask.device)
        cards = self.analysis(x["rows"], mask)
        rec = torch.where(mask[..., None], F.relu(self.record(torch.cat([cards, x["place"], x["gframe"]], -1))), zero)

        def mean(sel):
            w = (sel & mask).float()
            return (rec * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)

        g = F.relu(self.group(torch.cat([mean(x["members"]), x["tokens"]], -1)))
        e = F.relu(self.enemy(mean(x["targets"])))
        o = F.relu(self.order(x["order"]))
        s = self.mission(x["mission"])
        M = mask.shape[0]
        seq = torch.cat([rec, g[:, None], e[:, None], o[:, None]], 1)
        mm = torch.cat([mask, torch.ones((M, 3), dtype=torch.bool, device=mask.device)], 1)
        after, sm = self.council(seq, s, mm)
        oa = after[:, -1]
        Cn = x["cand"].shape[1]
        z = torch.cat([sm[:, None].expand(M, Cn, sm.shape[-1]), oa[:, None].expand(M, Cn, oa.shape[-1]), x["cand"]], -1)
        logit = self.s2(F.relu(self.s1(z)))[..., 0]
        if self.use_pick:
            logit = logit + (self.pick(torch.cat([sm, oa], -1))[:, None, :] * x["cand"][..., :2]).sum(-1)
        logit = logit + F.one_hot(torch.zeros((), dtype=torch.long, device=mask.device), Cn).float() * self.keep_bias
        return torch.where(x["legal"], logit, torch.full_like(logit, NEG))


class NetPlayer:
    """(module doc) greedy on OpNet1; None = KEEP."""
    name = "NET"
    wants_view = True

    def __init__(self, net):
        self.net = net.eval()

    def __call__(self, f, order, c=None, obs=None, b=None):
        with torch.no_grad():
            lg = self.net(stack([obs]))[0]
        k = int(lg.argmax())
        return None if k == 0 else CANDIDATES[k]

