"""The strategist's formula (plan.md "Формула стратега"), the plain strategies the exam compares with, the teacher's term.

The formula, by code, from the strategist's own input only (the composition, the task, the zones): k groups = min(ceil
(enemies / per_group), floor(mine / per_group)), 1 to 6; the infantry and the shooters dealt to the groups in turn (by
slot); the groups left to right, their centres (i + 1/2 - k/2) x my zone's width x spread / k aside (two groups: +-1/3 of the width); the task 'hold' - the
group's zone at my zone's back edge, 'attack' - at its front edge (edge_m inside); a square, sector 0 (facing the enemy's
zone); the operative's verb = the task ('destroy' / 'hold'), the pace 'any'. The cell: the open one nearest the wanted
point (the walk with the formula's wishes as logits, greedy - so the bans are the network's own).

The plain strategies (mode): 'blob' - one group of all in the middle; 'front' - every zone at the front edge whatever the
task; 'back' - every zone at the back edge; 'hold' / 'attack' - the formula under the task 'hold' / 'attack' whatever it says.
"""
import torch

from tools.nn.operative import step as SP
from tools.nn.strategist import cfg
from tools.nn.strategist import net as NN
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import decide as DC

MODES = ("formula", "blob", "front", "back", "hold", "attack")
BIG = 20.0


def wishes(inp, ctx, mode="formula"):
    """The bias of net.choose (its doc) the formula's choices come from."""
    c = cfg()
    tc = c["teacher"]
    G, S = int(c["groups"]), int(c["sectors"])
    own, enemy, shooter = inp["own"], inp["enemy"], inp["shooter"]
    B, K = own.shape
    dev = own.device
    destroy = inp["command"][:, DC.VERBS.index("destroy")] > 0.5
    if mode == "hold":
        destroy = torch.zeros_like(destroy)
    elif mode == "attack":
        destroy = torch.ones_like(destroy)
    per = int(tc["per_group"])
    n_own, n_en = own.sum(1), enemy.sum(1)
    k = torch.minimum(torch.div(n_en + per - 1, per, rounding_mode="floor"), torch.div(n_own, per, rounding_mode="floor"))
    k = k.clamp(1, G)
    if mode == "blob":
        k = torch.ones_like(k)
    # the deal: the i-th infantry (the i-th shooter) of my rows, by slot, to the group i mod k
    inf, sh = own & ~shooter, own & shooter
    ri = torch.cumsum(inf.long(), 1) - 1
    rs = torch.cumsum(sh.long(), 1) - 1
    g = torch.where(inf, ri % k[:, None], rs % k[:, None])
    g = torch.where(own, g, torch.zeros_like(g))
    comp = F_one(g, G) * BIG
    form = torch.zeros((B, G, 2), device=dev)
    form[..., NN.SQUARE] = BIG
    turn = torch.zeros((B, G, S), device=dev)
    turn[..., 0] = BIG
    # the wanted points, in my zone's axes
    n = (F_one(g, G) * own[..., None].float()).sum(1)                   # [B, G]
    _, depth = SP.size(n.long().clamp(min=1), torch.zeros((B, G), dtype=torch.bool, device=dev))
    Wz, Dz = ctx.mine["front"][:, None], ctx.mine["depth"][:, None]
    gi = torch.arange(G, device=dev)[None].float()
    kk = k[:, None].float()
    s = (gi + 0.5 - kk / 2) * float(tc["spread"]) * Wz / kk
    edge = float(tc["edge_m"])
    front = (destroy[:, None] | torch.full_like(destroy[:, None], mode == "front")) & (mode != "back")
    f = torch.where(front, Dz / 2 - depth / 2 - edge, -Dz / 2 + depth / 2 + edge)
    gx, gz = ZS.world(ctx.mine, s, f)
    verb = torch.zeros((B, len(DC.VERBS)), device=dev)
    verb[torch.arange(B, device=dev), torch.where(destroy, DC.VERBS.index("destroy"), DC.VERBS.index("hold"))] = BIG
    return {"comp": comp, "form": form, "turn": turn, "goal_x": gx, "goal_z": gz, "verb": verb}


def F_one(g, G):
    return torch.nn.functional.one_hot(g.clamp(min=0), G).float()


def play_formula(inp, ctx, mode="formula"):
    """The walk with the formula's wishes and no network: net.choose's output."""
    B, K = inp["own"].shape
    G = int(cfg()["groups"])
    dev = inp["own"].device
    return NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, dev), greedy=True,
                     bias=wishes(inp, ctx, mode))


def labels_of(inp, ctx):
    return play_formula(inp, ctx)["act"]


def loss(net, inp, ctx, labels=None):
    """(term, agreement {head: share}): the network's cross-entropies against the formula's choices, those in force."""
    lab = labels_of(inp, ctx) if labels is None else labels
    out = net(inp, ctx, act=lab, labels=lab)
    t = out["teach"]
    st = {h: (t[f"{h}_ok"] / t[f"{h}_n"]) if t[f"{h}_n"] > 0 else None for h in NN.HEADS}
    return t["term"] / max(t["n"], 1.0), st


def weight_at(update):
    c = cfg()["train"]
    return float(c["teach_weight"]) * max(0.0, 1.0 - update / max(float(c["decay_updates"]), 1.0))
