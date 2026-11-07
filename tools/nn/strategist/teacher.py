"""The strategist's formula (plan.md "Формула стратега"), the plain strategies the exam compares with, the teacher's term.

The formula, by code, from the strategist's own input only (the composition, the task, the zones): the operative's line of
the tactician's drill groups (operative/teacher.line_of, its numbers config/nn/operative.json teacher line_units, line_m):
k groups = ceil(mine / line_units), 1 to 6; the melee units (the lord too) cut by slot into k runs as even as can be
(numpy array_split), the shooters the same, run i to group i (up to 2 infantry + 1 shooters: what the tactician's formula
places; the operative cuts by place, the strategist by slot - it has no places); the groups side by side, left to
right, their centres line_m apart around my zone's middle (one clump for the operative); the task 'hold' - the
group's zone at my zone's back edge, 'attack' - at its front edge (edge_m inside); a square, sector 0 (facing the enemy's
zone); the operative's verb = the task ('destroy' / 'hold'), the pace 'any'. The cell: the open one nearest the wanted
point (the walk with the formula's wishes as logits, greedy - so the bans are the network's own).

The plain strategies (mode): 'blob' - one group of all in the middle; 'front' - every zone at the front edge whatever the
task; 'back' - every zone at the back edge; 'hold' / 'attack' - the formula under the task 'hold' / 'attack' whatever it says.
"""
import torch

from tools.nn.operative import cfg as ocfg
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
    per = int(ocfg()["teacher"]["line_units"])
    n_own = own.sum(1)
    k = torch.div(n_own + per - 1, per, rounding_mode="floor")
    if bool(tc.get("drill_cap", False)):
        # every group at most 2 melee + 1 shooters (the tactician's drill exactly), not only at most line_units units
        k = torch.maximum(k, torch.maximum(torch.div((own & ~shooter).sum(1) + 1, 2, rounding_mode="floor"),
                                           (own & shooter).sum(1)))
    k = k.clamp(1, G)
    if mode == "blob":
        k = torch.ones_like(k)
    # the cut: the melee units by slot into k runs as even as can be, the shooters the same; run i -> group i
    inf, sh = own & ~shooter, own & shooter
    g = torch.where(inf, runs(inf, k), runs(sh, k))
    g = torch.where(own, g, torch.zeros_like(g))
    comp = F_one(g, G) * BIG
    form = torch.zeros((B, G, 2), device=dev)
    form[..., NN.SQUARE] = BIG
    turn = torch.zeros((B, G, S), device=dev)
    turn[..., 0] = BIG
    # the wanted points, in my zone's axes
    n = (F_one(g, G) * own[..., None].float()).sum(1)                   # [B, G]
    _, depth = SP.size(n.long().clamp(min=1), torch.zeros((B, G), dtype=torch.bool, device=dev))
    Dz = ctx.mine["depth"][:, None]
    gi = torch.arange(G, device=dev)[None].float()
    kk = k[:, None].float()
    s = (gi + 0.5 - kk / 2) * float(ocfg()["teacher"]["line_m"])
    edge = float(tc["edge_m"])
    front = (destroy[:, None] | torch.full_like(destroy[:, None], mode == "front")) & (mode != "back")
    f = torch.where(front, Dz / 2 - depth / 2 - edge, -Dz / 2 + depth / 2 + edge)
    gx, gz = ZS.world(ctx.mine, s, f)
    verb = torch.zeros((B, len(DC.VERBS)), device=dev)
    verb[torch.arange(B, device=dev), torch.where(destroy, DC.VERBS.index("destroy"), DC.VERBS.index("hold"))] = BIG
    return {"comp": comp, "form": form, "turn": turn, "goal_x": gx, "goal_z": gz, "verb": verb}


def runs(sel, k):
    """[B, K] long: each selected row's run when the selected rows (by slot) are cut into k [B] runs as numpy's
    array_split cuts: the first m mod k runs one longer."""
    i = torch.cumsum(sel.long(), 1) - 1
    m = sel.sum(1, keepdim=True)
    kk = k[:, None]
    q = torch.div(m, kk, rounding_mode="floor")
    r = m - q * kk
    long_part = r * (q + 1)
    run = torch.where(i < long_part, torch.div(i, q + 1, rounding_mode="floor"),
                      r + torch.div(i - long_part, q.clamp(min=1), rounding_mode="floor"))
    return torch.where(sel, run, torch.zeros_like(run))


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
