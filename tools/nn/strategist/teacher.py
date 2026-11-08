"""The strategist's formula (plan.md "Формула стратега"), the plain strategies the exam compares with, the teacher's term.

The formula, by code, from the strategist's own input only (the composition, the task, the zones): the operative's line of
the tactician's drill groups (operative/teacher.line_of, its numbers config/nn/operative.json teacher line_units, line_m):
k groups = ceil(mine / line_units), 1 to 6; the melee units (the lord too) cut by slot into k runs as even as can be
(numpy array_split), the shooters the same, run i to group i (up to 2 infantry + 1 shooters: what the tactician's formula
places; the operative cuts by place, the strategist by slot - it has no places); the groups side by side, left to
right, their centres line_m apart around my zone's middle (one clump for the operative) - closer when my zone is
narrower, and in rows when one row does not hold them (a group's zone at the deployment: its units' slots,
strategist/deploy.footprint); the task 'hold' - the first row at my zone's back edge, the next in front of it,
'attack' - from its front edge back (edge_m inside); a square, sector 0 (facing the enemy's zone); the operative's verb = the task ('destroy' / 'hold'), the pace 'any'. The cell: the open one nearest the wanted
point (the walk with the formula's wishes as logits, greedy - so the bans are the network's own).

The plain strategies (mode): 'blob' - one group of all in the middle; 'front' - every zone at the front edge whatever the
task; 'back' - every zone at the back edge; 'hold' / 'attack' - the formula under the task 'hold' / 'attack' whatever it says.
"""
import torch

from tools.nn.operative import cfg as ocfg
from tools.nn.strategist import cfg
from tools.nn.strategist import deploy as DP
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
    # the wanted points, in my zone's axes: the groups' zones at the deployment (their units' slots, deploy.footprint)
    # side by side line_m apart - closer if my zone is narrower, and in rows from the task's edge inwards if one row
    # does not hold them (the game's zone 416 x 176 m: 6 groups of 3-4 in two rows of 3)
    mem = F_one(g, G).transpose(1, 2).bool() & own[:, None, :]          # [B, G, K]
    n = mem.sum(2)
    sw, sd = DP.slot(mem, inp["front"], inp["depth"])
    fw, fd = DP.footprint(n.clamp(min=1), torch.zeros((B, G), dtype=torch.bool, device=dev), sw, sd)
    on = n > 0
    fw = torch.where(on, fw, torch.zeros_like(fw)).amax(1)               # [B] the widest, the deepest
    fd = torch.where(on, fd, torch.zeros_like(fd)).amax(1)
    edge = float(tc["edge_m"])
    room = (ctx.mine["front"] - 2 * edge).clamp(min=0)
    per = torch.minimum(torch.floor(room / fw.clamp(min=1.0)).clamp(min=1), k.float())   # groups a row holds
    sp = torch.where(per > 1, torch.minimum(torch.full_like(fw, float(ocfg()["teacher"]["line_m"])),
                                            (room - fw).clamp(min=0) / (per - 1).clamp(min=1)), torch.zeros_like(fw))
    gi = torch.arange(G, device=dev)[None].float()
    row = torch.floor(gi / per[:, None])
    j = gi - row * per[:, None]
    kr = torch.minimum(per[:, None], k[:, None].float() - row * per[:, None])          # the groups in this row
    s = (j + 0.5 - kr / 2) * sp[:, None]
    Dz = ctx.mine["depth"][:, None]
    front = (destroy[:, None] | torch.full_like(destroy[:, None], mode == "front")) & (mode != "back")
    back = -Dz / 2 + fd[:, None] / 2 + edge + row * fd[:, None]
    f = torch.where(front, -back, back)
    if mode in ("formula", "hold", "attack"):
        # an army (the operative's army formula from army.min own units): its groups by role, laid out as the person
        # lays his out (strategist/army.py) - the operative's teacher keeps them from its first decision
        g, s, f = army_wishes(inp, ctx, destroy, g, s, f, G)
        comp = F_one(g, G) * BIG
    gx, gz = ZS.world(ctx.mine, s, f)
    verb = torch.zeros((B, len(DC.VERBS)), device=dev)
    verb[torch.arange(B, device=dev), torch.where(destroy, DC.VERBS.index("destroy"), DC.VERBS.index("hold"))] = BIG
    return {"comp": comp, "form": form, "turn": turn, "goal_x": gx, "goal_z": gz, "verb": verb}


def army_wishes(inp, ctx, destroy, g, s, f, G):
    """(g [B, K], s, f [B, G]) with the battles of an army (army.min own units or more) by strategist/army.py; the
    others as given."""
    import math
    from tools.nn.strategist import army as SA
    own = inp["own"]
    B, K = own.shape
    amin = int(ocfg()["teacher"]["army"]["min"])
    if not bool((own.sum(1) >= amin).any()):
        return g, s, f
    edge = float(cfg()["teacher"]["edge_m"])
    g, s, f = g.clone(), s.clone(), f.clone()
    cards = inp["rows"][..., :69].cpu().numpy()
    slot = inp["slot"].cpu().tolist()
    fr, dp = inp["front"].cpu().tolist(), inp["depth"].cpu().tolist()
    cs, cf = ZS.local(ctx.mine, ctx.cx, ctx.cz)                    # the cells' centres in my zone's axes
    for b in range(B):
        mine = sorted((j for j in range(K) if bool(own[b, j])), key=lambda j: slot[b][j])
        if len(mine) < amin:
            continue
        rl = SA.roles(cards[b][mine])
        gs = SA.groups(rl, mine)
        if len(gs) > G:
            continue
        fw, fd = [], []
        for _, rows in gs:
            n = len(rows)
            sw = max(fr[b][j] for j in rows) + DP.GAP_M
            sd = max(dp[b][j] for j in rows) + DP.ROW_GAP_M
            cols = math.ceil(math.sqrt(n) - 1e-6)
            fw.append(cols * sw)
            fd.append(math.ceil(n / cols) * sd)
        F_, D_ = float(ctx.mine["front"][b]), float(ctx.mine["depth"][b])
        inside = (cs[b].abs() <= F_ / 2) & (cf[b].abs() <= D_ / 2)
        s_grid = sorted({round(float(v), 1) for v in cs[b][inside]})
        f_grid = sorted({round(float(v), 1) for v in cf[b][inside]})
        pl = SA.layout(gs, fw, fd, s_grid, f_grid, F_, edge, bool(destroy[b]))
        num = {i: k for k, i in enumerate(SA.order(gs, pl))}
        s[b], f[b] = 0.0, 0.0
        g[b] = 0
        for i, (_, rows) in enumerate(gs):
            k = num[i]
            for j in rows:
                g[b, j] = k
            s[b, k], f[b, k] = pl[i]
    return g, s, f


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
