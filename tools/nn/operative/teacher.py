"""The operative's formula (plan.md "Формула оперативника"), the plain operators the exam compares with, the teacher's
term.

The formula, on every decision, by code: the seen enemies cut into pieces by the squares' rule (in the field's axes);
my units into clumps (single link, join_m). Two pieces or more and two clumps or more: each clump its own group, its
target the nearest piece; else one group of all, the target the piece nearest my centre. The groups' numbers: a group
the same as one in force keeps its number; else the group in force it shares most units with; else the least free.
Form 'square', turn sector 0 (facing the target), verb 'hold', pace 'any'. The goal: the command 'hold' - the
group's centre (set once: then 'keep', the turn mended when the target's bearing moved more than turn_deg); 'destroy'
- a point approach_m short of the target's centre on the line from the group's centre ('move' when the zone the step
gives is move_m or more from the one in force). A cell the zone would leave the field from: the nearest open one.
A group of more than line_units (an army) is cut into a line of the tactician's drill groups (line_of): side by side
line_m apart across the way to the target, around the centre of the group it was cut from; such a group moves its zone
under 'hold' too, until it stands in its place.

The plain operators (mode): 'each' - every unit its own group where it stands; 'blob' - one group of all on my centre;
'stay' / 'go' - the formula under the command 'hold' / 'destroy' whatever the case says.

formula() -> (act, labels) in the walk's format (net.choose); the play runs the walk with that act and no network
(net.NullHeads); loss() - the term of the training (the cross-entropies of choose(labels=...)).
"""
import math

import numpy as np
import torch

from tools.nn.operative import cfg
from tools.nn.operative import groups as GR
from tools.nn.operative import net as NN
from tools.nn.operative import step as SP
from tools.nn.operative import target as TG
from tools.nn.tactic import decide as DC
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR

MODES = ("formula", "each", "blob", "stay", "go")


def pieces(raw, axes, enemy):
    """[B] lists of the seen enemies' pieces (lists of slots) by the squares' rule in the field's axes."""
    B, N = enemy.shape
    dev = raw.device
    h = axes.heading * TR.DEG
    z = torch.zeros(B, dtype=torch.bool, device=dev)
    p = PC.Parcel(torch.arange(B, device=dev), torch.zeros_like(enemy), enemy, axes.cx, axes.cz, torch.sin(h),
                  torch.cos(h), z, torch.zeros(B, dtype=torch.long, device=dev))
    s, f, hs, hf, on = PC.target_boxes(raw, p, sel=enemy)
    touch = ((s[:, :, None] - s[:, None, :]).abs() <= hs[:, :, None] + hs[:, None, :]) & \
            ((f[:, :, None] - f[:, None, :]).abs() <= hf[:, :, None] + hf[:, None, :])
    touch = (touch & on[:, :, None] & on[:, None, :]).cpu().numpy()
    on = on.cpu().numpy()
    out = []
    for b in range(B):
        left = set(np.nonzero(on[b])[0].tolist())
        comps = []
        while left:
            st = [min(left)]
            comp = set(st)
            left -= comp
            while st:
                i = st.pop()
                for j in list(left):
                    if touch[b, i, j]:
                        left.discard(j)
                        comp.add(j)
                        st.append(j)
            comps.append(sorted(comp))
        out.append(comps)
    return out


def clumps(xy, join_m):
    """Lists of indices of the points xy [n, 2] joined by single link within join_m."""
    n = len(xy)
    left = set(range(n))
    out = []
    while left:
        st = [min(left)]
        comp = set(st)
        left -= comp
        while st:
            i = st.pop()
            for j in list(left):
                if math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) <= join_m:
                    left.discard(j)
                    comp.add(j)
                    st.append(j)
        out.append(sorted(comp))
    return out


def formula(inp, ctx, destroy, mode="formula"):
    """(act, labels): destroy [B] bool the command (mode 'stay' / 'go' override it)."""
    c = cfg()
    tc = c["teacher"]
    G = int(c["groups"])
    raw, axes, old = ctx.raw, ctx.axes, ctx.groups
    B, K = ctx.slot.shape
    dev = raw.device
    I = TR.I
    if mode == "stay":
        destroy = torch.zeros_like(destroy)
    elif mode == "go":
        destroy = torch.ones_like(destroy)
    enemy_slots = NN.to_slots(ctx.enemy, ctx.slot, raw.shape[1], False)
    comps = pieces(raw, axes, enemy_slots)
    slot = ctx.slot.cpu().numpy()
    own = ctx.own.cpu().numpy()
    enemy = ctx.enemy.cpu().numpy()
    J = ctx.J[..., [I["x"], I["z"]]].cpu().numpy()
    rg = ctx.row_group.cpu().numpy()
    on = old.on.cpu().numpy()
    old_t = old.target.cpu().numpy()
    row_of = [{int(slot[b, j]): j for j in range(K) if slot[b, j] >= 0} for b in range(B)]
    # phase A: the groups wanted and their numbers
    keep = np.full((B, G), GR.REFORM, dtype=np.int64)
    comp_l = np.full((B, K), -1, dtype=np.int64)
    tgt_rows = np.zeros((B, G, K + 1), dtype=np.float32)
    tgt_act = np.full((B, G), K, dtype=np.int64)
    want = np.zeros((B, G), dtype=bool)
    ident = np.zeros((B, G), dtype=bool)
    changed_t = np.zeros((B, G), dtype=bool)
    gxy = np.zeros((B, G, 2), dtype=np.float32)
    txy = np.zeros((B, G, 2), dtype=np.float32)
    has_t = np.zeros((B, G), dtype=bool)
    axy = np.zeros((B, G, 2), dtype=np.float32)          # the line's anchor: the centre of the group it was cut from
    rxy = np.zeros((B, G, 2), dtype=np.float32)          # the line's right
    off = np.zeros((B, G), dtype=np.float32)             # the group's place in the line (m right of the anchor)
    rng = ctx.J[..., I["range"]].cpu().numpy()
    line_n, lat_m = int(tc["line_units"]), float(tc["line_m"])
    for b in range(B):
        mine = [j for j in range(K) if own[b, j]]
        if not mine:
            continue
        pcs = comps[b]
        pc_xy = [np.mean([J[b, row_of[b][s]] for s in pc], 0) for pc in pcs]

        def nearest(xy):
            if not pcs:
                return None
            return int(np.argmin([math.hypot(*(p - xy)) for p in pc_xy]))
        if mode == "each":
            W = [([j], nearest(J[b, j])) for j in mine[:G]]
            for j in mine[G:]:                           # (more units than groups: the nearest one's)
                k = min(range(len(W)), key=lambda i: math.hypot(*(J[b, W[i][0][0]] - J[b, j])))
                W[k][0].append(j)
        else:
            cl = [[mine[i] for i in k] for k in clumps([J[b, j] for j in mine], float(tc["join_m"]))]
            if mode != "blob" and len(pcs) >= 2 and len(cl) >= 2:
                W = [(k, nearest(np.mean([J[b, j] for j in k], 0))) for k in cl]
            else:
                W = [(mine, nearest(np.mean([J[b, j] for j in mine], 0)))]
        cut = line_n if mode in ("formula", "stay", "go") else 10 ** 6     # (the plain operators keep their groups)
        have = {g: set(j for j in mine if rg[b, j] == g) for g in range(G) if on[b, g]}
        W = [line_of(b, rows, pc, J, rng, pc_xy, axes, cut, lat_m, have) for rows, pc in W]
        W = [w for ws in W for w in ws]
        W.sort(key=lambda w: min(w[0]))
        taken = set()
        index = []
        for rows, *_ in W:
            rs = set(rows)
            g = next((g for g, m in have.items() if m == rs and g not in taken), None)
            if g is None:
                best = sorted(((len(m & rs), -g) for g, m in have.items() if g not in taken and m & rs), reverse=True)
                g = -best[0][1] if best else None
            if g is None:
                g = next((g for g in range(G) if g not in taken and not on[b, g]),
                         next(g for g in range(G) if g not in taken))
            taken.add(g)
            index.append(g)
        for (rows, pc, anchor, o, right), g in zip(W, index):
            axy[b, g], off[b, g], rxy[b, g] = anchor, o, right
            want[b, g] = True
            ident[b, g] = bool(on[b, g]) and have.get(g) == set(rows)
            gxy[b, g] = np.mean([J[b, j] for j in rows], 0)
            if pc is not None:
                has_t[b, g] = True
                txy[b, g] = pc_xy[pc]
                for s in pcs[pc]:
                    tgt_rows[b, g, row_of[b][s]] = 1.0 / len(pcs[pc])
                near_j = min((row_of[b][s] for s in pcs[pc]), key=lambda j: math.hypot(*(J[b, j] - gxy[b, g])))
                tgt_act[b, g] = near_j
                changed_t[b, g] = not any(old_t[b, g, s] for s in pcs[pc])
            else:
                tgt_rows[b, g, K] = 1.0
                changed_t[b, g] = bool(old_t[b, g].any())
            for j in rows:
                comp_l[b, j] = g
    T = lambda a, dt=None: torch.as_tensor(a, device=dev, dtype=dt)               # noqa: E731
    want, ident, changed_t, has_t = T(want), T(ident), T(changed_t), T(has_t)
    gxy, txy, axy, rxy, off = T(gxy), T(txy), T(axy), T(rxy), T(off)
    in_line = off.abs() > 1e-6
    # phase B: the goals, the cells, keep or move (tensors over B x G)
    gx, gz, tx, tz = gxy[..., 0], gxy[..., 1], txy[..., 0], txy[..., 1]
    up = axes.heading[:, None].expand(B, G)
    base = torch.where(has_t, TG.bearing(gx, gz, tx, tz), up)
    ax, az = axy[..., 0], axy[..., 1]
    dx, dz = tx - ax, tz - az
    d = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-6)
    ap = float(tc["approach_m"])
    go = destroy[:, None] & has_t
    qx = torch.where(go, tx - dx / d * torch.minimum(torch.full_like(d, ap), d), ax) + off * rxy[..., 0]
    qz = torch.where(go, tz - dz / d * torch.minimum(torch.full_like(d, ap), d), az) + off * rxy[..., 1]
    members_n = torch.zeros((B, G), dtype=torch.long, device=dev)
    cl_t = T(comp_l)
    for g in range(G):
        members_n[:, g] = ((cl_t == g) & ctx.own).sum(1)
    n = members_n.clamp(min=1).reshape(-1)
    line = torch.zeros_like(n, dtype=torch.bool)
    keep_tiles = (ident & ~old.line).reshape(-1)
    n0 = torch.where(keep_tiles, old.n0.reshape(-1), n)
    front, depth = SP.size(n0, line)
    cx_, cz_ = SP.cells(axes)
    cxr, czr = cx_.repeat_interleave(G, 0), cz_.repeat_interleave(G, 0)
    hd = base.reshape(-1)
    fok = SP.field_ok(cxr, czr, front, depth, hd, axes.take(torch.arange(B, device=dev).repeat_interleave(G)).field)
    fok = fok | ~fok.any(1, keepdim=True)
    d2 = (cxr - qx.reshape(-1, 1)) ** 2 + (czr - qz.reshape(-1, 1)) ** 2
    cell = d2.masked_fill(~fok, 1e18).argmin(1)
    cqx, cqz = cxr.gather(1, cell[:, None])[:, 0], czr.gather(1, cell[:, None])[:, 0]
    fld = axes.take(torch.arange(B, device=dev).repeat_interleave(G)).field
    zx, zz = SP.clamp(gx.reshape(-1), gz.reshape(-1), cqx, cqz, SP.near(n, line), front, depth, hd, fld)
    zx, zz = zx.view(B, G), zz.view(B, G)
    dturn = (torch.remainder(base - old.heading + 180.0, 360.0) - 180.0).abs()
    far = (zx - old.cx) ** 2 + (zz - old.cz) ** 2 >= float(tc["move_m"]) ** 2
    turned = dturn > float(tc["turn_deg"])
    # (a group of a line moves under 'hold' only while it takes its place - its first place_s: its anchor is its army's
    # centre now, which creeps forward as the infantry goes to the front rows)
    placing = in_line & (old.age < float(tc["place_s"]))
    need = changed_t | turned | ((destroy[:, None] | placing) & far)
    keepT = torch.where(ident & ~need, GR.KEEP, torch.where(ident, GR.MOVE, GR.REFORM))
    keepT = torch.where(old.on, keepT, torch.full_like(keepT, GR.REFORM))
    # the act and the labels
    hold_i, any_i = DC.VERBS.index("hold"), DC.PACES.index("any")
    act = {"keep": keepT, "comp": cl_t, "form": torch.zeros((B, G), dtype=torch.long, device=dev),
           "target": T(tgt_act), "turn": torch.zeros((B, G), dtype=torch.long, device=dev),
           "cell": cell.view(B, G), "verb": torch.full((B, G), hold_i, dtype=torch.long, device=dev),
           "pace": torch.full((B, G), any_i, dtype=torch.long, device=dev)}
    labels = dict(act)
    labels["target"] = T(tgt_rows)
    return act, labels


def line_of(b, rows, pc, J, rng, pc_xy, axes, line_n, lat_m, have=None):
    """[(rows, piece, anchor xy, offset m, right xy)]: a group of more than line_n units cut into a line of groups of
    the tactician's drill (up to 2 infantry + 1 shooters; a lord is infantry), side by side lat_m apart across the way to
    its target (plan.md; 07.11: the tactician's formula places 2 infantry and 1 shooters - an army of 8 in one group sent
    it to its last resort on 17 % of its decisions). Neighbours by place: the infantry and the shooters each sorted
    from left to right and cut into k runs."""
    c = np.mean([J[b, j] for j in rows], 0)
    if pc is not None:
        f = pc_xy[pc] - c
    else:
        h = math.radians(float(axes.heading[b]))
        f = np.array([math.sin(h), math.cos(h)])
    f = f / max(float(np.hypot(*f)), 1e-6)
    right = np.array([f[1], -f[0]], dtype=np.float32)
    if len(rows) <= line_n:
        return [(rows, pc, c, 0.0, right)]
    k = math.ceil(len(rows) / line_n)
    side = lambda j: float((J[b, j] - c) @ right)                                       # noqa: E731
    # the line in force kept: the groups now cover exactly these units, each of up to line_n (any number of them) - the
    # same groups in their order from left to right (a cut again by places would re-form them as units shift: the
    # formula re-formed 0.84 times a battle on the armies' exam, 07.11; and a valid line the network made is kept too:
    # with 'as many as the cut' 44 % of run8's re-forms were the teacher's own 'reform' - reform_diag_run8.json)
    rs = set(rows)
    now = [m for m in (have or {}).values() if m and m <= rs]
    if now and set().union(*now) == rs and all(len(m) <= line_n for m in now):
        k = len(now)
        now.sort(key=lambda m: float(np.mean([side(j) for j in m])))
        return [(sorted(m), pc, c, (i - (k - 1) / 2) * lat_m, right) for i, m in enumerate(now)]
    inf = sorted([j for j in rows if rng[b, j] <= 0], key=side)
    sh = sorted([j for j in rows if rng[b, j] > 0], key=side)
    runs_i = np.array_split(np.array(inf, dtype=np.int64), k)
    runs_s = np.array_split(np.array(sh, dtype=np.int64), k)
    out = []
    for i in range(k):
        g = [int(v) for v in runs_i[i]] + [int(v) for v in runs_s[i]]
        if g:
            out.append((g, pc, c, (i - (k - 1) / 2) * lat_m, right))
    return out


def play_formula(inp, ctx, destroy, mode="formula"):
    """The walk with the formula's choices and no network: net.choose's output."""
    act, _ = formula(inp, ctx, destroy, mode)
    B, K = ctx.slot.shape
    G = int(cfg()["groups"])
    enc = NN.null_enc(B, K, G, ctx.raw.device)
    return NN.choose(enc, inp, ctx, NN.NullHeads(B, K, G, ctx.raw.device), act=act)


def loss(net, inp, ctx, destroy):
    """(term, stats): the formula's choices in force, the network's cross-entropies against them."""
    act, labels = formula(inp, ctx, destroy)
    out, _ = net(inp, ctx, act=act, labels=labels)
    t = out["teach"]
    st = {h: (t[f"{h}_ok"] / t[f"{h}_n"]) if t[f"{h}_n"] > 0 else None for h in NN.HEADS}
    n = max(t["n"], 1.0)
    st["keep_part"] = t["keep_term"] / n                 # (the 'keep / move / re-form' head's share of the term)
    return t["term"] / n, st


def weight_at(update, decay=None):
    c = cfg()["teacher"]
    return float(c["weight"]) * max(0.0, 1.0 - update / max(float(c["decay_updates"] if decay is None else decay), 1.0))
