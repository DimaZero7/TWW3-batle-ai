"""Drill T0-2 'strike the pinned one': the training buffer and the LABEL PROBE (tmp/tactic_slice/design_attack_v23.md
sections 6-7; review_attack_v23 C, G, K; METHOD step 4 - a small model on the level's input repeats the script's move
on held-out seeds; if not, the cause is not visible to the network).

    features(v, T, teacher, ana)  one tick's stored numbers of the groups with a head-free unit: the frozen card
                 ('Анализ отряда', 128) and the row columns COLS (right, forward, v_right, v_forward - the rows' own; mt
                 capped orders.net.mt_cap; the X control's pinned(e) and waited(e) at the labeller's touch, 'free and
                 not come_own' on my units' rows), own() [G, 16], the walk fed with the labeller's choices (open [G, K],
                 pair numbers [G, K, 7] at each g's turn: later slots see the earlier choices), the X control's pair
                 column 'g is the nearest eligible unit of its side to k' (the teacher's own walk), the label (row, -1
                 keep), dropped / don't-care label, don't-care keys, kinds (bits of tags.KINDS), trivial, near2 (the
                 label's half)
    Buffer       the hook of tags.Record (the one recorder): features of every tick, by battle; a tick whose head-free
                 units are all trivial keeps is kept one in loss.trivial_every with that weight
    weights(d)   section 7 (config drills.attack.loss): per situation type (a family's cause scenes, the no-cause
                 scenes, the general battles; a scene's class by play) attack labels = keeps in mass, taught kinds
                 (tags.TAUGHT) at least taught_min of a type's keeps (a tick's weight raised at most cap times), trivial
                 at most trivial_max; families families x their share of S, no-cause nocause, general general
    train_probe  the probe's model (heads.ProbeNet on input A / B / C / X) by Adam on the buffer
    Reader       the probe's read on the teacher's path (the walk fed with the teacher's earlier choices): per head-free
                 unit-tick the pick of every probe (null, the label's target, a don't-care key, a false attack) - over
                 the loss's own set ({null} + open keys less the don't-care keys), and over every open key ('~dcin')
                 beside; --read-only reads the probes saved in --out again
    tables       followed (the teacher's 'attack e' answered by the same unit with the same e at that tick) by recipe and
                 half, false attacks a head-free unit-tick by recipe and of each kind's keeps; don't-care out
    verdict      design section 6's rules fixed before the run (config drills.attack.probe.rules)

    DOCK_NAME=agent-s4 DOCK_CPUS=6 DOCK_MEM=6g bash ops/dock.sh net.tactic.learn probe --inputs A,B,C,X \
        --out tmp/tactic_slice/v23/stage4/probe --sim 1925110
`learn train` and `learn after` (the first training, the exam after by the move marks): tactic/imitate.py.

Drill T0-2 'side of the strike' (tmp/drill2/design_merged.md; `learn probe --side`): TEACH2 drives and labels; per
group-tick with a new attack label SideBuffer keeps the probe's input (the card, D2's and the S2 row columns, own() of 19,
the walk fed with the teacher's targets AND approaches: the pair numbers, heads.side_feats and the held flanks at the
label's target) and the label's approach; heads.SideProbe learns it (held-back battles, plateau) on the training draw
(T0-1's scenes and general battles of the T0-1 probe's seeds + S2 / S2N scenes of SIDE_PROBE's seeds) and is read on
held-out seeds: S2 / S2N scenes of new seeds, the held axes (scenes.held_specs_side of new seeds), S' and G' (T0-1's
probe sets). The lines (written before the run, the brief): followed >= 0.90 per class front / left / right on the
held-out reads pooled, false flank (left / right where the teacher says front) on the no-cause scenes S2N <= 3 %.
    bash ops/dock.sh net.tactic.learn probe --side --teacher TEACH2 --out tmp/t02b/probe --sim <commit>
weights(d, meta): S2 / S2N battles with an attack label are their own type 'side' (mass SIDE_LEARN.mass), without one
'nocause'.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP
from net.tactic import exam_pin as EP
from net.tactic import heads as HD
from net.tactic import orders as OR
from net.tactic import pin as PN
from net.tactic import tags as TG
from net.tactic.cols import COLS, D2, PAIR_EXTRA, _D2, ingredients, row_cols  # noqa: F401 (one formula: cols.py)

# input: (the row columns of COLS it reads, the pair columns of PAIR_EXTRA beside the 7) - review_attack_v23 M
INPUTS = {"A": ((0, 1), ()), "B": ((0, 1, 2, 3), ()), "C": ((0, 1, 2, 3, 4), ()),
          "D": (tuple(range(10)), ()), "E": (tuple(range(10)), (1, 2, 3, 4)),
          "D2": (_D2, ()), "E2": (_D2, (1, 2, 3, 4, 5, 6)),
          "X": ((0, 1, 2, 3, 4, 10, 11, 12), (0,))}
GMAX = 8
FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6")
NOCAUSE, GENERAL = len(FAMILIES), len(FAMILIES) + 1
SIDE = GENERAL + 1
SIDE_RECIPES = ("S2", "S2N")
# drill T0-2's training and label-probe settings: config tactic.json drills.side (learn, probe; their _why)
from net.tactic import cfg as _tcfg  # noqa: E402
SIDE_LEARN = _tcfg()["drills"]["side"]["learn"]
SIDE_PROBE = _tcfg()["drills"]["side"]["probe"]


def _c():
    return scfg()["drills"]["attack"]


def _pad_g(t, fill):
    G = t.shape[1]
    if G == GMAX:
        return t
    assert G < GMAX, f"{G} units of mine in a group: at most {GMAX}"
    pad = torch.full((t.shape[0], GMAX - G) + tuple(t.shape[2:]), fill, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], 1)


def nearest_eligible(fx, v, teacher, given):
    """([R, G, K] g is the nearest eligible unit of its side class to k at its turn of the walk fed with given - the
    teacher's 'i the nearest free unit of its side not yet given an order in this walk', pin.Teacher; [R, G] eligible
    at the walk's start: free or forced, not in melee, and (the teacher's co) no enemy coming at it - come_own)."""
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    elig = fx.able & ~v.melee & (fx.free | v.must)
    if getattr(teacher, "co", False):
        elig = elig & ~PN.come_own(fx, getattr(teacher, "come_free", False), getattr(teacher, "come_reach", None))
    e0 = elig.clone()
    d = fx.g_dist + torch.arange(G, device=dev)[None, :, None] * 1e-4
    sd = fx.g_side
    near = torch.zeros((R, G, K), dtype=torch.bool, device=dev)
    for g in range(G):
        same = sd == sd[:, g:g + 1, :]
        best = torch.where(same & elig[:, :, None], d, torch.full_like(d, OR.INF)).amin(1)
        near[:, g] = (d[:, g] <= best) & elig[:, g:g + 1]
        elig[:, g] = elig[:, g] & ~(given[:, g] >= 0)
    return near, e0




def features(v, T, teacher, ana):
    """(module doc) {name: tensor [R, ...]} on the play's device, every group row (the caller selects)."""
    fx = v.fx
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    given = T["k"]
    wi = HD.walk_inputs(fx, given)
    own = OR.own(v.order, v.ctx.prev, fx, v.now)
    rows, mask = v.rr["rows"], v.rr["mask"]
    with torch.no_grad():
        card = ana(rows, mask)
    pin = PN.pinned(fx, touch=getattr(teacher, "touch", None))
    near, e0 = nearest_eligible(fx, v, teacher, given)
    on = HD.g_on_of(fx)                       # (the network's rule, heads.net_inputs: the unit has its own row)
    fnc = torch.zeros((R, K), device=dev)
    fnc.scatter_add_(1, fx.g_row, (e0 & on).float())
    _, _, pe = ingredients(fx)
    cols = row_cols(fx, torch.stack([pin.float(), (fx.mt >= 2).float(), fnc.clamp(max=1.0)], -1))
    head = T["head_free"]
    kinds = torch.zeros((R, G), dtype=torch.int32, device=dev)
    for i, n in enumerate(TG.KINDS):
        kinds |= T["kinds"][n].int() << i
    return {"mask": mask, "card": card, "cols": cols, "g_row": fx.g_row, "g_on": on, "head": head,
            "label": torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1)),
            "drop": T["dropped"] | T["dc_label"], "kinds": kinds, "trivial": T["trivial"],
            "near2": T["tags"]["near2"] & T["attack"], "open": wi["open"], "dc": T["dc_keys"], "near": near,
            "pair": wi["pair"], "pe": pe, "own": own}


def batch_of(f, inp, device=None, w=None):
    """The probe's batch (heads.ProbeNet.forward) of features f (stored or live) for input inp."""
    ci, pi = INPUTS[inp]
    dev = device or f["mask"].device
    g = lambda k: f[k].to(dev)                                                      # noqa: E731
    pair = g("pair").float()
    if pi:
        ex = torch.cat([g("near").float()[..., None], g("pe").float()], -1)          # [.., PAIR_EXTRA]
        pair = torch.cat([pair, ex[..., list(pi)]], -1)
    b = {"card": g("card").float(), "cols": g("cols")[..., list(ci)].float(), "mask": g("mask"), "own": g("own").float(),
         "g_row": g("g_row").long(), "g_on": g("g_on"), "pair": pair, "open": g("open")}
    if w is not None:
        b.update(label=g("label").long(), w=w.to(dev), dc=g("dc"))
    return b


# ------------------------------------------------------------------ the buffer
class Buffer:
    """(module doc) ticks: a list of dicts (CPU) with b = the buffer's battle id."""

    def __init__(self, ana, every=None):
        self.ana = ana
        self.every = int(_c()["loss"]["trivial_every"] if every is None else every)
        self.pending, self.ticks, self.meta = [], [], []
        self.n_triv = 0

    def hook(self, teacher):
        return lambda v, lab, T: self(v, T, teacher)

    def __call__(self, v, T, teacher):
        head = T["head_free"]
        sel = head.any(1)
        if not bool(sel.any()):
            return
        f = features(v, T, teacher, self.ana)
        triv = sel & ~(head & ~T["trivial"]).any(1)                                  # all head-free: trivial keeps
        tw = torch.ones(sel.shape, device=sel.device)
        for r in triv.nonzero()[:, 0].tolist():
            self.n_triv += 1
            if self.n_triv % self.every:
                sel[r] = False
            else:
                tw[r] = float(self.every)
        if not bool(sel.any()):
            return
        x = {k: t[sel] for k, t in f.items()}
        out = {"b": v.idx[sel].cpu(), "sec": torch.full((int(sel.sum()),), int(round(float(v.now))), dtype=torch.int16),
               "tw": tw[sel].cpu()}
        for k, t in x.items():
            if t.dim() >= 2 and t.shape[1] == v.fx.G and k not in ("mask", "card", "cols"):
                t = _pad_g(t, -1 if k in ("label",) else 0)
            if t.is_floating_point():
                t = t.half()
            out[k] = t.cpu()
        out["g_row"] = out["g_row"].to(torch.int16)
        out["label"] = out["label"].to(torch.int8)
        self.pending.append(out)

    def flush(self, ii, specs, kind):
        """After a chunk (exam_pin.run_player's on_chunk): local rows -> battle ids; the padding dropped; meta."""
        base = len(self.meta)
        for i in ii:
            s = specs[i]
            self.meta.append({"kind": kind, "recipe": s.get("recipe", "G"), "twin_of": s.get("twin_of"),
                              "held": s.get("held"), "n": len(s["mine"])})
        for t in self.pending:
            keep = t["b"] < len(ii)
            if not bool(keep.any()):
                continue
            t = {k: x[keep] for k, x in t.items()}
            t["b"] = t["b"] + base
            m = t["mask"]
            t["card"] = t["card"][m]                                                   # (the rows with a unit only)
            self.ticks.append(t)
        self.pending = []

    def data(self):
        """One dict of tensors: card_flat [rows, 128] fp16 and pos [T, K] its index (-1: no row)."""
        keys = [k for k in self.ticks[0] if k != "card"]
        d = {k: torch.cat([t[k] for t in self.ticks], 0) for k in keys}
        d["card_flat"] = torch.cat([t["card"] for t in self.ticks], 0)
        m = d["mask"]
        pos = m.flatten().long().cumsum(0) - 1
        d["pos"] = torch.where(m.flatten(), pos, torch.full_like(pos, -1)).view(m.shape).int()
        d["b"] = d["b"].long()
        return d


def stored(d, idx):
    """The features of the ticks idx of the buffer's data (card restored from card_flat)."""
    f = {k: d[k][idx] for k in ("mask", "cols", "g_row", "g_on", "head", "label", "drop", "kinds", "trivial", "near2",
                                "open", "dc", "near", "pair", "pe", "own")}
    p = d["pos"][idx].long()
    f["card"] = d["card_flat"][p.clamp(min=0)] * f["mask"][..., None]
    return f


# ------------------------------------------------------------------ the plays
def _args(device, sim):
    return SimpleNamespace(device=device, sim=sim)


def play(name, specs, scene, seeds, chunk, a, P, hook_of, on_chunk):
    """exam_pin.run_player with TEACH (built under its own name, as exam_pin.record_run) driving through tags.Record,
    hook_of(labeller) -> the Record's hook; on_chunk(ii) after each chunk."""
    def make():
        lab = EP.build_under(name)
        return TG.Record(EP.build_under(name), lab, hook=hook_of(lab))
    kw = {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}
    return EP.run_player(name, specs, seeds, chunk, a, P, make=make, on_chunk=lambda ii, w, s: on_chunk(ii),
                         full=True, by_limit=True, **kw)


def round_specs(k, scale=1.0, gscale=None, seeds=None):
    """(scenes, general battles) of training round k (config drills.attack.probe: seeds apart from every set; the
    round's mix, E1-E3 on top; general battles on G's trained sizes); scale / gscale - a smaller draw (the smoke);
    seeds {scenes, round_step, general} - another draw's seeds (the training's: config tactic.json train.learn.seeds)."""
    gscale = scale if gscale is None else gscale
    from net.tactic import scenes as SN
    c = _c()
    mix = c["mix"]
    sd = c["probe"]["seeds"] if seeds is None else seeds
    rn = c["sets"]["round"]
    n = int(rn["scenes"])
    counts = {r: max(1, round(v * n * scale)) for r, v in mix.items() if r != "E_on_top"}
    e = max(1, round(float(mix.get("E_on_top", 0.0)) * n * scale / 3))
    counts.update({"E1": e, "E2": e, "E3": e})
    s0 = int(sd["scenes"]) + int(sd["round_step"]) * k
    scenes = []
    for j, (rec, m) in enumerate(counts.items()):
        scenes += [SN.make(rec, s0 + 10000 * j + i) for i in range(int(m))]
    G = c["sets"]["G"]
    gen = SP.drill_generals(max(1, round(int(rn["general"]) * gscale)), SP.seeded(int(sd["general"]) + k),
                            sizes=G["sizes"], limit=int(G["limit"]))
    return scenes, gen


def eval_specs(scale=None):
    """(S', H', G') of the probe's read: S's counts x eval.S on the probe's S seed; H's axes with eval.H x its cause /
    twin counts on H's own seeds (a subset of H); G' eval.G x G's two parts on the probe's G seeds."""
    from net.tactic import scenes as SN
    c = _c()
    ev = c["probe"]["eval"] if scale is None else {"S": scale, "H": scale, "G": scale}
    sd = c["probe"]["seeds"]
    S = SN.set_specs("S", seed0=int(sd["S"]),
                     counts={r: max(1, round(n * float(ev["S"]))) for r, n in c["sets"]["S"]["counts"].items()})
    st = c["sets"]["H"]
    H = []
    nc, nt = max(1, round(int(st["cause"]) * float(ev["H"]))), max(1, round(int(st["twin"]) * float(ev["H"])))
    for j, (axis, recs) in enumerate(st["axes"].items()):
        for i in range(nc):
            H.append(SN.make(recs[i % len(recs)], int(st["seed"]) + 10000 * j + i, held=axis))
        tw = [t for t in ("N1", "N2") if any(b in recs for b in SN.TWINS[t][1])] or ["N1"]
        for i in range(nt):
            H.append(SN.make(tw[i % len(tw)], int(st["seed"]) + 10000 * j + 5000 + i, held=axis, bases=recs))
    Gc = c["sets"]["G"]
    G = SP.drill_generals(max(1, round(int(Gc["n"]) * float(ev["G"]))), SP.seeded(int(sd["G"])), sizes=Gc["sizes"],
                          limit=int(Gc["limit"]))
    G += SP.drill_generals(max(1, round(int(Gc["n_held"]) * float(ev["G"]))), SP.seeded(int(sd["G_held"])),
                           sizes=Gc["sizes_held"], limit=int(Gc["limit"]))
    return S, H, G


def _seeds(base, n, chunk):
    return [base + k for k in range((n + chunk - 1) // chunk)]


def collect(rounds, name, a, P, ana, chunk, scale=1.0, log=print, gscale=None):
    """The buffer of `rounds` training rounds with the teacher driving (module doc)."""
    buf = Buffer(ana)
    tseed = int(_c()["probe"]["seeds"]["tseed"])
    for k in range(rounds):
        scenes, gen = round_specs(k, scale, gscale)
        for kind, sp, scene in (("scene", scenes, True), ("general", gen, False)):
            t0 = time.time()
            ch = min(chunk, len(sp))
            seeds = _seeds(tseed + 100 * k + (0 if scene else 50), len(sp), ch)
            play(name, sp, scene, seeds, ch, a, P, buf.hook, lambda ii, sp=sp, kind=kind: buf.flush(ii, sp, kind))
            n = sum(t["b"].shape[0] for t in buf.ticks)
            log(json.dumps({"collect": kind, "round": k, "battles": len(sp), "group_ticks_total": n,
                            "min": round((time.time() - t0) / 60, 1)}))
    return buf


# ------------------------------------------------------------------ the weights (section 7)
def family_of(m):
    """A battle's family for the weights: a cause recipe, A4L / A5L -> A4 / A5, a twin -> its base, else A1."""
    r = m["recipe"]
    if r in FAMILIES:
        return r
    if r in ("A4L", "A5L"):
        return r[:2]
    if m.get("twin_of") and m["twin_of"][0] in FAMILIES:
        return m["twin_of"][0]
    return "A1"


def weights(d, meta):
    """(w [T, G] float: section 7's weights normalised to mean 1 over the weighted unit-ticks, {type: masses})."""
    L = _c()["loss"]
    head, drop, label = d["head"], d["drop"], d["label"].long()
    tw = d["tw"].float()[:, None].expand_as(head)
    item = head & ~drop
    att = item & (label >= 0)
    keep = item & (label < 0)
    B = len(meta)
    bt = d["b"][:, None].expand_as(head)
    cause = torch.zeros(B, dtype=torch.bool)
    cause[bt[att]] = True
    ty = torch.empty(B, dtype=torch.long)
    for i, m in enumerate(meta):
        if m["kind"] != "general" and m["recipe"] in SIDE_RECIPES:
            ty[i] = SIDE if cause[i] else NOCAUSE                 # (drill T0-2's scenes: their own type)
            continue
        ty[i] = GENERAL if m["kind"] == "general" else (FAMILIES.index(family_of(m)) if cause[i] else NOCAUSE)
    t_ = ty[bt]
    cnt = _c()["sets"]["S"]["counts"]
    tot_f = sum(cnt[f] for f in FAMILIES)
    mass = {i: float(L["families"]) * cnt[f] / tot_f for i, f in enumerate(FAMILIES)}
    mass.update({NOCAUSE: float(L["nocause"]), GENERAL: float(L["general"]), SIDE: float(SIDE_LEARN["mass"])})
    present = {i for i in mass if bool((item & (t_ == i)).any())}
    z = sum(mass[i] for i in present) or 1.0
    tb = sum(1 << TG.KINDS.index(n) for n in TG.TAUGHT if n in TG.KINDS)
    taught = keep & ((d["kinds"] & tb) != 0)
    triv = keep & d["trivial"] & ~taught
    other = keep & ~taught & ~triv
    w = torch.zeros(head.shape)
    info = {}
    r = float(L["attack_keep"])
    tm, cap, tv = float(L["taught_min"]), float(L["cap"]), float(L["trivial_max"])
    for i in present:
        sel = t_ == i
        M = mass[i] / z
        na, nt, nv, no = (float((tw * (x & sel)).sum()) for x in (att, taught, triv, other))
        nk = nt + nv + no
        Ma = 0.0 if na == 0 else (M if nk == 0 else M * r / (1 + r))
        Mk = M - Ma
        a_, b_ = 1.0, 1.0
        for _ in range(30):
            if nt > 0:
                a_ = min(cap, max(1.0, tm / (1 - tm) * (b_ * nv + no) / nt))
            if nv > 0 and (a_ * nt + no) > 0:
                b_ = min(1.0, tv / (1 - tv) * (a_ * nt + no) / nv)
        den = a_ * nt + b_ * nv + no
        wo = Mk / den if den > 0 else 0.0
        if na:
            w = torch.where(att & sel, torch.full_like(w, Ma / na), w)
        w = torch.where(taught & sel, torch.full_like(w, a_ * wo), w)
        w = torch.where(triv & sel, torch.full_like(w, b_ * wo), w)
        w = torch.where(other & sel, torch.full_like(w, wo), w)
        name = FAMILIES[i] if i < NOCAUSE else {NOCAUSE: "nocause", GENERAL: "general", SIDE: "side"}[i]
        info[name] = {"mass": round(M, 4), "battles": int((ty == i).sum()), "attack": int(na), "taught": int(nt),
                      "trivial": int(nv), "other": int(no), "taught_x": round(a_, 3), "trivial_x": round(b_, 3),
                      "taught_share_of_keep_mass": round(a_ * nt / den, 4) if den else 0.0,
                      "trivial_share_of_keep_mass": round(b_ * nv / den, 4) if den else 0.0}
    w = w * tw
    n = float((w > 0).sum())
    w = w * (n / max(float(w.sum()), 1e-12))
    return w, info


# ------------------------------------------------------------------ the probe's training
def held_split(d, share=0.1, seed=4242):
    """[T] bool: the ticks of a fixed tenth of the buffer's battles held back (review M: by battle, fixed seed)."""
    bs = sorted(set(d["b"].tolist()))
    rnd = random.Random(seed)
    held = set(rnd.sample(bs, max(1, round(len(bs) * share))))
    return torch.tensor([b in held for b in d["b"].tolist()], dtype=torch.bool)


def _held_loss(net, d, w, idx, inp, device, mb):
    tot, wt = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, idx.numel(), mb):
            j = idx[i:i + mb]
            b = batch_of(stored(d, j), inp, device, w[j])
            loss, _ = HD.loss(net(b), b["label"], b["w"], b["dc"])
            sw = float(b["w"].sum())
            tot += float(loss) * sw
            wt += sw
    return tot / max(wt, 1e-9)


def train_probe(d, w, inp, width, device, log=print, seed=0, passes=None, held=None):
    """The probe by Adam; with held [T] bool (held_split): trained on the rest to a plateau - stop when the held-back
    loss has not improved for config probe.patience passes, at most probe.max_passes; the best held-back state kept.
    Returns (net, history)."""
    pr = _c()["probe"]
    if held is not None:
        return _train_plateau(d, w, inp, width, device, log, seed, held, passes)
    nc, npair = (len(x) for x in INPUTS[inp])
    torch.manual_seed(seed)
    net = HD.ProbeNet(nc, npair, width=width, heads=int(pr["heads"]), ff=int(pr["ff"])).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=float(pr["lr"]))
    use = (w.sum(1) > 0).nonzero()[:, 0]
    mb = int(pr["minibatch"])
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    hist = []
    for p in range(int(pr["passes"] if passes is None else passes)):
        perm = use[torch.randperm(use.numel(), generator=gen)]
        tot, nb = 0.0, 0
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            b = batch_of(stored(d, idx), inp, device, w[idx])
            lg = net(b)
            loss, _ = HD.loss(lg, b["label"], b["w"], b["dc"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append(round(tot / max(nb, 1), 5))
        log(json.dumps({"train": inp, "width": width, "pass": p, "loss": hist[-1], "min": round((time.time() - t0) / 60, 1)}))
    return net.eval(), hist


def _train_plateau(d, w, inp, width, device, log, seed, held, passes=None):
    import copy
    pr = _c()["probe"]
    nc, npair = (len(x) for x in INPUTS[inp])
    torch.manual_seed(seed)
    net = HD.ProbeNet(nc, npair, width=width, heads=int(pr["heads"]), ff=int(pr["ff"])).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=float(pr["lr"]))
    pos = w.sum(1) > 0
    use = (pos & ~held).nonzero()[:, 0]
    hidx = (pos & held).nonzero()[:, 0]
    mb = int(pr["minibatch"])
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    hist = {"train": [], "held": []}
    best, best_p, best_state, wait = float("inf"), -1, None, 0
    top = int(pr["max_passes"]) if passes is None else int(passes)
    for p in range(top):
        net.train()
        perm = use[torch.randperm(use.numel(), generator=gen)]
        tot, nb = 0.0, 0
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            b = batch_of(stored(d, idx), inp, device, w[idx])
            loss, _ = HD.loss(net(b), b["label"], b["w"], b["dc"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        net.eval()
        hl = _held_loss(net, d, w, hidx, inp, device, mb)
        hist["train"].append(round(tot / max(nb, 1), 5))
        hist["held"].append(round(hl, 5))
        if hl < best:
            best, best_p, wait = hl, p, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
        log(json.dumps({"train": inp, "width": width, "pass": p, "loss": hist["train"][-1], "held": hist["held"][-1],
                        "min": round((time.time() - t0) / 60, 1)}))
        if wait >= int(pr["patience"]):
            break
    net.load_state_dict(best_state)
    hist.update(passes=len(hist["held"]), best_pass=best_p, best_held=round(best, 5),
                train_at_best=hist["train"][best_p])
    log(json.dumps({"trained": inp, "width": width, "passes": hist["passes"], "best_pass": best_p,
                    "held_loss": hist["best_held"], "train_loss": hist["train_at_best"]}))
    return net.eval(), hist


def pick_codes(lg, f, dc_out=True):
    """[R, G] int8 per unit-tick: 0 null, 1 the label's target, 2 a don't-care key, 3 a false attack. dc_out: the pick
    over the loss's own set - {null} + the open keys less the don't-care keys (the loss never ranks a don't-care key
    against the label, so an argmax over them reads what was not taught); False: over every open key (printed beside,
    '~dcin')."""
    if dc_out:
        lg = lg.masked_fill(torch.cat([torch.zeros_like(f["dc"][..., :1]), f["dc"]], -1), HD.NEG)
    k = lg.argmax(-1) - 1
    kc = k.clamp(min=0)
    lab = f["label"].long()
    dc = f["dc"].gather(-1, kc[..., None])[..., 0]
    code = torch.where(k < 0, 0, torch.where(k == lab, 1, torch.where(dc, 2, 3)))
    return code.to(torch.int8)


class Reader:
    """(module doc) probes {name: ProbeNet}; rows: per head-free unit-tick (CPU) b, att, near2, drop, kinds, trivial,
    codes [n_probes]."""

    def __init__(self, probes, ana):
        self.probes, self.ana = probes, ana
        self.names = list(probes) + [n + "~dcin" for n in probes]
        self.pending, self.rows, self.meta = [], [], []
        self.pick, self.diag = {}, {}
        self.bad_open = 0

    def hook(self, teacher):
        return lambda v, lab, T: self(v, T, teacher)

    def __call__(self, v, T, teacher):
        head = T["head_free"]
        if not bool(head.any()):
            return
        f = features(v, T, teacher, self.ana)
        lab = f["label"]
        lc = lab.clamp(min=0)
        self.bad_open += int((head & (lab >= 0) & ~f["open"].gather(-1, lc[..., None])[..., 0]).sum())
        codes = []
        with torch.no_grad():
            raw = []
            for n in self.probes:
                lg = self.probes[n](batch_of(f, n.split("@")[0]))
                codes.append(pick_codes(lg, f))
                raw.append(pick_codes(lg, f, dc_out=False))
                if n.startswith(("E@", "E2@")):
                    m = lg.masked_fill(torch.cat([torch.zeros_like(f["dc"][..., :1]), f["dc"]], -1), HD.NEG)
                    self.pick[n] = m.argmax(-1) - 1
            codes += raw
        cd = torch.stack(codes, -1)                                                      # [R, G, P]
        r, g = head.nonzero(as_tuple=True)
        for n, k in self.pick.items():
            # the ingredients of (g, k) where E picked k or the label is k (k = E's pick, else the label)
            kk = torch.where(k >= 0, k, lab).clamp(min=0)[r, g]
            ck = f["cols"][r, kk]                                                     # [n, COLS]
            cg = f["cols"][r, f["g_row"][r, g]]
            pr = f["pair"][r, g, kk]
            ex = torch.cat([f["near"][r, g, kk].float()[:, None], f["pe"][r, g, kk]], -1)
            self.diag.setdefault(n, []).append(torch.cat([ck[:, 5:], cg[:, 5:], pr[:, :2], ex], -1).cpu())
        self.pending.append({**{"ing_" + n: self.diag[n].pop() for n in list(self.diag)}, "b": v.idx[r].cpu(), "att": (lab[r, g] >= 0).cpu(), "near2": f["near2"][r, g].cpu(),
                             "drop": f["drop"][r, g].cpu(), "kinds": f["kinds"][r, g].cpu(),
                             "trivial": f["trivial"][r, g].cpu(), "codes": cd[r, g].cpu()})

    def flush(self, ii, specs, kind):
        base = len(self.meta)
        for i in ii:
            s = specs[i]
            self.meta.append({"kind": kind, "recipe": s.get("recipe", "G"), "held": s.get("held"), "n": len(s["mine"])})
        for t in self.pending:
            keep = t["b"] < len(ii)
            t = {k: x[keep] for k, x in t.items()}
            t["b"] = t["b"].long() + base
            self.rows.append(t)
        self.pending = []

    def data(self):
        return {k: torch.cat([t[k] for t in self.rows], 0) for k in self.rows[0]}


DIAG_COLS = (tuple("k_" + c for c in COLS[5:]) + tuple("g_" + c for c in COLS[5:]) + ("dist", "gap") + PAIR_EXTRA)


def ingredient_diag(rd, names, kinds=None):
    """For each E probe and taught kind: the mean ingredients of (g, k) at E's false attacks on that kind's keeps
    against at the cause's followed attack labels (k_ the row k, g_ my unit's row; dist / gap / pos_m)."""
    kinds = kinds or TG.KINDS
    out = {}
    for n in [x for x in names if x.startswith(("E@", "E2@")) and "~" not in x]:
        j = names.index(n)
        code = rd["codes"][:, j].long()
        ing = rd["ing_" + n].float()
        ok = ~rd["drop"]
        fol = rd["att"] & ok & (code == 1)
        res = {"cols": DIAG_COLS, "followed_attacks": {"n": int(fol.sum()),
                                                      "mean": [round(float(x), 3) for x in ing[fol].mean(0)] if fol.any() else None}}
        for i, k in enumerate(kinds):
            if k not in TG.TAUGHT:
                continue
            sel = ((rd["kinds"] >> i) & 1).bool() & ~rd["att"] & ok & (code == 3)
            res[k] = {"n": int(sel.sum()), "mean": [round(float(x), 3) for x in ing[sel].mean(0)] if sel.any() else None}
        out[n] = res
    return out


# ------------------------------------------------------------------ the tables and the verdict
def share(x, n):
    """{share, two_errors (binomial), n}."""
    if n == 0:
        return {"share": None, "two_errors": None, "n": 0}
    p = x / n
    return {"share": round(p, 4), "two_errors": round(2 * math.sqrt(max(p * (1 - p), 1e-12) / n), 4), "n": int(n)}


def group_of(m, setname):
    if setname == "H":
        return "H:" + str(m.get("held"))
    if m["kind"] == "general":
        return "G57" if m["n"] in tuple(_c()["sets"]["G"]["sizes_held"]) else "G"
    return m["recipe"]


def tables(rd, meta, names, setname, kinds=None):
    """(module doc) {probe: {pooled, by_recipe {r: {followed {all, NO, YES}, false_tick}}, by_kind {k: false}}}."""
    kinds = kinds or TG.KINDS
    grp = [group_of(m, setname) for m in meta]
    gb = [grp[b] for b in rd["b"].tolist()]
    groups = sorted(set(gb))
    gi = torch.tensor([groups.index(x) for x in gb], dtype=torch.long)
    ok = ~rd["drop"]
    att = rd["att"] & ok
    keep = ~rd["att"] & ok
    near2 = rd["near2"]
    out = {}
    for j, n in enumerate(names):
        code = rd["codes"][:, j].long()
        fol = att & (code == 1)
        fal = ok & (code == 3)

        def fsum(sel):
            return {"all": share(int((fol & sel).sum()), int((att & sel).sum())),
                    "NO": share(int((fol & sel & ~near2).sum()), int((att & sel & ~near2).sum())),
                    "YES": share(int((fol & sel & near2).sum()), int((att & sel & near2).sum()))}
        everyone = torch.ones_like(ok)
        res = {"pooled": {"followed": fsum(everyone), "false_tick": share(int(fal.sum()), int(ok.sum())),
                          "dc_attack_tick": share(int((ok & (code == 2)).sum()), int(ok.sum()))}, "by_recipe": {},
               "by_kind": {}}
        for q, g in enumerate(groups):
            sel = gi == q
            res["by_recipe"][g] = {"followed": fsum(sel), "false_tick": share(int((fal & sel).sum()), int((ok & sel).sum()))}
        for i, k in enumerate(kinds):
            kb = ((rd["kinds"] >> i) & 1).bool() & keep
            res["by_kind"][k] = {"false": share(int((fal & kb).sum()), int(kb.sum())), "taught": k in TG.TAUGHT}
        out[n] = res
    return out


def verdict_of(tab, judged=("S", "G")):
    """Design section 6's rules (config drills.attack.probe.rules) for one probe's tables {set: table}: (passed,
    fails [text])."""
    R = _c()["probe"]["rules"]
    nmin = int(R["cell_min_n"])
    fails = []
    for s in judged:
        t = tab[s]
        p = t["pooled"]
        f = p["followed"]["all"]
        if f["share"] is None or f["share"] < float(R["followed"]):
            fails.append(f"{s} followed pooled {f['share']} (n {f['n']})")
        if p["false_tick"]["share"] is not None and p["false_tick"]["share"] > float(R["false_tick"]):
            fails.append(f"{s} false a tick pooled {p['false_tick']['share']} (n {p['false_tick']['n']})")
        for r, x in t["by_recipe"].items():
            for h in ("NO", "YES"):
                c = x["followed"][h]
                if c["n"] >= nmin and c["share"] < float(R["followed"]):
                    fails.append(f"{s} {r} {h} followed {c['share']} (n {c['n']})")
            c = x["false_tick"]
            if c["n"] >= nmin and c["share"] > float(R["false_tick"]):
                fails.append(f"{s} {r} false a tick {c['share']} (n {c['n']})")
        for k, x in t["by_kind"].items():
            c = x["false"]
            if x["taught"] and c["n"] >= nmin and c["share"] > float(R["false_kind"]):
                fails.append(f"{s} kind {k} false {c['share']} (n {c['n']})")
    return not fails, fails


def x_ok(tab, judged=("S", "G")):
    R = _c()["probe"]["rules"]
    return all((tab[s]["pooled"]["followed"]["all"]["share"] or 0.0) >= float(R["x_followed"]) for s in judged)


def verdict(tabs, inputs, widths):
    """{width used, X reached, the input (the smallest of A, B, C that passes), fails, invisible: cells where X passes
    and C fails}. tabs {probe name 'A@64': {set: table}}."""
    R = _c()["probe"]["rules"]
    out = {"rules": R, "by_width": {}}
    used = None
    for wd in widths:
        xs = f"X@{wd}"
        xr = x_ok(tabs[xs]) if xs in tabs else None
        res = {"X_reached": xr, "X_followed": {s: tabs[xs][s]["pooled"]["followed"]["all"] for s in ("S", "G", "H")}
               if xs in tabs else None, "inputs": {}}
        for i in inputs:
            n = f"{i}@{wd}"
            if n in tabs:
                ok, fails = verdict_of(tabs[n])
                res["inputs"][i] = {"passed": ok, "fails": fails}
        sm = [i for i in ("A", "B", "C") if res["inputs"].get(i, {}).get("passed")]
        res["smallest_passing"] = sm[0] if sm else None
        sm2 = [i for i in ("D", "E") if res["inputs"].get(i, {}).get("passed")]
        res["smallest_passing_D_E"] = sm2[0] if sm2 else None          # (review M: the smallest of D, E is the input)
        sm3 = [i for i in ("D2", "E2") if res["inputs"].get(i, {}).get("passed")]
        res["smallest_passing_D2_E2"] = sm3[0] if sm3 else None        # (review M, the first mend read: the candidates)
        for i in ("C", "E", "E2"):
            if "X" in res["inputs"] and i in res["inputs"]:
                fx_, fc = set(res["inputs"]["X"]["fails"]), res["inputs"][i]["fails"]
                res[f"invisible_with_{i}"] = [x for x in fc if x not in fx_]
        out["by_width"][str(wd)] = res
        if used is None and (xr or wd == widths[-1]):
            used = wd
    out["width_used"] = used
    u = out["by_width"][str(used)]
    out["input"] = u.get("smallest_passing_D2_E2")
    out["X_reached"] = u["X_reached"]
    out["stop"] = (not u["X_reached"]) and "X below the line at every width: the probe's model is too small - stop and report"
    return out


def brief(tab):
    """One line a probe and a set: followed all / NO / YES, false a tick."""
    p = tab["pooled"]
    f = p["followed"]
    return (f"followed {f['all']['share']} (n {f['all']['n']}), NO {f['NO']['share']} ({f['NO']['n']}), "
            f"YES {f['YES']['share']} ({f['YES']['n']}); false/tick {p['false_tick']['share']} (n {p['false_tick']['n']})")


# ------------------------------------------------------------------ drill T0-2: the side's label probe (module doc)
def side_row_cols():
    """The side probe's row columns: D2's (cols.D2) and cols.SIDE_ROW - indices into cols.COLS."""
    from net.tactic.cols import SIDE_ROW
    return tuple(HD.d2_cols()[1]) + tuple(COLS.index(n) for n in SIDE_ROW)


def at_label(t, k):
    """t [R, G, K, c] at the rows k [R, G] (clamped) -> [R, G, c]."""
    return t.gather(2, k.clamp(min=0)[..., None, None].expand(-1, -1, 1, t.shape[-1]))[:, :, 0]


def side_label_inputs(v, lab, T):
    """The side's label and the pair's side inputs of a tick: {k [R, G] the label's row (-1 none), ap [R, G] its approach
    (-1 none), pair_k [R, G, 7], x_k [R, G, SIDE_X] (side_feats + held at k)} - the walk fed with the labeller's targets
    and approaches (TEACH2: lab['approach'])."""
    fx = v.fx
    assert "approach" in lab, "the labeller gives no approach: drill T0-2 needs TEACH2"
    att = T["attack"]
    k = torch.where(att, T["k"], torch.full_like(T["k"], -1))
    ap = torch.where(att, lab["approach"], torch.full_like(lab["approach"], -1))
    wi = HD.walk_inputs(fx, T["k"], approach=lab["approach"])
    x = torch.cat([at_label(HD.side_feats(fx), k), at_label(wi["held"], k)], -1)
    return {"k": k, "ap": ap, "pair_k": at_label(wi["pair"], k), "x_k": x, "wi": wi}


class SideBuffer:
    """(module doc) ticks: dicts (CPU) of the group-ticks with an attack label; meta by battle."""

    def __init__(self, ana):
        self.ana = ana
        self.pending, self.ticks, self.meta = [], [], []
        self.frontal = {"rows": 0, "differ": 0, "rows_labelled": 0, "differ_labelled": 0}

    def hook(self, teacher):
        return lambda v, lab, T: self(v, lab, T, teacher)

    def check_frontal(self, fx, T, teacher):
        """cols.anvil_off(touch) <= frontal_deg against the teacher's cols.frontal(touch) on every row of the tick (and
        on the labels' targets): the column must carry the teacher's fact exactly."""
        from net.tactic import cols as CL
        from net.unit import scripts as SC
        tch = getattr(teacher, "touch", None)
        tch = SC.TOUCH_M if tch is None else tch
        a = CL.anvil_off(fx, tch) <= float(CL.c_side()["frontal_deg"])
        b = CL.frontal(fx, tch)
        m = fx.mask
        self.frontal["rows"] += int(m.sum())
        self.frontal["differ"] += int(((a != b) & m).sum())
        k = T["k"].clamp(min=0)
        att = T["attack"]
        self.frontal["rows_labelled"] += int(att.sum())
        self.frontal["differ_labelled"] += int(((a.gather(1, k) != b.gather(1, k)) & att).sum())

    def __call__(self, v, lab, T, teacher=None):
        fx = v.fx
        self.check_frontal(fx, T, teacher)
        sel = T["attack"].any(1)
        if not bool(sel.any()):
            return
        s = side_label_inputs(v, lab, T)
        with torch.no_grad():
            card = self.ana(v.rr["rows"], v.rr["mask"])
        f = {"card": card, "mask": v.rr["mask"], "cols": row_cols(fx)[..., list(side_row_cols())],
             "own": OR.own(v.order, v.ctx.prev, fx, v.now, approach=True), "g_row": fx.g_row, "g_on": HD.g_on_of(fx),
             "k": s["k"], "ap": s["ap"], "pair_k": s["pair_k"], "x_k": s["x_k"],
             "lab_open": s["wi"]["open"].gather(-1, s["k"].clamp(min=0)[..., None])[..., 0] & (s["k"] >= 0)}
        out = {"b": v.idx[sel].cpu()}
        for k, t in f.items():
            t = t[sel]
            if k not in ("mask", "card", "cols"):
                t = _pad_g(t, -1 if k in ("k", "ap") else (True if k == "lab_open" else 0))
            out[k] = (t.half() if t.is_floating_point() else t).cpu()
        self.pending.append(out)

    def flush(self, ii, specs, kind):
        base = len(self.meta)
        for i in ii:
            s = specs[i]
            self.meta.append({"kind": kind, "recipe": s.get("recipe", "G"), "held": s.get("held"),
                              "cause": s.get("s2_cause"), "n": len(s["mine"])})
        for t in self.pending:
            keep = t["b"] < len(ii)
            if bool(keep.any()):
                t = {k: x[keep] for k, x in t.items()}
                t["b"] = t["b"].long() + base
                self.ticks.append(t)
        self.pending = []

    def data(self):
        return {k: torch.cat([t[k] for t in self.ticks], 0) for k in self.ticks[0]}


def side_band(x, ap):
    """[..] bool: a side label at the rule's threshold (config drills.side.bands, its _why) - its DECIDING column, for the
    side the teacher chose or refused, inside a band: the way's clearance within way_m of its line (clear_enemy_m,
    clear_own_m), the extra path within extra_m of extra_cap_m, the anvil's off-facing angle within anvil_deg of
    frontal_deg. x [.., SIDE_X] the side columns at the label's pair (heads.side_feats + held), ap [..] the teacher's
    approach (-1 none: False). A flank s is 'at the threshold' when it is open and not held and either feasible with
    one of its threshold columns in a band, or not feasible with every failing threshold column in a band. A flank
    label: its own flank at the threshold, or the anvil's angle in its band. A front label: the anvil's angle in its
    band while a flank is feasible or at the threshold, or the target frontal and a flank at the threshold."""
    from net.tactic import cols as CL
    c = CL.c_side()
    b = _tcfg()["drills"]["side"]["bands"]
    pos = float(_tcfg()["pos_m"])
    x = x.float()
    wm, am, em = float(b["way_m"]), float(b["anvil_deg"]), float(b["extra_m"])
    ce, co, cap, fd = (float(c[k]) for k in ("clear_enemy_m", "clear_own_m", "extra_cap_m", "frontal_deg"))
    se, so, ex = x[..., [0, 4]] * ce, x[..., [1, 5]] * co, x[..., [2, 6]] * pos
    opn = (x[..., [8, 9]] >= 1 - 1e-3) & (x[..., [11, 12]] < 0.5)
    pe, po, px = se >= ce, so >= co, ex <= cap
    ie, io, ix = (se - ce).abs() <= wm, (so - co).abs() <= wm, (ex - cap).abs() <= em
    feas = opn & pe & po & px
    amb = opn & ((feas & (ie | io | ix)) | (~feas & (pe | ie) & (po | io) & (px | ix)))
    anv = x[..., 10] * 180.0
    ia = (anv - fd).abs() <= am
    fl = (ap == 1) | (ap == 2)
    own = torch.where(ap == 2, amb[..., 1], amb[..., 0])
    flank_b = fl & (own | ia)
    front_b = (ap == 0) & ((ia & (feas | amb).any(-1)) | ((anv <= fd) & amb.any(-1)))
    return flank_b | front_b


def side_batch(d, idx, dev):
    g = lambda k: d[k][idx].to(dev)                                                  # noqa: E731
    return {"card": g("card").float(), "cols": g("cols").float(), "mask": g("mask"), "own": g("own").float(),
            "g_row": g("g_row").long(), "g_on": g("g_on"), "k": g("k").long(), "pair_k": g("pair_k").float(),
            "x_k": g("x_k").float(), "ap": g("ap").long(),
            # (buffers collected before REVIEW2 finding 2 have no lab_open: taken as open)
            "lab_open": g("lab_open").bool() if "lab_open" in d else torch.ones_like(g("ap"), dtype=torch.bool)}


def side_probe_train(d, width, dev, log, held, seed=17, passes=160):
    """heads.SideProbe (its SideHead reads the threshold-centred columns) by recipe R3 (tmp/t02/fix/b/probe_recipes.md):
    Adam lr 1e-3 falling by cosine to 1e-5 over passes (160), minibatch 128, no early stop, the LAST state; the labels
    at the rule's threshold (side_band) out of the loss. The held-back tenth only logged."""
    pr = _c()["probe"]
    torch.manual_seed(seed)
    net = HD.SideProbe(len(side_row_cols()), width=width, heads=int(pr["heads"]), ff=int(pr["ff"])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=passes, eta_min=1e-5)
    use, hidx = (~held).nonzero()[:, 0], held.nonzero()[:, 0]
    mb = 128
    gen = torch.Generator().manual_seed(seed)

    def lossof(b):
        lg = net(b)
        m = (b["ap"] >= 0) & ~side_band(b["x_k"], b["ap"]) & b["lab_open"]
        return HD.side_loss(lg[m], b["ap"][m], torch.ones(int(m.sum()), device=dev))[0]
    hist = {"train": [], "held": []}
    for p in range(passes):
        net.train()
        perm = use[torch.randperm(use.numel(), generator=gen)]
        tot, nb = 0.0, 0
        for i in range(0, perm.numel(), mb):
            loss = lossof(side_batch(d, perm[i:i + mb], dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        sched.step()
        net.eval()
        with torch.no_grad():
            hl = [float(lossof(side_batch(d, hidx[i:i + mb], dev))) for i in range(0, hidx.numel(), mb)]
        hl = sum(hl) / max(len(hl), 1)
        hist["train"].append(round(tot / max(nb, 1), 5))
        hist["held"].append(round(hl, 5))
        log(json.dumps({"side_probe": width, "pass": p, "loss": hist["train"][-1], "held": hist["held"][-1]}))
    return net.eval(), hist


def side_picks(net, d, dev, mb=1024):
    """[n, 4] (b, label ap, pick, at the rule's threshold - side_band) of every labelled unit-tick of d."""
    out = []
    with torch.no_grad():
        for i in range(0, d["b"].shape[0], mb):
            idx = torch.arange(i, min(i + mb, d["b"].shape[0]))
            b = side_batch(d, idx, dev)
            pk = net(b).argmax(-1)
            m = b["ap"] >= 0
            r, g = m.nonzero(as_tuple=True)
            bd = (side_band(b["x_k"], b["ap"]) | ~b["lab_open"])[m].long()
            out.append(torch.stack([d["b"][idx][r.cpu()], b["ap"][m].cpu(), pk[m].cpu(), bd.cpu()], 1))
    return torch.cat(out, 0) if out else torch.zeros((0, 4), dtype=torch.long)


def side_group(m):
    if m["kind"] == "general":
        return "G"
    if m["recipe"] in SIDE_RECIPES:
        return m["recipe"] + ("" if m.get("held") is None else f" held {m['held']}")
    return "T0-1 scenes"


def side_tables(rows, meta, bands=True):
    """{group: {per class c: followed {share, two_errors, n}, false_flank (front labels picked left / right), banded
    (the share of the group's labels at the rule's threshold)}}; groups by side_group, + 'all held-out' and 'S2N (no
    cause)' pooled; bands - the labels at the threshold (rows[:, 3]) left out of the shares."""
    gr = [side_group(m) for m in meta]
    gb = [gr[int(b)] for b in rows[:, 0].tolist()]
    out = {}
    bd = rows[:, 3].bool() if rows.shape[1] > 3 else torch.zeros(rows.shape[0], dtype=torch.bool)

    def one(sel0):
        sel = sel0 & ~bd if bands else sel0
        lab, pk = rows[sel, 1], rows[sel, 2]
        res = {}
        for c, name in enumerate(OR.APPROACHES):
            n = int((lab == c).sum())
            res[name] = share(int(((lab == c) & (pk == c)).sum()), n)
        res["false_flank"] = share(int(((lab == 0) & (pk > 0)).sum()), int((lab == 0).sum()))
        res["banded"] = share(int((sel0 & bd).sum()), int(sel0.sum()))
        return res
    for g in sorted(set(gb)):
        out[g] = one(torch.tensor([x == g for x in gb], dtype=torch.bool))
    out["all"] = one(torch.ones(len(gb), dtype=torch.bool))
    out["S2N (no cause)"] = one(torch.tensor([x.startswith("S2N") for x in gb], dtype=torch.bool))
    return out


def side_verdict(tab):
    """The brief's lines on the held-out read: (passed, fails)."""
    L_ = SIDE_PROBE["lines"]
    fails = []
    for c in OR.APPROACHES:
        x = tab["all"][c]
        if x["share"] is None or x["share"] < float(L_["followed"]):
            fails.append(f"followed {c} {x['share']} (n {x['n']})")
    x = tab["S2N (no cause)"]["false_flank"]
    if x["share"] is None or x["share"] > float(L_["false_flank"]):
        fails.append(f"false flank on S2N {x['share']} (n {x['n']})")
    x = tab["all"]["banded"]
    if x["share"] is None or x["share"] > float(_tcfg()["drills"]["side"]["bands"]["band_share_max"]):
        fails.append(f"labels at the rule's threshold {x['share']} (n {x['n']})")
    return not fails, fails


def side_buffer_key(a):
    """sha256 (16) of what the side probe's buffers depend on: the column code (cols.py), the side head's inputs
    (heads.side_feats, learn.side_label_inputs), the teacher (pin.py, orders.py, scenes.py), the configs, the simulator."""
    import hashlib
    from net.tactic import cfg as tc
    src = "".join((ROOT_T / f).read_text(encoding="utf-8") for f in ("cols.py", "heads.py", "pin.py", "orders.py",
                                                                      "scenes.py", "learn.py"))
    x = {"src": hashlib.sha256(src.encode()).hexdigest(), "side": tc()["drills"].get("side"), "orders": tc()["orders"],
         "stage0": scfg()["drills"], "sim_key": EP.sim_key(), "teacher": a.teacher}
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


ROOT_T = Path(__file__).resolve().parent


def side_probe(a, P, log, D):
    from net.tactic import scenes as SN
    t0 = time.time()
    sp = SIDE_PROBE
    dev = a.device
    ar = _args(dev, a.sim)
    ana = HD.frozen_analysis(dev)
    sm = a.smoke
    pr = _c()["probe"]
    tseed = int(pr["seeds"]["tseed"]) + 7000

    def s2(seed0, n):
        return [SN.side_scene(seed0 + i) for i in range(n)] + [SN.side_scene(seed0 + i, twin=True) for i in range(n)]

    def plays(buf, parts, base):
        for j, (kind, specs, scene) in enumerate(parts):
            if not specs:
                continue
            if sm:
                for x in specs:
                    x["limit"] = min(int(x["limit"]), 150)
            t1 = time.time()
            ch = min(a.chunk, len(specs))
            play(a.teacher, specs, scene, _seeds(tseed + base + 10 * j, len(specs), ch), ch, ar, P, buf.hook,
                 lambda ii, specs=specs, kind=kind: buf.flush(ii, specs, kind))
            log(json.dumps({"side_collect": kind, "battles": len(specs), "group_ticks": sum(t["b"].shape[0] for t in buf.ticks),
                            "frontal_check": dict(buf.frontal), "min": round((time.time() - t1) / 60, 1)}))
    n_tr, n_ev, n_h = (sm, max(2, sm // 4), 2) if sm else (sp["n_train"], sp["n_eval"], sp["n_held"])
    sc, gs = (sm / 1536.0, sm / 512.0) if sm else (sp["scale"], sp["gscale"])
    bpath = D / "side_train.pt"
    epath = D / "side_eval.pt"
    fc = {}
    key = side_buffer_key(a)
    if not sm and (bpath.exists() or epath.exists()):
        # (cached buffers are never reused silently - REVIEW.md finding 2: only with --reuse, their key printed and
        # compared, and the frontal check run anew on a small fresh draw of this code)
        if not a.reuse:
            raise SystemExit(f"{D}: side_train.pt / side_eval.pt exist - give a clean --out, or --reuse to read them")
        bufc = SideBuffer(ana)
        plays(bufc, [("scene", s2(sp["seeds"]["eval"], 16), True)], 900)
        fc["reuse_check"] = dict(bufc.frontal)
        log(json.dumps({"reuse_frontal_check": fc["reuse_check"], "battles": 32}))
        assert bufc.frontal["differ"] == 0 and bufc.frontal["differ_labelled"] == 0, bufc.frontal
    if bpath.exists() and not sm:
        x = torch.load(bpath, weights_only=False)
        d, meta = x["data"], x["meta"]
        log(json.dumps({"loaded": str(bpath), "key": x.get("key"), "key_now": key}))
        assert x.get("key") in (None, key), f"{bpath}: collected by other code / config ({x.get('key')} != {key})"
    else:
        scenes, gen = round_specs(0, sc, gs)
        if sm:
            scenes, gen = scenes[:sm], gen[:sm]
        buf = SideBuffer(ana)
        plays(buf, [("scene", scenes + s2(sp["seeds"]["train"], n_tr), True), ("general", gen, False)], 0)
        d, meta = buf.data(), buf.meta
        fc["train"] = dict(buf.frontal)
        if not sm:
            torch.save({"data": d, "meta": meta, "key": key}, bpath)
    held = held_split(d)
    lab = d["ap"][d["ap"] >= 0]
    log(json.dumps({"train_labels": {c: int((lab == i).sum()) for i, c in enumerate(OR.APPROACHES)},
                    "group_ticks": int(d["b"].shape[0]), "battles": len(meta), "min": round((time.time() - t0) / 60, 1)}))
    # the held-out reads
    if epath.exists() and not sm:
        x = torch.load(epath, weights_only=False)
        de, me = x["data"], x["meta"]
        log(json.dumps({"loaded": str(epath), "key": x.get("key"), "key_now": key}))
        assert x.get("key") in (None, key), f"{epath}: collected by other code / config ({x.get('key')} != {key})"
    else:
        S, _, G = eval_specs(sp["eval_scale"] if not sm else 0.02)
        if sm:
            S, G = S[:sm], G[:sm]
        Hs = SN.held_specs_side(seed0=sp["seeds"]["held"], n=n_h)
        bufe = SideBuffer(ana)
        plays(bufe, [("scene", s2(sp["seeds"]["eval"], n_ev) + Hs + S, True), ("general", G, False)], 500)
        de, me = bufe.data(), bufe.meta
        fc["eval"] = dict(bufe.frontal)
        if not sm:
            torch.save({"data": de, "meta": me, "key": key}, epath)
    res = {"teacher": a.teacher, "sim": a.sim, "sim_key": EP.sim_key(), "lines": sp["lines"], "by_width": {},
           "frontal_check": fc}
    log(json.dumps({"frontal_check": fc}))
    widths = [int(x) for x in a.widths.split(",") if x] if a.widths else [int(pr["width"]), int(pr["width_retry"])]
    for wd in widths:
        net, hist = side_probe_train(d, wd, dev, log, held)
        torch.save(net.state_dict(), D / f"side_probe_{wd}.pt")
        rows_e = side_picks(net, de, dev)
        tab = side_tables(rows_e, me)
        tab_un = side_tables(rows_e, me, bands=False)
        tab_in = side_tables(side_picks(net, d, dev), meta)
        ok, fails = side_verdict(tab)
        res["by_width"][str(wd)] = {"history": hist, "held_out": tab, "held_out_unbanded": {
            k: tab_un[k] for k in ("all", "S2N (no cause)")}, "training_draw": tab_in["all"], "passed": ok,
            "fails": fails}
        log(json.dumps({"side_probe_width": wd, "passed": ok, "fails": fails, "all": tab["all"],
                        "S2N": tab["S2N (no cause)"], "unbanded_all": tab_un["all"],
                        "unbanded_S2N": tab_un["S2N (no cause)"]}))
        if ok:
            break
    res["passed"] = any(x["passed"] for x in res["by_width"].values())
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (D / "side_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(json.dumps({"SIDE_PROBE": {"passed": res["passed"], "minutes": res["minutes"]}}))
    return res


# ------------------------------------------------------------------ the command
def probe(a):
    from sim.params import load
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    logf = open(D / "probe.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    if a.side:
        a.teacher = "TEACH2" if a.teacher == "TEACH" else a.teacher
        res = side_probe(a, P, log, D)
        logf.close()
        return res
    t0 = time.time()
    pr = _c()["probe"]
    dev = a.device
    ar = _args(dev, a.sim)
    ana = HD.frozen_analysis(dev)
    inputs = [x for x in a.inputs.split(",") if x]
    widths = [int(x) for x in a.widths.split(",") if x] if a.widths else [int(pr["width"]), int(pr["width_retry"])]
    scale = 1.0 if not a.smoke else a.smoke / float(_c()["sets"]["round"]["scenes"])
    rounds = int(pr["rounds"]) if not a.smoke else 1
    log(json.dumps({"start": time.strftime("%H:%M"), "teacher": a.teacher, "reach_m": OR.c_orders()["reach_m"],
                    "come_reach_m": OR.c_orders()["teach"].get("come_reach_m"), "inputs": inputs, "widths": widths,
                    "rounds": rounds, "scale": scale, "device": dev, "sim": a.sim, "sim_key": EP.sim_key()}))
    bpath = Path(a.buffer) if a.buffer else None
    if a.read_only:
        pass
    elif bpath is not None and bpath.exists():
        saved = torch.load(bpath, weights_only=False)
        d, meta = saved["data"], saved["meta"]
        log(json.dumps({"buffer": str(bpath), "loaded": True}))
    else:
        gscale = None if not a.smoke else a.smoke / float(_c()["sets"]["round"]["general"])
        buf = collect(rounds, a.teacher, ar, P, ana, a.chunk, scale, log, gscale)
        d, meta = buf.data(), buf.meta
        if bpath is not None:
            torch.save({"data": d, "meta": meta, "kinds": TG.KINDS, "cols": COLS, "pair_extra": PAIR_EXTRA}, bpath)
        if a.collect_only:
            log(json.dumps({"collected": str(bpath), "group_ticks": int(d["head"].shape[0]),
                            "min": round((time.time() - t0) / 60, 1)}))
            return None
    if not a.read_only:
        nbytes = sum(t.numel() * t.element_size() for t in d.values())
        w, winfo = weights(d, meta)
        item = d["head"] & ~d["drop"]
        log(json.dumps({"buffer_group_ticks": int(d["head"].shape[0]), "unit_ticks_head_free": int(d["head"].sum()),
                        "weighted_unit_ticks": int((w > 0).sum()), "attack_labels": int((item & (d["label"] >= 0)).sum()),
                        "battles": len(meta), "bytes": nbytes, "min": round((time.time() - t0) / 60, 1)}))
        log(json.dumps({"weights": winfo}))
    probes, hist = {}, {}
    if a.read_only:
        for wd in widths:
            for inp in inputs:
                nc, npair = (len(x) for x in INPUTS[inp])
                net = HD.ProbeNet(nc, npair, width=wd, heads=int(pr["heads"]), ff=int(pr["ff"])).to(dev)
                net.load_state_dict(torch.load(D / f"probe_{inp}_{wd}.pt", map_location=dev))
                probes[f"{inp}@{wd}"] = net.eval()
        old = json.loads((D / "probe.json").read_text(encoding="utf-8"))
        winfo, hist, meta = old["weights"], old["loss_history"], [None] * old["sizes"]["train_battles"]
        log(json.dumps({"read_only": True, "probes": list(probes)}))
    held = None if a.read_only else held_split(d)
    for wd in ([] if a.read_only else widths):
        for inp in inputs:
            net, h = train_probe(d, w, inp, wd, dev, log, seed=17, passes=2 if a.smoke else None, held=held)
            probes[f"{inp}@{wd}"] = net
            hist[f"{inp}@{wd}"] = h
            torch.save(net.state_dict(), D / f"probe_{inp}_{wd}.pt")
    if not a.read_only:
        del d
    S, H, G = eval_specs(0.02 if a.smoke else None)
    if a.smoke:
        S, H, G = S[:a.smoke], H[:max(4, a.smoke // 4)], G[:a.smoke]
    rd = {}
    tseed = int(pr["seeds"]["tseed"])
    for sname, sp, scene, sb in (("S", S, True, 900), ("H", H, True, 950), ("G", G, False, 980)):
        rdr = Reader(probes, ana)
        t1 = time.time()
        ch = min(a.chunk, len(sp))
        play(a.teacher, sp, scene, _seeds(tseed + sb, len(sp), ch), ch, ar, P, rdr.hook,
             lambda ii, sp=sp, scene=scene: rdr.flush(ii, sp, "scene" if scene else "general"))
        rd[sname] = (rdr.data(), rdr.meta)
        log(json.dumps({"read": sname, "battles": len(sp), "unit_ticks": int(rd[sname][0]["b"].shape[0]),
                        "label_not_open": rdr.bad_open, "min": round((time.time() - t1) / 60, 1)}))
    names = list(probes) + [n + "~dcin" for n in probes]
    tabs = {n: {} for n in names}
    diag = {}
    for sname, (x, m) in rd.items():
        t = tables(x, m, names, sname)
        for n in names:
            tabs[n][sname] = t[n]
        diag[sname] = ingredient_diag(x, names)
    v = verdict(tabs, inputs, widths)
    res = {"teacher": a.teacher, "reach_m": OR.c_orders()["reach_m"], "sim": a.sim, "sim_key": EP.sim_key(),
           "sizes": {"train_battles": len(meta), "S": len(S), "H": len(H), "G": len(G)}, "weights": winfo,
           "loss_history": hist, "tables": tabs, "verdict": v, "ingredients_E": diag, "minutes": round((time.time() - t0) / 60, 1)}
    (D / ("probe_read.json" if a.read_only else "probe.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n in names:
        for s in ("S", "G", "H"):
            log(f"{n} {s}: {brief(tabs[n][s])}")
    log(json.dumps({"verdict": v}, indent=1))
    if bpath is not None and a.drop_buffer and bpath.exists():
        bpath.unlink()
    logf.close()
    return res


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("train", "after"):
        # the first training and the exam after by the move marks (tactic/imitate.py)
        from net.tactic import imitate as IM
        return IM.main(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("probe",))
    ap.add_argument("--inputs", default="A,B,C,X")
    ap.add_argument("--widths", default="", help="the probe's widths (default: config probe.width, width_retry)")
    ap.add_argument("--teacher", default="TEACH", help="the driver and labeller (a player name of exam_pin)")
    ap.add_argument("--out", default="tmp/tactic_slice/v23/stage4/probe")
    ap.add_argument("--sim", default="", help="the simulator's commit (said by the caller)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--smoke", type=int, default=0, help="n scenes + n general battles, one round, one pass; tiny reads")
    ap.add_argument("--buffer", default="", help="a file for the buffer: loaded if it exists, else written")
    ap.add_argument("--drop-buffer", action="store_true", help="delete the buffer file at the end")
    ap.add_argument("--collect-only", action="store_true", help="collect the buffer into --buffer and stop")
    ap.add_argument("--read-only", action="store_true", help="the reads only, of the probes saved in --out (probe_*.pt, "
                                                                 "probe.json): written to probe_read.json")
    ap.add_argument("--side", action="store_true", help="drill T0-2's side probe (module doc; --teacher TEACH2)")
    ap.add_argument("--reuse", action="store_true", help="--side: read side_train.pt / side_eval.pt already in --out "
                                                         "(their key compared, the frontal check run on a fresh draw)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    probe(a)


if __name__ == "__main__":
    main()
