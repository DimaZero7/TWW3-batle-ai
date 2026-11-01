"""The first drill's teacher (design/tactic "Строй учителя", "Учитель"; the user's decisions 01.11): a formula for the
formation on every decision of the network, while the network itself drives.

    Two wide infantry zones side by side in the front row of the group's zone (80 x 48 m each), the shooters' zone
    (64 x 48 m) in the middle of the back row. The left infantry goes to the left zone, the right one to the right, by
    where they stand now, so they do not cross and the answer is one.

    ideal()     per unit the size's steps and the place wanted (in the zone's axes)
    label()     the where head's soft label: the cell of the place and, partly, its neighbours (exp(-d^2 / 2 sigma^2) of
                the distance in cells; 11 m off is not 100 m off), closed cells out
    drive()     the same formation played by the code: the frames the formula gives through uzone.Seq (the yardstick of
                the exam: the network chooses among the same cells)
    loss()      the term of the training: the size's cross-entropy + the where's against the soft label, with the
                teacher's earlier frames in force
"""
import torch
import torch.nn.functional as F

from tools.nn.tactic import cfg
from tools.nn.tactic import raw as TR
from tools.nn.tactic import uzone as U
from tools.nn.tactic.decide import left_after
from tools.nn.unit import zmap as Z
from tools.nn.unit.net import NEG


def _step(m, device):
    """The index of the size m in size_steps_m."""
    return (U.steps(device) - float(m)).abs().argmin()


def ideal(raw_mine, able, tiles):
    """{width, depth} [R, G] long (the steps), {s, f} [R, G] the places, in the zone's axes. raw_mine [R, G, RAW] my units'
    raw state, able [R, G] the ones that choose."""
    c = cfg()["teach"]
    dev = raw_mine.device
    G = able.shape[1]
    I = TR.I
    s, _ = Z.local(raw_mine[..., I["x"]], raw_mine[..., I["z"]], tiles.cx[:, None], tiles.cz[:, None],
                   tiles.heading[:, None])
    shooter = raw_mine[..., I["range"]] > 0
    inf = able & ~shooter
    zs, zf = tiles.front[:, None] / 2, tiles.depth[:, None] / 2
    # left / right: of two infantry units the one further left goes left; one alone: on its own side of the middle
    key = torch.where(inf, s, torch.full_like(s, 1e9))
    two = inf.sum(1) >= 2
    left_one = F.one_hot(key.argmin(1), G).bool() & two[:, None]
    side = torch.where(two[:, None], torch.where(left_one, -1.0, 1.0), torch.where(s < 0, -1.0, 1.0))
    out_s = torch.where(shooter, torch.zeros_like(s), side * zs / 2)
    out_f = torch.where(shooter, -zf / 2, zf / 2)
    iw, id_ = _step(c["infantry_m"][0], dev), _step(c["infantry_m"][1], dev)
    sw, sd = _step(c["shooters_m"][0], dev), _step(c["shooters_m"][1], dev)
    return {"width": torch.where(shooter, sw, iw), "depth": torch.where(shooter, sd, id_), "s": out_s, "f": out_f}


def cell_dist2(seq, s, f):
    """[R, C, C] the squared distance (in cells) of each cell's centre from the place (s, f) [R]."""
    cm = seq.cell.view(-1, 1, 1)
    return ((seq.s0 - s.view(-1, 1, 1)) ** 2 + (seq.f0 - f.view(-1, 1, 1)) ** 2) / (cm * cm)


def label(seq, s, f, ok):
    """([R, 900] the soft label: closed cells 0; [R] bool: some cell open near the place)."""
    sigma = float(cfg()["teach"]["sigma_cells"])
    w = torch.exp(-cell_dist2(seq, s, f) / (2 * sigma * sigma)) * ok.float()
    tot = w.flatten(1).sum(1, keepdim=True)
    return w.flatten(1) / tot.clamp(min=1e-12), tot[:, 0] > 1e-12


def _steps_of(m, st):
    return (m[:, None] - st).abs().argmin(1)


def drive(raw_mine, able, tiles):
    """The formula's frames: {width, depth, cell} [R, G], frames, fell - as decide.choose gives."""
    R, G = able.shape
    dev = able.device
    seq = U.Seq(tiles, G)
    want = ideal(raw_mine, able, tiles)
    st = U.steps(dev)
    left = left_after(able)
    out = {k: torch.zeros((R, G), dtype=torch.long, device=dev) for k in ("width", "depth", "cell")}
    fell = torch.zeros(R, device=dev)
    for j in range(G):
        a = able[:, j]
        ok, s, f, wm, dm, fb = seq.place_or_least(j, st[want["width"][:, j]], st[want["depth"][:, j]], left[:, j], a)
        d = cell_dist2(seq, want["s"][:, j], want["f"][:, j]).flatten(1)
        c = d.masked_fill(~ok.flatten(1), 1e12).argmin(1)
        seq.commit(j, c, s, f, wm, dm, a)
        out["width"][:, j], out["depth"][:, j], out["cell"][:, j] = _steps_of(wm, st), _steps_of(dm, st), c
        fell = fell + fb.float()
    return {**out, "frames": seq.frames(), "fell": fell}


def loss(enc, ctx, heads):
    """(term, stats): the teacher's term on the groups given: for each unit the size's two cross-entropies and the where's
    cross-entropy against the soft label, the teacher's earlier frames in force. stats: size_ok, cell_ok (the network's
    best cell within two cells of the wanted), per unit."""
    tiles, able, rows = ctx.tiles, ctx.able, ctx.rows
    R, G = able.shape
    dev = able.device
    seq = U.Seq(tiles, G)
    want = ideal(ctx.mine_raw, able, tiles)
    st = U.steps(dev)
    left = left_after(able)
    idx = torch.arange(R, device=dev)
    tot, n = torch.zeros((), device=dev), 0.0
    size_ok, cell_ok = 0.0, 0.0
    for j in range(G):
        a = able[:, j]
        card = enc["cards"][idx, rows[:, j]]
        w, d = want["width"][:, j], want["depth"][:, j]
        ok, s, f, wm, dm, _ = seq.place_or_least(j, st[w], st[d], left[:, j], a)
        lg = heads.where(card, _steps_of(wm, st), _steps_of(dm, st), enc["map"]).masked_fill(~ok.flatten(1), NEG)
        q, some = label(seq, want["s"][:, j], want["f"][:, j], ok)
        lw, ld = heads.width(card), heads.depth(card)
        ce = F.cross_entropy(lw, w, reduction="none") + F.cross_entropy(ld, d, reduction="none")
        ce = ce - torch.where(q > 0, q * F.log_softmax(lg, -1), torch.zeros_like(q)).sum(1)
        use = a & some
        tot = tot + (ce * use).sum()
        n += float(use.sum())
        size_ok += float(((lw.argmax(1) == w) & (ld.argmax(1) == d) & use).sum())
        dist = cell_dist2(seq, want["s"][:, j], want["f"][:, j]).flatten(1)
        cell_ok += float(((dist.gather(1, lg.argmax(1)[:, None])[:, 0] <= 4.0) & use).sum())
        seq.commit(j, dist.masked_fill(~ok.flatten(1), 1e12).argmin(1), s, f, wm, dm, a)
    n = max(n, 1.0)
    return tot / n, {"size_ok": size_ok / n, "cell_ok": cell_ok / n, "n": n}


def weight_at(update):
    """The term's weight at this update of the run: falling to zero linearly (config teach)."""
    c = cfg()["teach"]
    return float(c["weight"]) * max(0.0, 1.0 - update / max(float(c["decay_updates"]), 1.0))
