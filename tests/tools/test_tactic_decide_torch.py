"""The tactician's output (tools/nn/tactic uzone, decide, teacher; design/tactic): the frames' rules (the nudge, the bans,
the last resort), the council's identity by slot, the teacher's formation with real numbers in a 2 x 2 zone, the choices'
log-probability read back. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.tactic import cfg  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import parcel as PC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import teacher as TC  # noqa: E402
from tools.nn.tactic import uzone as U  # noqa: E402
from tools.nn.tactic import zone as ZN  # noqa: E402
from tools.nn.unit.net import NEG  # noqa: E402

CELL = 320.0 / 30


def tiles_2x2(R=1, heading=0.0):
    p = PC.Parcel.make(0, [0, 1, 2], [], 0.0, 0.0, heading, line=False, n=6)
    p = PC.Parcel.cat([p] * R)
    return ZN.Tiles.of(p)


def tiles_line(n):
    p = PC.Parcel.make(0, list(range(n)), [], 0.0, 0.0, 0.0, line=True, n=2 * n)
    return ZN.Tiles.of(p)


def cell_at(s, f):
    """The cell index of a place (row x 30 + col)."""
    col = int(torch.floor(torch.tensor(s / CELL + 15.0)))
    row = int(torch.floor(torch.tensor(15.0 - f / CELL)))
    return row * 30 + col


def put(seq, j, w, d, cell, need=0):
    ok, s, f = seq.place(j, torch.tensor([float(w)]), torch.tensor([float(d)]), torch.tensor([need]))
    able = torch.tensor([True])
    assert bool(ok.flatten(1)[0, cell]), "the cell is closed"
    seq.commit(j, torch.tensor([cell]), s, f, torch.tensor([float(w)]), torch.tensor([float(d)]), able)
    return float(seq.s[0, j]), float(seq.f[0, j])


def test_page_example_two_80_m_frames_flush():
    """The page: 2 zones of 80 m in a 2 x 2, the cell +-37.3 m is nudged to +-40 m, flush at the middle and the edge."""
    seq = U.Seq(tiles_2x2(), 3)
    s, _ = put(seq, 0, 80, 48, cell_at(-37.0, 37.0))
    assert s == pytest.approx(-40.0, abs=1e-3)
    s2, _ = put(seq, 1, 80, 48, cell_at(37.0, 37.0))
    assert s2 == pytest.approx(40.0, abs=1e-3)
    assert not U.bad_frames(seq.frames(), seq.tiles)[0].any()


def test_cell_closed_when_it_laps_over_more_than_a_cell():
    seq = U.Seq(tiles_2x2(), 3)
    ok, _, _ = seq.place(0, torch.tensor([80.0]), torch.tensor([48.0]), torch.tensor([0]))
    # 80 m wide, half 40: a centre at s = 37.33 + 2 x cell laps over the edge by 2 cells - 2.67 > 1 cell
    assert not bool(ok.flatten(1)[0, cell_at(37.0 + 2 * CELL, 0.0)])
    assert bool(ok.flatten(1)[0, cell_at(37.0 + CELL, 0.0)])                 # lap 8 m: nudged in
    assert not bool(ok.flatten(1)[0, cell_at(150.0, 0.0)])
    # over a frame: a neighbour's centre closer than the frames' half widths by more than a cell
    put(seq, 0, 80, 48, cell_at(-37.0, 37.0))
    ok, s, _ = seq.place(1, torch.tensor([80.0]), torch.tensor([48.0]), torch.tensor([0]))
    assert not bool(ok.flatten(1)[0, cell_at(16.0, 37.0)])                   # the lap 24 - ... > a cell
    assert bool(ok.flatten(1)[0, cell_at(37.0, 37.0)])


def test_gap_under_a_cell_is_closed_flush():
    seq = U.Seq(tiles_2x2(), 3)
    put(seq, 0, 48, 48, cell_at(-50.0, 50.0))
    x0 = float(seq.s[0, 0])
    w = 48
    # the next frame a little to the right of the first: the gap under a cell closes
    target = x0 + 48 + 5.0
    s, f = put(seq, 1, w, 48, cell_at(target, 50.0))
    assert s - x0 == pytest.approx(48.0, abs=1e-3)
    assert f == pytest.approx(float(seq.f[0, 0]), abs=1e-3)


def test_room_keeps_a_place_for_the_units_still_to_choose():
    """An 80 x 80 m frame in the very middle of a 2 x 2 leaves no room for a 48 m frame: closed while others still choose."""
    seq = U.Seq(tiles_2x2(), 3)
    w = torch.tensor([80.0])
    ok, _, _ = seq.place(0, w, w, torch.tensor([2]))
    mid = cell_at(1.0, 1.0)
    assert not bool(ok.flatten(1)[0, mid])
    free, _, _ = seq.place(0, w, w, torch.tensor([0]))
    assert bool(free.flatten(1)[0, mid])


def test_last_resort_least_size_then_error():
    seq = U.Seq(tiles_line(2), 2)                                               # 160 x 80 m
    w = torch.tensor([80.0])
    put(seq, 0, 80, 80, cell_at(-40.0, 0.0))
    # room for an 80 x 80 m at s = 40: no last resort
    *_, fell = seq.place_or_least(1, w, w, torch.tensor([0]), torch.tensor([True]))
    assert not bool(fell[0])
    # every tile taken: the least size cannot help - the error of the level above
    seq2 = U.Seq(tiles_line(2), 3)
    put(seq2, 0, 80, 80, cell_at(-40.0, 0.0))
    put(seq2, 1, 80, 80, cell_at(40.0, 0.0))
    with pytest.raises(PC.ParcelError):
        seq2.place_or_least(2, w, w, torch.tensor([0]), torch.tensor([True]))
    # not able: no error, its answer is never used
    assert not bool(seq2.place_or_least(2, w, w, torch.tensor([0]), torch.tensor([False]))[-1][0])
    # the middles of both rows taken: no cell for 80 x 80 m, the sides still hold 48 x 48 m: the least size, and the
    # frame that comes of it is whole
    seq3 = U.Seq(tiles_2x2(), 3)
    put(seq3, 0, 64, 48, cell_at(0.0, -40.0))
    put(seq3, 1, 80, 80, cell_at(0.0, 40.0))
    ok, s, f, wm, dm, fell = seq3.place_or_least(2, w, w, torch.tensor([0]), torch.tensor([True]))
    assert bool(fell[0]) and wm.item() == 48.0 and dm.item() == 48.0 and bool(ok.any())
    c = ok.flatten(1).float().argmax(1)
    seq3.commit(2, c, s, f, wm, dm, torch.tensor([True]))
    over, out = U.bad_frames(seq3.frames(), seq3.tiles)
    assert not over.any() and not out.any()


def test_holes_close_their_tile():
    t = tiles_2x2()
    t.drop(torch.tensor([True]), torch.tensor([-40.0]), torch.tensor([40.0]))    # the front left tile is gone
    assert t.mask[0].sum() == 3
    seq = U.Seq(t, 2)
    ok, _, _ = seq.place(0, torch.tensor([48.0]), torch.tensor([48.0]), torch.tensor([0]))
    assert not bool(ok.flatten(1)[0, cell_at(-40.0, 40.0)])
    assert bool(ok.flatten(1)[0, cell_at(40.0, 40.0)])


def test_random_sequences_keep_the_promise():
    """Any sizes and any open cells, in slot order: the frames stay inside the zone and do not overlap, and the last
    resort's error never fires while the tiles are at least the units."""
    gen = torch.Generator().manual_seed(1)
    R, G = 64, 3
    t = tiles_2x2(R, heading=33.0)
    st = U.steps()
    for _ in range(20):
        seq = U.Seq(t, G)
        able = torch.ones((R, G), dtype=torch.bool)
        left = DC.left_after(able)
        for j in range(G):
            w = st[torch.randint(3, (R,), generator=gen)]
            d = st[torch.randint(3, (R,), generator=gen)]
            ok, s, f, wm, dm, _ = seq.place_or_least(j, w, d, left[:, j], able[:, j])
            c = torch.multinomial(ok.flatten(1).float(), 1, generator=gen)[:, 0]
            seq.commit(j, c, s, f, wm, dm, able[:, j])
        over, out = U.bad_frames(seq.frames(), t)
        assert not over.any() and not out.any()


def raw_of(places):
    """[1, N, RAW] with my units at (right, forward) in a zone at the origin, heading 0: (s, f, range)."""
    raw = torch.zeros((1, len(places), len(TR.RAW)))
    for i, (s, f, rng) in enumerate(places):
        raw[0, i, TR.I["x"]], raw[0, i, TR.I["z"]], raw[0, i, TR.I["range"]] = s, f, rng
        raw[0, i, TR.I["alive"]] = 1.0
    return raw


@pytest.mark.parametrize("start", ["shooters_front", "infantry_front"])
@pytest.mark.parametrize("swap", [False, True])
def test_teacher_real_numbers(start, swap):
    """Both starts, the 2 x 2 zone: 80 x 48 m flush at s = +-40 m, f = 37.33 m (the nearest cell's centre to 40 m), the
    left infantry left; the shooters 64 x 48 m in the back row's middle (the cell +-5.33 m from the middle: a cell's edge
    runs along it); nothing overlaps."""
    fwd = 30.0 if start == "shooters_front" else -30.0
    inf = [(-35.0, -fwd, 0.0), (35.0, -fwd, 0.0)]
    if swap:
        inf = inf[::-1]
    places = [inf[0], (0.0, fwd, 160.0), inf[1]]          # the shooters in the middle slot
    raw = raw_of(places)
    able = torch.ones((1, 3), dtype=torch.bool)
    t = tiles_2x2()
    out = TC.drive(raw, able, t)
    fr = out["frames"]
    a, b, sh = fr["s"][0, 0].item(), fr["s"][0, 2].item(), fr["s"][0, 1].item()
    left, right = (a, b) if not swap else (b, a)
    assert left == pytest.approx(-40.0, abs=1e-3) and right == pytest.approx(40.0, abs=1e-3)
    for j in (0, 2):
        assert fr["f"][0, j].item() == pytest.approx(37.333, abs=1e-2)
        assert 2 * fr["hw"][0, j].item() == 80.0 and 2 * fr["hd"][0, j].item() == 48.0
    assert abs(sh) == pytest.approx(5.333, abs=1e-2)
    assert fr["f"][0, 1].item() == pytest.approx(-37.333, abs=1e-2)
    assert 2 * fr["hw"][0, 1].item() == 64.0 and 2 * fr["hd"][0, 1].item() == 48.0
    assert not out["fell"].any()
    over, out_ = U.bad_frames(fr, t)
    assert not over.any() and not out_.any()


def test_teacher_label_partial_for_neighbours():
    t = tiles_2x2()
    seq = U.Seq(t, 1)
    ok = torch.ones((1, 30, 30), dtype=torch.bool)
    q, some = TC.label(seq, torch.tensor([-37.33]), torch.tensor([37.33]), ok)
    assert bool(some[0])
    c = cell_at(-37.0, 37.0)
    assert q.argmax(1).item() == c
    assert 0.3 < (q[0, c + 1] / q[0, c]).item() < 0.9               # a neighbour: partly
    assert q[0, c + 3].item() / q[0, c].item() < 0.05               # three cells off: hardly
    closed = ok.clone()
    closed.view(1, -1)[0, c] = False
    q2, _ = TC.label(seq, torch.tensor([-37.33]), torch.tensor([37.33]), closed)
    assert q2[0, c].item() == 0.0


def test_teacher_weight_falls_to_zero():
    n = cfg()["teach"]["decay_updates"]
    assert TC.weight_at(0) == pytest.approx(cfg()["teach"]["weight"])
    assert 0 < TC.weight_at(n // 2) < TC.weight_at(0)
    assert TC.weight_at(n) == 0.0 and TC.weight_at(n + 5) == 0.0


def test_order_vec():
    o = DC.order_vec(2, "hold", "any")
    assert o.shape == (2, 7)
    assert o[0].tolist() == [0, 1, 0, 0, 1, 0, 0]
    assert DC.order_vec(1, "avoid", "fast")[0].tolist() == [0, 0, 0, 1, 0, 0, 1]


def test_council_masks_and_keeps_the_rows():
    """An empty row gives zeros and changes nothing; the rows keep their places (identity by slot)."""
    net = DC.Council().eval()
    cards = torch.randn(2, 5, 128)
    mask = torch.tensor([[True, True, True, False, False], [True, False, True, True, False]])
    s = torch.randn(2, 128)
    a, sm = net(cards, s, mask)
    assert a.shape == (2, 5, 128) and sm.shape == (2, 128)
    assert (a[~mask] == 0).all()
    c2 = cards.clone()
    c2[~mask] = 99.0
    b, _ = net(c2, s, mask)
    assert torch.allclose(a, b, atol=1e-5)
    # swapping two rows swaps the answers
    perm = torch.tensor([2, 1, 0, 3, 4])
    d, _ = net(cards[:, perm], s, mask[:, perm])
    assert torch.allclose(d[0, 0], a[0, 2], atol=1e-4)


def test_bans_are_neg_and_frozen_blocks_do_not_train():
    net = DC.TacticNet()
    assert all(not p.requires_grad for p in net.enc.inp.analysis.parameters())
    assert all(not p.requires_grad for p in net.enc.inp.squeeze.parameters())
    assert all(p.requires_grad for p in net.enc.inp.map.parameters())
    assert all(p.requires_grad for p in net.enc.council.parameters())
    r = DC.report(net)
    assert r["to_train"] == r["total"] - r["analysis_frozen"] - r["squeeze_frozen"]
    assert NEG == -1e9


def frames_of(places):
    """{s, f, hw, hd, on} [1, G] from (s, f, width, depth) or None (no frame)."""
    G = len(places)
    out = {k: torch.zeros(1, G) for k in ("s", "f", "hw", "hd")}
    out["on"] = torch.zeros(1, G, dtype=torch.bool)
    for j, q in enumerate(places):
        if q is not None:
            out["s"][0, j], out["f"][0, j], out["hw"][0, j], out["hd"][0, j] = q[0], q[1], q[2] / 2, q[3] / 2
            out["on"][0, j] = True
    return out


TEACHER_FRAMES = [(-40.0, 37.33, 80.0, 48.0), (40.0, 37.33, 80.0, 48.0), (5.33, -37.33, 64.0, 48.0)]


def test_must_change_no_frame_or_a_hole_under_it():
    able = torch.ones((1, 3), dtype=torch.bool)
    t = tiles_2x2()
    assert U.Seq(t, 3).must_change(None, able).all()
    prev = frames_of(TEACHER_FRAMES[:2] + [None])
    assert U.Seq(t, 3).must_change(prev, able)[0].tolist() == [False, False, True]
    t.drop(torch.tensor([True]), torch.tensor([-40.0]), torch.tensor([40.0]))     # the front left tile is gone
    assert U.Seq(t, 3).must_change(frames_of(TEACHER_FRAMES), able)[0].tolist() == [True, False, False]
    out_of_zone = frames_of([(70.0, 0.0, 80.0, 48.0), None, None])
    assert U.Seq(tiles_2x2(), 3).must_change(out_of_zone, able)[0, 0].item()


def test_keepers_close_their_cells_first():
    """The last slot keeps, the first slots change: they may not lap over it."""
    seq = U.Seq(tiles_2x2(), 3)
    able = torch.ones((1, 3), dtype=torch.bool)
    prev = frames_of(TEACHER_FRAMES)
    kept = seq.keep(torch.tensor([[False, False, True]]), prev, able)
    assert kept[0].tolist() == [False, False, True]
    assert seq.s[0, 2].item() == pytest.approx(5.33) and seq.on[0].tolist() == [False, False, True]
    ok, _, _ = seq.place(0, torch.tensor([64.0]), torch.tensor([48.0]), torch.tensor([1]))
    assert not bool(ok.flatten(1)[0, cell_at(5.0, -37.0)])          # on the kept frame
    assert bool(ok.flatten(1)[0, cell_at(-40.0, 37.0)])


def test_keep_lets_go_when_too_few_tiles_remain():
    """A frame across the zone's middle blocks all four tiles: with two units still to change it is let go, from the last
    slot back; alone (nobody to change) it stays."""
    able = torch.ones((1, 3), dtype=torch.bool)
    big = (0.0, 0.0, 80.0, 80.0)
    prev = frames_of([None, None, big])
    kept = U.Seq(tiles_2x2(), 3).keep(torch.tensor([[False, False, True]]), prev, able)
    assert not kept.any()
    only = torch.tensor([[False, False, True]])
    kept = U.Seq(tiles_2x2(), 3).keep(only.clone(), prev, only)
    assert kept[0, 2].item()


def test_random_keep_and_change_keep_the_promise():
    """Any mix of keepers and changers over many decisions: inside the zone, no overlaps, no error of the level above."""
    gen = torch.Generator().manual_seed(2)
    R, G = 64, 3
    t = tiles_2x2(R, heading=33.0)
    st = U.steps()
    able = torch.ones((R, G), dtype=torch.bool)
    prev = None
    for _ in range(30):
        seq = U.Seq(t, G)
        must = seq.must_change(prev, able)
        want = (torch.rand((R, G), generator=gen) < 0.6) & ~must
        kept = seq.keep(want.clone(), prev, able)
        changing = able & ~kept
        left = DC.left_after(changing)
        for j in range(G):
            a = changing[:, j]
            w, d = st[torch.randint(3, (R,), generator=gen)], st[torch.randint(3, (R,), generator=gen)]
            ok, s, f, wm, dm, _ = seq.place_or_least(j, w, d, left[:, j], a)
            c = torch.multinomial(ok.flatten(1).float() + 1e-9, 1, generator=gen)[:, 0]
            seq.commit(j, c, s, f, wm, dm, a)
        fr = seq.frames()
        over, out = U.bad_frames(fr, t)
        assert not over.any() and not out.any() and fr["on"].all()
        if prev is not None:
            for name in ("s", "f", "hw", "hd"):
                assert torch.equal(fr[name][kept], prev[name][kept])
        prev = {k: v.clone() for k, v in fr.items()}


class Inp:
    """The smallest network's input for the walk: my 3 rows, a random map."""
    def __new__(cls, R=4):
        return {"cards": torch.randn(R, 6, 128), "map": torch.randn(R, 64, 30, 30)}


def ctx_for(R=4, prev=None):
    t = tiles_2x2(R)
    able = torch.ones((R, 3), dtype=torch.bool)
    rows = torch.arange(3)[None].expand(R, 3)
    return DC.Ctx(t, rows, able, rows, torch.zeros(R, 3, len(TR.RAW)), prev)


def repeat(fr, R):
    return {k: v.expand(R, *v.shape[1:]).clone() for k, v in fr.items()}


def test_first_decision_must_change_and_keep_head_starts_on_keep():
    h = DC.ZoneHeads()
    torch.manual_seed(1)
    out = DC.choose(Inp(), ctx_for(), h)
    assert not out["kept"].any() and out["must"].all() and (out["keep"] == DC.CHANGE).all()
    assert out["frames"]["on"].all()
    c = torch.randn(1, 128)
    p = torch.softmax(h.keep_logits(c[None], None), -1)[0, 0, DC.KEEP].item()
    b = cfg()["zone_head"]["keep_bias"]
    assert p == pytest.approx(torch.sigmoid(torch.tensor(float(b))).item(), abs=1e-5)


def test_keepers_size_and_cell_do_not_count():
    """The log-probability and the entropy of a unit that keeps hold no size and no cell; the frame stays bit for bit."""
    torch.manual_seed(2)
    h = DC.ZoneHeads()
    with torch.no_grad():
        h.keep.bias[DC.KEEP] = 50.0          # all keep
    prev = repeat(frames_of(TEACHER_FRAMES), 4)
    enc = Inp()
    out = DC.choose(enc, ctx_for(prev=prev), h)
    assert out["kept"].all()
    assert out["ent"]["where"].abs().max() == 0 and out["ent"]["width"].abs().max() == 0
    for name in ("s", "f", "hw", "hd"):
        assert torch.equal(out["frames"][name], prev[name])
    act = {k: out[k].clone() for k in ("keep", "width", "depth", "cell")}
    act["width"][:] = 2
    act["cell"][:] = 7
    again = DC.choose(enc, ctx_for(prev=prev), h, act=act)
    assert torch.equal(again["lp"], out["lp"])


def test_the_walk_reads_back_with_keepers_and_changers():
    torch.manual_seed(3)
    h = DC.ZoneHeads()
    with torch.no_grad():
        h.keep.bias[DC.KEEP] = 0.0
    prev = repeat(frames_of(TEACHER_FRAMES), 32)
    enc = Inp(32)
    out = DC.choose(enc, ctx_for(32, prev), h)
    assert 0 < out["kept"].sum() < 96
    again = DC.choose(enc, ctx_for(32, prev), h, act={k: out[k] for k in ("keep", "width", "depth", "cell")})
    assert (again["lp"] - out["lp"]).abs().max() < 1e-4
    for name in ("s", "f", "hw", "hd"):
        assert torch.equal(again["frames"][name], out["frames"][name])


def test_teacher_says_keep_only_on_the_formula_s_frame():
    raw = raw_of([(-35.0, -30.0, 0.0), (35.0, -30.0, 0.0), (0.0, 30.0, 160.0)])
    t = tiles_2x2()
    able = torch.ones((1, 3), dtype=torch.bool)
    rows = torch.arange(3)[None]
    h = DC.ZoneHeads()
    enc = Inp(1)
    # the formula's frames: all three keep, nothing else is compared
    ctx = DC.Ctx(t, rows, able, rows, raw, frames_of(TEACHER_FRAMES))
    term, st = TC.loss(enc, ctx, h)
    assert st["keep_said"] == 1.0 and term.item() > 0
    # one frame 20 m off (over change_m): that one changes; size_ok and cell_ok then read that unit alone
    off = [(-40.0, 17.0, 80.0, 48.0)] + TEACHER_FRAMES[1:]
    _, st = TC.loss(enc, DC.Ctx(t, rows, able, rows, raw, frames_of(off)), h)
    assert st["keep_said"] == pytest.approx(2 / 3)
    assert st["n"] == 1
    # within change_m: still the formula's
    near = [(-40.0, 37.33 - 10.0, 80.0, 48.0)] + TEACHER_FRAMES[1:]
    _, st = TC.loss(enc, DC.Ctx(t, rows, able, rows, raw, frames_of(near)), h)
    assert st["keep_said"] == 1.0
    # another size: changes
    other = [(-40.0, 37.33, 64.0, 48.0)] + TEACHER_FRAMES[1:]
    _, st = TC.loss(enc, DC.Ctx(t, rows, able, rows, raw, frames_of(other)), h)
    assert st["keep_said"] == pytest.approx(2 / 3)
    # no frames yet: every unit must change, the head has no term
    _, st = TC.loss(enc, DC.Ctx(t, rows, able, rows, raw, None), h)
    assert st["keep_said"] == 0.0 and st["n"] == 3


def test_keep_head_reads_the_frame():
    """Beside the card the head reads the unit's frame in force: another frame, another answer once it has learned."""
    h = DC.ZoneHeads()
    torch.nn.init.normal_(h.keep.weight)
    c = torch.randn(1, 3, 128)
    a = h.keep_logits(c, frames_of(TEACHER_FRAMES))
    b = h.keep_logits(c, frames_of([(-20.0, 0.0, 48.0, 48.0)] + TEACHER_FRAMES[1:]))
    assert not torch.allclose(a[0, 0], b[0, 0])
    assert torch.equal(a[0, 1:], b[0, 1:])
