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
