"""tools/nn/general/map_cnn.py: the general's map analysis (design/general/index.html, "Проверки анализа карты",
group 1 "При сборке")."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import map_batch as mb                      # noqa: E402
from tools.nn.general import map_cnn as mc                        # noqa: E402
from tools.nn.general import map_input as mi                      # noqa: E402
from tools.nn.general import map_probe as mp                      # noqa: E402

F64 = torch.float64
N = mi.GRID


def model(seed=0, dtype=F64, **kw):
    torch.manual_seed(seed)
    m = mc.MapCNN(**kw).to(dtype)
    with torch.no_grad():                    # biases nonzero, so "outside" padding differs from zero padding
        for c in list(m.stem) + list(m.convs):
            if c.bias is not None:
                c.bias.uniform_(-0.2, 0.2)
    return m


def plain(fill="field", dtype=F64):
    """[1, 6, 128, 128] a map of nothing but field (or outside)."""
    x = torch.zeros(1, 6, N, N, dtype=dtype)
    x[:, 0 if fill == "field" else 1] = 1
    return x


def put_field(x, z0, z1, x0, x1):
    x[:, 0, z0:z1, x0:x1] = 1
    x[:, 1, z0:z1, x0:x1] = 0


def arena_layers(side=1, dtype=F64):
    return mi.to_tensor(mi.build(mi.arena_spec()), side).to(dtype)[None]


def test_shape_and_size():
    m = mc.MapCNN()
    assert mc.weights(m) == 29504
    assert mc.weights(mc.MapCNN(64)) == 114304
    assert (m.regions, m.region_m()) == (32, 64.0)
    assert m(plain(dtype=torch.float32)).shape == (1, 1024, 32)
    assert mc.MapCNN(64)(plain(dtype=torch.float32)).shape == (1, 1024, 64)
    m96 = mc.MapCNN(pools=(2, 3))
    assert (m96.regions, m96.region_m()) == (22, 96.0)
    assert m96(plain(dtype=torch.float32)).shape == (1, 22 * 22, 32)


def test_the_window_looks_only_nearby():
    """One cell changed on a plain map: each step answers only within its reach of that cell."""
    m = model()
    base = plain()
    pt = plain()
    cz, cx = 61, 70
    pt[0, 2, cz, cx] = 1                                       # one cell of "our zone"
    with torch.no_grad():
        a, b = m.steps(base), m.steps(pt)
    # the box each step may change: stem +-1 cell, conv1 +-2, pool1 //2, conv2 +-1, pool2 //2, head +-1
    lo, hi = np.array([cz, cx]) - 2, np.array([cz, cx]) + 2
    boxes = [(np.array([cz, cx]) - 1, np.array([cz, cx]) + 1), (lo, hi)]
    p1 = (lo // 2, hi // 2)
    c2 = (p1[0] - 1, p1[1] + 1)
    p2 = (c2[0] // 2, c2[1] // 2)
    boxes += [p1, c2, p2, (p2[0] - 1, p2[1] + 1)]
    for k, (fa, fb, (blo, bhi)) in enumerate(zip(a, b, boxes)):
        d = (fa - fb).abs().amax(1)[0]
        mask = torch.zeros_like(d, dtype=torch.bool)
        mask[blo[0]:bhi[0] + 1, blo[1]:bhi[1] + 1] = True
        assert float(d[~mask].max()) == 0.0, mp.STEP_NAMES[k]
        assert float(d[mask].max()) > 0, mp.STEP_NAMES[k]
    assert float((a[0] - b[0]).abs().amax(1)[0, cz, cx]) > 0
    assert (boxes[-1][1] - boxes[-1][0] + 1).tolist() == [5, 5]   # the head's reach: 5 x 5 regions (320 m)


def test_a_shift_by_a_cell_shifts_the_features():
    """A patch moved by 1 cell: the full-size steps move by 1 cell; moved by 4 cells: the regions by 1."""
    m = model(1)
    torch.manual_seed(5)
    x = plain()
    x[0, 2:, 50:62, 40:56] = (torch.rand(4, 12, 16, dtype=F64) > 0.5).to(F64)
    x[0, 0, 55:58, 44:50] = 0.5
    x[0, 1, 55:58, 44:50] = 0.5
    for dz, dx, check in ((1, 0, (0, 1)), (0, 1, (0, 1)), (4, 4, (0, 1, 2, 3, 4, 5)), (-4, 8, (0, 1, 2, 3, 4, 5))):
        y = torch.roll(x, (dz, dx), (2, 3))
        with torch.no_grad():
            a, b = m.steps(x), m.steps(y)
        for k in check:
            s = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4}[k]
            want = torch.roll(a[k], (dz // s, dx // s), (2, 3))
            e = 16 // s                                           # away from the grid's edge (the map's edge)
            assert torch.allclose(want[..., e:-e, e:-e], b[k][..., e:-e, e:-e], atol=1e-12, rtol=0), \
                (dz, dx, mp.STEP_NAMES[k])
    with torch.no_grad():                                         # 2 cells: inside a region, the summary changes
        assert not torch.allclose(m.grid(x), m.grid(torch.roll(x, (2, 0), (2, 3))))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_new_layer_with_zero_weights_changes_nothing(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda")
    torch.manual_seed(2)
    m = mc.MapCNN().to(device)
    x = torch.cat([arena_layers(1, torch.float32), arena_layers(2, torch.float32)]).to(device)
    before = m(x)
    m.add_layers(2, outside=(1.0, 0.0))
    extra = torch.rand(2, 2, N, N, device=device)
    after = m(torch.cat([x, extra], 1))
    assert torch.equal(before, after)
    assert m.in_layers == 8
    after.square().mean().backward()                      # the new weights do learn
    assert float(m.stem[1].weight.grad.abs().sum()) > 0
    with pytest.raises(AssertionError):
        m(x)


@pytest.mark.parametrize("pools", [(2, 2), (2, 3)])
def test_the_grid_edge_reads_as_the_middle(pools):
    """Beyond the grid is "outside": a patch of field at the grid's corner gives the features of the same
    patch in the middle of an all-outside grid (moved by a whole number of regions)."""
    m = model(3, pools=pools)
    corner, middle = plain("outside"), plain("outside")
    put_field(corner, 0, 10, 0, 6)
    corner[0, 2, 2:5, 1:3] = 1
    sh = 60                                                     # 15 regions of 4 cells, 10 of 6
    put_field(middle, sh, sh + 10, sh, sh + 6)
    middle[0, 2, sh + 2:sh + 5, sh + 1:sh + 3] = 1
    with torch.no_grad():
        a, b = m.steps(corner), m.steps(middle)
        full = m.steps(plain("outside"))
    S = pools[0] * pools[1]
    for k, s in enumerate((1, 1, 2, 2, S, S)):
        n = a[k].shape[-1]
        o = sh // s
        assert torch.allclose(a[k][..., :n - o, :n - o], b[k][..., o:, o:], atol=1e-12, rtol=0), mp.STEP_NAMES[k]
        # nothing but outside: the same numbers at every position, the grid's edge included
        f = full[k][0]
        assert torch.allclose(f, f[:, n // 2, n // 2, None, None].expand_as(f), atol=1e-12, rtol=0)


def test_zero_padding_would_show_the_edge():
    """With zeros beyond the grid the edge test above fails: it really tests the margin."""
    m = model(3)
    x = plain("outside")
    m.stem_outside = [(0.0,) * 6]
    with torch.no_grad():
        f = m.steps(x)
    for k in range(6):
        assert not torch.allclose(f[k][0, :, 0, 0], f[k][0, :, -1 + f[k].shape[-1] // 2, f[k].shape[-1] // 2])


def test_numbers_keep_their_scale():
    """On real maps (arena both sides, the example, turned drills) every step's numbers neither blow up nor
    die out at the start (He init, no normalisation needed), for several seeds and for 64 features."""
    descs = [mc._drill_desc(i) for i in range(0, 64, 4)]
    drill = mb.layers(mb.rects_tensor([mb.zones_of(d) for d in descs]), [1 + i % 2 for i in range(len(descs))])
    maps = torch.cat([arena_layers(1, torch.float32), arena_layers(2, torch.float32),
                      mi.to_tensor(mi.build(mi.example_spec()), 1)[None], drill])
    x_rms = float(maps.square().mean().sqrt())
    for features in (32, 64):
        for seed in range(3):
            torch.manual_seed(seed)
            m = mc.MapCNN(features)
            with torch.no_grad():
                st = m.steps(maps)
            for k, f in enumerate(st):
                rms = float(f.square().mean().sqrt())
                assert 0.2 * x_rms < rms < 10 * x_rms, (features, seed, mp.STEP_NAMES[k], rms, x_rms)
            d = mp.dead_all(m, maps)
            assert max(d.values()) <= 0.25, (features, seed, d)


def test_side_2_of_the_turned_battle_is_side_1():
    """A battle turned half round the map's centre with the sides swapped: side 2 there gets side 1's layers
    and features exactly (map_input / map_batch turn the map for side 2)."""
    m = model(4)
    desc = mc._drill_desc(5)
    z = mb.zones_of(desc)

    def turn(r):
        cx, cz, ux, uz, hf, hl = r
        return (-cx, -cz, -ux, -uz, hf, hl)
    turned = {1: turn(z[2]), 2: turn(z[1])}
    lay = mb.layers(mb.rects_tensor([z, turned], dtype=F64), [1, 2])
    assert torch.equal(lay[0], lay[1])
    assert lay[0, 2].sum() > 0 and not torch.equal(lay[0, 2], lay[0, 3])
    f = m(lay)
    assert torch.equal(f[0], f[1])
    a1, a2 = arena_layers(1), arena_layers(2)
    assert torch.equal(a2, a1.flip(-1, -2)[:, [0, 1, 3, 2, 5, 4]])


def test_indexed_and_the_cache():
    m = model(5, torch.float32)
    maps = torch.cat([arena_layers(1, torch.float32), arena_layers(2, torch.float32)])
    idx = torch.tensor([0, 1, 1, 0, 0])
    tok = m.indexed(maps, idx)
    assert torch.equal(tok[2], m(maps[1:2])[0])
    data = {7: mi.build(mi.arena_spec())}
    cache = mc.SummaryCache(m)
    keys = [(7, 1), (7, 2), (7, 1)]
    got = cache.get(keys, mc.map_layers(data))
    assert cache.computed == 2 and len(cache) == 2
    assert torch.allclose(got, tok[[0, 1, 0]], atol=1e-6)
    cache.get([(7, 2)], mc.map_layers(data))
    assert cache.computed == 2                                # held: not computed again
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m(maps).mean().backward()
    opt.step()
    new = cache.get([(7, 1)], mc.map_layers(data))           # the weights changed: computed again
    assert cache.computed == 3
    assert torch.allclose(new[0], m(maps[:1])[0], atol=1e-6)
