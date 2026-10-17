"""The fighters lab's GPU kernels (tools/nn/fighters/kernels.py) give the plain torch code's numbers. Needs torch and
CUDA with Triton: run in the container on the GPU."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.fighters import cells, kernels as K  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available() or K.triton is None, reason="CUDA + Triton")


def _men(B=3, M=600, N=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((B, M), generator=g) * 50
    z = torch.rand((B, M), generator=g) * 50
    valid = torch.rand((B, M), generator=g) < 0.9
    hit = valid & (torch.rand((B, M), generator=g) < 0.9)
    uid = torch.randint(0, N, (B, M), generator=g)
    radius = 0.3 + 0.6 * torch.rand((B, N), generator=g)
    height = 1.5 + 1.5 * torch.rand((B, N), generator=g)
    return g, x, z, valid, hit, uid, radius, height


@cuda
def test_path_first_triton_matches_torch():
    g, x, z, valid, hit, uid, radius, height = _men()
    B, M = x.shape
    P = 200
    lx = torch.rand((B, P), generator=g) * 50
    lz = torch.rand((B, P), generator=g) * 50
    a = torch.rand((B, P), generator=g) * 6.283
    ex, ez = torch.cos(a), torch.sin(a)
    tan = 0.02 + torch.rand((B, P), generator=g) * 0.8
    sh = torch.randint(0, M, (B, P), generator=g)
    ok = torch.rand((B, P), generator=g) < 0.7
    dev = "cuda"
    to = lambda *t: [v.to(dev) for v in t]
    x, z, valid, hit, uid, radius, height, lx, lz, ex, ez, tan, sh, ok = to(x, z, valid, hit, uid, radius, height,
                                                                          lx, lz, ex, ez, tan, sh, ok)
    table = cells.Table(x, z, valid, 3.6, 4096)
    args = (table, 12, 6, x, z, hit, uid, radius, height, lx, lz, ex, ez, tan, sh, ok)
    want = K.path_first(*args, use_triton=False)
    got = K.path_first(*args, use_triton=True)
    assert int((want >= 0).sum()) > 50                          # the test hits men
    assert torch.equal(got, want)


def _near(x, z, valid, uid):
    from tools.nn.fighters import hook
    n = hook.Near.__new__(hook.Near)
    n.px, n.pz, n.cap, n.uid, n._cand = x, z, 12, uid, None
    n.table = cells.Table(x, z, valid, 3.6, 4096, cap=12)
    return n


@cuda
def test_neighbour_scans_triton_match_torch():
    from tools.nn.fighters import scans as SC
    g, x, z, valid, hit, uid, radius, height = _men(M=900)
    x, z = x * 0.5, z * 0.5                                     # dense: many men in reach
    B, M = x.shape
    N = radius.shape[1]
    side = torch.where(torch.arange(N) < N // 2, 1, 2).expand(B, N).contiguous()
    permit = torch.rand((B, N, N), generator=g) < 0.8
    tgt = torch.where(torch.rand((B, M), generator=g) < 0.5, torch.randint(0, M, (B, M), generator=g), -1)
    pref = torch.where(torch.rand((B, M), generator=g) < 0.5, torch.randint(0, N, (B, M), generator=g), -1)
    up = hit
    ax, az = x + 0.3 * torch.rand((B, M), generator=g), z - 0.3 * torch.rand((B, M), generator=g)
    rad_m = radius.gather(1, uid)
    t = [v.to("cuda") for v in (x, z, valid, uid, side, radius, permit, tgt, pref, up, ax, az, rad_m)]
    x, z, valid, uid, side, radius, permit, tgt, pref, up, ax, az, rad_m = t
    near = _near(x, z, valid, uid)
    a = SC.contact_scan(near, up, uid, side, radius, 2.5, use_triton=False)
    b = SC.contact_scan(near, up, uid, side, radius, 2.5, use_triton=True)
    assert int((a[1] >= 0).sum()) > 100
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    args = (near, up, valid, uid, side, radius, permit, tgt, pref, 2.5, 2.0)
    a = SC.melee_scan(*args, use_triton=False)
    b = SC.melee_scan(*args, use_triton=True)
    has = a[1]
    assert int(has.sum()) > 100 and int(a[0].sum()) > 10
    for k in (0, 1, 3):
        assert torch.equal(a[k], b[k])
    assert torch.equal(a[2][has], b[2][has]) and torch.equal(a[4][a[3]], b[4][b[3]])
    a = SC.push_scan(near, ax, az, up, rad_m, use_triton=False)
    b = SC.push_scan(near, ax, az, up, rad_m, use_triton=True)
    assert float(a[0].abs().sum()) > 1.0
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
