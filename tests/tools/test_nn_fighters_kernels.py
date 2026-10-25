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
    # each projectile its own number of points (lab.json missile.path_full), up to n_pts
    npts = torch.randint(1, 15, (B, P), generator=g).to(dev)
    args = (table, 12, 14, x, z, hit, uid, radius, height, lx, lz, ex, ez, tan, sh, ok)
    want = K.path_first(*args, use_triton=False, npts=npts)
    got = K.path_first(*args, use_triton=True, npts=npts)
    assert int((want >= 0).sum()) > 50
    assert torch.equal(got, want)
    # the shooter's own unit passed (lab.json missile.own_pass)
    want = K.path_first(*args, use_triton=False, npts=npts, own_pass=True)
    got = K.path_first(*args, use_triton=True, npts=npts, own_pass=True)
    assert torch.equal(got, want)
    hit_own = (want >= 0) & (uid.gather(1, want.clamp(min=0)) == uid.gather(1, sh))
    assert not bool(hit_own.any())


def _path_after_case(dev):
    """The path_after kernel's case (25.10.2026, a projectile through several men): torch and Triton on device dev."""
    g, x, z, valid, hit, uid, radius, height = _men(seed=4)
    B, M = x.shape
    P = 200
    lx = torch.rand((B, P), generator=g) * 50
    lz = torch.rand((B, P), generator=g) * 50
    a = torch.rand((B, P), generator=g) * 6.283
    ex, ez = torch.cos(a), torch.sin(a)
    tan = 0.02 + torch.rand((B, P), generator=g) * 0.3
    sh = torch.randint(0, M, (B, P), generator=g)
    ok = torch.rand((B, P), generator=g) < 0.8
    to = lambda *t: [v.to(dev) for v in t]
    x, z, valid, hit, uid, radius, height, lx, lz, ex, ez, tan, sh, ok = to(x, z, valid, hit, uid, radius, height,
                                                                          lx, lz, ex, ez, tan, sh, ok)
    table = cells.Table(x, z, valid, 3.6, 4096)
    npts = torch.randint(1, 15, (B, P), generator=g).to(dev)
    args = (table, 12, 14, x, z, hit, uid, radius, height, lx, lz, ex, ez, tan, sh, ok)
    first = K.path_first(*args, use_triton=False, npts=npts, own_pass=True)
    fc = first.clamp(min=0)
    bmax = -((x.gather(1, fc) - lx) * ex + (z.gather(1, fc) - lz) * ez) - 1e-4
    ok2 = ok & (first >= 0)
    args2 = args[:-1] + (ok2,)
    want = K.path_first(*args2, use_triton=False, npts=npts, own_pass=True, bmax=bmax)
    got = K.path_first(*args2, use_triton=True, npts=npts, own_pass=True, bmax=bmax)
    return first, want, got


@cuda
def test_path_after_triton_matches_torch():
    first, want, got = _path_after_case("cuda")
    assert int((want >= 0).sum()) > 20 and not bool(((want == first) & (want >= 0)).any())   # the next man, not the first
    assert torch.equal(got, want)


def test_a_projectile_passes_the_men_of_its_own_unit():
    # lab.json missile.own_pass: a flat bullet over its own front man (same unit) goes on to the enemy behind him
    x, z = torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[-2.0, 0.0, 20.0]])      # shooter, his front man, the enemy
    valid = torch.tensor([[True, True, True]])
    uid = torch.tensor([[0, 0, 1]])
    table = cells.Table(x, z, valid, 3.6, 4096)
    one = lambda v: torch.tensor([[v]])
    args = (table, 12, 48, x, z, valid, uid, torch.tensor([[0.3, 0.3]]), torch.tensor([[1.8, 1.8]]), one(0.0),
            one(40.0), one(0.0), one(1.0), one(0.03), torch.tensor([[0]]), torch.tensor([[True]]))
    assert int(K.path_first(*args, use_triton=False)) == 1
    assert int(K.path_first(*args, use_triton=False, own_pass=True)) == 2


def test_the_whole_low_path_finds_a_man_far_behind_the_landing_point():
    # a flat bullet (tan 0.05: a man 1.8 m tall is under it for the last 36 m) lands 30 m past a lone man: he is
    # on its low path; 6 points (21.6 m) miss him, the projectile's own 11 points find him (lab.json path_full)
    x, z = torch.tensor([[0.0]]), torch.tensor([[0.0]])
    valid = torch.tensor([[True]])
    uid = torch.zeros((1, 1), dtype=torch.long)
    table = cells.Table(x, z, valid, 3.6, 4096)
    one = lambda v: torch.tensor([[v]])
    args = (table, 12, 6, x, z, valid, uid, torch.tensor([[0.65]]), torch.tensor([[1.8]]), one(0.0), one(30.0),
            one(0.0), one(1.0), one(0.05), torch.tensor([[-1]]), torch.tensor([[True]]))
    assert int(K.path_first(*args, use_triton=False)) == -1
    args = (table, 12, 48) + args[3:]
    assert int(K.path_first(*args, use_triton=False, npts=torch.tensor([[11]]))) == 0
    assert int(K.path_first(*args, use_triton=False, npts=torch.tensor([[6]]))) == -1


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
    # move.stop_own's scan (block_scan): men stopped by a man of another unit of their side in contact ahead
    ang = torch.rand((B, M), generator=torch.Generator().manual_seed(5)).to("cuda") * 6.283
    ex, ez = torch.sin(ang), torch.cos(ang)
    side_m = torch.ones((B, N), dtype=side.dtype, device="cuda")                # one side: many own of other units
    fight = torch.rand((B, M), generator=torch.Generator().manual_seed(6)).to("cuda") < 0.3
    prio = torch.where(torch.rand((B, M), generator=torch.Generator().manual_seed(7)).to("cuda") < 0.5,
                       torch.rand((B, M), generator=torch.Generator().manual_seed(8)).to("cuda") * 10, -1e9)
    c_me = torch.rand((B, M), generator=torch.Generator().manual_seed(9)).to("cuda") * 10
    base = up & ~fight
    a = SC.block_scan(near, base, uid, ex, ez, c_me, side_m, radius, valid, fight, prio, 1.95, use_triton=False)
    b = SC.block_scan(near, base, uid, ex, ez, c_me, side_m, radius, valid, fight, prio, 1.95, use_triton=True)
    assert int(a.sum()) > 50 and torch.equal(a, b)


@cuda
def test_unit_shape_by_the_contact_kernel_matches_torch():
    # shape.unit_features' share of men in reach of an enemy: the contact kernel (no neighbour lists) = torch's lists
    from tools.nn.fighters import shape
    g, x, z, valid, hit, uid, radius, height = _men(M=900)
    x, z = x * 0.5, z * 0.5                                     # dense: many men in reach
    B, M = x.shape
    N = radius.shape[1]
    side_m = torch.where(uid < N // 2, 1, 2)
    bearing = torch.rand((B, N), generator=g) * 360
    men0 = torch.full((B, N), 120.0)
    hp = torch.rand((B, N), generator=g)
    args = [t.to("cuda") for t in (x, z, uid, valid, side_m, bearing, men0, hp)]
    want = shape.unit_features(*args, N, use_triton=False)
    got = shape.unit_features(*args, N, use_triton=True)
    i = shape.FEATURES.index("contact")
    assert float(want[..., i].sum()) > 0.5                       # men in reach of an enemy
    diff = (got - want).abs().amax((0, 1))
    # (the other features: the same sums, atomics' order on the GPU - float noise)
    assert float(diff[i]) < 1e-6 and torch.allclose(got, want, rtol=1e-4, atol=1e-5), diff.tolist()
