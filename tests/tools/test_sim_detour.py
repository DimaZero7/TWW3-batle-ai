"""The map's part of the way round 'cannot stand' ground (tools/nn/sim/detour.py; build/diag_map/pathing.md): the
clearance is the exact distance to the nearest closed node (the grid's edge closed), the level traced into loops with
the closed side on the left, the corners of a grown block on its convex corners, the graph's ways the shortest, and a
way round a block as long as the geometry says. numpy alone (the host)."""
import math

import numpy as np

from tools.nn.sim import detour as D


def test_clearance_is_the_exact_distance_to_the_nearest_closed_node_and_the_edge():
    rng = np.random.default_rng(3)
    b = (rng.random((23, 31)) < 0.04).astype(np.uint8)
    F = D.clearance(b, 2.0, fmax=200.0)
    R, C = b.shape
    pts = [(j, i) for j in range(-1, R + 1) for i in range(-1, C + 1)
           if j in (-1, R) or i in (-1, C) or b[j, i]]                 # closed nodes and the ring outside the grid
    pj, pi = np.array(pts).T
    for j in range(R):
        for i in range(C):
            want = 2.0 * np.sqrt(((pj - j) ** 2 + (pi - i) ** 2).min())
            assert abs(F[j, i] - want) < 1e-4, (j, i)
    assert (F[b > 0] == 0).all()
    assert D.clearance(b, 2.0, fmax=5.0).max() == 5.0                 # capped


def _square(n=41, lo=15, hi=25):
    b = np.zeros((n, n), np.uint8)
    b[lo:hi, lo:hi] = 1
    return b


def test_the_loops_keep_the_closed_side_on_the_left():
    b = _square()
    F = D.clearance(b, 1.0)
    loops = D.contours(F.astype(np.float64) - 3.0, 0.0, 0.0, 1.0)
    assert len(loops) == 2                                             # round the block, inside the map's edge
    for loop in loops:
        p, q = loop, np.roll(loop, -1, 0)
        mid = (p + q) / 2
        d = q - p
        L = np.hypot(d[:, 0], d[:, 1])[:, None].clip(min=1e-9)
        left = mid + np.stack([-d[:, 1], d[:, 0]], 1) / L * 0.4      # a little to the left of each step
        right = mid - np.stack([-d[:, 1], d[:, 0]], 1) / L * 0.4
        fl = D.at(F, 0.0, 0.0, 1.0, left[:, 0], left[:, 1])
        fr = D.at(F, 0.0, 0.0, 1.0, right[:, 0], right[:, 1])
        assert (fl <= fr + 1e-6).mean() > 0.95


def test_a_blocks_corners_stand_off_its_convex_corners_at_the_growth():
    b = _square()                       # nodes 15..24: the block's corner nodes (15, 15) (15, 24) (24, 15) (24, 24)
    F = D.clearance(b, 1.0)
    r = 4.0
    c = D.corners(F, 0.0, 0.0, 1.0, r, contour_m=1.0)
    assert len(c) >= 4
    assert (D.at(F, 0.0, 0.0, 1.0, c[:, 0], c[:, 1]) >= r).all()
    for cx, cz in ((15, 15), (15, 24), (24, 15), (24, 24)):
        d = np.hypot(c[:, 0] - cx, c[:, 1] - cz)
        assert d.min() < r + 2.0, (cx, cz)                          # a corner beside each of the block's corners
    # (none at the map's edge: its corners are concave from the field)
    assert (np.hypot(c[:, 0] - 0, c[:, 1] - 0) > 10).all() and (np.hypot(c[:, 0] - 40, c[:, 1] - 40) > 10).all()


def test_the_way_round_a_wall_is_the_geometrys_shortest():
    # a wall x 40..99 at z 48..51 (closed), a unit at (60, 20) to (60, 80): round its nearer west end (x 40), growth r
    n = 121
    b = np.zeros((n, n), np.uint8)
    b[48:52, 40:100] = 1
    F = D.clearance(b, 1.0, fmax=64.0)
    r = 6.0
    c = D.corners(F, 0.0, 0.0, 1.0, r)
    Dm, NX = D.graph(F, 0.0, 0.0, 1.0, c, r)
    assert np.allclose(Dm, Dm.T) and (np.diag(Dm) == 0).all()
    s, g = (60.0, 20.0), (60.0, 80.0)
    way = D.route_np(F, 0.0, 0.0, 1.0, c, Dm, NX, s, g, r)
    assert way is not None and len(way) >= 2
    pts = np.concatenate([[s], way, [g]])
    L = np.hypot(*np.diff(pts, axis=0).T).sum()
    # the taut string round the grown end: by the points r west of the wall's end corners (40, 48) / (40, 51) - the
    # arcs round them make the true one a little shorter
    ideal = math.hypot(60 - (40 - r), 48 - 20) + 3 + math.hypot(60 - (40 - r), 80 - 51)
    assert 0.97 * ideal < L < 1.04 * ideal
    assert (pts[1:-1, 0] < 40).all()                                   # round the open (west) end
    # every leg keeps clear of r - tol_m - sample_m / 2 (the samples 2 m apart, the clearance 1-Lipschitz)
    for a, bb in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, 200)[:, None]
        q = a + (bb - a) * t
        assert (D.at(F, 0.0, 0.0, 1.0, q[:, 0], q[:, 1]) >= r - 1.5).all()
    # a clear line: no corner
    assert len(D.route_np(F, 0.0, 0.0, 1.0, c, Dm, NX, (20.0, 20.0), (25.0, 30.0), r)) == 0
    # a wall across the whole map: no way
    b2 = np.zeros((n, n), np.uint8)
    b2[48:52, :] = 1
    F2 = D.clearance(b2, 1.0)
    c2 = D.corners(F2, 0.0, 0.0, 1.0, r)
    D2, NX2 = D.graph(F2, 0.0, 0.0, 1.0, c2, r)
    assert D.route_np(F2, 0.0, 0.0, 1.0, c2, D2, NX2, s, g, r) is None


def test_a_wider_growth_takes_a_longer_way():
    n = 121
    b = np.zeros((n, n), np.uint8)
    b[48:52, 40:100] = 1
    F = D.clearance(b, 1.0)
    out = []
    for r in (4.0, 8.0, 12.0):
        c = D.corners(F, 0.0, 0.0, 1.0, r)
        Dm, NX = D.graph(F, 0.0, 0.0, 1.0, c, r)
        way = D.route_np(F, 0.0, 0.0, 1.0, c, Dm, NX, (70.0, 20.0), (70.0, 80.0), r)
        pts = np.concatenate([[(70.0, 20.0)], way, [(70.0, 80.0)]])
        out.append(np.hypot(*np.diff(pts, axis=0).T).sum())
    assert out[0] < out[1] < out[2]
