"""tools/nn/general/deploy.py: the units' layout inside their groups' places and the deployment zone (S28 skeleton,
docs/en/training/general.md "Deployment"). Plain numpy: runs on the host."""
import math

import numpy as np
import pytest

from tools.nn.general import deploy as d

RULE = d.ZoneRule(gap_m=350.0, deployment_m=300.0)


def army(rng, n):
    return [{"slot": int(i), "group": int(rng.integers(0, 4)), "width": float(rng.uniform(1, 60)),
             "depth": float(rng.uniform(1, 30))} for i in range(n)]


def zone_point(rng, side, edge=False):
    """(x, z) of a point of the side's zone (edge: on its corner)."""
    if edge:
        f, lat = RULE.back_m, RULE.half_m * (1 if rng.random() < 0.5 else -1)
    else:
        f, lat = rng.uniform(RULE.back_m, RULE.front_m), rng.uniform(-RULE.half_m, RULE.half_m)
    return d.from_zone(f, lat, side, RULE.gap_m)


def test_the_zone_frame_goes_both_ways_and_mirrors_the_sides():
    for side in (1, 2):
        for f, lat in ((0.0, 0.0), (-200.0, 120.0), (40.0, -150.0)):
            x, z = d.from_zone(f, lat, side, RULE.gap_m)
            assert d.to_zone(x, z, side, RULE.gap_m) == pytest.approx((f, lat))
    # side 1 west facing east, side 2 east facing west: the same zone point mirrored through the centre
    assert d.from_zone(-100.0, 30.0, 1, 350.0) == pytest.approx((-275.0, -30.0))
    assert d.from_zone(-100.0, 30.0, 2, 350.0) == pytest.approx((275.0, 30.0))
    assert RULE.back_m == -260.0 and RULE.half_m == 150.0                    # place.py: 40 m ahead, 300 m deep
    assert RULE.inside(-175.0, 0.0, 1) and not RULE.inside(175.0, 0.0, 1) and RULE.inside(175.0, 0.0, 2)
    assert not RULE.inside(-175.0 + 41.0, 0.0, 1) and RULE.inside(-175.0 + 41.0, 0.0, 1, slack=2.0)


@pytest.mark.parametrize("side", (1, 2))
def test_the_layout_stays_in_the_zone(side):
    """Random armies of 1-20 units, their groups on random tiles of the zone (also its corners, all on one tile): every
    unit's footprint stays inside the zone."""
    rng = np.random.default_rng(side)
    enemy = (175.0, 0.0) if side == 1 else (-175.0, 0.0)
    for trial in range(200):
        units = army(rng, int(rng.integers(1, 21)))
        one = trial % 3 == 0
        p = zone_point(rng, side, edge=trial % 5 == 0)
        groups = {g: (p if one else zone_point(rng, side, edge=trial % 7 == 0)) for g in range(4)}
        pts = d.layout(units, groups, side, enemy, RULE)
        assert set(pts) == {u["slot"] for u in units}
        assert d.inside(pts, units, side, RULE), (trial, pts)


def test_a_group_stands_in_a_line_on_its_tile_facing_the_enemy():
    units = [{"slot": i, "group": 0, "width": 30.0, "depth": 10.0} for i in range(3)]
    tile = d.from_zone(-100.0, 0.0, 1, RULE.gap_m)
    enemy = (175.0, 0.0)
    pts = d.layout(units, {0: tile}, 1, enemy, RULE)
    xs = [pts[i][0] for i in range(3)]
    zs = [pts[i][1] for i in range(3)]
    assert max(xs) - min(xs) == pytest.approx(0.0, abs=1e-6)                 # one line across the facing
    assert zs[1] == pytest.approx(tile[1])                                    # centred on the tile, slot order
    assert abs(zs[0] - zs[1]) == pytest.approx(30.0 + d.LATERAL_GAP_M)
    for x, z, b in pts.values():
        fx, fz = math.sin(math.radians(b)), math.cos(math.radians(b))
        ex, ez = enemy[0] - tile[0], enemy[1] - tile[1]
        assert (fx * ex + fz * ez) / math.hypot(ex, ez) == pytest.approx(1.0, abs=1e-6)   # faces the enemy's centre


def test_two_groups_on_one_tile_stand_one_behind_the_other():
    units = [{"slot": 0, "group": 0, "width": 30.0, "depth": 10.0}, {"slot": 1, "group": 1, "width": 30.0, "depth": 10.0}]
    tile = d.from_zone(-100.0, 0.0, 1, RULE.gap_m)
    pts = d.layout(units, {0: tile, 1: tile}, 1, (175.0, 0.0), RULE)
    f0, _ = d.to_zone(pts[0][0], pts[0][1], 1, RULE.gap_m)
    f1, _ = d.to_zone(pts[1][0], pts[1][1], 1, RULE.gap_m)
    assert f0 - f1 == pytest.approx(10.0 + d.ROW_GAP_M)                       # the second a row behind


def test_a_line_wider_than_the_zone_goes_on_in_the_next_row():
    units = [{"slot": i, "group": 0, "width": 60.0, "depth": 8.0} for i in range(8)]   # 8 x 66 m > 290 m
    tile = d.from_zone(-50.0, 0.0, 1, RULE.gap_m)
    pts = d.layout(units, {0: tile}, 1, (175.0, 0.0), RULE)
    rows = {round(d.to_zone(x, z, 1, RULE.gap_m)[0], 3) for x, z, _ in pts.values()}
    assert len(rows) == 2 and d.inside(pts, units, 1, RULE)


def test_units_without_a_group_are_not_laid_out():
    units = [{"slot": 0, "group": -1, "width": 30.0, "depth": 10.0}, {"slot": 1, "group": 0, "width": 30.0, "depth": 10.0}]
    pts = d.layout(units, {0: d.from_zone(-100.0, 0.0, 1, RULE.gap_m)}, 1, (175.0, 0.0), RULE)
    assert set(pts) == {1}
    x, z, b, laid = d.as_arrays(pts, 4)
    assert laid.tolist() == [False, True, False, False]


def test_a_formations_depth_is_its_ranks():
    assert d.depth_of(120, 30.0, 1.0, 1.5) == pytest.approx(4 * 1.5)        # 30 files, 4 ranks
    assert d.depth_of(1, 1.0, 1.0, 1.5) == pytest.approx(1.5)               # a lone man
