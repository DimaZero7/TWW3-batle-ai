"""tools/nn/general/map_input.py and geometry.py: the general's map layers (design/general/index.html, "Ландшафт")."""
import math
import re

import numpy as np
import pytest

from tools.nn import scenario as nn_scenario
from tools.nn.general import geometry as geo
from tools.nn.general import map_input as mi


def samples(grid, iz, ix, step=0.5):
    """Points every `step` m over the cell (its edges included)."""
    xa, xb, za, zb = (np.asarray(g)[iz, ix] for g in grid)
    xs = np.arange(xa, xb + 1e-9, step)
    zs = np.arange(za, zb + 1e-9, step)
    return np.meshgrid(xs, zs)


def poly_distance(poly, x, z):
    """Distance from points to the polygon (0 inside), brute force."""
    a, b = np.asarray(poly), np.roll(np.asarray(poly), -1, axis=0)
    d = geo._point_seg_dist(x[..., None], z[..., None], a[:, 0], a[:, 1], b[:, 0], b[:, 1]).min(-1)
    return np.where(geo.inside(poly, x, z), 0.0, d)


@pytest.fixture(scope="module")
def example():
    spec = mi.example_spec()
    return spec, mi.build(spec)


@pytest.fixture(scope="module")
def arena():
    spec = mi.arena_spec()
    return spec, mi.build(spec)


class TestGrid:
    def test_sizes(self, arena):
        _, data = arena
        assert data["cells"].shape == (3, 128, 128) and data["cells"].dtype == np.uint8
        lay = mi.unpack(data, 1)
        assert lay.shape == (6, 128, 128) and lay.dtype == np.float32
        assert mi.SIZE_M == 2048.0

    def test_a_1024_map_fills_the_middle_64_by_64(self):
        spec = mi.MapSpec(name="t", field_box=(1000.0, 2024.0, 3000.0, 4024.0),
                          zones={1: geo.rect(1300, 3512, 200, 200), 2: geo.rect(1724, 3512, 200, 200)})
        assert mi.origin(spec) == (1512.0 - 1024.0, 3512.0 - 1024.0)
        f = mi.unpack(mi.build(spec))[0]
        inner = np.zeros((128, 128), bool)
        inner[32:96, 32:96] = True
        assert (f[inner] == 1).all() and (f[~inner] == 0).all()

    def test_the_map_stands_in_the_middle_with_outside_cells_on_every_side(self, example):
        _, data = example
        f = mi.unpack(data)[0]
        assert np.array_equal(f, f[::-1, :]) and np.array_equal(f, f[:, ::-1])
        assert (f[0] == 0).all() and (f[-1] == 0).all() and (f[:, 0] == 0).all() and (f[:, -1] == 0).all()

    def test_cell_of(self, arena):
        _, data = arena
        org = data["meta"]["origin"]
        assert org == [-1024.0, -1024.0]
        assert mi.cell_of(org, -1024.0, -1024.0) == (0, 0) and mi.cell_of(org, 1023.9, 1023.9) == (127, 127)


class TestShares:
    def test_an_edge_cutting_cells_gives_their_share(self, example):
        """1000 x 600 m field centred in the grid: the x edge at -500 m cuts cells x -512..-496 (0.25 inside),
        the z edge at 300 m cuts cells z 288..304 (0.75 inside); the corner cell 0.25 x 0.75."""
        spec, data = example
        g = mi.grid(spec)
        exact = mi.field_share(spec, g)
        iz, ix = mi.cell_of(data["meta"]["origin"], 0.0, 0.0)[0], mi.cell_of(data["meta"]["origin"], -500.0, 0.0)[1]
        assert ix == 32 and exact[iz, ix] == pytest.approx(0.25)
        izc = mi.cell_of(data["meta"]["origin"], 0.0, 299.0)[0]
        assert exact[izc, 64] == pytest.approx(0.75) and exact[izc, ix] == pytest.approx(0.1875)
        assert exact.sum() * 256.0 == pytest.approx(1000.0 * 600.0)
        f = mi.unpack(data)[0]
        assert np.abs(f - exact).max() <= 0.5 / 255 + 1e-6

    def test_polygon_share_is_exact(self):
        """A turned rounded rectangle: the clipped share against 0.5 m sampling, and the whole area."""
        poly = geo.rounded_rect(13.0, -7.0, 300.0, 170.0, 40.0, math.radians(23))
        g = geo.cells(-256.0, -256.0, 32, 16.0)
        s = geo.share(poly, g)
        assert s.sum() * 256.0 == pytest.approx(geo.area(poly), rel=1e-9)
        for iz, ix in list(zip(*np.nonzero((s > 0) & (s < 1))))[::7]:
            x, z = samples(g, iz, ix, 0.25)
            x, z = x[:-1, :-1] + 0.125, z[:-1, :-1] + 0.125      # cell centres of a 0.25 m split
            assert geo.inside(poly, x, z).mean() == pytest.approx(s[iz, ix], abs=0.01)

    def test_a_rectangle_polygon_matches_the_box_formula(self):
        g = geo.cells(-100.0, -100.0, 16, 16.0)
        box = (-37.0, 61.5, -90.25, 13.0)
        poly = np.array([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]])
        assert np.allclose(geo.share(poly, g), geo.share_box(box, g))

    def test_shares_sum_to_one(self, example, arena):
        for _, data in (example, arena):
            assert (data["cells"][:2].astype(int).sum(0) == 255).all()
            assert np.allclose(mi.unpack(data)[:2].sum(0), 1.0, atol=1e-6)

    def test_quantise_keeps_the_sum_for_more_types(self):
        rng = np.random.default_rng(0)
        s = rng.dirichlet([0.5, 0.5, 0.5], size=(20, 20)).transpose(2, 0, 1)
        q = mi.quantise(s)
        assert (q.astype(int).sum(0) == 255).all()
        assert np.abs(q / 255.0 - s).max() < 1.0 / 255


class TestZones:
    def test_wholly_inside_on_a_rounded_corner(self, example):
        """A cell counts only when every point of it is in the zone: the rounded corner leaves out cells whose
        centre is in the zone but a corner sticks out."""
        spec, _ = example
        g = mi.grid(spec)
        zone = spec.zones[1]
        flag = geo.wholly_inside(zone, g)
        s = geo.share(zone, g)
        assert np.array_equal(flag, s >= 1 - 1e-9)
        cx, cz = (g[0] + g[1]) / 2, (g[2] + g[3]) / 2
        centre_in = geo.inside(zone, cx, cz)
        cut = centre_in & ~flag
        assert cut.any()
        for iz, ix in zip(*np.nonzero(cut)):
            x, z = samples(g, iz, ix)
            assert not geo.inside(zone, x, z).all()
        for iz, ix in list(zip(*np.nonzero(flag)))[::11]:
            x, z = samples(g, iz, ix)
            assert (poly_distance(zone, x, z) == 0).all()

    def test_the_vanguard_flag_is_clear_of_the_band(self, example):
        spec, data = example
        g = mi.grid(spec)
        lay = mi.unpack(data, 1)
        enemy = spec.zones[2]
        field_full = mi.field_share(spec, g) >= 1 - 1e-9
        brute = np.zeros((128, 128))
        for iz, ix in zip(*np.nonzero(field_full)):
            x, z = samples(g, iz, ix, 1.0)
            brute[iz, ix] = poly_distance(enemy, x, z).min()
        yes = lay[4] > 0.5
        assert (yes <= field_full).all()
        sure_clear = field_full & (brute >= spec.band_m)
        sure_in = field_full & (brute < spec.band_m - 1.0)      # 1 m sampling: the true minimum is up to 0.71 m less
        assert yes[sure_clear].all() and not yes[sure_in].any()

    def test_the_zones_do_not_overlap_and_lie_in_their_own_vanguard_area(self, arena, example):
        for _, data in (arena, example):
            lay = mi.unpack(data, 1)
            assert not (lay[2] * lay[3]).any()
            assert (lay[4][lay[2] > 0] == 1).all() and (lay[4][lay[3] > 0] == 0).all()


class TestSides:
    def test_side_two_sees_the_map_half_turned_with_the_zones_swapped(self, example):
        _, data = example
        a, b = mi.unpack(data, 1), mi.half_turn(mi.unpack(data, 2))
        assert np.array_equal(a[:2], b[:2])
        assert np.array_equal(a[2], b[3]) and np.array_equal(a[3], b[2])
        assert np.array_equal(a[4], b[5]) and np.array_equal(a[5], b[4])

    def test_a_half_turned_battle_seen_by_side_2_is_the_same_as_by_side_1(self, example):
        """Turn the world half round the map's centre and swap the sides: side 2 of the turned map gets side 1's
        layers of the first (as the network's side frame, tools/nn/model/frame.py)."""
        spec, data = example
        turned = mi.MapSpec(name="turned", field_box=spec.field_box, centre=spec.centre,
                            zones={1: -spec.zones[2], 2: -spec.zones[1]}, band_m=spec.band_m)
        assert np.array_equal(mi.unpack(mi.build(turned), 2), mi.unpack(data, 1))

    def test_the_arena_looks_the_same_from_both_sides(self, arena):
        """The arena's zones are symmetric about the map's centre: both sides get the same layers."""
        _, data = arena
        assert np.array_equal(mi.unpack(data, 2), mi.unpack(data, 1))


class TestArena:
    def test_the_zones_are_those_of_the_battle_file(self):
        arena = nn_scenario.load_arena()
        xml = nn_scenario.scenario_xml(arena, "enemy")
        areas = re.findall(r'<centre x="([-\d.]+)" y="([-\d.]+)"/><width metres="([\d.]+)"/><height metres="([\d.]+)"/>',
                           xml)
        zones = mi.arena_zones(arena)
        for side, (x, z, w, h) in zip((1, 2), areas):
            p = zones[side]
            assert p[:, 0].mean() == pytest.approx(float(x)) and p[:, 1].mean() == pytest.approx(float(z))
            assert np.ptp(p[:, 0]) == pytest.approx(float(w)) and np.ptp(p[:, 1]) == pytest.approx(float(h))

    def test_the_arena_layers(self, arena):
        """Field +-1020 m in the grid's +-1024: the outer ring of cells 0.75 field (corners 0.5625), the rest 1.
        Zones x -435..-135 / 135..435, z -150..150 -> 18 x 18 whole cells."""
        _, data = arena
        lay = mi.unpack(data, 1)
        f = lay[0]
        assert (f[1:-1, 1:-1] == 1).all() and (lay[1][1:-1, 1:-1] == 0).all()
        assert np.allclose(f[0, 1:-1], 0.75, atol=0.5 / 255) and np.allclose(f[1:-1, -1], 0.75, atol=0.5 / 255)
        assert f[0, 0] == pytest.approx(0.5625, abs=0.5 / 255)
        assert lay[2].sum() == 18 * 18 == lay[3].sum()
        assert lay[2][:, : 128 // 2].sum() == 18 * 18
        assert (lay[4][0] == 0).all() and (lay[4][1:-1, 1:-1].sum() > 0)
        assert data["meta"]["band_m"] == mi.BAND_M

    def test_save_and_load(self, arena, tmp_path):
        spec, data = arena
        path = mi.save(data, tmp_path / "a.npz")
        back = mi.load(path)
        assert np.array_equal(back["cells"], data["cells"]) and back["meta"]["name"] == spec.name
        assert np.array_equal(mi.unpack(back, 2), mi.unpack(data, 2))
        assert path.stat().st_size < 3 * 128 * 128 + 2000

    def test_to_tensor(self, arena):
        torch = pytest.importorskip("torch")
        _, data = arena
        t = mi.to_tensor(data, 2)
        assert t.shape == (6, 128, 128) and t.dtype == torch.float32
