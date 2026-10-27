"""The zone map (tools/nn/unit/zmap.py): the frame turned front-up, density 0..1 by the share of a cell covered, the
fronts' direction layers, the map's edge, my area, the cell size. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

L = Z.L


def t(*v):
    return torch.tensor(v, dtype=torch.float32)


def one(x, z, b, front, depth, fill=1.0, role=Z.ENEMY):
    """A single unit for one map: [1, 1] tensors."""
    return {"x": t(x)[None], "z": t(z)[None], "b": t(b)[None], "front": t(front)[None], "depth": t(depth)[None],
            "fill": t(fill)[None], "role": torch.tensor([[role]])}


def frame(heading=0.0, cell=8.0, cx=0.0, cz=0.0):
    return t(cx), t(cz), t(heading), t(cell)


def centroid(layer):
    """(row, col) of a layer's mass."""
    C = layer.shape[-1]
    r = torch.arange(C, dtype=torch.float32)
    w = layer.sum()
    return float((layer.sum(1) * r).sum() / w), float((layer.sum(0) * r).sum() / w)


class TestFrame:
    @pytest.mark.parametrize("heading", [0.0, 37.0, 90.0, 200.0, 315.0])
    def test_an_enemy_straight_ahead_is_at_the_top_whatever_way_i_face(self, heading):
        h = math.radians(heading)
        x, z = 60 * math.sin(h), 60 * math.cos(h)                      # 60 m ahead of the frame
        m = Z.build(*frame(heading), units=one(x, z, heading + 180, 30, 10))[0]
        r, c = centroid(m[L["enemy"]])
        assert r == pytest.approx(15 - 60 / 8 - 0.5, abs=0.3) and c == pytest.approx(14.5, abs=0.3)
        # it faces me: cos -1, sin 0 (relative 180 deg)
        dens = m[L["enemy"]]
        assert float((m[L["enemy_cos"]] * dens).sum() / (dens * dens).sum()) == pytest.approx(-1, abs=1e-4)
        assert float(m[L["enemy_sin"]].abs().max()) < 1e-4

    def test_right_is_right(self):
        m = Z.build(*frame(90.0), units=one(0, -50, 90, 10, 10))[0]   # facing east, the unit to the south: my right
        r, c = centroid(m[L["enemy"]])
        assert r == pytest.approx(14.5, abs=0.3) and c == pytest.approx(14.5 + 50 / 8, abs=0.3)

    def test_cell_centre_and_cell_of_and_world_round_trip(self):
        s, f = Z.cell_centre(torch.tensor(3), torch.tensor(20), 8.0)
        row, col = Z.cell_of(s, f, 8.0)
        assert (int(row), int(col)) == (3, 20)
        x, z = Z.world(s, f, 10.0, -5.0, 30.0)
        s2, f2 = Z.local(x, z, 10.0, -5.0, 30.0)
        assert (float(s2), float(f2)) == pytest.approx((float(s), float(f)), abs=1e-4)


class TestDensity:
    def test_a_cell_half_covered_is_half_dense(self):
        # a 4 x 4 m box on the left half of the cell right of the centre-front (cell 8 m: cells' edges at 0, 8...)
        m = Z.build(*frame(0.0, 8.0), units=one(2.0, 6.0, 0, 4, 4))[0]
        row, col = Z.cell_of(t(2.0), t(6.0), 8.0)
        assert float(m[L["enemy"], row, col]) == pytest.approx(0.25, abs=1e-3)   # 16 of 64 m2
        assert float(m[L["enemy"]].sum()) == pytest.approx(0.25, abs=1e-3)

    def test_a_few_men_on_a_corner_give_a_little_not_one(self):
        m = Z.build(*frame(0.0, 8.0), units=one(0.0, 8.0, 0, 2, 2))[0]    # a 2 x 2 m box over 4 cells' corner
        cells = m[L["enemy"]][m[L["enemy"]] > 1e-6]
        assert len(cells) == 4 and cells.tolist() == pytest.approx([1 / 64] * 4, abs=1e-4)
        thin = Z.build(*frame(0.0, 16.0), units=one(3.0, 3.0, 30, 0.3, 0.3))[0]   # a sliver of 0.09 m2: not lost
        assert float(thin[L["enemy"]].sum()) * 256 == pytest.approx(0.09, rel=0.05)

    def test_the_fill_scales_the_density(self):
        full = Z.build(*frame(), units=one(0, 40, 0, 30, 10, fill=1.0))[0]
        half = Z.build(*frame(), units=one(0, 40, 0, 30, 10, fill=0.4))[0]
        assert torch.allclose(half[L["enemy"]], 0.4 * full[L["enemy"]])

    def test_every_cell_the_formation_covers_is_painted(self):
        m = Z.build(*frame(0.0, 8.0), units=one(0, 40, 0, 30, 10))[0]
        # 30 x 10 m centred (0, 40): columns -15..15 m -> cells 13..16, rows forward 35..45 m -> 2 rows
        painted = (m[L["enemy"]] > 0).nonzero().tolist()
        assert {c for _, c in painted} == {13, 14, 15, 16} and len({r for r, _ in painted}) == 2
        assert float(m[L["enemy"]].sum()) * 64 == pytest.approx(300, rel=1e-3)


class TestTurnedFormation:
    def test_a_turned_formation_covers_its_turned_shape_and_its_layers_give_its_angle(self):
        flat = Z.build(*frame(0.0, 4.0), units=one(0, 30, 0, 32, 8))[0][L["enemy"]]
        turned = Z.build(*frame(0.0, 4.0), units=one(0, 30, 90, 32, 8))[0][L["enemy"]]
        rows_f = (flat.sum(1) > 0).sum()
        rows_t = (turned.sum(1) > 0).sum()
        assert int(rows_t) > int(rows_f) * 2                       # long across -> long along the way
        assert float(turned.sum()) == pytest.approx(float(flat.sum()), rel=0.03)
        for rel in (0.0, 45.0, 90.0, 135.0, 250.0):
            m = Z.build(*frame(30.0, 8.0), units=one(0, 0, 30.0 + rel, 30, 10))[0]
            d = m[L["enemy"]]
            cos_ = float(m[L["enemy_cos"]].sum() / d.sum())
            sin_ = float(m[L["enemy_sin"]].sum() / d.sum())
            assert math.degrees(math.atan2(sin_, cos_)) % 360 == pytest.approx(rel, abs=0.01)

    def test_two_units_in_a_cell_average_their_fronts_by_density(self):
        u = {"x": t(-2.0, 2.0)[None], "z": t(4.0, 4.0)[None], "b": t(0.0, 90.0)[None], "front": t(4.0, 8.0)[None],
             "depth": t(8.0, 4.0)[None], "fill": t(1.0, 1.0)[None], "role": torch.tensor([[Z.OWN, Z.OWN]])}
        m = Z.build(*frame(0.0, 8.0), units=u)[0]
        row, col = Z.cell_of(t(0.0), t(4.0), 8.0)          # the cell s 0..8, f 0..8
        r, c = int(row), int(col)
        # the first box (s -4..0, f 0..8) lies in the cell left of it, the second (turned: s 0..4) in it
        assert float(m[L["own_cos"], r, c - 1]) == pytest.approx(0.5, abs=1e-3)
        assert float(m[L["own_sin"], r, c]) == pytest.approx(0.5, abs=1e-3)
        both = dict(u, x=t(4.0, 4.0)[None])                                  # both in the cell x 0..8
        m2 = Z.build(*frame(0.0, 8.0), units=both)[0]
        assert float(m2[L["own"], r, c]) == pytest.approx(1.0, abs=1e-3)     # 0.5 + 0.5
        assert float(m2[L["own_cos"], r, c]) == pytest.approx(0.5, abs=1e-3)
        assert float(m2[L["own_sin"], r, c]) == pytest.approx(0.5, abs=1e-3)


class TestEdgeAndArea:
    def test_the_maps_edge(self):
        # the frame 20 m from the east edge (bounds 100), facing east: the field ends 20 m ahead
        m = Z.build(*frame(90.0, 8.0, cx=80.0), bounds=100.0)[0]
        field = m[L["field"]]
        assert float(field[14, 14]) == pytest.approx(1.0) and float(field[0, 14]) == 0.0
        assert torch.allclose(m[L["off_map"]], 1 - field)
        rows_on = field[:, 14]
        # rows forward 16..24 m: the edge at 20 -> half
        assert float(rows_on[12]) == pytest.approx(0.5, abs=1e-3)
        # an enemy across the edge is painted where it is (the map shows what lies in its field)
        e = Z.build(*frame(90.0, 8.0, cx=80.0), units=one(130, 0, 0, 10, 10), bounds=100.0)[0]
        assert float(e[L["enemy"]].sum()) > 0

    def test_a_unit_off_the_maps_field_is_painted_only_where_it_reaches_in(self):
        m = Z.build(*frame(0.0, 4.0), units=one(0, 62, 0, 30, 10))[0]      # 4 m cells: the field ends 60 m ahead
        assert float(m[L["enemy"]].sum()) * 16 == pytest.approx(30 * 3, rel=1e-3)  # 57..60 m of its 57..67

    def test_my_area_layer_is_the_share_inside_the_turned_rectangle(self):
        area = {"x": t(0.0), "z": t(0.0), "b": t(0.0), "front": t(40.0), "depth": t(36.0)}
        m = Z.build(*frame(0.0, 8.0), area=area)[0][L["area"]]
        assert float(m[14, 14]) == pytest.approx(1.0) and float(m[14, 0]) == 0.0
        assert float(m.sum()) * 64 == pytest.approx(40 * 36, rel=1e-3)
        assert float(m[12, 14]) == pytest.approx(0.25, abs=1e-3)         # 18 m: the row 16..24 m holds 2 of 8
        off = dict(area, on=torch.tensor([False]))
        assert float(Z.build(*frame(), area=off)[0][L["area"]].sum()) == 0.0
        turned = Z.build(*frame(45.0, 8.0), area=dict(area, b=t(45.0)))[0][L["area"]]
        assert torch.allclose(turned, m, atol=1e-4)                    # turned with me: the same picture


class TestBatchAndState:
    def test_cell_size_is_a_parameter_and_chunks_give_the_same(self):
        R = 7
        g = torch.Generator().manual_seed(1)
        u = {"x": torch.rand(R, 3, generator=g) * 80 - 40, "z": torch.rand(R, 3, generator=g) * 80 - 40,
             "b": torch.rand(R, 3, generator=g) * 360, "front": torch.full((R, 3), 30.0),
             "depth": torch.full((R, 3), 10.0), "fill": torch.rand(R, 3, generator=g),
             "role": torch.tensor([[Z.ENEMY, Z.OWN, Z.SELF]] * R)}
        u["x"][:, 2], u["z"][:, 2] = 0.0, 0.0                       # the frame's own unit at its centre
        cell = t(4, 8, 16, 4, 8, 16, 8)
        heading = torch.rand(R, generator=g) * 360
        u["b"][:, 2] = heading
        a = Z.build(torch.zeros(R), torch.zeros(R), heading, cell, units=u, bounds=60.0)
        b = Z.build(torch.zeros(R), torch.zeros(R), heading, cell, units=u, bounds=60.0, chunk=2)
        assert a.shape == (R, len(Z.LAYERS), 30, 30) and torch.equal(a, b)
        assert float(a[:, L["obstacle"]].abs().max()) == 0.0
        # the same unit at 4 / 8 / 16 m cells: the same area in m2 x its fill
        for r in range(R):
            assert float(a[r, L["self"]].sum()) * float(cell[r]) ** 2 == pytest.approx(
                300 * float(u["fill"][r, 2]), rel=1e-3)

    def test_rects_of_a_simulator_state(self):
        P = load()
        st = scenario.build([{"attacker": 1, "sides": {
            1: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_spearmen_0", "x": 0, "z": 0, "b": 0}]},
            2: {"faction": "wh2_main_skv_skaven", "units": [{"key": "wh2_main_skv_inf_clanrat_spearmen_0", "x": 0,
                                                             "z": 60, "b": 180}]}}}], P)
        r = Z.rects_of(st.u, P.sim["formation"]["spacing_m"])
        assert float(r["front"][0, 0]) == pytest.approx(20 * 1.48, abs=0.01)   # 120 men / 6 ranks, h 1.48
        assert float(r["fill"][0, 0]) == pytest.approx(1.0)
        st.u["men"][0, 0] = 90.0                                              # 4.5 ranks of 20
        r = Z.rects_of(st.u, P.sim["formation"]["spacing_m"])
        assert float(r["fill"][0, 0]) == pytest.approx(90 / 100)
