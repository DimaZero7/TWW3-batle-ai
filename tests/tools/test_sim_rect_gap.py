"""The formations' true rectangle gap (geometry.rect_gap, config/nn/sim.json contact.rect_gap): equal to the centre-line
gap head-on, the real edge distance or overlap at an angle or offset; the contact test uses it. Needs torch."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import geometry  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()


def units(rows):
    """Two formations, 1.5 m spacing (30 m wide, 120 men: 20 files x 6 ranks = 30 x 9 m), or a lone man."""
    t = lambda k: torch.tensor([[float(r[k]) for r in rows]])     # noqa: E731
    u = {k: t(k) for k in ("x", "z", "b", "men", "width")}
    u["men0"] = u["men"].clone()
    u["side"] = torch.tensor([[1, 2]])
    u["radius"] = torch.full_like(u["x"], 0.65)
    return u


def gaps(rows):
    u = units(rows)
    ray = geometry.pairwise(u, 1.5)["gap"]
    rect = geometry.pairwise(u, 1.5, rect=True)["gap"]
    return float(ray[0, 0, 1]), float(rect[0, 0, 1]), float(rect[0, 1, 0])


A = dict(x=0, z=0, b=0, men=120, width=30)


class TestRectGap:
    def test_head_on_is_the_centre_line_gap(self):
        for z in (6.0, 9.0, 11.5, 30.0):                    # overlap 3 m, touching, 2.5 m apart, 21 m apart
            ray, rect, back = gaps([A, dict(x=0, z=z, b=180, men=120, width=30)])
            assert rect == pytest.approx(ray, abs=1e-4) == pytest.approx(z - 9.0, abs=1e-4)
            assert back == pytest.approx(rect, abs=1e-4)

    def test_offset_fronts_by_their_depth_gap(self):
        # 20 m to the side (10 m of the fronts overlap), the fronts 2 m apart: the edges are 2 m apart, the centre
        # line (through the corners' side) reads more
        ray, rect, _ = gaps([A, dict(x=20, z=11, b=180, men=120, width=30)])
        assert rect == pytest.approx(2.0, abs=1e-4)
        assert ray > 4.0

    def test_corner_overlap_at_an_angle(self):
        # j faces 225 deg (f = (-c, -c), its front along l = (-c, c)); its corner centre + hf f - hl l, the one nearest
        # i, 2 m inside i's front edge (z 4.5) 10 m right of i's centre: they overlap, the centre line reads them apart
        c = math.cos(math.radians(45))
        hf, hl = 4.5, 15.0
        x, z = 10.0 - c * (hl - hf), 2.5 + c * (hf + hl)
        ray, rect, back = gaps([A, dict(x=x, z=z, b=225, men=120, width=30)])
        assert rect < -1.0
        assert ray > 0.0
        assert back == pytest.approx(rect, abs=1e-4)

    def test_apart_diagonally_is_the_corner_distance(self):
        # j square to i, 40 m right and 30 m ahead: corner to corner (40 - 30, 30 - 9) apart
        ray, rect, _ = gaps([A, dict(x=40, z=30, b=180, men=120, width=30)])
        assert rect == pytest.approx(math.hypot(10, 21), abs=1e-3)
        assert ray > rect

    def test_a_lone_man_keeps_the_centre_line_gap(self):
        u = units([dict(x=0, z=0, b=0, men=1, width=5), dict(x=12, z=8, b=200, men=120, width=30)])
        assert float(geometry.pairwise(u, 1.5, rect=True)["gap"][0, 0, 1]) == pytest.approx(
            float(geometry.pairwise(u, 1.5)["gap"][0, 0, 1]))

    def test_the_switch_is_off(self):
        # off: the walkpass2 twin gets worse with it (config/nn/sim.json contact.rect_gap_why)
        assert P.sim["contact"].get("rect_gap") is False
