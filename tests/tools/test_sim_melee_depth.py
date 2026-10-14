"""A formation in melee keeps room for its dead (geometry.dims melee_depth, config/nn/sim.json formation.melee_depth):
ranks = max(ceil(r), min(r + c (1 - men / men0), ceil(men0 / files0))), r = men / files, only while its melee clock
runs (contact_s > 0); a whole unit, a unit out of melee and the switch off keep the closed-up box; the companion's mirror
(exchange.gaps, the game's melee flag) agrees. Needs torch."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import exchange  # noqa: E402
from tools.nn.sim import geometry  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
C = 2.8


def units(rows, melee=(True, True)):
    """Empire swordsmen 1.48 x 1.6 (30 m: 20 files, 6 ranks whole) and clanrats 1.6 x 1.7 (30 m: 18 files, 9 ranks)."""
    t = lambda k: torch.tensor([[float(r[k]) for r in rows]])     # noqa: E731
    u = {k: t(k) for k in ("x", "z", "b", "men", "men0", "width", "sp_h", "sp_v", "radius")}
    u["side"] = torch.tensor([[1, 2]])
    u["contact_s"] = torch.tensor([[5.0 if m else 0.0 for m in melee]])
    u["m"] = torch.tensor([list(melee)])
    return u


def emp(**kw):
    return dict(dict(x=0, z=0, b=0, men=120, men0=120, width=30, sp_h=1.48, sp_v=1.6, radius=0.65), **kw)


def rat(**kw):
    return dict(dict(x=0, z=12, b=180, men=160, men0=160, width=30, sp_h=1.6, sp_v=1.7, radius=0.65), **kw)


def depth(u, c=C):
    return geometry.dims(u, 1.5, c)[1][0]


class TestDims:
    def test_whole_units_unchanged(self):
        u = units([emp(), rat()])
        assert depth(u).tolist() == pytest.approx(depth(u, 0.0).tolist())
        assert depth(u).tolist() == pytest.approx([6 * 1.6, 9 * 1.7])

    def test_half_lost_in_melee_keeps_room(self):
        # 60 of 120 swordsmen: 20 files, r = 3; 3 + 2.8 x 0.5 = 4.4 ranks (closed up: 3)
        u = units([emp(men=60), rat()])
        assert float(depth(u)[0]) == pytest.approx(4.4 * 1.6)
        assert float(depth(u, 0.0)[0]) == pytest.approx(3 * 1.6)

    def test_out_of_melee_closes_up(self):
        u = units([emp(men=60), rat()], melee=(False, True))
        assert float(depth(u)[0]) == pytest.approx(3 * 1.6)

    def test_never_deeper_than_whole(self):
        u = units([emp(men=60), rat()])
        assert float(depth(u, 50.0)[0]) == pytest.approx(6 * 1.6)

    def test_fewer_men_than_files(self):
        # 10 swordsmen: 10 files, r = 1; 1 + 2.8 x 110 / 120
        u = units([emp(men=10), rat()])
        assert float(depth(u)[0]) == pytest.approx((1 + C * 110 / 120) * 1.6)
        assert float(geometry.dims(u, 1.5, C)[0][0, 0]) == pytest.approx(10 * 1.48)

    def test_never_shallower_than_closed_up(self):
        # 119 of 120 men: r 5.95, + 0.02: below ceil(r) = 6 -> the closed-up 6 ranks
        u = units([emp(men=119), rat()])
        assert float(depth(u)[0]) == pytest.approx(6 * 1.6)

    def test_no_clock_no_rule(self):
        u = units([emp(men=60), rat()])
        del u["contact_s"]
        assert float(depth(u)[0]) == pytest.approx(3 * 1.6)


class TestPairwise:
    def test_head_on_gap_shrinks_by_the_extra_half_depths(self):
        u = units([emp(men=60), rat(men=80)])
        g0 = float(geometry.pairwise(u, 1.5, rect=True)["gap"][0, 0, 1])
        g1 = float(geometry.pairwise(u, 1.5, rect=True, melee_depth=C)["gap"][0, 0, 1])
        extra = (4.4 - 3) * 1.6 / 2 + ((80 / 18 + C * 0.5) - 5) * 1.7 / 2
        assert g0 - g1 == pytest.approx(extra, abs=1e-4)

    def test_config_reader(self):
        p = copy.deepcopy(P)
        p.sim["formation"]["melee_depth"] = {"ranks": C}
        assert geometry.melee_depth(p.sim) == C
        p.sim["formation"].pop("melee_depth")
        assert geometry.melee_depth(p.sim) == 0.0


class TestCompanionMirror:
    def test_exchange_gaps_match_pairwise(self):
        rows = [emp(men=60), rat(men=80, x=3)]
        u = units(rows, melee=(True, False))
        state = {k: np.array([[r[k] for r in rows]], dtype=float) for k in ("x", "z", "b", "men")}
        state["m"] = np.array([[True, False]])
        shape = {k: np.array([r[k] for r in rows], dtype=float) for k in ("width", "sp_h", "sp_v", "men0", "radius")}
        shape.update(spacing=1.5, reach_m=-2.5, lord_reach_m=1.0, hold_m=2.0, engage_m=None, melee_depth=C)
        gap, _ = exchange.gaps(state, shape)
        sim = geometry.pairwise(u, 1.5, melee_depth=C)["gap"][0].numpy()
        assert gap[0, 1] == pytest.approx(float(sim[0, 1]), abs=1e-4)
        shape["melee_depth"] = 0.0
        gap0, _ = exchange.gaps(state, shape)
        assert gap0[0, 1] > gap[0, 1] + 0.5
