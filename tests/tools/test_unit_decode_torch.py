"""The unit network's choices -> orders and the leash (tools/nn/unit/decode.py): a pursuit stops at the area's edge
facing the target and goes on when the target comes back; a unit pushed out in melee is not leashed; the leash shows
in the unit's own row. Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import SP, battle, card, view_of  # noqa: E402
from tools.nn.sim import battle as Bt  # noqa: E402
from tools.nn.sim import geometry, replay  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def square(st, cx=0.0, cz=0.0, side=100.0):
    f = lambda v: torch.full((st.B, st.N), float(v))       # noqa: E731
    return {"x": f(cx), "z": f(cz), "b": f(0.0), "front": f(side), "depth": f(side)}


def own(st):
    m = torch.zeros((st.B, st.N), dtype=torch.bool)
    m[:, 0] = True
    return m


class TestToOrders:
    def test_target_cell_face_and_pace(self):
        st = battle([(SPEAR, 0, 0, 30)], [(CLANRAT, 0, 60, 180)])
        H = st.N // 2
        v = view_of(st)
        now = V.raw(st, SP)
        K = v["slot"].shape[1]
        here = 15 * Z.CELLS + 15                                # the cell I stand in (cell_of(0, 0))
        o = Dc.to_orders({"where": torch.tensor([here]), "face": torch.tensor([16]), "target": torch.tensor([K]),
                          "pace": torch.tensor([0])}, v, now, 8.0, st.B, st.N)
        assert int(o.kind[0, 0]) == O.HOLD and float(o.face[0, 0]) == pytest.approx(30 + 90)
        assert int(o.kind[0, H]) == O.KEEP and not bool(o.run[0, 0])
        cell = 0 * Z.CELLS + 15                                 # the top row, right of the centre line: 116 m ahead
        o = Dc.to_orders({"where": torch.tensor([cell]), "face": torch.tensor([0]), "target": torch.tensor([K]),
                          "pace": torch.tensor([1])}, v, now, 8.0, st.B, st.N)
        x, z = Z.world(torch.tensor(4.0), torch.tensor(116.0), 0.0, 0.0, 30.0)
        assert int(o.kind[0, 0]) == O.MOVE and bool(o.run[0, 0])
        assert (float(o.x[0, 0]), float(o.z[0, 0])) == pytest.approx((float(x), float(z)), abs=1e-4)
        o = Dc.to_orders({"where": torch.tensor([cell]), "face": torch.tensor([3]), "target": torch.tensor([1]),
                          "pace": torch.tensor([1])}, v, now, 8.0, st.B, st.N)
        assert int(o.kind[0, 0]) == O.ATTACK and int(o.target[0, 0]) == H and float(o.face[0, 0]) < 0


class TestLeash:
    def test_a_pursuit_stops_at_the_edge_facing_the_target_and_goes_on_when_it_comes_back(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 0)])
        H = st.N // 2
        area, mine = square(st), own(st)
        leash = Dc.Leash(st.B, st.N)
        intended = replay.hold(st)
        intended.kind[0, 0], intended.target[0, 0], intended.run[0, 0] = O.ATTACK, H, True
        flee = (0.0, 400.0)
        was_on, worst = False, 1e9
        for k in range(160):
            e = replay.hold(st)
            e.kind[0, H], e.x[0, H], e.z[0, H], e.run[0, H] = O.MOVE, flee[0], flee[1], True
            o = O.merge(e, leash.apply(st, intended, area, mine), mine)
            Bt.step(st, o, P)
            assert not bool(st.u["m"][0, 0])
            worst = min(worst, float(Dc.signed_dist(st.u["x"][0, 0], st.u["z"][0, 0], {k_: a[0, 0] for k_, a in area.items()})))
            was_on = was_on or bool(leash.on[0, 0])
        assert was_on and bool(leash.on[0, 0])
        assert worst >= -0.5                                     # never out of the area (half a metre of slack)
        assert abs((float(st.u["b"][0, 0]) + 180) % 360 - 180) < 6       # standing at the edge facing the target (north)
        # the target comes back into the area: the leash comes off and the attack goes on
        for k in range(200):
            e = replay.hold(st)
            e.kind[0, H], e.x[0, H], e.z[0, H], e.run[0, H] = O.MOVE, 0.0, 30.0, True
            o = O.merge(e, leash.apply(st, intended, area, mine), mine)
            Bt.step(st, o, P)
            if not bool(leash.on[0, 0]):
                break
        assert not bool(leash.on[0, 0]) and int(o.kind[0, 0]) == O.ATTACK

    def test_a_unit_pushed_out_in_melee_is_not_leashed(self):
        st = battle([(SPEAR, -50, 0, 90)], [(CLANRAT, 50, 0, 270)])
        H = st.N // 2
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        st.u["x"][0, 0] = -(depth[0, 0] / 2 - 1.0)
        st.u["x"][0, H] = depth[0, H] / 2 - 1.0
        area, mine = square(st, cx=-200.0), own(st)              # both far outside the area
        leash = Dc.Leash(st.B, st.N)
        intended = replay.hold(st)
        intended.kind[0, 0], intended.target[0, 0] = O.ATTACK, H
        for _ in range(4):
            out = leash.apply(st, intended, area, mine)
            Bt.step(st, out, P)
        assert bool(st.u["m"][0, 0])
        out = leash.apply(st, intended, area, mine)
        assert not bool(leash.on[0, 0]) and int(out.kind[0, 0]) == O.ATTACK and int(out.target[0, 0]) == H

    def test_the_leash_shows_in_my_own_row_only_and_the_area_is_in_the_map(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 180)])
        on = torch.zeros((st.B, st.N), dtype=torch.bool)
        on[0, 0] = True
        on[0, st.N // 2] = True                                   # (an enemy's flag never shows)
        now = V.raw(st, SP, leash=on)
        who = own(st)
        v = V.build(now, V.Memory(st.B, st.N).hist, card(st), who, area=square(st, cz=20.0))
        col = V.ROW.index("leashed")
        assert float(v["rows"][0, 0, col]) == 1.0 and float(v["rows"][0, 1, col]) == 0.0
        off = V.build(V.raw(st, SP), V.Memory(st.B, st.N).hist, card(st), who, area=square(st, cz=20.0))
        assert float(off["rows"][0, 0, col]) == 0.0
        assert float(v["map"][0, Z.L["area"]].sum()) * 64 == pytest.approx(100 * 100, rel=0.01)
        assert len(Nt.ORDER) and v["cells_ok"][0].sum() < Z.CELLS ** 2
