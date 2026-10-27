"""Drill U2 'reach and hold' (tools/nn/unit/drill_go.py), the area's numbers in the order (net.order_of), the 'where'
cells when the area is beyond the map (view.allowed), the enemy's script to_area, the end rule. Needs torch."""
import math

import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import battle, card  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_go as DG  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def rect(st, x, z, b=0.0, w=64.0, d=64.0):
    f = lambda v: torch.full((st.B, st.N), float(v))       # noqa: E731
    return {"x": f(x), "z": f(z), "b": f(b), "front": f(w), "depth": f(d)}


def test_skirmishes_start_out_of_the_area_with_the_race_wanted():
    sp = DG.specs(90, torch.Generator().manual_seed(4))
    st, own, area, pace = DG.build(sp, P, "cpu")
    H = st.N // 2
    a0 = {k: v[:, 0] for k, v in area.items()}
    sd = Dc.signed_dist(st.u["x"][:, 0], st.u["z"][:, 0], a0)
    assert (sd < 0).all()
    for b, s in enumerate(sp):
        assert -float(sd[b]) == pytest.approx(s["edge"], abs=1e-2)               # the nearest point of the area
        e_sd = float(Dc.signed_dist(st.u["x"][b, H], st.u["z"][b, H], {k: v[b] for k, v in a0.items()}))
        if s["race"] == "held":
            assert e_sd > 0
        else:
            assert e_sd < 0
    assert {s["race"] for s in sp} == set(DG.RACES) and {s["pace"] for s in sp} == set(DG.PACES)
    assert any(s["edge"] > 240 for s in sp) and any(s["edge"] < DG.NEAR_M[1] for s in sp)
    assert set(pace.tolist()) == {Nt.PACE_FAST, Nt.PACE_SLOW}
    assert len(DG.exam_specs()) == 360 and DG.exam_specs() == DG.exam_specs()


def test_the_order_carries_the_area_in_my_frame():
    st = battle([(SPEAR, 0, 0, 90)], [(CLANRAT, 600, 600, 180)])
    now = V.raw(st, SP)
    area = {"x": torch.tensor([300.0]), "z": torch.tensor([-50.0]), "b": torch.tensor([90.0]),
            "front": torch.tensor([64.0]), "depth": torch.tensor([80.0])}
    o = Nt.order_of(now[:, 0], area, torch.tensor([Nt.PACE_SLOW]), torch.tensor([False]))[0]
    g = dict(zip(Nt.ORDER, o.tolist()))
    assert g["area_forward"] == pytest.approx(3.0, abs=1e-4) and g["area_right"] == pytest.approx(0.5, abs=1e-4)
    assert g["area_front"] == pytest.approx(0.64) and g["area_cos"] == pytest.approx(1.0) and g["going"] == 1.0
    assert g["pace_slow"] == 1.0 and g["in_place"] == 0.0


def test_an_area_beyond_the_map_allows_the_edge_cells_towards_it():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 600, 600, 180)])
    who = torch.zeros((1, st.N), dtype=torch.bool)
    who[0, 0] = True
    far = rect(st, 400.0, 300.0)                                  # 500 m away, bearing ~53 deg
    v = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=far)
    ok = v["cells_ok"][0]
    assert 0 < int(ok.sum()) < 30
    rows, cols = ok.nonzero(as_tuple=True)
    assert all(int(r) in (0, Z.CELLS - 1) or int(c) in (0, Z.CELLS - 1) for r, c in zip(rows, cols))
    s, f = Z.cell_centre(rows.float(), cols.float(), 8.0)
    way = math.degrees(math.atan2(400.0, 300.0))
    assert all(abs((math.degrees(math.atan2(float(a), float(b))) - way + 180) % 360 - 180) <= V.TOWARD_DEG + 1e-3
               for a, b in zip(s, f))
    assert float(v["map"][0, Z.L["area"]].sum()) == 0.0           # not on the map: no layer, the order carries it
    near = rect(st, 0.0, 80.0)
    v2 = V.build(V.raw(st, SP), V.Memory(1, st.N).hist, card(st), who, area=near)
    assert float(v2["map"][0, Z.L["area"]].sum()) > 0 and int(v2["cells_ok"][0].sum()) == 64


def test_the_enemy_goes_to_the_area_holds_it_and_fights_who_comes():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 300, 180)])
    H = st.N // 2
    area = rect(st, 0.0, 150.0)
    o = DG.to_area(st, area)
    assert int(o.kind[0, H]) == O.MOVE and bool(o.run[0, H])
    assert (float(o.x[0, H]), float(o.z[0, H])) == (0.0, 150.0)
    st.u["z"][0, H] = 150.0                                        # there; we are far
    assert int(DG.to_area(st, area).kind[0, H]) == O.HOLD
    st.u["z"][0, 0] = 110.0                                        # we come within GRAB_M of the area
    o = DG.to_area(st, area)
    assert int(o.kind[0, H]) == O.ATTACK and int(o.target[0, H]) == 0


def test_the_stage_and_the_limit_follow_the_arrival():
    sp = DG.specs(2, torch.Generator().manual_seed(1))
    st, own, area, pace = DG.build(sp, P, "cpu")
    end = DG.Ender(sp, "cpu")
    a0 = {k: v[:, 0] for k, v in area.items()}
    st.u["x"][0, 0], st.u["z"][0, 0] = a0["x"][0], a0["z"][0]      # battle 0: our unit in its area
    end.after_step(st, SP, area)
    assert end.arrived.tolist() == [True, False] and float(end.arrive_t[0]) == float(st.t[0])
    st.t[:] = float(end.arrive_t[0]) + DG.ARRIVED_S
    end.after_step(st, SP, area)
    assert bool(st.done[0]) and int(end.why[0]) == 3
    assert bool(st.done[1]) == bool(st.t[1] >= end.never[1])


def test_a_short_rollout_of_the_drill_runs_within_the_bans():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 30, "cell_m": 8.0, "bounds": None}
    sp = DG.specs(4, torch.Generator().manual_seed(6))
    batch, stats = T.rollout(Nt.UnitNet(), Nt.Critic(), cfg, P, "cpu", T.Duty("cpu"), T.go_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
    assert batch["arrived"].shape == batch["valid"].shape and batch["pace"].shape == (4,)
