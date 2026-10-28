"""Drill U6 'withdraw from melee' (tools/nn/unit/drill_withdraw.py): in melee at the order, the new area behind or aside
at 40-150 m on our grid, the stage and the end rule, a short rollout. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_withdraw as DW  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]


def test_in_melee_at_the_order_the_new_area_behind_or_aside():
    sp = DW.specs(24, torch.Generator().manual_seed(3))
    st, own, area, pace = DW.build(sp, P, "cpu")
    H = st.N // 2
    fighting = 0
    for b, s in enumerate(sp):
        if bool(st.done[b]):
            continue
        fighting += int(bool(st.u["m"][b, 0]))
        a0 = {k: v[b, 0] for k, v in area.items() if k != "on"}
        sd = float(Dc.signed_dist(st.u["x"][b, 0], st.u["z"][b, 0], a0))
        assert -sd == pytest.approx(s["edge_real"], abs=0.5) and DW.EDGE_M[0] - 8 <= -sd <= DW.EDGE_M[1] + 8
        s_, f_ = Z.local(a0["x"], a0["z"], st.u["x"][b, 0], st.u["z"][b, 0], st.u["b"][b, 0])
        way = math.degrees(math.atan2(float(s_), float(f_))) % 360
        mid, spread = DW.DIRS[s["way"]]
        assert min(abs((way - m + 180) % 360 - 180) for m in (mid, -mid)) <= spread + 12
    assert fighting >= 0.8 * (~st.done).sum()
    assert set(pace.tolist()) == {Nt.PACE_FAST, Nt.PACE_SLOW} and len(DW.exam_specs()) == 240


def test_the_end_after_the_arrival_once_the_enemy_is_out_of_the_zone():
    sp = DW.specs(2, torch.Generator().manual_seed(4))
    st, own, area, pace = DW.build(sp, P, "cpu")
    H = st.N // 2
    end = DW.Ender(sp, st)
    a0 = {k: v[:, 0] for k, v in area.items() if k != "on"}
    st.u["x"][:, 0], st.u["z"][:, 0] = a0["x"], a0["z"]              # both arrive
    st.u["x"][1, H], st.u["z"][1, H] = 900.0, 900.0                   # battle 1: the enemy far away
    st.u["x"][0, H], st.u["z"][0, H] = a0["x"][0] + 40.0, a0["z"][0]  # battle 0: it followed us, near
    st.u["r"][:, 0] = False
    end.after_step(st, SP, area)
    assert end.arrived.all() and st.done.tolist() == [False, True] and int(end.why[1]) == 2


def test_a_short_rollout_of_the_drill():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 30, "cell_m": 8.0, "bounds": None}
    sp = DW.specs(4, torch.Generator().manual_seed(5))
    batch, stats = T.rollout(Nt.UnitNet(), Nt.Critic(), cfg, P, "cpu", T.Duty("cpu"),
                             T.withdraw_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
