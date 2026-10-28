"""Drill U5c 'the shooter's width' (tools/nn/unit/drill_shootw.py, exam_shootw.py): the layouts (a gap between our
infantry ahead, the gap at a slant, the open), the position wide enough to choose, the friends closing a wide line's
fire and not a narrow one's, the scripts' width, a short rollout, the mix. Needs torch."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import drill_shootw as DW5  # noqa: E402
from tools.nn.unit import exam_shootw as EW  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
HANDGUN, SLING = "wh_main_emp_inf_handgunners", "wh2_main_skv_inf_skavenslave_slingers_0"


def test_the_layouts():
    sp = DW5.specs(60, torch.Generator().manual_seed(3))
    st, own, area, goal = DW5.build(sp, P, "cpu")
    assert {s["layout"] for s in sp} == set(DW5.LAYOUTS)
    for b, s in enumerate(sp):
        assert float(st.u["range"][b, 0]) > 0 and bool(own[b, 0]) and not bool(own[b, 1:].any())
        off = (s["sb"] - s["b"] + 180) % 360 - 180
        if s["layout"] == "open":
            assert len(s["friends"]) == 0 and DW5.OPEN_FRONT_M[0] <= s["pos_front"] <= DW5.OPEN_FRONT_M[1]
            assert abs(off) <= DW5.SECTOR_OFF + 1e-6
            continue
        # the friends: the gap between their inner edges straight ahead, their rear edges beyond our front edge
        (xl, zl), (xr, zr) = DW5.friend_places(s)
        fl, fr = s["friends"]
        assert (xr - fr["front"] / 2) - (xl + fl["front"] / 2) == pytest.approx(s["gap"])
        assert zl - fl["depth"] / 2 > s["depth"] / 2 and zr - fr["depth"] / 2 > s["depth"] / 2
        for i, (x, z) in ((1, (xl, zl)), (2, (xr, zr))):
            sx, sf = Z.local(st.u["x"][b, i], st.u["z"][b, i], 0.0, 0.0, s["b"])
            assert float(sx) == pytest.approx(x, abs=1e-3) and float(sf) == pytest.approx(z, abs=1e-3)
        # the position: wide enough to choose a line wider than the gap
        assert s["pos_front"] >= s["gap"] + 2 * DW5.MARGIN_M - 1e-6
        if s["layout"] == "angle":
            assert DW5.ANGLE_DEG[0] - 1e-6 <= abs(off) <= DW5.ANGLE_DEG[1] + 1e-6
    assert len(DW5.exam_specs()) == 240 and [s["layout"] for s in DW5.exam_specs()[:3]] == list(DW5.LAYOUTS)


def _lost_share(front_factor, steps=40):
    """A handgunners' gap layout (a 16 m gap), slingers standing 70 m ahead: our share of fire lost to friends."""
    s = copy.deepcopy(DW5.specs(1, torch.Generator().manual_seed(1), fixed=["gap"])[0])
    pp = DH._passports()
    s.update(ours="emp", theirs="skv", me=HANDGUN, gap=16.0, ahead=4.0, sb=s["b"])
    s["front"], s["depth"] = DH.box(pp[HANDGUN], front_factor)
    s["friends"] = [{"key": "wh_main_emp_inf_spearmen_0", "front": 25.0, "depth": 12.0}] * 2
    s["foes"] = [{"key": SLING, "shooter": True, "rel": 0.0, "gap": 70.0}]
    st, own, area, goal = DW5.build([s], P, "cpu")
    stand = DW5.shooter_slots([s], st)
    cap = skip = 0.0
    for _ in range(steps):
        o = DW5.script(st, goal, stand)
        o.kind[:, 0], o.x[:, 0], o.z[:, 0] = O.HOLD, st.u["x"][:, 0], st.u["z"][:, 0]
        battle.step(st, o, P)
        cap += float(st.u["fc_cap"][0, 0])
        skip += float(st.u["fc_skip"][0, 0])
    assert cap > 0
    return skip / cap, float(st.u["width"][0, 0])


def test_friends_close_a_wide_line_s_fire_and_not_a_narrow_one_s():
    wide, w_wide = _lost_share(1.6)
    narrow, w_narrow = _lost_share(0.35)
    assert w_wide > 16.0 >= w_narrow - 2.0
    assert wide > 0.2 and narrow < wide / 3


def test_the_scripts_width_and_a_short_rollout_on_the_mix():
    sp = DW5.specs(6, torch.Generator().manual_seed(4), fixed=list(DW5.LAYOUTS))
    st, own, area, goal = DW5.build(sp, P, "cpu")
    want = EW.gap_width(sp, area)
    for b, s in enumerate(sp):
        assert float(want[b]) == pytest.approx(s["gap"] if s["layout"] != "open" else s["pos_front"])
    assert "shootw" in T.MIX and "shootw" in T.SETUPS
    torch.manual_seed(0)
    cfg = {"battles": 3, "seconds": 6, "cell_m": 8.0, "bounds": None}
    setup = T.shootw_setup(cfg, P, "cpu", sp=sp[:3])
    net = Nt.UnitNet()
    _, rs = T.rollout(net, None, cfg, P, "cpu", T.Duty("cpu"), setup)
    assert rs["bans_broken"] == 0 and rs["nan"] == 0
