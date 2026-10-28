"""Drill U5b 'a clean line of fire' (tools/nn/unit/drill_shoot2.py): the two layouts, the sectors and targets per unit,
the friends' bookkeeping of lost fire, a short rollout; every drill of the mix has its exam. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_shoot2 as D2  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()


def test_the_layouts_positions_and_sectors():
    sp = D2.specs(40, torch.Generator().manual_seed(3))
    st, own, area, goal = D2.build(sp, P, "cpu")
    for b, s in enumerate(sp):
        for i in (0, 1):
            ai = {k: v[b, i] for k, v in area.items() if k in ("x", "z", "b", "front", "depth")}
            assert float(Dc.signed_dist(st.u["x"][b, i], st.u["z"][b, i], ai)) > 0
        shooters = [bool(area["sw"][b, i] > 0) for i in (0, 1)]
        assert shooters == ([True, True] if s["layout"] == "shooters" else [False, True])
        if s["sub"] in ("file", "screen"):
            a0 = {k: v[b, 0] for k, v in area.items()}
            _, f1 = Z.local(area["x"][b, 1], area["z"][b, 1], a0["x"], a0["z"], s["b"])
            assert float(f1) < 0                                            # the second one behind the first
        assert float(st.u["range"][b, 1]) > 0
        assert (float(st.u["range"][b, 0]) > 0) == (s["layout"] == "shooters")
    assert {s["sub"] for s in sp} == {"side", "file", "screen"} and len(D2.exam_specs()) == 240


def test_a_short_rollout_and_the_lost_fire_bookkeeping():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 40, "cell_m": 8.0, "bounds": None}
    sp = D2.specs(4, torch.Generator().manual_seed(5), fixed=["screen"])
    for s in sp:
        for f in s["foes"]:
            f["gap"] = 120.0
    setup = T.shoot2_setup(cfg, P, "cpu", sp=sp)
    batch, stats = T.rollout(Nt.UnitNet(), None, cfg, P, "cpu", T.Duty("cpu"), setup)
    assert stats["bans_broken"] == 0 and stats["nan"] == 0 and batch["act"]["where"].shape[1] == 8
    assert "fc_cap" in setup["st"].u and "fc_skip" in setup["st"].u


def test_every_drill_of_the_mix_has_its_exam():
    import tools.nn.unit.exam_shoot2  # noqa: F401
    assert "shoot2" in T.MIX and set(T.MIX) == set(T.SETUPS) - {"smoke"}
