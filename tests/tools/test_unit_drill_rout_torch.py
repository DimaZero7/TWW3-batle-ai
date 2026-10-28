"""Drill U7 'a routing enemy' (tools/nn/unit/drill_rout.py): a worn enemy in melee that the simulator's morale breaks, a
fleeing one near our front, the end rule, a short rollout. Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_rout as DR  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()


def test_the_worn_enemy_breaks_by_the_simulators_morale_and_the_crossing_one_crosses():
    sp = DR.specs(40, torch.Generator().manual_seed(3), fixed=list(DR.KINDS))
    st, own, area = DR.build(sp, P, "cpu")
    H = st.N // 2
    brk = torch.tensor([s["kind"] == "breaking" for s in sp])
    engaged = st.u["m"][:, 0] | st.u["r"][:, H]                          # breaking: in melee, or broken already
    assert engaged[brk].float().mean() > 0.9
    assert st.u["r"][~brk, H].all()                                      # crossing: routing
    assert not st.done[~brk].any()                                       # (the drill's end rule, not the simulator's)
    for b in range(st.B):
        a0 = {k: v[b, 0] for k, v in area.items() if k != "on"}
        assert float(Dc.signed_dist(st.u["x"][b, 0], st.u["z"][b, 0], a0)) > 0
    routed = st.u["r"][:, H].clone()
    from tools.nn.sim import geometry
    from tools.nn.unit import zmap as Z
    crossed = torch.zeros(st.B, dtype=torch.bool)
    end = DR.Ender(st)
    for _ in range(60):                                                 # 30 s, ours holding (the drill's end rule)
        o = DR.script(st)
        o.kind[:, 0], o.target[:, 0] = 0, -1
        battle.step(st, o, P)
        end.after_step(st, P.sim["formation"]["spacing_m"])
        routed = routed | st.u["r"][:, H]
        s_, _ = Z.local(st.u["x"][:, H], st.u["z"][:, H], st.u["x"][:, 0], st.u["z"][:, 0], st.u["b"][:, 0])
        crossed = crossed | (s_ > 0)                                    # it got past our centre's line, rightwards
    assert (st.u["order_kind"][~brk, 1] == 2).all() or not (st.u["men"][~brk, 1] > 0).all()   # it chases
    assert routed[brk].float().mean() >= 0.5                            # most of the worn ones broke
    assert crossed[~brk].float().mean() >= 0.5                          # most routers ran across our front
    assert len(DR.exam_specs()) == 240


def test_a_short_rollout_of_the_drill():
    torch.manual_seed(0)
    cfg = {"battles": 4, "seconds": 30, "cell_m": 8.0, "bounds": None}
    sp = DR.specs(4, torch.Generator().manual_seed(4))
    batch, stats = T.rollout(Nt.UnitNet(), Nt.Critic(), cfg, P, "cpu", T.Duty("cpu"), T.rout_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
