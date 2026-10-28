"""The unit network's width of the line and the shooter's map (tools/nn/unit: net.py width head, decode.py width_of /
width_limits, train.py order_changes / restore / the cell by range, view.py fire_arc / cell_size / allowed, zmap.py the
reach and fire_arc layers). Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import ARCHER, CLANRAT, SPEAR, battle, card, net_inputs  # noqa: E402
from tools.nn.sim import replay  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
BI, II = torch.tensor([0]), torch.tensor([0])


class TestHead:
    def test_the_width_head_starts_at_keep(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
        v, o, cm = net_inputs(st)
        lg = Nt.UnitNet()(v, o, cm)
        assert lg["width"].shape == (1, Nt.WIDTHS) and "width" in Nt.HEADS
        p = torch.softmax(lg["width"].detach(), 1)
        assert float(p[0, Nt.WIDTH_KEEP]) > 0.98

    def test_the_choices_go_from_the_narrowest_to_the_widest(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
        lo, hi = Dc.width_limits(st, P)
        assert float(lo[0, 0]) == pytest.approx(5 * 1.48) and float(hi[0, 0]) == Dc.WIDTH_MAX_M
        k = torch.arange(Nt.WIDTHS)
        w = Dc.width_of(k, lo[0, 0].expand(Nt.WIDTHS), hi[0, 0].expand(Nt.WIDTHS))
        assert float(w[0]) == -1.0 and float(w[1]) == pytest.approx(5 * 1.48) and float(w[-1]) == pytest.approx(80.0)
        assert bool((w[2:] > w[1:-1]).all())

    def test_the_decoder_gives_a_width_but_not_with_an_attack(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 60, 180)])
        v, o, cm = net_inputs(st)
        now = V.raw(st, SP)
        K = v["slot"].shape[1]
        here = 15 * Z.CELLS + 15
        wl = Dc.width_limits(st, P)
        act = {"where": torch.tensor([here]), "face": torch.tensor([0]), "target": torch.tensor([K]),
               "pace": torch.tensor([0]), "width": torch.tensor([Nt.WIDTHS - 1])}
        assert float(Dc.to_orders(act, v, now, cm, st.B, st.N, wlim=wl).width[0, 0]) == pytest.approx(80.0)
        act["target"] = torch.tensor([1])
        assert float(Dc.to_orders(act, v, now, cm, st.B, st.N, wlim=wl).width[0, 0]) == -1.0
        act["target"], act["width"] = torch.tensor([K]), torch.tensor([Nt.WIDTH_KEEP])
        assert float(Dc.to_orders(act, v, now, cm, st.B, st.N, wlim=wl).width[0, 0]) == -1.0


class TestCost:
    def test_another_width_is_a_change_keep_and_the_same_are_not(self):
        st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
        base = replay.hold(st)
        w0 = float(st.u["width"][0, 0])
        for w, want in ((-1.0, False), (w0, False), (w0 + 0.5, False), (60.0, True), (10.0, True)):
            o = replay.hold(st)
            o.width[0, 0] = w
            assert bool(T.order_changes(o, base, BI, II, widths=T.widths_of(st, P, o, BI, II))[0]) == want, w


class TestRestore:
    def test_an_old_checkpoint_gets_zero_map_channels_and_the_width_head_at_keep(self, tmp_path):
        torch.manual_seed(3)
        net, critic = Nt.UnitNet(), Nt.Critic()
        opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()))
        old = Z.LAYERS_BEFORE_FIRE
        nsd, csd = dict(net.state_dict()), dict(critic.state_dict())
        for sd in (nsd, csd):
            sd["trunk.map.c1.weight"] = sd["trunk.map.c1.weight"][:, :old].clone()
        for k in [k for k in nsd if k.startswith("width.")]:
            del nsd[k]
        torch.save({"kind": T.KIND, "net": nsd, "critic": csd, "opt": opt.state_dict(), "cfg": {}, "updates": 2},
                   tmp_path / "old.pt")
        n2, c2, ck = T.restore(tmp_path / "old.pt")
        assert ck["padded"]
        assert float(n2.trunk.map.c1.weight[:, old:].abs().sum()) == 0.0
        assert float(c2.trunk.map.c1.weight[:, old:].abs().sum()) == 0.0
        assert float(n2.width.bias[Nt.WIDTH_KEEP]) == Nt.WIDTH_KEEP_BIAS and float(n2.width.weight.abs().sum()) == 0
        # the old network's decisions are unchanged: the new layers feed nothing (an archer: they are not zero)
        st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
        v, o, cm = net_inputs(st)
        assert float(v["map"][0, Z.L["reach"]].sum()) > 0
        with torch.no_grad():
            net.trunk.map.c1.weight[:, old:] = 0.0
            a, b = net.eval()(v, o, cm), n2.eval()(v, o, cm)
        for k in ("where", "face", "target", "pace"):
            assert torch.allclose(a[k], b[k]), k


class TestShooterMap:
    def test_the_cell_is_the_range_over_14_for_a_shooter_8_m_for_infantry(self):
        st = battle([(ARCHER, 0, 0, 0), (SPEAR, 60, 0, 0)], [(CLANRAT, 0, 300, 180)])
        now = V.raw(st, SP)
        cm = V.cell_size(now[0, :2], 8.0)
        rng = float(st.u["range"][0, 0])
        assert float(cm[0]) == pytest.approx(max(8.0, rng / V.RANGE_CELLS)) and float(cm[1]) == 8.0
        assert rng < float(cm[0]) * Z.CELLS / 2                     # the range circle fits the map

    def test_the_reach_circle_and_the_fire_arc_now(self):
        st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 300, 180)])
        now = V.raw(st, SP)
        rng = float(st.u["range"][0, 0])
        assert float(now[0, 0, V.I["arc"]]) == P.sim["missile"]["stand_fire_arc_deg"]
        cm = V.cell_size(now[0, :1], 8.0)
        reach, arc = Z.fire_layers(now[0, :1, V.I["range"]], now[0, :1, V.I["arc"]], cm)
        circle = math.pi * rng ** 2 / float(cm[0]) ** 2
        assert float(reach.sum()) == pytest.approx(circle, rel=0.03)
        assert float(arc.sum()) == pytest.approx(circle * 90 / 360, rel=0.08)          # +-45 deg
        assert float(arc[0, :15].sum()) > 0.95 * float(arc.sum())                      # ahead (the map's top half)
        assert float(reach[0, 15, 15]) == 1.0 and float(reach[0, 15, 0]) == 0.0

    def test_on_the_move_without_fire_on_the_move_no_arc_infantry_nothing(self):
        st = battle([(ARCHER, 0, 0, 0), (SPEAR, 60, 0, 0)], [(CLANRAT, 0, 300, 180)])
        st.u["mv"][0, 0] = True
        now = V.raw(st, SP)
        assert float(now[0, 0, V.I["arc"]]) == 0.0
        st.u["fire_move"][0, 0] = True
        assert float(V.raw(st, SP)[0, 0, V.I["arc"]]) == P.sim["missile"]["move_fire_arc_deg"]
        assert float(now[0, 1, V.I["arc"]]) == 0.0 and float(now[0, 1, V.I["range"]]) == 0.0
        who = torch.zeros((1, st.N), dtype=torch.bool)
        who[0, 1] = True
        v = V.build(now, V.Memory(1, st.N).hist, card(st), who)
        assert float(v["map"][0, Z.L["reach"]:].abs().sum()) == 0.0

    def test_a_position_smaller_than_a_cell_allows_the_cell_it_lies_in(self):
        st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 300, 180)])
        now = V.raw(st, SP)
        cm = V.cell_size(now[0, :1], 8.0)
        c = float(cm[0])
        f = lambda val: torch.full((1, st.N), float(val))       # noqa: E731
        area = {"x": f(0.3 * c), "z": f(2.2 * c), "b": f(0), "front": f(0.3 * c), "depth": f(0.3 * c)}
        who = torch.zeros((1, st.N), dtype=torch.bool)
        who[0, 0] = True
        v = V.build(now, V.Memory(1, st.N).hist, card(st), who, area=area, cell_m=cm)
        ok = v["cells_ok"][0]
        assert int(ok.sum()) == 1
        r, col = Z.cell_of(torch.tensor(0.3 * c), torch.tensor(2.2 * c), cm[0])
        assert bool(ok[int(r), int(col)])


def test_a_rollout_with_the_width_chosen_re_forms_the_unit():
    torch.manual_seed(0)
    cfg = {"battles": 2, "seconds": 20, "cell_m": 8.0, "bounds": None}
    sp = DH.specs(2, torch.Generator().manual_seed(5))
    net = Nt.UnitNet()
    with torch.no_grad():
        net.width.bias.zero_()
        net.width.bias[1] = 30.0                     # always the narrowest
        net.t_none.bias.fill_(30.0)                  # never an attack (an attack carries no width)
    setup = T.hold_setup(cfg, P, "cpu", sp=sp)
    st = setup["st"]
    lo, _ = Dc.width_limits(st, P)
    w_start = st.u["width"][:, 0].clone()
    batch, stats = T.rollout(net, None, cfg, P, "cpu", T.Duty("cpu"), setup)
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
    assert (batch["act"]["width"] == 1).all()
    goal = torch.where(st.u["wnew"] >= 0, st.u["wnew"], st.u["width"])[:, 0]
    assert bool((goal <= lo[:, 0] + 1e-3).all()) and bool((goal < w_start).all())
