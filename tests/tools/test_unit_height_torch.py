"""The unit network's ground (tools/nn/unit: view.py RAW gh / terr and the row's dh, zmap.py the height / slope_fwd /
slope_right layers, train.restore padding an older checkpoint): exact numbers from the battle's height map
(tools/nn/sim/terrain.py), zeros on a flat battle. Needs torch (the container)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_unit_view_net_torch import ARCHER, CLANRAT, SPEAR, battle, net_inputs, view_of  # noqa: E402
from tools.nn.sim import terrain  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
GX = np.arange(-400.0, 401.0, 2.0)


def plane(ax=0.0, az=0.0, c=0.0):
    Zg, Xg = np.meshgrid(GX, GX, indexing="ij")
    return terrain.register((ax * Xg + az * Zg + c).astype(np.float32), -400.0, -400.0, 2.0)


def test_flat_battle_zeros_and_the_raw_ground():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
    v = view_of(st)
    for name in ("height", "slope_fwd", "slope_right"):
        assert float(v["map"][0, Z.L[name]].abs().sum()) == 0
    assert float(v["rows"][0, :, V.ROW.index("dh")].abs().sum()) == 0
    raw = V.raw(st, SP)
    assert raw[0, :, V.I["terr"]].tolist() == [-1.0] * st.N and float(raw[0, :, V.I["gh"]].abs().sum()) == 0
    terrain.set_map(st, plane(0.0, 0.1, 200.0))
    raw = V.raw(st, SP)
    H = st.N // 2
    assert float(raw[0, 0, V.I["gh"]]) == pytest.approx(200.0, abs=1e-3)
    assert float(raw[0, H, V.I["gh"]]) == pytest.approx(210.0, abs=1e-3)


def test_the_rows_exact_height_difference():
    st = battle([(SPEAR, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
    terrain.set_map(st, plane(0.0, 0.1, 200.0))
    v = view_of(st)
    dh = v["rows"][0, :, V.ROW.index("dh")]
    assert float(dh[0]) == 0.0 and float(dh[1]) == pytest.approx(10.0 / V.DH_M, abs=1e-4)


@pytest.mark.parametrize("b, fwd, right", [(0.0, 0.1, 0.0), (90.0, 0.0, -0.1), (180.0, -0.1, 0.0)])
def test_the_height_and_slope_layers_in_the_units_frame(b, fwd, right):
    # the ground rises 0.1 a metre to +z; the unit faces b
    st = battle([(SPEAR, 0, 0, b)], [(CLANRAT, 300, 300, 180)])
    terrain.set_map(st, plane(0.0, 0.1, 50.0))
    m = view_of(st)["map"][0]
    assert torch.allclose(m[Z.L["slope_fwd"]], torch.full((Z.CELLS, Z.CELLS), fwd), atol=1e-4)
    assert torch.allclose(m[Z.L["slope_right"]], torch.full((Z.CELLS, Z.CELLS), right), atol=1e-4)
    # the top row is the farthest forward: (14.5 cells x 8 m) x the rise ahead, / HEIGHT_M
    top = float(m[Z.L["height"], 0].mean())
    assert top == pytest.approx(14.5 * 8.0 * fwd / Z.HEIGHT_M, abs=1e-3)


def test_an_old_checkpoint_gets_zero_ground_inputs(tmp_path):
    torch.manual_seed(5)
    net, critic = Nt.UnitNet(), Nt.Critic()
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()))
    old_l, old_r = Z.LAYERS_BEFORE_HEIGHT, len(V.ROW) - 1
    nsd, csd = dict(net.state_dict()), dict(critic.state_dict())
    for sd in (nsd, csd):
        sd["trunk.map.c1.weight"] = sd["trunk.map.c1.weight"][:, :old_l].clone()
        sd["trunk.rows.state.weight"] = sd["trunk.rows.state.weight"][:, :old_r].clone()
        g = sd["trunk.memory.gru.weight_ih_l0"]
        sd["trunk.memory.gru.weight_ih_l0"] = torch.cat([g[:, :old_r], g[:, -1:]], 1).clone()
    torch.save({"kind": T.KIND, "net": nsd, "critic": csd, "opt": opt.state_dict(), "cfg": {}, "updates": 3},
               tmp_path / "old.pt")
    n2, c2, ck = T.restore(tmp_path / "old.pt")
    assert ck["padded"]
    for m in (n2, c2):
        assert float(m.trunk.map.c1.weight[:, old_l:].abs().sum()) == 0
        assert float(m.trunk.rows.state.weight[:, old_r:].abs().sum()) == 0
        assert float(m.trunk.memory.gru.weight_ih_l0[:, old_r:-1].abs().sum()) == 0
        assert torch.equal(m.trunk.memory.gru.weight_ih_l0[:, -1], (nsd if m is n2 else csd)[
            "trunk.memory.gru.weight_ih_l0"][:, -1])
    # the old network's decisions on a hill are its flat ones: the ground's inputs feed nothing
    st = battle([(ARCHER, 0, 0, 0)], [(CLANRAT, 0, 100, 180)])
    v0, o, cm = net_inputs(st)
    terrain.set_map(st, plane(0.05, 0.1, 30.0))
    v1, _, _ = net_inputs(st)
    assert float(v1["map"][0, Z.L["height"]].abs().sum()) > 0
    with torch.no_grad():
        a, b = n2.eval()(v0, o, cm), n2.eval()(v1, o, cm)
    for k in ("where", "face", "target", "pace"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k
