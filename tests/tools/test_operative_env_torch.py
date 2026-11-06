"""The operative's battle (tools/nn/operative/env.py, lower.py, watch.py): a short batch plays to its end with the
network and with the formula, every parcel passes parcel.check, the stored decisions rebuild to the same
log-probability, the readings are in their ranges. Needs torch (the container; the simulator on the CPU)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.operative import cases as CS  # noqa: E402
from tools.nn.operative import env as EN  # noqa: E402
from tools.nn.operative import net as NN  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import train as TT  # noqa: E402

P = load()


@pytest.fixture(scope="module")
def unit_net():
    return TT.unit_network("cpu")


def short(n=3, limit=40.0, seed=0):
    sp = CS.specs(n, torch.Generator().manual_seed(seed), fixed=True)
    s = CS.Setup(sp, P, "cpu")
    s.limit = torch.full_like(s.limit, limit)
    return s


def test_the_network_plays_stores_and_rebuilds(unit_net):
    torch.manual_seed(0)
    net, critic = NN.OperativeNet(), NN.OperativeCritic()
    s = short()
    batch, rs, rd = EN.play(s, unit_net, net=net, critic=critic)
    assert rs["parcel_errors"] == 0 and rs["parcels"] >= 3
    D = batch["raw"].shape[0]
    assert D == 3                                           # decisions at 0, 15, 30 s
    di, bi = batch["valid"].nonzero(as_tuple=True)
    inp, ctx, act = EN.rebuild(batch, di, bi)
    with torch.no_grad():
        again, _ = net(inp, ctx, act=act)
    assert torch.allclose(again["lp"], batch["lp"][di, bi], atol=1e-3)
    assert float(rd["between"].min()) >= 0 and float(rd["between"].max()) <= 1


def test_the_formula_plays_without_errors(unit_net):
    s = short(limit=20.0, seed=1)
    _, rs, rd = EN.play(s, unit_net, mode="formula", keep=False)
    assert rs["parcel_errors"] == 0
    assert rs["groups_mean"] >= 1.0
    m = EN.measures(s, rd)
    assert set(m) == set(CS.CASES)
