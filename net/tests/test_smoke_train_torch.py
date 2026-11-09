"""One training step a level (the smoke test of net): a short batch of the level's own battles with a fresh network, one
update that changes its weights, the checkpoint saved and read back through <level>/ckpt.py (the bridge's way in) with
the same weights. Needs torch (the container, CPU); the levels above the unit run the frozen unit network
(net/config/tactic.json train.unit_ckpt -> weights/unit/..., net/weights.json)."""
import pytest

torch = pytest.importorskip("torch")

from sim.params import load  # noqa: E402

P = load()


def _params(net):
    return [p.detach().clone() for p in net.parameters() if p.requires_grad]


def _changed(before, net):
    return any(not torch.equal(a, b) for a, b in zip(before, [p for p in net.parameters() if p.requires_grad]))


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture(scope="module")
def unit_net():
    from net.tactic import ckpt as TK
    return TK.unit_network("cpu")


def test_the_unit_level_trains_a_step(tmp_path):
    from net.unit import ckpt as UC
    from net.unit import train as T
    log, same = T.main(["--drill", "hold", "--battles", "6", "--seconds", "20", "--updates", "1",
                        "--minibatch", "256", "--out", str(tmp_path)])
    assert len(log) == 1 and log[0]["samples"] > 0 and same            # (main: the checkpoint's output bit for bit)
    net, _, ck = UC.restore(tmp_path / "latest.pt")
    assert ck["kind"] == UC.KIND and ck["updates"] == 1 and not ck["padded"]
    assert not any(p.requires_grad for p in UC.frozen(tmp_path / "latest.pt").parameters())


def test_the_tactician_trains_a_step(tmp_path, monkeypatch, unit_net):
    from net.tactic import cfg
    from net.tactic import ckpt as TK
    from net.tactic import decide as DC
    from net.tactic import drill1 as DR
    from net.tactic import env as EN
    from net.tactic import train as TT
    monkeypatch.setitem(cfg()["drill1"], "limit_s", 20)
    torch.manual_seed(0)
    net, critic = DC.TacticNet(), DC.TacticCritic()
    setup = EN.Setup(DR.specs(2, torch.Generator().manual_seed(3)), P, "cpu")
    batch, rs, _ = EN.play(setup, unit_net, net=net, critic=critic)
    assert rs["decisions"] > 0
    pa, pc = TT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    tc = dict(cfg()["train"], minibatch=8, epochs=1)
    before = _params(net)
    us = TT.update(net, critic, opt, batch, tc, "cpu", None, 1.0)
    assert us["samples"] > 0 and _changed(before, net)
    TK.save(tmp_path / "t.pt", net, critic, opt, tc, 1)
    n2, c2, ck = TK.restore(tmp_path / "t.pt")
    assert ck["kind"] == TK.KIND and ck["updates"] == 1 and _same(net, n2) and _same(critic, c2)


def test_the_operative_trains_a_step(tmp_path, unit_net):
    from net.operative import cases as CS
    from net.operative import cfg
    from net.operative import ckpt as OK
    from net.operative import env as EN
    from net.operative import net as NN
    from net.operative import train as OT
    torch.manual_seed(0)
    net, critic = NN.OperativeNet(), NN.OperativeCritic()
    s = CS.Setup(CS.specs(3, torch.Generator().manual_seed(0), fixed=True), P, "cpu")
    s.limit = torch.full_like(s.limit, 31.0)
    batch, rs, _ = EN.play(s, unit_net, net=net, critic=critic)
    assert rs["parcel_errors"] == 0 and rs["parcels"] >= 3
    pa, pc = OT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    tc = dict(cfg()["train"], minibatch=8, epochs=1)
    before = _params(net)
    us = OT.update(net, critic, opt, batch, tc, "cpu", None, 1.0)
    assert us["samples"] > 0 and _changed(before, net)
    OK.save(tmp_path / "o.pt", net, critic, opt, tc, 1)
    n2, c2, ck = OK.restore(tmp_path / "o.pt")
    assert ck["kind"] == OK.KIND and _same(net, n2) and _same(critic, c2)


def test_the_strategist_trains_a_step(tmp_path, unit_net):
    from net.strategist import battle as BT
    from net.strategist import cases as CS
    from net.strategist import cfg
    from net.strategist import ckpt as SK
    from net.strategist import net as NN
    from net.strategist import train as ST
    torch.manual_seed(0)
    net, critic = NN.StrategistNet(), NN.StrategistCritic()
    tc = cfg()["train"]
    pre = CS.Pre(CS.specs(2, torch.Generator().manual_seed(31), cases=tuple(tc["cases"])), P, "cpu")
    pre.limit = torch.full_like(pre.limit, 20.0)
    inp, ctx = BT.inputs_of(pre)
    with torch.no_grad():
        o = net(inp, ctx)
    rd, rs = BT.play(pre, o, inp, unit_net, op_net=None)
    assert rs["parcel_errors"] == 0 and rs["deploy_errors"] == 0
    rw = cfg()["reward"]
    ret = float(rw["measure"]) * BT.measure_of(pre, rd) + float(rw["trade"]) * rd["trade"]
    params = [p for p in net.parameters() if p.requires_grad] + [p for p in critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(tc["lr"]))
    before = _params(net)
    ST.update(net, critic, opt, inp, ctx, o["act"], o["lp"].detach(), ret, tc, 1.0)
    assert _changed(before, net)
    SK.save(tmp_path / "s.pt", net, critic, opt, 1, 0)
    n2, c2, ck = SK.restore(tmp_path / "s.pt")
    assert ck["kind"] == SK.KIND and _same(net, n2) and _same(critic, c2)
