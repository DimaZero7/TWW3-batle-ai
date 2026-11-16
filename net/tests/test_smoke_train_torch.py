"""One training step a level with a network (the smoke test of net): a short batch of the level's own battles with a
fresh network, one update that changes its weights, the checkpoint saved and read back through <level>/ckpt.py (the
bridge's way in) with the same weights. Needs torch (the container, CPU); the levels run the frozen unit network
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


def test_the_tactic_pointer_learns_a_step(tmp_path):
    """Drill T0-2's network (tactic/heads.TacticNet2 as pin.NetPlayer) drives two short pilot battles through the side
    wrapper, TEACH labels the same views (tags.Record); one update on those ticks with the walk fed with the
    labeller's choices changes the weights; the tactic_v2 checkpoint reads back the same."""
    from net.stage0 import battle as BA
    from net.tactic import ckpt as TK
    from net.tactic import exam_pin as EP
    from net.tactic import heads as HD
    from net.tactic import pin as PN
    from net.tactic import tags as TG
    torch.manual_seed(0)
    net = HD.TacticNet2()
    got, calls = [], []

    def hook(v, lab, T):
        calls.append(int(T["head_free"].sum()))
        if bool(T["head_free"].any()):
            got.append((v, T))

    def make():
        rec = TG.Record(PN.NetPlayer(net), PN.player("TEACH"), hook=hook)
        rec.needs_input = True
        return rec
    BA.play_tactic(EP.pilot_specs(2, 0, limit=160), make, P, "cpu", chunk=2, compile=False)
    assert got, calls
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    before = _params(net)
    total = 0.0
    for v, T in got:
        label = torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1))
        o = net.act(v, greedy=True, given=label)
        w = (T["head_free"] & ~T["dropped"] & ~T["dc_label"]).float()
        total = total + HD.loss(o["logits"], label, w, T["dc_keys"])[0]
    opt.zero_grad()
    total.backward()
    opt.step()
    assert _changed(before, net)
    TK.save_v2(tmp_path / "t2.pt", net, opt, ["T0-2"])
    n2, c2, ck = TK.restore(tmp_path / "t2.pt")
    assert ck["kind"] == TK.KIND2 and c2 is None and ck["row_width"] == 185 and _same(net, n2)


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


def test_the_operatives_memory_trains_a_step():
    """net/operative/memory.py (a prototype, not in the network yet): a short battle seen every second, two decisions'
    records; the groups' sequences through the time attention and the units' summary through its block, one step."""
    from net.operative import frame as FR
    from net.operative import memory as MM
    from net.tactic import raw as TR
    from sim import battle, scenario
    from sim import scripts as opponents
    torch.manual_seed(0)

    def line(keys, z, b):
        return [{"key": k, "x": -30.0 + 30.0 * i, "z": z, "b": b} for i, k in enumerate(keys)]
    spec = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": line(["wh_main_emp_inf_spearmen_0"] * 2, -60.0, 0.0)},
        2: {"faction": "wh2_main_skv_skaven", "units": line(["wh2_main_skv_inf_clanrat_spearmen_0"] * 2, 60.0, 180.0)}}}
    st = scenario.build([spec], P)
    seer, store, axes = TR.Seer(P), MM.Store(1, st.N, 1, T=4), None
    for sec in range(1, 31):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
        raw = seer.raw(st)
        axes = axes or FR.Axes.of(raw, torch.ones(1, dtype=torch.long), st.bounds)
        store.see(raw, float(sec))
        if sec % 15 == 0:
            store.tick(float(sec), axes)
    members = torch.stack([store.enemy == 0, store.enemy], 1) & (raw[..., TR.I["side"]] > 0)[:, None]
    seq, mask, age = store.sequence(members)
    att, summ = MM.TimeAttention(), MM.Summary()
    G = members.shape[1]
    out = att(seq.flatten(0, 1), mask.flatten(0, 1), age.flatten(0, 1))
    out2 = summ(store.units()[0][:, :, 0].flatten(0, 1), store.summary(30.0, axes).flatten(0, 1))
    assert out.shape == (G, 64) and out2.shape == (st.N, 64) and bool(mask[:, :, :2].all())
    opt = torch.optim.Adam(list(att.parameters()) + list(summ.parameters()), lr=1e-3)
    before = _params(att) + _params(summ)
    (out.pow(2).mean() + out2.pow(2).mean()).backward()
    opt.step()
    assert _changed(before[:len(_params(att))], att) and _changed(before[len(_params(att)):], summ)


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
