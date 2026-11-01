"""The tactician's first drill in the simulator (tools/nn/tactic drill1, env, exam1, train; design/tactic "Первое
упражнение"): the battle's table, the enemy's script, the contact's reading, a random network through a whole battle, the
update's choices read back, the unit's parcel. Needs torch (the container), a few minutes on the CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import cfg  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import drill1 as DR  # noqa: E402
from tools.nn.tactic import env as EN  # noqa: E402
from tools.nn.tactic import exam1 as EX  # noqa: E402
from tools.nn.tactic import parcel as PC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import train as TT  # noqa: E402
from tools.nn.tactic import uzone as U  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()


def sp(n=6, seed=3, fixed=list(DR.STARTS)):                      # noqa: B006
    return DR.specs(n, torch.Generator().manual_seed(seed), fixed=fixed)


def test_roster_is_lobbing_front_firing_shooters_and_infantry():
    """The base: the shooters arc over friends (direct false, dual_low_fixed), fire ahead only; the infantry is melee."""
    pp = DH._passports()
    for side in cfg()["drill1"]["roster"].values():
        for k in side["shooters"]:
            m = pp[k]["missile"]
            assert m["trajectory"] == "dual_low_fixed" and not m["direct"], k
            assert m["fire_arc_deg"] <= 45.0, k
            assert pp[k]["caste"] == "missile_infantry", k
        for k in side["infantry"]:
            assert pp[k]["caste"] == "melee_infantry" and not pp[k].get("missile"), k


def test_battle_table_and_correct_parcels():
    s = sp(40, fixed=None)
    assert {x["start"] for x in s} == set(DR.STARTS)
    c = cfg()["drill1"]
    st, parcel, order, own, shooter, foe = DR.build(s, P, "cpu")          # (build checks the parcels: PC.ensure)
    assert st.N == 6 and own[:, :3].all() and foe[:, 3:].all()
    assert shooter.sum(1).eq(1).all()
    assert order[0].tolist() == [0, 1, 0, 0, 1, 0, 0]                     # 'hold', 'any'
    assert parcel.target[:, 3:].all() and not parcel.target[:, :3].any()
    assert parcel.line.eq(False).all() and (parcel.mine.sum(1) == 3).all()
    for b, x in enumerate(s):
        ex, ez = DR.enemy_centre(x)
        zx, zz = float(parcel.cx[b]), float(parcel.cz[b])
        assert c["shift_m"][0] - 1e-3 <= math.hypot(zx, zz) <= c["shift_m"][1] + 1e-3      # the zone shifted from the group
        assert c["distance_m"][0] - 1e-3 <= math.hypot(ex - zx, ez - zz) <= c["distance_m"][1] + 1e-3
        fx, fz = Z.local(st.u["x"][b, 3:].mean(), st.u["z"][b, 3:].mean(), parcel.cx[b], parcel.cz[b], parcel.heading()[b])
        assert abs(float(fx)) < 1.0 and float(fz) > 0                     # turned at the enemy


def test_starts_put_the_right_unit_in_front():
    for start in DR.STARTS:
        x = sp(1, fixed=[start])[0]
        st, _, _, _, shooter, _ = DR.build([x], P, "cpu")
        b0 = DR.facing(x)
        _, f = Z.local(st.u["x"][0, :3], st.u["z"][0, :3], 0.0, 0.0, b0)
        ahead = float(f[shooter[0, :3]].mean()) > float(f[~shooter[0, :3]].mean())
        assert ahead == (start == "shooters_front")


def test_script_walks_at_the_nearest_and_looks_every_5_s():
    s = sp(2)
    st, _, _, own, _, _ = DR.build(s, P, "cpu")
    script = DR.Script()
    o = script(st)
    assert (o.kind[:, 3:] == O.ATTACK).all() and not o.run[:, 3:].any()
    same = script(st)
    assert same is o
    st.t = st.t + 5.0
    assert script(st) is not o


def test_segment_crosses_a_turned_rectangle():
    t = lambda v: torch.tensor(float(v))                          # noqa: E731
    args = (t(0), t(0), t(0), t(40), t(10))                        # at the origin, 40 m wide, 10 m deep
    assert DR.crosses(t(0), t(50), t(0), t(-50), *args)
    assert DR.crosses(t(10), t(50), t(-10), t(-50), *args)
    assert not DR.crosses(t(30), t(50), t(30), t(-50), *args)
    assert not DR.crosses(t(0), t(50), t(0), t(10), *args)         # stops before it
    turned = (t(0), t(0), t(90), t(40), t(10))                     # now 40 m along z, 10 m across x
    assert not DR.crosses(t(30), t(0), t(10), t(0), *turned)
    assert DR.crosses(t(30), t(0), t(-10), t(0), *turned)


def fake_touch(st, pairs):
    B, N = st.B, st.N
    hs = torch.zeros((B, N, N))
    for b, i, j in pairs:
        hs[b, i, j] = 1.0
    st.u["hit_sec"] = hs.view(B, N * N)


def test_watch_between_and_hit():
    """Four cases at the first contact: the infantry between (+), a touch on the shooters (hit, not between), the
    infantry touched but the shooters to one side of it (not between, no hit), no contact yet."""
    s = sp(4, fixed=["infantry_front"])
    st, _, _, own, shooter, _ = DR.build(s, P, "cpu")
    u = st.u
    inf = (own & ~shooter)
    for b in range(4):
        sl = int(shooter[b].float().argmax())
        ii = inf[b].nonzero()[:, 0].tolist()
        # a line along z: the enemy at z = 100, the infantry at 80, the shooters at 0
        u["x"][b, 3:], u["z"][b, 3:] = 0.0, 100.0
        u["b"][b, :] = 0.0
        u["x"][b, ii[0]], u["z"][b, ii[0]] = 0.0, 80.0
        u["x"][b, ii[1]], u["z"][b, ii[1]] = 500.0, 80.0
        u["x"][b, sl], u["z"][b, sl] = 0.0, 0.0
    # battle 2: the shooters far to the side: the line misses the infantry
    sl = int(shooter[2].float().argmax())
    u["x"][2, sl], u["z"][2, sl] = 200.0, 100.0
    w = DR.Watch(st, own, shooter, P.sim["formation"]["spacing_m"], "cpu")
    i0 = int(inf[0].nonzero()[0, 0])
    i2 = int(inf[2].nonzero()[0, 0])
    fake_touch(st, [(0, i0, 3), (1, int(shooter[1].float().argmax()), 3), (2, i2, 3)])
    new, got, hit = w.step(st, torch.zeros(4, dtype=torch.bool), 0.5, None)
    assert new.tolist() == [True, True, True, False]
    assert got.tolist() == [True, False, False, False]
    assert hit.tolist() == [False, True, False, False]


def test_ender_fight_time_after_contact():
    s = sp(2)
    st, _, _, own, _, _ = DR.build(s, P, "cpu")
    e = DR.Ender(st, own)
    assert not e.after_step(st).any()
    st.t = st.t + 10.0
    tc = torch.tensor([5.0, -1.0])
    st.t = st.t + cfg()["drill1"]["fight_s"]
    new = e.after_step(st, tc)
    assert new.tolist() == [True, False]
    assert e.why[0].item() == 3


def test_unit_area_shooter_sector_and_no_target():
    s = sp(2)
    setup = EN.Setup(s, P, "cpu")
    area = EN.fresh_area(2, 6, setup.own, "cpu")
    t = EN.ZN.Tiles.of(setup.parcel)
    f = {"s": torch.tensor([[-40.0, 0.0, 40.0]] * 2), "f": torch.tensor([[37.3, -37.3, 37.3]] * 2),
         "hw": torch.tensor([[40.0, 32.0, 40.0]] * 2), "hd": torch.full((2, 3), 24.0),
         "on": torch.ones((2, 3), dtype=torch.bool)}
    EN.unit_area(area, f, t, setup.my, setup.shooter)
    # the sector: only on the shooters, forward along the zone
    assert ((area["sw"] > 0) == setup.shooter).all()
    sl = setup.shooter[0].float().argmax()
    assert area["sb"][0, sl].item() == pytest.approx(float(t.heading[0]), abs=1e-3)
    assert area["sw"][0, sl].item() == cfg()["sector_deg"]
    # the frame back in the zone's axes
    s_, f_ = Z.local(area["x"][0, 0], area["z"][0, 0], t.cx[0], t.cz[0], t.heading[0])
    assert s_.item() == pytest.approx(-40.0, abs=1e-2) and f_.item() == pytest.approx(37.3, abs=1e-2)
    assert area["front"][0, 0].item() == 80.0 and area["depth"][0, 0].item() == 48.0


def test_tile_keeper_drop_and_rally():
    s = sp(1)
    setup = EN.Setup(s, P, "cpu")
    keep = EN.TileKeeper(setup.parcel, setup.my)
    seer = TR.Seer(P)
    raw = seer.raw(setup.st).clone()
    t = keep.update(raw)
    assert t.mask[0].sum() == 4
    raw[0, 0, TR.I["r"]] = 1.0                       # slot 0 routs
    t = keep.update(raw)
    assert t.mask[0].sum() == 3
    t = keep.update(raw)
    assert t.mask[0].sum() == 3                      # once
    raw[0, 0, TR.I["r"]] = 0.0                       # it rallies: the tile is back
    t = keep.update(raw)
    assert t.mask[0].sum() == 4


@pytest.fixture(scope="module")
def unit_net():
    return TT.unit_network("cpu")


def _short(monkeypatch, limit=40):
    monkeypatch.setitem(cfg()["drill1"], "limit_s", limit)


def test_random_network_plays_a_whole_battle(unit_net, monkeypatch):
    """A random network through a battle: no error, every frame inside the zone, none overlap; the update's walk reads back
    the same log-probabilities (the raw numbers stored are enough); the unit reaches for the frames; the personal areas'
    layer holds the last output."""
    _short(monkeypatch, 40)
    torch.manual_seed(0)
    net, critic = DC.TacticNet(), DC.TacticCritic()
    setup = EN.Setup(sp(4), P, "cpu")
    b, rs, rd = EN.play(setup, unit_net, net=net, critic=critic)
    assert rs["frames_overlap"] == 0 and rs["frames_out"] == 0
    assert rs["decisions"] > 0
    di, bi = b["valid"].nonzero(as_tuple=True)
    inp, o, p, t, act, prev = EN.rebuild(b, di, bi)
    res, _, _ = net(inp, o, p, t, prev=prev, act=act)
    assert (res["lp"] - b["lp"][di, bi]).abs().max() < 1e-3
    # the areas' layer: empty at the first decision, filled at the later ones
    first = di == 0
    li = EN.IN.W.near_layers().index("areas")
    assert inp["near"][first, li].abs().sum() == 0
    assert inp["near"][~first, li].sum((1, 2)).min() > 0
    # one training step goes through
    pa, pc = TT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    tc = dict(cfg()["train"], minibatch=8, epochs=1)
    before = next(iter(net.heads.where_a.parameters())).clone()
    us = TT.update(net, critic, opt, b, tc, "cpu", None, 1.0)
    assert us["samples"] > 0
    assert not torch.equal(before, next(iter(net.heads.where_a.parameters())))


def test_formula_drives_and_frozen_part_stays(unit_net, monkeypatch):
    _short(monkeypatch, 15)
    setup = EN.Setup(sp(2), P, "cpu")
    _, rs, rd = EN.play(setup, unit_net, formula=True, keep=False)
    assert rs["frames_overlap"] == 0 and rs["frames_out"] == 0 and rs["decisions"] > 0
    assert rs["changes_per_battle"] < 1.0                           # the formula does not fidget


def test_exam_table_reads_both_starts(unit_net, monkeypatch):
    _short(monkeypatch, 15)
    t = EX.run(unit_net, P, "cpu", None, True, n=4)
    assert t["who"] == "formula" and t["shooters_front"]["n"] == 2 and t["infantry_front"]["n"] == 2
    for k in ("between", "contact", "hit", "shooters_in_melee", "shooters_fired_s", "ammo_used", "trade"):
        assert k in t["all"]


def test_zone_changes_cost_and_changed_rule():
    a = {"s": torch.tensor([[0.0, 0.0]]), "f": torch.tensor([[0.0, 0.0]]), "hw": torch.tensor([[24.0, 24.0]]),
         "hd": torch.tensor([[24.0, 24.0]]), "on": torch.tensor([[True, True]])}
    b = {k: v.clone() for k, v in a.items()}
    assert not U.changed(b, a).any()
    b["s"][0, 0] = 10.0
    assert not U.changed(b, a).any()                  # under change_m
    b["s"][0, 0] = 20.0
    assert U.changed(b, a)[0].tolist() == [True, False]
    b = {k: v.clone() for k, v in a.items()}
    b["hw"][0, 1] = 32.0
    assert U.changed(b, a)[0].tolist() == [False, True]


def test_the_same_frame_through_change_costs_as_keep(unit_net, monkeypatch):
    """A change is paid only when the frame really changed and the network chose it: a keeper pays 0, a unit that chose
    'change' and landed on the same frame (within change_m) pays 0, not less."""
    a = {"s": torch.tensor([[0.0, 0.0]]), "f": torch.tensor([[0.0, 0.0]]), "hw": torch.tensor([[24.0, 24.0]]),
         "hd": torch.tensor([[24.0, 24.0]]), "on": torch.tensor([[True, True]])}
    same = {k: v.clone() for k, v in a.items()}
    same["s"][0, 0] = 5.0
    assert not U.changed(same, a).any()                    # the same frame: nothing to pay, as for a keeper
    assert not U.changed(a, a).any()
    # the cost in a battle: all keep from the second decision: only the first decision's frames, no cost at all
    _short(monkeypatch, 20)
    net = DC.TacticNet()
    with torch.no_grad():
        net.heads.keep.weight.zero_()
        net.heads.keep.bias[DC.KEEP] = 50.0
    setup = EN.Setup(sp(4), P, "cpu")
    b, rs, rd = EN.play(setup, unit_net, net=net)
    assert rs["changes"] == 0 and rs["keep_share"] == 1.0
    assert b["reward"].abs().sum() < 1e-3                  # (no contact in 20 s, no end but the trade 0)
    # always change: paid per real change
    with torch.no_grad():
        net.heads.keep.bias[DC.KEEP] = -50.0
    setup = EN.Setup(sp(4), P, "cpu")
    b, rs, _ = EN.play(setup, unit_net, net=net)
    assert rs["changes"] > 0 and rs["keep_share"] == 0.0
    assert float(b["reward"].sum()) == pytest.approx(-rs["changes"] * cfg()["reward"]["change"], abs=0.3)


def test_old_checkpoint_gets_the_keep_head(tmp_path):
    """A checkpoint from before the head reads with the head at its start; the rest of the weights as they were."""
    net, critic = DC.TacticNet(), DC.TacticCritic()
    pa, pc = TT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    p = tmp_path / "old.pt"
    TT.save(p, net, critic, opt, {}, 21)
    ck = torch.load(p, weights_only=False)
    for k in [k for k in ck["net"] if k.startswith("heads.keep")]:
        del ck["net"][k]
    torch.save(ck, p)
    n2, _, ck2 = TT.restore(p)
    assert ck2["padded"] and ck2["updates"] == 21
    assert n2.heads.keep.weight.abs().sum() == 0
    assert n2.heads.keep.bias[DC.KEEP].item() == cfg()["zone_head"]["keep_bias"]
    assert torch.equal(n2.heads.where_a.weight, net.heads.where_a.weight)


def test_checkpoint_reads_back(tmp_path):
    net, critic = DC.TacticNet(), DC.TacticCritic()
    pa, pc = TT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=1e-4)
    p = tmp_path / "x.pt"
    TT.save(p, net, critic, opt, {}, 3)
    n2, c2, ck = TT.restore(p)
    assert ck["updates"] == 3 and ck["kind"] == TT.KIND and not ck["padded"]
    for a, b in zip(net.state_dict().values(), n2.state_dict().values()):
        assert torch.equal(a, b)
    assert all(not q.requires_grad for q in n2.enc.inp.analysis.parameters())


def test_gpu_lock_only_on_the_gpu():
    """The chain's lock on a CUDA device (not entered here), nothing on the CPU."""
    import contextlib
    assert isinstance(TT.lock("cpu", "x"), contextlib.nullcontext)
    assert hasattr(TT.lock("cuda", "x"), "__enter__")


def test_rows_k_is_all_slots():
    """The drill's K = N rows give the same answers as 32: the rows past N are empty under the mask."""
    assert EN.rows_of(6) == 6
    assert EN.rows_of(100) == cfg()["rows"]
    torch.manual_seed(1)
    net = DC.TacticNet().eval()
    setup = EN.Setup(sp(2), P, "cpu")
    raw = TR.Seer(P).raw(setup.st)
    t = EN.ZN.Tiles.of(setup.parcel)
    with torch.no_grad():
        a = net.enc(EN.IN.build(raw, setup.card, setup.parcel, t, None, bounds=setup.st.bounds, rows=6), setup.order)
        b = net.enc(EN.IN.build(raw, setup.card, setup.parcel, t, None, bounds=setup.st.bounds, rows=32), setup.order)
    assert b["cards"].shape[1] == 32 and a["cards"].shape[1] == 6
    assert torch.allclose(a["cards"], b["cards"][:, :6], atol=1e-4)
    assert torch.allclose(a["map"], b["map"], atol=1e-5)
    assert PC.Parcel is not None and V.T == 10
