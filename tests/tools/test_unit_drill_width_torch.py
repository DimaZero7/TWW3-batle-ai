"""Drill U9 'the width for the task' (tools/nn/unit/drill_width.py, exam_width.py, train.py width_setup / the mix): the
area's front and our line within their ranges, one enemy or a line of two side by side ahead, the scripts' width order
(and none for 'keep'), the exam's watch (the line beyond the area, re-forms, the men striking), a short rollout and
update. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle as Bt  # noqa: E402
from tools.nn.sim import geometry  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import decode as Dc  # noqa: E402
from tools.nn.unit import drill_width as DW  # noqa: E402
from tools.nn.unit import exam_width as EW  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
CFG = {"battles": 4, "seconds": 20, "cell_m": 8.0, "bounds": None}


def test_the_area_our_line_and_the_enemies():
    sp = DW.specs(60, torch.Generator().manual_seed(3))
    assert {s["kind"] for s in sp} == set(DW.KINDS)
    st, own, area = DW.build(sp, P, "cpu")
    H = st.N // 2
    front, _ = geometry.dims(st.u, SP)
    for b, s in enumerate(sp):
        lo, hi = DW.width_limits(s["me"])
        assert DW.AREA_FRONT_M[0] <= float(area["front"][b, 0]) <= DW.AREA_FRONT_M[1]
        assert DW.AREA_DEPTH_M[0] <= float(area["depth"][b, 0]) <= DW.AREA_DEPTH_M[1]
        # our line: the width wanted (snapped to whole files), within the unit's limits; our centre in the area
        assert lo - 1e-3 <= s["width"] <= hi + 1e-3 and abs(float(front[b, 0]) - s["width"]) <= 2.0
        assert float(Dc.signed_dist(st.u["x"][b, 0], st.u["z"][b, 0], {k: v[b, 0] for k, v in area.items()})) > 0
        n = 1 if s["kind"] == "one" else 2
        assert int((st.u["men0"][b, H:] > 0).sum()) == n
        for k in range(n):
            # ahead of the area's front, facing us, at the gap wanted, with its own width
            dx, dz = float(st.u["x"][b, H + k] - st.u["x"][b, 0]), float(st.u["z"][b, H + k] - st.u["z"][b, 0])
            ahead = dx * math.sin(math.radians(s["b"])) + dz * math.cos(math.radians(s["b"]))
            assert ahead >= DW.GAP_M[0] / 2                                 # ahead of the area's front
            face = (math.degrees(math.atan2(-dx, -dz)) - float(st.u["b"][b, H + k]) + 180) % 360 - 180
            assert abs(face) <= 50.0                                      # turned towards us (a line: its own front)
            assert abs(float(front[b, H + k]) - s["foe_widths"][k]) <= 2.0
        if n == 2:
            # side by side: the centres (wA + wB) / 2 + the gap apart across the way, at about the same distance
            a = math.radians(s["b"] + s["bearing"])
            ax = lambda j: float(st.u["x"][b, j]) * math.cos(a) - float(st.u["z"][b, j]) * math.sin(a)  # noqa: E731
            across = abs(ax(H) - ax(H + 1))
            assert across == pytest.approx((s["foe_widths"][0] + s["foe_widths"][1]) / 2 + s["line_gap"], abs=0.1)
            assert DW.enemy_span(s) == pytest.approx(sum(s["foe_widths"]) + s["line_gap"])
    ex = DW.exam_specs()
    assert len(ex) == 240 and [s["kind"] for s in ex[:4]] == ["one", "two", "one", "two"]
    assert ex[0] == DW.exam_specs()[0]


def test_the_scripts_width_order_and_none_for_keep():
    sp = DW.specs(4, torch.Generator().manual_seed(4), fixed=["two"])
    st, own, area = DW.build(sp, P, "cpu")
    span = torch.tensor([DW.enemy_span(s) for s in sp])
    wl = Dc.width_limits(st, P)
    for name in EW.SCRIPTS:
        o = EW.script_orders(st, name, span, area, wl, 0.0)
        if name == "keep":
            assert bool((o.kind[:, 0] == O.ATTACK).all()) and bool((o.width[:, 0] < 0).all())
            continue
        want = {"match": span, "area": area["front"][:, 0], "match_area": torch.minimum(span, area["front"][:, 0])}[name]
        assert bool((o.kind[:, 0] == O.HOLD).all())
        assert torch.allclose(o.width[:, 0], torch.minimum(torch.maximum(want, wl[0][:, 0]), wl[1][:, 0]))
    # after START_S: the attack, no width
    st.t[:] = EW.START_S
    o = EW.script_orders(st, "match", span, area, wl, 0.0)
    assert bool((o.kind[:, 0] == O.ATTACK).all())


def _watch_run(st, area, o, steps):
    with EW.Strikers() as sk:
        w = EW.Watch(st.B, st.N, P, "cpu", sk)
        for _ in range(steps):
            was = st.done.clone()
            hp0 = st.u["hp_abs"][:, 0].clone()
            Bt.step(st, o, P)
            w(st, hp0, was, area, 0)
    # the exam's helper changed nothing: the simulator's own function is back
    assert EW.melee.strikes is sk.orig
    return w


def test_the_watch_counts_the_line_beyond_the_area_and_the_reforms():
    sp = DW.specs(2, torch.Generator().manual_seed(5), fixed=["one"])
    st, own, area = DW.build(sp, P, "cpu")
    area["front"][0, :] = 10.0                 # battle 0: our line sticks out of a 10 m area
    area["front"][1, :] = 200.0
    o = O.hold(st.B, st.N)
    o.width[:, 0] = torch.where(st.u["width"][:, 0] > 40, 12.0, 70.0)    # a new width, given every step
    w = _watch_run(st, area, o, 6)
    assert float(w.beyond_s[0]) > 0 and float(w.beyond_s[1]) == 0
    assert w.reforms.tolist() == [1.0, 1.0]                     # one re-form begun each, not one a step
    assert bool((w.melee_s == 0).all()) and bool((w.front_contact < 0).all())


def test_the_watch_counts_the_men_striking_in_melee():
    sp = DW.specs(2, torch.Generator().manual_seed(5), fixed=["one"])
    st, own, area = DW.build(sp, P, "cpu")
    H = st.N // 2
    _, depth = geometry.dims(st.u, SP)
    for b in range(2):
        # the enemy touching our front
        d = float(depth[b, 0] / 2 + depth[b, H] / 2) - 0.5
        a = math.radians(float(st.u["b"][b, 0]))
        st.u["x"][b, H] = float(st.u["x"][b, 0]) + d * math.sin(a)
        st.u["z"][b, H] = float(st.u["z"][b, 0]) + d * math.cos(a)
        st.u["b"][b, H] = (float(st.u["b"][b, 0]) + 180) % 360
    o = opponents.nearest(st)
    w = _watch_run(st, area, o, 6)
    assert bool((w.melee_s > 0).all()) and bool((w.ours_strike > 0).all()) and bool((w.theirs_strike > 0).all())
    assert bool((w.front_contact > 0).all()) and float(w.reforms.sum()) == 0


def test_a_short_rollout_and_update_and_the_mix():
    assert "width" in T.MIX and "width" in T.SETUPS
    torch.manual_seed(0)
    cfg = dict(CFG, epochs=1, minibatch=32, gamma=0.9994, lam=0.9025, clip=0.2, vf=0.5, entropy=0.01)
    net, critic = Nt.UnitNet(), Nt.Critic()
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=3e-4)
    batch, stats = T.rollout(net, critic, cfg, P, "cpu", T.Duty("cpu"), T.width_setup(cfg, P, "cpu",
                                                                                      torch.Generator().manual_seed(6)))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0
    us = T.update(net, critic, opt, batch, cfg, "cpu", T.Duty("cpu"))
    assert math.isfinite(us["policy_loss"]) and math.isfinite(us["value_loss"])


class TestWidthBan:
    def test_widths_wider_than_the_area_are_banned_keep_never(self):
        lo, hi = torch.tensor([7.4, 7.4, 7.4]), torch.tensor([80.0, 80.0, 80.0])
        ok = Dc.width_ok(lo, hi, torch.tensor([30.0, 5.0, 200.0]))
        w = Dc.width_of(torch.arange(Nt.WIDTHS), lo[0].expand(Nt.WIDTHS), hi[0].expand(Nt.WIDTHS))
        assert ok[0].tolist() == [True] + [float(x) <= 30.0 for x in w[1:]] and int(ok[0, 1:].sum()) == 2
        # none fits: 'keep' and the narrowest
        assert ok[1].tolist() == [True, True] + [False] * (Nt.WIDTHS - 2)
        assert bool(ok[2].all()) and bool(Dc.width_ok(lo, hi, None).all())

    def test_the_network_never_picks_a_banned_width(self):
        sp = DW.specs(4, torch.Generator().manual_seed(8))
        setup = T.width_setup(CFG, P, "cpu", sp=sp)
        st, area = setup["st"], setup["area"]
        area["front"][:] = torch.tensor([20.0, 30.0, 6.0, 80.0])[:, None]
        wl = Dc.width_limits(st, P)
        raws = V.raw(st, SP)[None]
        z = torch.zeros(4, dtype=torch.long)
        v = T.views(raws, z, torch.arange(4), z, T.cards(st, "cpu"), area, CFG, wlim=wl)
        assert v["width_ok"].shape == (4, Nt.WIDTHS) and bool(v["width_ok"][:, Nt.WIDTH_KEEP].all())
        assert int(v["width_ok"][2].sum()) == 2 and bool(v["width_ok"][3].all())
        net = Nt.UnitNet()
        with torch.no_grad():
            net.width.bias.zero_()                    # every width as likely as 'keep'
        order = T.orders_now(raws[0], area, torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool),
                             torch.arange(4), z)
        lg = net(v, order, v["cell_m"])
        assert bool((lg["width"][~v["width_ok"]] <= Nt.NEG / 2).all())
        for _ in range(20):
            k = Nt.sample(lg)["width"]
            assert bool(v["width_ok"].gather(1, k[:, None]).all())
        # without the limits: no ban (the old view)
        assert "width_ok" not in T.views(raws, z, torch.arange(4), z, T.cards(st, "cpu"), area, CFG)

    def test_the_rollout_keeps_the_ban_and_the_update_sees_it(self):
        torch.manual_seed(1)
        cfg = dict(CFG, epochs=1, minibatch=32, gamma=0.9994, lam=0.9025, clip=0.2, vf=0.5, entropy=0.01)
        net, critic = Nt.UnitNet(), Nt.Critic()
        with torch.no_grad():
            net.width.bias.zero_()
        setup = T.width_setup(cfg, P, "cpu", torch.Generator().manual_seed(9))
        setup["area"]["front"][:] = 20.0
        batch, stats = T.rollout(net, critic, cfg, P, "cpu", T.Duty("cpu"), setup)
        assert stats["bans_broken"] == 0 and batch["wlim"] is not None
        d, r = batch["valid"].nonzero(as_tuple=True)
        v = T.views(batch["raws"], d, batch["bi"][r], batch["ii"][r], batch["card"], batch["area"], cfg,
                    wlim=batch["wlim"])
        assert bool(v["width_ok"].gather(1, batch["act"]["width"][d, r][:, None]).all())
        assert bool((~v["width_ok"]).any())                      # some widths really were banned
        opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=3e-4)
        us = T.update(net, critic, opt, batch, cfg, "cpu", T.Duty("cpu"))
        assert math.isfinite(us["policy_loss"])
