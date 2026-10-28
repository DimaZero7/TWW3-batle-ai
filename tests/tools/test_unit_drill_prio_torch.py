"""Drill U8 'a priority target' (tools/nn/unit/drill_prio.py, train.py prio / obedience / --prio-cost): the two layouts
and their slots, the order's main target (has_target, the row's my_target flag) only while it is there, no target - the
view and the order as before, obedience counted only while the target is allowed (not a mask: the other enemies stay
allowed), the cost on the reward, a short rollout and update. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import drill_prio as DQ  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
CFG = {"battles": 4, "seconds": 20, "cell_m": 8.0, "bounds": None}
FLAG = V.ROW.index("my_target")


def test_the_layouts_slots_and_priorities():
    sp = DQ.specs(80, torch.Generator().manual_seed(3))
    assert {s["layout"] for s in sp} == set(DQ.LAYOUTS)
    given = sum(s["prio"] is not None for s in sp) / len(sp)
    assert 0.5 < given < 0.9
    st, own, area, goal, prio, duel = DQ.build(sp, P, "cpu")
    H = st.N // 2
    for b, s in enumerate(sp):
        assert int(prio[b, 0]) == (-1 if s["prio"] is None else H + DQ.PRIOS[s["layout"]].index(s["prio"]))
        assert bool(duel[b]) == (s["layout"] == "duel") and (float(area["sw"][b, 0]) > 0) == (s["layout"] == "duel")
        assert float(st.u["range"][b, H + 1]) == 0                                # infantry: the duel's and U3's B
        if s["layout"] == "duel":
            # the enemy shooter in our sector, within our range
            dx, dz = float(st.u["x"][b, H]), float(st.u["z"][b, H])
            off = abs((math.degrees(math.atan2(dx, dz)) - s["sb"] + 180) % 360 - 180)
            assert float(st.u["range"][b, H]) > 0 and off <= s["sw"] / 2
            assert math.hypot(dx, dz) <= float(st.u["range"][b, 0])
        else:
            assert float(st.u["range"][b, 0]) == 0 and float(st.u["range"][b, H]) == 0
    assert bool((prio[:, 1:] == -1).all())
    ex = DQ.exam_specs()
    assert len(ex) == 240 and {(s["layout"], s["prio"]) for s in ex} == set(DQ.EXAM_CELLS)


def _views(setup, prio):
    st = setup["st"]
    now = V.raw(st, SP)
    raws = now[None]
    B = st.B
    z = torch.zeros(B, dtype=torch.long)
    v = T.views(raws, z, torch.arange(B), z, T.cards(st, "cpu"), setup["area"], CFG, prio=prio)
    return now, v


def test_the_order_names_the_target_while_it_is_there():
    sp = DQ.specs(4, torch.Generator().manual_seed(1), fixed=[("duel", "shooter"), ("two", "B"), ("duel", None),
                                                               ("two", "A")])
    setup = T.prio_setup(CFG, P, "cpu", sp=sp)
    st, prio = setup["st"], setup["prio"]
    H = st.N // 2
    now, v = _views(setup, prio)
    bi, ii = torch.arange(4), torch.zeros(4, dtype=torch.long)
    o = T.orders_now(now, setup["area"], torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool), bi, ii,
                     prio=prio)
    assert o[:, Nt.ORDER.index("has_target")].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert v["prio"].tolist() == [H, H + 1, -1, H]
    for b in range(4):
        flag = v["rows"][b, :, FLAG]
        if int(v["prio"][b]) < 0:
            assert float(flag.abs().sum()) == 0
        else:
            k = (v["slot"][b] == v["prio"][b]).nonzero()[:, 0]
            assert flag.nonzero()[:, 0].tolist() == k.tolist()               # the flag on its row only (if in the zone)
    # the target gone (dead): no target named any more, no flag
    st.u["men"][0, H] = 0.0
    now, v = _views(setup, prio)
    o = T.orders_now(now, setup["area"], torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool), bi, ii,
                     prio=prio)
    assert float(o[0, Nt.ORDER.index("has_target")]) == 0.0 and int(v["prio"][0]) == -1
    assert float(v["rows"][0, :, FLAG].abs().sum()) == 0


def test_no_target_the_view_and_the_order_as_before():
    sp = DQ.specs(4, torch.Generator().manual_seed(2), fixed=[("duel", "shooter"), ("two", "A")])
    setup = T.prio_setup(CFG, P, "cpu", sp=sp)
    none = torch.full_like(setup["prio"], -1)
    _, v0 = _views(setup, None)
    _, v1 = _views(setup, none)
    for k in ("rows", "mem", "map", "target_ok", "cells_ok", "slot"):
        assert torch.equal(v0[k], v1[k])
    now = V.raw(setup["st"], SP)
    bi, ii = torch.arange(4), torch.zeros(4, dtype=torch.long)
    pace, arr = torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool)
    assert torch.equal(T.orders_now(now, setup["area"], pace, arr, bi, ii),
                       T.orders_now(now, setup["area"], pace, arr, bi, ii, prio=none))


def test_obedience_only_while_the_target_is_allowed_and_no_mask():
    sp = DQ.specs(2, torch.Generator().manual_seed(4), fixed=[("duel", "shooter"), ("duel", "infantry")])
    for s in sp:
        s["start"] = 340.0                                                     # the infantry beyond every range
    setup = T.prio_setup(CFG, P, "cpu", sp=sp)
    st, prio = setup["st"], setup["prio"]
    H = st.N // 2
    _, v = _views(setup, prio)
    K = v["slot"].shape[1]
    row = lambda b, j: int((v["slot"][b] == j).nonzero()[0, 0])               # noqa: E731
    assert bool(v["target_ok"][0, row(0, H)])                                  # the shooter within our range
    # the named shooter (battle 0): its row obeys, the 'nobody' does not; the infantry (battle 1) is not allowed yet
    act = {"target": torch.tensor([row(0, H), row(1, H)])}
    allow, obey = T.obedience(v, act)
    assert allow.tolist() == [True, False] and obey.tolist() == [True, False]
    act = {"target": torch.tensor([K, K])}
    allow, obey = T.obedience(v, act)
    assert allow.tolist() == [True, False] and obey.tolist() == [False, False]
    # not a mask: the same targets allowed with or without the named one
    _, v0 = _views(setup, None)
    assert torch.equal(v0["target_ok"], v["target_ok"])


def test_the_cost_lands_on_the_reward():
    sp = DQ.specs(4, torch.Generator().manual_seed(5), fixed=[("duel", "shooter")])
    out = {}
    for c in (0.0, 0.01):
        torch.manual_seed(0)
        b, s = T.rollout(Nt.UnitNet(), None, dict(CFG, prio_cost=c), P, "cpu", T.Duty("cpu"),
                         T.prio_setup(CFG, P, "cpu", sp=sp))
        pr = s["_prio"]
        out[c] = (float(b["reward"].sum()), float((pr["allowed_s"] - pr["obeyed_s"]).sum()), s["prio"])
    assert out[0.0][1] == out[0.01][1] > 0 and out[0.0][2]["allowed_s"] > 0
    assert out[0.0][0] - out[0.01][0] == pytest.approx(0.01 * out[0.0][1], rel=1e-4)


def test_a_short_rollout_and_update():
    torch.manual_seed(0)
    cfg = dict(CFG, epochs=1, minibatch=32, gamma=0.9994, lam=0.9025, clip=0.2, vf=0.5, entropy=0.01, prio_cost=0.003)
    net, critic = Nt.UnitNet(), Nt.Critic()
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=3e-4)
    sp = DQ.specs(4, torch.Generator().manual_seed(6))
    batch, stats = T.rollout(net, critic, cfg, P, "cpu", T.Duty("cpu"), T.prio_setup(cfg, P, "cpu", sp=sp))
    assert stats["bans_broken"] == 0 and stats["nan"] == 0 and batch["prio"] is not None
    us = T.update(net, critic, opt, batch, cfg, "cpu", T.Duty("cpu"))
    assert math.isfinite(us["policy_loss"]) and math.isfinite(us["value_loss"])
