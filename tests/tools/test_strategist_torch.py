"""The strategist (tools/nn/strategist; build/diag_strategist/plan.md): the blind input, the deployment by code and its
checks, the formula and the plain strategies, the walk (sampling and the update give the same log-probability), the
teacher's term, a short battle under the operative. Needs torch (the container; the simulator on the CPU)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim.params import load  # noqa: E402
from tools.nn.strategist import battle as BT  # noqa: E402
from tools.nn.strategist import cases as CS  # noqa: E402
from tools.nn.strategist import deploy as DP  # noqa: E402
from tools.nn.strategist import inputs as IN  # noqa: E402
from tools.nn.strategist import net as NN  # noqa: E402
from tools.nn.strategist import teacher as TC  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402

P = load()


def pre_of(n=6, seed=3):
    return CS.Pre(CS.exam_specs(n=n, seed=seed), P, "cpu")


def test_the_input_is_blind_to_every_place():
    sp = CS.exam_specs(n=4, seed=1)
    a = CS.Pre(sp, P, "cpu")
    moved = [dict(s, side=[v + 37.0 for v in s["side"]]) for s in sp]       # the enemy elsewhere in its zone
    b = CS.Pre(moved, P, "cpu")
    ia, _ = BT.inputs_of(a)
    ib, _ = BT.inputs_of(b)
    for k in ("rows", "map", "numbers", "command", "mask", "own", "enemy", "shooter"):
        assert torch.allclose(ia[k].float(), ib[k].float(), atol=1e-5), k
    # my units elsewhere (the pre-state's row across the zone shifted): the same input
    raw = a.raw0.clone()
    from tools.nn.tactic import raw as TR
    raw[:, :CS.H, TR.I["x"]] += 55.0
    ic, _ = BT.inputs_of(a, raw)
    assert torch.allclose(ia["rows"], ic["rows"], atol=1e-5)


def test_an_enemy_row_keeps_only_the_composition():
    pre = pre_of(4)
    inp, _ = BT.inputs_of(pre)
    _, keep_enemy = IN.blind_masks()
    rows = inp["rows"][inp["enemy"]]
    assert bool((rows[:, ~keep_enemy] == 0).all())
    for name in ("right", "forward", "front_cos", "dh", "mp", "fat"):
        assert not bool(keep_enemy[RW.RI[name]])
    assert bool(keep_enemy[RW.RI["men"]]) and bool(keep_enemy[RW.RI["hp"]])
    assert bool((rows[:, :69] != 0).any())                                  # the passport stays


def test_the_cases_give_both_sides_a_lord_and_3_to_8_units():
    sp = CS.exam_specs(n=24, seed=5)
    pre = CS.Pre(sp, P, "cpu")
    u = pre.st0.u
    for b in range(len(sp)):
        for side in (1, 2):
            on = u["side"][b] == side
            assert 3 <= int(on.sum()) <= 8
            assert int((u["lord"][b] & on).sum()) == 1
        assert bool(((u["range"][b] > 0) & (u["side"][b] == 2)).any())       # the enemy has shooters
    arena = CS.exam_specs(n=4, arena=True)
    assert all(len(s["mine"]) == 7 and len(s["foes"]) == 7 for s in arena)


def test_the_formula_groups_places_and_commands():
    import math
    pre = pre_of(8)
    out, inp, ctx = BT.decide(pre)
    pl = out["plan"]
    n_own = inp["own"].sum(1)
    want = torch.tensor([math.ceil(int(k) / 3) for k in n_own])                 # the operative's line: up to 3 a group
    assert bool((pl["on"].sum(1) == want).all())
    assert bool((pl["no_cell"] == 0).all())
    assert bool((pl["members"].sum(1)[inp["own"]] == 1).all())                # every unit in one group
    # the groups side by side, touching zones allowed, none overlapping
    from tools.nn.strategist import zones as ZS0
    for g in range(1, int(pl["on"].sum(1).max())):
        on = pl["on"][:, g]
        prev = {k: pl[k][:, :g] for k in ("cx", "cz", "heading", "front", "depth")}
        ok = ZS0.apart(pl["cx"][:, g:g + 1], pl["cz"][:, g:g + 1], pl["front"][:, g], pl["depth"][:, g],
                       pl["heading"][:, g], prev, pl["on"][:, :g])
        assert bool(ok[on].all())
    destroy = pl["verb"] == DC.VERBS.index("destroy")
    assert bool((destroy == pre.destroy).all())
    st, g0, places, mem = BT.placed(pre, out, inp)
    errs = DP.errors(st, pre.own, g0.member, pre.mine_zone, P.sim["formation"]["spacing_m"])
    assert int(errs.sum()) == 0
    step = 360.0 / DP.SECTORS
    b = places["b"][pre.own]
    assert bool(((b / step - torch.floor(b / step) - 0.5).abs() < 1e-3).all())     # the sector's middle, as the game
    assert bool((places["width"][pre.own] <= 74.0 + 1e-4).all())
    # 'hold' at the back, 'attack' at the front of my zone
    from tools.nn.strategist import zones as ZS
    _, f = ZS.local(pre.mine_zone, pl["cx"], pl["cz"])
    assert bool((f[:, 0][pre.destroy] > 0).all()) and bool((f[:, 0][~pre.destroy] < 0).all())


def test_the_plain_strategies_differ():
    pre = pre_of(8)
    blob, _, _ = BT.decide(pre, mode="blob")
    assert bool((blob["plan"]["on"].sum(1) == 1).all())
    hold, _, _ = BT.decide(pre, mode="hold")
    assert bool((hold["plan"]["verb"] == DC.VERBS.index("hold")).all())


def test_the_deployment_checks_catch_overlaps_and_the_zone():
    pre = pre_of(2)
    out, inp, _ = BT.decide(pre)
    st, g0, _, _ = BT.placed(pre, out, inp)
    sp = P.sim["formation"]["spacing_m"]
    st.u["x"][0, 1] = st.u["x"][0, 0]
    st.u["z"][0, 1] = st.u["z"][0, 0]
    st.u["x"][1, 0] = st.u["x"][1, 0] + 2000.0
    e = DP.errors(st, pre.own, g0.member, pre.mine_zone, sp)
    assert int(e[0]) >= 1 and int(e[1]) >= 1
    g0.member[0, 0] = -1
    assert int(DP.errors(st, pre.own, g0.member, pre.mine_zone, sp)[0]) >= 2


def test_the_walk_rebuilds_its_log_probability_and_the_teacher_learns():
    torch.manual_seed(0)
    pre = pre_of(4)
    inp, ctx = BT.inputs_of(pre)
    net = NN.StrategistNet()
    with torch.no_grad():
        o = net(inp, ctx)
        again = net(inp, ctx, act=o["act"])
    assert torch.allclose(o["lp"], again["lp"], atol=1e-4)
    assert bool(torch.isfinite(o["lp"]).all())
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lab = TC.labels_of(inp, ctx)
    first = None
    for _ in range(15):
        term, _ = TC.loss(net, inp, ctx, lab)
        first = float(term.detach()) if first is None else first
        opt.zero_grad()
        term.backward()
        opt.step()
    assert float(term.detach()) < first


def test_a_short_battle_hands_the_groups_to_the_operative():
    from tools.nn.tactic import train as TT
    pre = pre_of(2)
    pre.limit = torch.full_like(pre.limit, 20.0)
    out, inp, _ = BT.decide(pre)
    rd, rs = BT.play(pre, out, inp, TT.unit_network("cpu"), op_net=None)
    assert rs["parcel_errors"] == 0 and rs["deploy_errors"] == 0
    assert rs["parcels"] >= int(out["plan"]["on"].sum())                    # the strategist's parcels at second 0
    m = BT.measures(pre, rd)
    for v in m.values():
        assert bool(((v >= 0) & (v <= 1)).all())


def test_the_runs_cut_as_numpy_array_split():
    import numpy as np
    from tools.nn.strategist import teacher as T
    for m in range(0, 9):
        for k in range(1, 4):
            sel = torch.zeros((1, 12), dtype=torch.bool)
            idx = sorted(np.random.default_rng(m * 7 + k).choice(12, m, replace=False).tolist())
            sel[0, idx] = True
            got = T.runs(sel, torch.tensor([k]))[0, idx].tolist()
            want = [i for i, part in enumerate(np.array_split(np.arange(m), k)) for _ in part]
            assert got == want, (m, k, got, want)
