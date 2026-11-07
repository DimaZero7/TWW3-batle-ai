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
    assert bool((places["width"][pre.own] == pre.width0[pre.own]).all())        # the unit's own frontage
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


GAME_ZONES = {"own": {"x": -184.0, "z": 0.0, "b": 90.0, "front": 416.0, "depth": 176.0},
              "enemy": {"x": 184.0, "z": 0.0, "b": 270.0, "front": 416.0, "depth": 176.0}}
ROSTER_20 = (["wh_main_emp_cha_general_0"] + ["wh_main_emp_inf_greatswords"] * 6 + ["wh_main_emp_inf_halberdiers"] * 5
             + ["wh_main_emp_inf_swordsmen"] * 2 + ["wh2_dlc13_emp_inf_huntsmen_0"] * 2
             + ["wh3_dlc25_emp_inf_hochland_long_rifles"] * 2 + ["wh_main_emp_art_mortar"] * 2)


def game_zone_doc(task_zones=GAME_ZONES):
    """The bridge's deployment document of 20 Empire units against 20 Skaven in the game's zones of ksl_plains_01
    (terrain/deployment east / west: 416 x 176 m, fronts 192 m apart)."""
    from tests.tools import test_tactic_bridge_parity_torch as TP
    from tools.nn.sim import scenario
    rows = lambda keys, x: [{"key": k, "x": x, "z": -190.0 + 20.0 * i, "b": 90.0 if x < 0 else 270.0}   # noqa: E731
                            for i, k in enumerate(keys)]
    skv = ["wh2_main_skv_cha_warlord_0"] + ["wh2_main_skv_inf_clanrats_0"] * 19
    st = scenario.build([{"attacker": 2, "sides": {1: {"faction": "wh_main_emp_empire", "units": rows(ROSTER_20, -184.0)},
                                                   2: {"faction": "wh2_main_skv_skaven", "units": rows(skv, 184.0)}}}],
                        P, device="cpu", per_side=20)
    d = TP.doc_of_sim(st, 0, 0)
    units = [r if r["side"] == 1 else dict({k: r[k] for k in ("n", "side", "key", "men", "hp")}, v=True)
             for r in d["units"]]
    return dict(d, units=units, deploy=True, zones=task_zones, batch="game_zone", attacker=2,
                factions={"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"})


@pytest.mark.parametrize("task", ["hold", "destroy"])
def test_the_formula_fits_20_units_into_the_games_zone(task):
    """The game's zone 416 x 176 m holds 20 units: the groups' zones are their units' slots (deploy.footprint), the
    formula lays 6 groups in two rows, every unit inside the zone, none overlapping, facing the enemy (07.11: with the
    tactician's 80 m tile a unit the formula had 16 of 20 out of the zone)."""
    from tools.nn.companion import deploy as CD
    _, rec = CD.Strategist("formula", task, params=P).decide(game_zone_doc())
    assert rec["errors"] == 0 and len(rec["places"]) == 20
    assert len(rec["groups"]) == 6
    for n, (x, z, b, w) in rec["places"].items():
        assert -272.0 <= x <= -96.0 and -208.0 <= z <= 208.0, (n, x, z)
        assert abs(b - 92.8125) < 1e-3                                            # facing the enemy (east), 64 sectors
    # 'hold' - the first row at the back edge, 'attack' - at the front
    xs = [g["cx"] for g in rec["groups"][:3]]
    assert all(x < -184.0 for x in xs) if task == "hold" else all(x > -184.0 for x in xs)


def test_a_groups_slot_is_its_widest_formation():
    m = torch.tensor([[[True, True, False], [False, False, True]]])
    front, depth = torch.tensor([[30.0, 50.0, 20.0]]), torch.tensor([[10.0, 8.0, 12.0]])
    w, d = DP.slot(m, front, depth)
    assert torch.allclose(w, torch.tensor([[56.0, 26.0]])) and torch.allclose(d, torch.tensor([[16.0, 18.0]]))
    fw, fd = DP.footprint(torch.tensor([[4, 3]]), torch.tensor([[False, True]]), w, d)
    assert torch.allclose(fw, torch.tensor([[112.0, 78.0]])) and torch.allclose(fd, torch.tensor([[32.0, 18.0]]))
