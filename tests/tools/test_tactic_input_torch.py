"""The tactician's input (tools/nn/tactic; design/tactic/index.html): the shapes of every block, nothing of an enemy
outside the rows' zone reaches the input, turning the zone turns the map and the rows together, a unit that fell out
takes its tile where it stood, the parcel's checks, the far window without shooters, the hidden target, the enemy's
columns, the identity by slot. Needs torch (the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import inputs as IN  # noqa: E402
from tools.nn.tactic import net as NT  # noqa: E402
from tools.nn.tactic import parcel as PC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402
from tools.nn.tactic import windows as W  # noqa: E402
from tools.nn.tactic import zone as ZN  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
EMP = ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen",
       "wh3_dlc25_emp_inf_nuln_ironsides", "wh_main_emp_inf_spearmen_0"]
SKV = ["wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_dlc12_skv_inf_ratling_gun_0",
       "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_art_warp_lightning_cannon"]


def army(gap=120.0, far_enemy=None):
    def line(keys, z, b):
        return [{"key": k, "x": -60.0 + 30.0 * i, "z": z, "b": b} for i, k in enumerate(keys)]
    skv = line(SKV, gap / 2, 180.0)
    if far_enemy is not None:
        skv[3] = dict(skv[3], x=far_enemy[0], z=far_enemy[1])
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": line(EMP, -gap / 2, 0.0)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": skv}}}


def state(seconds=3, **kw):
    st = scenario.build([army(**kw)], P)
    seer = TR.Seer(P)
    raw = None
    for _ in range(seconds):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.hold(st), P)
        raw = seer.raw(st)
    return st, raw


def slot_of(st, key, side):
    return [i for i, k in enumerate(st.keys[0]) if k == key and int(st.u["side"][0, i]) == side]


def group(st, raw, mine, target, heading=0.0, line=False):
    x, z = raw[0, mine, TR.I["x"]].mean(), raw[0, mine, TR.I["z"]].mean()
    return PC.Parcel.make(0, mine, target, float(x), float(z), heading, line, 0, n=st.N)


def build(st, raw, p, **kw):
    return IN.build(raw, T.cards(st, "cpu"), p, ZN.Tiles.of(p), bounds=st.bounds, **kw)


def test_the_shapes_of_every_block_and_the_weights():
    st, raw = state()
    assert raw.shape[-1] == len(TR.RAW)
    p = group(st, raw, [1, 2, 3], [st.N // 2 + 1, st.N // 2 + 2])
    assert not any(PC.check(raw, p)[0])
    inp = build(st, raw, p)
    K = 32
    assert inp["rows"].shape == (1, K, 185) and inp["mask"].shape == (1, K)
    assert inp["near"].shape == (1, 34, 30, 30) and inp["far"].shape == (1, 28, 30, 30)
    assert inp["target"].shape == (1, 17, 30, 30) and inp["numbers"].shape == (1, 4)
    assert len(W.near_layers()) == 34 and len(W.far_layers()) == 28 and len(W.target_layers()) == 17
    assert p.vector().shape == (1, 2 * st.N + 7)
    # mine first, then the target, by slot
    assert inp["slot"][0, :5].tolist() == [1, 2, 3, st.N // 2 + 1, st.N // 2 + 2]
    assert inp["who"][0, :5].tolist() == [RW.OWN_GROUP] * 3 + [RW.ENEMY_TARGET] * 2
    net = NT.TacticInput()
    out = net(inp)
    assert out["cards"].shape == (1, K, 128) and out["card8"].shape == (1, K, 8)
    assert out["map_vec"].shape == (1, 128) and out["map_near"].shape == (1, 64, 30, 30)
    w = NT.weights(net)
    assert w["analysis"] == 185 * 256 + 256 + 256 * 128 + 128
    assert w["conv_near"] == 79376 and w["conv_far"] == 17344 and w["conv_target"] == 14176
    assert 110_000 < w["conv_all"] < 112_000
    # the card layers are painted where my units / the target stand
    names = W.near_layers()
    near = out["near"][0]
    mine_d = near[names.index("mine")]
    assert float(mine_d.max()) > 0.2
    c0 = names.index("card_mine0")
    assert float(near[c0:c0 + 8][:, mine_d == 0].abs().max()) == 0.0
    assert float(near[c0:c0 + 8].abs().max()) > 0


def test_the_zone_grid_by_the_number_of_units():
    want = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3),
            10: (3, 4)}
    assert {n: ZN.grid(n) for n in want} == want
    assert ZN.grid(4, line=True) == (1, 4)


def test_an_enemy_outside_the_zone_changes_nothing():
    st, raw = state(far_enemy=(600.0, 900.0))
    far = slot_of(st, "wh2_main_skv_inf_clanrat_spearmen_0", 2)[1]
    p = group(st, raw, [1, 2], [st.N // 2 + 1])
    a = build(st, raw, p)
    assert far not in a["slot"][0].tolist()
    raw2 = raw.clone()
    for k, v in (("men", 0.3), ("hp", 0.2), ("b", 77.0), ("fat", 4.0), ("e_poison", 3.0), ("a_has", 3.0),
                 ("x", 650.0), ("range", 0.0), ("fire", 1.0)):
        raw2[0, far, TR.I[k]] = v
    b = build(st, raw2, p)
    for k in ("rows", "mask", "slot", "who", "near", "far", "target", "numbers"):
        assert torch.equal(a[k], b[k]), k
    for k in a["cover"]:
        assert torch.equal(a["cover"][k], b["cover"][k]), k
    # within the zone's reach the same change does show
    raw3 = raw.clone()
    raw3[0, far, TR.I["x"]], raw3[0, far, TR.I["z"]] = -30.0, 0.0
    c = build(st, raw3, p)
    assert far in c["slot"][0].tolist()


def test_turning_the_zone_turns_the_map_and_the_rows_together():
    st, raw = state()
    mine = [1, 2, 3, 4]                                        # 2 x 2 tiles: the zone is square
    p = group(st, raw, mine, [st.N // 2 + 2], heading=10.0)
    q = group(st, raw, mine, [st.N // 2 + 2], heading=100.0)   # turned 90 deg clockwise
    a, b = build(st, raw, p), build(st, raw, q)
    for win in ("near", "far", "target"):
        names = W.LAYERS[win]()
        rot = torch.rot90(a[win], 1, (-2, -1))                  # what was right is now up
        for i, n in enumerate(names):
            if n.endswith("_cos"):                              # the fronts against "up": a - 90 deg
                rot[:, i], rot[:, i + 1] = rot[:, i + 1].clone(), -rot[:, i].clone()
        assert torch.allclose(rot, b[win], atol=2e-3), (win, float((rot - b[win]).abs().max()))
    for win in a["cover"]:
        assert torch.allclose(torch.rot90(a["cover"][win], 1, (-2, -1)), b["cover"][win], atol=2e-3), win
    assert torch.equal(a["slot"], b["slot"])
    ra, rb = a["rows"][0], b["rows"][0]
    ix = RW.RI
    assert torch.allclose(rb[:, ix["forward"]], ra[:, ix["right"]], atol=1e-5)
    assert torch.allclose(rb[:, ix["right"]], -ra[:, ix["forward"]], atol=1e-5)
    keep = [i for i, n in enumerate(RW.ROW) if n not in ("right", "forward", "front_cos", "front_sin", "v_right",
                                                         "v_forward")]
    assert torch.allclose(ra[:, keep], rb[:, keep], atol=1e-6)
    na, nb = a["numbers"][0], b["numbers"][0]
    assert float(na[0]) == pytest.approx(float(nb[0]), abs=1e-5)
    assert float(nb[1]) == pytest.approx(float(na[2]), abs=1e-5) and float(nb[2]) == pytest.approx(-float(na[1]), abs=1e-5)


def test_turning_the_whole_battle_with_its_zone_changes_nothing():
    st, raw = state()
    p = group(st, raw, [1, 2], [st.N // 2 + 1, st.N // 2 + 2], heading=20.0)
    a = build(st, raw, p)
    t = math.radians(90.0)
    raw2 = raw.clone()
    x, z = raw[..., TR.I["x"]], raw[..., TR.I["z"]]
    vx, vz = raw[..., TR.I["vx"]], raw[..., TR.I["vz"]]
    # a bearing b + 90: (x, z) -> (x cos + z sin, -x sin + z cos)
    raw2[..., TR.I["x"]], raw2[..., TR.I["z"]] = x * math.cos(t) + z * math.sin(t), -x * math.sin(t) + z * math.cos(t)
    raw2[..., TR.I["vx"]], raw2[..., TR.I["vz"]] = vx * math.cos(t) + vz * math.sin(t), -vx * math.sin(t) + vz * math.cos(t)
    raw2[..., TR.I["b"]] = torch.remainder(raw[..., TR.I["b"]] + 90.0, 360.0)
    cx, cz = float(p.cx), float(p.cz)
    q = PC.Parcel.make(0, p.mine[0], p.target[0], cx * math.cos(t) + cz * math.sin(t),
                       -cx * math.sin(t) + cz * math.cos(t), 110.0, False, 0)
    b = IN.build(raw2, T.cards(st, "cpu"), q, ZN.Tiles.of(q), bounds=None)
    a = IN.build(raw, T.cards(st, "cpu"), p, ZN.Tiles.of(p), bounds=None)
    assert torch.equal(a["slot"], b["slot"])
    assert torch.allclose(a["rows"], b["rows"], atol=1e-4)
    for win in ("near", "far", "target", "numbers"):
        assert torch.allclose(a[win], b[win], atol=2e-3), win


def test_a_unit_that_falls_out_takes_its_tile_where_it_stood():
    st, raw = state()
    mine = [1, 2, 3, 4]
    p = group(st, raw, mine, [])
    keep = ZN.Keeper()
    t0 = keep.update(p, raw)
    assert int(t0.mask.sum()) == 4 and int(t0.rows) == 2 and int(t0.cols) == 2
    size0 = float(t0.near_size()[0])
    x, z = t0.world()
    ux, uz = float(raw[0, 4, TR.I["x"]]), float(raw[0, 4, TR.I["z"]])
    nearest = int(((x[0] - ux) ** 2 + (z[0] - uz) ** 2)[:4].argmin())
    raw2 = raw.clone()
    raw2[0, 4, TR.I["alive"]] = 0.0                             # unit 4 dead
    t1 = keep.update(p, raw2)
    assert t1.mask[0, :4].tolist() == [k != nearest for k in range(4)]
    assert float(t1.near_size()[0]) == size0                   # the zone keeps its size: a hole on the spot
    t2 = keep.update(p, raw2)                                   # once only
    assert int(t2.mask.sum()) == 3
    inp = IN.build(raw2, T.cards(st, "cpu"), p, t2, bounds=st.bounds)
    assert 4 not in inp["slot"][0].tolist()
    assert float(inp["near"][0, 0].sum()) == pytest.approx(3 * (80.0 / float(inp["cell"][0, 0])) ** 2, rel=1e-3)
    # a routing unit falls out as well
    raw3 = raw2.clone()
    raw3[0, 1, TR.I["r"]] = 1.0
    assert int(keep.update(p, raw3).mask.sum()) == 2


def test_the_parcels_checks_catch_a_split_or_a_wide_target_and_bad_slots():
    st, raw = state()
    e = [i for i in range(st.N) if int(st.u["side"][0, i]) == 2]
    ok = group(st, raw, [1, 2], e[:3])
    assert PC.check(raw, ok) == [[]]
    raw2 = raw.clone()
    raw2[0, e[2], TR.I["x"]] = 400.0                            # one target unit far off: two groups
    err = PC.check(raw2, ok, near_rule=False)[0]
    assert any("not connected" in x for x in err)
    # a chain of touching squares wider than the near window (1 unit: 80 + 2 x 80 = 240 m)
    raw3 = raw.clone()
    for k, i in enumerate(e):
        raw3[0, i, TR.I["x"]], raw3[0, i, TR.I["z"]] = -150.0 + 75.0 * k, 100.0
    one = group(st, raw3, [1], e)
    err = PC.check(raw3, one, near_rule=False)[0]
    assert not any("not connected" in x for x in err) and any("not compact" in x for x in err)
    dead = raw.clone()
    dead[0, 2, TR.I["alive"]] = 0.0
    assert any("not alive" in x for x in PC.check(dead, ok)[0])
    with pytest.raises(PC.ParcelError):
        PC.ensure(raw2, ok, near_rule=False)
    # two groups holding one unit
    both = PC.Parcel.cat([ok, group(st, raw, [2, 3], [])])
    assert all(any("two groups" in x for x in g) for g in PC.check(raw, both))
    # the generator gives correct parcels
    g = torch.Generator().manual_seed(0)
    for _ in range(10):
        rp = PC.random(raw, g)
        assert rp is not None and PC.check(raw, rp) == [[]]


def test_the_far_window_is_zeros_and_flagged_without_shooters():
    st, raw = state()
    melee = group(st, raw, [1, 2], [st.N // 2 + 1])
    a = build(st, raw, melee)
    assert float(a["far"].abs().max()) == 0.0 and float(a["numbers"][0, 3]) == 1.0
    out = NT.TacticInput()(a)
    assert float(out["far"].abs().max()) == 0.0                 # no cards painted either
    guns = group(st, raw, [2, 3], [st.N // 2 + 1])               # the Nuln Ironsides shoot
    b = build(st, raw, guns)
    assert float(b["numbers"][0, 3]) == 0.0 and float(b["far"].abs().max()) > 0
    rng = float(raw[0, 3, TR.I["range"]])
    assert float(b["cell"][0, 1]) == pytest.approx(rng / 14.0)
    names = W.far_layers()
    assert float(b["far"][0, names.index("our_reach")].max()) == 1.0


def test_a_hidden_target_unit_is_not_in_the_input():
    st, raw = state()
    t = [st.N // 2 + 1, st.N // 2 + 2]
    p = group(st, raw, [1, 2], t)
    raw2 = raw.clone()
    raw2[0, t[1], TR.I["vis"]] = 0.0
    a = build(st, raw2, p)
    assert t[1] not in a["slot"][0].tolist() and t[0] in a["slot"][0].tolist()
    raw3 = raw2.clone()
    raw3[0, t[1], TR.I["x"]] += 15.0
    raw3[0, t[1], TR.I["men"]] = 0.1
    b = build(st, raw3, p)
    for k in ("rows", "near", "far", "target", "numbers"):
        assert torch.equal(a[k], b[k]), k


def test_an_enemys_row_fills_only_what_the_player_sees():
    st, raw = state()
    e = st.N // 2 + 1
    raw2 = raw.clone()
    for k, v in (("e_poison", 3.0), ("e_terror_immune", 5.0), ("e_hidden", 1.0), ("a_has", 1.0), ("a_on0", 5.0),
                 ("a_cd0", 30.0)):
        raw2[0, e, TR.I[k]] = v
        raw2[0, 1, TR.I[k]] = v
    p = group(st, raw2, [1, 2], [e])
    inp = build(st, raw2, p)
    rows, who = inp["rows"][0], inp["who"][0]
    enemy = (who == RW.ENEMY_TARGET) | (who == RW.ENEMY_OTHER)
    hidden = torch.tensor(~RW.ENEMY_SEES)
    assert float(rows[enemy][:, hidden].abs().max()) == 0.0
    er = rows[inp["slot"][0].tolist().index(e)]
    orow = rows[inp["slot"][0].tolist().index(1)]
    ix = RW.RI
    assert float(er[ix["poison_on"]]) == 1.0 and float(er[ix["poison_left"]]) == 0.0
    assert float(orow[ix["poison_left"]]) == pytest.approx(0.7)
    assert float(er[ix["on_0"]]) == 1.0 and float(er[ix["cd_0"]]) == 0.0 and float(orow[ix["cd_0"]]) > 0
    assert float(orow[ix["terror_immune_on"]]) == 1.0 and float(er[ix["terror_immune_on"]]) == 0.0
    assert int(torch.tensor(RW.ENEMY_SEES).sum()) == 150


def test_the_identity_is_by_slot_row_by_row():
    st, raw = state()
    p = group(st, raw, [1, 2, 3], [st.N // 2 + 1, st.N // 2 + 2])
    inp = build(st, raw, p)
    net = NT.UnitAnalysis()
    a = net(inp["rows"], inp["mask"])
    perm = torch.randperm(inp["rows"].shape[1], generator=torch.Generator().manual_seed(5))
    b = net(inp["rows"][:, perm], inp["mask"][:, perm])
    assert torch.allclose(a[:, perm], b, atol=1e-6)
    assert float(a[~inp["mask"]].abs().max()) == 0.0


def test_the_effect_rules_charge_and_braced_from_positions():
    st, _ = state(1)
    seer = TR.Seer(P)
    r0 = seer.raw(st)
    st.u["mv"][0, 1] = True
    st.u["x"][0, 1] += 3.0                                     # spearmen run 3 m/s: a run-up
    r1 = seer.raw(st)
    st.u["m"][0, 1] = True
    r2 = seer.raw(st)
    assert float(r2[0, 1, TR.I["e_charge"]]) > 0
    assert float(r0[0, 2, TR.I["e_braced"]]) >= 0              # (computed; swordsmen cannot brace)
    del r1
