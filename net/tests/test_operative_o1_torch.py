"""Drill O0-1 'advance on the holder' (net/operative: scenes_o1, advance, lines, sally, exam_o1) - the area's smoke test:
the twins share the cause scene's draws, the teacher's rule holds on its hand-made states, and a 45 s play of a cause
scene and its role twin under TEACH moves the zone once at 30 s in the cause scene only. Needs torch (the container,
CPU)."""
import pytest

torch = pytest.importorskip("torch")


def test_the_drill_plays_its_cause_and_keeps_in_its_twin():
    from net.operative import advance as AD
    from net.operative import exam_o1 as EX
    from net.operative import scenes_o1 as SO
    AD.table()                                          # (asserts each hand-made state)
    a, come, role = SO.scene(5), SO.scene(5, twin="come"), SO.scene(5, twin="role")
    assert a["places_mine"] == come["places_mine"] == role["places_mine"] and a["mine"] == come["mine"]
    assert [p[:2] for p in a["places_theirs"]] == [tuple(p) for p in come["o1"]["pose"]]
    assert (a["o1"]["att"], role["o1"]["att"], come["o1"]["enemy"]) == (1, 0, "walkin")
    for s in (a, role):
        s["limit"] = 45
    res = EX.play("TEACH", [a, role], [0], 2, "cpu", compile=False)
    assert res["per_battle"]["changes"] == [1.0, 0.0]
    t30 = {r[0]: r for r in res["ticks"] if r[1] == 30}
    assert t30[0][3] == 2 and t30[1][3] == -1           # (LS in the cause scene, KEEP in the role twin)


def test_the_o1_input_masks_and_action_on_a_turned_battle_a_hidden_enemy_and_a_moved_line():
    """The network's whole deployed input (heads.observe of a recorded 30 s decision), the head's mask and a
    non-trivial head's greedy action: the same on the battle turned +90 deg about the flat field's centre (positions,
    velocities, headings, the zone, the goal and the stand-off line - the field's 'up' turns with it); a hidden enemy
    leaves no trace (moved 50 m: the same input); the stand-off line moved 37 m from the zone in force is legal again,
    within same_line_m it is not."""
    import math
    from net.drill.levels import operative as OPL
    from net.operative import exam_o1 as EX
    from net.operative import frame as FR
    from net.operative import heads as HD
    from net.operative import lines as LN
    from net.operative import scenes_o1 as SO
    from net.tactic import raw as TR
    I = TR.I
    seed = next(s for s in range(5, 300) if SO.scene(s)["map"] == "flat")
    a = SO.scene(seed)
    a["limit"] = 31
    rec = OPL.Recorder()
    EX.play("TEACH", [a], [0], 1, "cpu", compile=False, recorder=rec)
    rows, card = rec.rows(), rec.cards(1)[0][None]
    raw, mem, E = rows["raw"][0], rows["mem"][0], rows["E"][0]
    ax = rows["ax"][0]
    half = float(ax[6]) / 2
    sec = int(rows["sec"][0])
    od = OPL._order(rows, 0)
    C = tuple(float(v) for v in rows["C"][0])
    ls = tuple(float(v) for v in rows["ls"][0])
    torch.manual_seed(0)
    net = HD.OpNet1().eval()
    torch.nn.init.normal_(net.s2.weight, std=1.0)

    def turn(x, z):
        return z, -x

    def inp(r, order, c, line, ls_same=False):
        axes = FR.Axes.of(r[None], torch.ones(1, dtype=torch.long), half)
        o = HD.inputs_of(r[None], card, mem, E, order["zone"], sec, axes)
        o["order"] = torch.tensor(HD.order_feats(order, c, line, sec))
        o["mission"] = torch.tensor([1.0, sec / 600.0])
        o["cand"] = torch.tensor(HD.cand_feats(order, c, line))
        o["legal"] = torch.tensor(HD.legal_of({"ls_same": ls_same}))
        return o

    r2 = raw.clone()
    r2[:, I["x"]], r2[:, I["z"]] = raw[:, I["z"]], -raw[:, I["x"]]
    r2[:, I["vx"]], r2[:, I["vz"]] = raw[:, I["vz"]], -raw[:, I["vx"]]
    r2[:, I["b"]] = torch.remainder(raw[:, I["b"]] + 90.0, 360.0)
    od2 = {**od, "zone": turn(*od["zone"][:2]) + ((od["zone"][2] + 90.0) % 360.0,), "goal": turn(*od["goal"])}
    x1, x2 = inp(raw, od, C, ls), inp(r2, od2, turn(*C), turn(*ls[:2]) + (ls[2],))
    assert float(x1["rows"].abs().max()) > 0 and float(x1["gframe"].abs().max()) > 0
    for k in HD.NET_KEYS:
        assert torch.allclose(x1[k].float(), x2[k].float(), atol=1e-4), (k, float((x1[k].float() - x2[k].float()).abs().max()))
    with torch.no_grad():
        l1, l2 = net(HD.stack([x1])), net(HD.stack([x2]))
    assert torch.allclose(l1, l2, atol=1e-4) and int(l1.argmax()) == int(l2.argmax())
    m1 = OPL.mask_of([{"melee": False, "ls_same": False}])
    assert m1.legal.tolist() == [[True, True]]
    # a hidden enemy: no row, not in E, and nothing of it reaches the input
    j = int(E.nonzero()[0, 0])
    rh = raw.clone()
    rh[j, I["vis"]] = 0.0
    rm = rh.clone()
    rm[j, I["x"]] += 50.0
    h1, h2 = inp(rh, od, C, ls), inp(rm, od, C, ls)
    assert j not in h1["slot"].tolist()
    assert int(h1["targets"].sum()) == int(x1["targets"].sum()) - 1
    for k in HD.NET_KEYS:
        assert torch.equal(h1[k], h2[k]), k
    # the stand-off line: the order in force on it - not legal; the line moved 37 m - legal again (lines' 20 m rule)
    same = float(LN.conf()["same_line_m"])
    on = {**od, "ptr": "LS", "zone": ls}
    for shift, legal in ((0.0, False), (37.0, True)):
        line = (ls[0] + shift, ls[1], ls[2])
        ls_same = math.hypot(on["zone"][0] - line[0], on["zone"][1] - line[1]) <= same
        x = inp(raw, on, C, line, ls_same)
        assert bool(x["legal"][1]) == legal
        assert OPL.mask_of([{"melee": False, "ls_same": ls_same}]).legal[0, 1].item() == legal
        with torch.no_grad():
            k = int(net(HD.stack([x])).argmax())
        assert legal or k == 0
