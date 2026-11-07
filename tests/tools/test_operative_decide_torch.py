"""The operative's walk, target, step and formula (tools/nn/operative: net, target, step, teacher; plan.md "Выход",
"Проверка посылки"): every parcel a walk sends passes tactic/parcel.check, the choices replay to the same log-probability,
the bans hold, the formula groups the cases as the plan says. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.operative import cases as CS  # noqa: E402
from tools.nn.operative import cfg  # noqa: E402
from tools.nn.operative import env as EN  # noqa: E402
from tools.nn.operative import frame as FR  # noqa: E402
from tools.nn.operative import groups as GR  # noqa: E402
from tools.nn.operative import inputs as IN  # noqa: E402
from tools.nn.operative import lower as LW  # noqa: E402
from tools.nn.operative import net as NN  # noqa: E402
from tools.nn.operative import step as SP  # noqa: E402
from tools.nn.operative import target as TG  # noqa: E402
from tools.nn.operative import teacher as TC  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import parcel as PC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402

P = load()


def setup_of(n=6, seed=0, cases=CS.OLD):
    sp = CS.specs(n, torch.Generator().manual_seed(seed), fixed=True, cases=cases)
    s = CS.Setup(sp, P, "cpu")
    return s, TR.Seer(P).raw(s.st)


def decide(net, setup, raw, groups, act=None):
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds,
                   K=EN.rows_of(raw.shape[1]))
    ctx = NN.ctx_of(inp, raw, axes, groups)
    out, _ = net(inp, ctx, act=act)
    return out, inp, ctx, axes


def sent_ok(raw, out, axes, B, N):
    tac = LW.Tacticians(B, N, int(cfg()["groups"]), "cpu")
    p, idx = tac.apply(out["new"], out["task"], out["moved"], raw)
    if idx.numel() == 0:
        return 0
    errs = PC.check(raw, p, EN.field_dict(axes, None))
    assert not any(errs), [e for e in errs if e][:3]
    return idx.numel()


def test_a_random_network_sends_only_correct_parcels_and_replays_its_choices():
    torch.manual_seed(0)
    net = NN.OperativeNet()
    setup, raw = setup_of(9, seed=1)
    B, N = raw.shape[:2]
    groups = GR.Groups.empty(B, N)
    sent = 0
    for step in range(4):
        out, inp, ctx, axes = decide(net, setup, raw, groups)
        sent += sent_ok(raw, out, axes, B, N)
        again, _ = net(inp, ctx, act=out["act"])
        assert torch.allclose(again["lp"], out["lp"], atol=1e-4)
        new = out["new"]
        # every own alive unit in one group at most, groups of 1-10, the verb and the pace only the open ones
        mem = new.members() & new.on[..., None]
        assert int(mem.sum(1).max()) <= 1
        n = mem.sum(2)
        assert bool(((n >= 1) | ~new.on).all()) and int(n.max()) <= 10
        assert bool((new.verb[new.on] == DC.VERBS.index("hold")).all())
        assert bool((new.pace[new.on] == DC.PACES.index("any")).all())
        # a kept group: the same members and task
        kept = groups.on & (out["act"]["keep"] == GR.KEEP)
        for k in ("cx", "cz", "heading", "n0"):
            assert torch.equal(getattr(new, k)[kept], getattr(groups, k)[kept])
        groups = new
    assert sent > 9


def test_the_first_decision_groups_every_own_unit():
    torch.manual_seed(1)
    net = NN.OperativeNet()
    setup, raw = setup_of(6, seed=2)
    B, N = raw.shape[:2]
    out, *_ = decide(net, setup, raw, GR.Groups.empty(B, N))
    own_alive = setup.own & (raw[..., TR.I["alive"]] > 0)
    assert bool(((out["new"].member >= 0) == own_alive).all())


def test_grow_takes_the_touching_piece_only_and_fits_the_window():
    setup, raw = setup_of(3, seed=4, cases=("split",))
    B, N = raw.shape[:2]
    I = TR.I
    enemy = (raw[..., I["side"]] == 2) & (raw[..., I["alive"]] > 0) & (raw[..., I["vis"]] > 0)
    seed = torch.tensor([int(enemy[b].nonzero()[0, 0]) for b in range(B)])
    gx = raw[:, :3, I["x"]].mean(1)
    gz = raw[:, :3, I["z"]].mean(1)
    near = SP.near(torch.full((B,), 3), torch.zeros(B, dtype=torch.bool))
    t = TG.grow(raw, torch.arange(B), seed, enemy, gx, gz, near)
    assert bool((t.sum(1) == 3).all())                      # the first three (one piece), not the far three
    H = CS.H
    assert bool((t[:, H:H + 3]).all()) and not bool(t[:, H + 3:].any())
    base = TG.base_heading(raw, torch.arange(B), t, gx, gz, torch.zeros(B))
    ok = TG.turn_ok(raw, torch.arange(B), t, base, near, gx, gz)
    assert bool(ok[:, 0].all())
    none = TG.grow(raw, torch.arange(B), torch.full((B,), -1), enemy, gx, gz, near)
    assert not bool(none.any())


def test_the_step_keeps_the_near_rule_and_the_field():
    B = 64
    g = torch.Generator().manual_seed(5)
    gx, gz = (torch.rand(B, generator=g) - 0.5) * 1900, (torch.rand(B, generator=g) - 0.5) * 1900
    qx, qz = (torch.rand(B, generator=g) - 0.5) * 2100, (torch.rand(B, generator=g) - 0.5) * 2100
    n = torch.randint(1, 7, (B,), generator=g)
    line = torch.rand(B, generator=g) < 0.3
    hd = torch.rand(B, generator=g) * 360
    field = {"x": torch.zeros(B), "z": torch.zeros(B), "b": torch.zeros(B), "front": torch.full((B,), 2040.0),
             "depth": torch.full((B,), 2040.0)}
    front, depth = SP.size(n, line)
    nm = SP.near(n, line)
    x, z = SP.clamp(gx, gz, qx, qz, nm, front, depth, hd, field)
    assert bool((torch.sqrt((x - gx) ** 2 + (z - gz) ** 2) <= nm / 2 + 1e-3).all())
    assert bool(SP.field_ok(x[:, None], z[:, None], front, depth, hd, field).all())


def test_the_formula_groups_the_cases_as_the_plan_says():
    setup, raw = setup_of(6, seed=6)
    B, N = raw.shape[:2]
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    for mode, want in (("formula", {"gather": 1, "split": 2, "go": 1}), ("each", {"gather": 3, "split": 6, "go": 3}),
                       ("blob", {"gather": 1, "split": 1, "go": 1})):
        out = TC.play_formula(inp, ctx, setup.destroy, mode)
        for b in range(B):
            assert int(out["new"].on[b].sum()) == want[CS.CASES[int(setup.case[b])]], (mode, b)
    out = TC.play_formula(inp, ctx, setup.destroy, "formula")
    new = out["new"]
    for b in range(B):
        if CS.CASES[int(setup.case[b])] == "split":
            # each clump its own group, its own enemy three
            for k in (0, 1):
                g = int(new.member[b, setup.clump[b] == k][0])
                assert bool((new.member[b, setup.clump[b] == k] == g).all())
                xz = raw[b][..., [TR.I["x"], TR.I["z"]]]
                t = xz[new.target[b, g]].mean(0)
                m = xz[setup.clump[b] == k].mean(0)
                o = xz[setup.clump[b] == 1 - k].mean(0)
                assert float((t - m).norm()) < float((t - o).norm())
    tac = LW.Tacticians(B, N, int(cfg()["groups"]), "cpu")
    p, idx = tac.apply(new, out["task"], out["moved"], raw)
    assert not any(PC.check(raw, p, EN.field_dict(axes, None)))


def test_the_teachers_term_is_finite_and_agrees_with_itself():
    torch.manual_seed(2)
    net = NN.OperativeNet()
    setup, raw = setup_of(6, seed=7)
    B, N = raw.shape[:2]
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    term, st = TC.loss(net, inp, ctx, setup.destroy)
    assert torch.isfinite(term) and float(term) > 0
    term.backward()
    assert net.heads.where_b.weight.grad is not None and float(net.heads.where_b.weight.grad.abs().sum()) > 0
    assert st["comp"] is not None and st["cell"] is not None


def test_the_armies_cases_have_a_lord_first_and_the_formula_groups_them():
    sp = CS.specs(12, torch.Generator().manual_seed(9), fixed=True, cases=CS.ARMY)
    setup = CS.Setup(sp, P, "cpu")
    raw = TR.Seer(P).raw(setup.st)
    B, N = raw.shape[:2]
    lords = cfg()["cases"]["army"]["lords"]
    for b, s in enumerate(sp):
        n = int(setup.own[b].sum())
        assert 3 <= n <= 8
        assert setup.st.keys[b][0] == lords[s["ours"]] and setup.st.keys[b][CS.H] == lords[s["theirs"]]
        assert bool(setup.st.u["lord"][b, 0]) and bool(setup.st.u["lord"][b, CS.H])
        assert float(setup.st.u["range"][b, 0]) == 0.0                   # the lord: a melee unit of its group
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    out = TC.play_formula(inp, ctx, setup.destroy, "formula")
    for b, s in enumerate(sp):
        mem = out["new"].members()[b] & out["new"].on[b][:, None]
        assert bool((mem.sum(0)[setup.own[b]] == 1).all())               # every unit of mine in one group
        if s["case"] == "split_a":
            assert int(out["new"].on[b].sum()) >= 2
    tac = LW.Tacticians(B, N, int(cfg()["groups"]), "cpu")
    p, idx = tac.apply(out["new"], out["task"], out["moved"], raw)
    assert not any(PC.check(raw, p, EN.field_dict(axes, None)))


def big_battle(n=80, seed=0):
    """One battle of n + n units (the mod's 80 a side): lords first, rows of infantry and shooters 40 m apart."""
    from tools.nn.sim import scenario
    from tools.nn.tactic import cfg as tcfg
    from tools.nn.unit import drill_hold as DH
    g = torch.Generator().manual_seed(seed)
    roster = tcfg()["drill1"]["roster"]
    lords = cfg()["cases"]["army"]["lords"]

    def side(f, z0, b):
        keys = [lords[f]] + [roster[f]["infantry" if i % 3 else "shooters"][int(torch.randint(2, (1,), generator=g))]
                             for i in range(n - 1)]
        out = []
        for i, k in enumerate(keys):
            r, c = divmod(i, 20)
            out.append({"key": k, "x": (c - 9.5) * 45.0, "z": z0 + (r * 40.0 if z0 > 0 else -r * 40.0), "b": b})
        out[0]["general"] = True
        return {"faction": DH.FACTIONS[f], "units": out}
    army = {"attacker": 2, "sides": {1: side("emp", -250.0, 0.0), 2: side("skv", 250.0, 180.0)}}
    st = scenario.build([army], P, device="cpu", per_side=n)
    return st


def test_forty_groups_and_eighty_a_side_walk_and_send_correct_parcels():
    import time
    G = int(cfg()["groups"])
    assert G == 40 and GR.PREV == 2 * (G + 1)
    st = big_battle()
    from tools.nn.unit import train as UT
    raw = TR.Seer(P).raw(st)
    B, N = raw.shape[:2]
    assert N == 160
    side = torch.ones(B, dtype=torch.long)
    card = UT.cards(st, "cpu").float()
    axes = FR.Axes.of(raw, side, st.bounds)
    from tools.nn.tactic.decide import order_vec
    cmd = order_vec(B, "hold", "any")
    torch.manual_seed(3)
    net = NN.OperativeNet()
    groups = GR.Groups.empty(B, N)
    t0 = time.time()
    for _ in range(2):
        inp = IN.build(raw, card, side, axes, groups, cmd, bounds=st.bounds)
        assert inp["rows"].shape[1] == int(cfg()["rows"]) == 160
        ctx = NN.ctx_of(inp, raw, axes, groups)
        with torch.no_grad():
            out, _ = net(inp, ctx)
        new = out["new"]
        assert int(new.on.sum()) <= G
        mem = new.members() & new.on[..., None]
        assert int(mem.sum(1).max()) <= 1 and int(mem.sum(2).max()) <= 10
        tac = LW.Tacticians(B, N, G, "cpu")
        p, idx = tac.apply(new, out["task"], out["moved"], raw)
        assert not any(PC.check(raw, p, EN.field_dict(axes, None)))
        groups = new
    assert time.time() - t0 < 600


def test_the_operative_takes_the_strategists_six_groups():
    """A Groups of fewer than config groups (the strategist's 6): the walk and the formula follow it, the rows pad."""
    torch.manual_seed(4)
    net = NN.OperativeNet()
    setup, raw = setup_of(6, seed=3)
    B, N = raw.shape[:2]
    groups = GR.Groups.empty(B, N, G=6)
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    assert inp["prev"].shape[-1] == GR.PREV and inp["tokens"].shape[1] == 6
    ctx = NN.ctx_of(inp, raw, axes, groups)
    with torch.no_grad():
        out, _ = net(inp, ctx)
    assert out["new"].G == 6 and out["act"]["keep"].shape[1] == 6
    f = TC.play_formula(inp, ctx, setup.destroy, "formula")
    assert f["new"].G == 6 and bool(f["new"].on.any())


def test_groups_on_one_target_stand_side_by_side_and_others_keep_80m():
    """zone_sep_m: only a group with no common enemy is kept 80 m off; two groups on one target may stand together."""
    setup, raw = setup_of(3, seed=5, cases=("go",))
    B, N = raw.shape[:2]
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    act, _ = TC.formula(inp, ctx, setup.destroy)
    # every own unit its own group, all on the same enemy, every goal on the same cell
    K = ctx.slot.shape[1]
    own = ctx.own[0].nonzero()[:, 0]
    for k, j in enumerate(own.tolist()):
        act["comp"][:, j] = k
    act["target"][:, :len(own)] = act["target"][:, :1]
    act["cell"][:, :len(own)] = act["cell"][:, :1]
    out = NN.choose(NN.null_enc(B, K, groups.G, "cpu"), inp, ctx, NN.NullHeads(B, K, groups.G, "cpu"), act=act)
    on = out["new"].on
    assert int(on[0].sum()) == len(own)
    xs, zs = out["new"].cx[0, on[0]], out["new"].cz[0, on[0]]
    d = torch.sqrt((xs[:, None] - xs[None]) ** 2 + (zs[:, None] - zs[None]) ** 2)
    assert float(d.max()) < 80.0                       # (one target: side by side, no ban; the formula's cell kept)


def test_a_checkpoint_of_another_group_count_does_not_load(tmp_path):
    from tools.nn.operative import train as OT
    net, critic = NN.OperativeNet(), NN.OperativeCritic()
    sd = net.state_dict()
    sd["enc.number"] = torch.zeros(6, sd["enc.number"].shape[1])
    path = tmp_path / "old.pt"
    torch.save({"kind": OT.KIND, "net": sd, "critic": critic.state_dict()}, path)
    with pytest.raises(ValueError, match="6 groups"):
        OT.restore(str(path))


def test_the_zone_separation_bans_only_another_target_within_80m():
    B, N, G = 1, 6, 3
    g = GR.Groups.empty(B, N, G=G)
    g.on[0, :2] = True
    g.cx[0, 0], g.cz[0, 0] = 0.0, 0.0                   # group 0: target slot 3, zone at the origin
    g.target[0, 0, 3] = True
    g.cx[0, 1], g.cz[0, 1] = 500.0, 0.0                 # group 1: no target, zone at x 500
    other = torch.tensor([[True, True, False]])
    cx = torch.tensor([[50.0, 79.0, 81.0, 500.0, 530.0]])
    cz = torch.zeros_like(cx)
    sep = float(cfg()["zone_sep_m"])
    assert sep == 80.0
    t_other = torch.zeros((B, N), dtype=torch.bool)
    t_other[0, 4] = True                                # another target (no enemy in common)
    ok = NN.sep_ok(cx, cz, g, t_other, other, sep)
    assert ok.tolist() == [[False, False, True, True, True]]     # beside group 0 closed; beside the target-less open
    t_same = t_other.clone()
    t_same[0, 3] = True                                 # one enemy in common: side by side allowed
    assert bool(NN.sep_ok(cx, cz, g, t_same, other, sep).all())
    none = torch.zeros((B, N), dtype=torch.bool)         # this group holds without a target: anywhere
    assert bool(NN.sep_ok(cx, cz, g, none, other, sep).all())
