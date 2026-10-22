"""The general's repairs after the module atlas of g17r (docs/en/training/general.md): the groups by the rule
(groups.py), the executor's own kind table, the KL's k3 estimate, the tiles' standardised and v2 numbers, the groups'
threat numbers, the task without the group card, older checkpoints' old behaviour, the branches' other kinds with their
own targets. torch: the training container, on the CPU."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import army_input as ai                  # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import groups as ggroups                 # noqa: E402
from tools.nn.general import learn as glearn                   # noqa: E402
from tools.nn.general import net as gnet                       # noqa: E402
from tools.nn.general import tiles as gtiles                   # noqa: E402
from tools.nn.model import passport as pp                      # noqa: E402

CFG = gconfig.GeneralConfig(card=32, reader_hidden=32, council_heads=4)
SECT = 4
G, S = CFG.places, SECT * SECT
CASTE = ai.LABELS.index("caste")


def army(P=1):
    """Our 8 units and the enemy's 8 (module test layout), places in m: our shooters in a line, melee 300 m on with
    the lord 36 m from its centre, a horse 500 m aside; the enemy alike but its lord 600 m from everyone."""
    spec = [("missile_infantry", 0, 0), ("missile_infantry", 30, 0), ("missile_infantry", 60, 0),
            ("melee_infantry", 0, 300), ("melee_infantry", 40, 300), ("melee_infantry", 80, 300),
            ("lord", 20, 330), ("melee_cavalry", 500, 0)]
    foe = [(c, x, -y - 100) for c, x, y in spec[:6]] + [("lord", 0, -900), ("melee_cavalry", -500, -100)]
    rows = [(c, x, y, True) for c, x, y in spec] + [(c, x, y, False) for c, x, y in foe]
    N = len(rows)
    num = torch.zeros(P, N, ai.SIZE, dtype=torch.float64)
    lab = torch.zeros(P, N, len(ai.LABELS), dtype=torch.long)
    pos = torch.zeros(P, N, len(gnet.POS_FEATURES), dtype=torch.float64)
    own = torch.zeros(P, N, dtype=torch.bool)
    for i, (c, x, y, mine) in enumerate(rows):
        lab[:, i, CASTE] = pp.CASTES.index(c)
        num[:, i, ai.INDEX["run"]] = 0.9 if "cavalry" in c else 0.45
        num[:, i, ai.INDEX["range"]] = 0.3 if c.startswith("missile") else 0.0
        num[:, i, ai.INDEX["cost"]] = 0.7
        num[:, i, ai.INDEX["hp_come"]] = 1.0
        pos[:, i, 0], pos[:, i, 1], pos[:, i, -1] = x / 500.0, y / 500.0, 1.0
        own[:, i] = mine
    return num, lab, pos, own


def inputs(P=2):
    num, lab, pos, own = army(P)
    N = num.shape[1]
    on = torch.ones(P, N, dtype=torch.bool)
    g = torch.Generator().manual_seed(1)
    return {"num": num, "lab": lab, "pos": pos, "on": on, "own": own, "free": on.clone(),
            "tiles": torch.rand(P, S, 8, generator=g, dtype=torch.float64), "tile_ok": torch.ones(P, S, dtype=torch.bool),
            "zone": torch.zeros(P, S, dtype=torch.bool), "deploy": torch.zeros(P, dtype=torch.bool),
            "prev": torch.zeros(P, 2 * G, CFG.card, dtype=torch.float64), "has_prev": torch.zeros(P, dtype=torch.bool),
            "prev_main": torch.full((P, N), -1, dtype=torch.long)}


def model(cfg=CFG, seed=0):
    torch.manual_seed(seed)
    return gnet.General(cfg, SECT).double().eval()


def test_the_rule_groups_by_role_then_place_and_a_lone_lord_joins_the_melee_near():
    num, lab, pos, own = army()
    free = torch.ones_like(own)
    raw = ggroups.collect(num, lab, pos, own, free, G)[0].tolist()
    ours, theirs = raw[:8], raw[8:]
    assert ours[0] == ours[1] == ours[2]                          # shooters together
    assert ours[3] == ours[4] == ours[5] == ours[6]               # melee with the lord 36 m off
    assert len({ours[0], ours[3], ours[7]}) == 3                  # the horse apart
    assert theirs[6] not in theirs[:6] + theirs[7:]               # their lord far: his own group
    assert len(set(theirs)) == 4 and all(G <= t < 2 * G for t in theirs) and all(0 <= t < G for t in ours)
    roles = ggroups.roles(num, lab)[0].tolist()
    assert roles[:8] == [0, 0, 0, 1, 1, 1, 2, 3]


def test_the_rule_keeps_at_most_the_places_a_side_and_leaves_the_routing_out():
    num, lab, pos, own = army()
    pos = pos.clone()
    pos[0, :8, 0] = torch.arange(8, dtype=torch.float64) * 0.8      # our units 400 m apart: 8 clusters by place
    free = torch.ones_like(own)
    free[0, 2] = False
    raw = ggroups.collect(num, lab, pos, own, free, 3)[0]
    assert raw[2] == -1
    assert len(set(raw[:8].tolist()) - {-1}) == 3


def test_rule_groups_in_the_net_learn_nothing_of_the_attention_and_recompute_alike():
    m, inp = model(), inputs()
    c, d = m(inp)
    raw = ggroups.collect(inp["num"], inp["lab"], inp["pos"], inp["own"], inp["free"], G)
    assert torch.equal(c["main"], gnet.renumber(c["perm"], raw))
    assert torch.equal(c["shares"].argmax(-1), c["main"].clamp(min=0))
    (d["kind_logp"].exp() * torch.arange(CFG.kinds, dtype=torch.float64)).sum().backward()
    names = dict(m.named_parameters())
    for n in ("groups.k.weight", "groups.q.weight", "groups.k_pos.weight", "groups.log_sharp", "groups.center",
              "groups.stick"):
        assert names[n].grad is None, n
    assert names["groups.gru.weight_ih"].grad is not None
    with torch.no_grad():
        c2 = m.cards(dict(inp, main=c["main"], perm=c["perm"]))
    torch.testing.assert_close(c2["groups"], c["groups"].detach(), rtol=0, atol=1e-12)


def test_the_executors_own_kind_table_cuts_its_gradient_from_the_general():
    for split, task_group in ((True, False), (True, True), (False, True)):
        m = model(dataclasses.replace(CFG, split_kinds=split, task_group=task_group))
        inp = inputs()
        c, d = m(inp)
        mix, kc = m.unit_cards(c, d["kind"], c["main"])
        assert bool((mix == 0).all()) != task_group
        (mix.square().sum() + kc.sum()).backward()
        grads = {n for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        if split:
            assert grads == {"exec_kinds"}, grads
        else:
            assert {"kinds", "reader.layer1.weight", "groups.gru.weight_ih"} <= grads and "exec_kinds" not in grads


def test_k3_is_never_below_zero_and_zero_only_at_the_same_policy():
    ref = torch.tensor([-1.0, -2.0, -0.1, -3.0])
    now = torch.tensor([-0.2, -2.0, -2.5, -1.0])
    k3 = glearn.kl_est(ref, now, "k3")
    assert bool((k3 >= 0).all()) and float(k3[1]) == 0.0
    assert float(glearn.kl_est(ref, now, "k1").mean()) < 0        # the old estimate goes below zero here
    with pytest.raises(ValueError):
        glearn.LearnConfig(kl="k2")


def test_the_tiles_are_standardised_over_the_map():
    m, inp = model(), inputs()
    ok = inp["tile_ok"].clone()
    ok[:, :3] = False
    t = m.tile_numbers(inp["tiles"], ok)
    assert bool((t[:, :3] == 0).all())
    torch.testing.assert_close(t[:, 3:].mean(1), torch.zeros(2, 8, dtype=torch.float64), rtol=0, atol=1e-9)
    torch.testing.assert_close(t[:, 3:].std(1, unbiased=False), torch.ones(2, 8, dtype=torch.float64), rtol=0,
                               atol=1e-2)
    off = model(dataclasses.replace(CFG, tile_norm=False))
    assert torch.equal(off.tile_numbers(inp["tiles"], ok), inp["tiles"])


def test_the_v2_tile_numbers_put_the_shooters_and_the_fast_where_they_stand():
    from tools.nn.model import config as mconfig
    mcfg = mconfig.preset("v2")
    pos = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [300.0, 300.0], [-300.0, -300.0]]])
    own = torch.tensor([[True, True, False, False]])
    alive = torch.ones_like(own)
    vis = torch.tensor([[True, True, True, False]])
    shooter = torch.tensor([[True, False, True, True]])
    fast = torch.tensor([[False, False, True, False]])
    cost, hp = torch.full((1, 4), 500.0), torch.ones(1, 4)
    ex = gtiles.extra(mcfg, pos, own, alive, vis, cost, hp, shooter, fast)
    assert ex.shape == (1, mcfg.sectors ** 2, 3)
    assert int((ex[0, :, 0] > 0).sum()) == 1 and int((ex[0, :, 1] > 0).sum()) == 1   # the hidden shooter not
    assert torch.equal(ex[0, :, 1] > 0, ex[0, :, 2] > 0)          # the seen enemy shooter is the fast one
    assert list(gtiles.EXTRA) == [4, 5, 6]


def test_the_threat_numbers_start_at_zero_weight_and_read_the_nearest_enemy_group():
    on, off = model(), model(dataclasses.replace(CFG, threat=False))
    inp = inputs()
    with torch.no_grad():
        c, d = on(inp, greedy=True)
        _, d0 = off(inp, greedy=True)
        torch.testing.assert_close(d["kind_logp"], d0["kind_logp"], rtol=0, atol=0)
        t = on.threat(inp, c)
    assert t.shape == (2, G, len(gnet.THREAT))
    g_sh = c["main"][0, 0]                                        # our shooters' group
    assert float(t[0, g_sh, 2]) == 1.0                            # all shooters
    assert float(t[0, g_sh, 5]) == 1.0                            # an enemy group there is
    # the nearest enemy group is theirs shooters' (100 m behind ours): distance 100 / THREAT_D, all shooters
    assert abs(float(t[0, g_sh, 0]) - 100.0 / gnet.THREAT_D) < 1e-4 and float(t[0, g_sh, 3]) == 1.0
    assert bool((t[0][~c["active"][0, :G]] == 0).all())


def test_an_older_checkpoint_keeps_its_old_behaviour_and_loads_its_kind_table_twice():
    old = {k: v for k, v in gconfig.to_dict(CFG).items() if k not in gconfig.LEGACY}
    o = gconfig.of_dict(old, warn=False)
    assert all(getattr(o, k) == v for k, v in gconfig.LEGACY.items())
    assert gconfig.of_dict(gconfig.to_dict(CFG)) == CFG
    from tools.nn.general import executor
    from tools.nn.model import config as mconfig, policy
    tiny = mconfig.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                          sector_d=16)
    gcfg = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
    torch.manual_seed(0)
    a = executor.wrap(policy.Actor(tiny), gcfg)
    state = {k: v for k, v in a.state_dict().items() if not k.startswith(("general.exec_kinds", "general.threat_in"))}
    torch.manual_seed(1)
    b = executor.GeneralActor(tiny, gcfg)
    b.load_state_dict(state)
    assert torch.equal(b.general.exec_kinds, a.general.kinds) and torch.equal(b.general.kinds, a.general.kinds)
    assert bool((b.general.threat_in.weight == 0).all())


def test_the_branches_other_kinds_take_their_own_targets():
    from tools.nn.general import branch as gbranch
    from tests.tools.test_general_branch_torch import grabbed, GCFG as BCFG
    env, actor, snap, cand = grabbed()
    rt = env.general_rt
    rows = torch.tensor([(s - 1) * env.B + b for b, s, _, _ in cand])
    group = torch.tensor([g for _, _, g, _ in cand])
    K = BCFG.kinds
    alt = torch.arange(K).repeat(len(rows), 1)
    target, tgt = gbranch.alt_targets(rt, rows, group, alt)
    pair = actor.general.pair_ok
    ok = target >= 0
    assert bool(ok.any())
    assert bool(pair[alt[ok], target[ok]].all())                  # every kind with a target of its type
    C = BCFG.target_grid ** 2
    tile = ok & (target >= 2 * BCFG.places) & (target < 2 * BCFG.places + C)
    assert bool(torch.isfinite(tgt[tile]).all()) and bool(torch.isnan(tgt[ok & ~tile]).all())
    # the group's own target kept where the kind takes it
    cur = rt.r["target"][rows].gather(1, group[:, None])[:, 0]
    keep = pair[alt, cur[:, None].expand_as(alt)]
    assert torch.equal(target[keep], cur[:, None].expand_as(alt)[keep])
    with pytest.raises(ValueError):
        gbranch.Config(kinds="some")


def test_a_rule_groups_card_reads_how_its_units_differ_from_their_side():
    """GeneralConfig.group_centre: the units' values less their side's mean - a shift common to a side's units does not
    change its groups' cards (the value layer and its norm made identities here)."""
    m = model()
    m.groups.v, m.groups.norm_in = torch.nn.Identity(), torch.nn.Identity()
    inp = inputs(P=1)
    N = inp["own"].shape[1]
    g = torch.Generator().manual_seed(3)
    u = torch.randn(1, N, CFG.card, generator=g, dtype=torch.float64)
    main = ggroups.collect(inp["num"], inp["lab"], inp["pos"], inp["own"], inp["free"], G)
    shift = u + 5.0 * inp["own"][..., None].to(u.dtype)
    with torch.no_grad():
        _, a = m.groups.fixed(u, main, own=inp["own"])
        _, b = m.groups.fixed(shift, main, own=inp["own"])
        _, c = m.groups.fixed(shift, main)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-9)
    assert not torch.allclose(a[:, :G], c[:, :G])
