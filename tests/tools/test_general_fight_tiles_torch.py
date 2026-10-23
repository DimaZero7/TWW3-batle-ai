"""The general's group cards read the tiles under the groups (GeneralConfig.group_tiles, net.General.group_tiles) and
the fight's numbers (net.FIGHT with GeneralConfig.threat: the shares under fire and in melee; Runtime.fight_of of the
state's flags); both layers start at zero, an older checkpoint loads them so with a warning (docs/en/training/general.md).
torch: the training container, on the CPU."""
import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import net as gnet                       # noqa: E402
from tools.nn.general import runtime as gruntime               # noqa: E402
from tests.tools.test_general_repairs_torch import CFG, G, S, SECT, inputs, model   # noqa: E402


def test_the_tiles_under_a_group_are_its_centres_neighbours_weighted_by_distance():
    m = model()
    inp = inputs()
    with torch.no_grad():
        c = m.cards(inp)
        gt = m.group_tiles(inp, c)
        cent, n, _ = m.centres(inp, c)
    assert gt.shape == (2, 2 * G, 8)
    feats = m.tile_numbers(inp["tiles"].float(), inp["tile_ok"])
    tc = m.tile_pos[:, :2].float() * gnet.POS_M
    step = m.tile_step
    for g in range(2 * G):
        if not bool(c["active"][0, g]):
            assert bool((gt[0, g] == 0).all())
            continue
        # by hand: the tiles within 1.5 steps of the centre, exp(-d^2 / 2 step^2)
        w = []
        for s in range(S):
            d = float((cent[0, g] - tc[s]).norm()) / step
            w.append(math.exp(-0.5 * d * d) if d <= gnet.GROUP_TILE_R else 0.0)
        w = torch.tensor(w)
        assert float(w.sum()) > 0
        ref = (w[:, None] * feats[0]).sum(0) / w.sum()
        torch.testing.assert_close(gt[0, g], ref, rtol=1e-5, atol=1e-5)
        # a tile farther than 1.5 steps from the centre is not read
        far = int(torch.argmax(torch.where(w == 0, torch.arange(S, dtype=torch.float32), torch.tensor(-1.0))))
        if float(w[far]) == 0.0:
            bumped = dict(inp, tiles=inp["tiles"].clone())
            bumped["tiles"][0, far] += 1.0
            # (the standardisation over the map moves every tile a little: compare with it switched off)
            m2 = model(dataclasses.replace(CFG, tile_norm=False))
            with torch.no_grad():
                a = m2.group_tiles(inp, c)[0, g]
                b = m2.group_tiles(bumped, c)[0, g]
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_group_tiles_start_at_zero_and_then_change_the_kind_by_a_direct_path():
    on, off = model(), model(dataclasses.replace(CFG, group_tiles=False))
    inp = inputs()
    blank = dict(inp, tiles=torch.zeros_like(inp["tiles"]))
    with torch.no_grad():
        _, d = on(inp, greedy=True)
        _, d0 = off(inp, greedy=True)
    torch.testing.assert_close(d["kind_logp"], d0["kind_logp"], rtol=0, atol=0)
    # the layer at zero still gets a gradient from the kind's choice (its weights learn from the first update)
    c, d = on(inp, greedy=True)
    d["kind_logp"][..., 0].sum().backward()
    assert float(on.gtile_in.weight.grad.abs().sum()) > 0
    # once it has weights, the tiles change the kind's choice (blank tiles: another distribution)
    with torch.no_grad():
        on.gtile_in.weight.normal_(0, 1.0)
        _, d1 = on(inp, greedy=True)
        _, d2 = on(blank, greedy=True)
    mine = d1["kind_ok"].any(-1)
    diff = (d1["kind_logp"] - d2["kind_logp"]).abs().amax(-1)[mine]
    assert float(diff.max()) > 1e-2


def test_the_targets_keys_read_the_tiles_under_the_enemy_groups():
    m = model()
    inp = inputs()
    blank = dict(inp, tiles=torch.zeros_like(inp["tiles"]))
    with torch.no_grad():
        m.gtile_in.weight.normal_(0, 1.0)
        # the same kind for both: only the targets' step is compared
        c1, d1 = m(inp, greedy=True)
        _, d2 = m(blank, act={"kind": d1["kind"], "target": d1["target"], "place": d1["place"]})
    enemy = slice(G, 2 * G)
    ok = d1["target_ok"][..., enemy]
    a, b = d1["target_logits"][..., enemy][ok], d2["target_logits"][..., enemy][ok]
    if a.numel():
        assert float((a - b).abs().max()) > 1e-3


def fight_state(N=6):
    """Units 0-2 ours (side 1), 3-5 theirs: 3 fires at 0, 4 fires at 1 but is in melee, 5 fires with no target; our
    2 fires at 5; 0 and 3 in melee."""
    side = torch.tensor([[1, 1, 1, 2, 2, 2]])
    fire = torch.tensor([[False, False, True, True, True, True]])
    m = torch.tensor([[True, False, False, True, True, False]])
    target = torch.tensor([[-1, -1, 5, 0, 1, -1]])
    return {"side": side, "fire": fire, "m": m, "target": target}


def test_under_fire_is_a_firing_enemy_not_in_melee_with_the_unit_its_target():
    u = fight_state()
    alive = torch.ones(1, 6, dtype=torch.bool)
    f = gruntime.Runtime.fight_of(u, alive)
    assert f.shape == (1, 6, len(gnet.FIGHT))
    # unit 3 fires at 0 while in melee itself: not a volley (the game's and the simulator's shooter in melee does not
    # shoot); unit 4 likewise; our 2 fires at their 5
    assert f[0, :, 0].tolist() == [0, 0, 0, 0, 0, 1]
    assert f[0, :, 1].tolist() == [1, 0, 0, 1, 1, 0]
    u["m"] = torch.tensor([[True, False, False, False, True, False]])
    f = gruntime.Runtime.fight_of(u, alive)
    assert f[0, :, 0].tolist() == [1, 0, 0, 0, 0, 1]
    # a dead shooter shoots nobody, a dead target is not under fire
    dead = alive.clone()
    dead[0, 3] = False
    assert gruntime.Runtime.fight_of(u, dead)[0, 0, 0] == 0
    # a target of its own side is not under fire
    u["target"] = torch.tensor([[-1, -1, 1, 0, 1, -1]])
    assert gruntime.Runtime.fight_of(u, alive)[0, 1, 0] == 0


def test_the_groups_fight_shares_and_their_layer_from_zero():
    # (the melee rule off in both: here the layer from zero alone; melee_stay closes "withdraw" to fighting groups)
    on, off = model(dataclasses.replace(CFG, melee_stay=False)), model(dataclasses.replace(CFG, threat=False))
    inp = inputs()
    N = inp["on"].shape[1]
    fight = torch.zeros(2, N, 2, dtype=torch.float64)
    fight[:, 0, 0] = 1.0                                           # one of our three shooters under fire
    fight[:, 3:5, 1] = 1.0                                         # two of our four melee (with the lord) in melee
    fight[:, 8:, :] = 1.0                                          # the enemy's: not read
    with torch.no_grad():
        c, d = on(dict(inp, fight=fight), greedy=True)
        _, d0 = off(inp, greedy=True)
        f = on.fight(dict(inp, fight=fight), c)
    torch.testing.assert_close(d["kind_logp"], d0["kind_logp"], rtol=0, atol=0)
    sh, me = int(c["main"][0, 0]), int(c["main"][0, 3])
    assert abs(float(f[0, sh, 0]) - 1 / 3) < 1e-6 and float(f[0, sh, 1]) == 0
    assert abs(float(f[0, me, 1]) - 2 / 4) < 1e-6 and float(f[0, me, 0]) == 0
    assert bool((f[0][~c["active"][0, :G]] == 0).all())
    # without the input: zeros
    assert bool((on.fight(inp, c) == 0).all())


def test_every_input_has_its_record():
    rt = gruntime.Runtime.__new__(gruntime.Runtime)
    rt.G, rt.N, rt.S, rt.D, rt.dev, rt.gcfg = G, 16, S, CFG.card, torch.device("cpu"), CFG
    r = rt._rows(4)
    assert all("in/" + k in r for k in gnet.INPUTS)
    assert r["in/fight"].shape == (4, 16, len(gnet.FIGHT))


def test_an_older_general_loads_the_new_layers_at_zero_with_a_warning(capsys):
    from tools.nn.general import executor
    from tools.nn.model import config as mconfig, policy
    tiny = mconfig.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                          sector_d=16)
    gcfg = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
    torch.manual_seed(0)
    a = executor.wrap(policy.Actor(tiny), gcfg)
    state = {k: v for k, v in a.state_dict().items() if not k.startswith(executor.NEW_LAYERS)}
    executor._WARNED.clear()
    torch.manual_seed(1)
    b = executor.GeneralActor(tiny, gcfg)
    b.load_state_dict(state)
    b.load_state_dict(state)
    assert bool((b.general.fight_in.weight == 0).all()) and bool((b.general.gtile_in.weight == 0).all())
    assert torch.equal(b.general.kinds, a.general.kinds)
    out = capsys.readouterr().out
    assert "fight_in" in out and "gtile_in" in out and out.count("start fresh at zero") == 1
    assert gconfig.GeneralConfig().group_tiles
