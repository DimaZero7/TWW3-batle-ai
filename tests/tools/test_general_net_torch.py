"""tools/nn/general/net.py: the general's network (S28 skeleton): masks, permutations, the choices' rules
(docs/en/training/general.md "Checks"). torch: runs in the training container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import army_input as ai                  # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import net as gnet                       # noqa: E402

CFG = gconfig.GeneralConfig(card=32, reader_hidden=32, council_heads=4)
SECT = 4                                                       # 4 x 4 tiles: small tests
G, S = CFG.places, SECT * SECT


def model(seed=0):
    torch.manual_seed(seed)
    return gnet.General(CFG, SECT).double().eval()


def inputs(P=3, N=12, seed=1, own_n=6):
    g = torch.Generator().manual_seed(seed)
    num = torch.rand(P, N, ai.SIZE, generator=g, dtype=torch.float64)
    lab = torch.stack([torch.randint(0, 4, (P, N), generator=g), torch.randint(0, 12, (P, N), generator=g),
                       torch.randint(0, 4, (P, N), generator=g)], -1)
    own = torch.zeros(P, N, dtype=torch.bool)
    own[:, :own_n] = True
    on = torch.ones(P, N, dtype=torch.bool)
    on[:, -2:] = False                                            # two empty slots
    free = on.clone()
    free[:, 1] = False                                            # a routing unit: at the council, in no group
    tiles = torch.rand(P, S, 8, generator=g, dtype=torch.float64)
    tile_ok = torch.ones(P, S, dtype=torch.bool)
    tile_ok[:, :2] = False                                        # outside the map
    zone = torch.zeros(P, S, dtype=torch.bool)
    zone[:, 4:8] = True
    return {"num": num, "lab": lab, "on": on, "own": own, "free": free, "tiles": tiles, "tile_ok": tile_ok,
            "zone": zone, "deploy": torch.ones(P, dtype=torch.bool), "prev": torch.zeros(P, 2 * G, CFG.card,
                                                                                         dtype=torch.float64),
            "has_prev": torch.zeros(P, dtype=torch.bool)}


def test_the_shares_keep_the_side_and_add_up_to_one():
    m, inp = model(), inputs()
    with torch.no_grad():
        c = m.cards(inp)
    sh = c["shares"]
    own, free = inp["own"], inp["free"]
    assert torch.all(sh[..., G:][own] < 1e-6) and torch.all(sh[..., :G][~own] < 1e-6)         # the side mask
    assert torch.allclose(sh.sum(-1)[free], torch.ones_like(sh.sum(-1)[free]))               # 100 % of a unit
    assert torch.all(sh[~free] == 0)                                                           # dead, empty, routing
    assert torch.all(c["main"][~free] == -1)
    assert torch.all((c["main"][own & free] >= 0) & (c["main"][own & free] < G))
    assert torch.all(c["main"][~own & free] >= G)
    # a place is a group when some unit's main place: the active ones
    for p in range(inp["on"].shape[0]):
        assert set(torch.nonzero(c["active"][p]).flatten().tolist()) == set(c["main"][p][c["main"][p] >= 0].tolist())


def test_the_keys_less_their_sides_mean_make_alike_units_split_evenly():
    """GeneralConfig.center: the group collection's keys less their side's mean - units that are all alike have no
    difference to compete by, every place gets the same share of each; plain slot attention (center 0) puts them by
    the common key (a fresh general's single group)."""
    import dataclasses
    inp = inputs(P=2)
    for k in ("num", "lab"):
        inp[k][:, 1:6] = inp[k][:, :1]                               # our 6 units alike (unit 1 routes: not free)
    inp["free"][:, 1] = True
    torch.manual_seed(0)
    m = gnet.General(CFG, SECT).double().eval()
    with torch.no_grad():
        sh = m.cards(inp)["shares"][:, :6, :G]
    torch.testing.assert_close(sh, torch.full_like(sh, 1.0 / G), rtol=0, atol=1e-9)
    torch.manual_seed(0)
    plain = gnet.General(dataclasses.replace(CFG, center=0.0), SECT).double().eval()
    assert plain.groups.center is None
    with torch.no_grad():
        sp = plain.cards(inp)["shares"][:, :6, :G]
    assert float((sp - 1.0 / G).abs().max()) > 1e-3


def test_empty_slots_change_nothing():
    m, inp = model(), inputs()
    big = {k: v.clone() for k, v in inp.items()}
    big["num"][:, -2:] = 7.0                                       # garbage in the empty slots
    with torch.no_grad():
        a = m.cards(inp)
        b = m.cards(big)
    for k in ("groups", "tiles", "shares"):
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=1e-12)


def test_a_permutation_of_the_slots_permutes_the_units_and_changes_nothing_else():
    m, inp = model(), inputs()
    perm = torch.tensor([5, 0, 3, 1, 4, 2, 9, 6, 11, 7, 10, 8])
    pin = {k: (v[:, perm] if k in ("num", "lab", "on", "own", "free") else v) for k, v in inp.items()}
    with torch.no_grad():
        a, da = m(inp, greedy=True)
        b, db = m(pin, greedy=True)
    torch.testing.assert_close(b["shares"], a["shares"][:, perm], rtol=0, atol=1e-10)
    torch.testing.assert_close(b["groups"], a["groups"], rtol=0, atol=1e-10)
    torch.testing.assert_close(b["main"], a["main"][:, perm])
    for k in ("kind", "t1", "t2", "place"):
        assert torch.equal(db[k], da[k]), k
    torch.testing.assert_close(db["lp"], da["lp"], rtol=0, atol=1e-10)
    torch.testing.assert_close(db["value"], da["value"], rtol=0, atol=1e-10)


def test_the_choices_keep_their_rules():
    """Only our groups choose; a target is an active group (not itself), an inside tile, or none (target 2), never the
    same twice; a place is a zone tile, only when the plan deploys."""
    m, inp = model(), inputs(P=4)
    torch.manual_seed(3)
    for trial in range(30):
        with torch.no_grad():
            c, d = m(inp)
        act, ok = c["active"], inp["tile_ok"]
        mine = act[:, :G]
        assert torch.all(d["lp"][~mine] == 0) and torch.all(d["ent"][~mine] == 0)
        for p in range(4):
            for g in torch.nonzero(mine[p]).flatten().tolist():
                for t in (int(d["t1"][p, g]), int(d["t2"][p, g])):
                    if t == 2 * G + S:
                        continue                                    # none
                    assert t != g
                    if t < 2 * G:
                        assert bool(act[p, t])
                    else:
                        assert bool(ok[p, t - 2 * G])
                assert int(d["t1"][p, g]) != 2 * G + S              # target 1 is never none
                assert int(d["t1"][p, g]) != int(d["t2"][p, g])
                assert bool(inp["zone"][p, int(d["place"][p, g])])
    no = dict(inp, deploy=torch.zeros(4, dtype=torch.bool))
    with torch.no_grad():
        _, d = m(no)
    assert torch.all(d["place"] == -1)


def test_the_log_probability_under_the_choices_made_is_the_sampled_ones():
    m, inp = model(), inputs()
    torch.manual_seed(5)
    with torch.no_grad():
        c, d = m(inp)
        c2, d2 = m(dict(inp, main=c["main"]), act={k: d[k] for k in gnet.CHOICES})
    torch.testing.assert_close(d2["lp"], d["lp"], rtol=0, atol=1e-10)
    torch.testing.assert_close(d2["value"], d["value"], rtol=0, atol=1e-10)


def test_the_unit_cards_carry_the_gradient_to_the_group_collection_and_the_readers():
    m, inp = model(), inputs()
    c, d = m(inp)
    mix, kc = m.unit_cards(c, d["kind"], c["main"])
    assert torch.all(mix[~(inp["own"] & inp["free"])] == 0) and torch.all(kc[~(inp["own"] & inp["free"])] == 0)
    (mix.square().sum() + kc.sum()).backward()
    for name in ("reader.layer1.weight", "groups.k.weight", "groups.gru.weight_ih", "listen.0.q.weight", "kinds",
                 "council.0.qkv.weight", "tile_in.0.weight"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name


def test_a_replan_starts_from_the_last_places():
    m, inp = model(), inputs()
    with torch.no_grad():
        c = m.cards(inp)
        again = m.cards(dict(inp, prev=c["places"], has_prev=torch.ones(3, dtype=torch.bool)))
        fresh = m.cards(inp)
    assert not torch.allclose(again["places"], fresh["places"])
    torch.testing.assert_close(fresh["places"], c["places"], rtol=0, atol=0)


def test_the_skeletons_simplifications_are_settings_and_the_missing_ones_are_refused():
    with pytest.raises(gconfig.NotBuilt):
        gconfig.GeneralConfig(tiles="cnn")
    with pytest.raises(gconfig.NotBuilt):
        gconfig.GeneralConfig(effects="on")
    with pytest.raises(gconfig.NotBuilt):
        gconfig.GeneralConfig(zone="map")
    assert gconfig.GeneralConfig(deploy="script").deploy == "script"
    from tools.nn.general import learn
    with pytest.raises(gconfig.NotBuilt):
        learn.LearnConfig(branches=0.1)
    with pytest.raises(gconfig.NotBuilt):
        learn.LearnConfig(distinct=0.1)
    c = gconfig.GeneralConfig(card=96)
    assert gconfig.of_dict(gconfig.to_dict(c)) == c and c.task_dim == 2 * 96 + 8
