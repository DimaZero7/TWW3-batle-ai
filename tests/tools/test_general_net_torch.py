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
    pos = torch.rand(P, N, len(gnet.POS_FEATURES), generator=g, dtype=torch.float64) * 2 - 1
    pos[..., -1] = 1.0                                            # known
    return {"num": num, "lab": lab, "pos": pos, "on": on, "own": own, "free": free, "tiles": tiles,
            "tile_ok": tile_ok, "zone": zone, "deploy": torch.ones(P, dtype=torch.bool),
            "prev": torch.zeros(P, 2 * G, CFG.card, dtype=torch.float64), "has_prev": torch.zeros(P, dtype=torch.bool),
            "prev_main": torch.full((P, N), -1, dtype=torch.long)}


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
    for k in ("num", "lab", "pos"):
        inp[k][:, 1:6] = inp[k][:, :1]                               # our 6 units alike (unit 1 routes: not free)
    inp["free"][:, 1] = True
    torch.manual_seed(0)
    m = gnet.General(dataclasses.replace(CFG, center=1.0), SECT).double().eval()
    with torch.no_grad():
        sh = m.cards(inp)["shares"][:, :6, :G]
    torch.testing.assert_close(sh, torch.full_like(sh, 1.0 / G), rtol=0, atol=1e-9)
    torch.manual_seed(0)
    plain = gnet.General(dataclasses.replace(CFG, center=0.0), SECT).double().eval()
    assert plain.groups.center is None
    with torch.no_grad():
        sp = plain.cards(inp)["shares"][:, :6, :G]
    assert float((sp - 1.0 / G).abs().max()) > 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_fresh_generals_groups_are_sharp_and_differ(seed):
    """GeneralConfig.sharp: with layer-normed keys and queries x the learned sharpness a fresh general gives every unit
    one clear main place (share > 0.8) and its groups' cards differ (mean cosine of a plan's groups < 0.9); the plain
    dot product (sharp 0) split every unit ~evenly (share ~0.1). And groups of several units come out (not one each)."""
    import dataclasses
    inp = inputs(P=8, N=28, seed=10 + seed, own_n=14)
    inp["prev"] = torch.zeros(8, 2 * G, 64, dtype=torch.float64)
    sel = inp["own"] & inp["free"]
    torch.manual_seed(seed)
    m = gnet.General(gconfig.GeneralConfig(card=64, reader_hidden=64), SECT).double().eval()
    with torch.no_grad():
        c = m.cards(inp)
    share = float(c["shares"][..., :G].max(-1).values[sel].mean())
    act = c["active"][:, :G]
    g = c["groups"][:, :G]
    cos = torch.nn.functional.cosine_similarity(g[:, :, None], g[:, None, :], dim=-1)
    pair = act[:, :, None] & act[:, None, :] & ~torch.eye(G, dtype=torch.bool)[None]
    assert share > 0.8, share
    assert float(cos[pair].mean()) < 0.9, float(cos[pair].mean())
    size = float(sel.sum()) / float(act.sum())
    assert size >= 1.5, size                       # (random units, all unlike; real armies: ~3.6, build/gfix)
    torch.manual_seed(seed)
    plain = gnet.General(gconfig.GeneralConfig(card=64, reader_hidden=64, sharp=0.0), SECT).double().eval()
    assert plain.groups.log_sharp is None and m.groups.log_sharp is not None
    with torch.no_grad():
        sp = plain.cards(inp)["shares"][..., :G].max(-1).values[sel].mean()
    assert float(sp) < 0.3
    assert dataclasses.replace(m.cfg).sharp == gconfig.GeneralConfig().sharp


def test_alike_units_at_different_places_are_told_apart_by_their_places():
    """The units' places (pos) reach the group collection: units alike in everything but where they stand do not
    split evenly over the places (they would without positions: the test above)."""
    inp = inputs(P=2)
    for k in ("num", "lab"):
        inp[k][:, 1:6] = inp[k][:, :1]
    inp["free"][:, 1] = True
    inp["pos"][:, 1:6] = inp["pos"][:, :1]                          # the same facing and velocity
    inp["pos"][:, :3, :2] =torch.tensor([-0.8, -0.8], dtype=torch.float64)     # three on the left ...
    inp["pos"][:, 3:6, :2] = torch.tensor([0.8, 0.8], dtype=torch.float64)      # ... three on the right
    torch.manual_seed(0)
    m = gnet.General(CFG, SECT).double().eval()
    with torch.no_grad():
        sh = m.cards(inp)["shares"][:, :6, :G]
    assert float((sh - 1.0 / G).abs().max()) > 1e-3
    # the three of a place have the same shares; the two places' differ
    torch.testing.assert_close(sh[:, 0], sh[:, 2], rtol=0, atol=1e-9)
    assert float((sh[:, 0] - sh[:, 4]).abs().max()) > 1e-4


def test_the_groups_keep_their_numbers_by_their_units():
    """match_groups: without a last plan the numbering is the identity; a last plan with the same units under other
    numbers gives the numbers back; the pair sharing more units wins; ours are matched only with ours."""
    G2 = 3
    main = torch.tensor([[0, 0, 1, 2, 3, 4, -1]])                    # ours 0..2, theirs 3..5
    ident = gnet.match_groups(main, torch.full_like(main, -1), G2)
    assert ident.tolist() == [list(range(6))]
    assert torch.equal(gnet.renumber(ident, main), main)
    prev = torch.tensor([[2, 2, 0, 1, 5, 3, -1]])                    # the same groups, other numbers
    perm = gnet.match_groups(main, prev, G2)
    assert torch.equal(gnet.renumber(perm, main), prev)
    assert sorted(perm[0].tolist()) == list(range(6)) and all(perm[0, :G2] < G2)
    # a group split: new place 0 has units {a, b}, new place 1 has {c}; the last plan had all three in group 1
    main = torch.tensor([[0, 0, 1, 3]])
    prev = torch.tensor([[1, 1, 1, 3]])
    got = gnet.renumber(gnet.match_groups(main, prev, G2), main)
    assert got[0, :2].tolist() == [1, 1] and int(got[0, 2]) not in (1,) and int(got[0, 3]) == 3


def test_a_replans_groups_are_numbered_by_the_last_plans_and_computed_again_alike():
    """A re-plan whose last plan had the same units under shuffled numbers: the new groups take the last plan's
    numbers; computed again under the plan's membership and numbering (the update), the cards and probabilities are
    the play's."""
    m, inp = model(), inputs()
    torch.manual_seed(5)
    with torch.no_grad():
        c0, _ = m(inp)
        shuffle = torch.tensor([3, 0, 1, 2, 4, 5, 6, 7, 8, 9] + [G + i for i in (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)])
        prev_main = torch.where(c0["main"] >= 0, shuffle[c0["main"].clamp(min=0)], c0["main"])
        again = dict(inp, prev_main=prev_main)
        c, d = m(again)
        assert torch.equal(c["main"], prev_main)                    # (same inputs: the same groups, the old numbers)
        c2, d2 = m(dict(again, main=c["main"], perm=c["perm"]), act={k: d[k] for k in gnet.CHOICES})
    for k in ("groups", "places", "shares"):
        torch.testing.assert_close(c2[k], c[k], rtol=0, atol=1e-10)
    torch.testing.assert_close(d2["lp"], d["lp"], rtol=0, atol=1e-10)


def test_a_replan_keeps_its_groups_units_by_the_stick():
    """GeneralConfig.stick: in a re-plan a unit's last group gets a bonus in the group collection - a strong one keeps
    every free unit where it was (on new inputs: the units moved); 0 has no parameter and the units go where the
    attention says."""
    import dataclasses
    inp = inputs(P=3)
    torch.manual_seed(0)
    m = gnet.General(dataclasses.replace(CFG, stick=50.0), SECT).double().eval()
    with torch.no_grad():
        c0 = m.cards(inp)
        moved = dict(inp, pos=inp["pos"].flip(1), prev=c0["places"], has_prev=torch.ones(3, dtype=torch.bool),
                     prev_main=c0["main"])
        c1 = m.cards(moved)
    assert torch.equal(c1["main"], c0["main"])
    torch.manual_seed(0)
    plain = gnet.General(dataclasses.replace(CFG, stick=0.0), SECT).double().eval()
    assert plain.groups.stick is None and m.groups.stick is not None


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
    pin = {k: (v[:, perm] if k in ("num", "lab", "pos", "on", "own", "free", "prev_main") else v)
           for k, v in inp.items()}
    with torch.no_grad():
        a, da = m(inp, greedy=True)
        b, db = m(pin, greedy=True)
    torch.testing.assert_close(b["shares"], a["shares"][:, perm], rtol=0, atol=1e-10)
    torch.testing.assert_close(b["groups"], a["groups"], rtol=0, atol=1e-10)
    torch.testing.assert_close(b["main"], a["main"][:, perm])
    for k in gnet.CHOICES:
        assert torch.equal(db[k], da[k]), k
    torch.testing.assert_close(db["lp"], da["lp"], rtol=0, atol=1e-10)
    torch.testing.assert_close(db["value"], da["value"], rtol=0, atol=1e-10)


def test_the_choices_keep_their_rules():
    """Only our groups choose; a target is an active group (ours not itself, or theirs) or a coarse tile with a tile
    inside the map (2G + target_grid^2 candidates); a place is a zone tile, only when the plan deploys."""
    m, inp = model(), inputs(P=4)
    C = CFG.target_grid ** 2
    torch.manual_seed(3)
    kinds = set()
    for trial in range(30):
        with torch.no_grad():
            c, d = m(inp)
        act, ok = c["active"], c["coarse_ok"]
        assert c["coarse"].shape[1] == C
        mine = act[:, :G]
        assert torch.all(d["lp"][~mine] == 0) and torch.all(d["ent"][~mine] == 0)
        assert d["ent_parts"].shape[-1] == 3
        for p in range(4):
            for g in torch.nonzero(mine[p]).flatten().tolist():
                t = int(d["target"][p, g])
                assert t != g and 0 <= t < 2 * G + C
                if t < 2 * G:
                    assert bool(act[p, t])
                else:
                    assert bool(ok[p, t - 2 * G])
                kinds.add(int(gnet.target_kind(d["target"][p, g], G, C)))
                assert bool(inp["zone"][p, int(d["place"][p, g])])
    assert kinds == {0, 1, 2}                                         # a fresh general picks groups too
    no = dict(inp, deploy=torch.zeros(4, dtype=torch.bool))
    with torch.no_grad():
        _, d = m(no)
    assert torch.all(d["place"] == -1)


def test_the_log_probability_under_the_choices_made_is_the_sampled_ones():
    m, inp = model(), inputs()
    torch.manual_seed(5)
    with torch.no_grad():
        c, d = m(inp)
        c2, d2 = m(dict(inp, main=c["main"], perm=c["perm"]), act={k: d[k] for k in gnet.CHOICES})
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
    assert learn.LearnConfig(branches=0.1).branches == 0.1                          # built (branch.py)
    with pytest.raises(ValueError):
        learn.LearnConfig(branches=1.5)
    assert learn.LearnConfig(distinct=0.1, distinct_decay=100).distinct == 0.1      # built (distinct.py)
    with pytest.raises(ValueError):
        learn.LearnConfig(distinct=-0.1)
    c = gconfig.GeneralConfig(card=96)
    assert gconfig.of_dict(gconfig.to_dict(c)) == c and c.task_dim == 2 * 96 + 4


def test_the_coarse_tiles_are_blocks_of_the_sectors():
    from tools.nn.general import tiles as gtiles
    from tools.nn.model import config as mconfig
    idx = gtiles.coarse_index(16, 4)
    assert idx.shape == (256,) and int(idx[0]) == 0 and int(idx[3]) == 0 and int(idx[4]) == 1
    assert int(idx[4 * 16]) == 4 and int(idx[255]) == 15 and torch.bincount(idx).tolist() == [16] * 16
    with pytest.raises(ValueError):
        gtiles.coarse_index(16, 5)
    cfg = mconfig.preset("v2")
    c = gtiles.coarse_centres(cfg, 4)
    step = 2 * cfg.grid_half / 4
    assert float(c[0, 0]) == pytest.approx(-cfg.grid_half + step / 2) and float(c[5, 1]) == pytest.approx(
        -cfg.grid_half + 1.5 * step)


def test_the_mission_kinds_are_one_list():
    """config.MISSION_KINDS: every kind's number and name in one place; the network has the first cfg.kinds."""
    ks = gconfig.MISSION_KINDS
    assert [k.index for k in ks] == list(range(len(ks))) and len({k.name for k in ks}) == len(ks)
    assert gconfig.GeneralConfig().kinds == len(ks)
    assert gconfig.mission_kinds(gconfig.GeneralConfig(kinds=4)) == ks[:4]
    with pytest.raises(ValueError):
        gconfig.GeneralConfig(kinds=len(ks) + 1)
