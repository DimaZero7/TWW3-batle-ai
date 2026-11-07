"""The operative's input (tools/nn/operative: frame, inputs, groups; build/diag_operative/plan.md "Вход"). Needs torch."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.operative import cases as CS  # noqa: E402
from tools.nn.operative import cfg  # noqa: E402
from tools.nn.operative import frame as FR  # noqa: E402
from tools.nn.operative import groups as GR  # noqa: E402
from tools.nn.operative import inputs as IN  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402
from tools.nn.tactic import zone as ZN  # noqa: E402

P = load()


def state(n=6, seed=0):
    sp = CS.specs(n, torch.Generator().manual_seed(seed))
    setup = CS.Setup(sp, P, "cpu")
    raw = TR.Seer(P).raw(setup.st)
    return setup, raw


def build(setup, raw, groups=None):
    B, N = raw.shape[:2]
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = groups or GR.Groups.empty(B, N, device="cpu")
    return IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds), axes, groups


def test_grid_is_the_tacticians():
    for line in (False, True):
        for n in range(1, 11):
            r, c = GR.grid(torch.tensor([n]), torch.tensor([line]))
            assert (int(r), int(c)) == ZN.grid(n, line), (n, line)


def test_rows_every_own_unit_and_seen_enemy_the_origin_at_the_unit():
    setup, raw = state()
    inp, axes, _ = build(setup, raw)
    K = int(cfg()["rows"])
    assert inp["rows"].shape == (6, K, len(RW.ROW))
    I = TR.I
    alive = raw[..., I["alive"]] > 0
    for b in range(6):
        slots = set(inp["slot"][b][inp["mask"][b]].tolist())
        want = set((alive[b] & ((raw[b, :, I["side"]] == 1) | (raw[b, :, I["vis"]] > 0))).nonzero()[:, 0].tolist())
        assert slots == want
    on = inp["mask"]
    for k in ("right", "forward", "dh"):
        assert float(inp["rows"][..., RW.RI[k]][on].abs().max()) < 1e-5, k
    who = inp["who"][on]
    assert set(who.tolist()) <= {RW.OWN_GROUP, RW.ENEMY_OTHER}
    # the place: the unit's own position in the field's axes / 1024 m
    s, f = axes.local(inp["J"][..., I["x"]], inp["J"][..., I["z"]])
    assert torch.allclose(inp["place"][..., 0][on], (s / 1024.0)[on], atol=1e-5)
    assert torch.allclose(inp["place"][..., 1][on], (f / 1024.0)[on], atol=1e-5)
    assert float(inp["place"][on].abs().max()) < 1.0


def test_a_hidden_enemy_has_no_row_and_leaves_no_trace():
    setup, raw = state(2, seed=3)
    inp, _, _ = build(setup, raw)
    raw2 = raw.clone()
    j = int((raw[0, :, TR.I["side"]] == 2).nonzero()[0, 0])
    raw2[0, j, TR.I["vis"]] = 0.0
    inp2, _, _ = build(setup, raw2)
    assert j not in inp2["slot"][0].tolist()
    raw3 = raw2.clone()
    raw3[0, j, TR.I["x"]] += 50.0
    raw3[0, j, TR.I["hp"]] = 0.3
    inp3, _, _ = build(setup, raw3)
    for k in ("rows", "map", "cover", "place", "prev"):
        assert torch.equal(inp2[k], inp3[k]), k


def test_axes_snap_to_the_field_and_face_the_enemy():
    setup, raw = state(6, seed=1)
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    assert torch.allclose(torch.remainder(axes.heading, 90.0), torch.zeros(6), atol=1e-4)
    mine, enemy = FR.sides(raw, setup.side)
    mx, mz, _ = FR.centre(raw, mine)
    ex, ez, _ = FR.centre(raw, enemy)
    way = torch.rad2deg(torch.atan2(ex - mx, ez - mz))
    off = (torch.remainder(way - axes.heading + 180.0, 360.0) - 180.0).abs()
    assert float(off.max()) <= 45.0 + 1e-3


def test_the_map_and_the_groups_numbers():
    setup, raw = state(3, seed=2)
    B, N = raw.shape[:2]
    g = GR.Groups.empty(B, N)
    g.member[0, :3] = 0
    g.on[0, 0] = True
    g.cx[0, 0], g.cz[0, 0] = float(raw[0, :3, TR.I["x"]].mean()), float(raw[0, :3, TR.I["z"]].mean())
    g.n0[0, 0] = 3
    foe = (raw[0, :, TR.I["side"]] == 2).nonzero()[:, 0]
    g.target[0, 0, foe] = True
    inp, axes, _ = build(setup, raw, g)
    C = int(cfg()["cells"])
    assert inp["map"].shape == (B, len(IN.LAYERS), C, C)
    assert len(IN.LAYERS) == 33
    k = IN.LI["card_mine0"]
    assert float(inp["map"][:, k:k + 16].abs().max()) == 0.0         # the cards: painted by the network
    assert float(inp["map"][0, IN.LI["zones"]].sum()) > 0 and float(inp["map"][1, IN.LI["zones"]].sum()) == 0
    assert float(inp["map"][:, IN.LI["mine"]].sum()) > 0 and float(inp["map"][:, IN.LI["enemy"]].sum()) > 0
    # prev: my rows' group, the enemy's target-of
    own = inp["own"][0]
    pv = inp["prev"][0]
    assert torch.equal(pv[own][:, 0], torch.ones(int(own.sum())))
    en = inp["enemy"][0]
    assert torch.equal(pv[en][:, GR.GMAX + 1], torch.ones(int(en.sum())))
    assert float(inp["prev"][1][inp["enemy"][1]][:, 2 * GR.GMAX + 1].min()) == 1.0
    assert inp["members"][0, 0].sum() == 3 and inp["targets"][0, 0].sum() == en.sum()
    assert inp["tokens"][0, 0, 0] == 1.0 and float(inp["tokens"][1].abs().max()) == 0.0
    assert math.isclose(float(inp["numbers"][0, 0]), 1 / GR.GMAX, rel_tol=1e-5)
    assert inp["prev"].shape[-1] == GR.PREV == 2 * (int(cfg()["groups"]) + 1)
