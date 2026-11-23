"""The operative's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
contract/parity.md, section "оперативник"): the bridge's documents of a simulator battle (the tactician's
parity test's doc_of_sim: what the bridge writes) through bridge/companion/operative_input.py give the same input as
the simulator's raw state - the rows (but the known columns: the velocity and running), the place, the last groups, the
whole field map, the groups' numbers - on a flat field and on a captured map in the game's coordinates. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from contract import test_tactic_bridge_parity_torch as TP  # noqa: E402
from bridge.companion import operative_input as OPI  # noqa: E402
from net.operative import frame as FR  # noqa: E402
from net.operative import gframe as GF  # noqa: E402
from net.operative import heads as HD  # noqa: E402
from net.operative import groups as GR  # noqa: E402
from net.operative import inputs as IN  # noqa: E402
from sim import scenario, terrain  # noqa: E402
from net.tactic import decide as DC  # noqa: E402
from net.tactic import raw as TR  # noqa: E402
from net.tactic import rows as RW  # noqa: E402
from net.unit import ckpt as T  # noqa: E402

P = TP.P


def groups_of(raw, place=None):
    """One group of side 1's first three units with its zone at their centre and a target of side 2's first two."""
    B, N = raw.shape[:2]
    g = GR.Groups.empty(B, N)
    I = TR.I
    mine = (raw[0, :, I["side"]] == 1).nonzero()[:3, 0]
    foe = (raw[0, :, I["side"]] == 2).nonzero()[:2, 0]
    g.member[0, mine] = 0
    g.on[0, 0] = True
    g.target[0, 0, foe] = True
    x, z = float(raw[0, mine, I["x"]].mean()), float(raw[0, mine, I["z"]].mean())
    hd = 30.0
    if place is not None:
        x, z = TP.to_game(x, z, place)
        hd = (hd + place[2]) % 360
    g.cx[0, 0], g.cz[0, 0], g.heading[0, 0], g.n0[0, 0], g.age[0, 0] = x, z, hd, 3, 30.0
    return g


def group_masks(raw, side):
    """(mem, E) [1, N]: the own side's alive units, the other side's alive and seen ones (drill O0-1's group and E)."""
    mine, enemy = FR.sides(raw, torch.full((1,), side, dtype=torch.long))
    return mine, enemy


def o1_parity(rs, rg, side, place):
    """Drill O0-1's new input (tmp/engine/design_merged.md 5): gframe on both raw states (the game's in the map's
    turned frame: a rotated moving twin, the velocities from the shared one-second history - view.shift_velocity on
    both), the order tokens and candidates of a stand-off line shifted 37 m across (its zone given in each frame), and
    the same with an enemy hidden: no row, out of E, gframe equal."""
    I = TR.I
    sd = torch.full((1,), side, dtype=torch.long)
    for hide in (False, True):
        a_raw, b_raw = rs.clone(), rg.clone()
        if hide:
            foe = ((a_raw[0, :, I["side"]] != side) & (a_raw[0, :, I["alive"]] > 0) & (a_raw[0, :, I["side"]] > 0) &
                   (a_raw[0, :, I["vis"]] > 0)).nonzero()[:, 0]
            j = int(foe[0])
            a_raw[0, j, I["vis"]] = 0.0
            b_raw[0, j, I["vis"]] = 0.0
        sa, sb = IN.select(a_raw, sd)[0], IN.select(b_raw, sd)[0]
        assert torch.equal(sa, sb)
        ma, ea = group_masks(a_raw, side)
        mb, eb = group_masks(b_raw, side)
        assert torch.equal(ma, mb) and torch.equal(ea, eb)
        if hide:
            assert j not in sa[0].tolist() and not bool(ea[0, j])
        ga, gb = GF.gframe(a_raw, sa, ma, ea), GF.gframe(b_raw, sb, mb, eb)
        assert float(ga.abs().max()) > 0
        assert torch.allclose(ga, gb, atol=2e-3), float((ga - gb).abs().max())
        C = [(float(r[0, :, I["x"]][m[0]].mean()), float(r[0, :, I["z"]][m[0]].mean())) for r, m in ((a_raw, ma), (b_raw, mb))]
        zone, ls = (C[0][0] + 10.0, C[0][1] - 20.0, 30.0), (C[0][0] + 37.0, C[0][1] + 140.0, 30.0)
        ga_ = TP.to_game(zone[0], zone[1], place) + (zone[2],)
        gl_ = TP.to_game(ls[0], ls[1], place) + (ls[2],)
        oa = {"ptr": "LS", "zone": zone, "goal": C[0], "born": 30.0}
        ob = {"ptr": "LS", "zone": ga_, "goal": C[1], "born": 30.0}
        fa = HD.order_feats(oa, C[0], ls, 75.0) + sum(HD.cand_feats(oa, C[0], ls), [])
        fb = HD.order_feats(ob, C[1], gl_, 75.0) + sum(HD.cand_feats(ob, C[1], gl_), [])
        assert max(abs(x - y) for x, y in zip(fa, fb)) < 2e-3, (fa, fb)


def parity(sim, game, st, place=None, bounds=None):
    card = T.cards(st, "cpu")[:1].float()
    cmd = DC.order_vec(1, "hold", "any")
    keep = [i for i, n in enumerate(RW.ROW) if n not in TP.ROW_NOT_YET]
    seen = 0
    for side in (1, 2):
        sd = torch.full((1,), side, dtype=torch.long)
        for d in (2, len(sim) // 2, len(sim) - 1):
            rs, rg = sim[d], game[side][d]
            a_ax = FR.Axes.of(rs, sd, bounds)
            b_ax = FR.Axes.of(rg, sd, bounds)
            a = IN.build(rs, card, sd, a_ax, groups_of(rs), cmd, bounds=bounds)
            b = IN.build(rg, card, sd, b_ax, groups_of(rs, place), cmd, bounds=bounds)
            assert torch.equal(a["slot"], b["slot"]) and torch.equal(a["who"], b["who"])
            assert torch.allclose(a["rows"][..., keep], b["rows"][..., keep], atol=2e-3)
            for k in ("place", "prev", "cover", "tokens", "numbers"):
                assert torch.allclose(a[k], b[k], atol=2e-3), k
            dd = (a["map"] - b["map"]).abs().amax((0, 2, 3))
            assert float(dd.max()) < 2e-3, (IN.LAYERS[int(dd.argmax())], float(dd.max()))
            assert torch.equal(a["members"], b["members"]) and torch.equal(a["targets"], b["targets"])
            o1_parity(rs, rg, side, place)
            seen += 1
    return seen


def test_the_bridge_gives_the_operatives_input_on_a_flat_field():
    torch.manual_seed(3)
    st = scenario.build([TP.skirmish()], P)
    seers = {k: OPI.OperativeSeer(allow_flat=True, params=P, own=k) for k in (1, 2)}
    sim, game = TP.run_both(st, 30, seers)
    assert parity(sim, game, st, bounds=st.bounds) == 6


def test_the_bridge_gives_the_operatives_input_on_a_map_in_the_games_coordinates(tmp_path):
    folder = TP.write_map(tmp_path / "testmap")
    mid = terrain.load_npz(folder / "heights.npz")
    torch.manual_seed(4)
    st = scenario.build([TP.skirmish(gap=100.0)], P)
    place = (150.0, 380.0, 35.0)
    terrain.set_map(st, mid, place=tuple(torch.tensor([v]) for v in place))
    seers = {k: OPI.OperativeSeer(map_root=tmp_path, params=P, own=k) for k in (1, 2)}
    sim, game = TP.run_both(st, 20, seers, place=place, terrain_path="testmap")
    assert parity(sim, game, st, place=place) == 6
    a = IN.build(sim[5], T.cards(st, "cpu")[:1].float(), torch.ones(1, dtype=torch.long),
                 FR.Axes.of(sim[5], torch.ones(1, dtype=torch.long)), groups_of(sim[5]), DC.order_vec(1, "hold", "any"))
    for n in ("field", "forest", "height"):
        assert float(a["map"][0, IN.LI[n]].abs().max()) > 0, n
