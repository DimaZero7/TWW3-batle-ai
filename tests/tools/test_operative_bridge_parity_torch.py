"""The operative's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md, section "оперативник"): the bridge's documents of a simulator battle (the tactician's
parity test's doc_of_sim: what the bridge writes) through tools/nn/companion/operative_input.py give the same input as
the simulator's raw state - the rows (but the known columns: the velocity and running), the place, the last groups, the
whole field map, the groups' numbers - on a flat field and on a captured map in the game's coordinates. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools import test_tactic_bridge_parity_torch as TP  # noqa: E402
from tools.nn.companion import operative_input as OPI  # noqa: E402
from tools.nn.operative import frame as FR  # noqa: E402
from tools.nn.operative import groups as GR  # noqa: E402
from tools.nn.operative import inputs as IN  # noqa: E402
from tools.nn.sim import scenario, terrain  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

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
