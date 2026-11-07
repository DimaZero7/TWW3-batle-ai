"""The strategist's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md, section "стратег"): the bridge's deployment document of a strategist battle (what
src/apps/bridge/services.lua deploy_document writes: my units' full rows - the tactician's parity test's doc_of_sim -,
the enemy's by its composition only, both zones) through tools/nn/companion/strategist_input.py gives the same input as
the simulator's state before the deployment - the rows, the map, the numbers, the command - and the companion's plan
(tools/nn/companion/deploy.py) the same places as the simulator's deployment, with no error. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools import test_tactic_bridge_parity_torch as TP  # noqa: E402
from tools.nn.companion import deploy as CD  # noqa: E402
from tools.nn.companion import strategist_input as SI  # noqa: E402
from tools.nn.strategist import battle as BT  # noqa: E402
from tools.nn.strategist import cases as CS  # noqa: E402
from tools.nn.strategist import inputs as IN  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402
from tools.nn.unit import drill_hold as DH  # noqa: E402

P = TP.P
COMPOSITION = ("n", "side", "key", "men", "hp")       # services.lua COMPOSITION


def doc_of_pre(pre, b):
    """The deployment document of battle b of a strategist batch, in the bridge's format."""
    d = TP.doc_of_sim(pre.st0, b, 0)
    units = []
    for r in d["units"]:
        if r["side"] == 2:
            r = dict({k: r[k] for k in COMPOSITION}, v=True)
        units.append(r)
    zones = {name: {k: float(z[k][b]) for k in SI.KEYS} for name, z in (("own", pre.mine_zone),
                                                                          ("enemy", pre.enemy_zone))}
    s = pre.sp[b]
    return dict(d, units=units, deploy=True, zones=zones, batch=f"parity{b}", attacker=2,
                factions={"own": DH.FACTIONS[s["ours"]], "enemy": DH.FACTIONS[s["theirs"]]})


def test_the_bridge_gives_the_strategists_input_and_the_same_places():
    pre = CS.Pre(CS.exam_specs(n=6, seed=5), P, "cpu")
    a, _ = BT.inputs_of(pre)
    out, _, _ = BT.decide(pre)
    _, _, places, _ = BT.placed(pre, out, a)
    for b, s in enumerate(pre.sp):
        doc = doc_of_pre(pre, b)
        seer = SI.StrategistSeer(allow_flat=True, params=P)
        g, _, _ = seer.input(doc, pre.command[b:b + 1])
        for k in ("mask", "who", "own", "enemy", "shooter"):
            assert torch.equal(a[k][b:b + 1], g[k]), k
        d = (a["rows"][b:b + 1] - g["rows"]).abs().amax((0, 1))
        assert float(d.max()) < 2e-3, (RW.ROW[int(d.argmax())], float(d.max()))
        dm = (a["map"][b:b + 1] - g["map"]).abs().amax((0, 2, 3))
        assert float(dm.max()) < 2e-3, (IN.LAYERS[int(dm.argmax())], float(dm.max()))
        for k in ("numbers", "command"):
            assert torch.allclose(a[k][b:b + 1], g[k], atol=1e-5), k
        # the enemy is in the rows by its composition: as many rows as units, men and health
        assert int(g["enemy"].sum()) == len(s["foes"])
        # the companion's plan: the simulator's places, unit by unit (doc_of_sim names a unit u<slot>)
        text, rec = CD.Strategist("formula", s["task"], params=P).decide(doc)
        assert rec["errors"] == 0 and len(rec["places"]) == len(s["mine"])
        for name, (x, z, bear, w) in rec["places"].items():
            i = int(name[1:])
            sim = [float(places[k][b, i]) for k in ("x", "z", "b", "width")]
            assert abs(x - sim[0]) < 0.05 and abs(z - sim[1]) < 0.05 and abs(bear - sim[2]) < 1e-3 \
                and abs(w - sim[3]) < 0.05, (name, (x, z, bear, w), sim)
        assert text.startswith(CD.FORMAT) and text.rstrip().endswith("end")
        assert sum(line.startswith("place ") for line in text.splitlines()) == len(s["mine"])


def test_the_enemys_place_and_state_never_reach_the_input():
    pre = CS.Pre(CS.exam_specs(n=2, seed=7), P, "cpu")
    doc = doc_of_pre(pre, 0)
    seer = SI.StrategistSeer(allow_flat=True, params=P)
    g0, _, _ = seer.input(doc, pre.command[:1])
    # my units' places moved, the enemy given places and a state it never has in the document: the same input
    moved = dict(doc, units=[dict(u, x=u["x"] + 90.0, z=u["z"] - 40.0) if u["side"] == 1 else
                             dict(u, x=0.0, z=0.0, mv=True, m=True, mp=0.2, fat="threshold_tired")
                             for u in doc["units"]])
    g1, _, _ = SI.StrategistSeer(allow_flat=True, params=P).input(moved, pre.command[:1])
    for k in ("rows", "map", "numbers"):
        assert torch.allclose(g0[k], g1[k], atol=1e-5), k
