"""The strategist's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
contract/parity.md, section "стратег"): the bridge's deployment document of a strategist battle (what
bridge/lua/apps/bridge/services.lua deploy_document writes: my units' full rows - the tactician's parity test's doc_of_sim -,
the enemy's by its composition only, both zones) through bridge/companion/strategist_input.py gives the same input as
the simulator's state before the deployment - the rows, the map, the numbers, the command - and the companion's plan
(bridge/companion/deploy.py) the same places as the simulator's deployment, with no error. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from contract import test_tactic_bridge_parity_torch as TP  # noqa: E402
from bridge.companion import deploy as CD  # noqa: E402
from bridge.companion import strategist_input as SI  # noqa: E402
from net.strategist import battle as BT  # noqa: E402
from net.strategist import cases as CS  # noqa: E402
from net.strategist import inputs as IN  # noqa: E402
from net.tactic import rows as RW  # noqa: E402
from net.unit import drill_hold as DH  # noqa: E402

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


def test_the_point_sent_is_half_a_depth_ahead_of_the_centre():
    # the game stands the formation's centre ~half its depth behind the teleport point (07.11.2026): the plan sends
    # centre + depth / 2 along the facing; a unit of one (a lord) its centre
    x, z = CD.teleport_point(-248.0, -376.0, 90.0, 9.6, 120)
    assert abs(float(x) + 243.2) < 1e-4 and abs(float(z) + 376.0) < 1e-4
    x, z = CD.teleport_point(10.0, 20.0, 0.0, 9.6, 120)
    assert abs(float(x) - 10.0) < 1e-4 and abs(float(z) - 24.8) < 1e-4
    x, z = CD.teleport_point(-248.0, 456.0, 92.8125, 1.4, 1)
    assert float(x) == -248.0 and float(z) == 456.0
    # the plan's lines: the point sent and the centre; the record's places stay the simulator's centres
    from sim import geometry
    pre = CS.Pre(CS.exam_specs(n=3, seed=5), P, "cpu")
    out, a, _ = BT.decide(pre)
    st, _, places, _ = BT.placed(pre, out, a)
    _, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    import math
    for b in range(3):
        text, rec = CD.Strategist("formula", pre.sp[b]["task"], params=P).decide(doc_of_pre(pre, b))
        lines = {ln.split()[1]: [float(v) for v in ln.split()[2:]] for ln in text.splitlines() if ln.startswith("place ")}
        for name, (px, pz, bear, w, cx, cz) in lines.items():
            i = int(name[1:])
            half = float(depth[b, i]) / 2 if float(st.u["men0"][b, i]) > 1 else 0.0
            assert half > 1.0 or float(st.u["men0"][b, i]) <= 1
            a_ = math.radians(bear)
            assert abs(cx - float(places["x"][b, i])) < 0.01 and abs(cz - float(places["z"][b, i])) < 0.01
            assert abs(px - (cx + half * math.sin(a_))) < 0.02 and abs(pz - (cz + half * math.cos(a_))) < 0.02
            assert rec["sent"][name] == [round(px, 2), round(pz, 2)]


def test_the_companion_takes_the_same_default_strategist_as_the_simulator():
    from net.strategist import cfg, default
    pre = CS.Pre(CS.exam_specs(n=2, seed=11), P, "cpu")
    a, _ = BT.inputs_of(pre)
    who = default()
    sim_net = None
    if who != "formula":
        from bridge.config import resolve
        from net.strategist import ckpt as SC
        sim_net, _, _ = SC.restore(str(resolve(who)))
    out, _, _ = BT.decide(pre, sim_net, greedy=True)
    _, _, places, _ = BT.placed(pre, out, a)
    for b in range(len(pre.sp)):
        st = CD.Strategist(task=pre.sp[b]["task"], params=P)           # no 'who': config default
        assert st.who == who and (st.net is None) == (cfg()["default"] == "formula")
        _, rec = st.decide(doc_of_pre(pre, b))
        assert rec["errors"] == 0
        assert rec["verb"] == ("destroy" if bool(pre.destroy[b]) else "hold")
        assert len(rec["groups"]) == int(out["plan"]["on"][b].sum())
