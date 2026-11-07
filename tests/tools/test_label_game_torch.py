"""Разметка в игре (tools/label/game.py) on a scratch game folder fed with the bridge's documents of a simulator arena
battle: the deployment document -> the strategist's pause, the person's decision -> the plan file; the state documents
-> the operative's pause on seconds 0 and 15 (no orders file for that move until the person decides), the other
seconds answered by the chain alone; a refused decision is not written; the record in the simulator's format with
'where: game'; the input rebuilt from a sample is the live one. Needs torch."""
import json
import time

import pytest

torch = pytest.importorskip("torch")

from tests.tools import test_strategist_bridge_parity_torch as SP  # noqa: E402
from tests.tools import test_tactic_bridge_parity_torch as TP  # noqa: E402
from tools.label import game as GM  # noqa: E402
from tools.label import record as RC  # noqa: E402
from tools.label import strategist as LS  # noqa: E402
from tools.nn.companion import deploy as CD  # noqa: E402
from tools.nn.companion import exchange  # noqa: E402
from tools.nn.sim import battle  # noqa: E402
from tools.nn.strategist import battle as BT  # noqa: E402
from tools.nn.strategist import cases as CS  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402

P = TP.P


def until(cond, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.01)
    raise TimeoutError


def answered(game, move):
    f = game / exchange.ORDERS
    d = exchange.parse_orders(f.read_text(encoding="utf-8")) if f.exists() else None
    return d is not None and d["move"] == move


def test_the_person_deploys_and_commands_the_operative_over_the_game(tmp_path):
    game, out = tmp_path / "game", tmp_path / "label"
    game.mkdir()
    out.mkdir()
    torch.manual_seed(0)
    sp = [s for s in CS.exam_specs(n=8, seed=3, arena=True) if s["case"] == "hold"][:1]
    pre = CS.Pre(sp, P, "cpu")
    o, inp, _ = BT.decide(pre)
    st, _, _, _ = BT.placed(pre, o, inp)
    s = GM.GameSession(game, {"unit": Nt.UnitNet().eval()}, task="hold", out_root=out, request=False)
    try:
        dep = SP.doc_of_pre(pre, 0)
        exchange.write_atomic(game / CD.DOC, json.dumps(dep))
        p = until(lambda: s.pending)
        assert p["level"] == "strategist" and s.state()["phase"] == "deployment"
        bad = {"groups": [], "verb": "hold", "pace": "any"}
        assert not s.decide(bad)["ok"] and not (game / CD.PLAN).exists()      # no group: refused, nothing written
        assert s.decide(p["hint"])["ok"]
        plan = (game / CD.PLAN).read_text(encoding="utf-8")
        assert plan.startswith(CD.FORMAT) and plan.count("\nplace ") == len(sp[0]["mine"])
        for sec in range(17):
            doc = TP.doc_of_sim(st, 0, 1000 * sec, own=1)
            doc.update(batch=dep["batch"], move=sec + 1)
            exchange.write_atomic(game / exchange.STATE, json.dumps(doc))
            if sec in (0, 15):
                q = until(lambda: s.pending)
                assert q["level"] == "operative" and q["sec"] == sec and q["can_keep"]
                time.sleep(0.1)
                assert not answered(game, sec + 1)               # the bridge waits: no answer before the decision
                assert s.decide({"keep": True} if sec == 0 else q["hint"])["ok"]
            until(lambda: answered(game, sec + 1))
            for _ in range(round(1.0 / P.dt)):
                battle.step(st, opponents.nearest(st), P)
        assert s.chain.stats["from_strategist"] and s.chain.stats["op_decisions"] == 2
        assert s.chain.stats["parcel_errors"] == 0 and s.n_decisions == 3
        done = TP.doc_of_sim(st, 0, 17000, own=1)
        done.update(batch=dep["batch"], move=17, done=True)          # the bridge keeps the last move's number
        exchange.write_atomic(game / exchange.STATE, json.dumps(done))
        until(lambda: s.over)
    finally:
        s.stop()
    f = s.folder
    info = json.loads((f / "session.json").read_text(encoding="utf-8"))
    assert info["where"] == "game" and info["format"] == RC.FORMAT
    smp = RC.load(f)
    assert [x["level"] for x in smp] == ["strategist", "operative", "operative"]
    assert [x["source"] for x in smp] == ["hint", "keep", "hint"] and all(x["where"] == "game" for x in smp)
    assert len((f / "decisions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert len((f / "chain.jsonl").read_text(encoding="utf-8").splitlines()) == 17
    res = json.loads((f / "result.json").read_text(encoding="utf-8"))
    assert res["why"] == "the battle's end" and res["decisions"] == 3 and res["where"] == "game"
    inp, ctx, act = RC.rebuild(smp[2])
    assert torch.equal(act["comp"], smp[2]["act"]["comp"]) and inp["rows"].shape[0] == 1
    sinp, _, _ = LS.rebuild(smp[0])
    assert torch.allclose(sinp["rows"], s.strat.inp["rows"])
    replay = json.loads((f / "replay.json").read_text(encoding="utf-8"))
    assert len(replay["snaps"]) == 18                     # the deployment's and 17 seconds
