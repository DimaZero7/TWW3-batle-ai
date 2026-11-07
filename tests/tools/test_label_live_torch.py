"""Разметка вживую (tools/label/live.py) on a scratch game folder fed with the live state documents of a simulator battle:
15 s windows - no groups yet 'none', the groups made within a window 'human' (the walk takes them: every own unit in a
group, the rest as one group more), the same groups 'keep'; the player's orders a row a second; the marks over the
groups' units; the input rebuilt from a sample is the window start's. Needs torch."""
import json
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from tests.tools import test_tactic_bridge_parity_torch as TP  # noqa: E402
from tools.label import game as GM  # noqa: E402
from tools.label import live as LV  # noqa: E402
from tools.label import record as RC  # noqa: E402
from tools.nn.companion import exchange  # noqa: E402
from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.train import opponents  # noqa: E402

P = TP.P


def test_windows_decisions_skips_and_the_orders_stream(tmp_path):
    game, out = tmp_path / "game", tmp_path / "label"
    game.mkdir()
    out.mkdir()
    torch.manual_seed(1)
    st = scenario.build([TP.skirmish(gap=200.0)], P)
    rec = LV.LiveRecorder(game, out, "t")
    th = threading.Thread(target=rec.run, kwargs={"idle_exit_s": 20, "out": lambda *_: None}, daemon=True)
    th.start()
    groups = {"own": [], "enemy": []}
    for sec in range(47):
        if sec == 20:     # the person groups (Ctrl+G) and marks an enemy group
            groups = {"own": [[1, ["u0", "u1"]], [2, ["u2"]]], "enemy": [[1, ["u5"]]]}
        doc = TP.doc_of_sim(st, 0, 1000 * sec, own=1)
        doc.update(batch="live", move=sec + 1, live=True, groups=groups, selection=[], done=sec == 46)
        exchange.write_atomic(game / exchange.STATE, json.dumps(doc))
        t0 = time.time()
        while rec.stats["seconds"] < min(sec + 1, 46) and time.time() - t0 < 10:
            time.sleep(0.01)
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
    th.join(20)
    f = rec.folder
    dec = [json.loads(x) for x in (f / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["source"] for d in dec] == ["none", "human", "keep"]
    assert [d["sec"] for d in dec] == [0, 15, 30]
    assert dec[1]["changed"] and dec[1]["errors"] == [] and dec[1]["groups"]["rest"] == [3]   # u3, u4 in a group more
    smp = RC.load(f)
    assert [s["source"] for s in smp] == ["human", "keep"] and all(s["mode"] == "live" for s in smp)
    inp, ctx, act = RC.rebuild(smp[0])
    assert int(act["comp"].ge(0).sum()) == 5                         # all five own units placed in groups
    rows = (f / "orders.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 46 and set(json.loads(rows[0])["units"]) == {"u0", "u1", "u2", "u3", "u4"}
    ov = (game / GM.OVERLAY).read_text(encoding="utf-8").splitlines()
    assert ov[-1] == "end"
    res = json.loads((f / "result.json").read_text(encoding="utf-8"))
    assert res["mode"] == "live" and res["human"] == 1 and res["keep"] == 1 and res["none"] == 1
    info = json.loads((f / "session.json").read_text(encoding="utf-8"))
    assert info["mode"] == "live" and info["where"] == "game"


def test_the_marks_follow_the_groups(tmp_path):
    names = ["a", "b", "c", "x", "y"]
    text = GM.overlay_text(3, names, [{"g": 0, "members": [0, 1], "target": [3]}, {"g": 1, "members": [], "target": [3, 4]}])
    lines = text.splitlines()
    assert "mark a g 1" in lines and "mark b g 1" in lines and "mark x t 1" in lines and "mark y t 2" in lines
    assert sum(x.startswith("mark x ") for x in lines) == 1          # one icon a unit
