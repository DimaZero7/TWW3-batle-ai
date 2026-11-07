"""The live record's training export (tools/label/digest_export.py): the operative's samples go out in the labelling
record's format and rebuild exactly; a unit's order becomes a unit_order sample with what maps one to one (target,
facing, pace, width) and what is missing. Needs torch (the container)."""
import copy
import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from tools.label import digest as DG  # noqa: E402
from tools.label import digest_export as DX  # noqa: E402
from tools.label import record as RC  # noqa: E402
from tools.label import session as SE  # noqa: E402
from tools.nn.tactic import train as TT  # noqa: E402
from tests.tools.test_label_torch import MATCH  # noqa: E402


def test_export(tmp_path):
    s = SE.Session(copy.deepcopy(MATCH), level="operative", nets={"unit": TT.unit_network("cpu")}, out_root=tmp_path)
    s.advance()
    live_inp = s.op[0]
    s.decide(s.pending["hint"])
    s.advance()
    s.decide({"keep": True})
    s.stop()
    lab = tmp_path / "live"
    lab.mkdir()
    shutil.copy(s.folder / "samples.pt", lab / "samples.pt")
    names = ["own_a", "own_b", "own_c", "enemy_a", "enemy_b", "enemy_c"]
    (lab / "session.json").write_text(json.dumps({"batch": "X", "names": names, "keys": s.info["keys"],
                                                  "sides": [1, 1, 1, 2, 2, 2]}), encoding="utf-8")
    rows = lambda ox, t: {"own_a": {"x": 0, "z": 0, "b": 90, "ox": ox, "oz": 0, "t": t, "f": True, "men": 100,  # noqa
                                    "ow": 30}}
    (lab / "orders.jsonl").write_text("\n".join(json.dumps({"sec": k, "units": r}) for k, r in
                                                ((1, rows(0, "")), (2, rows(50, "")), (3, rows(50, "enemy_b")))),
                                      encoding="utf-8")
    rec = DG.Record(lab, run=None)
    out = tmp_path / "out"
    out.mkdir()
    info = DX.export(rec, out)
    assert info["operative"] == {"samples": 2, "human": 0, "keep": 1, "with_act": 2, "refused": 0, "rebuilt": 2,
                                 "hint": 1}
    S = RC.load(out)
    inp, ctx, act = RC.rebuild(S[0])
    for k, v in live_inp.items():
        if torch.is_tensor(v):
            assert torch.equal(inp[k], v), k
    uo = [x for x in S if x["level"] == "unit_order"]
    assert [u["order"]["kind"] for u in uo] == ["move", "attack"]       # (the first second's places are no order)
    assert uo[1]["act_partial"]["target_slot"] == 4 and uo[0]["act_partial"]["pace"] == 1
    assert info["tactic"]["samples"] == 0 and info["tactic"]["missing"]
