"""The bridge (tools/label/bridge_infer.py; no torch): groups read off a made-up battle - two units under one command and
marching together are one group, a far unit shooting its own target another; a unit that leaves its company in a
later window makes a decision, one window without change is 'keep'; the commands' kinds (bool1, target, the run flag
then seen as moving fast); the Ctrl+G groups compared; the deployment's commands (their own clock) dropped."""
import json

import pytest

from tools.label import bridge_infer as BI
from tools.label import digest as DG


def unit(n, side, x, z, **k):
    return {"n": n, "side": side, "x": x, "z": z, "b": 90, "men": 100, "ow": 30, "ox": x, "oz": z, "t": "", "f": False,
            "m": False, "mv": False, "r": False, **k}


@pytest.fixture()
def rec(tmp_path):
    lab, run = tmp_path / "lab", tmp_path / "runs" / "r"
    lab.mkdir()
    run.mkdir(parents=True)
    names = ["own_a", "own_b", "own_c", "own_d", "enemy_x", "enemy_y"]
    (lab / "session.json").write_text(json.dumps({"batch": "B", "names": names, "keys": [""] * 6,
                                                  "sides": [1, 1, 1, 1, 2, 2]}), encoding="utf-8")
    (lab / "decisions.jsonl").write_text(json.dumps({"sec": 0, "until": 0, "level": "strategist", "groups": {
        "own": {"1": ["own_a", "own_b", "own_c"], "2": ["own_d"]}, "enemy": {}}}), encoding="utf-8")
    ev, orders = [], []
    for s in range(0, 46):
        a = (-200 + 3 * s, 0)                           # a and b march together; c stands far; d stands far the other way
        b = (-200 + 3 * s, 30)
        c = (-200, 300) if s < 30 else (-200 + 3 * (s - 30), 60)    # c joins a and b in the third window
        us = [unit("own_a", 1, *a, mv=True, f=True), unit("own_b", 1, *b, mv=True, f=True), unit("own_c", 1, *c),
              unit("own_d", 1, -200, -300, t="enemy_y", fire=True), unit("enemy_x", 2, 100, 0), unit("enemy_y", 2, 100, -300)]
        ev.append({"batch": "B", "event": "nn_sample", "t": s * 1000, "units": us})
        orders.append({"sec": s, "units": {u["n"]: u for u in us if u["side"] == 1}})
    cmds = [{"sec": 300.0, "name": "Move Orientation Width", "units": ["own_a"], "bool1": False, "target": None,
             "ordered": {}},                                                    # (the deployment's clock: dropped)
            {"sec": 1.5, "name": "Attack Unit", "units": ["own_a", "own_b"], "bool1": True, "target": "enemy_x",
             "ordered": {"own_a": {"fast": True}, "own_b": {"fast": True}}},
            {"sec": 31.0, "name": "Move Orientation Width", "units": ["own_c", "own_a"], "bool1": False, "target": None,
             "ordered": {"own_c": {"fast": True, "ox": -100, "oz": 60, "ob": 90, "ow": 30}}}]
    (lab / "commands.jsonl").write_text("\n".join(json.dumps(c) for c in cmds), encoding="utf-8")
    (lab / "orders.jsonl").write_text("\n".join(json.dumps(o) for o in orders), encoding="utf-8")
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    return DG.Record(lab, run)


def test_groups_targets_decisions(rec):
    wins = BI.infer(rec)
    assert [w.sec for w in wins] == [1, 16, 31]
    w0 = wins[0]
    together = [g for g in w0.groups.values() if "own_a" in g][0]
    assert set(together) == {"own_a", "own_b"}
    d = [g for g in w0.groups.values() if "own_d" in g][0]
    assert d == ["own_d"]
    k_d = next(k for k, g in w0.groups.items() if g == ["own_d"])
    assert w0.seed[k_d] == "enemy_y"
    k_ab = next(k for k, g in w0.groups.items() if "own_a" in g)
    assert w0.seed[k_ab] == "enemy_x" and w0.area[k_ab]["front"] > 0
    assert wins[1].decision == "keep"
    assert set([g for g in wins[2].groups.values() if "own_a" in g][0]) >= {"own_a", "own_c"}


def test_commands_and_compare(rec):
    assert [c["sec"] for c in BI.battle_commands(rec)] == [1.5, 31.0]
    uo = BI.unit_orders(rec)
    assert uo["Attack Unit"]["bool1"] == {True: 1} and uo["Attack Unit"]["with_target"] == 1
    assert uo["Attack Unit"]["fast_then_moving_fast"] == "2/2"
    cmp = BI.compare(rec, BI.infer(rec))
    assert cmp[1]["windows"] == 3 and cmp[1]["together"] < 1.0          # c was apart from a and b for two windows


def test_limits():
    pos = {f"u{i}": (i * 10.0, 0.0) for i in range(23)}
    parts = BI._cut(sorted(pos), pos)
    assert all(len(p) <= BI.MAX_GROUP for p in parts) and sum(map(len, parts)) == 23
    groups = [[f"u{i}"] for i in range(9)]
    assert len(BI._merge_to(groups, pos)) == BI.G
