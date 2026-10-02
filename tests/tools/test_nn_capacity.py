"""tools.nn.train.capacity: forgetting against the previous iteration and the training log's signals
(docs/en/training/training.md "Capacity and forgetting"). Plain numpy: runs in the project's .venv."""
import json

import numpy as np
import pytest

from tools.nn.train import capacity

EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def evaluation(won, trade, version="abc123def456", opp="ai_like"):
    """A paired evaluation of len(won) battles: pairs of two (seed k, sides 1 and 2), EMP v SKV then
    SKV v EMP, our side attacking in the first battle of a pair."""
    n = len(won)
    seeds = [1000 + i // 2 for i in range(n)]
    battles = {"won": list(won), "own": [EMP if i % 2 == 0 else SKV for i in range(n)],
               "enemy": [SKV if i % 2 == 0 else EMP for i in range(n)], "attacks": [i % 2 == 0 for i in range(n)],
               "side": [1 + i % 2 for i in range(n)], "pair": [i // 2 for i in range(n)], "seed": seeds,
               "trade": list(trade), "destroyed": [500 + 500 * t for t in trade], "lost": [500 - 500 * t for t in trade],
               "budget": [1000.0] * n}
    return {"by_opponent": {opp: {"baseline": {"cache": f"{opp}_19u_3600s_{version}.json"}}},
            "battles": {opp: battles}, "skill": {"overall": {"value": 0.0, "se": 0.05}}}


def test_a_paired_drop_beyond_noise_is_forgetting_and_a_small_one_is_not():
    rng = np.random.default_rng(0)
    n = 400
    won = rng.random(n) < 0.6
    trade = rng.normal(0.1, 0.2, n)
    prev = capacity.cells(evaluation(won, trade))
    worse = trade - np.where(np.arange(n) % 2 == 0, 0.10, 0.0) + rng.normal(0, 0.01, n)   # EMP-SKV loses 10 points
    rows = {r["name"]: r for r in capacity.compare(capacity.cells(evaluation(won, worse)), prev)}
    assert rows["ai_like EMP-SKV trade"]["flag"] == "forgetting" and rows["ai_like EMP-SKV trade"]["paired"]
    assert rows["ai_like EMP-SKV trade"]["diff"] == pytest.approx(-0.10, abs=0.01)
    assert rows["ai_like SKV-EMP trade"]["flag"] == ""
    assert rows["ai_like pair score"]["diff"] == 0.0 and rows["ai_like EMP attack win"]["n"] == n // 2
    small = capacity.compare(capacity.cells(evaluation(won, trade - 0.02)), prev)   # pair gold: two battles
    assert not [r for r in small if r["flag"] in ("forgetting", "drop?")]                  # under 5 points


def test_cells_of_an_older_evaluation_come_from_its_summary_and_scene_rows():
    old = {"by_opponent": {"nearest": {"roles": {"attack": {"games": 100, "win_rate": 0.6, "gold_trade": 0.1}}}},
           "by_scene": {"nearest": {"Empire v Skaven, attack": {"games": 40, "wins": 30},
                                    "Empire v Skaven, defend": {"games": 60, "wins": 30}}}}
    c = capacity.cells(old)
    assert c["nearest EMP-SKV win"]["value"] == pytest.approx(0.6) and c["nearest EMP attack win"]["value"] == 0.75
    assert c["nearest attack trade"]["se"] is None and c["nearest attack win"]["se"] == pytest.approx(0.049, abs=1e-3)
    new = {"nearest EMP-SKV win": {"value": 0.4, "se": 0.05, "per": None},
           "nearest attack trade": {"value": 0.0, "se": None, "per": None}}
    rows = {r["name"]: r for r in capacity.compare(new, c)}
    assert rows["nearest EMP-SKV win"]["flag"] == "forgetting" and not rows["nearest EMP-SKV win"]["paired"]
    assert rows["nearest attack trade"]["flag"] == "drop?"                                 # no noise estimate


def log(n, ev=0.98, entropy=0.1, grad=0.3, vl=lambda i: 0.015):
    return [{"update": i + 1, "ev": ev, "value_loss": vl(i), "policy_loss": 0.01 * (-1) ** i, "entropy": entropy,
             "grad_norm": grad, "kl": 0.001, "clip": 0.002} for i in range(n)]


def test_the_training_signals_see_a_plateau_a_collapse_and_a_rising_grad_norm():
    flat = capacity.training(log(60, vl=lambda i: 0.015 + 0.001 * (-1) ** i))
    assert flat["value_loss"]["plateau"] and flat["ev"]["late"] == pytest.approx(0.98)
    falling = capacity.training(log(60, vl=lambda i: 0.03 - 0.0003 * i))
    assert not falling["value_loss"]["plateau"] and falling["value_loss"]["slope_100"] < 0
    lines = log(30) + log(30, entropy=0.02, grad=0.9)
    v, why = capacity.verdict([], [], capacity.training(lines), None, None)
    assert v == "watch" and any("entropy" in w for w in why) and any("grad norm" in w for w in why)
    assert capacity.verdict([], [], capacity.training(log(30)), (0.0, 0.05), (0.5, 0.05))[0] == "ok"


def test_the_verdict_tells_interference_from_a_trade_off_and_from_losing_skill():
    drop = {"name": "ai_like EMP-SKV trade", "diff": -0.1, "flag": "forgetting"}
    gain = {"name": "ai_like SKV-EMP trade", "diff": 0.1, "flag": "gain"}
    train = capacity.training(log(30))
    flat, up = ((0.0, 0.05), (0.02, 0.05)), ((0.0, 0.05), (0.5, 0.05))
    assert capacity.verdict([], [drop, gain], train, *flat)[0] == "widen-candidate"
    assert capacity.verdict([], [drop, gain], train, *up)[0] == "watch"                    # a trade-off
    v, why = capacity.verdict([], [drop], train, *flat)
    assert v == "watch" and "settings" in why[0]                                           # nothing gained


def test_the_report_finds_the_previous_iteration_and_doubts_it_with_another_simulator(tmp_path):
    root = tmp_path / "test5"
    won = [True, False] * 50
    for label, trade, version in (("it1", [0.2] * 100, "aaaaaaaaaaaa"), ("it2", [0.0] * 100, "bbbbbbbbbbbb")):
        d = root / label
        d.mkdir(parents=True)
        for f in ("before.json", "after.json"):
            (d / f).write_text(json.dumps(evaluation(won, trade, version)), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "log.jsonl").write_text("\n".join(json.dumps(x) for x in log(12)), encoding="utf-8")
    (root / "it2" / "report.json").write_text(json.dumps({"init": "build/nn-train/test5/it1/m10.pt", "run": str(run)}))
    rep = capacity.report(root / "it2")
    assert rep["prev"].endswith("it1") and rep["training"]["updates"] == 12
    assert {r["flag"] for r in rep["across"] if "trade" in r["name"]} == {"drop?"}       # another simulator
    assert any("differ" in w for w in rep["reasons"])
    text = "\n".join(capacity.lines(rep))
    assert "verdict: **watch**" in text and "within the iteration (minute 0 → end)" in text
    assert capacity.main([str(root / "it2"), "--prev", str(root / "it1" / "report.json")]) == 0
