"""The live record's digest and pictures (tools/label/digest.py, replay.py; no torch): a small made-up record and run -
the feed by windows (groups, changes, compressed orders, chat, losses, the result), a frame a window and the gif."""
import json

import pytest

from tools.label import digest as DG


def unit(n, side, x, z, men=100, **k):
    return {"n": n, "side": side, "x": x, "z": z, "b": 90 if side == 1 else 270, "men": men, "ow": 30, "ox": x, "oz": z,
            "t": "", "f": False, "m": False, "mv": False, "r": False, "ob": 90 if side == 1 else 270, **k}


@pytest.fixture()
def rec(tmp_path):
    lab, run = tmp_path / "2026-11-07_000000_live", tmp_path / "runs" / "r1"
    lab.mkdir()
    run.mkdir(parents=True)
    names = ["own_spear_1", "own_spear_2", "own_gun_1", "enemy_rat_1", "enemy_rat_2"]
    keys = ["wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_handgunners",
            "wh2_main_skv_inf_clanrats_0", "wh2_main_skv_inf_clanrats_0"]
    (lab / "session.json").write_text(json.dumps({"batch": "B1", "map": "flat", "names": names, "keys": keys,
                                                  "sides": [1, 1, 1, 2, 2]}), encoding="utf-8")
    wins = [{"sec": 1, "until": 16, "source": "none", "groups": {"own": {}, "enemy": {}}},
            {"sec": 16, "until": 31, "source": "human", "groups": {"own": {"1": ["own_spear_1", "own_spear_2"]},
                                                                   "enemy": {"1": ["enemy_rat_1"]}},
             "chat": [{"n": 1, "sec": 20.0, "text": "копья держат рат"}]}]
    (lab / "decisions.jsonl").write_text("\n".join(json.dumps(w, ensure_ascii=False) for w in wins), encoding="utf-8")
    events, orders = [{"batch": "B1", "event": "ready"}], []
    for s in range(1, 31):
        us = [unit("own_spear_1", 1, -100, 0), unit("own_spear_2", 1, -100, 40), unit("own_gun_1", 1, -140, 20),
              unit("enemy_rat_1", 2, 100, 0, men=100 - s), unit("enemy_rat_2", 2, 100, 40)]
        if s >= 20:                                     # the spears go to (0, 20) at a run; the gun fires at rat 1
            for u in us[:2]:
                u.update(ox=0, oz=20, f=True, mv=True)
            us[2].update(t="enemy_rat_1")
        events.append({"batch": "B1", "event": "nn_sample", "t": s * 1000, "units": us})
        orders.append({"sec": s, "units": {u["n"]: u for u in us if u["side"] == 1}})
    events.append({"batch": "B1", "event": "nn_soldiers", "t": 20000,
                   "units": [{"n": "own_spear_1", "side": 1, "xz_dm": [-1000, 0, -1000, 15]}]})
    events.append({"batch": "B1", "event": "result", "winner": 1, "duration_model_ms": 30000, "side_1_men": 300,
                   "side_2_men": 170, "side_1_standing_units": 3, "side_2_standing_units": 2})
    (run / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
    (lab / "orders.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in orders), encoding="utf-8")
    return DG.Record(lab, run)


def test_the_feed(rec):
    md = DG.digest_md(rec)
    assert "## 16–31 с" in md and "решение человека" in md
    assert "группа 1:" in md and "×2" in md and "враг 1:" in md
    assert "идут к (0, 20) бегом" in md                       # the two spears' order, compressed into one line
    assert "атакуют" in md                                     # the gun's target
    assert "копья держат рат" in md
    assert "потери: враг" in md and "## Итог" in md and "победа" in md
    assert md.count("\n") < 60                                 # short


def test_frames_and_gif(rec, tmp_path):
    pytest.importorskip("PIL")
    from tools.label import replay as RP
    out = tmp_path / "out"
    out.mkdir()
    n = RP.frames(rec, out, gif=True, every_s=5)
    assert n == 2 and (out / "digest_02.png").exists() and (out / "battle.gif").exists()
    img = RP.frame(rec, 20)
    assert img.size == (RP.W, RP.H)


def test_the_run_is_found_by_its_batch(rec, tmp_path):
    assert DG.find_run("B1", tmp_path / "runs") == rec.run
    assert DG.find_run("nope", tmp_path / "runs") is None
