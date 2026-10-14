"""The step replay (tools/nn/sim/stepreplay.py) on a small made-up recording: each window starts from the recorded state
and the rebuilt clocks."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import stepreplay

OWN, ENEMY = "wh2_main_skv_inf_stormvermin_1", "wh_main_emp_inf_greatswords"


def recording(tmp_path, T=30):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = {"own_ai": "attack", "enemy_role": "defend", "arena": "arena",
           "factions": {"own": "wh2_main_skv_skaven", "enemy": "wh_main_emp_empire"},
           "units": {"own": [{"script_name": "own_a", "key": OWN, "width": 30.0}],
                     "enemy": [{"script_name": "enemy_b", "key": ENEMY, "width": 30.0}]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")
    lines = []
    for k in range(T):
        a = {"n": "own_a", "x": -100.0 if k < 10 else -60.0, "z": 0.0, "b": 90, "hp": 1.0 if k < 12 else 0.7,
             "men": 100 if k < 12 else 70, "mp": 1.0 if k < 10 else -0.2, "ms": 2 if k < 10 else 6, "r": k >= 10,
             "w": False, "m": False, "s": False, "a": 0, "k": 0, "fat": "threshold_fresh", "t": "", "ox": -100.0,
             "oz": 0.0}
        b = {"n": "enemy_b", "x": 100.0, "z": 0.0, "b": 270, "hp": 1.0, "men": 90, "mp": 1.0, "ms": 2, "r": False,
             "w": False, "m": False, "s": False, "a": 0, "k": 0, "fat": "threshold_fresh", "t": "", "ox": 100.0,
             "oz": 0.0}
        lines.append({"event": "nn_sample", "t": 1000 * k, "units": [a, b]})
    lines.append({"event": "result", "winner": 2})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in lines),
                                           encoding="utf-8")
    return tmp_path


def test_windows_start_from_the_recorded_state(tmp_path):
    run = recording(tmp_path / "run1")
    out = tmp_path / "out"
    out.mkdir()
    got = stepreplay._batch(([str(run)], 10, str(out), [], 2, "cpu", 1))
    a = np.load(got["run1"])
    assert list(a["starts"]) == [0, 10]
    assert a["sim_x"].shape == (2, 10, 2, 2)              # [W, H, C, n]
    # window 1 starts where the game had the unit (moved 40 m), routing, its rout counted
    assert abs(float(a["sim_x"][1, 0, :, 0].mean()) + 60.0) < 6.0
    assert a["sim_r"][1, 0, :, 0].all() and a["c_rout_count"][1, 0] == 1
    # the health is the recorded one at each window start: nobody fights, nothing is lost in the windows
    assert np.allclose(a["sim_hp"][0, :, :, 0], 1.0) and np.allclose(a["sim_hp"][1, :, :, 0], 1.0)
    assert np.allclose(a["sim_hp"][:, :, :, 1], 1.0)
    # the second window's start lost nothing yet in the game (the loss comes at row 12): no casualty window
    assert a["c_recent"][1, 0] == 0.0


def test_forced_places_and_health_follow_the_recording_every_second(tmp_path):
    run = recording(tmp_path / "run1")
    out = tmp_path / "out"
    out.mkdir()
    got = stepreplay._batch(([str(run)], 10, str(out), [], 1, "cpu", 1, ("pos", "hp")))
    a = np.load(got["run1"])
    # rows 12+ lost 30 % in the game: from the window's second 2 (row 12) the simulator's health is the game's
    assert np.allclose(a["sim_hp"][1, 1:, 0, 0], 0.7, atol=1e-4) and np.allclose(a["sim_hp"][1, 0, 0, 0], 1.0)
    assert np.allclose(a["sim_x"][1, :, 0, 0], -60.0, atol=1e-3)
    assert str(a["force"]) == "pos,hp"
