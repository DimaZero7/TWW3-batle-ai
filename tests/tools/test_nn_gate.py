"""The in-game gate: nn-arena builds of generated battles (--army-seed, --own-role) and
tools/nn/gate.py (the plan, the outcome of a run, the summary). Levels 1-2: no game."""
import json

import pytest

from tools import build
from tools import config as project
from tools.nn import gate
from tools.nn import scenario as nn_scenario
from tools.nn.armies import generate


def capture(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "BUILD", tmp_path)
    monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
    written = []
    monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append((config, scenario)) or {})
    return written


class TestBuild:
    @pytest.mark.parametrize("role,enemy_role,timeout_winner,attacker", [("attack", "defend", 1, 1),
                                                                         ("defend", "attack", 0, 2)])
    def test_a_generated_battle_under_the_network(self, role, enemy_role, timeout_winner, attacker, tmp_path,
                                                   monkeypatch):
        written = capture(monkeypatch, tmp_path)
        seed = 1_000_900_008
        assert build.main(["nn-arena", "--army-seed", str(seed), "--own-role", role, "--timeout", "3600"]) == 0
        (config, scenario), = written
        assert config["own_ai"] == "net" and config["own_role"] == role and config["enemy_role"] == enemy_role
        assert config["arena"] == f"random_{seed}" and config["timeout_ms"] == 3_600_000
        assert config["army"]["seed"] == seed and config["army"]["split"] == "eval"
        arena = generate.battle(seed)
        assert config["factions"] == {s: arena["sides"][s]["faction"] for s in ("own", "enemy")}
        assert [u["script_name"] for u in config["units"]["own"]] == [
            f"own_{u['slot']}" for u in arena["sides"]["own"]["units"]]
        assert len(config["units"]["enemy"]) == 20 and config["army"]["men"]["enemy"] > 1000
        # The battle file goes next to the build, not into scenarios/; the game's own limit is past ours.
        path = tmp_path / "nn-arena" / f"random_{seed}.xml"
        assert scenario == str(path) and not (tmp_path / "nn_arena.xml").exists()
        xml = path.read_text(encoding="utf-8")
        assert f"<timeout_winning_alliance_index>{timeout_winner}</timeout_winning_alliance_index>" in xml
        assert "<duration>3660</duration>" in xml and xml.count("<general>") == 2
        assert xml.count("<unit ") == len(config["units"]["own"]) + len(config["units"]["enemy"])
        # What the Lua side puts into the companion's state: attacker 1 when our side attacks.
        assert (1 if config["enemy_role"] == "defend" else 2) == attacker

    def test_the_build_reads_a_battle_file_outside_scenarios(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        xml = tmp_path / "random_1.xml"
        xml.write_text("<battle/>", encoding="utf-8")
        manifest = build.build("nn-arena", {"speed": 20, "timeout_ms": 60000, "tick_ms": 1000, "scenario": "x"},
                               dependencies=[], scenario=str(xml))
        assert manifest["syntax_checked"] is True and manifest["scenario"].endswith("nn_arena.xml")

    def test_defaults_and_wrong_combinations(self, tmp_path, monkeypatch):
        written = capture(monkeypatch, tmp_path)
        assert build.main(["nn-arena", "--own-ai", "net"]) == 0
        assert written[-1][0]["own_role"] == "defend" and written[-1][0]["enemy_role"] == "attack"
        assert written[-1][1] is None                      # the named arenas still use scenarios/
        assert build.main(["nn-arena"]) == 0
        assert written[-1][0]["own_ai"] == "attack" and "own_role" not in written[-1][0]
        for bad in (["--own-ai", "attack", "--own-role", "defend"], ["--army-seed", "5", "--arena", "whole_emp_v_skv"],
                    ["--timeout", "3601"]):
            with pytest.raises(SystemExit):
                build.main(["nn-arena", *bad])


class TestPlan:
    def test_eval_seeds_by_army_size_and_alternating_roles(self):
        rows = gate.plan(4)
        assert [r["seed"] for r in rows] == [1_000_900_000, 1_000_900_003, 1_000_900_007, 1_000_900_008]
        assert [r["role"] for r in rows] == ["attack", "defend", "attack", "defend"]
        sizes = [r["own_units"] - 1 for r in rows]
        assert [lo <= n <= hi for n, (lo, hi) in zip(sizes, gate.SIZE_BINS)] == [True] * 4
        assert all(generate.split(r["seed"]) == "eval" for r in rows)
        assert gate.plan(1, offset=3) == [rows[3]]
        assert gate.min_wins(4) == 3 and gate.min_wins(200) == 150

    def test_the_battle_limit_is_the_simulator_s(self):
        assert gate.battle_limit_s() == 3600 and gate.deadline_s(3600) == 840


def result(**kw):
    row = {"event": "result", "status": "completed", "winner": 1, "duration_model_ms": 250_000, "duration_wall_s": 30,
           "side_1_men": 500, "side_2_men": 20, "side_1_standing_units": 5, "side_2_standing_units": 0,
           "nn_moves": 250, "nn_answered": 250, "nn_missed": 0, "nn_orders_given": 900, "nn_keeps": 10,
           "nn_bad_files": 0}
    row.update(kw)
    return row


def write_run(root, name, seed, role, res, difficulty=1, errors=()):
    d = root / name
    d.mkdir(parents=True)
    config = {"own_role": role, "army": {"seed": seed, "men": {"own": 600, "enemy": 700}, "budget": 1000},
              "factions": {"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"},
              "units": {"own": [{}] * 3, "enemy": [{}] * 4}}
    (d / "manifest.json").write_text(json.dumps({"config": config}))
    (d / "launch.json").write_text("﻿" + json.dumps({"battle_difficulty": difficulty}), encoding="utf-8")
    (d / "status.json").write_text("﻿" + json.dumps({"status": "completed", "preferences_restored": True}),
                                   encoding="utf-8")
    lines = [json.dumps({"event": "error", "message": m}, separators=(",", ":")) for m in errors]
    if res:
        lines.append(json.dumps(res, separators=(",", ":")))
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    return str(d)


class TestSummary:
    @pytest.mark.parametrize("status,winner,role,expected", [
        ("completed", 1, "attack", (1, "completed")), ("completed", 2, "defend", (2, "completed")),
        ("timeout", 0, "attack", (2, "timeout")), ("timeout", 0, "defend", (1, "timeout")),
        ("stalled", 0, "defend", (1, "stalled")), ("deadline", 0, "defend", (None, "deadline")),
        ("incomplete", 0, "attack", (None, "incomplete"))])
    def test_timeouts_go_to_the_defender(self, status, winner, role, expected):
        assert gate.outcome({"status": status, "winner": winner}, role) == expected
        assert gate.outcome(None, role) == (None, "no_result")

    def test_a_gate_folder(self, tmp_path, capsys):
        runs = tmp_path / "runs"
        battles = [
            {"battle": 1, "seed": 11, "role": "attack", "run": write_run(runs, "r1", 11, "attack", result())},
            {"battle": 2, "seed": 12, "role": "defend",
             "run": write_run(runs, "r2", 12, "defend", result(status="timeout", winner=0))},
            {"battle": 3, "seed": 13, "role": "attack",
             "run": write_run(runs, "r3", 13, "attack", result(status="completed", winner=2), errors=["boom"])},
            {"battle": 4, "seed": 14, "role": "defend", "run": write_run(runs, "r4", 14, "defend", result(), difficulty=3)},
        ]
        gate_dir = tmp_path / "gate"
        gate_dir.mkdir()
        (gate_dir / "battles.json").write_text(json.dumps({"checkpoint": "c.pt", "planned": 4, "min_wins": 3,
                                                           "battles": battles}))
        assert gate.main(["summary", str(gate_dir)]) == 1
        s = json.loads((gate_dir / "summary.json").read_text(encoding="utf-8"))
        assert (s["wins"], s["losses"], s["no_result"], s["passed"], s["fair"]) == (2, 1, 1, False, False)
        rows = s["battles"]
        assert [r["outcome"] for r in rows] == ["win", "win", "loss", "no_result"]
        assert rows[1]["how"] == "timeout" and rows[1]["winner"] == "net"
        assert rows[2]["lua_errors"] == ["boom"] and rows[3]["how"] == "not_normal_difficulty"
        assert rows[0]["units"] == {"own": 3, "enemy": 4} and rows[0]["men_left"] == {"own": 500, "enemy": 20}
        assert rows[0]["nn"]["moves"] == 250 and rows[0]["duration_s"] == 250.0
        out = capsys.readouterr().out
        assert "NOT PASSED" in out and "EMP-SKV" in out

    def test_a_missing_run_is_no_result_and_a_short_gate_does_not_pass(self, tmp_path):
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 2, "battles": [
            {"battle": 1, "seed": 1, "role": "attack", "run": write_run(tmp_path, "r1", 1, "attack", result())},
            {"battle": 2, "seed": 2, "role": "defend", "run": None}]}))
        s = gate.summarize(tmp_path)
        assert s["min_wins"] == 2 and s["wins"] == 1 and s["no_result"] == 1 and not s["passed"]
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 1, "battles": [
            {"battle": 1, "seed": 1, "role": "attack", "run": str(tmp_path / "r1")}]}))
        assert gate.summarize(tmp_path)["passed"] is True
