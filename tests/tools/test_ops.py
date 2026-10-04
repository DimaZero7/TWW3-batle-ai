"""tools/ops: the chain step from a few parameters, the run card, the leftovers check and the wait with a
timeout (docs/en/training/workflow.md). Plain Python; wait.sh needs bash (skipped without it)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.ops import card, leftovers, step

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")


# --- step ------------------------------------------------------------------------------------------

def chain_folders(tmp_path, label="prev", nets=("m5", "m10", "m25"), critic=True):
    t5, runs = tmp_path / "test5", tmp_path / "runs"
    for n in nets:
        (t5 / label / f"{n}.pt").parent.mkdir(parents=True, exist_ok=True)
        (t5 / label / f"{n}.pt").write_bytes(b"x")
    if critic:
        (runs / f"test5_{label}").mkdir(parents=True)
        (runs / f"test5_{label}" / "latest.pt").write_bytes(b"x")
    return t5, runs


class TestStep:
    def test_the_previous_step_s_last_network_and_its_critic(self, tmp_path):
        t5, runs = chain_folders(tmp_path)
        label, init, critic = step.chain_point("prev", t5, runs)
        assert (label, init.name, critic.name) == ("prev", "m25.pt", "latest.pt")      # m25 > m10 > m5 by number, not text

    def test_a_named_point_and_a_missing_critic(self, tmp_path):
        t5, runs = chain_folders(tmp_path, critic=False)
        label, init, critic = step.chain_point("prev/m10", t5, runs)
        assert init.name == "m10.pt" and critic is None
        with pytest.raises(SystemExit):
            step.chain_point("prev/m99", t5, runs)
        with pytest.raises(SystemExit):
            step.chain_point("nobody", t5, runs)

    def test_options_become_arguments(self):
        args = step.option_args({"anchor": 0.03, "drill-teach": "auto", "no-eval": True, "gone": None, "off": False,
                                 "mix": {"self": 0.05, "ai_like": 0.3}})
        assert args == ["--anchor", "0.03", "--drill-teach", "auto", "--no-eval", "--mix", '{"self":0.05,"ai_like":0.3}']

    def test_set_and_drop(self):
        out = step.apply_sets({"drills": 0.2, "teach-normal": "auto"}, ["drills=0.1", "--weights={\"a\":1}", "name=plain"], ["teach-normal"])
        assert out == {"drills": 0.1, "weights": {"a": 1}, "name": "plain"}
        with pytest.raises(SystemExit):
            step.apply_sets({}, ["novalue"], [])

    def test_the_command_of_a_step(self, tmp_path, monkeypatch):
        t5, runs = chain_folders(tmp_path)
        monkeypatch.setattr(step, "TEST5", t5)
        monkeypatch.setattr(step, "RUNS", runs)
        chain = {"test5": {"minutes": 25, "every": 5, "baseline_canary": 32, "dock_prefix": "orch-"},
                 "options": {"anchor": 0.03, "drills": 0.2, "mix": {"self": 0.05}}}
        lines, info = step.build("n4", "prev", chain, minutes=15, sets=["drills=0.1"], extra=["--lord-rout", "0.3"])
        cmd = lines[1]
        assert lines[0].startswith('cd "/') or lines[0].startswith('cd "')
        assert cmd.startswith("DOCK_NAME=orch-n4 bash tools/nn/dock.sh tools.nn.train.test5 --label n4 --init ")
        assert "--updates 0 --minutes 15 --every 5 --baseline-canary 32 -- --critic-init " in cmd
        assert "--reference " in cmd and "--anchor 0.03 --drills 0.1 --mix '{\"self\":0.05}' --lord-rout 0.3" in cmd
        assert cmd.index("--init ") < cmd.index(" -- ") < cmd.index("--critic-init")
        assert info["prev"] == "prev" and info["init"].endswith("m25.pt") and info["options"]["drills"] == 0.1
        assert info["container"] == "orch-n4" and info["out"] == "build/nn-train/test5/n4"

    def test_no_canary_when_the_config_says_zero(self, tmp_path, monkeypatch):
        t5, runs = chain_folders(tmp_path)
        monkeypatch.setattr(step, "TEST5", t5)
        monkeypatch.setattr(step, "RUNS", runs)
        lines, info = step.build("n4", "prev", {"test5": {"baseline_canary": 0}, "options": {}})
        assert "--baseline-canary" not in lines[1] and "--minutes 25 --every 5" in lines[1]

    def test_the_wall_time_counts_before_every_point_and_after(self):
        wall, evals = step.wall_minutes(25, 5, miss=False)
        assert evals == 7 and 45 < wall < 55
        assert step.wall_minutes(25, 5, miss=True)[0] > wall + 20

    def test_posix_paths_for_git_bash(self):
        assert step.posix("C:\\Users\\x\\repo") == "/c/Users/x/repo"

    def test_main_writes_the_script(self, tmp_path, monkeypatch, capsys):
        t5, runs = chain_folders(tmp_path)
        monkeypatch.setattr(step, "TEST5", t5)
        monkeypatch.setattr(step, "RUNS", runs)
        monkeypatch.setattr(step, "baseline_cache", lambda: ("abc", []))
        chain = tmp_path / "chain.json"
        chain.write_text(json.dumps({"test5": {"minutes": 10, "every": 5}, "options": {"anchor": 0.03}}), encoding="utf-8")
        out = tmp_path / "n4.sh"
        assert step.main(["--label", "n4", "--from", "prev", "--chain", str(chain), "--write", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert text.endswith("--anchor 0.03\n") and "\r" not in text
        said = capsys.readouterr().out
        assert "baseline cache: hit" in said and "expected wall time" in said and "tools.ops.card build/nn-train/test5/n4" in said


# --- card ------------------------------------------------------------------------------------------

def point(rating, se=0.06, gold=0.2, gse=0.015, win=0.7, move=0.25, lord=0.25, timeouts=0.0, kl=None, seconds=190):
    cell = {"win": 0.6, "lord_dead_own": lord, "games": 256, "timeouts": timeouts, "gold_ratio": 1.1, "missile_melee_share": 0.08}
    p = {"skill": {"rating": {"overall": {"value": rating, "se": se}},
                   "pair_gold": {o: {"pair_gold": {"value": gold, "se": gse}, "exchange": {"value": 1.3}} for o in card.OPPONENTS}},
         "drills": {"counter": {"win_rate": win, "games": 128, "scripts": {"skilled": {"win_rate": 0.9}}}},
         "transfer": {"counter": {"network": {"share": 0.4, "mistake": 0.25}, "ai_like": {"share": 0.5, "mistake": 0.27}}},
         "teach_auto": [{"drill": "counter", "share": 0.1, "deficit": 0.2}],
         "ai_like/all": {"kind_hold": 0.4, "kind_move": move, "kind_attack": 0.35}, "seconds": seconds}
    for o in card.OPPONENTS:
        p[f"{o}/attack"], p[f"{o}/defend"] = dict(cell), dict(cell)
    if kl is not None:
        p["distance"] = {"start_kl": kl}
    return p


def report_folder(tmp_path, points, name="run", init="build/nn-train/test5/prev/m25.pt"):
    folder = tmp_path / name
    folder.mkdir()
    rep = {"init": init, "train_s": 1500, "updates": 60, "trend": {str(m): p for m, p in points},
           "capacity": {"verdict": "ok", "reasons": ["steady"]}}
    (folder / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    return folder


class TestCard:
    def test_the_rows_carry_the_key_numbers_with_their_noise(self):
        rows = {r[0]: r for r in card.rows(point(0.5, win=0.75, kl=0.04))}
        assert rows["rating"][2] == 0.5 and rows["rating"][3] == 0.06
        assert rows["gold_ai_like"][2] == 0.2 and rows["exchange"][2] == 1.3
        assert rows["drill_counter"][2] == 0.75 and rows["drill_counter"][3] == pytest.approx((0.75 * 0.25 / 128) ** 0.5)
        assert "skilled 0.90" in rows["drill_counter"][1]
        assert rows["applied_counter"][2] == 0.4 and "0.50" in rows["applied_counter"][4]      # ai_like's beside
        assert rows["teach_auto_counter"][2] == 0.1 and rows["lord"][2] == 0.25 and rows["lord"][3] is not None
        assert rows["move"][2] == 0.25 and rows["kl"][2] == 0.04 and rows["seconds"][2] == 190
        assert card.fmt(0.5, 0.06, "{:+.2f}") == "+0.50 ± 0.12" and card.fmt(None, None, "{:.2f}") == "-"

    def test_a_change_beyond_two_standard_errors_or_the_plain_threshold(self):
        assert card.change(0.5, 0.8, 0.06, 0.06) == (pytest.approx(0.3), True)
        assert card.change(0.5, 0.6, 0.06, 0.06) == (pytest.approx(0.1), False)
        assert card.change(0.4, 0.46, None, None) == (pytest.approx(0.06), True)
        assert card.change(0.4, 0.42, None, None)[1] is False
        assert card.change(None, 0.4, None, None) == (None, False)

    def test_the_card_of_a_finished_run_marks_gains_and_lists_anomalies(self, tmp_path):
        pts = [(0, point(0.5)), (5, point(0.1, move=0.5, lord=0.45, timeouts=0.08)), (10, point(0.9, win=0.5, move=0.5, lord=0.45, kl=0.05))]
        folder = report_folder(tmp_path, pts)
        lines, data = card.card(folder)
        text = "\n".join(lines)
        assert lines[0] == "run card: run (init build/nn-train/test5/prev/m25.pt)"
        assert "| min 0 | min 5 | min 10 | change first -> last |" in text
        rating = next(r for r in data["rows"] if r["key"] == "rating")
        assert rating["values"] == [0.5, 0.1, 0.9] and rating["change"] == pytest.approx(0.4) and rating["beyond"]
        assert "| +0.50 ± 0.12 | +0.10 ± 0.12 | +0.90 ± 0.12 | +0.400 * |" in text
        assert "rating fell beyond noise min 0 -> 5" in text
        assert "move share x2.0: 0.25 -> 0.50" in text and "own lord dead x1.8" in text
        assert "timeouts 0.08 at min 5" in text and "counter drill win fell 0.70 -> 0.50" in text
        assert "capacity: ok - steady" in text and "training: 25 min, 60 updates" in text
        seconds = next(r for r in data["rows"] if r["key"] == "seconds")
        assert seconds["beyond"] is False                     # never marked

    def test_the_previous_step_s_end_beside(self, tmp_path):
        folder = report_folder(tmp_path, [(0, point(0.5)), (5, point(0.6))])
        prev = report_folder(tmp_path, [(0, point(0.2)), (25, point(0.3))], name="prev")
        lines, data = card.card(folder, prev)
        assert "| prev end (prev min 25) |" in lines[2] and "| vs prev end |" in lines[2]
        assert next(r for r in data["rows"] if r["key"] == "rating")["vs_prev"] == pytest.approx(0.3)
        assert "| +0.100 | +0.300 * |" in "\n".join(lines)

    def test_a_running_folder_is_read_from_its_evaluations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(card.skill, "summary", lambda res: {"rating": {"overall": {"value": res["r"], "se": 0.05}}})
        folder = tmp_path / "live"
        folder.mkdir()
        raw = {"r": 0.1, "seconds": 300, "drills": {"kiting": {"win_rate": 0.9, "games": 128}},
               "by_opponent": {"ai_like": {"kinds": {"hold": 0.3, "move": 0.3, "attack": 0.4},
                                           "roles": {"attack": {"games": 256, "win_rate": 0.6, "lord_dead_own": 0.2, "timeouts": 0.0,
                                                                "behaviour": {"missile_melee_share": 0.1}},
                                                     "defend": {"games": 0}}}}}
        (folder / "before.json").write_text(json.dumps(raw), encoding="utf-8")
        (folder / "eval_m5.json").write_text(json.dumps(dict(raw, r=0.4, seconds=150)), encoding="utf-8")
        (folder / "eval_m10.json").write_text(json.dumps(dict(raw, r=0.5)), encoding="utf-8")
        pts, rep = card.points_of(folder)
        assert [m for m, _ in pts] == ["0", "5", "10"] and rep is None
        assert pts[1][1]["ai_like/attack"]["lord_dead_own"] == 0.2 and "ai_like/defend" not in pts[1][1]
        lines, data = card.card(folder)
        assert next(r for r in data["rows"] if r["key"] == "rating")["values"] == [0.1, 0.4, 0.5]
        assert next(r for r in data["rows"] if r["key"] == "seconds")["values"] == [300, 150, 300]
        assert next(r for r in data["rows"] if r["key"] == "drill_kiting")["values"] == [0.9, 0.9, 0.9]
        assert "anomalies: none" in lines and not any(line.startswith("capacity") for line in lines)

    def test_an_empty_folder_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit):
            card.points_of(tmp_path)


# --- leftovers -------------------------------------------------------------------------------------

def fake_run(docker="", ps="[]", smi=""):
    calls = []

    def run(cmd, timeout=30):
        calls.append(cmd)
        return {"docker": docker, "powershell": ps, "nvidia-smi": smi}.get(cmd[0], "")
    run.calls = calls
    return run


def proc(pid, ppid, name, cmd, age_min=20):
    return {"pid": pid, "ppid": ppid, "name": name, "age_min": age_min, "cmd": cmd}


class TestLeftovers:
    def test_only_our_containers_are_listed(self, monkeypatch):
        monkeypatch.setattr(leftovers, "run", fake_run(docker="orch-n4\tsnake-ai-trainer\tUp 2 hours\t2 hours ago\n"
                                                              "explk-db\tpostgres:15-alpine\tUp 2 days\t2 days ago\n"
                                                              "tww3-bai-companion\tsnake-ai-trainer:latest\tUp 1 minute\t1 minute ago\n"))
        assert [c["name"] for c in leftovers.containers()] == ["orch-n4", "tww3-bai-companion"]

    def test_wait_loops_their_sleeps_and_not_our_own_shell(self):
        procs = [proc(1, 0, "explorer.exe", ""), proc(10, 1, "bash.exe", "bash -c until grep -qE 'passed|failed' t.log; do sleep 60; done"),
                 proc(11, 10, "sleep.exe", "sleep 60"), proc(20, 1, "bash.exe", "bash -c while true; do sleep 5; done", age_min=2),
                 proc(30, 1, "bash.exe", "bash -c python -m tools.ops.leftovers"), proc(31, 30, "python.exe", "python -m tools.ops.leftovers"),
                 proc(40, 1, "python.exe", "python -m tools.nn.gate summary x"), proc(50, 1, "Warhammer3.exe", "")]
        cl = leftovers.classify(procs, own=31)
        assert [p["pid"] for p in cl["loops"]] == [10, 20] and [p["pid"] for p in cl["sleeps"]] == [11]
        assert [p["pid"] for p in cl["nn"]] == [40] and len(cl["game"]) == 1
        assert leftovers.ancestors(procs, 31) == {31, 30, 1}
        assert "until grep" in leftovers.snippet(procs[1]["cmd"])

    def test_the_report_flags_two_containers_a_stale_lock_and_old_loops(self, tmp_path, monkeypatch):
        monkeypatch.setattr(leftovers, "run", fake_run(docker="a\tsnake-ai-trainer\tUp\t1h\nb\tsnake-ai-trainer\tUp\t1h\n", smi="7, 1200, 16000"))
        monkeypatch.setattr(leftovers, "LOCK", tmp_path / "gpu-train.lock")
        procs = [proc(10, 1, "bash.exe", "until x; do sleep 1; done", age_min=30), proc(20, 1, "bash.exe", "until x; do sleep 1; done", age_min=1)]
        lines, found = leftovers.report(procs, min_age=10)
        text = "\n".join(lines)
        assert found and "MORE THAN ONE" in text and "GPU lock: free" in text
        assert "wait loops (bash until/while ... sleep): 2, 1 older than 10 min" in text and "GPU: 7 % busy, 1200 / 16000 MiB" in text
        (tmp_path / "gpu-train.lock").write_text("test5 n4 2026-10-04 10:00:00\n", encoding="utf-8")
        monkeypatch.setattr(leftovers, "run", fake_run())                               # no container: the lock is stale
        lines, found = leftovers.report([], min_age=10)
        assert found and "STALE" in "\n".join(lines)
        monkeypatch.setattr(leftovers, "run", fake_run(docker="a\tsnake-ai-trainer\tUp\t1h\n"))
        lines, found = leftovers.report([], min_age=10)
        assert "STALE" not in "\n".join(lines)                                            # a training container holds it

    def test_nothing_left_is_quiet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(leftovers, "run", fake_run())
        monkeypatch.setattr(leftovers, "LOCK", tmp_path / "none.lock")
        lines, found = leftovers.report([], min_age=10)
        assert not found and lines[:3] == ["containers (ours): none", "GPU lock: free", "wait loops: none"]

    def test_kill_ends_only_old_loops_and_their_sleeps(self, monkeypatch):
        run = fake_run()
        monkeypatch.setattr(leftovers, "run", run)
        procs = [proc(10, 1, "bash.exe", "until x; do sleep 1; done", age_min=30), proc(11, 10, "sleep.exe", "sleep 1"),
                 proc(20, 1, "bash.exe", "until x; do sleep 1; done", age_min=1), proc(21, 20, "sleep.exe", "sleep 1")]
        monkeypatch.setattr(leftovers, "ancestors", lambda procs, pid: set())
        assert leftovers.kill(procs, min_age=10) == [10, 11]
        assert [c[-1] for c in run.calls] == ["Stop-Process -Id 10 -Force -ErrorAction SilentlyContinue",
                                              "Stop-Process -Id 11 -Force -ErrorAction SilentlyContinue"]

    def test_processes_parse_powershell_s_json(self, monkeypatch):
        ps = json.dumps({"ProcessId": 5, "ParentProcessId": 1, "Name": "sleep.exe", "CreationDate": "/Date(1791099674213)/", "CommandLine": "sleep 60"})
        monkeypatch.setattr(leftovers, "run", fake_run(ps=ps))
        p = leftovers.processes()
        assert len(p) == 1 and p[0]["pid"] == 5 and p[0]["name"] == "sleep.exe" and p[0]["age_min"] > 0
        monkeypatch.setattr(leftovers, "run", fake_run(ps="not json"))
        assert leftovers.processes() == []

    def test_main_returns_one_when_something_is_left(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(leftovers, "run", fake_run(docker="a\tsnake-ai-trainer\tUp\t1h\n"))
        monkeypatch.setattr(leftovers, "LOCK", tmp_path / "none.lock")
        assert leftovers.main([]) == 1
        monkeypatch.setattr(leftovers, "run", fake_run())
        assert leftovers.main([]) == 0


# --- wait.sh -----------------------------------------------------------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not available")
class TestWait:
    def run(self, *args, timeout=30):
        return subprocess.run([BASH, str(REPO / "tools" / "ops" / "wait.sh"), *args], capture_output=True, text=True, timeout=timeout)

    def test_a_matching_line_ends_the_wait_whatever_its_case(self, tmp_path):
        log = tmp_path / "t.log"
        log.write_text("collecting\n3 FAILED, 1 passed\n", encoding="utf-8")
        done = self.run("-t", "5", "-i", "1", str(log), "passed|failed")
        assert done.returncode == 0 and "3 FAILED, 1 passed" in done.stdout

    def test_a_timeout_returns_two_with_the_file_s_tail(self, tmp_path):
        log = tmp_path / "t.log"
        log.write_text("still running\n", encoding="utf-8")
        done = self.run("-t", "1", "-i", "1", str(log), "never")
        assert done.returncode == 2 and "timeout after 1 s" in done.stderr and "still running" in done.stderr

    def test_a_missing_file_is_waited_for_not_an_error(self, tmp_path):
        done = self.run("-t", "1", "-i", "1", str(tmp_path / "later.log"), "x")
        assert done.returncode == 2

    def test_gone_and_a_bad_call(self, tmp_path):
        assert self.run("-t", "2", "-i", "1", "gone", str(tmp_path / "nothere")).returncode == 0
        assert self.run().returncode == 1
        assert self.run("only-one").returncode == 1
