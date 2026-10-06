"""The script references with torch: a drill's battles end the same when only the running ones step
(drills/verify.run), the drill metrics' Tracker counts the same over the narrowed batch, and
tools/nn/train/refs.py finds the missing files and runs a task the way test5 reads it back."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import scenario  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train import evaluate, randomise, refs, reward, rollout  # noqa: E402
from tools.nn.train.drills import metrics as drill_metrics  # noqa: E402
from tools.nn.train.drills import verify  # noqa: E402


def counter_battles(n=8):
    drill = D.load(["counter"])["counter"]
    pairs = D.battles(drill, range(D.DRILL_EVAL_SEEDS.start, D.DRILL_EVAL_SEEDS.start + n))
    st = scenario.build([p[0] for p in pairs])
    randomise.apply(st, torch.ones(st.B, dtype=torch.bool), evaluate.SPREAD, torch.Generator().manual_seed(7))
    return drill, st, torch.tensor([p[1] for p in pairs])


def play(compact, limit_s=600.0):
    """counter's 8 battles to a short limit (they end at different times), with a Tracker as drill_scripts keeps it."""
    drill, st, ours = counter_battles()
    tr = drill_metrics.Tracker(SimpleNamespace(B=st.B, N=st.N, device=st.device), ours, drill.roles, limit_s=limit_s)
    box = {"done": st.done.clone()}

    def extra(cur, rows):
        was = box["done"] if rows is None else box["done"][rows]
        tr.update(cur, ~was, rows)
        box["done"][slice(None) if rows is None else rows] = cur.done
        box["sizes"] = box.get("sizes", set()) | {cur.B}
    verify.run(st, ours, drill.skilled, drill.enemy, rollout.params_with_limit(limit_s), extra=extra,
               compact=compact, check_every=2)
    return st, tr, box["sizes"]


class TestCompactDrillBattles:
    def test_the_battles_and_the_counts_end_the_same_in_a_shrinking_batch(self):
        a, ta, sizes_a = play(compact=True)
        b, tb, sizes_b = play(compact=False)
        assert len(sizes_a) > 1 and sizes_b == {8}                      # it did shrink (and the other did not)
        assert bool(a.done.all()) and torch.equal(a.winner, b.winner) and torch.equal(a.t, b.t)
        assert len(set(a.t.tolist())) > 1                               # the battles end at different times
        assert torch.equal(reward.gold_sides(a.u), reward.gold_sides(b.u))
        for k in ta.all:
            assert torch.equal(ta.all[k], tb.all[k]) and torch.equal(ta.tail[k], tb.tail[k]), k
        assert torch.equal(ta.first, tb.first)

    def test_narrowed_keeps_the_rows_and_put_back_restores_them(self):
        _, st, _ = counter_battles(4)
        keep = torch.tensor([1, 3])
        cur = verify.narrowed(st, keep)
        assert cur.B == 2 and torch.equal(cur.u["x"], st.u["x"][keep]) and cur.keys == [st.keys[1], st.keys[3]]
        cur.u["x"] = cur.u["x"] + 1.0
        cur.t = cur.t + 5.0
        verify.put_back(st, keep, cur)
        assert torch.equal(st.u["x"][keep], cur.u["x"]) and torch.equal(st.t[keep], cur.t)
        assert float(st.t[0]) == 0.0


class TestRefs:
    def test_missing_names_the_absent_files_and_a_task_writes_what_test5_reads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        assert refs.missing(scripts=("nearest",), drills=("counter",), pairs=4, drill_battles=8) == ["nearest", "drill:counter"]
        monkeypatch.setattr(evaluate, "CANARY", 0)                    # (the task sets it: restored after)
        calls = []
        monkeypatch.setattr(evaluate, "baselines", lambda names, n, *a, **k: calls.append(("base", names, n, evaluate.CANARY))
                            or {names[0]: {"cache": "f.json"}})
        monkeypatch.setattr(evaluate, "drill_script", lambda name, n, which, device: calls.append(("drill", name, n, which, device))
                            or {"win_rate": 1.0})
        monkeypatch.setattr(torch, "set_num_threads", lambda k: calls.append(("threads", k)))
        assert refs._task("nearest", None, 2, 32, 4, 8, None, None) == ("nearest", 0, "f.json")
        assert refs._task("drill:counter:skilled", None, 2, 0, 4, 8, None, None) == ("drill:counter:skilled", 0, {"win_rate": 1.0})
        assert calls == [("threads", 2), ("base", ["nearest"], 4, 32), ("threads", 2), ("drill", "counter", 8, "skilled", "cpu")]

    def test_the_drill_s_file_is_written_when_both_scripts_are_in(self, tmp_path):
        from concurrent.futures import Future

        def done(x):
            f = Future()
            f.set_result(x)
            return f
        said = []
        pool = SimpleNamespace(shutdown=lambda: said.append("shutdown"))
        part = {"win_rate": 0.5, "gold_trade": 0.1}
        failed = Future()
        failed.set_exception(RuntimeError("boom"))
        jobs = ["ai_like", "drill:kiting:naive", "drill:kiting:skilled", "drill:counter:naive", "drill:counter:skilled"]
        p = refs.Pending(pool, jobs, [done(("ai_like", 3, "a.json")), done(("drill:kiting:naive", 2, part)),
                                      done(("drill:kiting:skilled", 4, dict(part, win_rate=0.9))),
                                      done(("drill:counter:naive", 1, part)), failed], 0.0, said.append,
                         lambda name: tmp_path / f"{name}.json")
        assert p.wait() == {"ai_like": 3, "drill:kiting:naive": 2, "drill:kiting:skilled": 4, "drill:counter:naive": 1}
        doc = json.loads((tmp_path / "kiting.json").read_text(encoding="utf-8"))
        assert doc == {"naive": part, "skilled": dict(part, win_rate=0.9)} and "shutdown" in said
        # a failed job: logged, its drill's file not written (the evaluation plays the drill itself)
        assert not (tmp_path / "counter.json").exists() and any("drill:counter:skilled: FAILED" in x for x in said)
        assert p.wait() is p.done and said.count("shutdown") == 1               # once

    def test_a_present_file_is_not_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        path = tmp_path / evaluate.baseline_path("nearest", refs.MAX_UNITS, refs.LIMIT_S).name
        path.write_text(json.dumps({"seed": evaluate.eval_seeds(4)}), encoding="utf-8")
        evaluate.drill_ref_path("counter", 8).write_text("{}", encoding="utf-8")
        assert refs.missing(scripts=("nearest",), drills=("counter",), pairs=4, drill_battles=8) == []
        assert refs.missing(scripts=("nearest",), drills=(), pairs=8) == ["nearest"]      # fewer seeds than asked
        assert refs.ensure(tmp_path, scripts=("nearest",), drills=("counter",), pairs=4, drill_battles=8) == {}

    def test_the_evaluation_waits_for_the_pending_references_before_it_reads_them(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        waited = []
        monkeypatch.setattr(evaluate, "REFS_READY", lambda: waited.append(1))
        evaluate.drill_ref_path("counter", 8).write_text('{"naive": 1}', encoding="utf-8")
        assert evaluate.drill_scripts("counter", 8) == {"naive": 1} and waited == [1]
        path = tmp_path / evaluate.baseline_path("nearest", 19, 4.0).name
        path.write_text(json.dumps({"seed": evaluate.eval_seeds(2), "winner": [1, 2]}), encoding="utf-8")
        assert evaluate.baselines(["nearest"], 2, limit_s=4.0)["nearest"]["winner"] == [1, 2] and waited == [1, 1]

    def test_test5_starts_the_missing_references_and_hands_the_wait_to_the_evaluation(self, monkeypatch):
        from tools.nn.train import test5

        class Pending:
            def wait(self):
                return {}
        got = []
        monkeypatch.setattr(evaluate, "REFS_READY", None)
        monkeypatch.setattr(evaluate, "CANARY", 32)
        monkeypatch.setattr(refs, "start", lambda **k: got.append(k) or Pending())
        test5.references(SimpleNamespace(eval=512, drill_eval=128))
        assert got == [dict(canary=32, pairs=256, drill_battles=128, scripts=test5.OPPONENTS, drills=None)]
        assert evaluate.REFS_READY is not None and evaluate.REFS_READY() == {}
        monkeypatch.setattr(refs, "start", lambda **k: got.append(k) or None)       # nothing missing
        test5.references(SimpleNamespace(eval=512, drill_eval=0))
        assert got[-1]["drills"] == () and evaluate.REFS_READY is None

    def test_a_file_another_process_plays_is_waited_for_not_played_again(self, tmp_path, monkeypatch):
        import threading
        monkeypatch.setattr(refs, "POLL_S", 0.02)
        path = refs.target("nearest", tmp_path)
        assert refs.claim(refs.marker(path))                                # another process holds it

        def missing(out=None, scripts=refs.SCRIPTS, drills=None, *rest):
            return ["nearest"] if "nearest" in scripts and not path.exists() else []
        monkeypatch.setattr(refs, "missing", missing)
        threading.Timer(0.2, lambda: path.write_text("{}", encoding="utf-8")).start()
        said = []
        pending = refs.start(tmp_path, scripts=("nearest",), drills=(), log=said.append)
        assert pending.pool is None and pending.others == {"nearest": path}  # nothing played here
        assert list(pending.wait()) == ["nearest"] and any("another process" in x for x in said)
        assert refs.ensure(tmp_path, scripts=("nearest",), drills=()) == {}  # written: nothing missing
