"""tools/nn/train/refs.py on the host (no torch): how the missing script references are shared out over
the container's cores."""
from tools.nn.train import refs


class TestPlan:
    def test_a_process_per_task_and_the_cores_shared_out(self):
        assert refs.plan(["a"] * 6, 8) == (6, 1)
        assert refs.plan(["a"] * 3, 8) == (3, 2)
        assert refs.plan(["a"] * 6, 20) == (6, 3)

    def test_never_more_processes_than_cores_or_jobs(self):
        assert refs.plan(["a"] * 6, 4) == (4, 1)
        assert refs.plan(["a"] * 6, 8, jobs=2) == (2, 4)
        assert refs.plan([], 8) == (1, 8)

    def test_a_drill_is_two_jobs_and_the_longest_jobs_go_first(self):
        got = refs.jobs_of(["drill:counter", "nearest", "other", "drill:kiting", "ai_like"])
        assert got == ["ai_like", "drill:kiting:naive", "drill:kiting:skilled", "nearest", "drill:counter:naive",
                       "drill:counter:skilled", "other"]
        assert refs.jobs_of([]) == []


class TestCpuQuota:
    def test_the_container_s_quota_cgroup_v2(self, tmp_path):
        (tmp_path / "cpu.max").write_text("800000 100000\n")
        assert refs.cpu_quota(tmp_path) == 8

    def test_cgroup_v1(self, tmp_path):
        (tmp_path / "cpu").mkdir()
        (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("500000")
        (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
        assert refs.cpu_quota(tmp_path) == 5

    def test_no_quota_the_machine_s_cores(self, tmp_path, monkeypatch):
        (tmp_path / "cpu.max").write_text("max 100000\n")
        monkeypatch.setattr(refs.os, "cpu_count", lambda: 12)
        assert refs.cpu_quota(tmp_path) == 12


class TestMarker:
    """A reference another process is playing ("<file>.computing", touched while it plays) is waited for, not
    played twice; a marker untouched for STALE_S is a dead process's and is taken over."""

    def test_one_process_holds_a_marker_and_a_stale_one_is_taken_over(self, tmp_path):
        import os
        import time
        mark = refs.marker(tmp_path / "ai_like_19u_3600s_v.json")
        assert mark.name == "ai_like_19u_3600s_v.json.computing"
        assert refs.claim(mark) and refs.live(mark)
        assert not refs.claim(mark)                                  # held by a live process
        old = time.time() - refs.STALE_S - 5
        os.utime(mark, (old, old))
        assert not refs.live(mark) and refs.claim(mark) and refs.live(mark)

    def test_the_owner_touches_its_markers_and_removes_them(self, tmp_path):
        import os
        import time
        mark = tmp_path / "x.json.computing"
        assert refs.claim(mark)
        old = time.time() - 100
        os.utime(mark, (old, old))
        beat = refs.Beat([mark], every=0.05)
        time.sleep(0.3)
        assert time.time() - mark.stat().st_mtime < 50
        beat.release()
        assert not mark.exists() and beat.stop.is_set()

    def test_a_waiter_returns_when_the_file_is_written_or_the_other_process_is_gone(self, tmp_path, monkeypatch):
        import threading
        monkeypatch.setattr(refs, "POLL_S", 0.02)
        f1, f2 = tmp_path / "a.json", tmp_path / "b.json"
        for f in (f1, f2):
            refs.claim(refs.marker(f))
        threading.Timer(0.1, lambda: f1.write_text("{}")).start()
        threading.Timer(0.2, refs.marker(f2).unlink).start()             # stopped without writing it
        said = []
        p = refs.Pending(None, [], [], 0.0, said.append, None, others={"a": f1, "b": f2},
                         ready=lambda t: (tmp_path / f"{t}.json").exists())
        assert p.wait() == {"a": p.done["a"]} and "b" not in p.done
        assert any("another process" in x for x in said) and any("stopped without it" in x for x in said)
