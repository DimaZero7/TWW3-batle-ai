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
