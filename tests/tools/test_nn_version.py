"""tools/nn/train/version.py: the simulator version on the host (no torch) and the comparison that lets
the baseline canary adopt an older version's file."""
import json

from tools.nn.train import version


def doc(n, seed0=1000, winner=1, lost=10.0):
    """A baseline document of n pairs as evaluate.script_battles writes it."""
    return {"seed": [seed0 + i for i in range(n)], "winner": [winner] * n, "lost": [[lost, 20.0]] * n,
            "start": [[100.0, 100.0]] * n, "budget": [1000.0] * n, "factions": [["emp", "skv"]] * n, "attacker": [1, 2] * (n // 2)}


class TestVersion:
    def test_the_hash_is_stable_and_twelve_characters(self):
        v = version.sim_version()
        assert len(v) == 12 and v == version.sim_version()

    def test_the_evaluation_s_version_follows_its_own_code_not_the_simulator_s(self, tmp_path, monkeypatch):
        f = tmp_path / "evaluate.py"
        f.write_text("A = 1\n", encoding="utf-8")
        monkeypatch.setattr(version, "EVAL_FILES", (str(f),))
        v, sim = version.eval_version(), version.sim_version()
        f.write_text("A = 2\n", encoding="utf-8")
        assert version.eval_version() != v and version.sim_version() == sim
        f.write_bytes(b"A = 1\r\n")                                   # line ends do not count
        assert version.eval_version() == v

    def test_the_baseline_file_name(self):
        assert version.baseline_name("ai_like", 19, 3600.0, "abc") == "ai_like_19u_3600s_abc.json"


class TestOlderBaselines:
    def test_other_versions_of_the_same_opponent_newest_first_with_the_seeds(self, tmp_path):
        seeds = doc(4)["seed"]
        (tmp_path / "ai_like_19u_3600s_old1.json").write_text(json.dumps(doc(4)), encoding="utf-8")
        (tmp_path / "ai_like_19u_3600s_old2.json").write_text(json.dumps(doc(4)), encoding="utf-8")
        (tmp_path / "ai_like_19u_3600s_cur.json").write_text(json.dumps(doc(4)), encoding="utf-8")       # this version: not a candidate
        (tmp_path / "nearest_19u_3600s_old1.json").write_text(json.dumps(doc(4)), encoding="utf-8")       # another opponent
        (tmp_path / "ai_like_19u_900s_old1.json").write_text(json.dumps(doc(4)), encoding="utf-8")        # another limit
        (tmp_path / "ai_like_19u_3600s_other.json").write_text(json.dumps(doc(4, seed0=5)), encoding="utf-8")   # other seeds
        (tmp_path / "ai_like_19u_3600s_broken.json").write_text("{", encoding="utf-8")
        import os
        os.utime(tmp_path / "ai_like_19u_3600s_old2.json", (2_000_000_000, 2_000_000_000))               # the newest
        out = version.older_baselines(tmp_path, "ai_like", 19, 3600.0, "cur", seeds)
        assert [p.name for p, _ in out] == ["ai_like_19u_3600s_old2.json", "ai_like_19u_3600s_old1.json"]
        assert all(d["seed"] == seeds for _, d in out)

    def test_a_shorter_prefix_of_seeds_still_serves(self, tmp_path):
        (tmp_path / "ai_like_19u_3600s_old.json").write_text(json.dumps(doc(8)), encoding="utf-8")
        assert len(version.older_baselines(tmp_path, "ai_like", 19, 3600.0, "cur", doc(4)["seed"])) == 1
        assert version.older_baselines(tmp_path, "ai_like", 19, 3600.0, "cur", doc(10)["seed"]) == []


class TestSamePrefix:
    def test_identical_first_pairs_match_whatever_follows(self):
        old = doc(8)
        old["winner"][6] = 2                                  # a later pair differs: not compared
        assert version.same_prefix(old, doc(4), 4)

    def test_any_field_of_the_prefix_differing_breaks_the_match(self):
        for k in version.FIELDS:
            canary = doc(4)
            v = canary[k][1]
            canary[k][1] = (3 - v) if isinstance(v, int) else ([x + 1 for x in v] if isinstance(v, list) and v and isinstance(v[0], float)
                                                                 else list(reversed(v)) if isinstance(v, list) else v + 1)
            assert not version.same_prefix(doc(8), canary, 4), k

    def test_a_short_or_incomplete_document_never_matches(self):
        assert not version.same_prefix(doc(2), doc(4), 4)
        old = doc(8)
        del old["attacker"]
        assert not version.same_prefix(old, doc(4), 4)
