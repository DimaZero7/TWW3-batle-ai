"""tools/nn/train/teach_auto.py: the drills' adaptive teacher's rule (run.py --drill-teach auto).

Plain python: runs in the project's .venv. Level 1: the deficit and the share (k, cap, the match
dead zone), the state across evaluations (the first shares from the prior numbers or the default, a
drill without numbers keeps its share), where the prior numbers come from (a test5 point's eval json,
report.json's trend, after.json; a test5 run's latest.pt), the per-iteration table. The rollout's
share of labelled battles and the term's normalisation: tests/tools/test_nn_drill_teach.py (torch).
"""
import json

import pytest

from tools.nn.train import teach_auto as T


def ev(**drills):
    """A drill block {name: (net, skilled)} as evaluate.play_drills writes it."""
    return {n: {"win_rate": net, "games": 128, "scripts": {"naive": {"win_rate": 0.0}, "skilled": {"win_rate": sk}}}
            for n, (net, sk) in drills.items()}


class TestRule:
    def test_deficit_is_relative_to_the_script_and_never_negative(self):
        assert T.deficit(0.0, 1.0) == 1.0
        assert T.deficit(0.6875, 0.984) == pytest.approx(0.3013, abs=1e-4)
        assert T.deficit(0.9, 0.85) == 0.0                     # the network beats the script
        assert T.deficit(None, 1.0) is None and T.deficit(0.5, None) is None and T.deficit(0.5, 0.0) is None

    def test_share_is_k_times_the_deficit_within_the_cap(self):
        assert T.share(0.3, k=0.5, cap=0.25) == pytest.approx(0.15)
        assert T.share(0.98, k=0.5, cap=0.25) == 0.25          # kiting before the teacher: at the cap
        assert T.share(0.5, k=0.5, cap=0.25) == 0.25
        assert T.share(0.4, k=2.0, cap=0.6) == pytest.approx(0.6)
        assert T.share(1.0, k=0.5, cap=1.0) == 0.5

    def test_a_matched_drill_gets_nothing(self):
        assert T.share(0.0) == 0.0
        assert T.share(T.MATCH) == 0.0 and T.share(0.04) == 0.0       # noise around a match
        assert T.share(0.06) == pytest.approx(T.K * 0.06)
        assert T.share(None) is None

    def test_the_defaults(self):
        assert (T.WEIGHT, T.K, T.CAP, T.MATCH) == (0.15, 0.5, 0.25, 0.05)


class TestAuto:
    def test_first_shares_from_the_prior_then_from_every_evaluation(self):
        a = T.Auto(prior=ev(kiting=(0.016, 1.0), counter=(0.6875, 0.984)), source="before")
        rows = a.bind(["kiting", "counter", "hold_fire"])
        assert a.shares["kiting"] == 0.25 and a.shares["counter"] == pytest.approx(0.1507, abs=1e-3)
        assert a.shares["hold_fire"] == a.unknown == 0.125         # no numbers: half the cap
        assert [r["drill"] for r in rows] == ["kiting", "counter", "hold_fire"] and rows[0]["source"] == "before"
        assert rows[2]["net"] is None and rows[2]["deficit"] is None and rows[2]["share"] == 0.125
        assert a.weights() == {"kiting": 0.15, "counter": 0.15, "hold_fire": 0.15}
        # the network learned kiting (matches the script): its share goes to 0; hold_fire measured now
        rows = a.observe(ev(kiting=(1.0, 1.0), counter=(0.75, 0.984), hold_fire=(0.18, 0.852)),
                         agreement={"kiting": 0.81})
        assert a.shares == {"kiting": 0.0, "counter": pytest.approx(0.1189, abs=1e-3), "hold_fire": 0.25}
        k = rows[0]
        assert (k["net"], k["skilled"], k["deficit"], k["share_was"], k["share"], k["agree"]) == (1.0, 1.0, 0.0, 0.25, 0.0, 0.81)
        assert rows[1]["agree"] is None
        # a later evaluation without drills keeps the shares
        a.observe(None)
        assert a.shares["hold_fire"] == 0.25 and len(a.history) == 3

    def test_no_prior_the_default_for_all(self):
        a = T.Auto(k=1.0, cap=0.5, weight=0.2)
        rows = a.bind(["kiting"])
        assert a.shares == {"kiting": 0.25} and rows[0]["source"] == "default" and a.weights() == {"kiting": 0.2}
        assert a.meta() == {"k": 1.0, "cap": 0.5, "weight": 0.2, "match": T.MATCH}

    def test_the_table(self):
        a = T.Auto(prior=ev(kiting=(0.0, 1.0)))
        a.bind(["kiting", "counter"])
        lines = T.table(a.observe(ev(kiting=(0.5, 1.0)), {"kiting": 0.6}), "min 5:")
        assert lines[0] == "min 5:" and lines[2].startswith("| drill | net win | skilled win | deficit | teacher share")
        assert lines[4] == "| kiting | 0.500 | 1.000 | 0.500 | 0.25 → 0.25 | 0.600 |"
        assert lines[5] == "| counter | - | - | - | 0.12 → 0.12 | - |"
        assert T.table([]) == []


class TestPrior:
    def folder(self, tmp_path):
        f = tmp_path / "test5" / "it1"
        f.mkdir(parents=True)
        (f / "eval_m5.json").write_text(json.dumps({"drills": ev(kiting=(0.5, 1.0))}), encoding="utf-8")
        (f / "after.json").write_text(json.dumps({"drills": ev(kiting=(0.9, 1.0))}), encoding="utf-8")
        (f / "report.json").write_text(json.dumps({"train_args": {"minutes": 25.0},
                                                   "trend": {"10": {"drills": ev(kiting=(0.7, 1.0))}}}),
                                       encoding="utf-8")
        return f

    def test_a_test5_point_finds_its_own_evaluation(self, tmp_path):
        f = self.folder(tmp_path)
        d, src = T.prior(f / "m5.pt")
        assert d["kiting"]["win_rate"] == 0.5 and src.endswith("eval_m5.json")
        d, src = T.prior(f / "m10.pt")                               # only in report.json's trend
        assert d["kiting"]["win_rate"] == 0.7 and "trend minute 10" in src
        d, src = T.prior(f / "m25.pt")                               # the last minute: after.json
        assert d["kiting"]["win_rate"] == 0.9 and src.endswith("after.json")
        assert T.prior(f / "m15.pt") == (None, None)

    def test_a_test5_runs_latest_finds_the_after_evaluation(self, tmp_path):
        f = self.folder(tmp_path)
        run = tmp_path / "runs" / "test5_it1"
        run.mkdir(parents=True)
        d, src = T.prior(run / "latest.pt")
        assert d["kiting"]["win_rate"] == 0.9 and src == str(f / "after.json")

    def test_nothing_found(self, tmp_path):
        assert T.prior(None) == (None, None)
        assert T.prior(tmp_path / "best.pt") == (None, None)


class TestTransferOwnSettings:
    """A drill's own reference, cap and stop for the teacher in normal battles (drills.Drill transfer_ref,
    teach_cap, teach_stop: direct_fire, a skill ai_like never uses). Set by hand here (bind reads them from the
    drill: tests/tools/test_nn_drill_direct_fire.py needs torch)."""

    @staticmethod
    def _teacher():
        t = T.Transfer()
        key = "direct_fire@normal"
        t.caps[key], t.refs[key], t.stops[key] = 0.08, 0.6, 0.15
        T.Auto.bind(t, [key, "kiting@normal"])
        return t, key

    @staticmethod
    def _xfer(df, ai_df, kit, ai_kit):
        return {"direct_fire": {"network": {"share": df}, "ai_like": {"share": ai_df}},
                "kiting": {"network": {"share": kit}, "ai_like": {"share": ai_kit}}}

    def test_the_own_reference_and_cap(self):
        t, key = self._teacher()
        assert t.shares[key] == pytest.approx(0.04) and t.shares["kiting@normal"] == pytest.approx(T.NORMAL_CAP / 2)
        rows = {r["drill"]: r for r in t.observe(self._xfer(0.0, 0.0, 0.0, 0.7))}
        assert rows[key]["skilled"] == 0.6 and t.shares[key] == pytest.approx(0.08)      # ai_like's 0 is not the reference
        assert t.shares["kiting@normal"] == pytest.approx(T.NORMAL_CAP)                    # the others as before
        t.observe(self._xfer(0.57, 0.0, 0.7, 0.7))                                         # within the match of 0.6
        assert t.shares[key] == 0.0

    def test_the_stop_switches_the_drill_off_for_the_run(self):
        t, key = self._teacher()
        t.observe(self._xfer(0.0, 0.0, 0.0, 0.7), rating=1.0)
        assert t.shares[key] > 0
        rows = {r["drill"]: r for r in t.observe(self._xfer(0.0, 0.0, 0.0, 0.7), rating=0.8)}
        assert t.shares[key] == 0.0 and rows[key].get("stopped") and t.shares["kiting@normal"] > 0
        t.observe(self._xfer(0.0, 0.0, 0.0, 0.7), rating=1.2)                             # stays off
        assert t.shares[key] == 0.0

    def test_the_mistake_stop(self):
        t, key = self._teacher()
        t.stops_mistake[key] = 0.1
        x = self._xfer(0.0, 0.0, 0.0, 0.7)
        x["direct_fire"]["network"]["mistake"] = 0.2
        t.observe(x)
        assert t.shares[key] > 0 and t.mistake0[key] == 0.2
        x["direct_fire"]["network"]["mistake"] = 0.35
        t.observe(x)
        assert t.shares[key] == 0.0 and key in t.stopped and t.shares["kiting@normal"] > 0
