"""tools/nn/train/evaluate.py fair metrics with torch: swapped pairs of generated battles, the script
baseline (script against itself on the same battles, cached) and the rating over all battles."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, policy  # noqa: E402
from tools.nn.train import evaluate  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)


def actor():
    torch.manual_seed(0)
    return policy.Actor(CFG).eval()


class TestPairedOrder:
    def test_both_battles_of_a_pair_are_one_bank_battle_from_the_two_sides(self):
        opp, pair, side, seed_i = evaluate.paired_order(4, 2)
        assert len(opp) == 16 and sorted(np.bincount(opp)) == [8, 8]
        for o in (0, 1):
            for p in range(4):
                at = np.nonzero((opp == o) & (pair == p))[0]
                assert len(at) == 2 and sorted(side[at]) == [1, 2]
                assert at[0] % 2 == at[1] % 2                    # the same role (scenes.Generated: by position)
                assert seed_i[at[0]] == seed_i[at[1]] == p
        with pytest.raises(AssertionError):
            evaluate.paired_order(3, 1)
        assert evaluate.pair_count(512) == 256 and evaluate.pair_count(6) == 2 and evaluate.pair_count(1) == 2

    def test_the_small_share_ends_at_a_block(self):
        seeds = list(range(24))
        for share in (0.35, 0.5, 0.1):
            s = evaluate.padded(seeds, (share, 6))
            assert s[:24] == seeds and round(share * len(s)) % 4 == 0
        assert evaluate.padded(seeds, None) == seeds


class TestPairedEvaluation:
    def test_pairs_margin_baseline_and_rating(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        res = evaluate.play(actor(), opponents=("nearest", "hold"), generated=8, limit_s=6.0, together=True,
                            baseline=True)
        near, hold = res["by_opponent"]["nearest"], res["by_opponent"]["hold"]
        assert near["games"] == 8 and hold["games"] == 8
        per = res["battles"]["nearest"]
        for p in range(4):
            at = [i for i, x in enumerate(per["pair"]) if x == p]
            assert len(at) == 2 and sorted(per["side"][i] for i in at) == [1, 2]
            assert per["seed"][at[0]] == per["seed"][at[1]]
            # the armies swap with the sides; the attacking army stays: our role swaps too
            assert (per["own"][at[0]], per["enemy"][at[0]]) == (per["enemy"][at[1]], per["own"][at[1]])
            assert per["attacks"][at[0]] != per["attacks"][at[1]]
        assert near["pairs"]["pairs"] == 4 and "pairs" not in hold          # `hold` only defends: no pairs
        assert all(res["battles"]["hold"]["attacks"])
        assert all(-1 <= m <= 1 for m in per["margin"]) and -1 <= near["margin"] <= 1
        assert all("margin" in c for c in near["matchups"].values())
        # the script baseline: nearest against nearest, hold against hold, on the same seeds
        b = near["baseline"]
        assert b["all"]["games"] == 8 and b["cache"].startswith("nearest_19u_6s_")
        assert hold["baseline"]["all"]["win_script"] == 0.0              # hold attacking: loses on time
        files = sorted(p.name for p in tmp_path.iterdir())
        assert len(files) == 2
        # the rating over all battles of both opponents
        sk = res["skill"]
        assert sk["battles"] == 16 and set(sk["by_opponent"]) == {"nearest", "hold"}
        assert sk["overall"]["lo"] < sk["overall"]["value"] < sk["overall"]["hi"]

    def test_a_cached_baseline_is_read_back_and_serves_fewer_seeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        played = []
        real = evaluate.script_battles
        monkeypatch.setattr(evaluate, "script_battles", lambda *a, **k: played.append(a[0]) or real(*a, **k))
        first = evaluate.baselines(["nearest"], 4, limit_s=4.0)
        again = evaluate.baselines(["nearest"], 2, limit_s=4.0)
        assert played == [["nearest"]]
        assert again["nearest"]["winner"] == first["nearest"]["winner"][:2]
        assert again["nearest"]["cache"] == first["nearest"]["cache"]
        # a battle's attacker as in a paired evaluation: side 1 attacks at even positions
        assert first["nearest"]["attacker"] == [1, 2, 1, 2]

    def test_the_canary_adopts_an_older_version_s_baseline_when_its_first_pairs_are_identical(self, tmp_path, monkeypatch):
        """A code change outside a script battle changes the version but not the battles: with CANARY the
        cache plays the first pairs only, finds them identical in the older file and copies it under the
        new version; a file whose pairs differ (another simulator) is never adopted."""
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        full = evaluate.baselines(["nearest"], 4, limit_s=4.0)["nearest"]
        old = tmp_path / f"nearest_19u_4s_{evaluate.sim_version()}.json"
        other = old.with_name("nearest_19u_4s_oldversion.json")
        old.rename(other)                                               # the file of an "older version"
        played = []
        real = evaluate.script_battles
        monkeypatch.setattr(evaluate, "script_battles", lambda names, n, *a, **k: played.append((list(names), n)) or real(names, n, *a, **k))
        monkeypatch.setattr(evaluate, "CANARY", 2)
        got = evaluate.baselines(["nearest"], 4, limit_s=4.0)["nearest"]
        assert played == [(["nearest"], 2)]                             # the canary only, not the 4 pairs
        assert old.is_file() and got["winner"] == full["winner"] and got["cache"] == old.name
        # a differing older file: the canary does not match, the whole baseline is played
        doc = dict(json.loads(other.read_text(encoding="utf-8")))
        doc["winner"] = [3 - w for w in doc["winner"]]
        other.write_text(json.dumps(doc), encoding="utf-8")
        old.unlink()
        played.clear()
        got = evaluate.baselines(["nearest"], 4, limit_s=4.0)["nearest"]
        assert played == [(["nearest"], 2), (["nearest"], 4)] and got["winner"] == full["winner"]
        # CANARY 0 (the default): straight to the whole baseline
        monkeypatch.setattr(evaluate, "CANARY", 0)
        old.unlink()
        played.clear()
        evaluate.baselines(["nearest"], 4, limit_s=4.0)
        assert played == [(["nearest"], 4)]

    def test_the_script_view_takes_the_network_s_side(self):
        base = {"seed": [5, 6], "winner": [1, 2], "lost": [[100.0, 300.0], [50.0, 0.0]], "budget": [1000.0, 500.0]}
        won, trade = evaluate.script_view(base, [5, 5, 6], [1, 2, 2])
        assert won.tolist() == [True, False, True]
        assert trade.tolist() == pytest.approx([0.2, -0.2, 0.1])

    def test_the_version_hash_is_stable(self):
        v = evaluate.sim_version()
        assert len(v) == 12 and v == evaluate.sim_version()


class TestUnpaired:
    def test_generated_battles_once_each_still_report_the_rating(self):
        res = evaluate.play(actor(), opponents=("nearest",), generated=4, limit_s=3.0, paired=False)
        assert res["by_opponent"]["nearest"]["games"] == 4 and "pairs" not in res["by_opponent"]["nearest"]
        assert res["skill"]["battles"] == 4

    def test_scenes_come_in_pairs(self):
        res = evaluate.play(actor(), opponents=("nearest",), per_scene=2, limit_s=2.0,
                            scene_list=[("arena", "attack")])
        assert res["by_opponent"]["nearest"]["pairs"]["pairs"] == 1
        assert res["battles"]["nearest"]["side"] == [1, 2]


class TestReport:
    def test_the_skill_block_leads_the_report_and_the_trend(self):
        from tools.nn.train import skill, test5

        def point(r):
            return {"ai_like/attack": {"win": 0.5}, "ai_like/all": {"kind_hold": 0.2},
                    "skill": {"rating": {"overall": {"value": r, "se": 0.1}, "by_opponent": {"ai_like": {"value": r, "se": 0.1}},
                                         "faction_edge": {"EMP-SKV": {"value": -0.8, "se": 0.1}},
                                         "attack": {"value": 0.1, "se": 0.05}},
                              "pairs": {"ai_like": {"pairs": 4, "won_both": 0.5, "split": 0.25, "lost_both": 0.25,
                                                    "pair_score": 0.25}},
                              "advantage": {"ai_like": {"all": {"games": 8, "win_adv": 0.125, "gold_adv": 0.05},
                                                        "matchups": {}}},
                              "margin": {"ai_like": {"all": 0.2}}}}
        text = "\n".join(test5.trend([(0, point(0.2)), (10, point(0.5))]))
        assert text.startswith("skill (fair metrics")
        assert "| rating, overall (logit, ± 95%) | +0.20 ± 0.20 | +0.50 ± 0.20 |" in text
        assert "| pair score (both/split/neither), ai_like (n 4) | +0.250 (0.50/0.25/0.25) |" in text
        assert "| vs script win / gold adv, ai_like all (n 8) | +0.125 / +0.050 |" in text
        assert "| margin, ai_like all | +0.200 | +0.200 |" in text
        assert "| win rate, skill" not in text                                    # not an opponent
        table = "\n".join(test5.table(point(0.2), point(0.5)))
        assert "| rating, overall (logit, ± 95%) | +0.20 ± 0.20 → +0.50 ± 0.20 |" in table
        assert test5.trend([(0, {"ai_like/attack": {"win": 0.5}})])[0].startswith("| metric")   # an old evaluation
        assert skill.Z95 > 1.95

    def test_metrics_keep_the_old_fields_and_add_the_skill(self):
        from tools.nn.train import test5
        res = evaluate.play(actor(), opponents=("nearest",), generated=4, limit_s=3.0)
        m = test5.metrics(res)
        assert {"nearest/attack", "nearest/defend", "nearest/all", "nearest/factions", "skill"} <= set(m)
        assert m["skill"]["rating"]["battles"] == 4 and "nearest" in m["skill"]["pairs"]


class TestCompact:
    def test_a_narrowed_batch_plays_its_battles_as_the_whole_one(self, monkeypatch):
        from tools.nn.train import cadence as cad
        from tools.nn.train import league, rollout, scenes
        real = rollout.Battles._act
        # the past version samples its orders: greedy here, so both batches draw nothing at random
        monkeypatch.setattr(rollout.Battles, "_act", lambda self, a, o, f, r, h, g: real(self, a, o, f, r, h, True))
        sc = scenes.SCENES[:1]
        lay = evaluate.combined(("nearest", "past", "hold_shoot"), 2, 1, scenes.attackers(sc))
        # a decision a second with the orders one step late, not at random (a random landing step is drawn per
        # battle: the batch's size would set the draws)
        cadence = cad.Cadence(1.0, 0.5)
        envs = [rollout.Battles(lay, sc, params=rollout.params_with_limit(60.0), spread=evaluate.SPREAD, seed=1,
                                auto_reset=False, cadence=cadence) for _ in range(2)]
        torch.manual_seed(1)
        past = policy.Actor(CFG).eval()
        for env in envs:
            env.set_past(past)
            for _ in range(3):
                env.step(actor(), None, True)
        whole, part = envs
        keep = part.narrow([5, 0, 2, 3])
        assert keep.tolist() == [0, 2, 3, 5] and part.B == 4 and part.R == 4
        for _ in range(4):
            whole.step(actor(), None, True)
            part.step(actor(), None, True)
        # the state: the same up to float rounding too (a batch's size sets the order of the network's sums, and
        # a greedy point or bearing may differ in the last digit)
        for k, v in part.st.u.items():
            w = whole.st.u[k][keep]
            assert torch.allclose(v, w, atol=1e-3) if v.is_floating_point() else torch.equal(v, w), k
        assert torch.equal(part.st.t, whole.st.t[keep])
        assert torch.equal(part.kind_battle, whole.kind_battle[keep])
        assert torch.equal(part.order_battle, whole.order_battle[keep])
        at = torch.isin(whole.rows_learn % whole.B, keep)
        # the network's memory: the same up to float rounding (a batch's size sets the order of the sums)
        assert torch.allclose(part.h_learn, whole.h_learn[at], atol=1e-4)

    def test_the_batch_shrinks_to_the_smallest_size_that_holds_the_running_battles(self, monkeypatch):
        monkeypatch.setattr(evaluate, "BUCKETS", (256, 64))
        assert evaluate.bucket(300, 1536) == 1536
        assert evaluate.bucket(200, 1536) == 256
        assert evaluate.bucket(3, 1536) == 64
        assert evaluate.bucket(3, 64) == 64

    def test_script_battles_end_the_same_in_a_shrinking_batch(self, monkeypatch):
        # the scripts draw nothing at random: every battle ends the same, whatever the batch around it
        monkeypatch.setattr(evaluate, "CHECK_EVERY", 2)
        monkeypatch.setattr(evaluate, "BUCKETS", tuple(range(2, 8)))
        x = evaluate.script_battles(["nearest"], 8, max_units=4, limit_s=900.0)
        monkeypatch.setattr(evaluate, "BUCKETS", ())
        y = evaluate.script_battles(["nearest"], 8, max_units=4, limit_s=900.0)
        assert x == y and len(set(x["nearest"]["winner"])) == 2
