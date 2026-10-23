"""The exam's proxy in the simulator (tools/nn/train/exam_sim.py): the bank's layout (base / swap), the pairs read like
the game's exam, the row of the skill block and the run card, only where it was played. The battles themselves (torch)
at the end: evaluate.play's seeds / armies / defend, exam_sim.play on a tiny network."""
import numpy as np
import pytest

from tools.nn.train import exam_sim, skill
from tools.ops import card


def per_battle(shares, r=4, trades=None):
    """Per-battle lists of seeds with the given (base, swap) win shares over r battles each."""
    won, side, seed, trade = [], [], [], []
    for i, (pb, ps) in enumerate(shares):
        for s, p in ((1, pb), (2, ps)):
            k = int(round(p * r))
            won += [True] * k + [False] * (r - k)
            side += [s] * r
            seed += [100 + i] * r
            trade += [(trades or {}).get((i, s), 0.1 if s == 1 else -0.1)] * r
    return won, side, seed, trade


class TestLayout:
    def test_every_seed_twice_r_in_a_row_from_an_even_place(self):
        bank = exam_sim.battle_seeds((7, 8, 9), 3)
        assert bank == [7] * 6 + [8] * 6 + [9] * 6
        for s in (7, 8, 9):
            at = [i for i, x in enumerate(bank) if x == s]
            assert at[0] % 2 == 0 and len(at) == 6          # scenes.Generated: side 1 attacks at even places -> swap
        assert len(exam_sim.battle_seeds()) == 12 * exam_sim.R
        with pytest.raises(ValueError):
            exam_sim.battle_seeds((1,), 0)

    def test_the_exam_s_seeds(self):
        assert exam_sim.SEEDS == (1000800014, 1000800163, 1000800092, 1000800042, 1000800127, 1000800133)
        assert exam_sim.ENEMY == "build/v2/itS27_4.pt" and exam_sim.ARMIES == "v2"


class TestPairs:
    def test_a_pair_is_2_0_or_0_2_only_when_both_sides_agree(self):
        assert exam_sim.pair_of(0.75, 0.6) == "2-0"
        assert exam_sim.pair_of(0.25, 0.0) == "0-2"
        assert exam_sim.pair_of(0.75, 0.25) == "1-1"
        assert exam_sim.pair_of(0.5, 1.0) == "1-1" and exam_sim.pair_of(0.5, 0.0) == "1-1"   # exactly a half: split

    def test_summary_counts_the_pairs_the_expected_pairs_the_wins_and_the_trade(self):
        shares = [(1.0, 0.75), (0.75, 0.25), (0.0, 0.25), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)]
        won, side, seed, trade = per_battle(shares, r=4)
        s = exam_sim.summary(won, side, seed, trade, timeouts=0.02)
        assert s["battles"] == 48 and s["pairs"] == [2, 3, 1]
        e20 = sum(pb * ps for pb, ps in shares)
        e02 = sum((1 - pb) * (1 - ps) for pb, ps in shares)
        assert s["expected"] == [round(e20, 2), round(e02, 2)]
        assert s["win"] == pytest.approx(np.mean(won), abs=1e-4) and s["wins12"] == pytest.approx(12 * np.mean(won), abs=0.01)
        assert s["trade"]["value"] == pytest.approx(0.0, abs=1e-9)
        se = np.std(trade, ddof=1) / np.sqrt(48)
        assert s["trade"]["se"] == pytest.approx(se, abs=1e-4)
        assert s["trade"]["hi"] - s["trade"]["lo"] == pytest.approx(2 * exam_sim.Z95 * se, abs=2e-4)
        first = s["by_seed"]["100"]
        assert (first["base"], first["swap"], first["pair"], first["n_base"]) == (1.0, 0.75, "2-0", 4)
        assert first["trade_base"] == pytest.approx(0.1) and first["trade_swap"] == pytest.approx(-0.1)
        assert s["timeouts"] == 0.02

    def test_a_seed_played_on_one_side_only_has_no_pair(self):
        s = exam_sim.summary([True, True], [1, 1], [5, 5], [0.2, 0.2])
        assert s["pairs"] == [0, 0, 0] and "pair" not in s["by_seed"]["5"] and s["by_seed"]["5"]["n_swap"] == 0

    def test_the_cell_and_the_title(self):
        s = exam_sim.summary(*per_battle([(1.0, 1.0), (0.0, 0.0)], r=2))
        s.update(enemy="build/v2/itS27_4.pt", R=2, seeds=[100, 101], fighters=True)
        assert exam_sim.cell(s).startswith("1/0/1 (exp 1.0/1.0), 6.0/12, +0.000 ± ")
        assert exam_sim.title(s).startswith("exam vs itS27_4 (fighters, 2 seeds x 2 sides x R 2, we defend)")
        assert exam_sim.cell(None) == "-" and exam_sim.cell({}) == "-"
        assert "exam point vs itS27_4 R 2: 1/0/1" in exam_sim.line(dict(s, seconds=5), "point")


def fake_exam():
    s = exam_sim.summary(*per_battle([(1.0, 0.5), (0.25, 0.0)], r=4))
    s.update(enemy="build/v2/itS27_4.pt", R=4, seeds=[100, 101], fighters=True, seconds=1,
             per_battle={"won": [True]})
    return s


class TestTables:
    def test_the_skill_summary_carries_the_exam_without_its_battles(self):
        res = {"skill": None, "by_opponent": {}, "exam": fake_exam()}
        s = skill.summary(res)
        assert s["exam"]["pairs"] == [0, 1, 1] and "per_battle" not in s["exam"]
        assert "exam" not in skill.summary({"skill": None, "by_opponent": {}})

    def test_the_row_only_where_the_exam_was_played_and_the_block_unchanged_without(self):
        plain = skill.summary({"skill": None, "by_opponent": {}})
        rows = skill.rows([plain, plain])
        assert not any(t.startswith("exam") for t, _ in rows)
        with_exam = skill.summary({"skill": None, "by_opponent": {}, "exam": fake_exam()})
        rows2 = skill.rows([plain, with_exam])
        exam = [(t, c) for t, c in rows2 if t.startswith("exam vs itS27_4")]
        assert len(exam) == 1 and exam[0][1][0] == "-" and exam[0][1][1].startswith("0/1/1 (exp ")
        assert [r for r in rows2 if not r[0].startswith("exam")] == rows          # the rest as before

    def test_the_run_card_shows_the_exam_rows_where_played(self, tmp_path):
        import json
        base = {"skill": {}, "by_opponent": {}, "seconds": 10}
        (tmp_path / "before.json").write_text(json.dumps(base), encoding="utf-8")
        (tmp_path / "eval_m10.json").write_text(json.dumps(dict(base, exam=fake_exam())), encoding="utf-8")
        lines, data = card.card(tmp_path)
        keys = [r["key"] for r in data["rows"]]
        assert "exam_trade" in keys and "exam_wins" in keys
        row = next(r for r in data["rows"] if r["key"] == "exam_wins")
        assert row["values"][0] is None and row["values"][1] == pytest.approx(12 * 7 / 16, abs=0.01)
        assert any(line.startswith("| exam vs itS27_4: wins /12") and "(0/1/1)" in line for line in lines)
        (tmp_path / "eval_m10.json").write_text(json.dumps(base), encoding="utf-8")
        _, data = card.card(tmp_path)
        assert not any(r["key"].startswith("exam") for r in data["rows"])


# --- the battles (torch) ------------------------------------------------------------------------------------

def tiny():
    torch = pytest.importorskip("torch")
    from tools.nn.model import config, policy
    cfg = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
    torch.manual_seed(0)
    return policy.Actor(cfg).eval()


class TestPlay:
    def test_seeds_armies_and_defend_in_evaluate_play(self, monkeypatch):
        actor = tiny()
        from tools.nn.train import evaluate, scenes
        seen = []
        real = scenes.Generated

        class Spy(real):
            def __init__(self, seeds, *a, **k):
                seen.append((list(seeds), k.get("armies")))
                super().__init__(seeds, *a, **k)
        monkeypatch.setattr(scenes, "Generated", Spy)
        seeds = [1000800014] * 2 + [1000800163] * 2
        res = evaluate.play(actor, opponents=("past",), past=actor, generated=4, max_units=4, limit_s=4.0,
                            paired=False, seeds=seeds, armies="v2", defend=True, measure=())
        b = res["battles"]["past"]
        assert seen[-1] == (seeds, "v2")                                  # the bank: these seeds, the v2 scheme
        assert b["seed"] == seeds and not any(b["attacks"])               # our network always defends
        assert b["side"] == [2, 1, 2, 1]                                   # even place: side 1 attacks -> we are side 2
        res = evaluate.play(actor, opponents=("nearest",), generated=8, max_units=4, limit_s=4.0, seeds=list(range(5, 9)),
                            measure=())
        assert seen[-1][1] == "v1" and sorted(set(res["battles"]["nearest"]["seed"])) == [5, 6, 7, 8]   # paired
        assert "baseline" not in res["by_opponent"]["nearest"]           # baselines only for EVAL_SEEDS
        with pytest.raises(ValueError):
            evaluate.play(actor, opponents=("past",), past=actor, generated=4, paired=True, defend=True)
        with pytest.raises(ValueError):
            evaluate.play(actor, opponents=("past",), past=actor, generated=4, paired=False, seeds=[1, 2])
        with pytest.raises(ValueError):
            evaluate.play(actor, opponents=("hold",), generated=4, paired=False, defend=True)   # `hold` only defends

    def test_exam_sim_plays_r_battles_a_seed_and_side(self, monkeypatch):
        actor = tiny()
        monkeypatch.setattr(exam_sim, "MAX_UNITS", 4)
        res = exam_sim.play(actor, enemy=actor, r=1, seeds=(1000800014, 1000800163), fighters=False, limit_s=4.0)
        assert res["battles"] == 4 and res["R"] == 1 and res["fighters"] is False
        pb = res["per_battle"]
        assert pb["side"] == [2, 1, 2, 1] and pb["seed"] == [1000800014] * 2 + [1000800163] * 2
        assert set(res["by_seed"]) == {"1000800014", "1000800163"}
        assert all("pair" in v for v in res["by_seed"].values()) and sum(res["pairs"]) == 2
        # the behaviour sums (the friends in the way) in the proxy's battles; same outcome without them
        for k in ("friend_skip_share_direct", "friend_skip_share_arc", "forced_lob_share_arc", "missile_melee_share"):
            assert k in res["behaviour"]
        assert "order_changes_per_min" not in res["behaviour"]                       # no liveliness
        bare = exam_sim.play(actor, enemy=actor, r=1, seeds=(1000800014, 1000800163), fighters=False, limit_s=4.0,
                             measure=())
        assert bare["behaviour"] == {} and bare["per_battle"] == res["per_battle"]
        import torch
        torch.manual_seed(5)
        before = torch.rand(1)
        torch.manual_seed(5)
        exam_sim.play(actor, enemy=actor, r=1, seeds=(1000800014,), fighters=False, limit_s=2.0)
        assert torch.equal(torch.rand(1), before)                          # the caller's random state untouched

    def test_the_proxy_plays_in_the_fighters_lab(self, monkeypatch):
        actor = tiny()
        from tools.nn.fighters import hook as fighters_hook
        monkeypatch.setattr(exam_sim, "MAX_UNITS", 4)
        res = exam_sim.play(actor, enemy=actor, r=1, seeds=(1000800014,), fighters=True, limit_s=3.0)
        assert res["battles"] == 2 and res["fighters"] is True and fighters_hook.HOOK is None   # installed and removed


class TestTest5:
    def test_the_exam_plays_only_with_its_profile_and_at_the_last_point(self, monkeypatch):
        pytest.importorskip("torch")
        from types import SimpleNamespace

        from tools.nn.train import cadence as cad
        from tools.nn.train import evaluate, test5
        monkeypatch.setattr(evaluate, "play", lambda *a, **k: {"by_opponent": {}, "battles": {}, "skill": None})
        calls = []
        monkeypatch.setattr(exam_sim, "play", lambda actor, enemy, r, **k: calls.append((enemy, r)) or fake_exam())
        args = SimpleNamespace(eval=8, drill_eval=0, profiles=(), exam_r=2, exam_enemy="x.pt", exam_at="last")
        plain = test5.evaluation(None, args, "cpu", cad.GAME, last=True)
        assert "exam" not in plain and not calls and "exam" not in plain["key"]     # the profile off: as before
        m0 = test5.metrics(plain)
        assert "exam" not in m0["skill"] and not any("exam" in x for x in test5.trend([(0, m0), (10, m0)]))
        args.profiles = ("exam",)
        assert "exam" not in test5.evaluation(None, args, "cpu", cad.GAME) and not calls   # not the last point
        res = test5.evaluation(None, args, "cpu", cad.GAME, last=True)
        assert calls == [("x.pt", 2)] and res["key"]["exam"]["R"] == 2 and res["exam"]["pairs"] == [0, 1, 1]
        lines = test5.trend([(0, m0), (10, test5.metrics(res))])
        row = next(x for x in lines if x.startswith("| exam vs itS27_4"))
        assert row.endswith("| - | 0/1/1 (exp " + row.split("| 0/1/1 (exp ")[1])
        args.exam_at = "all"
        assert "exam" in test5.evaluation(None, args, "cpu", cad.GAME) and len(calls) == 2
        assert test5.chosen(SimpleNamespace()) == profiles_full()


def profiles_full():
    from tools.nn.train import profiles
    return tuple(p for p in profiles.TEST5 if p != "exam")
