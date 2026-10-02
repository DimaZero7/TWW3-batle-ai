"""tools/nn/train/matchups.py: wins by our faction and role and by matchup (plain numpy)."""
import numpy as np
import pytest

from tools.nn.train import matchups

EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def battles():
    """Six battles: (won, ours, theirs, we attack, own gold lost, enemy gold destroyed, budget)."""
    rows = [(1, EMP, SKV, 1, 100, 300, 1000), (0, EMP, SKV, 0, 300, 100, 1000), (1, EMP, EMP, 1, 0, 200, 1000),
            (0, SKV, EMP, 1, 400, 0, 1000), (1, SKV, EMP, 0, 100, 500, 1000), (1, SKV, SKV, 0, 200, 200, 1000)]
    won, own, enemy, attacks, lost, destroyed, budget = zip(*rows)
    return np.array(won, bool), list(own), list(enemy), np.array(attacks, bool), \
        (np.array(lost, float), np.array(destroyed, float), np.array(budget, float))


class TestGroup:
    def test_wins_by_our_faction_and_role(self):
        won, own, enemy, attacks, gold = battles()
        g = matchups.group(won, own, enemy, attacks, gold)
        assert list(g["factions"]) == ["EMP", "SKV"]
        assert g["factions"]["EMP"]["attack"] == {"games": 2, "wins": 2, "win_rate": 1.0}
        assert g["factions"]["EMP"]["defend"] == {"games": 1, "wins": 0, "win_rate": 0.0}
        assert g["factions"]["SKV"]["attack"] == {"games": 1, "wins": 0, "win_rate": 0.0}
        assert g["factions"]["SKV"]["defend"]["games"] == 2 and g["factions"]["SKV"]["defend"]["wins"] == 2

    def test_matchups_are_ours_first_with_the_gold_exchange(self):
        won, own, enemy, attacks, gold = battles()
        m = matchups.group(won, own, enemy, attacks, gold)["matchups"]
        assert list(m) == ["EMP-EMP", "EMP-SKV", "SKV-EMP", "SKV-SKV"]
        assert (m["EMP-SKV"]["games"], m["EMP-SKV"]["wins"]) == (2, 1)
        assert m["EMP-SKV"]["gold_destroyed"] == 200 and m["EMP-SKV"]["gold_lost"] == 200
        assert m["EMP-SKV"]["gold_ratio"] == pytest.approx(1.0) and m["EMP-SKV"]["gold_trade"] == pytest.approx(0.0)
        assert m["SKV-EMP"]["gold_ratio"] == pytest.approx(500 / 500)
        assert m["SKV-EMP"]["gold_trade"] == pytest.approx(((0 - 400) + (500 - 100)) / 2 / 1000)
        assert sum(c["games"] for c in m.values()) == 6

    def test_no_result_battles_count_as_games_not_wins(self):
        g = matchups.group([True, False], [EMP, EMP], [SKV, SKV], [True, False], void=[False, True])
        assert g["matchups"]["EMP-SKV"] == {"games": 2, "wins": 1, "win_rate": 0.5, "no_result": 1}
        assert g["factions"]["EMP"]["defend"]["no_result"] == 1

    def test_the_mean_margin_per_matchup(self):
        won, own, enemy, attacks, gold = battles()
        m = matchups.group(won, own, enemy, attacks, gold, margin=[0.5, -0.25, 0.9, -0.1, 0.2, 0.4])["matchups"]
        assert m["EMP-SKV"]["margin"] == pytest.approx(0.125) and m["SKV-SKV"]["margin"] == pytest.approx(0.4)
        assert "margin" not in matchups.group(won, own, enemy, attacks)["matchups"]["EMP-SKV"]

    def test_an_unknown_faction_keeps_the_end_of_its_key(self):
        g = matchups.group([True], ["wh3_main_cth_cathay"], [None], [True])
        assert list(g["matchups"]) == ["h_cathay-?"]


class TestText:
    def test_the_line_and_the_block_rows(self):
        won, own, enemy, attacks, gold = battles()
        g = matchups.group(won, own, enemy, attacks, gold)
        line = matchups.text(g)
        assert line.startswith("EMP attack 2/2, defend 0/1; SKV attack 0/1, defend 2/2 | EMP-EMP 1/1 1.00")
        assert "EMP-SKV 1/2 0.50 gold 1.00" in line
        rows = dict(matchups.rows([None, g]))                  # an old evaluation without factions: "-"
        assert rows["EMP win attack / defend (n 2/1)"] == ["- / -", "1.000 / 0.000"]
        assert rows["SKV-EMP win / gold ratio (n 2)"] == ["- / -", "0.500 / 1.00"]
