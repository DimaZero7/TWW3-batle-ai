"""tools/nn/train/skill.py: the network's skill apart from the faction matchup (plain numpy):
swapped pairs, the advantage over the script baseline, the margin and the rating with a faction term."""
import numpy as np
import pytest

from tools.nn.train import matchups, skill

EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


class TestPairs:
    def test_won_both_split_lost_both_and_the_score(self):
        won = [1, 1, 1, 0, 0, 0, 1, 0]
        pair = [0, 0, 1, 1, 2, 2, -1, -1]                       # the last two: not paired
        p = skill.pairs(won, pair)
        assert (p["pairs"], p["incomplete"]) == (3, 0)
        assert (p["won_both"], p["split"], p["lost_both"]) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
        assert p["pair_score"] == pytest.approx(0.0)

    def test_a_pair_with_no_result_or_a_missing_half_is_incomplete(self):
        p = skill.pairs([1, 1, 1, 0, 1], [0, 0, 1, 1, 2], void=[0, 0, 0, 1, 0])
        assert (p["pairs"], p["incomplete"], p["pair_score"]) == (1, 2, 1.0)
        assert skill.pairs([1], [-1])["pair_score"] is None

    def test_the_pair_score_ignores_a_faction_edge_that_decides_single_battles(self):
        # the favoured side always wins: every pair is split, whatever the share of wins
        rng = np.random.default_rng(0)
        n = 200
        favoured = rng.integers(1, 3, n)                         # per pair, the side that wins
        won = np.stack([favoured == 1, favoured == 2], 1).ravel()
        p = skill.pairs(won, np.repeat(np.arange(n), 2))
        assert p["split"] == 1.0 and p["pair_score"] == 0.0


class TestPairGold:
    def test_one_pair_by_hand(self):
        # battle 0: we play army A (side 1), destroy 600, lose 300, margin +0.4; battle 1: we play army B
        g = skill.pair_gold([0, 0], [1, 2], [600, 300], [300, 500], [1000, 1000], [0.4, -0.3])
        assert (g["pairs"], g["incomplete"]) == (1, 0)
        assert g["pair_gold"]["value"] == pytest.approx(0.1) and g["pair_gold"]["se"] == 0.0
        assert (g["net_ratio"], g["opp_ratio"]) == pytest.approx((900 / 800, 800 / 900))
        w, st = g["weak"], g["strong"]                          # B: our lower margin
        assert (w["trade_net"], w["trade_opp"]) == pytest.approx((-0.2, -0.3))
        assert (w["ratio_net"], w["ratio_opp"]) == pytest.approx((300 / 500, 300 / 600))
        assert (st["trade_net"], st["trade_opp"]) == pytest.approx((0.3, 0.2))
        assert (st["ratio_net"], st["ratio_opp"]) == pytest.approx((2.0, 500 / 300))
        # either army: our trade with it minus the opponent's with it is the pair gold
        assert w["trade_net"] - w["trade_opp"] == pytest.approx(g["pair_gold"]["value"])

    @staticmethod
    def battles(n, skill_x, rng):
        """n pairs of random unequal armies; an army's side destroys strength x the enemy's cost x a
        noise shared by both battles of the pair (the armies decide), our hands x skill_x."""
        s = rng.lognormal(0, 0.6, (n, 2))                       # armies A, B
        cost = rng.uniform(500, 3000, n)
        noise = rng.lognormal(0, 0.2, (n, 2))
        f = (lambda x, y, k: np.minimum(cost, cost * 0.4 * s[:, x] / s[:, y] * noise[:, k]))
        dA, dB = f(0, 1, 0), f(1, 0, 1)                         # what army A / B destroys with even hands
        # battle a: we play A; battle b: we play B (the opponent A)
        d = np.stack([np.minimum(cost, dA * skill_x), np.minimum(cost, dB * skill_x)], 1)
        lo = np.stack([dB, dA], 1)
        m = np.tanh((d - lo) / cost[:, None])
        return (np.repeat(np.arange(n), 2), np.tile([1, 2], n), d.ravel(), lo.ravel(),
                np.repeat(cost, 2), m.ravel())

    def test_even_hands_give_zero_whatever_the_armies(self):
        g = skill.pair_gold(*self.battles(300, 1.0, np.random.default_rng(5)))
        assert g["pair_gold"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert g["exchange"]["value"] == pytest.approx(1.0) and g["net_ratio"] == pytest.approx(1.0)
        assert g["weak"]["ratio_net"] == pytest.approx(g["weak"]["ratio_opp"])
        assert g["weak"]["ratio_net"] < 1 < g["strong"]["ratio_net"]

    def test_better_hands_show_with_an_interval(self):
        rng = np.random.default_rng(6)
        g = skill.pair_gold(*self.battles(300, 1.2, rng))
        pg = g["pair_gold"]
        assert pg["value"] > 0 and pg["lo"] > 0                 # every pair split or not, the gold shows it
        # ours x1.2 and the opponent's / 1.2 (1.44), a little less where an army is destroyed whole (the cap)
        assert 1.25 < g["exchange"]["value"] < 1.45 and g["exchange"]["lo"] > 1
        assert g["weak"]["ratio_net"] > g["weak"]["ratio_opp"] and g["weak"]["trade_net"] > g["weak"]["trade_opp"]
        worse = skill.pair_gold(*self.battles(300, 0.8, rng))
        assert worse["pair_gold"]["hi"] < 0 and worse["exchange"]["hi"] < 1

    def test_incomplete_pairs_and_ties(self):
        nan = float("nan")
        g = skill.pair_gold([0, 0, 1, 1, 2, 2, 3, 3, 4, -1], [1, 2, 1, 1, 1, 2, 1, 2, 1, 2],
                            [5, 5, 5, 5, 5, 5, nan, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [100] * 10,
                            [0, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0], void=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert (g["pairs"], g["incomplete"]) == (1, 4)          # same sides, no result, no gold, half a pair
        assert g["pair_gold"]["value"] == 0.0 and g["weak"] == {"pairs": 0}    # a tie: no weak army
        assert skill.pair_gold([], [], [], [], [], [])["pair_gold"] is None


class TestMargin:
    def test_the_winner_s_gold_left_signed_by_our_side(self):
        lost = np.array([[100.0, 900.0], [500.0, 250.0], [0.0, 0.0]])
        start = np.array([[1000.0, 900.0], [1000.0, 1000.0], [800.0, 800.0]])
        m = skill.margin([1, 2, 0], [1, 1, 2], lost, start)
        assert m.tolist() == pytest.approx([0.9, -0.75, 0.0])


class TestAdvantage:
    def test_the_network_minus_the_script_on_the_same_battles_per_matchup(self):
        own, enemy = [EMP, EMP, SKV, SKV], [SKV, SKV, EMP, EMP]
        a = skill.advantage([1, 0, 1, 1], [0, 0, 1, 1], [0.2, -0.1, 0.3, 0.1], [-0.2, -0.3, 0.3, 0.3], own, enemy)
        es, se = a["matchups"]["EMP-SKV"], a["matchups"]["SKV-EMP"]
        assert list(a["matchups"]) == ["EMP-SKV", "SKV-EMP"]
        assert (es["win_net"], es["win_script"], es["win_adv"]) == pytest.approx((0.5, 0.0, 0.5))
        assert es["gold_adv"] == pytest.approx(0.3) and es["games"] == 2
        assert (se["win_adv"], se["gold_adv"]) == pytest.approx((0.0, -0.1))
        assert a["all"]["win_adv"] == pytest.approx(0.25) and a["all"]["gold_adv"] == pytest.approx(0.1)


def synthetic(n, skv_share, rating, edge, attack, rng, opponent="ai_like"):
    """n battles of a network with `rating` (logit) against one opponent; Skaven beat the Empire by
    `edge` (the EMP-SKV edge is -edge); our faction even, theirs Skaven with skv_share."""
    own = np.where(rng.random(n) < 0.5, EMP, SKV)
    enemy = np.where(rng.random(n) < skv_share, SKV, EMP)
    att = rng.random(n) < 0.5
    sign = np.where((own == EMP) & (enemy == SKV), 1, np.where((own == SKV) & (enemy == EMP), -1, 0))
    logit = rating - edge * sign + attack * np.where(att, 1, -1)
    won = rng.random(n) < 1 / (1 + np.exp(-logit))
    return won, list(own), list(enemy), att, [opponent] * n


class TestFit:
    def test_a_known_rating_faction_edge_and_role_come_back(self):
        rng = np.random.default_rng(1)
        won, own, enemy, att, opp = synthetic(20000, 0.5, 0.5, 1.0, 0.3, rng)
        f = skill.fit(won, own, enemy, att, opp)
        assert f["battles"] == 20000
        r = f["by_opponent"]["ai_like"]
        assert r["value"] == pytest.approx(0.5, abs=0.06) and r["lo"] < 0.5 < r["hi"]
        assert f["faction_edge"]["EMP-SKV"]["value"] == pytest.approx(-1.0, abs=0.08)
        assert f["attack"]["value"] == pytest.approx(0.3, abs=0.06)
        assert f["overall"]["value"] == pytest.approx(r["value"])

    def test_the_rating_stays_when_the_matchup_mix_moves_the_win_rate(self):
        rng = np.random.default_rng(2)
        rates, ratings = [], []
        for skv in (0.15, 0.85):                                # few / many Skaven opponents
            won, own, enemy, att, opp = synthetic(20000, skv, 0.4, 1.2, 0.0, rng)
            f = skill.fit(won, own, enemy, att, opp)
            rates.append(won.mean())
            ratings.append(f["overall"]["value"])
        assert abs(rates[0] - rates[1]) > 0.12                   # the raw win rate moves a lot
        assert abs(ratings[0] - ratings[1]) < 0.08                # the rating does not

    def test_ratings_per_opponent_their_mean_and_a_wider_interval_for_fewer_battles(self):
        rng = np.random.default_rng(3)
        a = synthetic(8000, 0.5, 1.0, 0.8, 0.0, rng, "nearest")
        b = synthetic(800, 0.5, -0.5, 0.8, 0.0, rng, "ai_like")
        f = skill.fit(*(list(x) + list(y) for x, y in zip(a, b)))
        n, ai = f["by_opponent"]["nearest"], f["by_opponent"]["ai_like"]
        assert n["value"] == pytest.approx(1.0, abs=0.12) and ai["value"] == pytest.approx(-0.5, abs=0.3)
        assert ai["se"] > 2 * n["se"]
        assert f["overall"]["value"] == pytest.approx((n["value"] + ai["value"]) / 2)
        assert f["overall"]["se"] < max(n["se"], ai["se"])

    def test_all_wins_stay_finite_and_void_battles_are_left_out(self):
        f = skill.fit([True] * 20 + [False], [EMP] * 21, [SKV] * 21, [True, False] * 10 + [True], ["hold"] * 21,
                      void=[False] * 20 + [True])
        assert f["battles"] == 20 and 0 < f["overall"]["value"] < 10 and np.isfinite(f["overall"]["se"])
        assert skill.fit([True], [EMP], [SKV], [True], ["x"], void=[True]) == {"battles": 0}

    def test_mirrors_alone_have_no_faction_edge(self):
        f = skill.fit([True, False, True], [EMP, SKV, EMP], [EMP, SKV, EMP], [True, False, True], ["a"] * 3)
        assert f["faction_edge"] == {}


class TestRows:
    def test_the_skill_block_over_points(self):
        rng = np.random.default_rng(4)
        won, own, enemy, att, opp = synthetic(400, 0.5, 0.3, 1.0, 0.0, rng)
        g = matchups.group(won, own, enemy, att, margin=np.where(won, 0.5, -0.5))
        pg = skill.pair_gold(*TestPairGold.battles(200, 1.1, rng))
        res = {"skill": skill.fit(won, own, enemy, att, opp),
               "by_opponent": {"ai_like": dict(g, margin=0.1, pairs=skill.pairs(won, np.arange(400) // 2), pair_gold=pg,
                                               baseline=skill.advantage(won, ~won, np.zeros(400), np.zeros(400),
                                                                        own, enemy))}}
        s = skill.summary(res)
        rows = dict(skill.rows([None, s]))
        assert rows["rating, overall (logit, ± 95%)"][0] == "-"
        assert rows["rating, overall (logit, ± 95%)"][1].startswith(("+", "-")) and "±" in rows["rating, ai_like"][1]
        assert "faction edge EMP-SKV (fitted)" in rows
        assert any(k.startswith("pair score (both/split/neither), ai_like (n 200)") for k in rows)
        gold = [k for k in rows if k.startswith("pair gold (ours - opp's") and k.endswith("ai_like (n 200)")]
        assert gold and rows[gold[0]][0] == "-" and rows[gold[0]][1].startswith("+") and "±" in rows[gold[0]][1]
        assert rows["pair exchange (our destroyed/lost / opp's, same army), ai_like"][1].startswith("x1.")
        weak = [k for k in rows if k.startswith("weak army destroyed/lost net vs opp (trade), ai_like")]
        assert weak and " vs " in rows[weak[0]][1] and "(trade " in rows[weak[0]][1]
        adv = [k for k in rows if k.startswith("vs script win / gold adv, ai_like EMP-SKV")]
        assert adv and rows[adv[0]][1].endswith("/ +0.000")
        m = [k for k in rows if k.startswith("margin, ai_like all / EMP-EMP")]
        assert m and rows[m[0]][1].startswith("+0.100 / ")
        assert skill.text(s)[0].startswith("rating, overall")
