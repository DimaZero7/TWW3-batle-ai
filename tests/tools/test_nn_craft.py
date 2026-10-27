"""Craft by unit class (tools/nn/train/craft.py): the attribution of the enemy's loss to our classes (health exact,
the loss beyond health by each attacker's share of the health it struck off that enemy), the summary, the card's
rows; with torch the classes of the state's numbers and the proxy's battles (the classes add up to the exam's trade)."""
import numpy as np
import pytest

from tools.nn.train import craft
from tools.ops import card

L, ME, SP, MI, FA, ST = (craft.CLASSES.index(c) for c in ("lord", "melee", "spear", "missile", "fast", "stalk"))


def one_battle():
    """One battle, slots 0-2 ours (lord, spears, shooters), 3-4 the enemy's: enemy 3 (cost 600) lost 300 of health
    (the spears struck 200, the shooters 100) and then routed off the map (lost whole: 300 more); enemy 4 (cost 400)
    shattered untouched (the army's collapse: 400 nobody struck). Ours lost: the lord 0, the spears 250, the
    shooters 50."""
    gold_out = np.array([[0.0, 200.0, 100.0, 0.0, 0.0]])
    pair = np.zeros((1, 5, 5))
    pair[0, 1, 3], pair[0, 2, 3] = 200.0, 100.0
    lost = np.array([[0.0, 250.0, 50.0, 600.0, 400.0]])
    hp_part = np.array([[0.0, 250.0, 50.0, 300.0, 0.0]])
    enemy = np.array([[False, False, False, True, True]])
    mine = ~enemy
    cls = np.array([[L, SP, MI, ME, ME]])
    cost = np.array([[500.0, 500.0, 300.0, 600.0, 400.0]])
    ammo = np.array([[0.0, 0.0, 0.75, 0.0, 0.0]])
    missile = np.array([[False, False, True, False, False]])
    return craft.attribute(gold_out, pair, lost, hp_part, enemy, mine, cls, cost, ammo, missile)


class TestAttribute:
    def test_health_exact_the_rout_by_the_share_of_health_struck_the_rest_unattributed(self):
        a = one_battle()
        assert a["kill_hp"][0, SP] == 200 and a["kill_hp"][0, MI] == 100 and a["kill_hp"][0, L] == 0
        # enemy 3's 300 beyond health: 2/3 to the spears, 1/3 to the shooters
        assert a["kill_rout"][0, SP] == pytest.approx(200) and a["kill_rout"][0, MI] == pytest.approx(100)
        assert a["unattributed"][0] == pytest.approx(400)                   # enemy 4: nobody struck it
        assert a["destroyed"][0] == 1000
        assert a["kill_hp"].sum() + a["kill_rout"].sum() + a["unattributed"].sum() == pytest.approx(1000)
        assert a["lost"][0, SP] == 250 and a["lost"][0, MI] == 50 and a["lost"][0, ME] == 0   # enemy losses not ours
        assert a["cost"][0, L] == 500 and a["units"][0].tolist() == [1, 0, 1, 1, 0, 0]
        assert a["ammo_used_sum"][0, MI] == 0.75 and a["ammo_units"][0, MI] == 1

    def test_summary_trade_shares_and_unfielded_classes_left_out(self):
        b = one_battle()
        b.update(budget=np.array([1000.0]),
                 stand_s=np.array([[100.0, 0, 200, 100, 0, 0]]), melee_s=np.array([[25.0, 0, 100, 10, 0, 0]]),
                 field_s=np.array([[100.0, 0, 250, 100, 0, 0]]), rout_s=np.array([[0.0, 0, 50, 0, 0, 0]]),
                 ready_s=np.array([[0.0, 0, 0, 80, 0, 0]]), fire_s=np.array([[0.0, 0, 0, 60, 0, 0]]))
        s = craft.summary(b)
        assert s["battles"] == 1 and set(s["classes"]) == {"lord", "spear", "missile"}
        sp, mi = s["classes"]["spear"], s["classes"]["missile"]
        assert sp["killed"] == 400 and sp["lost"] == 250 and sp["trade"]["value"] == pytest.approx(0.15)
        assert sp["return"] == pytest.approx(0.8) and sp["melee_share"] == 0.5 and sp["rout_share"] == 0.2
        assert mi["trade"]["value"] == pytest.approx(0.15) and mi["fire_share"] == 0.75 and mi["fire_s"] == 60
        assert mi["ammo_used"] == 0.75 and "fire_share" not in sp
        assert s["classes"]["lord"]["trade"]["value"] == 0
        # the classes' trades plus the unattributed share = the battle's trade (1000 destroyed - 300 lost) / 1000
        total = sum(c["trade"]["value"] for c in s["classes"].values()) + s["unattributed"]
        assert total == pytest.approx(0.7)
        assert craft.lines(s)[0].startswith("craft by class") and len(craft.lines(s)) == 2 + 3 + 1
        assert craft.lines(None) == [] and craft.summary(b, np.array([False]))["classes"] == {}

    def test_the_card_rows_only_where_the_proxy_has_the_craft(self):
        b = one_battle()
        b.update(budget=np.array([1000.0]), **{k: np.ones((1, len(craft.CLASSES))) for k in craft.TIMES})
        exam = {"battles": 12, "pairs": [1, 1, 1], "wins12": 6.0, "trade": {"value": 0.1, "se": 0.02},
                "craft": craft.summary(b)}
        rows = {r[0]: r for r in card.rows({"skill": {"exam": exam}})}
        assert rows["craft_spear"][2] == pytest.approx(0.15) and "craft_fast" not in rows
        text = card.fmt(rows["craft_spear"][2], rows["craft_spear"][3], rows["craft_spear"][4])
        assert text.startswith("+0.150") and "killed 400 / lost 250 of 500" in text
        exam.pop("craft")
        assert not any(k.startswith("craft_") for k in (r[0] for r in card.rows({"skill": {"exam": exam}})))


# --- torch ---------------------------------------------------------------------------------------------------

def units(**cols):
    torch = pytest.importorskip("torch")
    base = {"lord": [False] * 7, "men0": [120.0] * 7, "stalk": [False] * 7, "range": [0.0] * 7, "ammo0": [0.0] * 7,
            "run": [3.0] * 7, "large": [False] * 7, "bonus_v_large": [0.0] * 7, "side": [1] * 7}
    base.update(cols)
    return {k: torch.tensor([v]) for k, v in base.items()}


class TestClasses:
    def test_the_first_match_wins(self):
        pytest.importorskip("torch")
        u = units(lord=[True, False, False, False, False, False, False],
                  men0=[1.0, 1.0, 120, 60, 80, 90, 160],                                   # 1: a hero (one man)
                  stalk=[False, False, False, True, False, False, False],
                  range=[0.0, 0, 0, 120, 150, 0, 0], ammo0=[0.0, 0, 0, 600, 900, 0, 0],
                  run=[4.0, 4, 3, 5.4, 3, 9.5, 3], bonus_v_large=[0.0, 0, 15, 0, 0, 0, 0])
        assert craft.classes(u)[0].tolist() == [L, L, SP, ST, MI, FA, ME]

    def test_the_battles_classes_add_up_to_their_trade(self):
        torch = pytest.importorskip("torch")
        from tools.nn.model import config, policy
        from tools.nn.train import evaluate
        cfg = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
        torch.manual_seed(0)
        actor = policy.Actor(cfg).eval()
        seeds = [1000800042] * 2 + [1000800127] * 2
        torch.manual_seed(1)
        res = evaluate.play(actor, opponents=("nearest",), generated=4, max_units=6, limit_s=240.0, paired=False,
                            seeds=seeds, armies="v2", defend=True, measure={"craft"})     # the script attacks us
        c, b = res["by_opponent"]["nearest"]["craft"], res["battles"]["nearest"]
        assert c["battles"] == 4 and "lord" in c["classes"]
        total = sum(x["trade"]["value"] for x in c["classes"].values()) + c["unattributed"]
        assert total == pytest.approx(np.mean(b["trade"]), abs=2e-3)
        lost = sum(x["lost"] for x in c["classes"].values())
        assert lost == pytest.approx(np.mean(b["lost"]), rel=1e-2, abs=1.0) and lost > 0
        assert sum(x["killed"] for x in c["classes"].values()) > 0                    # someone struck in 4 minutes
        assert res["by_opponent"]["nearest"]["behaviour"] == {}                       # only what was named
        torch.manual_seed(1)                                                          # the same draws: the same battles
        bare = evaluate.play(actor, opponents=("nearest",), generated=4, max_units=6, limit_s=240.0, paired=False,
                             seeds=seeds, armies="v2", defend=True, measure=())
        assert "craft" not in bare["by_opponent"]["nearest"] and bare["battles"]["nearest"]["trade"] == b["trade"]
