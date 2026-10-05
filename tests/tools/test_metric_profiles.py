"""Metric profiles (tools/nn/train/profiles.py), the numbers of one recorded battle
(tools/nn/battle_metrics.py), the game-vs-sim gap card's comparison (tools/ops/gapcard.py) and the
profiles' wiring into the gate summary, the chain step and the run card. Plain Python."""
import json

import numpy as np
import pytest

from tools.nn import battle_metrics as M
from tools.nn import gamedata, gate
from tools.nn.train import profiles
from tools.ops import card, gapcard, step


# --- profiles ---------------------------------------------------------------------------------------

class TestProfiles:
    def test_full_is_every_profile_and_mandatory_none(self):
        assert profiles.parse("full", profiles.TEST5) == profiles.TEST5
        assert profiles.parse("mandatory", profiles.TEST5) == ()
        assert profiles.parse("", profiles.TEST5) == () and profiles.parse(None, profiles.GAME) == ()

    def test_a_list_keeps_the_known_order_and_takes_aliases(self):
        got = profiles.parse("drills, shooters,transfer", profiles.TEST5, profiles.TEST5_ALIASES)
        assert got == ("behaviour", "transfer", "drills")
        assert profiles.parse("mandatory+fatigue", profiles.GAME) == ("fatigue",)

    def test_an_unknown_name_fails_and_names_the_known(self):
        with pytest.raises(ValueError, match="known: mandatory, behaviour"):
            profiles.parse("drils", profiles.TEST5)

    def test_text_says_full_or_the_mandatory_set_plus_the_chosen(self):
        assert profiles.text(profiles.TEST5, profiles.TEST5) == "full"
        assert profiles.text((), profiles.TEST5) == "mandatory"
        assert profiles.text(("fatigue", "drills"), profiles.TEST5) == "mandatory+fatigue+drills"

    def test_the_adaptive_teachers_need_their_blocks(self):
        assert set(profiles.test5_needs("auto", True)) == {"drills", "transfer"}
        assert profiles.test5_needs(None, False) == {} and set(profiles.test5_needs('{"kiting": 0.5}', False)) == set()


# --- one battle's numbers ---------------------------------------------------------------------------

LORD, SPEAR, ARCH = "wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_crossbowmen"
PASS = {LORD: {"multiplayer_cost": 1000, "caste": "lord", "men": 1},
        SPEAR: {"multiplayer_cost": 400, "caste": "melee_infantry", "men": 100},
        ARCH: {"multiplayer_cost": 600, "caste": "missile_infantry", "men": 100, "missile": {"ammo": 10}}}


def battle(T=5, t=None):
    """Own lord + crossbowmen (side 1) vs a lord + spearmen (side 2), T samples a second apart."""
    keys = (LORD, ARCH, LORD, SPEAR)
    side = np.array([1, 1, 2, 2])
    f = {k: np.zeros((T, 4)) for k in gamedata.FLOAT_FIELDS}
    f.update({k: np.zeros((T, 4), dtype=bool) for k in gamedata.BOOL_FIELDS})
    f["men"][:] = [1, 100, 1, 100]
    f["hp"][:] = 1.0
    f["fat"] = np.zeros((T, 4))
    return gamedata.Battle(run="x", own_ai="", enemy_role="", result={}, t=np.arange(T, dtype=float) if t is None else t,
                           f=f, names=("a", "b", "c", "d"), keys=keys, side=side)


class TestBattleMetrics:
    def test_the_gold_and_the_trade_as_the_gate_counts_them(self):
        b = battle()
        b.f["hp"][-1, 1] = 0.5                      # own crossbowmen: half lost, routing -> 0.5 + 0.5 * 0.5
        b.f["r"][-1, 1] = True
        b.f["s"][-1, 3] = True                      # enemy spearmen shattered: whole
        m = M.measure(b, 2, passports=PASS, chosen=())
        assert m["gold_own"] == pytest.approx(0.75 * 600 / 1600)
        assert m["gold_enemy"] == pytest.approx(400 / 1400)
        assert m["trade"] == pytest.approx((400 - 450) / 1500)
        assert m["win"] == 0.0 and m["length_s"] == 4.0 and m["lord_dead_own"] == 0.0
        assert set(m) == {k for k, g, _, _ in M.ROWS if g == "mandatory"}

    def test_the_window_ends_at_cut_but_the_result_and_length_do_not(self):
        b = battle(T=10)
        b.f["men"][6:, 0] = 0                       # our lord dies at 6 s
        m = M.measure(b, 1, cut_s=4, passports=PASS, chosen=())
        assert m["lord_dead_own"] == 0.0 and m["gold_own"] == 0.0 and m["length_s"] == 9.0 and m["win"] == 1.0
        assert M.measure(b, 1, passports=PASS, chosen=())["lord_dead_own"] == 1.0
        assert M.measure(b, 0, passports=PASS, chosen=())["win"] is None

    def test_routs_lords_and_abilities(self):
        b = battle(T=6)
        b.f["r"][2:4, 3] = True                     # enemy spearmen rout once, rally
        b.f["r"][5, 3] = True                       # ... and again
        b.f["m"][1:5, 0] = True                     # our lord in melee 4 s, loses 0.2 of his health
        b.f["hp"][3:, 0] = 0.8
        m = M.measure(b, 1, abilities=[3.0, 1.0, 9.0], cut_s=5, passports=PASS, chosen=("routs", "lords"))
        assert m["routs_enemy"] == 1.0 and m["routs_own"] == 0.0
        assert m["lord_lost_own"] == pytest.approx(0.2 / 4) and m["lord_lost_enemy"] is None
        assert m["abilities_own"] == 2.0 and m["ability_first_s"] == 1.0

    def test_fatigue_by_time_bin(self):
        t = np.array([0.0, 100.0, 101.0, 200.0, 201.0])
        b = battle(T=5, t=t)
        b.f["fat"][3:, :2] = 3                      # own units tired from 200 s
        b.f["fat"][4, 0] = 5
        m = M.measure(b, 1, passports=PASS, chosen=("fatigue",))
        assert m["tired_own_0-120"] == 0.0 and m["tired_own_120-240"] == 1.0 and m["tired_own_240-360"] is None
        assert m["exhausted_own"] == pytest.approx(1 / 10) and m["tired_enemy_120-240"] == 0.0

    def test_activity_and_missile_time_in_melee(self):
        b = battle(T=5)
        b.f["m"][:2, 1] = True                      # crossbowmen: 2 s melee, 1 firing, 1 moving, 1 still (last: 0 s)
        b.f["fire"][2, 1] = True
        b.f["mv"][3, 1] = True
        m = M.measure(b, 1, passports=PASS, chosen=("activity", "shooters"))
        assert m["missile_melee_own"] == pytest.approx(2 / 4) and m["missile_melee_enemy"] is None
        assert m["melee_own"] == pytest.approx(2 / 8) and m["fire_own"] == pytest.approx(1 / 8)
        assert m["move_own"] == pytest.approx(1 / 8) and m["still_own"] == pytest.approx(4 / 8)

    def test_the_bridge_s_ability_events(self, tmp_path):
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in (
            {"event": "nn_ability", "status": "used", "t": 2500, "u": "own_lord"},
            {"event": "nn_ability", "status": "refused", "t": 3000, "u": "own_lord"},
            {"event": "nn_ability", "status": "used", "t": 4000, "u": "own_hero"})), encoding="utf-8")
        assert M.abilities_used(p, "own_lord") == [(2.5, "own_lord")]
        assert len(M.abilities_used(p)) == 2


# --- the gap card ---------------------------------------------------------------------------------

class TestGapCard:
    def test_beyond_takes_the_larger_of_the_noise_and_the_tolerance(self):
        assert not gapcard.beyond(0.04, None, "trade") and gapcard.beyond(0.06, None, "trade")
        assert not gapcard.beyond(0.10, 0.06, "trade") and gapcard.beyond(0.13, 0.06, "trade")
        assert not gapcard.beyond(20.0, None, "length_s") and not gapcard.beyond(None, 0.1, "trade")

    def test_one_battle_and_pooled_flags(self):
        sims = [{"trade": x, "gold_own": 0.5} for x in (0.10, 0.12, 0.08, 0.10)]
        battles = [{"battle": 1, "role": "attack", "game": {"trade": -0.30, "gold_own": 0.52}, "sims": sims},
                   {"battle": 2, "role": "defend", "game": {"trade": 0.09, "gold_own": None}, "sims": sims}]
        rows = {r["key"]: r for r in gapcard.compare(battles, ())}
        tr = rows["trade"]
        assert tr["cells"][0]["flag"] and not tr["cells"][1]["flag"]
        assert tr["cells"][0]["sim"] == pytest.approx(0.10) and tr["cells"][0]["n"] == 4
        assert tr["pooled"]["game"] == pytest.approx(-0.105) and tr["pooled"]["diff"] == pytest.approx(0.205)
        assert tr["pooled"]["flag"] and tr["pooled"]["k"] == 2
        g = rows["gold_own"]
        assert g["pooled"]["k"] == 1 and not g["pooled"]["flag"]
        assert "routs_own" not in rows and "routs_own" in {r["key"] for r in gapcard.compare(battles, ("routs",))}
        text = "\n".join(gapcard.table(list(rows.values()), battles))
        assert "| b1 att game / sim |" in text and "-0.300 / +0.100 !" in text
        assert "differ beyond noise (pooled): trade" in text

    def test_reprint_from_the_saved_card(self, tmp_path, capsys):
        doc = {"gate": str(tmp_path), "checkpoint": "build/x/m20.pt", "copies": 2, "battles": [
            {"battle": 1, "role": "attack", "game": {"win": 0.0, "trade": -0.3}, "sims": [{"win": 1.0, "trade": 0.2}] * 2}]}
        (tmp_path / "gapcard.json").write_text(json.dumps(doc), encoding="utf-8")
        assert gapcard.main([str(tmp_path), "--from-json", "--profile", "mandatory"]) == 0
        out = capsys.readouterr().out
        assert "x/m20.pt vs ai_like" in out and "win (1 / 0)" in out and "rout" not in out


# --- wiring ---------------------------------------------------------------------------------------

class TestWiring:
    def test_the_step_passes_the_profile_to_test5(self, tmp_path, monkeypatch):
        (tmp_path / "test5" / "prev").mkdir(parents=True)
        (tmp_path / "test5" / "prev" / "m25.pt").write_bytes(b"x")
        monkeypatch.setattr(step, "TEST5", tmp_path / "test5")
        monkeypatch.setattr(step, "RUNS", tmp_path / "runs")
        lines, info = step.build("n4", "prev", {"test5": {"profile": "drills"}, "options": {}})
        assert "--profile drills --" in lines[1] or "--profile drills " in lines[1].split(" -- ")[0]
        assert info["profile"] == "drills"
        lines, info = step.build("n4", "prev", {"options": {}}, profile="full")
        assert "--profile" not in lines[1] and info["profile"] == "full"
        with pytest.raises(ValueError):
            step.build("n4", "prev", {"options": {}}, profile="nope")

    def test_the_card_says_what_was_computed(self):
        pts = [("0", {"profile": "mandatory+drills"}), ("5", {"profile": "mandatory+drills"})]
        assert card.computed(pts) == "computed: mandatory+drills (tools/nn/train/profiles.py)"
        assert card.computed([("0", {}), ("5", {"profile": "mandatory"})]) == "computed: min 0 full; min 5 mandatory"

    def test_the_gate_summary_keeps_only_the_chosen_profiles(self, monkeypatch):
        seen = {}

        def fake(run, winner, chosen=M.GROUPS, passports=None):
            seen["chosen"] = list(chosen)
            return {"lord_dead_own": 1.0, "routs_own": 0.5}
        monkeypatch.setattr(M, "game_battle", fake)
        rows = [{"battle": 1, "metrics": fake(None, 1, ("routs",))}, {"battle": 2, "metrics": None}]
        assert gate.metric_mean(rows, "lord_dead_own") == 1.0
        assert gate.metrics_summary(rows, ("routs",)) == {"routs_own": 0.5, "routs_enemy": None}
        assert gate.metrics_summary(rows, ("liveliness",)) == {}
        lines = gate.metric_lines({"battles": rows, "lord_dead_own": 1.0, "metrics": {"routs_own": 0.5},
                                   "profile": "mandatory+routs"})
        assert lines[0] == "own lord dead: 1 - (mean 1.00)" and "rout onsets per unit, own: 0.50 - | 0.50" in lines[-1]


class TestRunFarFromTheFight:
    def test_the_orders_in_force_from_the_bridge_s_given_orders(self, tmp_path):
        p = tmp_path / "events.jsonl"
        evs = [{"event": "nn_orders", "t": 300, "orders": [{"u": "a", "k": "move", "run": True, "status": "given"},
                                                         {"u": "b", "k": "move", "run": False, "status": "skipped"}]},
               {"event": "nn_orders", "t": 2100, "orders": [{"u": "a", "k": "attack", "status": "given"},
                                                          {"u": "b", "k": "withdraw", "run": False, "status": "given"}]}]
        p.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in evs), encoding="utf-8")
        o = M.orders_in_force(p, ("a", "b"), np.array([0.0, 1.0, 2.0, 3.0]))
        assert o["move"][:, 0].tolist() == [False, True, True, False] and o["run"][:, 0].tolist() == [False, True, True, False]
        assert o["move"][:, 1].tolist() == [False, False, False, True] and not o["run"][:, 1].any()
        assert M.orders_in_force(tmp_path / "none.jsonl", ("a",), np.zeros(2)) is None

    def test_only_far_and_before_contact_counts(self):
        b = battle(T=6)
        b.f["x"][:] = [0.0, 0.0, 400.0, 400.0]               # the enemy 400 m away
        b.f["x"][3:, 2] = 100.0                              # its lord comes within 150 m at 3 s
        b.f["m"][5, 3] = True                                # first melee at 5 s
        move = np.zeros((6, 4), dtype=bool)
        move[:, :2] = True
        run = np.zeros((6, 4), dtype=bool)
        run[:, 1] = True                                     # the crossbowmen run, the lord walks
        m = M.measure(b, 1, passports=PASS, chosen=("fatigue",), orders={"move": move, "run": run})
        assert m["run_far_own"] == pytest.approx(0.5)       # 0-2 s: both far; then the enemy lord is near
        assert M.measure(b, 1, passports=PASS, chosen=("fatigue",))["run_far_own"] is None
        far = M.far_from_fight(b, np.ones((6, 4), dtype=bool))
        assert far[:3, 0].all() and not far[3:, 0].any()
