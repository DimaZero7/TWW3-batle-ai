"""The melee probe's plans, battle files and measures (tools/nn/charge_probe.py) and the lord swarm's damage
plan (tools/nn/lord_swarm.py). The simulator twin needs torch (tested in the container by running it)."""
import json
import math

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn import lord_swarm as ls


def test_plans_have_few_battles_and_lanes_on_both_sides():
    assert len(cp.battles("charge")) == 8 and len(cp.battles("hit")) == 2
    for plan in cp.PLANS:
        for i in range(1, len(cp.battles(plan)) + 1):
            config, model_s, arena = cp.run_config(plan, i)
            assert 2 <= len(config["lanes"]) <= 5 and model_s < 400
            names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
            for lane in config["lanes"]:
                assert lane["attacker"] in names and lane["target"] in names
                assert lane["attacker"].split("_")[0] != lane["target"].split("_")[0]   # two sides
            # one lord a side, parked unless a lane uses him
            used = {n for l in config["lanes"] for n in (l["attacker"], l["target"], (l.get("lord") or {}).get("name"))}
            used |= {f["name"] for l in config["lanes"] for f in l.get("rally_friends") or []}
            assert {p["name"] for p in config["park"]} == {"own_lord", "enemy_lord"} - used
            places = [(l["x"], l["z"]) for l in config["lanes"]]
            assert all(math.dist(p, q) >= 200 for k, p in enumerate(places) for q in places[k + 1:])


def test_the_charge_set_puts_every_order_in_every_lane_place():
    a = cp.battles("charge")[:4]
    for k in range(4):
        assert {b[k]["mode"] for b in a} == {"attack_run", "attack_walk", "move_run", "recharge"}
    hit = cp.battles("hit")
    assert [b[-1]["lord"]["ability"] for b in hit] == [cp.SYG, ""]


def test_the_fresh_plan_puts_every_order_in_every_lane_with_a_second_target():
    b = cp.battles("fresh")
    assert len(b) == 2 and all(sorted(l["after"]["kind"] for l in x) == sorted(cp.FRESH) for x in b)
    config, _, arena = cp.run_config("fresh", 1)
    names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
    for lane in config["lanes"]:
        assert lane["target2"] in names and lane["target2"] != lane["target"]
        assert lane["t2_dx"] == 34 and lane["after"]["at_s"] == 10 and lane["target_mode"] == "both"


def test_the_meleeorders_plan_gives_hold_moves_and_a_control_10_s_after_contact():
    b = cp.battles("meleeorders")
    assert len(b) == 2 and all(len(x) == 5 for x in b)
    orders = [(l["after"]["kind"], l["after"]["dx"], l["after"]["dz"]) for x in b for l in x]
    assert ("halt", 0, 0) in orders and ("none", 0, 0) in orders
    assert {o for o in orders if o[0] == "move_near"} == {("move_near", 0, -5), ("move_near", 0, -15),
                                                           ("move_near", 0, -40), ("move_near", 5, 0), ("move_near", 15, 0)}
    assert all(l["after"]["at_s"] == 10 and not l["after"]["walk"] and l["target_mode"] == "both" for x in b for l in x)


def test_the_damaged_plan_damages_the_clanrats_two_on_one_and_a_shooting_lane():
    b = cp.battles("damaged")
    assert len(b) == 2 and all(len(x) == 5 for x in b)
    methods = [((l.get("damage") or {}).get("t") or {}).get("method") for x in b for l in x if l["mode"] != "shoot"]
    assert methods.count(None) == 3 and methods.count("reduce") == 3 and methods.count("kill") == 2
    shoot = [l for l in b[0] if l["mode"] == "shoot"]
    assert sorted(l["t2_gap_m"] for l in shoot) == [15, 40] and all(l["target_mode"] == "hold" for l in shoot)
    twos = [l for l in b[1] if l.get("t2_mode") == "attack"]
    assert len(twos) == 3 and all(l["target2"] == "clanrat" for l in twos)
    config, _, arena = cp.run_config("damaged", 1)
    names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
    assert all(l.get("target2", l["target"]) in names for l in config["lanes"])


def test_the_reform_plan_records_the_soldiers_all_fight():
    (b,) = cp.battles("reform")
    assert len(b) == 5 and all(l["fight_s"] == 240 and l["gap_m"] == 3 for l in b)
    config, model_s, _ = cp.run_config("reform", 1)
    assert config["men_after_s"] >= 250 and model_s < 400
    assert [l.get("t2_mode") for l in config["lanes"]].count("attack") == 1
    assert sum(1 for l in config["lanes"] if l.get("damage")) == 1 and "spear" in config["lanes"][2]["attacker"]


def test_the_reform2_plan_separates_own_losses_blow_and_depth():
    (b,) = cp.battles("reform2")
    assert len(b) == 5 and all(l["fight_s"] == 240 for l in b)
    assert b[0]["damage"]["a"]["method"] == "kill" and b[1]["attacker"] == "gs"
    assert [l.get("t_w") for l in b[2:4]] == [15, 50]
    config, model_s, _ = cp.run_config("reform2", 1)
    assert config["men_after_s"] >= 250 and model_s < 400
    assert config["lanes"][2]["t_depth"] > config["lanes"][4]["t_depth"] > config["lanes"][3]["t_depth"]


def test_the_defender_plan_gives_the_swordsmen_four_orders_1_s_after_contact():
    b = cp.battles("defender")
    assert len(b) == 2 and all(sorted(l["after"]["kind"] for l in x) == sorted(cp.DEFENDER) for x in b)
    assert [l["after"]["kind"] for l in b[0]][:2] != [l["after"]["kind"] for l in b[1]][:2]      # rotated
    for i in (1, 2):
        config, model_s, arena = cp.run_config("defender", i)
        assert config["men_after_s"] >= 90 and model_s < 400
        xs = [l["x"] for l in config["lanes"]]
        assert min(b_ - a for a, b_ in zip(xs, xs[1:])) == cp.DEFENDER_LANE_DX
        for lane in config["lanes"]:
            assert lane["attacker"].startswith("own_swords") and lane["target_mode"] == "both" and lane["gap_m"] == 3
            assert lane["after"]["at_s"] == 1 and not lane["after"]["walk"]
            if lane["after"]["kind"] == "attack_t2":
                assert abs(lane["t2_dx"]) == 180 and abs(lane["x"] + lane["t2_dx"]) < abs(lane["x"])   # inward
                others = [l["x"] for l in config["lanes"] if l is not lane]
                assert min(abs(lane["x"] + lane["t2_dx"] - o) for o in others) >= 120
            elif lane["after"]["kind"] == "move_near":
                assert lane["after"]["dx"] == 30
            else:
                assert "target2" not in lane


def test_the_wave_plan_varies_the_start_gap_and_the_clanrats_pace():
    b = cp.battles("wave")
    assert len(b) == 2 and all(len(x) == 5 for x in b)
    cells = sorted((l["gap_m"], l["target_mode"]) for l in b[0])
    assert cells == [(1, "both_walk"), (2.5, "both_walk"), (3, "both"), (3, "both_walk"), (30, "both")]
    assert sorted((l["gap_m"], l["target_mode"]) for l in b[1]) == cells and b[0][0] != b[1][0]   # rotated
    assert all(l["mode"] == "attack_walk" and l["fight_s"] == 90 and not l["answer"] for x in b for l in x)
    config, model_s, _ = cp.run_config("wave", 1)
    assert config["men_after_s"] >= 90 and model_s < 400


def test_the_routmob_plan_keeps_the_targets_morale_and_samples_the_mob_after_the_rout():
    b = cp.battles("routmob")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    for x in b:
        assert all(l["t_morale"] and l["mode"] == "attack_run" and l["target_mode"] == "stand" for l in x)
        assert sorted((l["attacker"], l["target"], l["at_rout"]) for l in x) == [
            ("spear", "slave", "halt"), ("spear", "slave", "none"), ("swords", "clanrat", "halt"), ("swords", "clanrat", "none")]
        assert all(l["after_rout_s"] == cp.ROUTMOB_END_S and l["max_s"] >= l["after_rout_s"] + 60 for l in x)
    assert b[1][0] == b[0][2]                                                # the lanes rotated
    config, model_s, _ = cp.run_config("routmob", 1)
    assert config["men_after_rout_s"] == cp.ROUTMOB_MEN_S and model_s < 400


def test_the_routmob2_plan_damages_the_slaves_forces_the_rout_and_adds_shooters_or_neighbours():
    b = cp.battles("routmob2")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    for x in b:
        assert all(l["t_morale"] and l["damage"] == {"t": {"method": "kill", "share": 0.5}} and l["rout_at_s"] == 30 for l in x)
        kinds = {l["kind"]: l for l in x}
        assert set(kinds) == {"chase", "fire", "neighbours", "control"}
        assert kinds["chase"]["at_rout"] == "none" and all(kinds[k]["at_rout"] == "halt" for k in ("fire", "neighbours", "control"))
        assert kinds["fire"]["extras"] == [{"short": "xbow", "dx": 0, "dz": 80, "fire": True}]
        assert [(e["dx"], e["dz"]) for e in kinds["neighbours"]["extras"]] == [(-25, -20), (25, -20)]
        assert "extras" not in kinds["control"] and "extras" not in kinds["chase"]
    config, model_s, arena = cp.run_config("routmob2", 1)
    names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
    for lane in config["lanes"]:
        for e in lane.get("extras") or []:
            assert e["name"] in names and e["name"].startswith("own_")               # the attacker's (Empire) side
    assert config["men_after_rout_s"] == cp.ROUTMOB_MEN_S and model_s < 400


def test_the_rallysecure_plan_routs_the_slaves_sends_the_attacker_away_and_brings_friends_at_the_rally():
    b = cp.battles("rallysecure")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    for x in b:
        kinds = {l["kind"]: l for l in x}
        assert set(kinds) == {"alone", "neighbours", "control", "lord"}
        assert all(l["t_morale"] and l["damage"] == {"t": {"method": "kill", "share": 0.5}} for l in x)
        assert all(l["target_mode"] == "hold" and not l["answer"] and "rout_at_s" not in l for l in x)
        for k in ("alone", "neighbours", "lord"):
            l = kinds[k]
            assert l["mode"] == "attack_run" and l["gap_m"] == 40 and l["at_rout"] == "away"
            assert l["away_dz"] == cp.RALLYSECURE_AWAY_DZ and l["after_rally_s"] == cp.RALLYSECURE_AFTER_S
        assert kinds["control"]["mode"] == "hold" and kinds["control"]["gap_m"] == 300
        assert "at_rout" not in kinds["control"] and "rally_friends" not in kinds["alone"]
        assert [(f["short"], f["dx"], f["dz"]) for f in kinds["neighbours"]["rally_friends"]] == [
            ("clanrat", -35, 0), ("clanrat", 35, 0)]
        assert [f["short"] for f in kinds["lord"]["rally_friends"]] == ["warlord"]
    assert b[1][0] == b[0][2]                                                # the lanes rotated
    for i in (1, 2):
        config, model_s, arena = cp.run_config("rallysecure", i)
        assert model_s < 400
        xs = sorted(l["x"] for l in config["lanes"])
        assert all(q - p >= 250 for p, q in zip(xs, xs[1:])) and max(abs(v) for v in xs) <= 480
        names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]} | {"enemy_lord"}
        lanes = {l["kind"]: l for l in config["lanes"]}
        for f in lanes["neighbours"]["rally_friends"] + lanes["lord"]["rally_friends"]:
            assert f["name"] in names and f["name"].startswith("enemy_") and f["pz"] == cp.RALLYSECURE_PARK_Z
        assert lanes["lord"]["rally_friends"][0]["name"] == "enemy_lord"
        assert "enemy_lord" not in {p["name"] for p in config["park"]}           # placed by the lane, not parked
        # parked friends: far from every lane's start (the attackers go to z = away_dz, the targets run to -z)
        for f in lanes["neighbours"]["rally_friends"]:
            assert all(math.hypot(f["px"] - l["x"], f["pz"] - l["z"]) >= 500 for l in config["lanes"])


def test_the_wavemiss_plan_charges_each_shooter_held_and_answering():
    b = cp.battles("wavemiss")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    cells = sorted((l["attacker"], l["target"], l["target_mode"]) for x in b for l in x)
    assert cells == sorted((a, t, m) for a, t in cp.WAVEMISS for m in ("hold", "stand"))
    assert all(l["mode"] == "attack_run" and l["gap_m"] == 30 and l["fight_s"] == 90 for x in b for l in x)
    assert all(l["answer"] == (l["target_mode"] == "stand") for x in b for l in x)
    assert b[0][0]["target"] != b[1][0]["target"]                         # the lanes rotated
    config, model_s, _ = cp.run_config("wavemiss", 2)
    assert config["men_after_s"] >= 90 and model_s < 400
    assert all(cp.FACTION[cp.UNITS[l["attacker"]][0]] != cp.FACTION[cp.UNITS[l["target"]][0]] for x in b for l in x)


def test_the_wavemiss2_plan_has_each_shooter_answering_and_leaving():
    b = cp.battles("wavemiss2")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    shooters = {t for _, t in cp.WAVEMISS}
    stand = sorted((l["attacker"], l["target"]) for x in b for l in x if l["target_mode"] == "stand")
    move = sorted((l["target"], l["attacker"]) for x in b for l in x if l["target_mode"] == "both")
    assert stand == move == sorted(cp.WAVEMISS)
    for x in b:
        for l in x:
            if l["target_mode"] == "both":       # the shooter leaves: unordered, then a run 40 m back 3 s after contact
                assert l["attacker"] in shooters and l["mode"] == "hold" and not l["answer"]
                assert l["after"] == {"at_s": 3, "kind": "move_near", "dx": 0, "dz": 40, "walk": False}
            else:
                assert l["target"] in shooters and l["mode"] == "attack_run" and l["answer"]
    assert b[0][0]["attacker"] != b[1][0]["attacker"]                     # the lanes rotated
    config, model_s, _ = cp.run_config("wavemiss2", 1)
    assert config["men_after_s"] >= 90 and model_s < 400


def test_the_dmgmelee_plan_fights_whole_and_cut_units_one_on_one_on_a_far_grid():
    b = cp.battles("dmgmelee")
    assert len(b) == 2 and all(len(x) == 5 for x in b)
    cells = sorted((l["attacker"], l["target"], l["kind"], json.dumps(l.get("damage"), sort_keys=True)) for l in b[0])
    cut = json.dumps({"method": "kill", "share": 0.7})
    assert cells == sorted([("gs", "svsh", "full", "null"), ("gs", "svsh", "t30", '{"t": %s}' % cut),
                            ("gs", "svsh", "a30", '{"a": %s}' % cut), ("spearsh", "clanrat", "full", "null"),
                            ("general", "svsh", "full", "null")])
    assert sorted(cp.cell(l) for l in b[0]) == sorted(cp.cell(l) for l in b[1]) and len({cp.cell(l) for l in b[0]}) == 5
    assert all(l["mode"] == "attack_walk" and l["target_mode"] == "both" and l["gap_m"] == 3 and l["fight_s"] == 240
               and l["morale_rows"] and "t_morale" not in l for x in b for l in x)
    assert cp.UNITS["svsh"][0] == "wh2_main_skv_inf_stormvermin_1" and cp.UNITS["spearsh"][0].endswith("spearmen_1")
    assert b[0][0]["kind"] != b[1][0]["kind"]                               # the lanes rotated
    for i in (1, 2):
        config, model_s, arena = cp.run_config("dmgmelee", i)
        assert config["men_after_s"] >= 250 and config["men_ms"] <= 1000 and model_s < 400
        places = [(l["x"], l["z"]) for l in config["lanes"]]
        assert sorted(places) == sorted(cp.DMGMELEE_PLACES)
        assert all(math.dist(p, q) > 250 for k, p in enumerate(places) for q in places[k + 1:])
        assert all(abs(x) <= 480 and abs(z) <= 480 for x, z in places)
        names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]} | {"own_lord"}
        assert all(l["attacker"] in names and l["target"] in names for l in config["lanes"])
        assert {p["name"] for p in config["park"]} == {"enemy_lord"}           # the General fights in a lane
        # the parked Warlord is far from every lane
        assert all(math.hypot(700 - x, -400 - z) > 250 for x, z in places)


def test_battle_file_is_written(tmp_path):
    path = cp.write_scenario("hit", 1, tmp_path / "x.xml")
    xml = path.read_text(encoding="utf-8")
    assert xml.count("<unit ") == 2 + 10 and "wh_main_emp_inf_greatswords" in xml


def test_depth_from_the_database_spacing():
    assert cp.depth("wh_main_emp_inf_swordsmen", 120) == 6 * 1.6      # 20 files of 1.48 m: 6 ranks of 1.6 m
    assert cp.depth("wh_main_emp_cha_general_0", 1) == 0.0


def test_near_counts():
    a = [0, 0, 100, 0]                      # decimetres: men at (0, 0) and (10, 0) m
    b = [0, 20, 500, 500]                   # (0, 2) and (50, 50)
    assert cp.near_counts(a, b, (1.5, 2.5)) == [0, 1]


def fake_lane(mode="attack_run", contact=20.0, rate_t=10.0, rate_a=5.0, recharge=False):
    spec = dict(cp.layout(cp.battles("charge")[0])[0][0], mode=mode)
    samples = []
    for k in range(0, 160):
        t = k * 0.5
        after = max(0.0, t - contact)
        z_a = max(20.0, 100 - 4 * t)
        samples.append((t, {"x": 0, "z": z_a, "hp": 8000 - rate_a * after, "men": 120, "m": t >= contact},
                        {"x": 0, "z": 0, "hp": 9600 - rate_t * after, "men": 160, "m": t >= contact}, None))
    return {"run": "r", "spec": spec, "samples": samples, "men": [(contact + 1, [0, 0], [0, 10])],
            "contacts": {1: contact}, "end": None, "abilities": [], "phases": []}


def test_measure_windows_and_steady_rates():
    m = cp.measure(fake_lane())
    assert m["contact_s"] == 20.0 and m["cell"] == "swords>clanrat attack_run/stand"
    assert abs(m["tg_hp_0_5"] - 50) < 1e-6 and abs(m["tg_hp_5_15"] - 100) < 1e-6
    assert abs(m["tg_hp_steady"] - 10) < 1e-6 and abs(m["a_hp_steady"] - 5) < 1e-6
    assert abs(m["speed_last30"] - 4.0) < 1e-6          # 2 m every 0.5 s
    assert m["a_near_0_5"] == [1.0, 1.0, 1.0]
    rows = cp.summary([m, cp.measure(fake_lane(rate_t=20))])
    assert rows[0]["n"] == 2 and abs(rows[0]["tg_hp_steady"][0] - 15) < 1e-6


def test_lord_swarm_damage_plan():
    config, model_s = ls.run_config(4, plan="damage")
    assert [t["name"] for t in config["trials"]] == ["sw1", "gs1", "gs1", "sw1"] * 2 and config["plan"] == "damage"
    keys = {u["slot"]: u["key"] for u in ls.arena("damage")["sides"]["own"]["units"]}
    assert keys["spear_1"] == ls.SWORDS and keys["ap_1"] == ls.GREATSWORDS
    assert len(config["lanes"][0]["spears"]) == 2 and model_s < 900


def test_lord_swarm_blows():
    # the lord loses 30 HP a tick now and then; the attacker loses 120 HP in two ticks 0.2 s apart (one blow)
    samples = []
    lord, att = 4068.0, 8000.0
    for k in range(50):
        if k % 10 == 5:
            lord -= 30
        if k in (20, 21):
            att -= 60
        samples.append({"t": 200 * k, "lord": {"hp": lord, "m": True}, "att": [{"hp": att, "men": 120 - (k > 21), "m": True}]})
    tr = {"lane": "warlord", "name": "gs1", "attackers": [{"name": "own_ap_1"}], "samples": samples}
    b = ls.blows(tr)
    assert b["on_lord"] == [30.0] * 5 and b["lord_blows"] == [120.0] and b["att_men"] == 1
    (row,) = ls.blows_table([tr])
    assert row["share_26_31"] == 1.0 and row["att_hp_per_lord_blow"] == 120.0
    json.dumps(row)
    assert np.isfinite(row["lord_hp_per_s"])
