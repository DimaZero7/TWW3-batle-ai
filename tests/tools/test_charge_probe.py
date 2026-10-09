"""The melee probe's plans, battle files and measures (tools/nn/charge_probe.py) and the lord swarm's damage
plan (tools/nn/lord_swarm.py). The simulator twin needs torch (tested in the container by running it)."""
import json
import math

import numpy as np
import pytest

from tools.nn import charge_probe as cp
from tools.nn import lord_swarm as ls


def test_plans_have_few_battles_and_lanes_on_both_sides():
    assert len(cp.battles("charge")) == 8 and len(cp.battles("hit")) == 2
    for plan in cp.PLANS:
        for i in range(1, len(cp.battles(plan)) + 1):
            config, model_s, arena = cp.run_config(plan, i)
            # (reengage: its tired pair runs up to 540 s before its fight; retarget2 / 3: 10 / 7 short lanes)
            assert 2 <= len(config["lanes"]) <= {"retarget2": 10, "retarget3": 7}.get(plan, 5)
            assert model_s < (720 if plan == "reengage" else 400)
            assert all(len(side["units"]) <= 20 for side in arena["sides"].values())
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


def test_the_reengage_plan_parts_and_rejoins_two_pairs_beside_a_fresh_and_a_tired_control():
    b = cp.battles("reengage")
    assert len(b) == 2 and all(len(x) == 4 for x in b)
    for x in b:
        kinds = sorted((l["attacker"], l["target"], l["kind"]) for l in x)
        assert kinds == sorted([("swords", "clanrat", "reengage"), ("gs", "svsh", "reengage"),
                                ("swords", "clanrat", "control"), ("swords", "clanrat", "tired")])
        assert all(l["gap_m"] == 30 and l["target_mode"] == "both" and not l["answer"] for l in x)
        for l in x:
            if l["kind"] == "reengage":
                assert l["mode"] == "reengage" and l["fight_s"] == 150
                assert (l["part_after_s"], l["part_m"], l["part_s"], l["after2_s"]) == (150, 30, 15, 60)
                assert l["max_s"] >= 30 / 4 + 150 + 15 + 60 / 4 + 60
            elif l["kind"] == "tired":
                assert l["mode"] == "tire" and l["fight_s"] == 60 and l["tire_until"] == "threshold_very_tired"
                assert l["max_s"] >= l["tire_max_s"] + l["ready_max_s"] + 60
                # the database's running cost (+4 a tick, 10 ticks a second) reaches very tired well inside the cap
                rules = json.loads((cp.project.ROOT / "config" / "nn" / "game_rules.json").read_text(encoding="utf-8"))
                fat = rules.get("fatigue", rules)
                assert fat["threshold_very_tired"] / (fat["running"] * 10) < l["tire_max_s"]
            else:
                assert l["mode"] == "attack_run" and l["fight_s"] == 60 and "after2_s" not in l
    assert [l["kind"] for l in b[1]] == [l["kind"] for l in cp.rotate(b[0], 2)] and b[0][0]["kind"] != b[1][0]["kind"]
    for i in (1, 2):
        config, model_s, arena = cp.run_config("reengage", i)
        assert config["men_after_s"] >= 250 and config["men_ms"] == 1000 and config["tick_ms"] == 500
        assert config["men_near_m"] >= 100 and model_s < 720
        xs = sorted(l["x"] for l in config["lanes"])
        assert all(q - p >= 250 for p, q in zip(xs, xs[1:])) and max(abs(v) for v in xs) <= 480
        # every unit moves along its own lane (z): the tired pair's shuttle stays inside the map
        tire = next(l for l in config["lanes"] if l["mode"] == "tire")
        assert tire["z"] + tire["gap_m"] + tire["a_depth"] + tire["tire_leg_m"] < 480
        assert {p["name"] for p in config["park"]} == {"own_lord", "enemy_lord"}


def test_the_retarget_plan_switches_crossbows_and_handguns_between_two_targets():
    b = cp.battles("retarget")
    assert len(b) == 2 and all(len(x) == 5 for x in b)
    for x in b:
        assert sorted((l["attacker"], l.get("switch_s") or 0) for l in x) == sorted(
            (s, sw or 0) for s, sw in cp.RETARGET_LANES)
        assert sorted(cp.cell(l) for l in x) == sorted(cp.cell(l) for l in b[0]) and len({cp.cell(l) for l in x}) == 5
        assert all(l["mode"] == "retarget" and l["target"] == l["target2"] == "slave" and l["target_mode"] == "hold"
                   and not l["answer"] and l["max_s"] == 120 for l in x)
    assert b[0][0]["kind"] != b[1][0]["kind"]                               # the lanes rotated
    assert cp.FACTION[cp.UNITS["hgun"][0]] == cp.EMP and cp.UNITS["xbow"][0].endswith("crossbowmen")
    for i in (1, 2):
        config, model_s, arena = cp.run_config("retarget", i)
        assert model_s < 200 and sorted((l["x"], l["z"]) for l in config["lanes"]) == sorted(cp.RETARGET_PLACES)
        names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
        units = json.loads(cp.UNITS_JSON.read_text(encoding="utf-8"))["units"]
        for l in config["lanes"]:
            assert l["target2"] in names and l["target2"] != l["target"] and l["t2_dx"] == 60 and l["a_dx"] == 30
            # both targets ~100 m from the shooter's centre, well inside its range
            d = math.hypot(l["a_dx"], l["gap_m"] + (l["a_depth"] + l["t_depth"]) / 2)
            assert 90 <= d <= 105 and d < units[l["a_key"]]["missile"]["range_m"] - 30
        # every target's centre 250+ m from every other lane's shooter
        for l in config["lanes"]:
            for o in config["lanes"]:
                if o is not l:
                    assert math.hypot(o["x"] + o["a_dx"] - l["x"], o["z"] - l["z"]) > 250
        assert {p["name"] for p in config["park"]} == {"own_lord", "enemy_lord"}


def test_the_retarget2_plan_adds_switches_bows_handguns_and_the_skavens_shooters():
    b = cp.battles("retarget2")
    assert len(b) == 2 and all(len(x) == 10 for x in b)
    want = sorted((s, sw or 0) for s, sw in cp.RETARGET2_LANES)
    assert ("xbow", 3) in cp.RETARGET2_LANES and ("xbow", 7) in cp.RETARGET2_LANES and ("xbow", None) in cp.RETARGET2_LANES
    assert {s for s, _ in cp.RETARGET2_LANES} == {"xbow", "archer", "hgun", "slinger", "nrun"}
    for x in b:
        assert sorted((l["attacker"], l.get("switch_s") or 0) for l in x) == want
        assert len({cp.cell(l) for l in x}) == 10
        for l in x:
            assert l["mode"] == "retarget" and l["target"] == l["target2"] and l["target_mode"] == "hold"
            assert not l["answer"] and l["max_s"] == 120 and l["kind"] == cp.RETARGET_KIND[l.get("switch_s")]
            assert l["target"] == ("spear" if l["attacker"] in ("slinger", "nrun") else "slave")
    assert [l["kind"] for l in b[0]] != [l["kind"] for l in b[1]]              # the lanes rotated
    units = json.loads(cp.UNITS_JSON.read_text(encoding="utf-8"))["units"]
    assert units[cp.UNITS["nrun"][0]]["missile"]["projectile"] == "wh2_main_skv_throwing_star"
    assert cp.FACTION[cp.UNITS["slinger"][0]] == cp.SKV and cp.UNITS["slinger"][0].endswith("slingers_0")
    for i in (1, 2):
        config, model_s, arena = cp.run_config("retarget2", i)
        assert model_s < 200 and sorted((l["x"], l["z"]) for l in config["lanes"]) == sorted(cp.RETARGET2_PLACES)
        assert all(len(side["units"]) <= 20 for side in arena["sides"].values())
        names = {f"{side}_{u['slot']}" for side in ("own", "enemy") for u in arena["sides"][side]["units"]}
        for l in config["lanes"]:
            assert l["target2"] in names and l["target2"] != l["target"] and abs(l["t2_dx"]) == 60
            assert l["a_dx"] == l["t2_dx"] / 2 and abs(l["x"] + l["t2_dx"]) <= 440    # the second target inward
            d = math.hypot(l["a_dx"], l["gap_m"] + (l["a_depth"] + l["t_depth"]) / 2)
            rng = units[l["a_key"]]["missile"]["range_m"]
            if cp.SHORT[l["a_key"]] in cp.RETARGET2_RANGE_SHARE:
                assert d == pytest.approx(0.7 * rng, abs=0.2)                     # slings ~84 m, stars ~49 m
            else:
                assert 90 <= d <= 105 and d < rng - 25
            assert l["z"] + l["gap_m"] + l["a_depth"] <= 480 and l["z"] - l["t_depth"] >= -480
        for l in config["lanes"]:
            for o in config["lanes"]:
                if o is not l:
                    assert math.hypot(o["x"] + o["a_dx"] - l["x"], o["z"] - l["z"]) > 250


def test_the_retarget3_plan_puts_the_targets_in_line_or_across():
    b = cp.battles("retarget3")
    assert len(b) == 2 and all(len(x) == 7 for x in b)
    assert [l["kind"] for l in b[0]] != [l["kind"] for l in b[1]]              # the lanes rotated
    assert sorted(l["kind"] for l in b[0]) == sorted(l["kind"] for l in b[1])
    units = json.loads(cp.UNITS_JSON.read_text(encoding="utf-8"))["units"]
    for i in (1, 2):
        config, model_s, arena = cp.run_config("retarget3", i)
        assert model_s < 200 and all(len(side["units"]) <= 20 for side in arena["sides"].values())
        kinds = sorted(cp.cell(l) for l in config["lanes"])
        assert len(set(kinds)) == 7
        for l in config["lanes"]:
            a = (l.get("a_dx", 0), l["gap_m"] + l["a_depth"] / 2)                 # the shooter from the target's front
            t1 = (0.0, -l["t_depth"] / 2)
            t2 = (l["t2_dx"], -l["t_depth"] / 2 + l.get("t2_dz", 0))
            d1, d2 = math.dist(a, t1), math.dist(a, t2)
            rng = units[l["a_key"]]["missile"]["range_m"]
            if l["kind"].endswith("-ray"):
                assert l["t2_dx"] == 0 and l.get("a_dx", 0) == 0 and 97 <= d1 <= 99 and 120 <= d2 <= 124 < rng
                assert -l["t2_dz"] - l["t_depth"] == cp.RETARGET3_RAY_GAP_M          # the blocks 6 m apart
                assert l["z"] + l["t2_dz"] - l["t_depth"] >= -480
            else:
                assert abs(l["t2_dx"]) == 60 and l["a_dx"] == l["t2_dx"] / 2 and l.get("t2_dz", 0) == 0
                assert abs(d1 - d2) < 1e-6 and 90 <= d1 <= 105
        for l in config["lanes"]:
            for o in config["lanes"]:
                if o is not l:
                    assert math.hypot(o["x"] + o.get("a_dx", 0) - l["x"], o["z"] - l["z"]) > 250


def _retarget_lane(switch_s=10):
    """A retarget lane: 90 men, the target switched every switch_s s from the go; each order silent 5 s, then 20
    shots in 0.5 s and 70 more 0.5 s later; the aimed target loses 2 HP a shot 2 s later for the first 20 of each
    order, 4 for the rest; nothing ever lands on the other."""
    spec = dict(cp.layout(cp.battles("retarget")[0])[0][1])
    spec["switch_s"] = switch_s
    samples, t2, ammo, hp = [], [], 1980.0, {1: 5400.0, 2: 5400.0}
    aims = [(k * switch_s, 1 + k % 2) for k in range(int(60 // switch_s))]
    pending = []
    for k in range(1, 121):
        t = k * 0.5
        o, who = max(a for a in aims if a[0] <= t - 0.5 + 1e-9)
        since = t - o
        n = 20 if abs(since - 5.5) < 1e-9 else 70 if abs(since - 6.0) < 1e-9 else 0
        ammo -= n
        if n:
            pending.append((t + 2, who, n * (2 if n == 20 else 4)))
        for p in [p for p in pending if p[0] <= t + 1e-9]:
            hp[p[1]] -= p[2]
            pending.remove(p)
        samples.append((t, {"x": 0, "z": 0, "men": 90, "hp": 7000, "ammo": ammo, "fire": n > 0},
                        {"x": 0, "z": 0, "men": 180, "hp": hp[1]}, None))
        t2.append((t, {"x": 60, "z": 0, "men": 180, "hp": hp[2]}))
    phases = [{"phase": "aim", "target": who, "n": k, "t": o * 1000} for k, (o, who) in enumerate(aims)]
    return {"run": "r", "spec": spec, "samples": samples, "t2": t2, "men": [], "contacts": {}, "end": None,
            "abilities": [], "phases": phases}


def test_retarget_measure_the_silence_the_first_burst_and_its_hits():
    m = cp.retarget_measure(_retarget_lane())
    assert len(m["orders"]) == 6 and m["shots"] == 540
    o = m["orders"][1]
    assert o["target"] == 2 and o["first_s"] == 5.5 and o["per_man_0_5"] == 0 and o["per_man_5_10"] == 1
    assert abs(o["burst_share"] - 1) < 1e-9 and o["shots_first"] == 90 and o["hp_first"] == 20 * 2 + 70 * 4
    assert o["spill_first"] == 0
    assert m["whole_volleys"] == 6 and m["partial_volleys"] == 0 and m["fire_share"] > 0
    assert m["t1_hp_lost"] + m["t2_hp_lost"] == 6 * 320 and abs(m["hp_per_shot"] - 6 * 320 / 540) < 1e-9
    row = cp.retarget_summary([m])[m["cell"]]
    assert row["first_s_median"] == 5.5 and row["orders_n"] == 6 and row["spill_share"] == 0
    assert abs(row["hp_per_shot_first"] - 320 / 90) < 1e-9 and row["hp_per_shot_later"] is None
    assert [cp.retarget_aim(t, 10) for t in (0, 9.99, 10, 25)] == [1, 1, 2, 1] and cp.retarget_aim(99, None) == 1


def _reengage_lane(**kw):
    """A reengage lane: contact at 10 s, the target losing 10 HP/s steady and 30 HP/s in the first 10 s of each fight;
    out at 160 s, contact 2 at 180 s, the end at 240 s."""
    spec = dict(cp.layout(cp.battles("reengage")[0])[0][0], **kw)
    assert spec["mode"] == "reengage"
    samples, hp = [], 9600.0
    for k in range(0, 481):
        t = k * 0.5
        if k:
            if 10 <= t - 0.5 < 160:
                hp -= 0.5 * (30 if t - 10 <= 10 else 10)
            elif t - 0.5 >= 180:
                hp -= 0.5 * (15 if t - 180 <= 10 else 10)
        fat = "threshold_active" if t < 180 else "threshold_tired"
        m = 10 <= t < 165 or t >= 180
        samples.append((t, {"x": 0, "z": 40, "hp": 8000, "men": 120, "m": m, "fat": fat},
                        {"x": 0, "z": 0, "hp": hp, "men": 160, "m": m, "fat": fat}, None))
    return {"run": "r", "spec": spec, "samples": samples, "men": [], "contacts": {1: 10.0, 2: 180.0}, "end": None,
            "abilities": [], "phases": [{"phase": "out", "t": 160000}, {"phase": "clear", "t": 166000},
                                        {"phase": "back", "t": 175000}]}


def test_measure_the_second_contact_wave_and_fatigue():
    m = cp.measure(_reengage_lane())
    assert m["contact2_s"] == 180.0 and m["cell"].endswith(" reengage")
    assert abs(m["tg_hp_steady"] - 10) < 0.2 and abs(m["tg_hp2_steady"] - 10) < 1e-6
    assert abs(m["tg_wave2_10"] - 1.5) < 0.05 and m["tg_wave_10"] > 2.8
    assert m["tg_fat_c1"] == 1 and m["tg_fat_c2"] == 3 and m["a_fat_c2"] == 3
    rows = cp.summary([m])
    assert "tg_wave2_10" in rows[0] and "a_fat_c2" in rows[0]


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
