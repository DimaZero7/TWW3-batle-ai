"""The melee probe's plans, battle files and measures (tools/nn/charge_probe.py) and the lord swarm's damage
plan (tools/nn/lord_swarm.py). The simulator twin needs torch (tested in the container by running it)."""
import json

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
            assert {p["name"] for p in config["park"]} == {"own_lord", "enemy_lord"} - used
            xs = sorted(l["x"] for l in config["lanes"])
            assert all(b - a >= 200 for a, b in zip(xs, xs[1:]))


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
