"""The Stormfiends with Grinderfists: passport, ability passport, the AI's rule, pools and the probe's plan (02.11.2026;
build/diag_roster/wh3_dlc29_skv_mon_stormfiend_grinderfists/units.md, mechanic.md, build/diag_stormfiend/). No torch (the simulator's rules: test_sim_stormfiend_torch.py)."""
import json
from pathlib import Path

from tools import config as project
from tools.nn import charge_probe as cp

SF = "wh3_dlc29_skv_mon_stormfiend_grinderfists"
SNARE = "wh3_dlc29_unit_abilities_stormfiend_seismic_snare"
GRINDERS = "wh2_dlc14_unit_abilities_seismic_snare"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][SF]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"]) == (12, 731, 8772, 90, 1200, 65)
    assert (p["mass"], p["radius_m"], p["size"], p["caste"]) == (1500.0, 1.25, "large", "monstrous_infantry")
    m = p["melee"]
    assert (m["attack"], m["defence"], m["charge_bonus"], m["damage"], m["ap_damage"], m["attack_interval_s"]) == (38, 31, 45, 25, 80, 3.8)
    assert m["magic"] and m["splash_max_attacks"] == 3 and m["splash_target_size"] == "large" and m["splash_power_multiplier"] == 1.1 and m["weapon_length"] == 6.0
    assert p["spacing"] == {"template": "wh_main_monstrous_infantry", "h": 6.0, "v": 6.0, "scatter": 1.0, "chaotic": False}
    assert SNARE in p["abilities"]
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][SF] == p


def test_seismic_snare_is_the_grinders_with_one_use():
    a = load("abilities.json")["abilities"]
    s, g = a[SNARE], a[GRINDERS]
    assert s["uses"] == 1 and g["uses"] == -1
    assert s["self_cast"] and s["cant_move"] and s["loop_stance"] and s["targets"]["enemies"] and s["enemy_units"] == -1
    same = lambda d: {k: v for k, v in d.items() if k not in ("uses", "phases")}
    assert same(s) == same(g)
    assert (s["active_s"], s["recharge_s"], s["range_m"]) == (13.0, 60.0, 35.0)


def test_the_ai_presses_it_in_melee_and_the_why_says_it_is_a_stand_in():
    ab = load("sim.json")["abilities"]
    assert SNARE in ab["model"] and ab["triggers"][SNARE] == "melee"
    why = ab["stormfiend_trigger_why"]
    assert "hex_melee_area_self" in why and "not found" in why and "35 m" in why


def test_the_edge_of_the_sparse_rule_and_the_measure_written_with_it():
    s = load("sim.json")["contact"]["sparse"]
    h, r = 6.0, load("units.json")["units"][SF]["radius_m"]
    assert h / (2 * r + 2 * s["clear_m"]) == 2.0                       # exactly on the edge: 2 bodies a place
    assert h / (2 * r + 2 * (s["clear_m"] + 0.01)) < 2.0               # at 0.26 only 1
    assert s["clear_m"] == 0.25 and "Stormfiend" in s["why"] and "20261102-132600" in s["why"]
    assert "610" in s["why"]                                           # the rat ogres' twin today


def test_the_catalogue_of_effects_does_not_grow():
    fx = load("effects.json")
    assert fx["units"][SF] == ["causes_fear", "guerrilla_deploy", "hide_forest", "moulder_monster", "wallbreaker", "wh2_main_unit_passive_scurry_away", "wh_main_unit_passive_frenzy"]
    assert SF not in fx["inflicts"]
    assert fx["order"][-1] == "wh2_dlc14_unit_contact_poisoned_wind"


def test_only_the_v3_scheme_fields_them():
    doc = load("pools.json")
    skv = doc["factions"]["wh2_main_skv_skaven"]
    assert SF in json.dumps(skv["units_v3"]) and SF not in json.dumps({k: v for k, v in skv.items() if k != "units_v3"})
    groups = load("army_templates.json")["factions"]
    rows = [f["units"][SF] for f in groups.values() if SF in f["units"]]
    assert rows and rows[0] == [{"group": "wh3_default_land_monsters_flanking_strong", "quality": "tier_3_step_3_units"}]


def test_the_probes_plan():
    b = cp.battles("stormfiend")[0]
    assert [x["kind"] for x in b] == ["charge", "snare_melee", "knights", "walk"]
    assert b[0]["attacker"] == "stormfiend" and b[0]["target"] == "spear" and b[2]["target"] == "stormfiend"
    assert b[1]["a_ability"] == SNARE and b[1]["a_ability_after_s"] == 10
    # the walk lane's second press past the first's 13 s and the recharge's 60 s; far from any enemy (35 m)
    w = b[3]
    assert [c["at_s"] for c in w["a_casts"]] == [3, 85] and 85 > 3 + 13 + 60 and w["gap_m"] > 35 + 5
    assert w["max_s"] > 85
    cfg = cp.run_config("stormfiend", 1)[0]
    assert cfg["men_near_m"] == 60 and cfg["men_after_s"] >= 60


def test_the_drills_cannot_take_them():
    """The caste monstrous_infantry passes no drill's filter (as the rat ogres): no need of NOT_IN_DRILLS."""
    src = Path(project.ROOT / "tools" / "nn" / "unit" / "drill_hold.py").read_text(encoding="utf-8")
    assert SF not in src
    assert load("units.json")["units"][SF]["caste"] not in ("melee_infantry", "missile_infantry")
