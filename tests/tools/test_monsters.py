"""The Skaven monsters' passports, effects and pools (29.10.2026; build/diag_skv_mon/units.md): Rat Ogres, the Brood
Horror, the Rat Ogre Mutant - the database's numbers in config/nn/units.json, abilities.json, effects.json and the v3
pools. No torch (the simulator's rules: test_sim_monsters.py)."""
import json

from tools import config as project
from tools.nn import effects
from tools.nn.armies import pools

OGRES, HORROR, MUTANT = ("wh2_main_skv_mon_rat_ogres", "wh2_dlc16_skv_mon_brood_horror_0",
                         "wh2_dlc16_skv_mon_rat_ogre_mutant")
REGEN, POISON = "wh_main_unit_passive_regeneration", "wh_main_unit_contact_poison"


def units():
    return json.loads((project.CONFIG_DIR / "nn" / "units.json").read_text(encoding="utf-8"))["units"]


def test_the_passports_by_the_database():
    u = units()
    o, h, m = u[OGRES], u[HORROR], u[MUTANT]
    # men, health, mass, size (build/diag_skv_mon/units.md "Числа")
    assert (o["men"], o["hp_per_man"], o["hp_total"], o["mass"], o["size"]) == (16, 483, 7728, 1500.0, "large")
    assert (h["men"], h["hp_total"], h["mass"], h["size"]) == (1, 6009, 2500.0, "very_large")
    assert (m["men"], m["hp_total"], m["mass"], m["size"]) == (1, 6508, 3000.0, "very_large")
    # the body: the Horror's capsule 3.8 x 0.36 (7.6 x 2.74 m), not a circle 7.6 m across; the Mutant a circle r 2.0
    assert (h["body_shape"], h["radius_m"], h["radii_ratio"]) == ("capsule", 3.8, 0.36)
    assert (m["body_shape"], m["radius_m"], m["radii_ratio"]) == ("circle", 2.0, 1.0)
    assert (o["body_shape"], o["radius_m"]) == ("circle", 1.0)
    # the chance to stay on the feet (knock_interrupts_ignore_chance) and to take no hit reaction
    assert (h["knock_ignore"], m["knock_ignore"], o["knock_ignore"]) == (60, 65, 20)
    assert (h["hit_reaction_ignore"], m["hit_reaction_ignore"], o["hit_reaction_ignore"]) == (70, 65, 20)
    assert u["wh_main_emp_inf_spearmen_0"]["knock_ignore"] == 0
    # the weapons: length, splash (4 targets up to medium / large), poison, magic
    assert (h["melee"]["weapon_length"], m["melee"]["weapon_length"], o["melee"]["weapon_length"]) == (7.0, 6.0, 2.0)
    assert (h["melee"]["splash_max_attacks"], h["melee"]["splash_target_size"]) == (4, "medium")
    assert (m["melee"]["splash_max_attacks"], m["melee"]["splash_target_size"]) == (4, "large")
    assert (o["melee"]["splash_max_attacks"], o["melee"]["splash_target_size"]) == (4, "medium")
    assert h["melee"]["contact_phase"] == POISON and m["melee"]["magic"] and not h["melee"]["magic"]
    assert m["damage_resist"]["physical"] == 20 and h["damage_resist"]["physical"] == 0
    # terror, fear, frenzy, Wounds, regeneration; the ogres brace
    assert {"causes_fear", "causes_terror"} <= set(h["attributes"]) and "causes_terror" in m["attributes"]
    assert "causes_terror" not in o["attributes"] and "causes_fear" in o["attributes"]
    assert REGEN in h["abilities"] and REGEN not in m["abilities"]
    assert all("wh_main_unit_passive_frenzy" in x["abilities"] for x in (o, h, m))
    assert all("wh3_main_unit_passive_single_entity" in x["abilities"] for x in (h, m))
    assert o["can_brace"] and not h["can_brace"]
    assert (o["spacing"]["h"], o["rank_depth"]) == (6.5, 3)
    # the multiplayer costs (the campaign's 800 / 1100 / 1200 are not used)
    assert (o["multiplayer_cost"], h["multiplayer_cost"], m["multiplayer_cost"]) == (700, 1050, 1200)


def test_regeneration_heals_a_thousandth_a_second_and_is_weak_to_fire():
    ab = json.loads((project.CONFIG_DIR / "nn" / "abilities.json").read_text(encoding="utf-8"))["abilities"]
    assert ab[REGEN]["heal"] == {"share": 0.001, "every_s": 1.0} and ab[REGEN]["passive"]
    assert [(e["stat"], e["value"]) for e in ab[REGEN]["effects"]] == [("stat_weakness_flame", -20.0)]
    doc = effects.load()
    e = doc["effects"][REGEN]
    assert e["modelled"] and e["heal"] == {"share": 0.001, "every_s": 1.0}
    assert [s["sim"] for s in e["stats"]] == ["resist_flame"]
    t = doc["effects"]["causes_terror"]
    assert t["modelled"] and t["rules"] == ["causes_terror"]
    assert doc["inflicts"][HORROR] == [POISON]
    assert REGEN in doc["units"][HORROR] and "causes_terror" in doc["units"][MUTANT]


def test_the_monsters_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert ({OGRES, HORROR, MUTANT} <= keys) is has, scheme
        assert not ({OGRES, HORROR, MUTANT} & keys) or has
    v3 = {x.key: x for x in pools.load(scheme="v3")["wh2_main_skv_skaven"].units}
    assert v3[OGRES].width == 39.0                 # 6 files x 6.5 m: the game's 16 / rank_depth 3
