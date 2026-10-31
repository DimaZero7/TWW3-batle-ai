"""Innate unit effects: the catalogue config/nn/effects.json (tools/nn/effects.py) built from the unit and
ability passports. Numpy-free, torch-free (the simulator's and the network's use: test_sim.py,
test_nn_model.py)."""
import json

from tools.nn import effects

UDOC = json.loads(effects.UNITS.read_text(encoding="utf-8"))
UNITS = UDOC["units"]
PHASES = UDOC["contact_phases"]
ABILITIES = json.loads(effects.ABILITIES.read_text(encoding="utf-8"))["abilities"]
SIN, SCURRY, PENITENT, FRENZY = ("wh2_main_unit_passive_strength_in_numbers", "wh2_main_unit_passive_scurry_away",
                                 "wh_dlc04_unit_passive_strength_of_the_penitent", "wh_main_unit_passive_frenzy")
POISON, CENSER, SUPPRESS = ("wh_main_unit_contact_poison", "wh2_main_unit_contact_plague_monk_censer_morale",
                            "wh2_dlc12_unit_contact_suppressive_fire")


def test_the_saved_catalogue_is_what_the_passports_give():
    saved = effects.load()
    built, links, order, inflicts = effects.build(UNITS, ABILITIES, saved["order"], PHASES)
    assert json.loads(json.dumps(effects.document(built, links, order, inflicts))) == saved


def test_every_unit_is_linked_to_its_attributes_and_innate_abilities_only():
    _, links, _, _ = effects.build(UNITS, ABILITIES, (), PHASES)
    for key, u in UNITS.items():
        innate = [a for a in u["abilities"] if ABILITIES[a]["passive"] or ABILITIES[a]["auto"]]
        assert sorted(links[key]) == sorted(u["attributes"] + innate), key
    assert "wh_main_character_abilities_rally" not in links["wh2_main_skv_cha_warlord_0"]      # cast, not innate


def test_effects_carry_the_database_values_and_conditions():
    built, _, _, _ = effects.build(UNITS, ABILITIES, (), PHASES)
    sin = built[SIN]
    assert {(s["stat"], s["value"]) for s in sin["stats"]} == {("scalar_speed", 0.9), ("stat_melee_defence", 8.0),
                                                                ("stat_morale", 6.0)}
    assert sin["off_when"] == ["hp_below_half"] and sin["modelled"]
    assert built[SCURRY]["off_when"] == ["not_wavering"]
    pen = built[PENITENT]
    # fired whenever ready (in melee: off out of it); its recharge context is when its 3 s run (build/effects P1)
    assert pen["kind"] == "timed" and pen["timed"] == {"active_s": 20.0, "recharge_s": 3.0, "fires_when": [],
                                                       "recharge_needs": ["losing_melee"]}
    assert pen["off_when"] == ["out_of_melee"]
    assert built[FRENZY]["rules"] == ["immune_to_psychology"] and built[FRENZY]["modelled"]
    assert built["wh3_main_unit_passive_single_entity"]["needs"] == ["hp_below_quarter"]
    assert built["unbreakable"]["modelled"] and not built["hide_forest"]["modelled"] and built["hide_forest"]["why"]


def test_contact_effects_come_from_the_weapons_with_the_database_numbers():
    built, links, _, inflicts = effects.build(UNITS, ABILITIES, (), PHASES)
    # who lays them: the censer (melee), the poisoned Giant Rats (melee), the Ratling Gun (its bullets)
    assert inflicts == {"wh2_main_skv_inf_plague_monk_censer_bearer": [CENSER],
                        "wh2_dlc16_skv_mon_wolf_rats_1": [POISON], "wh2_dlc12_skv_inf_ratling_gun_0": [SUPPRESS],
                        # line g43 (25.10.2026): the poisoned stars and slings, the Jezzails' shield breaker
                        "wh2_main_skv_inf_gutter_runners_1": [POISON],
                        "wh2_main_skv_inf_gutter_runner_slingers_1": [POISON],
                        "wh2_dlc12_skv_inf_warplock_jezzails_0": ["wh2_dlc09_unit_contact_shieldbreaker"],
                        # artillery step 1 (29.10.2026): the Plagueclaw Catapult's shell (its projectile and its blast)
                        "wh2_main_skv_art_plagueclaw_catapult": [CENSER],
                        # the Skaven monsters (29.10.2026): the Brood Horror's claws poison as the Giant Rats' bites
                        "wh2_dlc16_skv_mon_brood_horror_0": [POISON],
                        # the Pusbags (31.10.2026): the poisoned sword and the vortices of their dead (the censer's -10)
                        "wh3_dlc29_skv_inf_pusbags": [CENSER, POISON]}
    # nobody owns a contact effect (it lies on the enemy it hits)
    assert not any(k in v for v in links.values() for k in (POISON, CENSER, SUPPRESS))
    for k in (POISON, CENSER, SUPPRESS):
        e = built[k]
        assert e["kind"] == "contact" and e["duration_s"] == 10.0 and e["modelled"], k
    assert {(s["stat"], s["value"], s["sim"]) for s in built[SUPPRESS]["stats"]} == {("scalar_speed", 0.7, "speed")}
    assert {(s["stat"], s["value"], s["sim"]) for s in built[CENSER]["stats"]} == {("stat_morale", -10.0,
                                                                                      "leadership")}
    assert {s["sim"] for s in built[POISON]["stats"]} == {"speed", "damage", "ap", "m_damage", "m_ap"}
    assert all(s["value"] == 0.85 for s in built[POISON]["stats"])


def test_what_the_simulator_lacks_is_schema_only_with_the_reason():
    p = {"passive": True, "auto": False, "active_s": -1, "recharge_s": -1, "range_m": 0,
         "effects": [{"stat": "stat_bonus_vs_large", "how": "add", "value": 10, "on": ["self"]}],
         "attributes": [], "off_when": ["some_new_flag"], "recharge_when": []}
    e = effects.ability_effect("x_passive_new", p)
    assert not e["modelled"] and "stat_bonus_vs_large" in e["why"] and "some_new_flag" in e["why"]
    assert not effects.attribute_effect("brand_new_attribute")["modelled"]
    c = effects.contact_effect("x_contact", {"duration_s": 10.0, "stats": [{"stat": "stat_bonus_vs_large", "how": "add",
                                                                             "value": -5}]})
    assert not c["modelled"] and "stat_bonus_vs_large" in c["why"]


def test_the_order_is_append_only():
    _, _, order, _ = effects.build(UNITS, ABILITIES, ["zzz_gone", "unbreakable"], PHASES)
    assert order[:2] == ["zzz_gone", "unbreakable"] and set(order[2:]) >= {SIN, SCURRY, POISON}
    saved = effects.load()["order"]
    assert saved[:14] == ["charge_defense_vs_large", "charge_reflection", "encourages", "expendable",
                          "guerrilla_deploy", "hide_forest", "mounted_fire_move", "unbreakable",
                          SCURRY, SIN, "wh3_main_unit_passive_single_entity", PENITENT,
                          "wh_main_lord_passive_hold_the_line", FRENZY]
