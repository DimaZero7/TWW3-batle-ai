"""Innate unit effects: the catalogue config/nn/effects.json (tools/nn/effects.py) built from the unit and
ability passports. Numpy-free, torch-free (the simulator's and the network's use: test_sim.py,
test_nn_model.py)."""
import json

from tools.nn import effects

UNITS = json.loads(effects.UNITS.read_text(encoding="utf-8"))["units"]
ABILITIES = json.loads(effects.ABILITIES.read_text(encoding="utf-8"))["abilities"]
SIN, SCURRY, PENITENT, FRENZY = ("wh2_main_unit_passive_strength_in_numbers", "wh2_main_unit_passive_scurry_away",
                                 "wh_dlc04_unit_passive_strength_of_the_penitent", "wh_main_unit_passive_frenzy")


def test_the_saved_catalogue_is_what_the_passports_give():
    saved = effects.load()
    built, links, order = effects.build(UNITS, ABILITIES, saved["order"])
    assert json.loads(json.dumps(effects.document(built, links, order))) == saved


def test_every_unit_is_linked_to_its_attributes_and_innate_abilities_only():
    _, links, _ = effects.build(UNITS, ABILITIES)
    for key, u in UNITS.items():
        innate = [a for a in u["abilities"] if ABILITIES[a]["passive"] or ABILITIES[a]["auto"]]
        assert sorted(links[key]) == sorted(u["attributes"] + innate), key
    assert "wh_main_character_abilities_rally" not in links["wh2_main_skv_cha_warlord_0"]      # cast, not innate


def test_effects_carry_the_database_values_and_conditions():
    built, _, _ = effects.build(UNITS, ABILITIES)
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


def test_what_the_simulator_lacks_is_schema_only_with_the_reason():
    p = {"passive": True, "auto": False, "active_s": -1, "recharge_s": -1, "range_m": 0,
         "effects": [{"stat": "stat_accuracy", "how": "add", "value": 10, "on": ["self"]}],
         "attributes": [], "off_when": ["some_new_flag"], "recharge_when": []}
    e = effects.ability_effect("x_passive_new", p)
    assert not e["modelled"] and "stat_accuracy" in e["why"] and "some_new_flag" in e["why"]
    assert not effects.attribute_effect("brand_new_attribute")["modelled"]


def test_the_order_is_append_only():
    _, _, order = effects.build(UNITS, ABILITIES, ["zzz_gone", "unbreakable"])
    assert order[:2] == ["zzz_gone", "unbreakable"] and set(order[2:]) >= {SIN, SCURRY}
