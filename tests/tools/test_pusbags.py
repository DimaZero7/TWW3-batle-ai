"""The Pusbags' passport, ability passports, effects and pools (31.10.2026; build/diag_roster/wh3_dlc29_skv_inf_pusbags/
units.md, mechanic.md): the skavenslaves' body with defence 5, 43 HP, poison, and the two classes of rules as data - a
phase's direct damage (Unshakable Pathogen) and a vortex at a man's death (Pestilent Perfection). No torch (the
simulator's rules: test_sim_pusbags_torch.py)."""
import json

import pytest

from tools import config as project
from tools.nn import abilities, effects
from tools.nn.armies import pools

PUS, SLAVES = "wh3_dlc29_skv_inf_pusbags", "wh2_main_skv_inf_skavenslaves_0"
PATHOGEN, PERFECTION = "wh3_dlc29_unit_passive_unshakable_pathogen", "wh3_dlc29_unit_passive_pestilent_perfection"
CENSER_MORALE, POISON = "wh2_main_unit_contact_plague_monk_censer_morale", "wh_main_unit_contact_poison"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_skavenslaves_but_five_numbers():
    u = load("units.json")["units"]
    p, s = u[PUS], u[SLAVES]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["melee"]["defence"], p["multiplayer_cost"]) == (180, 43, 7740, 5, 200)
    assert p["melee"]["contact_phase"] == POISON and s["melee"]["contact_phase"] is None
    for k in ("mass", "size", "radius_m", "speed", "leadership", "armour", "rank_depth", "spacing"):
        assert p[k] == s[k], k
    assert {PATHOGEN, PERFECTION} <= set(p["abilities"]) and p["attributes"] == ["expendable", "hide_forest"]
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))["units"]
    assert probe[PUS] == p


def test_the_direct_damage_passport_is_the_database_and_the_program():
    ab = load("abilities.json")["abilities"][PATHOGEN]
    d = ab["direct"]
    assert ab["passive"] and ab["off_when"] == ["out_of_melee"] and ab["targets"]["self"]
    assert (d["amount"], d["entities"], d["every_s"], d["on"], d["scale"], d["min_unary"]) == (5, 3, 1.0, ["self"], 1.0, 0.5)
    # trunc(5 x U[0.5, 1]) at least 1: 2 / 3 / 4 HP (mechanic.md B2), the mean 3.2 (9.6 HP a second)
    assert d["hit_dist"] == [[2, 0.2], [3, 0.4], [4, 0.4]] and d["hit_hp"] == pytest.approx(3.2)
    assert d["magical"] and d["range_m"] == 0 and d["duration_s"] == -1.0
    # the size scale: large 0.75 -> trunc(3.75) = 3 -> 1 or 2 HP
    assert abilities.direct_hit_dist(5, 0.75, 0.5) == [[1, pytest.approx(1 / 3, abs=1e-6)], [2, pytest.approx(2 / 3, abs=1e-6)]]
    assert abilities.direct_hit_mean(5, 0.75, 0.5) == pytest.approx(5 / 3, abs=1e-5)


def test_the_vortex_passport_is_battle_vortexs():
    ab = load("abilities.json")["abilities"]
    p = ab[PERFECTION]
    # active_time 1 (the vortex's life) and no recharge: passive by its behaviour, not 'auto' (a timed one)
    assert p["passive"] and not p["auto"] and p["behaviour"] == "vortex_on_entity_death"
    v = p["vortex"]
    assert (v["delay_s"], v["duration_s"], v["start_radius_m"], v["goal_radius_m"], v["expansion_speed"]) == (1.0, 1.0, 1.0,
                                                                                                         2.0, 30.0)
    assert (v["damage"], v["damage_ap"], v["magical"], v["spell"], v["force"]) == (5, 0, True, False, 25.0)
    assert (v["contact"], v["allies"], v["enemies"], v["move_m_s"], v["n"]) == (CENSER_MORALE, False, True, 0.0, 1)
    # the Warlord's Verminous Valour names its vortex too (an active ability: not acted on - the catalogue's class)
    assert ab["wh2_main_character_abilities_verminous_valour"]["vortex"]["goal_radius_m"] == 25.0


def test_the_catalogue_models_both_and_the_vortex_lays_the_censers_morale():
    doc = effects.load()
    d, v = doc["effects"][PATHOGEN], doc["effects"][PERFECTION]
    assert d["modelled"] and d["off_when"] == ["out_of_melee"] and d["direct"]["melee_only"]
    assert v["modelled"] and v["vortex"]["behaviour"] == "vortex_on_entity_death" and not v["off_when"]
    assert {PATHOGEN, PERFECTION} <= set(doc["units"][PUS])
    assert doc["inflicts"][PUS] == [CENSER_MORALE, POISON]
    # appended: an older network's effect pairs keep their places (the Warp Grinders' wallbreaker after them, 31.10.2026)
    assert doc["order"][-3:-1] == [PERFECTION, PATHOGEN]


def test_what_the_class_does_not_model_says_why():
    # a passive's direct damage on enemies (the Cauldron's miasma; the Warp Grinders' active Warp-Quake is the simulator's
    # abilities.py, no innate effect) and another vortex
    p = {"passive": True, "direct": {"amount": 5, "entities": 18, "every_s": 0.5, "on": ["enemies"], "scale": 1.0,
                                     "min_unary": 0.5, "hit_hp": 3.2, "range_m": 35.0}}
    e = effects.ability_effect("x_passive_tremor", p)
    assert not e["modelled"] and "enemies" in e["why"]
    q = {"passive": True, "behaviour": "vortex_per_entity",
         "vortex": {"delay_s": 0, "n": 1, "move_m_s": 24.0, "allies": True, "enemies": True, "goal_radius_m": 12.0,
                    "damage": 20, "damage_ap": 0, "force": 135.0}}
    e = effects.ability_effect("x_passive_breath", q)
    assert not e["modelled"] and "moving" in e["why"] and "own side" in e["why"]


def test_the_pusbags_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert (PUS in keys) is has, scheme
    v3 = {x.key: x for x in pools.load(scheme="v3")["wh2_main_skv_skaven"].units}
    assert v3[PUS].width == v3[SLAVES].width == 30.0
    groups = load("army_templates.json")["factions"]["wh2_main_skv_skaven"]["units"][PUS]
    assert groups and groups[0]["group"].startswith("wh3_default_land_melee_infantry")
