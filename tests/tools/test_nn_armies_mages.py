"""Casters in the armies (config/nn/pools.json "mages", tools/nn/armies generate.add_mages, pools.mages): off by default,
on - each side its faction's caster beside the lord with exactly his spells in the battle file."""
import json

import numpy as np

from tools.nn import scenario
from tools.nn.armies import generate, place, pools as P
from tools.nn.sim import params as sim_params

GOLD, WARLOCK = "wh3_dlc25_emp_cha_wizard_metal", "wh2_main_skv_cha_warlock_engineer"
SD, WL = "wh_main_spell_metal_searing_doom", "wh2_main_spell_ruin_warp_lightning"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def _pools(tmp_path, share):
    doc = json.loads(P.POOLS.read_text(encoding="utf-8"))
    doc["mages"]["share"] = share
    path = tmp_path / "pools.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_the_config_keeps_the_casters_off_and_the_battles_as_before():
    share, mages = P.mages()
    assert share == 0.0 and mages == {}
    assert P.mage_passports() == {}                         # the simulator's units stay units.json's
    g = generate.Generator()
    for seed in (3, 11):
        for armies in ("v1", "v2"):
            a = g.generate(np.random.default_rng(seed), armies=armies)
            assert not any(u["key"] in (GOLD, WARLOCK) for s in a["sides"].values() for u in s["units"])
            assert all("mage" not in s for s in a["sides"].values())


def test_a_share_names_each_factions_caster_with_his_passport_spells(tmp_path):
    share, mages = P.mages(_pools(tmp_path, 0.5))
    assert share == 0.5 and set(mages) == {EMP, SKV}
    assert mages[EMP].unit.key == GOLD and mages[EMP].spells == (SD,) and mages[EMP].unit.cost == 250
    assert mages[SKV].unit.key == WARLOCK and mages[SKV].spells == (WL,) and mages[SKV].unit.cost == 400
    assert mages[EMP].unit.men == 1 and mages[EMP].unit.slot == "mage"
    assert set(P.mage_passports(_pools(tmp_path, 0.5))) == {GOLD, WARLOCK}


def test_on_every_side_gets_its_caster_beside_the_lord_on_top_of_its_army(tmp_path):
    g = generate.Generator()
    g.mage_share, g.mages = P.mages(_pools(tmp_path, 1.0))
    off = generate.Generator().generate(np.random.default_rng(5), sides=(EMP, SKV), armies="v2")
    a = g.generate(np.random.default_rng(5), sides=(EMP, SKV), armies="v2")
    for tag, key, spell, cost in (("own", GOLD, SD, 250), ("enemy", WARLOCK, WL, 400)):
        side, before = a["sides"][tag], off["sides"][tag]
        assert side["units"][:-1] == before["units"]           # the army as without casters (the draw comes last)
        mage = side["units"][-1]
        lord = side["units"][0]
        assert (mage["key"], mage["slot"], mage["men"], mage["special_abilities"]) == (key, "mage", 1, [spell])
        assert mage["forward"] == lord["forward"] and mage["lateral"] == lord["lateral"] + place.HERO_GAP_M
        assert side["cost"] == before["cost"] + cost and side["mage"] == key


def test_the_battle_file_gives_the_generated_caster_his_spell_and_innate_abilities(tmp_path):
    g = generate.Generator()
    g.mage_share, g.mages = P.mages(_pools(tmp_path, 1.0))
    a = g.generate(np.random.default_rng(7), sides=(SKV, EMP))
    xml = scenario.scenario_xml(a, "enemy", 600)
    block = [b for b in xml.split("<unit ") if WARLOCK in b][0]
    assert f"<special_ability>{WL}</special_ability>" in block
    assert "wh2_main_hero_abilities_doomrocket" in block and "wh3_main_unit_passive_single_entity" in block
    gold = [b for b in xml.split("<unit ") if GOLD in b][0]
    assert f"<special_ability>{SD}</special_ability>" in gold and gold.count("<special_ability>") == 2
    # the units without the field keep the game's defaults
    assert xml.count("<unit_capabilities>") == 2


def test_the_simulator_takes_the_casters_passports_only_while_the_share_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "POOLS", _pools(tmp_path, 1.0))
    sim_params.load.cache_clear()
    try:
        units = sim_params.load().units
        assert GOLD in units and WARLOCK in units and units[GOLD]["spells"] == [SD]
    finally:
        monkeypatch.undo()
        sim_params.load.cache_clear()
    assert GOLD not in sim_params.load().units
