"""The magic probe (tools/nn/magic_probe.py on the melee probe's frame) and the battle file's <unit_capabilities>
(tools/nn/scenario.py): the casters get exactly their spell and their innate abilities."""
import json

import pytest

from tools.nn import charge_probe as cp
from tools.nn import magic_probe as mp
from tools.nn import scenario


def test_capabilities_add_the_innate_abilities_and_keep_the_defaults_without_a_list(tmp_path):
    path = tmp_path / "units.json"
    path.write_text(json.dumps({"units": {"hero": {"abilities": ["passive_a", "bound_b"]}}}), encoding="utf-8")
    assert scenario.capabilities({"key": "hero"}, (path,)) is None               # no list: the game's defaults
    assert scenario.capabilities({"key": "hero", "special_abilities": ["spell", "passive_a"]}, (path,)) == \
        ["passive_a", "bound_b", "spell"]
    assert scenario.capabilities({"key": "hero", "special_abilities": []}, (path,)) == ["passive_a", "bound_b"]
    with pytest.raises(KeyError):                                                  # innate abilities unknown
        scenario.capabilities({"key": "nobody", "special_abilities": ["spell"]}, (path,))
    xml = scenario.capabilities_xml(["a", "b"])
    assert xml == ("        <unit_capabilities>\n          <special_ability>a</special_ability>\n"
                   "          <special_ability>b</special_ability>\n        </unit_capabilities>\n")
    assert scenario.capabilities_xml(None) == ""


def test_the_training_arena_has_no_unit_capabilities():
    xml = scenario.scenario_xml(scenario.load_arena(), "enemy")
    assert "unit_capabilities" not in xml


def test_the_magic_battle_file_gives_the_casters_their_spell_and_innate_abilities(tmp_path):
    xml = cp.write_scenario("magic", 1, tmp_path / "m.xml").read_text(encoding="utf-8")
    blocks = xml.split("<unit ")[1:]
    gold = [b for b in blocks if "wh3_dlc25_emp_cha_wizard_metal" in b]
    warlock = [b for b in blocks if "wh2_main_skv_cha_warlock_engineer" in b]
    assert len(gold) == len(warlock) == 2
    assert sum("unit_capabilities" in b for b in gold) == 1 and sum("unit_capabilities" in b for b in warlock) == 1
    (g,) = [b for b in gold if "unit_capabilities" in b]
    assert g.count("<special_ability>") == 2 and f"<special_ability>{mp.SEARING_DOOM}</special_ability>" in g
    assert "wh3_main_unit_passive_single_entity" in g and g.index("unit_capabilities") < g.index("unit_experience")
    (w,) = [b for b in warlock if "unit_capabilities" in b]
    assert f"<special_ability>{mp.WARP_LIGHTNING}</special_ability>" in w and "wh2_main_hero_abilities_doomrocket" in w
    assert w.count("<special_ability>") == 5
    assert '<width metres="5"/>' in g                                              # a lone hero


def test_the_magic_plan_casts_on_held_targets_and_logs_the_winds():
    plan = cp.battles("magic")
    assert len(plan) == 3 and all(len(b) == 4 for b in plan)
    for i, b in enumerate(plan, 1):
        config, model_s, arena = cp.run_config("magic", i)
        assert config["winds_ms"] == 1000 and config["men_ms"] == 1000 and config["men_near_m"] >= 300
        for ln in config["lanes"]:
            assert ln["mode"] == "cast" and not ln["answer"] and ln["target_mode"] in ("hold", "push")
            assert all(c["key"] in (mp.SEARING_DOOM, mp.WARP_LIGHTNING) and c["tg"] == "t" for c in ln["casts"])
            assert ln["a_width"] == 5 and ln["a_depth"] == 0
    b1 = cp.run_config("magic", 1)[0]
    assert b1["winds_set"] == []                                                   # the game's own winds
    l1 = b1["lanes"][0]
    assert [c["at_s"] for c in l1["casts"]][:3] == [40, 41, 42] and l1["casts"][-1]["at_s"] == 150
    assert [ln["casts"] for ln in b1["lanes"][2:]] == [[], []]                     # the defaults' casters never cast
    b2 = cp.run_config("magic", 2)[0]
    assert {s["side"]: (s["current"], s["reserve"]) for s in b2["winds_set"]} == {1: (30, 40), 2: (30, -60)}
    b3 = cp.battles("magic")[2]
    assert [ln["gap_m"] for ln in b3] == [mp.GAP_M, mp.FAR_M, mp.GAP_M, mp.GAP_M] and mp.FAR_M > 200 > mp.GAP_M + 20


def _run(tmp_path):
    """A recorded run: one Empire caster (L1) casting at 10 s (the winds fall 6 at 11 s), the target losing 600 HP and
    10 men over 2-6 s after the try; the Skaven side's winds rising 0.5 a second, no cast."""
    lanes = [{"name": "L1", "attacker": "own_goldwiz_1", "target": "enemy_clanrat_1", "mode": "cast", "kind": "k1",
              "gap_m": 120, "target_mode": "hold"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"plan": "magic", "battle": 1, "lanes": lanes}}),
                                            encoding="utf-8")
    rows = [{"event": "probe_winds_api", "side": 1, "found": ["cur", "rate"]},
            {"event": "probe_mage", "lane": "L1", "u": "own_goldwiz_1", "active": ["sd"], "passive": [], "can": {"sd": True},
             "cco": {}},
            {"event": "probe_ability", "lane": "L1", "who": "a", "key": "sd", "tg": "enemy_clanrat_1", "n": 1, "t": 10000,
             "status": "given"}]
    for k in range(0, 41):
        t = k / 2
        hp = 9600 - 600 * min(max(t - 12, 0), 4) / 4
        men = 160 - round(10 * min(max(t - 12, 0), 4) / 4)
        rows.append({"event": "probe_sample", "lanes": [{"lane": "L1", "t": int(t * 1000), "a": {"hp": 3420, "men": 1},
                                                          "tg": {"hp": hp, "men": men}}]})
    for s in range(0, 21):
        cur1 = 12 + 0.5 * s - (6 if s >= 11 else 0)
        rows.append({"event": "probe_winds", "t": s * 1000, "s1": {"cur": cur1, "rate": 0.5},
                     "s2": {"cur": 10 + 0.5 * s, "rate": 0.5}})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_measure_finds_the_cast_by_the_winds_drop_and_the_damage_after_it(tmp_path):
    doc = mp.measure(_run(tmp_path))
    (c,) = doc["casts"]
    assert (c["lane"], c["side"], c["try_s"], c["drop_s"], c["cost"]) == ("L1", 1, 10.0, 11.0, 5.5)
    assert c["hp0"] == 9600 and c["men0"] == 160
    assert c["windows"]["0-2"] == (0, 0) and c["windows"]["2-4"] == (300, 5) and c["windows"]["4-6"] == (300, 5)
    assert doc["sides"][2]["casts"] == [] and doc["sides"][2]["winds"]["flow"] == pytest.approx(0.5)
    assert doc["sides"][1]["winds"]["flow"] == pytest.approx(0.5 - 0.5 / 20)          # the cost: 6 - 0.5 seen
    assert doc["sides"][1]["api"] == ["cur", "rate"] and doc["mages"]["L1"]["can"] == {"sd": True}
