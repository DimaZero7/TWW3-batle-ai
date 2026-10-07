"""tools/nn/army_edit.py and tools.build nn-arena --army-add/--army-remove: our side's generated army
changed by hand, deployed again by the generator's rule. Levels 1-2: no game."""
import re
from collections import Counter

import pytest

from tools import build
from tools import config as project
from tools.nn import army_edit
from tools.nn import scenario as nn_scenario
from tools.nn.armies import generate

SEED = 1_000_900_014                     # Skaven (own) against the Empire (enemy)
ARCHERS, SPEARMEN = "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_spearmen_0"


def keys(units):
    return Counter(u["key"] for u in units)


def test_parse():
    assert army_edit.parse("x") == ("x", 1) and army_edit.parse("x=3") == ("x", 3)
    for bad in ("x=0", "=2", "x=a", "x=-1"):
        with pytest.raises(ValueError):
            army_edit.parse(bad)


def test_no_change_is_the_generators_army():
    arena = generate.battle(SEED)
    edited = army_edit.edit(arena, "enemy")
    assert edited["sides"]["enemy"]["units"] == arena["sides"]["enemy"]["units"]
    assert edited["sides"]["enemy"]["cost"] == arena["sides"]["enemy"]["cost"]
    assert edited["sides"]["own"] is arena["sides"]["own"] and edited["name"] == f"random_{SEED}_edit"


def test_added_archers_stand_with_the_shooters_and_the_cost_follows():
    arena = generate.battle(SEED)
    before = arena["sides"]["enemy"]
    edited = army_edit.edit(arena, "enemy", add=[(ARCHERS, 3)], remove=[(SPEARMEN, 1)])["sides"]["enemy"]
    assert keys(edited["units"]) == keys(before["units"]) + Counter({ARCHERS: 3}) - Counter({SPEARMEN: 1})
    assert edited["cost"] == before["cost"] + 3 * 350 - 300 and edited["budget"] == before["budget"]
    melee = [u["forward"] for u in edited["units"] if not u.get("general") and u["key"] not in
             (ARCHERS, "wh_dlc04_emp_inf_free_company_militia_0")]
    assert all(u["forward"] < min(melee) for u in edited["units"] if u["key"] == ARCHERS)
    assert edited["edit"] == {"add": {ARCHERS: 3}, "remove": {SPEARMEN: 1}}
    with pytest.raises(ValueError):
        army_edit.edit(arena, "enemy", remove=[(SPEARMEN, 3)])          # the army has 2
    with pytest.raises(ValueError):
        army_edit.edit(arena, "enemy", add=[("wh2_main_skv_inf_clanrats_1", 1)])   # not an Empire unit
    with pytest.raises(ValueError):
        army_edit.edit(arena, "own", add=[("wh2_main_skv_inf_clanrats_1", 1)])     # 19 units already


def test_the_build_edits_our_side_after_the_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "BUILD", tmp_path)
    monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
    written = []
    monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append((config, scenario)) or {})
    assert build.main(["nn-arena", "--army-seed", str(SEED), "--army-swap", "--army-add", f"{ARCHERS}=3",
                       "--army-remove", SPEARMEN, "--timeout", "3600"]) == 0
    (config, scenario), = written
    arena = generate.battle(SEED)
    assert config["arena"] == f"random_{SEED}_swap_edit" and scenario.endswith(f"random_{SEED}_swap_edit.xml")
    assert config["own_role"] == "defend" and config["army"]["swap"] is True
    assert config["army"]["edit"] == {"add": {ARCHERS: 3}, "remove": {SPEARMEN: 1}}
    assert keys(config["units"]["own"]) == keys(arena["sides"]["enemy"]["units"]) + Counter({ARCHERS: 3}) \
        - Counter({SPEARMEN: 1})
    assert keys(config["units"]["enemy"]) == keys(arena["sides"]["own"]["units"])
    # the battle file: alliance 0 is ours (the Empire, 16 with the lord), alliance 1 the Skaven
    xml = (tmp_path / "nn-arena" / f"random_{SEED}_swap_edit.xml").read_text(encoding="utf-8")
    own, enemy = xml.split("<alliance ")[1:]
    assert "wh_main_emp_empire" in own and Counter(re.findall(r'unit_type type="([^"]+)"', own)) \
        == keys(config["units"]["own"])
    assert len(re.findall(r"<unit ", enemy)) == 20
    for bad in (["--army-add", ARCHERS], ["--army-seed", str(SEED), "--army-add", f"{ARCHERS}=0"]):
        with pytest.raises(SystemExit):
            build.main(["nn-arena", *bad])
