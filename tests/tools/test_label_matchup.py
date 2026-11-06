"""Разметка's matchup and pool (tools/label/matchup.py, pool.py; no torch): the checks, the army description, the slots'
order, the json save / load."""
import copy

from tools.label import matchup as MU
from tools.label import pool as PL

M = {"name": "проверка 1", "map": "flat", "limit_s": 120, "command": {"verb": "hold", "pace": "any"}, "enemy": "attack",
     "sides": {"1": {"faction": "wh_main_emp_empire", "units": [
         {"key": "wh_main_emp_inf_spearmen_0", "x": -60, "z": -300, "b": 0},
         {"key": "wh_main_emp_cha_general_0", "x": 0, "z": -400, "b": 0}]},
         "2": {"faction": "wh2_main_skv_skaven", "units": [
             {"key": "wh2_main_skv_inf_clanrats_0", "x": 70, "z": 300, "b": 180}]}}}


def test_pool_has_both_factions_with_lords_first():
    f = PL.factions()
    assert {"wh_main_emp_empire", "wh2_main_skv_skaven"} <= set(f)
    for fac in f.values():
        assert fac["units"][0]["general"] and len({u["key"] for u in fac["units"]}) == len(fac["units"])
    assert PL.maps()[0]["name"] == "flat"


def test_a_correct_matchup_has_no_errors():
    assert MU.check(M) == []


def test_errors_are_named():
    m = copy.deepcopy(M)
    m["sides"]["1"]["units"][0]["z"] = 300                 # in the enemy's strip
    m["sides"]["2"]["units"].append({"key": "wh_main_emp_inf_spearmen_0", "x": 0, "z": 300, "b": 180})  # not skaven
    m["command"]["verb"] = "dance"
    m["enemy"] = "fly"
    e = " | ".join(MU.check(m))
    assert "deployment strip" in e and "not in the pool" in e and "verb" in e and "enemy" in e
    m = copy.deepcopy(M)
    m["sides"]["1"]["units"].append(dict(m["sides"]["1"]["units"][1]))
    assert any("generals" in x for x in MU.check(m))


def test_army_and_slots():
    a = MU.army(M)
    assert a["attacker"] == 2 and a["sides"][1]["units"][0]["width"] == 30
    assert a["sides"][1]["units"][1]["general"]
    slots, H = MU.order(M)
    assert H == 2 and slots == [(1, 1), (1, 0), (2, 0), None]      # the general first (scenario.build's rule)


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(MU, "folder", lambda: tmp_path)
    name = MU.save(M)
    assert name == "проверка_1" and MU.names() == [name]
    assert MU.load(name)["sides"] == M["sides"]
