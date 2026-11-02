"""The Death Runners' passport, ability passport, contact effect, pools and the companion's count of uses (02.11.2026;
build/diag_roster/wh2_main_skv_inf_death_runners_0/units.md, mechanic.md): 72 men of 80 HP, the first hiding melee unit, the weeping
blade (armour x0.5 for 35 s) and Concealment Bombs (stalk + unspottable, 73 s, recharge 90 s, 3 uses). No torch (the simulator's
rules: test_sim_runners_torch.py)."""
import json

import numpy as np
import pytest

from tools import config as project
from tools.nn.armies import pools
from tools.nn.companion import exchange
from tools.nn.sim import abilities as sab

DR = "wh2_main_skv_inf_death_runners_0"
BOMBS = "wh2_main_unit_abilities_concealment_bombs"
BLADE = "wh2_main_unit_contact_weeping_blade"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][DR]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"]) == (72, 80, 5760, 20,
                                                                                                                  850, 68)
    m = p["melee"]
    assert (m["attack"], m["defence"], m["charge_bonus"], m["damage"], m["ap_damage"], m["bonus_v_infantry"]) == (32, 28, 22, 8,
                                                                                                                    22, 10)
    assert m["contact_phase"] == BLADE and p["missile"] is None
    assert p["damage_resist"]["physical"] == 30 and p["hiding_scalar"] == 0.8 and p["spotting"]["scrub"] == 80
    assert {"stalk", "strider", "guerrilla_deploy"} == set(p["attributes"])
    assert BOMBS in p["abilities"]
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][DR] == p and probe["contact_phases"][BLADE] == doc["contact_phases"][BLADE]


def test_the_bombs_passport_active_three_uses_no_stats_two_attributes():
    p = load("abilities.json")["abilities"][BOMBS]
    assert (p["active_s"], p["recharge_s"], p["uses"], p["initial_s"]) == (73.0, 90.0, 3, 0.0)
    assert p["self_cast"] and not p["passive"] and not p["auto"] and p["effects"] == []
    assert {a["attribute"] for a in p["attributes"]} == {"stalk", "unspottable"}
    assert sab.rules_of(p) == ["stalk", "unspottable"]


def test_rules_of_only_on_the_owner_and_known_rules():
    p = {"attributes": [{"attribute": "unbreakable", "on": ["friends"]}, {"attribute": "stalk", "on": ["self"]},
                        {"attribute": "no_such_rule", "on": ["self"]}, {"attribute": "charge_defense", "on": ["self"]}]}
    assert sab.rules_of(p) == ["stalk", "charge_defence"]


def test_only_the_bombs_have_limited_uses_or_attributes_among_the_actives():
    # (old armies stay bit for bit: nothing else fires these rules)
    ab = load("abilities.json")["abilities"]
    assert [k for k, p in ab.items() if p["uses"] != -1] == [BOMBS]
    assert [k for k, p in ab.items() if p["attributes"] and not p["passive"] and not p["auto"]] == [BOMBS]


def test_the_blade_is_a_modelled_contact_effect_last_in_the_order():
    doc = load("effects.json")
    e = doc["effects"][BLADE]
    assert e["kind"] == "contact" and e["modelled"] and e["duration_s"] == 35.0
    assert [(s["stat"], s["how"], s["value"], s["sim"]) for s in e["stats"]] == [("stat_armour", "mult", 0.5, "armour")]
    # (the Warpfire Throwers' 'Flames!' after it, 02.11.2026)
    assert doc["order"][-2:] == [BLADE, "wh_main_unit_contact_morale"] and doc["inflicts"][DR] == [BLADE]
    assert sum(1 for x in doc["effects"].values() if x["kind"] == "contact") == 6


def test_the_runners_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert (DR in keys) is has, scheme
    groups = load("army_templates.json")["factions"]["wh2_main_skv_skaven"]["units"][DR]
    assert groups and groups[0]["group"] == "wh3_default_land_melee_infantry_main_flanking"


def test_the_stealth_number_is_the_database():
    assert load("sim.json")["stealth"]["unspottable_m"] == load("game_rules.json")["battle"]["unspottable_unit_reveal_distance"] == 20


def battle_doc(t_ms, used, counted=None, fx=None):
    row = {"n": "own_dr", "key": DR, "side": 1, "x": 0, "z": 0}
    if fx is not None:
        row["fx"] = fx
    return {"t": t_ms, "units": [row], "abilities_used": {"own_dr": {BOMBS: used}} if used is not None else {},
            "abilities_count": {"own_dr": {BOMBS: counted}} if counted else {}}


def timers(doc):
    return exchange.ability_timers(doc, ["own_dr"], [[BOMBS]])


def test_the_companions_timers_run_while_uses_are_left():
    s = timers(battle_doc(100000, 90000, 2, fx=[BOMBS]))       # used 10 s ago, the second time: 63 s on, 153 s to ready
    assert s["ab0_on"][0, 0] == pytest.approx(63) and s["ab0_cd"][0, 0] == pytest.approx(153)
    s = timers(battle_doc(300000, 90000, 2, fx=[]))            # 210 s ago, two used: ready again
    assert s["ab0_on"][0, 0] == 0 and s["ab0_cd"][0, 0] == 0


def test_the_companions_timers_freeze_with_no_use_left():
    s = timers(battle_doc(100000, 90000, 3, fx=[BOMBS]))       # the third use: on as usual
    assert s["ab0_on"][0, 0] == pytest.approx(63) and s["ab0_cd"][0, 0] == pytest.approx(153)
    s = timers(battle_doc(1000000, 90000, 3, fx=[]))           # long after: the recharge does not run out (the simulator's)
    assert s["ab0_on"][0, 0] == 0 and s["ab0_cd"][0, 0] == pytest.approx(90)
    assert np.all(s["ab0_cd"] > 0)


def test_the_probe_plan_has_its_lanes():
    from tools.nn import morale_probe as mp
    b = mp.battles("runners")
    assert [[ln["kind"] for ln in x] for x in b] == [["plain", "bomb", "bomb_walk"], ["blade_gs", "blade_leave", "uses"],
                                                     ["arrow_ctrl", "arrow_blade", "melee_sw"]]
    assert all(len(x) <= 3 for x in b)
    # a lane: the bombs key where pressed, every step's roles known
    uses = next(ln for ln in b[1] if ln["kind"] == "uses")
    assert [s["delay_s"] for s in uses["steps"] if s["do"] == "ability"] == [0, 170, 340, 510]
    assert all(s["key"] == BOMBS for s in uses["steps"] if s["do"] == "ability")
    layout = mp.layout(b[0], "runners")
    assert {u["key"] for ln in layout[0] for u in ln["units"]} == {DR, "wh_main_emp_inf_swordsmen"}


def test_the_card_shows_the_blade_and_the_companion_sets_its_bit():
    from tools.nn.model import effects as mfx
    doc = {"batch": "b-1", "move": 1, "t": 5000, "done": False, "attacker": 2, "decide_ms": 1000,
           "factions": {"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"},
           "units": [{"n": "own_sw", "side": 1, "key": "wh_main_emp_inf_swordsmen", "x": 0.0, "z": 0.0, "b": 0, "men": 100, "hp": 0.9,
                      "mp": 0.8, "ms": 3, "r": False, "s": False, "w": False, "m": True, "mv": False, "f": False, "a": 0,
                      "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": 0.0, "oz": 0.0, "lf": False, "rf": False,
                      "bf": False, "v": True, "fx": [BLADE]},
                     {"n": "enemy_dr", "side": 2, "key": DR, "x": 0.0, "z": 3.0, "b": 180, "men": 70, "hp": 0.9, "mp": 0.8, "ms": 3,
                      "r": False, "s": False, "w": False, "m": True, "mv": False, "f": False, "a": 0, "fire": False, "t": "",
                      "fat": "threshold_fresh", "k": 0, "ox": 0.0, "oz": 3.0, "lf": False, "rf": False, "bf": False, "v": True,
                      "fx": []}]}
    b = exchange.battle(doc)
    s = exchange.arrays(doc, b.names)
    exchange.effects_on(s, doc, b.names, b.setup)
    i = mfx.keys().index(BLADE)
    assert (int(s["fx_on"][0, 0]) >> i) & 1 == 1 and (int(s["fx_on"][0, 1]) >> i) & 1 == 0
    # who lays it: both know (the unit card), the Runners' 'owned' column of the effect
    assert mfx.owned([DR, "wh_main_emp_inf_swordsmen"])[0, i] == 1 and mfx.owned([DR, "wh_main_emp_inf_swordsmen"])[1, i] == 0


def test_the_blades_armour_is_the_tokens_number_and_its_new_columns_go_last():
    from tools.nn.model import effects as mfx
    from tools.nn.model import observation as ob
    assert ob.TAIL[-2:] == (ob.FX_STATS4, ob.FX_LATE) and ob.FX_STATS4 == ("fxs_stat_armour_mult",)
    # (the Warpfire Throwers' 'Flames!' after the blade, 02.11.2026: +2 columns)
    flames = "wh_main_unit_contact_morale"
    assert ob.FX_LATE == (f"fx_{BLADE}_owned", f"fx_{BLADE}_on", f"fx_{flames}_owned", f"fx_{flames}_on")
    assert ob.NAMES[-5:] == ob.FX_STATS4 + ob.FX_LATE
    # a token of the Warp Grinders' width (the last checkpoints) stays a beginning of today's
    old = ob.TOKEN - 5
    assert ob.old_columns(old) == list(range(old))
    on = np.zeros((1, 1, len(mfx.keys())), bool)
    on[0, 0, mfx.keys().index(BLADE)] = True
    val = mfx.stats_on(np, on)
    assert val[0, 0, mfx.FX_ALL.index(("stat_armour", "mult"))] == pytest.approx(-0.5)
