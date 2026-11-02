"""The Death Globe Bombardiers' passport, data and the probe's plan (02.11.2026; build/diag_roster/wh2_main_skv_inf_death_globe_bombardiers/units.md, mechanic.md, build/diag_bombardiers/): the class
'shells without guns' by data alone, no contact effect, the 4 m blast's numbers. No torch (the simulator's numbers: test_sim_bombardiers_torch.py)."""
import json

import pytest

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn.sim import params as sim_params

B = "wh2_main_skv_inf_death_globe_bombardiers"
G = "wh2_main_skv_inf_poison_wind_globadiers"
PROJ = "wh2_main_skv_death_globe"
SKV = "wh2_main_skv_skaven"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][B]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"], p["mass"]) == (48, 76, 3648, 100, 900, 58, 150.0)
    m = p["missile"]
    assert (m["projectile"], m["ammo"], m["category"], m["range_m"], m["reload_s"], m["damage"], m["ap_damage"]) == (PROJ, 18, "artillery", 85, 10.0, 6, 16)
    assert (m["trajectory"], m["fixed_elevation"], m["gravity"], m["expiry_m"], m["muzzle_velocity"], m["fire_arc_deg"]) == ("dual_low_fixed", 30, 6.5, 120, 30.0, 30.0)
    assert m["magic"] and not m["is_spell"] and not m["flaming"] and m["contact_phase"] is None
    b = m["blast"]
    assert (b["type"], b["radius_m"], b["damage"], b["ap_damage"], b["force"], b["speed"]) == ("shockwave", 4.0, 9.0, 33.0, 65.0, 15.0)
    assert b["allies"] and b["magic"] and not b["is_spell"] and b["contact_phase"] is None
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][B] == p


def test_the_catalogue_of_effects_does_not_grow():
    """No contact: the catalogue's order stays, it owns only what every Skaven of its kind owns."""
    fx = load("effects.json")
    assert B not in fx["inflicts"]
    assert fx["order"][-1] == "wh2_dlc14_unit_contact_poisoned_wind"
    assert fx["units"][B] == fx["units"][G]


def test_the_class_and_the_numbers_with_their_sources():
    p = sim_params.load()
    assert sim_params.is_round(p.units[B], p.sim)
    a, ms = p.sim["artillery"], p.sim["missile"]
    assert a["volley_land_s"][PROJ] == 2.4 and "1.67" in a["volley_land_why"]
    assert ms["reload_projectile_s"][PROJ] == 11.6 and "11.6" in ms["reload_projectile_why"] and ms["aim_projectile_s"][PROJ] == 2.0
    assert "Death Globe" in ms["aim_projectile_why"] and "Death Globe" in a["rounds"]["ring_why"]
    assert p.sim["terrain"]["trees"]["ignore_s"][PROJ] == 1.4


def test_the_static_fields():
    p = sim_params.load()
    s = p.static(B, SKV)
    assert s["rd"] and s["eng0"] == 0.0 and s["spv"] == 1.0 and s["bst_rate"] == 0.0 and s["cat_art"]
    assert s["lob_el"] == 30.0 and s["land_w"] == 2.4 and s["fix_el"] == 0.0
    assert s["m_g"] == 6.5 and s["exp_m"] == 120.0 and s["ex_r"] == 4.0 and s["ex_v"] == 15.0 and s["ex_f"] == 65.0 and s["ammo0"] == 864
    assert s["reload"] == 11.6 and s["aim_s"] == 2.0 and s["p_magic"] and s["ex_magic"]
    assert s["ex_fx"] == -1 and s["sfx"] == -1


def test_the_force_throws_a_man_to_176_kg_and_not_a_knight():
    """(1 + e) F S / (F + m), the game's shockwave thresholds: the Bombardiers' 65 and 15 m/s (the Globadiers' 30 knocks and never throws)."""
    rules = load("game_rules.json")["battle"]
    e, thr_fly, thr_down = rules["collision_restitution_coefficient"], rules["shockwave_speed_delta_threshold_knocked_flying"], rules["shockwave_speed_delta_threshold_knocked_down"]
    dv = lambda m, f=65.0: (1 + e) * f * 15.0 / (f + m)
    assert [round(dv(m), 2) for m in (100, 150, 1000)] == [7.68, 5.90, 1.19]
    assert dv(176) > thr_fly > dv(177)
    assert dv(1000) < thr_down and dv(100, 30.0) < thr_fly


def test_only_the_v3_scheme_fields_them():
    doc = load("pools.json")
    skv = doc["factions"][SKV]
    assert B in json.dumps(skv["units_v3"]) and B not in json.dumps({k: v for k, v in skv.items() if k != "units_v3"})
    groups = load("army_templates.json")["factions"]
    rows = [f["units"][B] for f in groups.values() if B in f["units"]]
    assert rows and rows[0] == [{"group": "wh3_default_land_ranged_infantry_main_heavy_short_range", "quality": "tier_2_step_5_units"}]


def test_the_probes_plan():
    b = cp.battles("dglobes")[0]
    assert [x["kind"] for x in b] == ["near", "knights", "lob"]
    assert all(x["attacker"] == "dglobes" and x["mode"] == "shoot" for x in b)
    assert b[2].get("t_morale") and not b[0].get("t_morale") and not b[1].get("t_morale")
    # the friends: their rank before the shooters, a metre from their front (the layout: from the target's place, the attacker at +z)
    a_d, t_d = cp.depth(*cp.UNITS["dglobes"][:2], 30), cp.depth(*cp.UNITS["spear"][:2], 30)
    e_d = cp.depth(*cp.UNITS["cspear"][:2], 30)
    assert b[2]["extras"][0]["dz"] + e_d / 2 == pytest.approx(t_d / 2 + b[2]["gap_m"] - 1.0, abs=0.1)
    cfg = cp.run_config("dglobes", 1)[0]
    assert cfg["mem_proj"] and cfg["mem_extras"] and cfg["mem_morale"]


def test_the_card_needs_nothing_new():
    """No effect on the target and no column: the companion reads the same fields; the ammunition of the passport."""
    from tools.nn.companion import exchange
    row = {"side": 1, "x": 0.0, "z": 0.0, "b": 0, "men": 100, "hp": 0.9, "mp": 0.8, "ms": 3, "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False,
           "a": 0, "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": 0.0, "oz": 0.0, "lf": False, "rf": False, "bf": False, "v": True}
    doc = {"batch": "b-1", "move": 1, "t": 5000, "done": False, "attacker": 2, "decide_ms": 1000,
           "factions": {"own": "wh_main_emp_empire", "enemy": SKV},
           "units": [dict(row, n="own_sp", key="wh_main_emp_inf_spearmen_0", fx=[]),
                     dict(row, n="enemy_b", side=2, key=B, z=60.0, oz=60.0, b=180, men=48, a=816, fire=True, fx=[])]}
    b = exchange.battle(doc)
    assert "enemy_b" in b.names


def test_the_drills_leave_them_out():
    from pathlib import Path
    src = Path(project.ROOT / "tools" / "nn" / "unit" / "drill_hold.py").read_text(encoding="utf-8")
    assert B in src.split("NOT_IN_DRILLS = (", 1)[1].split(")", 1)[0]
