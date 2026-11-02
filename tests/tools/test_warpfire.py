"""The Warpfire Throwers' passport, data and the probe's plan (02.11.2026; build/diag_roster/wh2_main_skv_inf_warpfire_thrower/units.md,
mechanic.md): the class 'shells without guns' by data alone. No torch (the simulator's numbers: test_sim_warpfire_torch.py)."""
import json

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn.armies import pools
from tools.nn.companion import exchange
from tools.nn.sim import params as sim_params

W = "wh2_main_skv_inf_warpfire_thrower"
RG = "wh2_dlc12_skv_inf_ratling_gun_0"
FLAMES = "wh_main_unit_contact_morale"
PROJ = "wh2_main_skv_warpfire"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][W]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"]) == (32, 123, 3936, 70, 550, 50)
    m = p["missile"]
    assert (m["projectile"], m["ammo"], m["category"], m["range_m"], m["reload_s"], m["damage"], m["ap_damage"]) == (PROJ, 72, "musket", 80, 13.0, 8, 3)
    assert (m["burst_size"], m["shots_per_volley"], m["projectile_number"], m["gravity"], m["expiry_m"]) == (4, 1, 1, 3.0, 85)
    assert m["burst_delay_s"] == 0.35 and m["muzzle_velocity"] == 30.0 and m["flaming"] and m["magic"] and not m["is_spell"]
    assert m["penetration"] == "low" and m["contact_phase"] == FLAMES and m["trajectory"] == "low"
    b = m["blast"]
    assert (b["type"], b["radius_m"], b["damage"], b["ap_damage"], b["force"], b["speed"]) == ("shockwave", 3.0, 20.0, 7.0, 40.0, 14.0)
    assert b["allies"] and b["magic"] and b["flaming"] and not b["is_spell"] and b["contact_phase"] is None
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][W] == p
    assert doc["contact_phases"][FLAMES]["duration_s"] == 10
    assert doc["contact_phases"][FLAMES]["stats"] == [{"stat": "stat_morale", "how": "add", "value": -8.0}]


def test_the_body_is_the_ratling_guns():
    u = load("units.json")["units"]
    for k in ("mass", "radius_m", "spacing", "speed", "armour", "damage_resist", "hiding_scalar", "spotting", "abilities", "men", "hp_per_man"):
        assert u[W][k] == u[RG][k], k


def test_every_other_unit_keeps_gravity_and_burst_of_one():
    # (the old units' passports hold the new fields; the sim reads gravity only for the class: the Warp Lightning Cannon's 6 is not read)
    u = load("units.json")["units"]
    for k, p in u.items():
        m = p.get("missile")
        if not m or k == W:
            continue
        assert m["burst_size"] == 1 and m["projectile_number"] == 1 and not m["flaming"], k
        assert m["gravity"] == (6.0 if k == "wh2_main_skv_art_warp_lightning_cannon" else 9.81), k


def test_the_class_is_by_data_and_the_numbers_with_their_sources():
    p = sim_params.load()
    assert sim_params.is_round(p.units[W], p.sim)
    # nobody else of the pool: a gun with a blast is no such shell, a shooter without a blast neither
    assert [k for k, u in p.units.items() if sim_params.is_round(u, p.sim)] == [W]
    off = p.with_cal("artillery", rounds={"on": 0})
    assert not sim_params.is_round(off.units[W], off.sim)
    ms = p.sim["missile"]
    assert ms["burst_rate_s"][PROJ] == 1.42 and ms["reload_projectile_s"][PROJ] == 17.2 and ms["aim_projectile_s"][PROJ] == 3.6
    assert PROJ in ms["fire_in_melee"]           # (the walk lane: 67 shells with the melee flag on)
    assert p.sim["artillery"]["rounds"]["lie_s"] == 5.8 and p.sim["artillery"]["rounds"]["fling"] == {"front": 0.15, "depth": 0.24}
    assert p.sim["terrain"]["trees"]["ignore_s"][PROJ] == 1.4
    assert p.sim["artillery"]["rounds"]["on"] and "20261101-050948" in p.sim["artillery"]["rounds"]["why"]


def test_the_static_fields():
    p = sim_params.load()
    s = p.static(W, "wh2_main_skv_skaven")
    assert s["spv"] == 4.0 and s["bst_rate"] == 1.42 and s["reload"] == 17.2 and s["ammo0"] == 2304 and not s["bst_cut"]
    assert s["aim_s"] == 3.6 and s["melee_fire"]
    assert not s["hslow"]                  # (its knock is its own rule: artillery.lie)
    assert s["eng0"] == 0.0 and s["rd"] and s["ex_r"] == 3.0 and s["m_g"] == 3.0 and s["exp_m"] == 85.0
    assert s["p_fire"] and s["ex_fire"] and s["p_magic"] and s["ex_magic"] and not s["p_spell"] and not s["ex_spell"]
    assert not s["cat_art"]                # (category musket: no artillery morale, the record)
    assert s["veg_m"] == 1.4 * 30.0
    r = p.static(RG, "wh2_main_skv_skaven")
    assert r["spv"] == 18.0 and r["bst_cut"] and not r["rd"] and r["m_g"] == 9.81 and r["exp_m"] == 0.0 and r["hslow"]
    for k in ("wh_main_emp_art_mortar", "wh2_main_skv_art_warp_lightning_cannon", "wh_main_emp_art_helblaster_volley_gun"):
        g = p.static(k, "wh_main_emp_empire")
        assert g["cat_art"] and not g["rd"] and g["m_g"] == 9.81 and not g["ex_fire"] and g["exp_m"] == 0.0, k
    assert not p.static("wh_main_emp_inf_handgunners", "wh_main_emp_empire")["cat_art"]


def test_a_stormfiends_burst_is_eight_shells_and_needs_only_data():
    # mechanic.md 4: projectile_number 2 x burst_size 4; no code of its own
    p = sim_params.load()
    u = json.loads(json.dumps(p.units[W]))
    u["missile"].update(projectile_number=2, gravity=3.0, expiry_m=85, range_m=90)
    q = sim_params.Params({**p.units, "x": u}, p.rules, p.sim, p.abilities, p.effects, p.spells)
    assert q.static("x", "wh2_main_skv_skaven")["spv"] == 8.0


def test_a_globadier_keeps_the_artillery_morale_by_its_category():
    p = sim_params.load()
    u = json.loads(json.dumps(p.units[W]))
    u["missile"].update(category="artillery", gravity=6.5, trajectory="dual_low_fixed", burst_size=1)
    q = sim_params.Params({**p.units, "x": u}, p.rules, p.sim, p.abilities, p.effects, p.spells)
    s = q.static("x", "wh2_main_skv_skaven")
    assert s["rd"] and s["cat_art"] and s["m_g"] == 6.5 and s["spv"] == 1.0


def test_the_flames_are_a_contact_on_the_catalogues_end():
    doc = load("effects.json")
    assert doc["order"][-1] == FLAMES and doc["inflicts"][W] == [FLAMES]
    e = doc["effects"][FLAMES]
    assert e["modelled"] and e["duration_s"] == 10.0 and e["stats"][0]["stat"] == "stat_morale" and e["stats"][0]["value"] == -8.0
    assert FLAMES not in doc["units"][W]


def test_the_throwers_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert (W in keys) is has, scheme
    groups = load("army_templates.json")["factions"]["wh2_main_skv_skaven"]["units"][W]
    assert groups == [{"group": "wh3_default_land_ranged_infantry_main_heavy_short_range", "quality": "tier_2_step_1_units"}]


def test_the_probes_plan():
    b = cp.battles("warpfire")
    assert len(b) == 1 and [ln["kind"] for ln in b[0]] == ["near", "far", "walk"]
    assert all(ln["attacker"] == "warpfire" and ln["target"] == "swords" and ln["mode"] == "shoot" for ln in b[0])
    assert [ln["target_mode"] for ln in b[0]] == ["hold", "hold", "both_walk"]
    assert [ln["gap_m"] for ln in b[0]] == [36, 61, 81]
    assert cp.UNITS["warpfire"][0] == W
    cfg = cp.run_config("warpfire", 1)[0]
    assert cfg["mem_proj"] and cfg["men_ms"] == 1000


def test_the_card_shows_the_flames_and_the_companion_sets_its_bit():
    from tools.nn.model import effects as mfx
    row = {"side": 1, "x": 0.0, "z": 0.0, "b": 0, "men": 100, "hp": 0.9, "mp": 0.8, "ms": 3, "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False,
           "a": 0, "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": 0.0, "oz": 0.0, "lf": False, "rf": False, "bf": False, "v": True}
    doc = {"batch": "b-1", "move": 1, "t": 5000, "done": False, "attacker": 2, "decide_ms": 1000,
           "factions": {"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"},
           "units": [dict(row, n="own_sw", key="wh_main_emp_inf_swordsmen", fx=[FLAMES]),
                     dict(row, n="enemy_w", side=2, key=W, z=45.0, oz=45.0, b=180, men=32, a=2304, fire=True, fx=[])]}
    b = exchange.battle(doc)
    s = exchange.arrays(doc, b.names)
    exchange.effects_on(s, doc, b.names, b.setup)
    i = mfx.keys().index(FLAMES)
    assert (int(s["fx_on"][0, 0]) >> i) & 1 == 1 and (int(s["fx_on"][0, 1]) >> i) & 1 == 0
    assert mfx.owned([W, "wh_main_emp_inf_swordsmen"])[0, i] == 1 and mfx.owned([W, "wh_main_emp_inf_swordsmen"])[1, i] == 0
