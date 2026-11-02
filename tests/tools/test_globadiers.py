"""The Poisoned Wind Globadiers' passport, data and the probe's plan (02.11.2026; build/diag_roster/wh2_main_skv_inf_poison_wind_globadiers/units.md, mechanic.md, build/diag_globadiers/): the class 'shells
without guns' by data, the poison a contact effect of direct damage. No torch (the simulator's numbers: test_sim_globadiers_torch.py)."""
import copy
import json

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn import effects as cat
from tools.nn.sim import effects as innate
from tools.nn.sim import params as sim_params

G = "wh2_main_skv_inf_poison_wind_globadiers"
W = "wh2_main_skv_inf_warpfire_thrower"
POISON = "wh2_dlc14_unit_contact_poisoned_wind"
PROJ = "wh2_main_skv_poison_wind_globe"
SKV = "wh2_main_skv_skaven"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][G]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"], p["mass"]) == (48, 76, 3648, 100, 650, 58, 150.0)
    m = p["missile"]
    assert (m["projectile"], m["ammo"], m["category"], m["range_m"], m["reload_s"], m["damage"], m["ap_damage"]) == (PROJ, 18, "artillery", 85, 12.0, 4, 12)
    assert (m["trajectory"], m["fixed_elevation"], m["gravity"], m["expiry_m"], m["muzzle_velocity"]) == ("dual_low_fixed", 30.0, 6.5, 120, 30.0)
    assert m["magic"] and not m["is_spell"] and not m["flaming"] and m["contact_phase"] == POISON
    b = m["blast"]
    assert (b["type"], b["radius_m"], b["damage"], b["ap_damage"], b["force"], b["speed"]) == ("shockwave", 2.0, 12.0, 16.0, 30.0, 15.0)
    assert b["allies"] and b["magic"] and not b["is_spell"] and b["contact_phase"] == POISON
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][G] == p


def test_the_poison_phase_is_the_database():
    ph = load("units.json")["contact_phases"]
    assert ph[POISON] == {"duration_s": 12.0, "effect": "negative", "stats": [], "allies": True,
                          "direct": {"amount": 4, "entities": 25, "every_s": 4.0, "scale": 1.0, "min_unary": 0.5, "hit_hp": 2.5}}
    # (the other phases: not on friends, no direct damage)
    for k, v in ph.items():
        if k != POISON:
            assert not v["allies"] and "direct" not in v, k


def test_the_catalogue_reads_the_blasts_contact_and_keeps_its_place_last():
    doc = cat.load()
    assert doc["order"][-1] == POISON and doc["inflicts"][G] == [POISON]
    e = doc["effects"][POISON]
    assert e["kind"] == "contact" and e["modelled"] and e["allies"] and e["direct"]["entities"] == 25 and e["stats"] == []
    assert cat.inflicted({"missile": {"contact_phase": None, "blast": {"contact_phase": POISON}}}) == [POISON]


def test_the_class_and_the_numbers_with_their_sources():
    p = sim_params.load()
    assert sim_params.is_round(p.units[G], p.sim)
    a, ms = p.sim["artillery"], p.sim["missile"]
    assert a["volley_land_s"][PROJ] == 2.4 and "1.37" in a["volley_land_why"]
    assert a["rounds"]["rank_stop"] == 0.49 and "343" in a["rounds"]["ring_why"]
    assert ms["reload_projectile_s"][PROJ] == 13.6 and ms["aim_projectile_s"][PROJ] == 2.0
    assert p.sim["terrain"]["trees"]["ignore_s"][PROJ] == 1.4


def test_the_static_fields():
    p = sim_params.load()
    s = p.static(G, SKV)
    assert s["rd"] and s["eng0"] == 0.0 and s["spv"] == 1.0 and s["bst_rate"] == 0.0 and s["cat_art"]
    assert s["lob_el"] == 30.0 and s["land_w"] == 2.4 and s["fix_el"] == 0.0
    assert s["m_g"] == 6.5 and s["exp_m"] == 120.0 and s["ex_r"] == 2.0 and s["ex_v"] == 15.0 and s["ammo0"] == 864
    assert s["reload"] == 13.6 and s["aim_s"] == 2.0 and s["p_magic"] and s["ex_magic"]
    order = p.effects["order"]
    # (the blast lays the poison, its own hit does not: its count is at the launch, 2-3 s before the shell lands)
    assert s["ex_fx"] == order.index(POISON) and s["sfx"] == -1


def test_the_warpfire_throwers_keep_their_contact_and_no_window():
    p = sim_params.load()
    s = p.static(W, SKV)
    assert s["land_w"] == 0.0 and s["lob_el"] == 0.0 and s["sfx"] == p.effects["order"].index("wh_main_unit_contact_morale") and s["ex_fx"] == -1


def test_an_old_unit_has_neither_field():
    p = sim_params.load()
    for k in ("wh2_main_skv_inf_gutter_runners_0", "wh_main_emp_art_mortar", "wh2_main_skv_art_plagueclaw_catapult"):
        s = p.static(k, SKV if k.startswith("wh2") else "wh_main_emp_empire")
        assert s["lob_el"] == 0.0 and s["land_w"] == 0.0 and not s["rd"]


def test_a_ninth_contact_effect_fits_by_data():
    """The queue's next ones: the Stormfiends' Pestilent Wind (the 8th), the Thunder Gauntlets' (the 9th): no code, the slots of the state hold them."""
    from tools.nn.sim.state import INTERNAL
    assert innate.CONTACT_SLOTS >= 9
    for j in range(innate.CONTACT_SLOTS):
        assert f"cfx{j}_s" in INTERNAL and f"cdd{j}_by" in INTERNAL
    p = sim_params.load()
    fx = copy.deepcopy(p.effects)
    for k in ("alt_a", "alt_b"):
        fx["effects"][k] = dict(fx["effects"][POISON])
        fx["order"].append(k)
    import dataclasses
    q = dataclasses.replace(p, effects=fx)
    assert len(innate.contact_keys(q)) == len(innate.contact_keys(p)) + 2


def test_only_the_v3_scheme_fields_them():
    doc = load("pools.json")
    skv = doc["factions"]["wh2_main_skv_skaven"]
    assert G in json.dumps(skv["units_v3"]) and G not in json.dumps({k: v for k, v in skv.items() if k != "units_v3"})
    groups = load("army_templates.json")["factions"]
    assert any(G in f["units"] for f in groups.values())


def test_the_probes_plans():
    b = cp.battles("globes")[0]
    assert [x["kind"] for x in b] == ["near", "far", "friends"]
    assert all(x["attacker"] == "globes" and x["mode"] == "shoot" for x in b)
    assert b[1].get("t_morale") and b[2]["extras"][0]["attack"] and b[2]["extras"][0]["short"] == "cspear"
    b2 = cp.battles("globes2")[0]
    assert [x["kind"] for x in b2] == ["lob", "far", "knights"]
    cfg = cp.run_config("globes2", 1)[0]
    assert cfg["mem_proj"] and cfg["mem_extras"] and cfg["mem_morale"]


def test_the_card_shows_the_poison_on_friends_too_and_the_companion_sets_its_bit():
    from tools.nn.companion import exchange
    from tools.nn.model import effects as mfx
    row = {"side": 1, "x": 0.0, "z": 0.0, "b": 0, "men": 100, "hp": 0.9, "mp": 0.8, "ms": 3, "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False,
           "a": 0, "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": 0.0, "oz": 0.0, "lf": False, "rf": False, "bf": False, "v": True}
    doc = {"batch": "b-1", "move": 1, "t": 5000, "done": False, "attacker": 2, "decide_ms": 1000,
           "factions": {"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"},
           "units": [dict(row, n="own_sp", key="wh_main_emp_inf_spearmen_0", fx=[POISON]),
                     dict(row, n="enemy_g", side=2, key=G, z=60.0, oz=60.0, b=180, men=48, a=816, fire=True, fx=[]),
                     dict(row, n="enemy_c", side=2, key="wh2_main_skv_inf_clanrat_spearmen_0", z=2.0, oz=2.0, b=180, men=160, a=0, fx=[POISON])]}
    b = exchange.battle(doc)
    s_ = exchange.arrays(doc, b.names)
    exchange.effects_on(s_, doc, b.names, b.setup)
    i = mfx.keys().index(POISON)
    on = {n: (int(s_["fx_on"][0, k]) >> i) & 1 for k, n in enumerate(b.names)}
    assert on["own_sp"] == 1 and on["enemy_c"] == 1 and on["enemy_g"] == 0
    assert mfx.owned([G, "wh_main_emp_inf_spearmen_0"])[0, i] == 1 and mfx.owned([G, "wh_main_emp_inf_spearmen_0"])[1, i] == 0
