"""The Eshin Triads' passport, data and the probe's plan (02.11.2026; build/diag_roster/wh2_dlc14_skv_inf_eshin_triads_0/units.md,
mechanic.md): the Death Runners' body, bombs and blade with a guandao against large, charge reflection and charge defence against
large. Nothing of their own in the code: every rule is the Runners' or older. No torch (the simulator's numbers:
test_sim_triads_torch.py; the Runners' rules run on both keys: test_sim_runners_torch.py)."""
import json

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn.armies import pools

T3 = "wh2_dlc14_skv_inf_eshin_triads_0"
DR = "wh2_main_skv_inf_death_runners_0"
BOMBS = "wh2_main_unit_abilities_concealment_bombs"
BLADE = "wh2_main_unit_contact_weeping_blade"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    doc = load("units.json")
    p = doc["units"][T3]
    assert (p["men"], p["hp_per_man"], p["hp_total"], p["armour"], p["multiplayer_cost"], p["leadership"]) == (72, 80, 5760, 20,
                                                                                                                  700, 64)
    m = p["melee"]
    assert (m["attack"], m["defence"], m["charge_bonus"], m["damage"], m["ap_damage"]) == (30, 28, 10, 8, 22)
    assert (m["bonus_v_large"], m["bonus_v_infantry"], m["attack_interval_s"], m["weapon_length"]) == (15, 0, 4.6, 2.0)
    assert m["contact_phase"] == BLADE and p["missile"] is None
    assert p["damage_resist"]["physical"] == 30 and p["hiding_scalar"] == 0.8 and p["spotting"]["scrub"] == 80
    assert {"charge_defense_vs_large", "charge_reflection", "guerrilla_deploy", "stalk", "strider"} == set(p["attributes"])
    assert BOMBS in p["abilities"]
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))
    assert probe["units"][T3] == p


def test_the_body_and_the_bombs_are_the_runners():
    u = load("units.json")["units"]
    for k in ("entity", "mass", "radius_m", "spacing", "speed", "armour", "damage_resist", "hiding_scalar", "spotting", "abilities"):
        assert u[T3][k] == u[DR][k], k


def test_no_new_ability_or_effect_for_the_triads():
    doc = load("effects.json")
    assert doc["inflicts"][T3] == [BLADE]
    assert set(doc["units"][T3]) == {"charge_defense_vs_large", "charge_reflection", "guerrilla_deploy", "stalk", "strider",
                                     "wh2_main_unit_passive_scurry_away", "wh2_main_unit_passive_strength_in_numbers"}
    assert all(k in doc["effects"] for k in doc["units"][T3])
    ab = load("abilities.json")["abilities"]
    assert [k for k, p in ab.items() if p["uses"] != -1] == [BOMBS, "wh3_dlc29_unit_abilities_stormfiend_seismic_snare"]         # (the other: the Stormfiends', 02.11.2026)


def test_the_triads_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert (T3 in keys) is has, scheme
    groups = load("army_templates.json")["factions"]["wh2_main_skv_skaven"]["units"][T3]
    assert groups == [{"group": "wh3_default_land_melee_infantry_main_flanking", "quality": "tier_2_step_3_units"}]


def test_the_probe_plan_has_its_lanes():
    b = cp.battles("triads")
    assert [[ln["kind"] for ln in x] for x in b] == [["knights", "bombs", "swords"]]
    kn, bo, sw = b[0]
    assert kn["attacker"] == bo["attacker"] == cp.KNIGHTS and sw["attacker"] == "swords"
    assert all(x["target"] == "triads" and x["target_mode"] == "hold" and x["a_on_sight"] and x["stealth_rows"] for x in b[0])
    assert bo["t_casts"] == [{"key": BOMBS, "at_s": 0}] and not kn.get("t_casts") and not sw.get("t_casts")
    assert kn["gap_m"] == 150 and sw["fight_s"] == 60
    cfg = cp.run_config("triads", 1)
    config = cfg[0] if isinstance(cfg, tuple) else cfg
    assert len(config["lanes"]) == 3


def lane_for_measure():
    """A swords lane: the target unseen to 8 s (gaps 80 ... 0 m by 1 m a tick of 0.5 s), the contact at 40 s."""
    spec = {"name": "L3", "kind": "swords", "a_key": cp.UNITS["swords"][0], "t_key": T3, "a_depth": 0.0, "t_depth": 0.0}
    samples = []
    for k in range(240):
        t = k * 0.5
        gap = max(0.0, 80.0 - 2.0 * t)
        samples.append((t, {"x": 0.0, "z": gap, "men": 120, "hp": 9000 - max(0, t - 40) * 30},
                        {"x": 0.0, "z": 0.0, "men": 72, "hp": 5760 - max(0, t - 40) * 60, "v": t >= 8, "hid": t < 8.5}, None))
    return {"run": "r", "spec": spec, "samples": samples, "contacts": {1: 40.0},
            "phases": [{"phase": "a_try_end", "kept": False, "t": 3000}, {"phase": "a_move", "t": 3000}]}


def test_the_measure_reads_the_sight_and_the_melee():
    m = cp.triads_measure(lane_for_measure())
    assert m["unseen_at_start"] and m["seen_t"] == 8.0 and m["seen_gap"] == 64.0 and m["seen_gap_before"] == 65.0
    assert m["hid_off_t"] == 8.5 and m["order_kept"] is False and m["contact_after_seen_s"] == 32.0
    # -1 ... 55 s: the rate from the contact on, the second before it none
    assert m["tg_hp_s_melee"] == round(60 * 55 / 56, 1) and m["a_hp_s_melee"] == round(30 * 55 / 56, 1)
    assert m["tg_hp_60"] == 3600.0 and m["a_hp_10"] == 300.0
