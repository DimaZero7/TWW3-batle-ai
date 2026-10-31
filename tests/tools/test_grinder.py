"""The Warp Grinders' passport, ability passports, pools and the probe's measure (31.10.2026;
build/diag_roster/wh2_dlc14_skv_inf_warp_grinder_0/units.md, mechanic.md): 24 crews of 184 HP, armour 70, a magical
drill, and two active abilities without a target on every enemy within 35 m - Warp-Quake (direct damage, speed x 0.75)
and Seismic Snare (no move). No torch (the simulator's rules: test_sim_grinder_torch.py)."""
import json

import pytest

from tools import config as project
from tools.nn import charge_probe as cp
from tools.nn.armies import pools

GR = "wh2_dlc14_skv_inf_warp_grinder_0"
QUAKE, SNARE = "wh2_dlc14_unit_abilities_warp_quake", "wh2_dlc14_unit_abilities_seismic_snare"


def load(name):
    return json.loads((project.CONFIG_DIR / "nn" / name).read_text(encoding="utf-8"))


def test_the_passport_is_the_database():
    p = load("units.json")["units"][GR]
    assert (p["men"], p["hp_per_man"], p["armour"], p["mass"], p["multiplayer_cost"]) == (24, 184, 70, 600.0, 600)
    m = p["melee"]
    assert (m["damage"], m["ap_damage"], m["magic"], m["attack_interval_s"], m["splash_max_attacks"]) == (18, 78, True,
                                                                                                         5.7, 4)
    assert {QUAKE, SNARE} <= set(p["abilities"]) and "wallbreaker" in p["attributes"]
    probe = json.loads((project.ROOT / "data" / "units" / "probe-units.json").read_text(encoding="utf-8"))["units"]
    assert probe[GR] == p


def test_both_abilities_are_pressed_without_a_target():
    ab = load("abilities.json")["abilities"]
    for k, (active, recharge) in ((QUAKE, (17.0, 90.0)), (SNARE, (13.0, 60.0))):
        p = ab[k]
        # affect_self 0, no target_friends / enemies / ground: no target to choose - the network and the bridge press it
        # as a lord's own (perform_special_ability(key, the unit))
        assert not p["targets_own"] and p["self_cast"] and not p["passive"] and not p["auto"]
        assert (p["active_s"], p["recharge_s"], p["range_m"], p["enemy_units"], p["update_targets"]) == (
            active, recharge, 35.0, -1, True)
        assert p["targets"] == {"self": False, "friends": False, "enemies": True} and p["loop_stance"]
    # the lords' own abilities stay self-cast, passives never
    assert ab["wh_main_character_abilities_rally"]["self_cast"] and not ab["wh_main_unit_passive_frenzy"]["self_cast"]
    assert not any(p.get("loop_stance") for k, p in ab.items() if k not in (QUAKE, SNARE))


def test_warp_quake_is_the_direct_class_on_enemies_and_the_snare_cant_move():
    ab = load("abilities.json")["abilities"]
    d = ab[QUAKE]["direct"]
    # the game's program (mechanic.md W1-W5): 5 x 18 men of EACH enemy unit every 0.5 s, 35 m held to 40 m
    assert (d["amount"], d["entities"], d["every_s"], d["on"], d["hit_hp"]) == (5, 18, 0.5, ["enemies"], pytest.approx(3.2))
    assert (d["range_m"], d["hold_m"], d["n_enemy"], d["every_frame"], d["duration_s"], d["magical"]) == (
        35.0, 40.0, -1, True, 17.0, True)
    assert [(e["stat"], e["value"], e["on"]) for e in ab[QUAKE]["effects"]] == [("scalar_speed", 0.75, ["enemies"])]
    assert ab[SNARE]["cant_move"] and not ab[SNARE]["effects"] and "direct" not in ab[SNARE]
    assert not any(p.get("cant_move") for k, p in ab.items() if k != SNARE)


def test_the_simulator_models_both_with_the_default_trigger():
    sim = load("sim.json")["abilities"]
    assert {QUAKE, SNARE} <= set(sim["model"]) and QUAKE not in sim["triggers"] and SNARE not in sim["triggers"]
    assert sim["default_trigger"] == "near" and sim["enemy_hold_m"] == 5
    assert sim["cant_move"]["on"] and sim["cant_move"]["routers"]
    assert "STAND-IN" in sim["grinders_why"]


def test_the_grinders_join_the_v3_pools_only():
    for scheme, has in (("v1", False), ("v2", False), ("v3", True)):
        keys = {x.key for x in pools.load(scheme=scheme)["wh2_main_skv_skaven"].units}
        assert (GR in keys) is has, scheme
    v3 = {x.key: x for x in pools.load(scheme="v3")["wh2_main_skv_skaven"].units}
    assert v3[GR].width == 60.0
    groups = load("army_templates.json")["factions"]["wh2_main_skv_skaven"]["units"][GR]
    assert groups and groups[0]["group"] == "wh3_default_land_melee_infantry_special"


def test_the_grinder_plan_is_one_battle_of_ten_lanes():
    (battle,) = cp.battles("grinder")
    assert [s["kind"] for s in battle] == [k for k in cp.GRINDER["kinds"] for _ in range(2)]
    config, model_s, _ = cp.run_config("grinder", 1)
    assert len(config["lanes"]) == 10 and model_s < 400 and config["tick_ms"] == cp.TURN_TICK_MS
    quake = config["lanes"][0]
    assert quake["a_casts"] == [{"key": QUAKE, "at_m": 30}] and quake["target_mode"] == "push"


def _lane(kind, samples, press_t, contact=None, steps=None):
    return {"run": "x", "spec": {"name": "L1", "kind": kind, "steps": steps or []}, "samples": samples,
            "abilities": [{"who": "a", "key": QUAKE, "n": 1, "t": press_t * 1000, "status": "used"}],
            "contacts": {1: contact} if contact is not None else {}, "phases": []}


def test_the_measure_reads_the_quake_damage_and_the_slowing():
    # the swordsmen walk 1.5 m/s, from the press at 10 s 1.125 m/s and 57.6 HP every 0.5 s (the first at once)
    samples = []
    for k in range(0, 4 * 40):
        t = k / 4
        z = 1.5 * min(t, 10) + 1.125 * max(0.0, t - 10)
        ticks = 0 if t < 10 else min(34, int((t - 10) / 0.5) + 1)
        samples.append((t, {"x": 0.0, "z": 50.0, "hp": 4416.0}, {"x": 0.0, "z": z, "hp": 8280.0 - 57.6 * ticks}, None))
    out = cp.grinder_measure(_lane("quake", samples, 10.0))
    assert out["tc"] == 10.0 and out["tg_hp_s"] == pytest.approx(115.2 * 17 / 17.3, rel=0.03)
    assert out["tg_v_before"] == pytest.approx(1.5) and out["tg_v_ratio"] == pytest.approx(0.75, abs=0.01)
    assert out["tg_hp_17s"] == pytest.approx(1958.4, abs=1)


def test_the_measure_reads_a_snared_stop():
    samples = []
    for k in range(0, 4 * 50):
        t = k / 4
        z = 1.5 * min(t, 10.25)
        samples.append((t, {"x": 0.0, "z": 50.0, "hp": 4416.0}, {"x": 0.0, "z": z, "hp": 8280.0}, None))
    out = cp.grinder_measure(_lane("snare", samples, 10.0))
    assert out["tg_v_snared"] == 0.0 and out["resume_s"] is None and out["stop_s"] == pytest.approx(0.25)


def test_the_network_may_press_both_in_the_simulators_slots():
    from tools.nn.model import abilities as mab
    from tools.nn.sim import abilities as sab
    idx, usable, _ = mab.slots([GR])
    keys = mab.keys()
    assert [keys[i] if 0 <= i < len(keys) else "" for i in idx[0, :2]] == [SNARE, QUAKE]
    assert usable[0, 0] and usable[0, 1] and not usable[0, 2:].any() and mab.sim_fits(idx)
    # the simulator's slots are the network's first ones (its ability branches price a press like a lord's)
    assert sab.slot_keys(GR, load("units.json")["units"], load("abilities.json")["abilities"])[:2] == [SNARE, QUAKE]
