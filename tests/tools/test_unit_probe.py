"""The new units' probe plans and measures (tools/nn/unit_probe.py, on the melee probe tools/nn/charge_probe.py)."""
import json

import pytest

from tools.nn import charge_probe as cp
from tools.nn import unit_probe as up


def test_any_empire_or_skaven_key_with_a_passport_stands_in_a_lane():
    key = "wh2_dlc16_skv_mon_wolf_rats_1"
    assert cp.UNITS[key] == (key, 80, cp.SKV) and cp.SHORT[key] == "wolfratp" and cp.FACTION[key] == cp.SKV
    # a key without a short name: men from its passport, faction from the key, the name kept
    other = "wh2_main_skv_inf_plague_monks"
    assert cp.UNITS[other][1] == 120 and cp.SHORT["wh_x_key"] == "wh_x_key"
    # the probe-only passports give the depth (the Ratling Gun's weapon team: 3.4 x 4.3 m)
    assert cp.spacing("wh2_dlc12_skv_inf_ratling_gun_0") == (3.4, 4.3)
    assert cp.depth("wh2_dlc12_skv_inf_ratling_gun_0", 32) == pytest.approx(4 * 4.3)
    with pytest.raises(KeyError):
        cp.UNITS["wh_main_emp_inf_no_such_unit"]
    with pytest.raises(KeyError):
        cp.UNITS["wh_main_dwf_inf_miners_0"]         # neither side of the probe


def test_the_new_units_plans_build_on_the_grid_with_effects_and_card_rows(tmp_path):
    keys = {"ratling": "wh2_dlc12_skv_inf_ratling_gun_0", "monks": "wh2_main_skv_inf_plague_monks",
            "censer": "wh2_main_skv_inf_plague_monk_censer_bearer", "wolfrat": "wh2_dlc16_skv_mon_wolf_rats_0",
            "wolfratp": "wh2_dlc16_skv_mon_wolf_rats_1", "halb": "wh_main_emp_inf_halberdiers"}
    seen = set()
    for plan in cp.UNIT_PLANS:
        b = cp.battles(plan)
        if plan == "suppress2":            # one battle, each kind twice
            assert len(b) == 1 and sorted(l["kind"] for l in b[0]) == sorted(list(up.SUPPRESS2_STOP) * 2)
        elif plan == "ratlord":            # one lane a battle (one General): each kind once
            assert [l["kind"] for bt in b for l in bt] == [x[0] for x in up.RATLORD_LANES]
        elif plan == "ratlone":            # one battle, each kind once (flat and arcing fire at lone heroes)
            assert len(b) == 1 and [l["kind"] for l in b[0]] == [x[0] for x in up.RATLONE_LANES]
        elif plan == "ratform":            # one battle, each kind once (the gun and handguns at formations)
            assert len(b) == 1 and [l["kind"] for l in b[0]] == [x[0] for x in up.RATFORM_LANES]
        else:
            assert len(b) == 2 and sorted(l["kind"] for l in b[0]) == sorted(l["kind"] for l in b[1])
            # every kind in two lane places
            assert all([l["kind"] for l in b[0]].index(k) != [l["kind"] for l in b[1]].index(k)
                       for k in (l["kind"] for l in b[0]))
        for i in range(1, len(b) + 1):
            config, model_s, arena = cp.run_config(plan, i)
            for l in config["lanes"]:
                assert l["morale_rows"] and l["card_rows"] and max(abs(l["x"]), abs(l["z"])) <= 440
                seen |= {l["a_key"], l["t_key"]}
            xml = cp.write_scenario(plan, i, path=tmp_path / "b.xml").read_text(encoding="utf-8")
            assert all(f'type="{k}"' in xml for k in {l["a_key"] for l in config["lanes"]})
    assert set(keys.values()) <= seen


def test_the_ratling_plan_shoots_held_targets_at_three_ranges_shields_half_crew_and_the_general():
    lanes = {l["kind"]: l for l in cp.battles("ratling")[0]}
    assert set(lanes) == {"d60", "d100", "d135", "shield100", "half100", "lord100"}
    assert all(l["attacker"] == "ratling" and l["mode"] == "shoot" and l["target_mode"] == "hold" and not l["answer"]
               for l in lanes.values())
    assert [lanes[k]["gap_m"] for k in ("d60", "d100", "d135")] == [60, 100, 135]
    assert lanes["half100"]["damage"] == {"a": {"method": "kill", "share": 0.5}} and "damage" not in lanes["d100"]
    assert lanes["shield100"]["target"] == "spearsh" and lanes["lord100"]["target"] == "general"
    assert lanes["d60"]["max_s"] == up.RATLING_S


def test_the_suppress_plan_charges_the_gun_firing_or_silent_at_a_walk_and_a_run():
    lanes = {l["kind"]: l for l in cp.battles("suppress")[0]}
    assert lanes["walk_fire"]["mode"] == "attack_walk" and lanes["run_fire"]["mode"] == "attack_run"
    assert lanes["walk_fire"]["target_mode"] == "skirmish" and lanes["walk_fire"]["t_skirmish"] is False
    assert lanes["walk_quiet"]["target_mode"] == "hold" and lanes["run_quiet"]["target"] == "ratling"
    assert all(l["gap_m"] == up.SUPPRESS_GAP and l["attacker"] == "swords" for l in lanes.values())
    stop = {l["kind"]: l for l in cp.battles("suppress2")[0]}
    assert stop["walk_stop25"]["t_stop_fire_s"] == 25 and stop["walk_stop25"]["mode"] == "attack_walk"
    assert stop["run_stop15"]["t_stop_fire_s"] == 15 and stop["run_stop15"]["target_mode"] == "skirmish"


def test_the_hitslow_plan_runs_clanrats_at_shooters_without_a_contact_effect():
    lanes = {l["kind"]: l for l in cp.battles("hitslow")[0]}
    assert set(lanes) == {"walk_hgun", "run_hgun", "walk_xbow", "run_xbow", "walk_quiet", "run_quiet"}
    assert all(l["attacker"] == up.HITSLOW_RUNNER and l["gap_m"] == up.SUPPRESS_GAP for l in lanes.values())
    assert lanes["run_hgun"]["target"] == "hgun" and lanes["run_hgun"]["target_mode"] == "skirmish"
    assert lanes["walk_xbow"]["target"] == "xbow" and lanes["walk_xbow"]["mode"] == "attack_walk"
    assert lanes["run_quiet"]["target_mode"] == "hold" and lanes["run_quiet"]["mode"] == "attack_run"
    assert up.MEASURE["hitslow"] is up.suppress_measure


def test_the_censer_and_wolf_plans_lanes():
    c = {l["kind"]: l for l in cp.battles("censer")[0]}
    assert c["censer_leave"]["mode"] == "withdraw" and c["censer_leave"]["t_at_leave"] == "halt"
    assert c["censer_leave"]["recharge_after_s"] == 20 and c["censer_morale"]["t_morale"]
    assert c["held_censer"]["target_mode"] == "hold" and c["held_censer"]["target"] == "censer"
    assert (c["halb"]["attacker"], c["halb"]["target"]) == ("halb", "svsh")
    w = {l["kind"]: l for l in cp.battles("wolfrats")[0]}
    assert w["charge120"]["gap_m"] == 120 and w["rear"]["target_mode"] == "rear"
    assert w["braced"]["target"] == "spear" and w["braced"]["target_mode"] == "hold"
    assert w["recharge"]["mode"] == "recharge" and w["recharge"]["pause_s"] == 3
    assert w["chase"]["rout_at_s"] == 15 and w["chase_clanrat"]["attacker"] == "clanrat"
    for b in cp.battles("wolfrats"):
        pace = next(l for l in b if l["kind"] == "pace")
        # the run aside goes towards the middle of the field, then back to its place
        assert pace["steps"][0]["dx"] == (-up.PACE_M if pace["place"][0] > 0 else up.PACE_M)
        assert pace["steps"][1] == {"at_s": up.PACE_BACK_S, "kind": "move", "dx": 0, "dz": 0, "run": True}
    p = {l["kind"]: l for l in cp.battles("wolfpoison")[0]}
    assert p["poison"]["attacker"] == "wolfratp" and p["plain"]["attacker"] == "wolfrat"
    assert p["poison_leave_halt"]["t_at_leave"] == "halt" and "t_at_leave" not in p["poison_leave_chase"]


def _lane(plan, kind, rows, contacts=None, phases=()):
    """A recorded lane of plan's battle 1: rows [(t, a row, tg row)]."""
    spec = next(l for l in cp.layout(cp.battles(plan)[0])[0] if l["kind"] == kind)
    return {"run": "r", "plan": plan, "battle": 1, "spec": spec, "samples": [(t, a, g, None) for t, a, g in rows],
            "men": [], "contacts": contacts or {}, "end": None, "abilities": [], "phases": list(phases)}


def test_ratling_measure_bullets_bursts_damage_and_the_targets_phase():
    """32 men, 270 ammo each; 18 bullets a man in 2 s bursts every 5 s from 3 s; the target loses 2 HP a bullet; the
    suppression phase on the target from the first burst and its card speed 33 -> 23."""
    rows, ammo, hp = [], 32 * 270.0, 8000.0
    for k in range(0, 61):
        t = k * 0.5
        burst = t >= 3 and (t - 3) % 5 < 2 - 1e-9 and t < 25
        if burst:
            ammo -= 32 * 18 / 4
            hp -= 2 * 32 * 18 / 4
        fx = "wh2_dlc12_unit_contact_suppressive_fire" if t >= 3.5 else ""
        rows.append((t, {"x": 0, "z": 100, "men": 32, "ammo": ammo, "dd": 0},
                     {"x": 0, "z": 0, "men": 120, "hp": hp, "fx": fx, "cs": {"scalar_speed": 23 if fx else 33}}))
    m = up.ratling_measure(_lane("ratling", "d100", rows))
    assert m["ammo0"] == 8640 and m["ammo_per_man"] == 270 and m["first_shot_s"] == 3
    assert m["bursts"] == 5 and m["burst_bullets"] == 32 * 18 and m["burst_every_s"] == 5 and m["burst_s"] == 2
    assert m["bullets"] == 5 * 32 * 18 and m["hp_per_bullet"] == pytest.approx(2)
    assert m["tg_fx"] == "wh2_dlc12_unit_contact_suppressive_fire" and m["tg_fx_first_s"] == 0.5
    assert (m["tg_card_speed"], m["tg_card_speed_fx"]) == (33, 23) and m["d0_m"] == 100


def test_suppress_measure_speed_under_fire_and_the_phase_after_the_last_shot():
    """The swordsmen walk 1.5 m/s from z 160 to the gun at 0, 1.0 m/s under the phase (6-20 s), contact at 90 s; the gun
    fires 2-16 s; the phase lasts to 20 s (4 s after the last shot)."""
    rows, z = [], 160.0
    for k in range(0, 200):
        t = k * 0.5
        on = 6 <= t < 20
        if k:
            z -= 0.5 * (1.0 if on else 1.5)
        shots = 1000 - 10 * min(max(0, int((t - 2) / 0.5)), 28)
        rows.append((t, {"x": 0, "z": z, "hp": 9000, "fx": "suppressive_fire" if on else "",
                         "cs": {"scalar_speed": 21 if on else 30}}, {"x": 0, "z": 0, "ammo": shots}))
    m = up.suppress_measure(_lane("suppress", "walk_fire", rows, {1: 90.0}))
    assert m["speed_way_fx"] == pytest.approx(1.0) and m["speed_way_nofx"] == pytest.approx(1.5)
    assert m["last_shot_s"] == 16 and m["fx_after_last_shot_s"] == pytest.approx(3.5) and m["fx_ended"]
    assert (m["card_speed0"], m["card_speed_fx"], m["card_speed_min"]) == (30, 21, 21)


def test_censer_measure_the_contagion_its_life_after_the_leave_and_the_cards():
    """Contact at 10 s, the contagion on the swordsmen from 11 s, their card leadership 70 -> 60; the censers leave at
    30 s, the last melee sample 33 s, the phase gone after 43 s."""
    rows = []
    for k in range(0, 121):
        t = k * 0.5
        m = 10 <= t <= 33
        on = 11 <= t <= 43
        rows.append((t, {"x": 0, "z": 40, "hp": 9600 - 10 * max(0, t - 10), "m": m, "fx": "frenzy" if t >= 10 else "",
                         "cs": {"stat_melee_attack": 44 if t >= 10 else 34}},
                     {"x": 0, "z": 0, "hp": 8760 - 50 * max(0, min(t, 33) - 10), "m": m, "mp": 0.9,
                      "fx": "wh2_main_unit_contact_censer_morale" if on else "",
                      "cs": {"stat_morale": 60 if on else 70}}))
    lane = _lane("censer", "censer_leave", rows, {1: 10.0}, [{"phase": "out", "t": 30000}])
    m = up.censer_measure(lane)
    assert m["tg_fx"] == "wh2_main_unit_contact_censer_morale" and m["tg_fx_first_s"] == 1
    assert m["tg_hp_0_5"] == pytest.approx(50) and m["a_hp_5_15"] == pytest.approx(10)
    assert (m["tg_stat_morale"], m["tg_stat_morale_fight"]) == (70, 60)
    assert (m["a_stat_melee_attack"], m["a_stat_melee_attack_fight"]) == (34, 44)
    assert m["leave_clear_s"] == 3 and m["tg_fx_after_clear_s"] == 10 and not m["tg_fx_still_on"]
    assert m["a_fx_new"] == "frenzy"


def test_wolfrats_pace_measure_the_peak_the_rise_and_the_turn():
    rows, x, x_leg1 = [], 0.0, None
    for k in range(0, 91):
        t = k * 0.5
        v = (min(9.5, 3 * (t - 1)) if 1 <= t < 18 else 0 if t < 20 else -min(9.5, 3 * (t - 20)))
        if k:
            x += 0.5 * v
        rows.append((t, {"x": x, "z": 100}, {"x": 0, "z": 0}))
        if t == 18.5:
            x_leg1 = x
    m = up.wolfrats_measure(_lane("wolfrats", "pace", rows))
    assert m["leg1_cruise"] == pytest.approx(9.5) and 3.5 < m["leg1_to90_s"] < 5
    assert m["leg2_cruise"] == pytest.approx(9.5) and m["leg1_s"] == pytest.approx(16.5)
    # moving from 1 s to 17.5 s (the last half second with a step)
    assert m["leg1_m"] == pytest.approx(x_leg1)


def test_report_tables_per_plan_and_kind(tmp_path, capsys):
    run = tmp_path / "run1"
    run.mkdir()
    config, _, _ = cp.run_config("ratling", 1)
    (run / "manifest.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    events = []
    for k in range(0, 20):
        lanes = [{"lane": l["name"], "t": k * 500, "a": {"x": 0, "z": 100, "men": 32, "ammo": 8640 - 100 * k},
                  "tg": {"x": 0, "z": 0, "men": 120, "hp": 8000 - 150 * max(0, k - 2)}} for l in config["lanes"]]
        events.append({"event": "probe_sample", "lanes": lanes})
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    result = up.report([run], out=tmp_path / "out.json")
    assert len(result["lanes"]) == 6
    s = {r["kind"]: r for r in result["summary"]}
    # the hits land a flight (100 m at 100 m/s: 1 s) after the bullets: the bullets of the lane's last 1.5 s are left out
    assert s["d60"]["hp_per_bullet"][0] == pytest.approx(1.5, rel=0.02) and s["d60"]["n"] == 1
    assert "== ratling" in capsys.readouterr().out
