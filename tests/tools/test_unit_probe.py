"""The new units' probe plans and measures (tools/nn/unit_probe.py, on the melee probe tools/nn/charge_probe.py)."""
import json
import math
from pathlib import Path

import numpy as np
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
        elif plan == "ratshape":           # one battle, each kind once (narrow and wide targets, a lone man)
            assert len(b) == 1 and [l["kind"] for l in b[0]] == [x[0] for x in up.RATSHAPE_LANES]
        elif plan == "snipe":              # two battles (long rifles, jezzails), the same four map lanes each
            assert [[l["kind"] for l in bt] for bt in b] == [list(up.SNIPE_SITES)] * 2
        elif plan == "losmove":            # one battle, each kind once (the target walks / runs at the shooter)
            assert len(b) == 1 and [l["kind"] for l in b[0]] == [x[0] for x in up.LOSMOVE_LANES]
            assert all(l["target_mode"] == "push" and l["push_m"] > up.LOSMOVE_GAP for l in b[0])
            assert sum(bool(l.get("a_free")) for l in b[0]) == 4 and sum(bool(l.get("push_run")) for l in b[0]) == 4
        elif plan == "losrange":           # the flat battle, then the two map lanes (one a battle)
            assert [l["kind"] for l in b[0]] == [x[0] for x in up.LOSRANGE_LANES]
            assert [bt[0]["kind"] for bt in b[1:]] == [x[0] for x in up.LOSRANGE_MAP]
            assert all(l.get("a_at") for l in b[0] if l["kind"].endswith("_back"))
        elif plan == "losfriend":          # one battle, each kind once (flat fire with own infantry ahead and alone)
            assert len(b) == 1 and [l["kind"] for l in b[0]] == [x[0] for x in up.LOSFRIEND_LANES]
            cover = [l for l in b[0] if l.get("extras")]
            assert len(cover) == 2 and all(l["extras"][0]["dz"] > 0 for l in cover)
        elif plan in ("hill", "flankslot", "flankslot2", "lobfriend", "forestwalk"):   # one battle (build/diag_stage0_probe)
            assert len(b) == 1 and len({l["kind"] for l in b[0]}) == len(b[0])
        else:
            assert len(b) == 2 and sorted(l["kind"] for l in b[0]) == sorted(l["kind"] for l in b[1])
            # every kind in two lane places
            assert all([l["kind"] for l in b[0]].index(k) != [l["kind"] for l in b[1]].index(k)
                       for k in (l["kind"] for l in b[0]))
        for i in range(1, len(b) + 1):
            config, model_s, arena = cp.run_config(plan, i)
            for l in config["lanes"]:
                # (a map lane: inside the Kislev field, |x|, |z| <= 470)
                assert l["morale_rows"] and l["card_rows"] and max(abs(l["x"]), abs(l["z"])) <= (470 if l.get("map") else 440)
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


def test_the_snipe_plan_puts_the_shooter_in_the_woods_and_the_target_on_open_ground():
    """The snipe plan (build/diag_snipe): the shooter block on forest ground (edge / deep / approach) or off it (open),
    facing its target SNIPE_GAP m off on open ground; the ranged attack waits SNIPE_GO_S; the approach lane's enemy walks
    to SNIPE_STOP_M m off the shooter's front; both rows carry the hiding (stealth_rows 'both')."""
    import json
    H = np.load("build/maps/ksl_plains_01/heights.npz") if Path("build/maps/ksl_plains_01/heights.npz").exists() else None
    names = (json.loads(Path("build/maps/ksl_plains_01/meta.json").read_text(encoding="utf-8"))["ground_names"]
             if H is not None else None)
    b = cp.battles("snipe")
    assert [{l["attacker"] for l in bt} for bt in b] == [{up.LR}, {up.JZ}]
    for bt in b:
        assert len({l["kind"] for l in bt}) == 4
        for l in bt:
            assert l["stealth_rows"] == "both" and l["map"] == cp.TERRAIN["map"] and l["answer"] is False
            ax, az, ab = l["a_at"]["x"], l["a_at"]["z"], l["a_at"]["b"]
            tx, tz = l["t_at"]["x"], l["t_at"]["z"]
            assert abs((math.degrees(math.atan2(tx - ax, tz - az)) - ab + 180) % 360 - 180) < 0.5   # facing it
            assert l["t_at"]["b"] == (ab + 180) % 360
            if l["kind"] == "approach":
                assert l["mode"] == "hold" and l["a_free"] and l["target_mode"] == "push" and not l["push_run"]
                px, pz, _, _ = up.SNIPE_SITES["approach"]
                assert math.hypot(l["push_to"]["x"] - px, l["push_to"]["z"] - pz) == pytest.approx(up.SNIPE_STOP_M, abs=0.2)
            else:
                assert l["mode"] == "shoot" and l["a_go_s"] == up.SNIPE_GO_S and l["max_s"] == up.SNIPE_S
            if H is None:
                continue
            g = H["ground"]
            forest = names.index("forest")
            on = lambda x, z: g[int(round(z - float(H["z0"]))), int(round(x - float(H["x0"])))] == forest
            assert on(ax, az) == (l["kind"] != "open")
            assert not on(tx, tz)
    # the lanes are apart: no enemy within 300 m of the approach shooter but its own (its fire at will)
    for bt in b:
        ap = next(l for l in bt if l["kind"] == "approach")
        for l in bt:
            if l is not ap:
                assert math.hypot(l["t_at"]["x"] - ap["a_at"]["x"], l["t_at"]["z"] - ap["a_at"]["z"]) > 300


def test_snipe_measure_reads_the_hiding_and_the_gap_at_which_the_enemy_saw_the_shooter():
    spec = {"kind": "approach", "target_mode": "push", "a_depth": 8.0, "t_depth": 16.0, "attacker": up.LR,
            "a_go_s": None}
    samples = []
    for k in range(40):
        t = k * 0.5
        gap = 100 - 4 * k                               # edge to edge: 100 m at the go, 4 m a sample closer
        d = gap + 12.0
        a = {"x": 0.0, "z": 0.0, "men": 32, "ammo": 768 - (32 if k >= 4 else 0) - (32 if k >= 20 else 0),
             "hid": gap > 60, "v": gap <= 60}
        tg = {"x": 0.0, "z": d, "men": 160, "hp": 1000.0 - (10 if k >= 6 else 0)}
        samples.append((t, a, tg, None))
    m = up.snipe_measure({"spec": spec, "samples": samples, "run": "r", "plan": "snipe"})
    assert m["hid_go"] == 1.0 and m["gap_seen_m"] == pytest.approx(60.0) and m["gap_unhid_m"] == pytest.approx(60.0)
    assert m["hid_100"] == 1.0 and m["v_100"] == 0.0 and m["hid_40"] == 0.0 and m["v_60"] == 1.0
    assert m["shots_unseen"] == 32.0 and m["first_shot_s"] == pytest.approx(2.0)


def _lane_points(l):
    """[(x, z)] a flankslot / hill lane's units: the target, the friends, the enemy's start and its step points."""
    pts = [(l["t_at"]["x"], l["t_at"]["z"])] + [(l["t_at"]["x"] + e["dx"], l["t_at"]["z"] + e["dz"])
                                                 for e in l.get("extras") or []]
    ax, az = l["a_at"]["x"], l["a_at"]["z"]
    pts.append((ax, az))
    for s in l.get("steps") or []:
        if s["kind"] == "move":
            pts.append((ax + s["dx"], az + s["dz"]))
    return pts


def test_the_hill_plan_walks_the_enemy_up_to_a_target_on_the_hill_and_on_level_ground():
    """The hill plan (build/diag_stage0_probe): the tested target (its morale kept, the script's flank flags) holds at
    each HILL_SITES place facing the enemy, who walks up from HILL_GAP to HILL_STOP m off its front; the sites' drop
    (target - ground HILL_GAP m out) on the Kislev heights: gentle and steep >= 15 m, level < 3 m; the lanes' units
    stay 150+ m from every other lane's target (the morale's 120 m lists)."""
    b = cp.battles("hill")[0]
    assert [l["kind"] for l in b] == list(up.HILL_SITES)
    td = cp.depth(cp.UNITS[up.HILL_T][0], cp.UNITS[up.HILL_T][1])
    ad = cp.depth(cp.UNITS[up.HILL_A][0], cp.UNITS[up.HILL_A][1])
    for l in b:
        assert l["mode"] == "script" and l["target_mode"] == "hold" and l["t_morale"] and l["flank_rows"]
        assert l["map"] == cp.TERRAIN["map"]
        px, pz, bear = up.HILL_SITES[l["kind"]]
        ax, az = l["a_at"]["x"], l["a_at"]["z"]
        assert abs((math.degrees(math.atan2(ax - px, az - pz)) - bear + 180) % 360 - 180) < 0.5
        (sx, sz), = [(s["dx"], s["dz"]) for s in l["steps"]]
        assert math.hypot(ax + sx - px, az + sz - pz) == pytest.approx(up.HILL_STOP + (td + ad) / 2, abs=0.2)
        assert math.hypot(ax - px, az - pz) == pytest.approx(up.HILL_GAP + (td + ad) / 2, abs=0.2)
    for l in b:
        for o in b:
            if o is not l:
                assert all(math.hypot(x - o["t_at"]["x"], z - o["t_at"]["z"]) > 150 for x, z in _lane_points(l))
    p = Path("build/maps/ksl_plains_01/heights.npz")
    if p.exists():
        H = np.load(p)
        h = lambda x, z: float(H["height"][int(round(z - float(H["z0"]))), int(round(x - float(H["x0"])))])
        for l in b:
            px, pz, bear = up.HILL_SITES[l["kind"]]
            fx, fz = math.sin(math.radians(bear)), math.cos(math.radians(bear))
            drop = h(px, pz) - h(px + fx * up.HILL_GAP, pz + fz * up.HILL_GAP)
            assert (drop < 3) if l["kind"] == "level" else (drop >= 15)


def test_the_flankslot_plan_moves_the_enemy_about_a_held_target_lanes_apart():
    """The flankslot plan: the held target faces +z at its place, its friends (side 't') beside it on its line, the
    enemy's 'script' steps from its start; every lane's units 190+ m from any other lane's target (the 120 m lists); the target's row
    carries the script's lf / rf / bf (flank_rows) and its morale (t_morale); the run reads the morale from memory."""
    b = cp.battles("flankslot")[0]
    assert [l["kind"] for l in b] == list(up.FLANK_LANES)
    assert [l["kind"] for l in cp.battles("flankslot2")[0]] == list(up.FLANK2_LANES)
    for bt in (b, cp.battles("flankslot2")[0]):
        for l in bt:
            assert l["t_at"]["b"] == 0.0 and l["t_morale"] and l["flank_rows"] and l["target_mode"] == "hold"
            for e in l.get("extras") or []:
                assert e["side"] == "t" and e["dz"] == 0.0 and abs(e["dx"]) >= 40
            for o in bt:
                if o is not l:
                    assert all(math.hypot(x - o["t_at"]["x"], z - o["t_at"]["z"]) > 190 for x, z in _lane_points(l))
            assert all(abs(x) <= 560 and abs(z) <= 560 for x, z in _lane_points(l))
    side = next(l for l in b if l["kind"] == "side_stand")
    gaps = [side["a_at"]["x"] + s["dx"] - side["t_at"]["x"] for s in side["steps"]]
    assert gaps == [120, 90, 65, 45, 30]
    away = next(l for l in b if l["kind"] == "side_away")
    assert [s["kind"] for s in away["steps"]] == ["face", "face"] and away["a_at"]["b"] == 90.0
    config, _, arena = cp.run_config("flankslot", 1)
    assert config["mem_morale"] and config["men_near_m"] >= 600 and "ksl" not in arena.get("map", "")


def test_the_lobfriend_plan_puts_the_cover_ahead_of_the_arcing_shooters():
    """The lobfriend plan: archers / crossbows / slingers ('shoot', arcing) at a held target, a unit of their side
    (fearless, never ordered) with its back edge LOB_LANES' metres ahead of the shooter's front - or none (the
    controls); the run reads the projectiles and the cover's men from memory."""
    b = cp.battles("lobfriend")[0]
    assert [l["kind"] for l in b] == [x[0] for x in up.LOB_LANES]
    for l, (kind, shooter, aw, target, gap, cover, ahead) in zip(b, up.LOB_LANES):
        assert l["mode"] == "shoot" and l["target_mode"] == "hold" and l["gap_m"] == gap and l["a_w"] == aw
        assert cp.passport(cp.UNITS[shooter][0])["missile"]["trajectory"] == "dual_low_fixed"
        if cover is None:
            assert not l.get("extras")
            continue
        e, = l["extras"]
        tk, tm, _ = cp.UNITS[target]
        ck, cm, _ = cp.UNITS[cover]
        # the cover's back edge `ahead` m from the shooter's front, the target's front gap m off it
        assert e["dz"] + cp.depth(ck, cm) / 2 == pytest.approx(cp.depth(tk, tm) / 2 + gap - ahead, abs=0.1)
    config, _, _ = cp.run_config("lobfriend", 1)
    assert config["mem_proj"] and config["mem_extras"]


def test_the_forestwalk_plan_shuttles_spearmen_in_forest_and_on_grass_walking_and_running():
    """The forestwalk plan: spearmen shuttle GROUND leg_m legs (a walk: shuttle_walk) at FOREST_SITES on Kislev; the
    forest sites' legs on forest ground, the grass ones' off it, both level (rise < 0.07 over the leg)."""
    b = cp.battles("forestwalk")[0]
    assert [l["kind"] for l in b] == list(up.FOREST_SITES)
    for l in b:
        x0, z0, bear, run = up.FOREST_SITES[l["kind"]]
        assert l["mode"] == "shuttle" and l["shuttle_walk"] == (not run) and l["map"] == cp.TERRAIN["map"]
        assert math.hypot(l["p2"]["x"] - x0, l["p2"]["z"] - z0) == pytest.approx(cp.GROUND["leg_m"], abs=0.2)
    p = Path("build/maps/ksl_plains_01/heights.npz")
    if p.exists():
        H = np.load(p)
        names = json.loads(Path("build/maps/ksl_plains_01/meta.json").read_text(encoding="utf-8"))["ground_names"]
        at = lambda a, x, z: a[int(round(z - float(H["z0"]))), int(round(x - float(H["x0"])))]
        for l in b:
            x0, z0, _, _ = up.FOREST_SITES[l["kind"]]
            pts = [(x0 + k / 10 * (l["p2"]["x"] - x0), z0 + k / 10 * (l["p2"]["z"] - z0)) for k in range(11)]
            share = np.mean([at(H["ground"], x, z) == names.index("forest") for x, z in pts])
            assert share == (1.0 if l["kind"].startswith("forest") else 0.0)
            hs = [float(at(H["height"], x, z)) for x, z in pts]
            assert abs(hs[-1] - hs[0]) / cp.GROUND["leg_m"] < 0.07
