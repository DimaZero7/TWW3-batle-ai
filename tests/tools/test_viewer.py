"""The battle viewer's records (tools/viewer): a game run and a simulation give the same format."""
import json
import math

import pytest

from tools.viewer import game, record, sim

LORD = {"id": "own_1", "key": "wh_main_emp_cha_general_0", "class": "com", "men": 1, "commanding": True, "range_m": 0,
        "shapes": [{"ordered_m": 5, "front_m": 1.0, "depth_m": 1.0}]}
ARCHERS = {"id": "own_2", "key": "wh2_dlc13_emp_inf_archers_0", "class": "inf_mis", "men": 4, "commanding": False, "range_m": 130,
           "shapes": [{"ordered_m": 20, "front_m": 18.0, "depth_m": 6.0}, {"ordered_m": 40, "front_m": 38.0, "depth_m": 3.0}]}
SPEARS = {"id": "enemy_2", "key": "wh_main_emp_inf_spearmen_0", "class": "inf_mel", "men": 10, "commanding": False, "range_m": 0,
          "shapes": [{"ordered_m": 40, "front_m": 38.6, "depth_m": 7.7}]}


def motion(x, z, bearing=90, width=20, men=4, moving=False):
    return {"x": x, "z": z, "bearing": bearing, "ordered_width": width, "number_of_men_alive": men,
            "is_moving": moving}


def enemy(men=10):
    return [{"script_name": "enemy_2", "motion": motion(300, 0, 270, 40, men)}]


def write_run(path):
    """A small run as formation-probe writes it: placed, a still stretch, one step, a hold with losses."""
    soldiers = [(-10 + 5 * i) * 10 for i in range(4)]
    flat = []
    for across in soldiers:            # bearing 90: across = -z, forward = +x
        flat += [0, -across]
    rows = [
        {"event": "plan", "model_ms": 1000, "strategy": "wall_and_arc"},
        {"event": "stage", "model_ms": 1000, "stage": "placed"},
        {"event": "stage_snapshot", "model_ms": 2000, "units": [
            {"script_name": "own_1", "role": "lord", "motion": motion(-10, 0, 90, 5, 1), "soldiers_dm": [-100, 0]},
            {"script_name": "own_2", "role": "arc", "motion": motion(0, 0), "soldiers_dm": flat}]},
        {"event": "alignment", "model_ms": 2000, "decision": "align",
         "check": {"needed": True, "angle_off_deg": 12, "offset_m": -30, "max_angle_deg": 10, "max_offset_m": 15},
         "target": {"anchor": {"x": 5, "z": 1}, "bearing": 90},
         "field": {"bearing": 90, "origin": {"x": 0, "z": 0}, "gap_m": 250, "half_width_m": 60,
                   "corners": [{"x": -10, "z": 60}, {"x": -10, "z": -60}, {"x": 310, "z": -60}, {"x": 310, "z": 60}],
                   "own": {"front_m": 3, "left_m": -10, "right_m": 10}, "enemy": {"front_m": 253, "left_m": -20, "right_m": 20}}},
        {"event": "stage", "model_ms": 2000, "stage": "mask"},
        {"event": "mask", "model_ms": 30000, "grid": {"frame": {"origin": {"x": 0, "z": 0}, "bearing": 90},
                                                      "along0": 0, "across0": -3, "step": 3, "rows": 1, "cols": 2},
         "code": ".#"},
        {"event": "stage_snapshot", "model_ms": 32000, "units": [
            {"script_name": "own_1", "role": "lord", "motion": motion(-10, 0, 90, 5, 1)},
            {"script_name": "own_2", "role": "arc", "motion": motion(0.5, 0)}]},
        {"event": "approach_decision", "model_ms": 33000, "decision": "approach", "reason": "step", "gap_m": 250,
         "stop_gap_m": 120, "step": {"action": "step", "advance_m": 50},
         "window": {"reason": "window", "reached": 1, "shooters": 1, "safe_to": 140, "advance_m": 135,
                    "window": {"from": 130, "to": 140}}},
        {"event": "approach_sample", "model_ms": 34000, "units": [
            {"script_name": "own_1", "motion": motion(-10, 0, 90, 5, 1, True)},
            {"script_name": "own_2", "motion": motion(10, 0, 90, 20, 4, True)}], "enemy": enemy()},
        {"event": "approach_manoeuvre", "model_ms": 60000, "kind": "approach", "reason": "stopped", "units": [
            {"script_name": "own_1", "motion": motion(40, 0, 90, 5, 1)},
            {"script_name": "own_2", "motion": motion(50, 0)}]},
        {"event": "hold_sample", "model_ms": 62000, "units": [
            {"script_name": "own_1", "motion": motion(40, 0, 90, 5, 1)},
            {"script_name": "own_2", "ammo": 50, "motion": motion(50, 0, 90, 20, 3)}], "enemy": enemy(8)},
        {"event": "result", "model_ms": 63000, "status": "completed"},
    ]
    (path / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    cfg = {"army": "tiny", "tree": {}, "own": {"units": [LORD, ARCHERS]}, "enemy": {"units": [SPEARS]}}
    (path / "manifest.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")
    return path


@pytest.fixture
def game_record(tmp_path):
    return game.convert(write_run(tmp_path))


def frame_units(rec, i):
    return {u["id"]: u for u in rec["frames"][i]["units"]}


def test_a_game_run_becomes_a_valid_record(game_record):
    r = game_record
    assert record.check(r) == []
    assert {u["id"]: u["role"] for u in r["units"]} == {"own:own_1": "lord", "own:own_2": "arc", "enemy:enemy_2": "wall"}
    arc = next(u for u in r["units"] if u["id"] == "own:own_2")
    assert arc["reach_m"] == pytest.approx(126)            # apps.reach: range 130, first arrow at 126
    assert r["meta"]["source"] == "game" and r["meta"]["strategy"] == "wall_and_arc"


def test_units_get_a_type_a_tag_and_a_title_in_both_languages(game_record):
    u = {x["id"]: x for x in game_record["units"]}
    assert u["own:own_1"]["tag"] == {"ru": "Лорд", "en": "Lord"}
    assert u["own:own_2"]["tag"] == {"ru": "С1", "en": "M1"} and u["own:own_2"]["title"] == "archers"
    assert u["enemy:enemy_2"]["tag"] == {"ru": "П1", "en": "I1"} and u["enemy:enemy_2"]["type"]["en"] == "Infantry"


def test_every_text_is_in_both_languages(game_record):
    assert game_record["meta"]["title"]["en"].startswith("Game")
    for e in game_record["events"]:
        assert e["text"]["ru"] and e["text"]["en"], e
    broken = json.loads(json.dumps(game_record))
    broken["events"][0]["text"] = {"ru": "только по-русски"}
    assert any("both languages" in p for p in record.check(broken))


def test_the_game_s_battlefield_and_alignment_become_overlays(game_record):
    kinds = {o["kind"]: o for o in game_record["overlays"]}
    f, a = kinds["field"], kinds["alignment"]
    assert f["corners"][0] == [-10, 60] and f["gap_m"] == 250
    # Fronts across the field: ours 3 m ahead of the origin, from 10 m left to 10 m right (bearing 90: right = -z).
    assert f["own_front"] == [[3, 10], [3, -10]] and f["enemy_front"][0] == [253, 20]
    assert a["needed"] and a["target"] == {"x": 5, "z": 1, "bearing": 90}


def test_the_enemy_is_carried_to_frames_that_did_not_see_it(game_record):
    first = frame_units(game_record, 0)
    assert "enemy:enemy_2" in first and first["enemy:enemy_2"]["men"] == 10
    assert frame_units(game_record, len(game_record["frames"]) - 1)["enemy:enemy_2"]["men"] == 8


def test_blocks_take_depth_from_the_shape_nearest_the_ordered_width(game_record):
    u = frame_units(game_record, 0)["own:own_2"]
    assert (u["front_m"], u["depth_m"]) == (18.0, 6.0)
    last = frame_units(game_record, len(game_record["frames"]) - 1)["own:own_2"]
    assert last["men"] == 3 and last["depth_m"] == pytest.approx(4.5) and last["ammo"] == 50


def test_a_still_stretch_is_cut_to_a_second(game_record):
    (cut,) = game_record["meta"]["skipped"]
    assert (cut["game_from_ms"], cut["game_to_ms"]) == (2000, 32000)
    times = [f["t_ms"] for f in game_record["frames"]]
    assert times == [0, 1000, 3000, 29000, 31000]          # game 2, 32, 34, 60, 62 s
    assert game_record["meta"]["duration_ms"] == 62000 - 2000 - cut["cut_ms"]
    mask_stage = next(e for e in game_record["events"] if e.get("stage") == "mask")
    assert mask_stage["t_ms"] == 0


def test_soldiers_are_kept_in_the_unit_s_frame(game_record):
    snap = game_record["soldiers"]["own:own_2"][0]
    pts = record.world_points(snap["pts"], 0, 0, 90)
    assert [round(v, 1) for v in pts[:4]] == [0.0, 10.0, 0.0, 5.0]
    # Across, in decimetres, left to right: the unit's right is -z at bearing 90.
    assert snap["pts"][0::2] == [-100, -50, 0, 50]


def test_a_decision_keeps_its_window_and_step(game_record):
    d = next(e for e in game_record["events"] if e["kind"] == "decision" and e["decision"] == "approach")
    assert d["advance_m"] == 50
    assert d["window"]["from"] == 130 and d["window"]["to"] == 140 and d["window"]["safe_to"] == 140
    assert "скачок вперёд" in d["text"]["ru"] and "окно" in d["text"]["ru"]
    assert "step forward" in d["text"]["en"] and "window" in d["text"]["en"]
    assert game_record["mask"]["rows"] == [".#"]


def test_points_round_trip_through_the_unit_frame():
    flat = [12.3, -4.5, 20.0, 7.5]
    local = record.local_points(flat, 10, 2, 37)
    back = record.world_points(local, 10, 2, 37)
    assert all(math.isclose(a, b, abs_tol=0.1) for a, b in zip(flat, back))


def test_check_catches_a_broken_record(game_record):
    broken = json.loads(json.dumps(game_record))
    broken["frames"][1]["units"][0]["id"] = "own:nobody"
    broken["frames"][2]["t_ms"] = -5
    assert len(record.check(broken)) == 2


def test_the_page_carries_the_records_and_opens_without_a_server(game_record, tmp_path):
    game_record["events"][0]["text"] = {"ru": "</script><b>", "en": "x"}
    html = record.page([game_record], tmp_path / "v.html", "fire", "en").read_text(encoding="utf-8")
    assert "/*RECORDS*/null" not in html and '"preset":"fire"' in html and '"lang":"en"' in html
    assert "<\\/script><b>" in html and html.count("</script>") == 1
    assert "src=" not in html and "fetch(" not in html


@pytest.fixture(scope="module")
def simulated():
    army, _ = sim.simulation.load_army("window_open")
    planner = sim.simulation.Planner()
    sides, _ = sim.simulation.simulate(army, planner)
    return sim.convert("window_open", planner), sides


def test_a_simulation_becomes_a_record_of_the_same_format(simulated):
    r, _ = simulated
    assert record.check(r) == []
    assert r["meta"]["source"] == "sim"
    assert {u["side"] for u in r["units"]} == {"own", "enemy"}
    assert any(e["kind"] == "decision" and e["decision"] == "approach" for e in r["events"])


def test_a_simulation_keeps_what_the_modules_saw(simulated):
    r, sides = simulated
    kinds = {}
    for o in r["overlays"]:
        kinds.setdefault(o["kind"], []).append(o)
    assert {"field", "groups", "alignment", "routes"} <= set(kinds)
    # A battlefield and the groups at the start and after every move.
    assert len(kinds["field"]) == len(kinds["groups"]) == len(sides["approach"]["moves"]) + 1
    own_ids = {u["id"] for u in r["units"] if u["side"] == "own"}
    main = next(g for g in kinds["groups"][-1]["own"] if g["main"])
    assert set(main["ids"]) <= own_ids and main["ids"]
    # The lord's routes start where the lord stands (apps.formation: along 0, back of the anchor).
    lord = next(p for p in sides["own"]["placements"] if p["role"] == "lord")
    start = kinds["routes"][-1]["lines"][0]["points"][0]
    assert start == pytest.approx([lord["x"], lord["z"]], abs=0.05)


def test_a_simulated_manoeuvre_takes_its_walking_time(simulated):
    r, sides = simulated
    moves = sides["approach"]["moves"]
    assert moves
    ends = [e["t_ms"] for e in r["events"] if e["kind"] == "manoeuvre"]
    starts = [e["t_ms"] for e in r["events"] if e["kind"] == "decision" and e["decision"] in ("align", "approach")
              and e.get("gap_m") is not None]
    for m, t0, t1 in zip(moves, starts, ends):
        assert t1 - t0 == pytest.approx(m["walk_s"] * 1000, abs=2)


def test_simulated_blocks_are_centred_like_the_game_s(simulated):
    # tools/sim placements hold the front rank's centre; the record, the block's centre (as the game).
    r, sides = simulated
    p = sides["own"]["placements"][1]
    last = {u["id"]: u for u in r["frames"][-1]["units"]}[f"own:{p['id']}"]
    b = math.radians(p["bearing"])
    assert last["x"] == pytest.approx(p["x"] - math.sin(b) * p["depth_m"] / 2, abs=0.05)
    assert last["z"] == pytest.approx(p["z"] - math.cos(b) * p["depth_m"] / 2, abs=0.05)


@pytest.mark.slow
def test_a_step_round_an_obstacle_becomes_two_records_to_compare():
    from tools.viewer import logistics
    (queue, at_once), = logistics.convert("rock_march_wide")
    for r in (queue, at_once):
        assert record.check(r) == []
        assert {u["side"] for u in r["units"]} == {"own"}           # the enemy plays no part in the queue
        assert r["frames"][1]["t_ms"] - r["frames"][0]["t_ms"] == 1000
        assert [s["name"]["en"] for s in r["series"]] == ["crowded soldiers", "soldiers on blocked cells"]
        plan = next(o for o in r["overlays"] if o["kind"] == "logistics")
        assert len(plan["band"]) == 2 and plan["units"]
    waits = lambda r: [u["release_s"] for u in next(o for o in r["overlays"] if o["kind"] == "logistics")["units"].values()]
    assert max(waits(queue)) > 0 and max(waits(at_once)) == 0
    assert queue["meta"]["title"]["ru"].endswith("с очередью") and at_once["meta"]["title"]["en"].endswith("all at once")
