"""tools.sim.formation: the first army example plans with the real roster."""
from tools.sim import formation


def test_first_attack_plans_both_sides():
    army, _ = formation.load_army("first_attack")
    sides, units = formation.simulate(army)
    assert sides["own"]["status"] == "ok" and not sides["own"]["overlaps"]
    assert sides["own"]["roles"] == {"wall": 1, "arc": 4, "lord": 1}
    assert len(units) == 6 and sides["enemy"]["roles"] == {"wall": 2, "arc": 2, "lord": 1}


def test_simulation_sees_both_sides_as_groups():
    army, _ = formation.load_army("first_attack")
    sides, _ = formation.simulate(army)
    enemy, own = sides["enemy"]["vision"], sides["own"]["vision"]
    assert len(enemy["groups"]) == 1 and len(enemy["main"]["ids"]) == 5 and enemy["seen_share"] == 1
    # Our wall, archers and lord stand as one group too.
    assert len(own["groups"]) == 1 and len(own["main"]["ids"]) == 6 and own["seen_share"] == 1


def test_soldier_points_fill_the_rectangle():
    p = {"x": 0, "z": 0, "bearing": 0, "front_m": 20, "depth_m": 10}
    pts = formation.soldier_points(p, 50)
    xs, zs = pts[0::2], pts[1::2]
    assert len(xs) == 50 and -10 < min(xs) and max(xs) < 10 and -10 < min(zs) and max(zs) < 0
    assert formation.soldier_points(p, 1) == [0, 0]


def test_simulation_builds_the_battlefield():
    army, _ = formation.load_army("first_attack")
    sides, _ = formation.simulate(army)
    field = sides["battlefield"]
    # Anchors 300 m apart along +Z: the axis looks north, the field is 40 m wider on each side.
    assert field["status"] == "ok" and min(field["bearing"], 360 - field["bearing"]) < 5
    assert 250 < field["centres_m"] < 350 and field["gap_m"] > 200
    assert field["half_width_m"] >= 40 + field["enemy"]["width_m"] / 2 - 1


def test_crooked_army_turns_to_face_the_enemy():
    army, _ = formation.load_army("first_attack_crooked")
    sides, _ = formation.simulate(army)
    a = sides["alignment"]
    # Started 25 degrees off: it only turns to face their centre (task 26), no shift aside.
    assert a["check"]["needed"] and set(a["check"]["reasons"]) == {"angle"}
    assert a["check"]["source"] == "centres" and abs(a["check"]["offset_m"]) < 1e-6
    after = a["after"]["check"]
    assert not after["needed"] and abs(after["angle_off_deg"]) < 3
    assert sides["own_before"]["placements"] and not sides["own"]["overlaps"]


def test_crooked_army_with_the_old_line_moves_opposite_their_front():
    army, _ = formation.load_army("first_attack_crooked")
    sides, _ = formation.simulate(dict(army, align={"line": "enemy_facing"}))
    a = sides["alignment"]
    # ~54 m aside of the line the enemy looks along: turned and shifted onto it.
    assert set(a["check"]["reasons"]) == {"angle", "offset"} and abs(a["check"]["offset_m"]) > 40
    after = a["after"]["check"]
    assert not after["needed"] and abs(after["angle_off_deg"]) < 1 and abs(after["offset_m"]) < 1


def test_aligned_army_is_left_alone():
    army, _ = formation.load_army("first_attack")
    sides, _ = formation.simulate(army)
    assert not sides["alignment"]["check"]["needed"] and "own_before" not in sides


def test_mask_shows_the_hamlet_and_the_lane_it_blocks():
    army, _ = formation.load_army("hamlet_attack")
    sides, _ = formation.simulate(army)
    mask = sides["mask"]
    s = mask["summary"]
    # The hamlet is one solid block (~1880 m2 at 1 m): a couple of hundred 3 m cells inside the field.
    assert s["known"] == s["cells"] and 150 < s["blocked"] < 400
    assert all(f["ok"] for f in mask["fits"])
    # Our front (with the lord) is wide enough to clip the hamlet's east corner on the way.
    assert not mask["lane"]["free"] and 100 < mask["lane"]["first_blocked_along"] < 200


def test_the_place_past_the_rock_is_not_taken_inside_their_reach():
    army, _ = formation.load_army("rock_attack")
    sides, _ = formation.simulate(army)
    log = sides["approach"]["log"]
    # The rock lies across the way close to them: a step aside along its edge (apps.approach
    # choose_aside), then every place past it is inside their archers' reach (apps.reach), so
    # the approach stops and the level above decides (before 28.09.2026 the formation went
    # past the rock under their fire).
    assert [r["decision"] for r in log] == ["approach", "approach", "blocked"]
    assert log[1]["path"]["aside_m"] and log[2]["reason"] == "no_place" and log[2]["window"]["reason"] == "window"
    assert all(r["path"]["advance_m"] <= r["window"]["safe_to"] for r in log if r["decision"] == "approach")
    # The formation is planned anew at each step (as in battle), not shifted as checked: after
    # the step aside one block may touch the rock's edge; the engine squeezes it
    # (tests/tools/test_sim_physics.py).
    assert sum(not f["ok"] for f in sides["mask"]["fits"]) <= 1


def test_long_wall_steps_aside_past_the_rock_out_of_their_reach():
    army, _ = formation.load_army("rock_attack_wide")
    sides, _ = formation.simulate(army)
    log = sides["approach"]["log"]
    # A step 15 m aside finds the place past the rock, still out of their archers' reach.
    assert [r["decision"] for r in log] == ["approach", "approach", "hold"]
    assert log[1]["path"]["aside_m"] and log[1]["path"]["advance_m"] <= log[1]["window"]["safe_to"]
    assert log[2]["window"]["safe_to"] >= 0 and not log[2]["window"]["under_fire_now"]
    assert len(sides["own"]["placements"]) == 16 and not sides["own"]["overlaps"]


def test_approach_in_open_field_holds_at_the_stop_line():
    army, _ = formation.load_army("first_attack")
    army["approach"] = True  # no map: every cell counts as standable
    sides, _ = formation.simulate(army)
    log = sides["approach"]["log"]
    steps = [r for r in log if r["decision"] == "approach"]
    assert log[-1]["decision"] == "hold" and all(r["advance_m"] <= 50 for r in steps)
    assert abs(log[-1]["gap_m"] - log[-1]["stop_gap_m"]) <= 2
    # Each step waits for the one before: one manoeuvre at a time.
    assert all(b["t_s"] > a["t_s"] for a, b in zip(log, log[1:]))


def test_scenario_xml_stays_well_formed_with_any_rout_points_and_zones():
    # Rout points and zones of other lengths than the template's broke the XML
    # (28.09.2026: the game crashed loading the battle).
    import xml.etree.ElementTree as ET
    unit = {"script_name": "u", "key": "wh_main_emp_inf_spearmen_0", "men": 120, "general": False, "x": 0, "z": 0}
    zone = {"centre": [-270, -290], "width": 400, "height": 260, "orientation": 0}
    for routs in (None, ((-600, 0), (600, 0)), ((-600, -290), (600, -290))):
        xml = formation.probe_xml({"own": [unit], "enemy": [dict(unit, script_name="e")]},
                                  zones=(zone, dict(zone, centre=[270, -290])), routs=routs)
        root = ET.fromstring(xml)
        alliances = root.findall(".//alliance")
        assert [a.get("id") for a in alliances] == ["0", "1"]
        if routs:
            assert [a.find(".//rout_position").get("y") for a in alliances] == [str(routs[0][1]), str(routs[1][1])]
