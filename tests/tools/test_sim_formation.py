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


def test_crooked_army_is_aligned_opposite_the_enemy():
    army, _ = formation.load_army("first_attack_crooked")
    sides, _ = formation.simulate(army)
    a = sides["alignment"]
    # Started 25 degrees off and ~54 m aside of the line the enemy looks along.
    assert a["check"]["needed"] and set(a["check"]["reasons"]) == {"angle", "offset"}
    assert a["check"]["source"] == "enemy_facing" and abs(a["check"]["offset_m"]) > 40
    after = a["after"]["check"]
    assert not after["needed"] and abs(after["angle_off_deg"]) < 1 and abs(after["offset_m"]) < 1
    assert sides["own_before"]["placements"] and not sides["own"]["overlaps"]


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


def test_approach_finds_a_place_past_the_rock():
    army, _ = formation.load_army("rock_attack")
    sides, _ = formation.simulate(army)
    log = sides["approach"]["log"]
    assert [r["decision"] for r in log] == ["approach", "approach", "hold"]
    past = log[1]["path"]
    # The 50 m target lies on the rock: the next place where the whole formation fits is past it;
    # the engine walks the units around (detour), the stop line is crossed (enemy not considered yet).
    assert past["advance_m"] > 50 and past["detour"] and past["beyond_stop_line"]
    # The rock (z 86..161, a diagonal band) is behind the wall and nobody stands on it.
    wall = next(p for p in sides["own"]["placements"] if p["role"] == "wall")
    assert wall["z"] > 161 and all(f["ok"] for f in sides["mask"]["fits"])


def test_long_wall_also_goes_past_the_rock():
    army, _ = formation.load_army("rock_attack_wide")
    sides, _ = formation.simulate(army)
    decisions = [r["decision"] for r in sides["approach"]["log"]]
    assert decisions[-1] == "hold" and decisions.count("approach") == 2
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
