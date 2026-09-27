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
