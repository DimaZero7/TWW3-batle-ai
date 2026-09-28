"""apps.formation.services: the line_and_blocks layout places units by the roles it is given.

Roles come from apps.strategy; here they are set by hand, so these tests do
not depend on any strategy.
"""
import json

import pytest

from tests.lua_runtime import load, new_runtime

SPEARS = [{"ordered_m": 60, "front_m": 58, "depth_m": 6}, {"ordered_m": 40, "front_m": 39, "depth_m": 8},
          {"ordered_m": 20, "front_m": 18, "depth_m": 15}]
ARCHERS = [{"ordered_m": 30, "front_m": 29, "depth_m": 13}, {"ordered_m": 20, "front_m": 19, "depth_m": 18},
           {"ordered_m": 15, "front_m": 14, "depth_m": 26}, {"ordered_m": 10, "front_m": 10, "depth_m": 36}]


def unit(uid, role):
    shapes = {"wall": SPEARS, "arc": ARCHERS}.get(role, [{"ordered_m": 5, "front_m": 1, "depth_m": 1}])
    return {"id": uid, "role": role, "range_m": 130 if role == "arc" else 0, "shapes": [dict(s) for s in shapes]}


@pytest.fixture
def plan():
    lua = new_runtime()
    f = load(lua, "apps.formation.services")
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")

    def run(units, params=None, enemy_lord=None, bearing=0):
        data = {"anchor": {"x": 0, "z": 0}, "bearing": bearing, "units": units}
        if enemy_lord:
            data["enemy_lord"] = {"x": enemy_lord[0], "z": enemy_lord[1]}
        result = f.plan("line_and_blocks", lua.table_from(data, recursive=True), lua.table_from(params or {}))
        result.overlaps = f.overlaps(result.placements)
        return json.loads(encode(result))
    run.lua, run.f = lua, f
    return run


def army(archers=4):
    return [unit("lord", "lord"), unit("spears", "wall")] + [unit(f"a{i}", "arc") for i in range(archers)]


def by_role(result):
    roles = {}
    for p in result["placements"]:
        roles.setdefault(p["role"], []).append(p)
    return roles


def test_first_example_is_covered_without_overlaps(plan):
    r = plan(army())
    assert r["status"] == "ok" and r["layout"] == "line_and_blocks" and not r["overlaps"]
    c = r["choice"]
    assert c["archer_front_m"] <= c["wall_front_m"] + 2 * 10 and c["wall_depth_m"] >= 5 and c["min_reach_m"] >= 80
    # Near-square archer blocks with room to turn between them.
    assert c["archer_aspect"] <= 1.3 and c["archer_gap_m"] > 1
    roles = by_role(r)
    assert len(roles["wall"]) == 1 and len(roles["arc"]) == 4 and len(roles["lord"]) == 1
    wall = roles["wall"][0]
    # Bearing 0 faces +Z: archers stand behind (lower Z), the lord in the centre behind the wall.
    assert all(a["z"] < wall["z"] - wall["depth_m"] for a in roles["arc"])
    assert roles["lord"][0]["x"] == pytest.approx(0) and roles["lord"][0]["z"] < wall["z"] - wall["depth_m"]


def test_the_wall_is_as_thick_as_the_archers_reach_allows(plan):
    units = army()[1:]
    default = plan(units)["choice"]
    assert default["min_reach_m"] >= 80
    # Asking less reach never makes the wall thinner.
    thick = plan(units, {"min_reach_m": 40})["choice"]
    assert thick["wall_depth_m"] >= default["wall_depth_m"] and thick["min_reach_m"] >= 40
    # With reach weighted like depth the thinnest wall wins, as before the rule.
    thin = plan(units, {"wall_depth_weight": 1, "min_reach_m": 60})["choice"]
    assert thin["wall_depth_m"] <= default["wall_depth_m"]


def test_impossible_cover_is_reported(plan):
    r = plan([unit("spears", "wall")] + [unit(f"a{i}", "arc") for i in range(6)],
             {"min_wall_depth_m": 15, "max_archer_rows": 1})
    assert r["status"] == "infeasible" and r["choice"]["failed"]


def test_no_wall(plan):
    assert plan([unit("a", "arc")])["status"] == "no_wall"


def test_units_without_a_role_are_rejected(plan):
    with pytest.raises(Exception, match="has no role"):
        plan([{"id": "x", "range_m": 0, "shapes": SPEARS}])


def test_other_roles_are_left_unplaced(plan):
    r = plan(army(1) + [unit("horse", "other")])
    assert r["unplaced"] == ["horse"] and all(p["id"] != "horse" for p in r["placements"])


def check_lord_in_the_centre(r, params=None):
    """The lord stands on the axis in the passage of the first archer row and can reach both flanks."""
    p = {"lord_passage_m": 10, **(params or {})}
    assert not r["overlaps"]
    roles = by_role(r)
    lords, first = roles.get("lord", []), [a for a in roles.get("arc", []) if a["row"] == 1]
    for lord in lords:
        assert lord["along_m"] == pytest.approx(0)
        assert lord["back_m"] > r["choice"]["wall_depth_m"]
    if not lords:
        return
    if first:
        # The two blocks either side of the axis leave the passage between them.
        left = [a["along_m"] + a["front_m"] / 2 for a in first if a["along_m"] < 0]
        right = [a["along_m"] - a["front_m"] / 2 for a in first if a["along_m"] > 0]
        assert len(right) - len(left) in (0, 1)
        assert min(right) - (max(left) if left else -min(right)) >= p["lord_passage_m"] - 1e-6
        middle = first[0]["back_m"] + max(a["depth_m"] for a in first) / 2
        assert sum(l["back_m"] for l in lords) / len(lords) == pytest.approx(middle)
    routes = r["lord_routes"]
    for side in ("left", "right"):
        assert routes[side]["clear"], (side, routes[side]["blocked_by"])
        assert routes[side]["length_m"] >= r["choice"]["wall_front_m"] / 2


def test_lord_stands_in_the_centre_in_a_passage(plan):
    r = plan(army(4))
    check_lord_in_the_centre(r)
    # Wider passage on request; the enemy lord's side no longer matters.
    check_lord_in_the_centre(plan(army(4), {"lord_passage_m": 16}), {"lord_passage_m": 16})
    assert plan(army(4), enemy_lord=(-30, 200))["placements"] == r["placements"]


def test_without_a_lord_there_is_no_passage(plan):
    r = plan(army(4)[1:])
    assert "lord_routes" not in r and "passage_m" not in r["choice"]


# Every mix from 2 to 20 units: walls, archers, none to two lords (user, 28.09.2026:
# the code must hold for any army, not the test one).
MIXES = [(w, a, l) for l in (0, 1, 2) for w in range(1, 19) for a in range(0, 19) if 2 <= w + a + l <= 20]


@pytest.mark.parametrize("bearing", [0, 90])
def test_any_mix_of_units(plan, bearing):
    bad = []
    for w, a, l in MIXES:
        units = ([unit(f"l{i}", "lord") for i in range(l)] + [unit(f"w{i}", "wall") for i in range(w)]
                 + [unit(f"a{i}", "arc") for i in range(a)])
        r = plan(units, bearing=bearing)
        try:
            assert len(r["placements"]) == w + a + l
            check_lord_in_the_centre(r)
        except AssertionError as e:
            bad.append(((w, a, l), r["status"], str(e)[:120]))
    assert not bad, bad[:10]


def test_archers_have_room_to_turn(plan):
    r = plan([unit("spears", "wall")] + [unit(f"a{i}", "arc") for i in range(2)], {"min_reach_m": 40})
    arcs = sorted(by_role(r)["arc"], key=lambda p: p["x"])
    if arcs[0]["z"] == arcs[1]["z"]:
        gap = arcs[1]["x"] - arcs[0]["x"] - (arcs[0]["front_m"] + arcs[1]["front_m"]) / 2
        diagonal = (arcs[0]["front_m"] ** 2 + arcs[0]["depth_m"] ** 2) ** 0.5
        assert gap == pytest.approx(diagonal - arcs[0]["front_m"])
    else:  # one behind the other: the row gap is the turning room
        back = sorted(arcs, key=lambda p: -p["z"])
        gap = back[0]["z"] - back[0]["depth_m"] - back[1]["z"]
        diagonal = (back[0]["front_m"] ** 2 + back[0]["depth_m"] ** 2) ** 0.5
        assert gap == pytest.approx(diagonal - back[0]["depth_m"])


def test_unknown_layout(plan):
    with pytest.raises(Exception, match="Unknown formation layout"):
        plan.f.plan("circle", plan.lua.table_from({}), None)


def test_facing(plan):
    t = plan.lua.table_from
    assert plan.f.facing(t({"x": 0, "z": 0}), t({"x": 0, "z": 10})) == pytest.approx(0)
    assert plan.f.facing(t({"x": 0, "z": 0}), t({"x": -10, "z": 0})) == pytest.approx(270)


def test_turn_in_place_keeps_the_middle(plan):
    t = plan.f.turn_in_place(plan.lua.table_from({"x": 0, "z": -9}), 0, 90, 18)
    # Facing +X now: the front rank centre is half the depth ahead of the middle.
    assert (t.x, t.z, t.bearing) == (pytest.approx(9), pytest.approx(-9), pytest.approx(90))
    back = plan.f.turn_in_place(plan.lua.table_from({"x": 0, "z": 0}), 10, -20, 10)
    assert back.bearing == pytest.approx(350)
