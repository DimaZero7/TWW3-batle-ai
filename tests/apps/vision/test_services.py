"""apps.vision: groups of visible enemy units.

Level 1: the gap and linking functions on synthetic blocks.
Level 2: tests/cases/vision/*.json — real layouts of the game AI in defence
(enemy-layout runs): the whole army is one group; shifted copies split.
"""
import json
import time
from pathlib import Path

import pytest

from tests.lua_runtime import load, new_runtime
from tools import roster
from tools.sim.formation import roster_units

CASES = sorted((Path(__file__).parents[2] / "cases" / "vision").glob("*.json"))


@pytest.fixture(scope="module")
def lua():
    runtime = new_runtime()
    runtime.globals().vision = load(runtime, "apps.vision.services")
    runtime.globals().assessment = load(runtime, "apps.assessment.services")
    return runtime


@pytest.fixture(scope="module")
def strength_of(lua):
    """Strength per unit key, from the real roster through apps.assessment."""
    entries = roster.load_roster()
    cache = {}

    def get(key):
        if key not in cache:
            unit = roster_units({"units": [{"key": key, "count": 1}]}, entries)[0]
            cache[key] = lua.globals().assessment.strength(lua.table_from(unit, recursive=True))
        return cache[key]
    return get


def block(uid, x, z, width=10, depth=2, strength=100):
    pts = []
    for i in range(0, width + 1, 2):
        for j in range(0, depth + 1, 2):
            pts += [x + i, z - j]
    return {"id": uid, "points": pts, "strength": strength}


def groups(lua, units, params=None, total=None):
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    result = lua.globals().vision.groups(lua.table_from(units, recursive=True),
                                        lua.table_from(params or {}), total)
    return json.loads(encode(result))


def test_gap_is_between_closest_soldiers(lua):
    one = groups(lua, [block("a", 0, 0), block("b", 25, 0)], {"link_gap_m": 14})
    two = groups(lua, [block("a", 0, 0), block("b", 25, 0)], {"link_gap_m": 16})
    # Block a spans x 0..10, block b 25..35: the gap is 15 m.
    assert len(one["groups"]) == 2 and len(two["groups"]) == 1


def test_links_chain_and_a_lone_unit_is_a_group(lua):
    units = [block("a", 0, 0), block("b", 20, 0), block("c", 40, 0), block("far", 300, 0, strength=10)]
    r = groups(lua, units)
    assert [sorted(h["ids"]) for h in r["groups"]] == [["a", "b", "c"], ["far"]]
    assert r["main"]["ids"] == r["groups"][0]["ids"]


def test_strongest_group_is_the_main_army_not_the_biggest(lua):
    units = [block("s1", 0, 0, strength=50), block("s2", 15, 0, strength=50), block("s3", 30, 0, strength=50),
             block("elite", 400, 0, strength=500)]
    r = groups(lua, units)
    assert r["main"]["ids"] == ["elite"] and r["groups"][1]["strength"] == 150


def test_centre_bounds_and_seen_share(lua):
    r = groups(lua, [block("a", 0, 0, strength=300), block("b", 20, 0, strength=100)], total=800)
    main = r["main"]
    # Middles at x 5 and 25; weighted 3:1 -> 10.
    assert main["centre"]["x"] == pytest.approx(10) and main["bounds"]["max_x"] == 30
    assert r["seen_share"] == pytest.approx(0.5) and r["link_gap_m"] == 30


def test_nothing_seen(lua):
    r = groups(lua, [])
    assert r["groups"] == {} or r["groups"] == [] and r.get("main") is None


def test_bad_input(lua):
    with pytest.raises(Exception, match="no soldier points"):
        groups(lua, [{"id": "x", "points": [], "strength": 1}])
    with pytest.raises(Exception, match="no strength"):
        groups(lua, [{"id": "x", "points": [0, 0]}])


def case_units(case, strength_of, shift=None):
    units = []
    for u in case["units"]:
        pts = [v / 10 for v in u["points_dm"]]
        if shift and u["id"] in shift:
            pts = [v + (shift[u["id"]] if i % 2 == 0 else 0) for i, v in enumerate(pts)]
        units.append({"id": u["id"], "points": pts, "strength": strength_of(u["key"])})
    return units


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_game_ai_army_is_one_group(path, lua, strength_of):
    case = json.loads(path.read_text(encoding="utf-8"))
    units = case_units(case, strength_of)
    total = sum(u["strength"] for u in units)
    started = time.perf_counter()
    r = groups(lua, units, total=total)
    elapsed = time.perf_counter() - started
    assert len(r["groups"]) == case["expect"]["groups"]
    assert len(r["main"]["ids"]) == case["expect"]["main_units"]
    assert r["seen_share"] == pytest.approx(case["expect"]["seen_share"])
    assert elapsed < 2.0, f"groups took {elapsed:.2f} s"


def test_a_detached_group_becomes_its_own_group(lua, strength_of):
    case = json.loads((Path(__file__).parents[2] / "cases" / "vision" / "game_ai_big_13.json").read_text())
    # The right wing (units whose middle is past x = 30) is moved 400 m away.
    middle_x = {u["id"]: sum(u["points_dm"][0::2]) / len(u["points_dm"][0::2]) / 10 for u in case["units"]}
    wing = {uid for uid, x in middle_x.items() if x > 30}
    r = groups(lua, case_units(case, strength_of, {uid: 400 for uid in wing}))
    assert len(r["groups"]) == 2
    assert {frozenset(h["ids"]) for h in r["groups"]} == {frozenset(wing), frozenset(set(middle_x) - wing)}


def test_there_are_cases():
    assert len(CASES) == 6


def test_picture_has_both_sides(lua):
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    data = {"own": [block("o1", 0, 0), block("o2", 15, 0), block("scout", 500, 0, strength=5)],
            "enemy": [block("e1", 0, 300, strength=200)], "enemy_total": 400}
    r = json.loads(encode(lua.globals().vision.picture(lua.table_from(data, recursive=True))))
    assert len(r["own"]["groups"]) == 2 and sorted(r["own"]["main"]["ids"]) == ["o1", "o2"]
    assert r["own"]["seen_share"] == 1 and r["enemy"]["seen_share"] == pytest.approx(0.5)
    with pytest.raises(Exception, match="Both sides required"):
        lua.globals().vision.picture(lua.table_from({"own": {}}))


def test_group_facing_is_the_mean_bearing(lua):
    a = dict(block("a", 0, 0), bearing=350)
    b = dict(block("b", 15, 0), bearing=10)
    r = heaps_or_groups(lua, [a, b])
    assert r["main"]["facing"] == pytest.approx(0, abs=1e-6) or r["main"]["facing"] == pytest.approx(360)
    assert "facing" not in heaps_or_groups(lua, [block("c", 0, 0)])["main"]


def heaps_or_groups(lua, units):
    encode = lua.eval("function(t) return require('apps.core.json').encode(t) end")
    return json.loads(encode(lua.globals().vision.groups(lua.table_from(units, recursive=True), None, None)))
