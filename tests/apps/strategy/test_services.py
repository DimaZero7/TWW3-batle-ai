"""apps.strategy: conditions, roles, choice and the decision contract (level 1: functions)."""
import pytest

from tests.lua_runtime import load, new_runtime


def features(**over):
    f = {"role": "attack",
         "own": {"melee_power": 100, "ranged_power": 400, "arc_share": 1, "count": {"infantry": 1}},
         "enemy": {"melee_power": 200, "ranged_power": 100, "infantry_armour": 30, "count": {"infantry": 2}},
         "ranged_ratio": 4}
    for path, v in over.items():
        side, _, key = path.partition("__")
        if key:
            f[side][key] = v
        else:
            f[side] = v
    return f


@pytest.fixture
def s():
    lua = new_runtime()
    m = load(lua, "apps.strategy.services")
    m.t = lambda x: lua.table_from(x, recursive=True)
    m.contract = load(lua, "apps.strategy.contract")
    m.lua = lua
    return m


def failed(s, f):
    key, candidates = s.select(s.t(f))
    return key, list(candidates[1].failed.values())


def test_all_six_conditions_hold(s):
    assert failed(s, features()) == ("wall_and_arc", [])


@pytest.mark.parametrize("over, condition", [
    ({"enemy__melee_power": 50}, "enemy_infantry_stronger"),
    ({"own__count": {"infantry": 0}}, "own_infantry_to_hold"),
    ({"enemy__ranged_power": 900}, "more_archers"),
    ({"enemy__infantry_armour": 90}, "archers_pierce"),
    ({"own__arc_share": 0.2}, "arc_fire"),
    ({"role": "defend"}, "we_attack"),
])
def test_each_condition_can_reject(s, over, condition):
    assert failed(s, features(**over)) == ("none", [condition])


def test_roles(s):
    entry = s.find("wall_and_arc")
    role = entry.role
    t = s.t
    assert role("lord", t({})) == "lord" and role("infantry", t({})) == "wall"
    assert role("shooter", t({"fire": "arc"})) == "arc" and role("shooter", t({"fire": "direct"})) == "other"
    assert role("other", t({})) == "other"


def test_decision_contract(s):
    units = s.t([{"id": "a"}, {"id": "b"}])
    good = s.t({"strategy": "wall_and_arc", "candidates": {}, "layout": "line_and_blocks",
                "roles": {"a": "wall", "b": "arc"}})
    s.contract.check_decision(good, units)
    s.contract.check_decision(s.t({"strategy": "none", "candidates": {}}), units)
    bad = s.t({"strategy": "wall_and_arc", "candidates": {}, "layout": "line_and_blocks", "roles": {"a": "wall"}})
    with pytest.raises(Exception, match="has no valid role"):
        s.contract.check_decision(bad, units)
