"""apps.tactics: the tactical trunk. Its decisions are checked on whole battles in
tests/golden and branch by branch in tests/tools/test_tree_branches.py; here the
manoeuvre bookkeeping and the switches."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def tt():
    lua = new_runtime()
    t = load(lua, "apps.tactics.services")
    tree = load(lua, "apps.tree.services")
    t.make = lambda switches=None: t.new(tree.new(lua.table_from(switches or {})))
    return t


def test_a_manoeuvre_is_under_way_until_finished(tt):
    s = tt.make()
    assert not tt.busy(s)
    s.commander.start("approach", 0, None)
    assert tt.busy(s)
    tt.finish(s, 1000, "stopped")
    assert not tt.busy(s)


def test_under_fire_breaks_off_only_with_the_branch(tt):
    assert tt.interrupt(tt.make(), True) and not tt.interrupt(tt.make(), False)
    assert not tt.interrupt(tt.make({"under_fire_stop": False}), True)
    assert not tt.interrupt(tt.make({"approach": False}), True)


def test_switches_are_the_tree(tt):
    s = tt.make({"align": False})
    assert not tt.on(s, "align") and tt.on(s, "approach") and not tt.on(s, "logistics")
