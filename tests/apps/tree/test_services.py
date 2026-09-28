"""apps.tree: switches; a node off switches off its subtree and every later phase."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def tree():
    lua = new_runtime()
    m = load(lua, "apps.tree.services")
    m.make = lambda switches=None: m.new(lua.table_from(switches or {}))
    return m


def test_everything_on_but_logistics_by_default(tree):
    t = tree.make()
    assert tree.on(t, "approach") and tree.on(t, "align") and tree.on(t, "formation_window")
    assert not tree.on(t, "logistics")
    assert tree.on(tree.make({"logistics": True}), "logistics")


def test_a_node_off_switches_off_its_subtree(tree):
    t = tree.make({"approach": False, "logistics": True})
    assert not tree.on(t, "approach")
    for branch in ("align", "logistics", "safe_detour", "under_fire_stop"):
        assert not tree.on(t, branch)
    assert tree.on(t, "deploy") and tree.on(t, "map_fit")


def test_only_the_branch_is_off(tree):
    t = tree.make({"align": False})
    assert not tree.on(t, "align") and tree.on(t, "approach") and tree.on(t, "safe_detour")


def test_bad_switches(tree):
    with pytest.raises(Exception, match="Unknown tree node"):
        tree.make({"flank_charge": False})
    with pytest.raises(Exception, match="true or false"):
        tree.make({"align": "no"})
    with pytest.raises(Exception, match="first phase"):
        tree.make({"deploy": False})
