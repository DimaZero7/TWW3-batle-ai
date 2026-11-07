"""apps.battle.box_pick: the box's enemies are the units whose banner point stands inside the box's rectangle."""
from tests.lua_runtime import load, new_runtime


def test_the_units_inside_the_box():
    lua = new_runtime()
    lua.globals().bp = load(lua, "apps.battle.box_pick")
    lua.execute("""
        local pts = {{'a', 100, 100}, {'b', 150, 120}, {'c', 400, 100}, {'d', 100, 300}}
        local got = bp.inside({x = 90, y = 90, w = 80, h = 40}, pts)
        assert(table.concat(got, '') == 'ab', table.concat(got, ''))
        assert(#bp.inside({x = 0, y = 0, w = 10, h = 10}, pts) == 0)
        -- the edges count
        assert(table.concat(bp.inside({x = 100, y = 100, w = 300, h = 0}, pts), '') == 'ac')
    """)
