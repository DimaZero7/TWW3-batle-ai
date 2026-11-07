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


def test_a_box_takes_a_unit_by_its_region_under_the_banner():
    lua = new_runtime()
    lua.globals().bp = load(lua, "apps.battle.box_pick")
    lua.execute("""
        local pts = {{'a', 100, 100}, {'b', 300, 100}}
        local reg = {left = -40, right = 40, top = 15, bottom = 170}
        -- a box over a's men (below its banner), not over the banner itself
        assert(table.concat(bp.covered_by_region({x = 90, y = 200, w = 20, h = 20}, pts, reg), '') == 'a')
        -- a box over both banners' rows
        assert(table.concat(bp.covered_by_region({x = 50, y = 150, w = 300, h = 10}, pts, reg), '') == 'ab')
        -- above the banners: nothing
        assert(#bp.covered_by_region({x = 0, y = 0, w = 400, h = 80}, pts, reg) == 0)
    """)


def test_a_click_on_the_field_hits_the_unit_whose_region_holds_it():
    lua = new_runtime()
    lua.globals().bp = load(lua, "apps.battle.box_pick")
    lua.execute("""
        local reg = {left = -40, right = 40, top = 15, bottom = 170}
        local pts = {{'a', 100, 100}, {'b', 160, 100}, {'c', 300, 100}}
        assert(bp.hit(pts, 300, 200, reg) == 'c')
        -- in two regions: the nearer banner's column
        assert(bp.hit(pts, 120, 180, reg) == 'a')
        assert(bp.hit(pts, 145, 180, reg) == 'b')
        assert(bp.hit(pts, 230, 180, reg) == nil)
        assert(bp.hit(pts, 100, 50, reg) == nil)       -- above the banner
    """)


def test_a_chat_message_is_one_trimmed_line():
    lua = new_runtime()
    lua.globals().lc = load(lua, "apps.battle.label_chat")
    lua.execute("""
        assert(lc.clean('  hold the flank\\r\\n') == 'hold the flank')
        assert(lc.clean('a\\nb') == 'a b')
        assert(lc.clean('   ') == nil and lc.clean(nil) == nil)
        assert(#lc.clean(string.rep('x', 1000)) == lc.MAX)
    """)
