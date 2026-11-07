"""apps.battle.screen_proj: the projection fitted to banners gives back the camera's field of view and the banners'
height; a formation's quad on the screen overlaps a box by any part, as the game's box takes a unit."""
from tests.lua_runtime import load, new_runtime


def test_the_fit_finds_the_field_of_view_and_the_banner_height():
    lua = new_runtime()
    lua.globals().sp = load(lua, "apps.battle.screen_proj")
    lua.execute("""
        local b = sp.basis({0, 120, -300}, {0, 0, 100})
        local pairs = {}
        for i = 0, 5 do for j = 0, 3 do
            local x, z, y = -200 + i * 80, -50 + j * 90, (i * 7 + j * 3) % 11
            local sx, sy = sp.project(b, 42, 1600, 1200, {x, y + 14, z})
            pairs[#pairs + 1] = {x, y, z, sx + 20, sy - 6}
        end end
        local fov, hb, dx, dy, err = sp.fit(b, 1600, 1200, pairs)
        assert(err < 2, err)
        assert(math.abs(dx - 20) < 3 and math.abs(dy + 6) < 3, dx .. ' ' .. dy)
        assert(math.abs(fov - 42) < 1 and math.abs(hb - 14) < 1.5, fov .. ' ' .. hb)
        -- what the box needs: the ground under the units where it shows (the banner's height and its screen offset
        -- trade off a little against each other; the ground's place does not)
        for _, g in ipairs({{-150, 2, 0}, {100, 5, 200}, {0, 0, -100}}) do
            local tx, ty = sp.project(b, 42, 1600, 1200, g)
            local x, y = sp.project(b, fov, 1600, 1200, g)
            assert(math.abs(x - tx) < 4 and math.abs(y - ty) < 4, x .. ' ' .. y .. ' vs ' .. tx .. ' ' .. ty)
        end
    """)


def test_a_box_takes_a_unit_by_any_part_of_its_formation():
    lua = new_runtime()
    lua.globals().sp = load(lua, "apps.battle.screen_proj")
    lua.execute("""
        local poly = {{100, 100}, {200, 110}, {190, 150}, {95, 140}}
        assert(sp.overlap({x = 180, y = 130, w = 50, h = 50}, poly))            -- a corner inside the box
        assert(sp.overlap({x = 120, y = 115, w = 10, h = 10}, poly))            -- the box inside the formation
        assert(sp.overlap({x = 140, y = 60, w = 10, h = 200}, poly))            -- a thin box across it
        assert(not sp.overlap({x = 300, y = 300, w = 50, h = 50}, poly))
        -- a quad: 0 = +z front, width along x
        local q = sp.quad(0, 0, 0, 0, 20, 4)
        local xs = {}
        for _, c in ipairs(q) do xs[#xs + 1] = c[1] end
        table.sort(xs)
        assert(math.abs(xs[1] + 10) < 1e-6 and math.abs(xs[4] - 10) < 1e-6)
    """)
