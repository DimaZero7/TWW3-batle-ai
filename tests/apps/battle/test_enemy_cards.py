"""apps.battle.enemy_cards places: on the own row after a separator when it fits; else rows above with a band kept free
for the own groups' frames (and the enemy's), inside the strip's edges."""
from tests.lua_runtime import load, new_runtime


def test_the_enemy_cards_leave_room_for_the_group_frames():
    lua = new_runtime()
    lua.globals().ec = load(lua, "apps.battle.enemy_cards")
    lua.execute("""
        local function row(n, x0, y, w, h)
            local out = {}
            for k = 1, n do out[k] = {x = x0 + (k - 1) * w, y = y, w = w, h = h} end
            return out
        end
        -- 7 own cards of 60: 4 enemy cards fit after them on the same row (7 would not: above)
        local p, sep = ec.places(row(7, 589, 995, 60, 130), 4, 255, 1345)
        assert(sep[5] == 'row' and p[1][2] == 995 and p[1][1] == 589 + 7 * 60 + ec.GAP)
        assert(p[4][1] + 60 <= 1345)
        local p3, sep3 = ec.places(row(7, 589, 995, 60, 130), 7, 255, 1345)
        assert(sep3[5] == "above" and p3[1][2] + 130 <= 995 - ec.FRAME_H)
        -- 20 own cards of 54: the enemy above, the own frames' band free, inside the strip
        local p2, sep2 = ec.places(row(20, 254, 995, 54, 130), 20, 255, 1345)
        assert(sep2[5] == 'above')
        for _, q in ipairs(p2) do
            assert(q[2] + 130 <= 995 - ec.FRAME_H, 'covers the own frames: ' .. q[2])
            assert(q[1] >= 255 and q[1] + 54 <= 1345 + 1)
        end
    """)


def test_a_groups_cards_stand_in_a_row_groups_first_by_number_then_the_rest():
    lua = new_runtime()
    lua.globals().ec = load(lua, "apps.battle.enemy_cards")
    lua.execute("""
        local names = {'a', 'b', 'c', 'd', 'e', 'f'}
        -- group 1 of the 2nd and the 6th card, group 2 of the 4th: 1, 1, 2, then the rest in their first order
        local o = ec.order(names, {[1] = {'f', 'b'}, [2] = {'d'}}, 6)
        assert(table.concat(o, '') == 'bfdace', table.concat(o, ''))
        -- a group dissolved: the first order back
        assert(table.concat(ec.order(names, {[2] = {'d'}}, 6), '') == 'dabcef')
        assert(table.concat(ec.order(names, {}, 6), '') == 'abcdef')
        -- an empty group left by a move: nothing
        assert(table.concat(ec.order(names, {[1] = {}, [3] = {'a'}}, 6), '') == 'abcdef')
    """)
