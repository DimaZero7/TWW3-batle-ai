"""apps.battle.adapter: health signature used by the stall rule."""
from tests.lua_runtime import load, new_runtime


def test_signature_keeps_small_damage_with_many_men():
    lua = new_runtime()
    lua.globals().battle = load(lua, "apps.battle.adapter")
    lua.execute("""
        local function unit(men, hp)
            return {number_of_men_alive = function() return men end, unary_hitpoints = function() return hp end}
        end
        local many = {}
        for i = 1, 20 do many[i] = unit(160, 1) end
        local before = battle.health_signature(many)
        many[20] = unit(160, 0.9998)
        local after = battle.health_signature(many)
        assert(before == '3200:200000', before)
        assert(after ~= before, after)
        many[1] = {number_of_men_alive = function() error('gone') end, unary_hitpoints = function() return 0 / 0 end}
        assert(battle.health_signature(many) == '3040:189998', battle.health_signature(many))
    """)
