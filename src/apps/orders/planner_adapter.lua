-- Hands units to the game's own tactical AI instead of issuing unit orders.
-- Uses CA's script_ai_planner (lib_battle_script_ai_planner.lua), the same
-- mechanism generated battles use for AI armies: it wraps
-- alliance:create_ai_unit_planner() and lets the engine fight the units.
-- Falls back to the engine planner directly if the library is not loaded.
local M = {}

local function closest(units, position)
    local best, best_distance
    for _, u in ipairs(units) do
        local ok, standing = pcall(function()
            return u:number_of_men_alive() > 0 and not u:is_routing() and not u:is_shattered()
        end)
        if ok and standing then
            local d = u:position():distance(position)
            if not best_distance or d < best_distance then best, best_distance = u, d end
        end
    end
    return best
end

-- units / enemies: engine units. Returns a handle with attack() to (re)issue
-- "attack the enemy force" and mode = 'script_ai_planner' | 'engine_planner'.
function M.hand_over(bm, name, alliance, units, enemies)
    if script_ai_planner and bm.get_scriptunit_for_unit then
        local own, foes = {}, {}
        for _, u in ipairs(units) do own[#own + 1] = bm:get_scriptunit_for_unit(u) end
        for _, u in ipairs(enemies) do foes[#foes + 1] = bm:get_scriptunit_for_unit(u) end
        local sai = script_ai_planner:new(name, own)
        assert(sai, 'script_ai_planner:new failed')
        return {mode = 'script_ai_planner', attack = function() sai:attack_force(foes) end,
            release = function() sai:release() end}
    end
    local planner = alliance:create_ai_unit_planner()
    for _, u in ipairs(units) do planner:add_units(u) end
    return {mode = 'engine_planner',
        attack = function()
            local anchor = closest(units, enemies[1]:position()) or units[1]
            local target = closest(enemies, anchor:position())
            if target then planner:attack_unit(target) end
        end,
        release = function() end}
end

return M
