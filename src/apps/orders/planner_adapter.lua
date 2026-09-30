-- Hands units to the game's own tactical AI instead of issuing unit orders.
-- Uses CA's script_ai_planner (lib_battle_script_ai_planner.lua), the same
-- mechanism generated battles use for AI armies: it wraps
-- alliance:create_ai_unit_planner() and lets the engine fight the units.
-- Falls back to the engine planner directly if the library is not loaded.
--
-- A unit that routs and rallies is not led by the planner again: it stands
-- idle to the end of the battle (arena runs, 30.09.2026; the game's own AI
-- does fight on with its rallied units). check_rallies() finds such units,
-- takes them out of the planner, puts them back and repeats the last order.
--
-- The planner also leaves units idle under arrows: standing, no target, not in
-- melee, losing men (our side 15-75% of the time it is shot at; the game's AI
-- 0-3%; arena runs 30.09.2026). check_idle() gives such a unit, after IDLE_MS,
-- first back to the planner with its order; if it still stands under fire, to a
-- planner of its own that attacks the nearest enemy (archers walk into range).
local M = {}

M.REJOIN_MS = 600   -- the library adds a unit to the engine planner 400 ms after add_sunits
M.IDLE_MS = 6000    -- idle under fire this long before a kick
M.KICK_GAP_MS = 15000

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

local function routing(u)
    local ok, r = pcall(function() return u:is_routing() or u:is_shattered() end)
    return ok and r == true
end

local function read(fn)
    local ok, v = pcall(fn)
    if ok then return v end
    return nil
end

-- Not moving, not in melee, no target (so not shooting either).
local function idle(u)
    return read(function() return u:is_moving() end) == false
        and read(function() return u:is_in_melee() end) == false
        and read(function() return u:current_target() end) == nil
end

local function standing(u)
    local ok, s = pcall(function()
        return u:number_of_men_alive() > 0 and not u:is_routing() and not u:is_shattered()
    end)
    return ok and s == true
end

-- units / enemies: engine units. Returns a handle:
--   attack()            (re)issue "attack the enemy force";
--   defend(centre, r)   defend a position;
--   check_rallies()     call every tick: rallied units rejoin; returns their indices;
--   check_idle(now_ms)  call every tick: kicks units idle under fire; returns {{unit = i, stage = 1|2}};
--   release(), mode = 'script_ai_planner' | 'engine_planner'.
function M.hand_over(bm, name, alliance, units, enemies)
    local handle = {units = units, rejoined = 0, kicks = 0, solo = 0}
    local was_routing, last = {}, nil
    local rejoin_unit, stand_alone
    if script_ai_planner and bm.get_scriptunit_for_unit then
        local own, foes = {}, {}
        for i, u in ipairs(units) do own[i] = bm:get_scriptunit_for_unit(u) end
        for _, u in ipairs(enemies) do foes[#foes + 1] = bm:get_scriptunit_for_unit(u) end
        local sai = script_ai_planner:new(name, own)
        assert(sai, 'script_ai_planner:new failed')
        handle.mode = 'script_ai_planner'
        handle.attack = function()
            last = function() sai:attack_force(foes) end
            last()
        end
        handle.defend = function(centre, radius_m)
            last = function() sai:defend_position(centre, radius_m) end
            last()
        end
        handle.release = function() sai:release() end
        rejoin_unit = function(i)
            if own[i].planner == sai then sai:remove_sunits(own[i]) end
            sai:add_sunits(own[i])
        end
        stand_alone = function(i)
            handle.solo = handle.solo + 1
            local solo = script_ai_planner:new(name .. '_solo_' .. handle.solo, {own[i]})
            if solo then
                bm:callback(function() pcall(function() solo:attack_force(foes) end) end, M.REJOIN_MS,
                    name .. '_solo_order_' .. handle.solo)
            end
        end
    else
        local planner = alliance:create_ai_unit_planner()
        for _, u in ipairs(units) do planner:add_units(u) end
        handle.mode = 'engine_planner'
        handle.attack = function()
            last = function()
                local anchor = closest(units, enemies[1]:position()) or units[1]
                local target = closest(enemies, anchor:position())
                if target then planner:attack_unit(target) end
            end
            last()
        end
        handle.defend = function(centre, radius_m)
            last = function() planner:defend_position(centre, radius_m) end
            last()
        end
        handle.release = function() end
        local solos = {}
        rejoin_unit = function(i)
            if solos[i] then
                pcall(function() solos[i]:remove_units(units[i]) end)
                solos[i] = nil
            end
            planner:remove_units(units[i])
            planner:add_units(units[i])
        end
        stand_alone = function(i)
            handle.solo = handle.solo + 1
            planner:remove_units(units[i])
            local solo = alliance:create_ai_unit_planner()
            solo:add_units(units[i])
            solos[i] = solo
            local target = closest(enemies, units[i]:position())
            if target then solo:attack_unit(target) end
        end
    end

    function handle.check_rallies()
        local rallied = {}
        for i, u in ipairs(units) do
            local r = routing(u)
            if was_routing[i] and not r and standing(u) then rallied[#rallied + 1] = i end
            was_routing[i] = r
        end
        if #rallied == 0 then return rallied end
        for _, i in ipairs(rallied) do rejoin_unit(i) end
        handle.rejoined = handle.rejoined + #rallied
        -- Repeat the last order once the units are back in the engine planner.
        if last then
            bm:callback(function() pcall(last) end, M.REJOIN_MS, name .. '_rejoin_' .. handle.rejoined)
        end
        return rallied
    end

    local prev_hp, idle_since, last_kick, stage = {}, {}, {}, {}
    function handle.check_idle(now_ms)
        local kicked = {}
        for i, u in ipairs(units) do
            local hp = read(function() return u:unary_hitpoints() end)
            local losing = hp and prev_hp[i] and prev_hp[i] - hp > 0.0005
            prev_hp[i] = hp
            local still = standing(u) and idle(u)
            if still and losing then
                idle_since[i] = idle_since[i] or now_ms
            elseif not still then
                idle_since[i], stage[i] = nil, nil
            end
            if idle_since[i] and now_ms - idle_since[i] >= M.IDLE_MS
                and now_ms - (last_kick[i] or -1e9) >= M.KICK_GAP_MS then
                stage[i] = (stage[i] or 0) + 1
                if stage[i] == 1 then
                    rejoin_unit(i)
                    if last then
                        bm:callback(function() pcall(last) end, M.REJOIN_MS, name .. '_kick_' .. handle.kicks)
                    end
                else
                    stand_alone(i)
                end
                handle.kicks = handle.kicks + 1
                last_kick[i], idle_since[i] = now_ms, nil
                kicked[#kicked + 1] = {unit = i, stage = math.min(stage[i], 2)}
            end
        end
        return kicked
    end

    return handle
end

return M
