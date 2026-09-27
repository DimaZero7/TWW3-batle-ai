-- Trusted private telemetry for post-terminal export research.
-- Never a policy input. Reads both sides, including hidden enemies.
local value = require('apps.core.value')

local M = {version = 1, profile = 'post-battle-telemetry-probe-v1'}
local finite = value.finite

local function attempt(out, key, kind, fn)
    local ok, result = pcall(fn)
    if ok and type(result) == kind and (kind ~= 'number' or finite(result)) then
        out[key] = result
    else
        out[key .. '_status'] = 'unknown'
    end
end

local function pos(u)
    local ok, p = pcall(function() return u:position() end)
    if not ok or not p then return nil end
    local good, x, y, z = pcall(function() return p:get_x(), p:get_y(), p:get_z() end)
    if not good or not finite(x) or not finite(z) then return nil end
    return {x = x, y = finite(y) and y or nil, z = z}
end

-- sides[i] = {alliance = <engine alliance>, units = {{unit=..., roster={id=...}}, ...}}
-- cco(unit, key) reads a CcoBattleUnit field.
function M.new(sides, cco)
    assert(type(sides) == 'table' and sides[1] and sides[2] and type(cco) == 'function',
        'two-side trusted registry required')
    local state = {sides = sides, cco = cco, ids = {}, ids_by_uid = {}, hits = {},
        previous_health = {}, previous_ammo = {}}
    for side = 1, 2 do
        for _, item in ipairs(sides[side].units) do
            assert(item.unit and item.roster and item.roster.id and not state.ids[item.unit],
                'unique unit registry required')
            state.ids[item.unit] = item.roster.id
            local ok, uid = pcall(function() return item.unit:unique_ui_id() end)
            if ok and uid ~= nil then
                uid = tostring(uid)
                assert(not state.ids_by_uid[uid], 'unique native unit id required')
                state.ids_by_uid[uid] = item.roster.id
            end
        end
    end
    return state
end

-- Engine handles are not guaranteed to be the same Lua object each call,
-- so fall back to unique_ui_id.
local function resolve_id(state, u)
    local id = state.ids[u]
    if id then return id end
    local ok, uid = pcall(function() return u:unique_ui_id() end)
    if ok and uid ~= nil then return state.ids_by_uid[tostring(uid)] end
end

function M.entity_hit(state, event, time_ms)
    assert(type(time_ms) == 'number' and finite(time_ms))
    local row = {time_ms = time_ms, event_name_status = 'unknown', target_status = 'unknown',
        source_status = 'unavailable_by_native_event'}
    attempt(row, 'event_name', 'string', function() return event:get_name() end)
    local ok, target = pcall(function() return event:get_unit() end)
    if ok and target then
        row.target_id = resolve_id(state, target)
        row.target_status = row.target_id and 'known' or 'unmapped'
    end
    attempt(row, 'artillery', 'boolean', function() return event:get_bool1() end)
    state.hits[#state.hits + 1] = row
    return row
end

function M.drain_hits(state)
    local result = state.hits
    state.hits = {}
    return result
end

local CCO_NUMBERS = {'HealthValue', 'HealthMax', 'NumEntities', 'PercentHpLostRecently',
    'DamageInflictedRecently', 'NumKills', 'MoralePercent', 'MoraleState'}
local CCO_BOOLEANS = {'IsTakingDamage', 'IsUnderMissileAttack', 'IsFiringMissiles',
    'IsWavering', 'IsRouting', 'IsShattered'}
local NATIVE_NUMBERS = {'number_of_men_alive', 'ammo_left', 'missile_range'}
local NATIVE_BOOLEANS = {'is_in_melee', 'is_moving', 'is_routing', 'is_wavering',
    'is_shattered', 'is_under_missile_attack'}

function M.sample(state, time_ms)
    assert(type(time_ms) == 'number' and finite(time_ms))
    local frame = {time_ms = time_ms, sides = {}, entity_hits = M.drain_hits(state)}
    local rows, handles = {}, {}
    for side = 1, 2 do
        local out = {side = side, units = {}}
        frame.sides[#frame.sides + 1] = out
        for _, item in ipairs(state.sides[side].units) do
            local u = item.unit
            local row = {id = item.roster.id, side = side, position = pos(u), visibility = {}}
            for observer = 1, 2 do
                attempt(row.visibility, 'side_' .. observer, 'boolean', function()
                    return u:is_visible_to_alliance(state.sides[observer].alliance)
                end)
            end
            for _, field in ipairs(CCO_NUMBERS) do
                attempt(row, field, 'number', function() return state.cco(u, field) end)
            end
            for _, field in ipairs(CCO_BOOLEANS) do
                attempt(row, field, 'boolean', function() return state.cco(u, field) end)
            end
            for _, field in ipairs(NATIVE_NUMBERS) do
                attempt(row, field, 'number', function() return u[field](u) end)
            end
            local previous_ammo = state.previous_ammo[row.id]
            if finite(previous_ammo) and finite(row.ammo_left) then
                row.ammo_delta = math.max(0, previous_ammo - row.ammo_left)
            end
            if finite(row.ammo_left) then state.previous_ammo[row.id] = row.ammo_left end
            for _, field in ipairs(NATIVE_BOOLEANS) do
                attempt(row, field, 'boolean', function() return u[field](u) end)
            end
            local ok, target = pcall(function() return u:current_target() end)
            if not ok then
                row.current_target_status = 'unknown'
            elseif not target then
                row.current_target_status = 'none'
            else
                row.current_target_id = resolve_id(state, target)
                row.current_target_status = row.current_target_id and 'known' or 'unmapped'
            end
            local ordered_ok, ordered = pcall(function() return u:ordered_position() end)
            if ordered_ok and ordered then
                local good, x, y, z = pcall(function()
                    return ordered:get_x(), ordered:get_y(), ordered:get_z()
                end)
                if good and finite(x) and finite(z) then
                    row.ordered_position = {x = x, y = finite(y) and y or nil, z = z}
                else
                    row.ordered_position_status = 'unknown'
                end
            else
                row.ordered_position_status = 'unknown'
            end
            row.distances = {}
            rows[row.id] = row
            handles[row.id] = u
            out.units[#out.units + 1] = row
        end
    end
    local damaged = {}
    for id, row in pairs(rows) do
        local prior = state.previous_health[id]
        if finite(prior) and finite(row.HealthValue) and row.HealthValue < prior then
            damaged[#damaged + 1] = id
        end
        if finite(row.HealthValue) then state.previous_health[id] = row.HealthValue end
    end
    for source_id, source in pairs(rows) do
        if (source.DamageInflictedRecently or 0) > 0 or source.IsFiringMissiles == true
            or source.is_in_melee == true then
            for _, target_id in ipairs(damaged) do
                if target_id ~= source_id then
                    attempt(source.distances, target_id, 'number', function()
                        return handles[source_id]:unit_distance(handles[target_id])
                    end)
                end
            end
        end
    end
    return frame
end

return M
