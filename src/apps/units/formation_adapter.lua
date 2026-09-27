-- Own-unit movement and soldier positions. Own side only: soldier positions
-- of an enemy unit would reveal hidden men (docs/en/apps/observation.md).
-- CCO ManList.At(i).Position returns X, Y, Z as three numbers
-- (docs/en/game/units/evidence.md). Soldier coordinates are whole decimetres:
-- game Lua numbers print as single precision, so rounded metres would not stay short.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.MAX_SOLDIERS = 256

local function round(n) return math.floor(n * 10 + 0.5) / 10 end
local function decimetres(n) return math.floor(n * 10 + 0.5) end

local function call(unit, name)
    local ok, v = pcall(function() return unit[name](unit) end)
    if ok then return v end
    return nil
end

local function point(unit, name)
    local p = call(unit, name)
    if not p then return nil end
    local ok, x, z = pcall(function() return p:get_x(), p:get_z() end)
    if ok and finite(x) and finite(z) then return x, z end
    return nil
end

-- Compact movement sample; missing readings are left out.
function M.motion(unit)
    local x, z = point(unit, 'position')
    assert(x, 'Unit position unavailable')
    local row = {x = round(x), z = round(z)}
    local ox, oz = point(unit, 'ordered_position')
    if ox then row.ordered_x, row.ordered_z = round(ox), round(oz) end
    for _, name in ipairs({'bearing', 'ordered_bearing', 'ordered_width', 'number_of_men_alive'}) do
        local v = call(unit, name)
        if finite(v) then row[name] = round(v) end
    end
    for _, name in ipairs({'is_moving', 'is_moving_fast', 'is_idle', 'is_in_melee'}) do
        local v = call(unit, name)
        if type(v) == 'boolean' then row[name] = v end
    end
    return row
end

-- cco(unit, key) -> engine values. Returns {status = 'ok', count, xz_dm = {x1, z1, ...}}
-- or {status = 'unavailable', reason}.
function M.soldiers(cco, unit)
    local ok, size = pcall(cco, unit, 'ManList.Size')
    if not ok then return {status = 'unavailable', reason = 'call_failed'} end
    if not (finite(size) and size % 1 == 0 and size >= 0 and size <= M.MAX_SOLDIERS) then
        return {status = 'unavailable', reason = 'invalid_size'}
    end
    local xz = {}
    for i = 0, size - 1 do
        local got, x, _, z = pcall(cco, unit, 'ManList.At(' .. i .. ').Position')
        if not (got and finite(x) and finite(z)) then
            return {status = 'unavailable', reason = 'invalid_position', index = i}
        end
        xz[#xz + 1] = decimetres(x)
        xz[#xz + 1] = decimetres(z)
    end
    return {status = 'ok', count = size, xz_dm = xz}
end

return M
