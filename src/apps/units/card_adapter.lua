-- Unit card as the player sees it (CCO UnitDetailsContext.StatList) and the
-- unit's static profile from native calls. The player can open a card of any
-- unit, but the caller still decides which units it may inspect
-- (docs/en/apps/observation.md). Values are flattened for telemetry:
-- a known value as is, an unknown one as 'unknown:<reason>'.
local value = require('apps.core.value')

local M = {}

M.MAX_STATS = 100
M.STAT_FIELDS = {'Value', 'DisplayedValue', 'ValueBase'}
-- UnitDetailsContext fields outside StatList; names not yet seen in game
-- are reported as unknown, never guessed.
M.DETAILS = {'Mass', 'Name'}
-- CcoBattleUnit fields of the whole unit, read with the card.
M.UNIT = {'HealthMax', 'NumEntitiesInitial'}
M.KIND_FLAGS = {'is_infantry', 'is_cavalry', 'is_pikemen', 'is_anti_cavalry_infantry', 'is_lancers',
    'is_chariot', 'is_war_beasts', 'is_elephants', 'is_artillery', 'is_war_machine'}
M.NUMBERS = {'initial_number_of_men', 'slow_speed', 'fast_speed', 'missile_range', 'starting_ammo'}

local function flat(r)
    if r.status == 'known' then return r.value end
    return 'unknown:' .. r.reason
end

-- Any scalar the card returns (numbers, texts, flags).
local function scalar(fn)
    local ok, v = pcall(fn)
    if not ok then return 'unknown:read_error' end
    if v == nil then return 'unknown:nil' end
    local t = type(v)
    if t == 'number' then return value.finite(v) and v or 'unknown:nonfinite' end
    if t == 'string' or t == 'boolean' then return v end
    return 'unknown:type_' .. t
end

-- cco(unit, key) -> engine values. Returns {status = 'ok', list = {{key, Value, DisplayedValue, ValueBase}}}
-- in card order, or {status = 'unavailable', reason}.
function M.stats(cco, unit)
    local count = value.read(function() return cco(unit, 'UnitDetailsContext.StatList.Size') end, 'number')
    if count.status ~= 'known' or count.value < 0 or count.value % 1 ~= 0 or count.value > M.MAX_STATS then
        return {status = 'unavailable', reason = count.reason or 'invalid_size'}
    end
    local list = {}
    for i = 0, count.value - 1 do
        local prefix = 'UnitDetailsContext.StatList.At(' .. i .. ').'
        local row = {key = scalar(function() return cco(unit, prefix .. 'Key') end)}
        for _, field in ipairs(M.STAT_FIELDS) do
            row[field] = scalar(function() return cco(unit, prefix .. field) end)
        end
        list[#list + 1] = row
    end
    return {status = 'ok', list = list}
end

function M.details(cco, unit)
    local row = {}
    for _, name in ipairs(M.DETAILS) do
        row[name] = scalar(function() return cco(unit, 'UnitDetailsContext.' .. name) end)
    end
    for _, name in ipairs(M.UNIT) do
        row[name] = scalar(function() return cco(unit, name) end)
    end
    return row
end

function M.profile(unit)
    local row = {
        type = flat(value.read(function() return unit:type() end, 'string')),
        commanding = flat(value.read(function() return unit:is_commanding_unit() end, 'boolean')),
        unit_class = scalar(function() return unit:unit_class() end),
        flags = {},
    }
    for _, name in ipairs(M.NUMBERS) do
        row[name] = flat(value.read(function() return unit[name](unit) end, 'number'))
    end
    for _, name in ipairs(M.KIND_FLAGS) do
        row.flags[name] = flat(value.read(function() return unit[name](unit) end, 'boolean'))
    end
    for _, kind in ipairs({'owned_non_passive_special_abilities', 'owned_passive_special_abilities'}) do
        local ok, list = pcall(function() return unit[kind](unit) end)
        if ok and type(list) == 'table' then
            local keys = {}
            for _, k in ipairs(list) do keys[#keys + 1] = tostring(k) end
            row[kind] = keys
        else
            row[kind] = 'unknown'
        end
    end
    return row
end

return M
