-- Optional trusted sensor; NOT part of policy API v1 and NOT a sandbox.
-- Only the trusted adapter may call observe or possess its engine arguments.
-- is_friendly must use the adapter's native alliance registry, never policy input.
-- Both endpoints are gated before any range/CCO/pair/position read. No memory.
local M = {version = 1}
local function finite(v)
    return type(v) == 'number' and v == v and v - v == 0
end
local function unknown(reason) return {status = 'unknown', reason = reason} end
local function read(fn, kind)
    local ok, value = pcall(fn)
    if not ok then return unknown('read_error') end
    if value == nil then return unknown('nil') end
    if type(value) ~= kind then return unknown('invalid_type') end
    if kind == 'number' and not finite(value) then return unknown('nonfinite') end
    return {status = 'known', value = value}
end
local function gate(unit, context)
    local relation = read(function() return context.is_friendly(unit) end, 'boolean')
    if relation.status ~= 'known' then return unknown('relationship_unknown') end
    if relation.value then return {status = 'known', value = true} end
    if not context.observer_alliance then return unknown('observer_missing') end
    return read(function() return unit:is_visible_to_alliance(context.observer_alliance) end, 'boolean')
end
local function permitted(g) return g.status == 'known' and g.value == true end
local function card_range(unit, context)
    local function query(key) return context.cco(unit, key) end
    local count = read(function() return query('UnitDetailsContext.StatList.Size') end, 'number')
    local index
    if count.status == 'known' and count.value >= 0 and count.value <= 100 and count.value % 1 == 0 then
        for i = 0, count.value - 1 do
            local key = read(function() return query('UnitDetailsContext.StatList.At(' .. i .. ').Key') end, 'string')
            if key.status == 'known' and key.value == 'scalar_missile_range' then index = i; break end
        end
    end
    local result = {}
    for _, field in ipairs({'Value', 'DisplayedValue', 'ValueBase'}) do
        result[field] = index and read(function()
            return query('UnitDetailsContext.StatList.At(' .. index .. ').' .. field)
        end, 'number') or unknown('stat_key_unavailable')
    end
    return result
end
local function centre_distance(source, target)
    return read(function()
        local a, b = source:position(), target:position()
        local ax, az, bx, bz = a:get_x(), a:get_z(), b:get_x(), b:get_z()
        assert(finite(ax) and finite(az) and finite(bx) and finite(bz))
        return math.sqrt((ax - bx)^2 + (az - bz)^2)
    end, 'number')
end
function M.observe(source, target, context)
    context = context or {}
    local result = {schema_version = M.version, access = 'withheld', sensors = {}}
    if finite(context.observed_ms) then result.observed_ms = context.observed_ms end
    result.source_gate = gate(source, context)
    if target ~= nil then result.target_gate = gate(target, context) end
    if not permitted(result.source_gate) or (target ~= nil and not permitted(result.target_gate)) then
        result.sensors.missile_range_m = unknown('endpoint_unavailable')
        result.sensors.card = {Value = unknown('endpoint_unavailable'),
            DisplayedValue = unknown('endpoint_unavailable'), ValueBase = unknown('endpoint_unavailable')}
        if target ~= nil then
            result.sensors.unit_in_range = unknown('endpoint_unavailable')
            result.sensors.unit_distance_m = unknown('endpoint_unavailable')
            result.sensors.centre_distance_xz_m = unknown('endpoint_unavailable')
        end
        return result
    end
    result.access = 'allowed'
    result.sensors.missile_range_m = read(function() return source:missile_range() end, 'number')
    result.sensors.card = card_range(source, context)
    if target ~= nil then
        result.sensors.unit_in_range = read(function() return source:unit_in_range(target) end, 'boolean')
        result.sensors.unit_distance_m = read(function() return source:unit_distance(target) end, 'number')
        result.sensors.centre_distance_xz_m = centre_distance(source, target)
    end
    return result
end
return M
