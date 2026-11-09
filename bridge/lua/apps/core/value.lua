-- Safe value checks and guarded engine reads. Pure: no engine objects, no I/O.
-- A read never raises: it returns {status='known', value=...} or
-- {status='unknown', reason=...}. Zero and false stay known values.
local M = {}

function M.finite(n)
    return type(n) == 'number' and n == n and n - n == 0
end

function M.integer(n)
    return M.finite(n) and n == math.floor(n)
end

function M.known(value)
    return {status = 'known', value = value}
end

function M.unknown(reason)
    return {status = 'unknown', reason = reason}
end

-- Calls fn in protected mode and checks the result type.
-- invalid_type_reason overrides the default 'type_<actual type>' reason.
function M.read(fn, kind, invalid_type_reason)
    local ok, value = pcall(fn)
    if not ok then return M.unknown('read_error') end
    if value == nil then return M.unknown('nil') end
    if type(value) ~= kind then
        return M.unknown(invalid_type_reason or ('type_' .. type(value)))
    end
    if kind == 'number' and not M.finite(value) then
        return M.unknown('nonfinite')
    end
    return M.known(value)
end

-- Copies an engine vector into a plain {x, y, z} table.
function M.vector(fn)
    local ok, result = pcall(function()
        local v = fn()
        local r = {x = v:get_x(), y = v:get_y(), z = v:get_z()}
        assert(M.finite(r.x) and M.finite(r.y) and M.finite(r.z))
        return r
    end)
    if ok then return M.known(result) end
    return M.unknown('invalid_vector')
end

function M.distance_xz(a, b)
    return math.sqrt((a.x - b.x) ^ 2 + (a.z - b.z) ^ 2)
end

return M
