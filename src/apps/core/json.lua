-- Minimal JSON encoder for telemetry rows. Pure.
-- Object keys are sorted so identical rows produce identical bytes.
-- Non-finite numbers become null: JSON has no NaN or infinity.
local value = require('apps.core.value')

local M = {}

local ESCAPES = {['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n',
    ['\r'] = '\\r', ['\t'] = '\\t'}

function M.string(text)
    local escaped = tostring(text):gsub('[%z\1-\31\\"]', function(c)
        return ESCAPES[c] or string.format('\\u%04x', string.byte(c))
    end)
    return '"' .. escaped .. '"'
end

local function is_array(t)
    local count = 0
    for k in pairs(t) do
        if type(k) ~= 'number' or k < 1 or k % 1 ~= 0 then return false end
        count = count + 1
    end
    return count == #t
end

function M.encode(data, depth)
    depth = depth or 0
    assert(depth < 32, 'JSON nesting too deep')
    local kind = type(data)
    if data == nil then return 'null' end
    if kind == 'boolean' then return tostring(data) end
    if kind == 'number' then
        if not value.finite(data) then return 'null' end
        return tostring(data)
    end
    if kind ~= 'table' then return M.string(data) end
    local parts = {}
    if next(data) ~= nil and is_array(data) then
        for i = 1, #data do parts[i] = M.encode(data[i], depth + 1) end
        return '[' .. table.concat(parts, ',') .. ']'
    end
    local keys = {}
    for k in pairs(data) do keys[#keys + 1] = tostring(k) end
    table.sort(keys)
    for _, k in ipairs(keys) do
        local v = data[k]
        if v == nil then v = data[tonumber(k)] end
        parts[#parts + 1] = M.string(k) .. ':' .. M.encode(v, depth + 1)
    end
    return '{' .. table.concat(parts, ',') .. '}'
end

return M
