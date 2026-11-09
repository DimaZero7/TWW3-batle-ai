-- Wall-clock helpers. os.date/os.time are standard Lua, no engine objects.
local M = {}

-- Epoch seconds lose precision in WH3's Lua numeric representation.
-- Test runs are capped below a day; seconds within the day remain exact.
function M.wall_seconds()
    local t = os.date('*t')
    return t.hour * 3600 + t.min * 60 + t.sec
end

function M.elapsed_wall_seconds(started)
    return (M.wall_seconds() - started) % 86400
end

function M.iso_utc()
    return os.date('!%Y-%m-%dT%H:%M:%SZ')
end

function M.batch_stamp()
    return os.date('%Y%m%dT%H%M%S')
end

return M
