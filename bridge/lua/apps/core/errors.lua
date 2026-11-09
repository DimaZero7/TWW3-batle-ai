-- Error codes and guarded callbacks. Pure: the caller supplies side effects.
local M = {}

-- Raises "[CODE] message" so logs keep a stable, searchable code.
function M.raise(code, message, level)
    error('[' .. code .. '] ' .. tostring(message), (level or 1) + 1)
end

function M.check(condition, code, message)
    if not condition then M.raise(code, message, 2) end
    return condition
end

-- Extracts CODE from "[CODE] message" (possibly prefixed by a location).
function M.code(message)
    return tostring(message):match('%[([%w_]+)%]')
end

-- Wraps fn so an error is reported once through on_error with a traceback
-- and never escapes into the engine callback. is_stopped lets several
-- callbacks share one "stop after first failure" flag.
function M.guard(fn, on_error, is_stopped)
    return function(...)
        if is_stopped and is_stopped() then return end
        local args = {...}
        local count = select('#', ...)
        local ok, err = xpcall(function() return fn(unpack(args, 1, count)) end,
            function(e) return debug.traceback(tostring(e), 2) end)
        if not ok then on_error(err) end
    end
end

return M
