-- Files shared with the companion (tools/nn/companion), in the game's working
-- directory. A reader must never see half a file: the text goes to a temp file
-- that is then renamed. Windows cannot rename over an existing file, so the old
-- one is removed first; for that moment the file is missing, and the reader
-- waits. Without os.rename the file is written in place, and the reader relies
-- on the format's end marker (JSON parse, the orders' 'end' line).
local telemetry = require('apps.telemetry.adapter')

local M = {}

local function call(fn, ...)
    if type(fn) ~= 'function' then return false end
    local ok, result = pcall(fn, ...)
    return ok and result ~= nil and result ~= false
end

-- Returns 'renamed' or 'direct' (how it was written).
function M.write(path, text)
    local tmp = path .. '.tmp'
    telemetry.write_file(tmp, text)
    call(os.remove, path)
    if call(os.rename, tmp, path) then return 'renamed' end
    telemetry.write_file(path, text)
    call(os.remove, tmp)
    return 'direct'
end

function M.read(path)
    local ok, text = pcall(telemetry.read_file, path)
    if ok then return text end
    return nil
end

function M.remove(path)
    call(os.remove, path)
    call(os.remove, path .. '.tmp')
end

return M
