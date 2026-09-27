-- File I/O for telemetry: JSONL event sink and small state files.
-- Paths are relative to the game's working directory (the WH3 install).
local json = require('apps.core.json')
local clock = require('apps.core.clock')

local M = {}

function M.read_file(path)
    local file = io.open(path, 'r')
    if not file then return nil end
    local text = file:read('*a')
    file:close()
    return text
end

function M.write_file(path, text)
    local file, err = io.open(path, 'w')
    assert(file, err)
    local ok, write_err = file:write(text)
    local closed, close_err = file:close()
    assert(ok and closed, write_err or close_err)
end

function M.append(path, text)
    local file, err = io.open(path, 'a')
    assert(file, 'cannot open telemetry: ' .. tostring(err))
    local ok, write_err = file:write(text)
    local closed, close_err = file:close()
    assert(ok and closed, write_err or close_err)
end

-- Returns emit(event, fields). stamp(row) adds caller context (build,
-- run id, model time...) to every row before it is written.
-- The file is reopened per row so a crash never loses buffered lines.
function M.sink(path, stamp)
    return function(event, fields)
        local row = fields or {}
        row.event = event
        row.wall_iso = clock.iso_utc()
        if stamp then stamp(row) end
        M.append(path, json.encode(row) .. '\n')
        return row
    end
end

-- Same rows as sink(), but kept in memory until flush(): one file open per
-- flush instead of per row. Use for high-volume diagnostics and flush once
-- per tick; flush on finish/error too, or buffered rows are lost.
function M.buffered_sink(path, stamp)
    local buffer = {}
    local function emit(event, fields)
        local row = fields or {}
        row.event = event
        row.wall_iso = clock.iso_utc()
        if stamp then stamp(row) end
        buffer[#buffer + 1] = json.encode(row)
        return row
    end
    local function flush()
        if #buffer == 0 then return end
        local text = table.concat(buffer, '\n') .. '\n'
        buffer = {}
        M.append(path, text)
    end
    return emit, flush
end

-- Monotonic counter kept on disk; prevents same-second batch ids
-- without consuming the game's random numbers.
function M.next_sequence(path)
    local sequence = (tonumber(M.read_file(path)) or 0) + 1
    M.write_file(path, tostring(sequence))
    return sequence
end

return M
