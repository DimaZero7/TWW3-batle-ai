-- The no-draw mode (V2; bridge.build --no-draw; bridge/docs/apps/memreader.md "Без кадров (V2)"; the policy:
-- bridge/config/memreader.json). ONE owned code patch through memreader Plus: the game's only view / present call
-- (5 bytes, the site no_draw of the sites file reverse/mem/sites_exe1103.json) becomes a 5-byte nop while the battle
-- runs, so the game asks the model at the full battle speed and draws no frame (the same game: tmp/farm/e1/REPORT.md,
-- tmp/farm/n/REPORT.md). Nothing else is written or hooked; the app's loop counter (site frame_count) is only read.
-- The patch lives in the game process's memory only: a killed game takes it with it, nothing is left on disk.
-- memreader Plus (its README, release 0_15_2): patcher(owner) gives the SAME handle for one owner (one registry per game
-- session; patches outlive a game mode); a batch that answers false leaves none of its patches written, but "does not
-- promise that no byte changed" - so every failure after patcher() still calls remove_all; remove_all may answer false
-- (a change it could not put back) - the handle is then kept and OFF is tried again.
--
-- cfg = config.no_draw = {site = {name, hi, lo, at, pattern, before, after} | nil (why: the build's reason),
--        frames = {hi, lo, ins_hi, ins_lo, ins} | nil, delay_ms, safety_ms}
--   site.hi / lo - the pattern's start (rva = hi * 2^20 + lo), site.at - the patched bytes' offset in the pattern
-- env = {bm, emit, flush, finished = fn -> true once the entry ended, mr = memreader Plus (default _G.memreader_plus),
--        clock = real seconds (default os.clock)}
-- M.new(cfg, env) -> h:
--   h.arm()     the entry's start (the battle has started): a leftover patch of this owner from an earlier battle of
--               the same process is taken back (v2_leftover), then ON after cfg.delay_ms of model time; a real-time
--               watcher turns it OFF once the entry has ended or the phase is VictoryCountdown / Complete (and tries
--               again every M.WATCH_MS, at most M.RETRIES times, while the restore is not done), and a real-time
--               safety net after cfg.safety_ms
--   h.off(why)  OFF (every end path of the entry calls it first: result, error, deadline, stall, timeout); a no-op when
--               nothing is held; event v2_off {why, ok, restored (the 5 bytes read back are the game's), frames, try};
--               the handle is let go only once remove_all succeeded. After a restored OFF, 1.5 s of real time later:
--               v2_drawn {frames_off, frames, rate_undrawn, rate, drawn} - the counter counts the app's loops (not drawn
--               frames): ~27,000 a second undrawn against ~190 drawn (game check 23.11, tmp/farm/v); drawn = the rate
--               after OFF under a tenth of the rate while undrawn; both rates finite or the row says why (no rates)
--   h.used      true once the patch was on (the result's field v2)
-- Refused (event v2_refused {reason}; the battle goes on DRAWN): no_site (cfg.site nil: cfg.why), no_memreader,
-- context (the pattern's bytes around the site differ), bytes (the 5 bytes are not the expected call), patcher,
-- batch (memreader Plus refused the write), read_back (written but not read back), error; after patcher() every
-- refusal first takes the owner's changes back (v2_off why failed_on).
local M = {}

M.OWNER = 'tww3_bai_no_draw'
M.NAME = 'skip_view_present'
M.WATCH, M.ON, M.SAFETY, M.DRAWN = 'tww3_bai_no_draw_watch', 'tww3_bai_no_draw_on', 'tww3_bai_no_draw_safety',
    'tww3_bai_no_draw_drawn'
M.WATCH_MS, M.DRAWN_MS, M.RETRIES = 250, 1500, 40
M.STOP_PHASES = {VictoryCountdown = true, Complete = true}

function M.bytes(hex)
    local t = {}
    for h in tostring(hex):gmatch('%x%x') do t[#t + 1] = string.char(tonumber(h, 16)) end
    return table.concat(t)
end

function M.hex(s)
    if type(s) ~= 'string' then return tostring(s) end
    return (s:gsub('.', function(c) return string.format('%02x', c:byte()) end))
end

-- (found, 'e8 ?? ...') -> true when every byte that is not ?? equals found's
function M.matches(found, pattern)
    if type(found) ~= 'string' then return false end
    local i = 0
    for tok in pattern:gmatch('%S+') do
        i = i + 1
        if tok ~= '??' and found:byte(i) ~= tonumber(tok, 16) then return false end
    end
    return #found >= i
end

local function tokens(pattern)
    local n = 0
    for _ in pattern:gmatch('%S+') do n = n + 1 end
    return n
end

-- a finite number, else nil (x - x is 0 only for a finite x; the game's Lua has no math.huge - game run 23.11)
local function finite(x)
    if type(x) ~= 'number' or x - x ~= 0 then return nil end
    return x
end
M.finite = finite

function M.new(cfg, env)
    local bm, emit = env.bm, env.emit
    local h = {used = false, on = false, refused = nil, done = false, tries = 0}
    local handle, at = nil, nil
    local before = cfg.site and M.bytes(cfg.site.before)
    local after = cfg.site and M.bytes(cfg.site.after)
    local function mr() return env.mr or _G.memreader_plus end
    local function flush() if env.flush then pcall(env.flush) end end
    local function clock()
        local ok, c = pcall(env.clock or os.clock)
        return ok and finite(c) or nil
    end
    local function addr(m, hi, lo) return m.add(m.add(m.base, hi * 1048576), lo) end
    local fr = nil
    local function frames()
        local f, m = cfg.frames, mr()
        if not f or type(m) ~= 'table' then return -1 end
        if fr == nil then
            local ok, ins = pcall(function() return m.read(addr(m, f.ins_hi, f.ins_lo), 0, #M.bytes(f.ins)) end)
            fr = ok and ins == M.bytes(f.ins)
        end
        if not fr then return -1 end
        local ok, v = pcall(function() return m.read_int32(addr(m, f.hi, f.lo), 0) end)
        return ok and finite(v) or -1
    end

    local function refuse(reason, extra)
        if h.refused then return end
        h.refused = reason
        local row = {reason = reason, site = cfg.site and cfg.site.name or nil}
        for k, v in pairs(extra or {}) do row[k] = v end
        pcall(emit, 'v2_refused', row)
        flush()
    end

    local function drawn_later(f0, c0)
        local undrawn = (f0 >= 0 and h.on_frames and h.on_frames >= 0 and c0 and h.on_clock and c0 > h.on_clock)
            and finite((f0 - h.on_frames) / (c0 - h.on_clock)) or nil
        pcall(function()
            bm:real_callback(function()
                local f1, c1 = frames(), clock()
                local rate = (undrawn and f1 >= 0 and c1 and c1 > c0) and finite((f1 - f0) / (c1 - c0)) or nil
                local row = {frames_off = f0, frames = f1}
                if undrawn and rate then
                    row.rate_undrawn, row.rate = math.floor(undrawn + 0.5), math.floor(rate + 0.5)
                    row.drawn = rate * 10 < undrawn
                else
                    row.drawn, row.why = false, 'no rates (counter or clock unreadable)'
                end
                pcall(emit, 'v2_drawn', row)
                flush()
            end, M.DRAWN_MS, M.DRAWN)
        end)
    end

    -- returns true once nothing of the owner is held
    function h.off(why)
        h.done = true
        if not handle then return true end
        local m, p = mr(), handle
        h.tries = h.tries + 1
        local ok, err = pcall(function()
            local r, w = p:remove_all()
            if not r then error('remove_all: ' .. tostring(w), 0) end
        end)
        local now = nil
        pcall(function() now = m.read(at, 0, 5) end)
        local restored = at == nil or now == before
        local f0, c0 = frames(), clock()
        if ok then handle, h.on = nil, false end
        pcall(emit, 'v2_off', {why = tostring(why), ok = ok, restored = restored,
            err = (not ok) and tostring(err) or nil, frames = f0, try = h.tries})
        flush()
        if ok and restored and h.used then drawn_later(f0, c0) end
        return ok
    end

    local function on()
        if h.on or h.refused or h.done or env.finished() then return end
        local m, s = mr(), cfg.site
        local ok, err = pcall(function()
            local n = tokens(s.pattern)
            local start = addr(m, s.hi, s.lo)
            local found = m.read(start, 0, n)
            if not M.matches(found, s.pattern) then error({reason = 'context', found = M.hex(found)}, 0) end
            at = m.add(start, s.at)
            local cur = m.read(at, 0, 5)
            if cur ~= before then error({reason = 'bytes', found = M.hex(cur), expected = M.hex(before)}, 0) end
            local p, why = m.patcher(M.OWNER)
            if not p then error({reason = 'patcher', err = tostring(why)}, 0) end
            handle = p             -- from here on every failure takes the owner's changes back
            local okb, whyb = p:batch({{'patch', M.NAME, at, before, after}})
            if not okb then error({reason = 'batch', err = tostring(whyb)}, 0) end
            if m.read(at, 0, 5) ~= after then error({reason = 'read_back'}, 0) end
            h.on = true
        end)
        if ok then
            h.used = true
            h.on_frames, h.on_clock = frames(), clock()
            pcall(emit, 'v2_on', {site = s.name, frames = h.on_frames, delay_ms = cfg.delay_ms})
            flush()
            return
        end
        local e = type(err) == 'table' and err or {reason = 'error', err = tostring(err)}
        if handle then h.off('failed_on') end
        local reason = e.reason
        e.reason = nil
        refuse(reason, e)
    end

    local function watch()
        if (h.done or h.refused) and not handle then return end
        local phase = nil
        pcall(function() phase = bm:get_current_phase_name() end)
        if env.finished() or M.STOP_PHASES[phase] or (h.done and handle) then
            if handle then h.off(env.finished() and 'finished' or (M.STOP_PHASES[phase] and phase or 'retry')) end
            if not handle or h.tries >= M.RETRIES then return end
        end
        bm:real_callback(function() pcall(watch) end, M.WATCH_MS, M.WATCH)
    end

    function h.arm()
        if h.armed then return end
        h.armed = true
        if not cfg.site then return refuse('no_site', {why = cfg.why}) end
        local m = mr()
        if type(m) ~= 'table' or type(m.patcher) ~= 'function' or type(m.read) ~= 'function'
            or type(m.add) ~= 'function' or m.base == nil then
            return refuse('no_memreader')
        end
        pcall(function()
            -- an earlier battle of this process that never reached its OFF (patches outlive a game mode)
            local p = m.patcher(M.OWNER)
            if not p then return end
            local r, removed = p:remove_all()
            if r and (tonumber(removed) or 0) > 0 then pcall(emit, 'v2_leftover', {removed = removed}) end
        end)
        bm:callback(function() pcall(on) end, cfg.delay_ms or 3000, M.ON)
        bm:real_callback(function() pcall(watch) end, M.WATCH_MS, M.WATCH)
        bm:real_callback(function() pcall(h.off, 'safety') end, cfg.safety_ms or 3600000, M.SAFETY)
    end

    return h
end

return M
