-- The pause probe (build/diag_chain/part2.md; tools.build nn-arena --pause-probe AT_S,REAL_MS,OFF_S): what a script
-- can do for a human who labels a paused battle. At at_s of battle time the script disables the player's orders
-- (bm:disable_orders(true), CA lib_battle_manager) and pauses the battle (bm:modify_battle_speed(0)); real-time
-- callbacks read the battle clock, the speed and the bridge's move count every real second of the pause (battle-time
-- timers stand still, real-time ones run: lib_timer_manager); after real_ms real ms the speed goes back; the orders
-- stay disabled until at_s + off_s of battle time (the bridge's script orders meanwhile show whether a script still
-- commands with the player's orders off). Events 'pause_probe' (stage, t, speed, written, the calls' results).
local M = {}

M.PREFIX = 'tww3_bai_pause_probe_'

-- opts: bm, cfg {at_s, real_ms, off_s}, speed (the battle's), emit(event, fields), now_ms() (battle ms since the start),
-- written() (the bridge's moves written), guard(fn) (the entry's error guard; default: fn).
function M.start(opts)
    local bm, cfg = opts.bm, opts.cfg
    local guard = opts.guard or function(fn) return fn end
    local h = {stage = 'wait'}

    local function call(name, ...)
        local f = bm[name]
        if type(f) ~= 'function' then return 'missing' end
        local ok, err = pcall(f, bm, ...)
        if ok then return 'ok' end
        return tostring(err)
    end

    local function reading(stage, extra)
        local ok, speed = pcall(function() return bm:current_battle_speed() end)
        local row = {stage = stage, t = opts.now_ms(), speed = ok and speed or nil, written = opts.written(),
            clock_ms = type(os.clock) == 'function' and math.floor(os.clock() * 1000 + 0.5) or nil}
        for k, v in pairs(extra or {}) do row[k] = v end
        opts.emit('pause_probe', row)
    end

    function h.tick()
        local t = opts.now_ms()
        if h.stage == 'wait' and t >= cfg.at_s * 1000 then
            h.stage = 'paused'
            local r = {orders_off = call('disable_orders', true)}
            r.pause = call('modify_battle_speed', 0)
            reading('pause', r)
            local n = math.floor((cfg.real_ms - 1) / 1000)
            for k = 1, n do
                bm:real_callback(guard(function() reading('paused', {k = k}) end), k * 1000, M.PREFIX .. 'r' .. k)
            end
            bm:real_callback(guard(function()
                local res = call('modify_battle_speed', opts.speed)
                h.stage = 'resumed'
                reading('resume', {resume = res})
                bm:real_callback(guard(function() reading('after_resume_1s_real') end), 1000, M.PREFIX .. 'after')
            end), cfg.real_ms, M.PREFIX .. 'resume')
        elseif h.stage == 'resumed' and t >= (cfg.at_s + cfg.off_s) * 1000 then
            h.stage = 'done'
            reading('orders_on', {orders_on = call('disable_orders', false)})
        end
    end

    return h
end

return M
