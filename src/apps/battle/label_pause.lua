-- The labelling over the live battle (tools.build nn-arena --label-game; tools/label/game.py, docs/ru/training/label.md
-- "Разметка в игре"): a person decides for the operative while the battle stands still.
--   lock(where)       the player's own orders, the pause key, the time controls and the speed keys off
--                     (bm:disable_orders, disable_pausing, disable_time_speed_controls, disable_cycle_battle_speed: CA
--                     lib_battle_manager); each call's result in the event 'label_lock' (ok / missing / the error)
--   after_decide()    called after the bridge wrote a state: on the operative's seconds (every every_s of battle time,
--                     from second 0, round(t / 1000) - the companion's chain counts them the same way) the battle is
--                     paused (speed 0) and the answer to that move is waited for on real-time callbacks (battle-time
--                     timers stand still on a pause; the bridge's own poll is one of them), every poll_ms real ms; the
--                     answer given (poll() took it) -> the speed back. Events 'label_pause', 'label_resume' (the real
--                     wait, the speed read after).
-- No time limit on the wait: the person decides as long as needed (the entry's real-time deadline still holds).
local M = {}

M.PREFIX = 'tww3_bai_label_'
M.LOCKS = {'disable_orders', 'disable_pausing', 'disable_time_speed_controls', 'disable_cycle_battle_speed'}

-- opts: bm, every_s, speed, now_ms() (battle ms since the start), written() / answered() (the bridge's moves),
-- poll() (the bridge's poll), emit(event, fields), flush(), guard(fn), poll_ms (default 100), finished() (true: stop).
function M.start(opts)
    local bm = opts.bm
    local guard = opts.guard or function(fn) return fn end
    local poll_ms = opts.poll_ms or 100
    local h = {next = 0, paused = false, pauses = 0, waited_ms = 0}

    local function clock_ms()
        return type(os.clock) == 'function' and math.floor(os.clock() * 1000 + 0.5) or nil
    end

    function h.lock(where)
        local row = {where = where}
        for _, name in ipairs(M.LOCKS) do
            local f = bm[name]
            if type(f) ~= 'function' then
                row[name] = 'missing'
            else
                local ok, err = pcall(f, bm, true)
                row[name] = ok and 'ok' or tostring(err)
            end
        end
        opts.emit('label_lock', row)
        return row
    end

    function h.after_decide()
        if h.paused then return false end
        local t = opts.now_ms()
        local sec = math.floor(t / 1000 + 0.5)
        if sec < h.next then return false end
        while h.next <= sec do h.next = h.next + opts.every_s end
        local move = opts.written()
        h.paused, h.pauses = true, h.pauses + 1
        bm:modify_battle_speed(0)
        local c0 = clock_ms()
        opts.emit('label_pause', {t = t, sec = sec, move = move})
        local function poll()
            if opts.finished and opts.finished() then return end
            opts.poll()
            if opts.answered() >= move then
                bm:modify_battle_speed(opts.speed)
                h.paused = false
                local c1 = clock_ms()
                local waited = (c0 and c1) and (c1 - c0) or nil
                if waited then h.waited_ms = h.waited_ms + waited end
                local ok, speed = pcall(function() return bm:current_battle_speed() end)
                opts.emit('label_resume', {t = opts.now_ms(), sec = sec, move = move, wait_real_ms = waited,
                    speed_read = ok and speed or nil})
                if opts.flush then opts.flush() end
                return
            end
            bm:real_callback(guard(poll), poll_ms, M.PREFIX .. 'poll')
        end
        bm:real_callback(guard(poll), poll_ms, M.PREFIX .. 'poll')
        return true
    end

    return h
end

return M
