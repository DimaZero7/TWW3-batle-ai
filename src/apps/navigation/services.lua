-- Pure movement bookkeeping for one ordered move ("leg").
-- Definitions: docs/ru/architecture/tasks/obstacles.md (arrived, stuck).
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {
    arrive_m = 5,          -- centre within this distance of the target
    stop_samples = 3,      -- consecutive samples with is_moving == false
    min_ms = 3000,         -- ignore stops right after the order (reform, start)
    stuck_window_ms = 10000,
    stuck_move_m = 2,      -- moved less than this over the window while moving
}

local function hypot(dx, dz) return math.sqrt(dx * dx + dz * dz) end

-- opts: target {x, z}, timeout_ms, detect_stuck (default true; off for a
-- reform in place, where the centre legitimately stays), plus any M.DEFAULTS override.
-- update(t_ms, sample{x, z, moving}) -> nil while the leg continues, or
-- 'arrived' | 'stopped' | 'stuck' | 'timeout'. t_ms counts from the order.
function M.new_leg_monitor(opts)
    assert(opts and opts.target and finite(opts.target.x) and finite(opts.target.z), 'Leg target required')
    assert(finite(opts.timeout_ms) and opts.timeout_ms > 0, 'Leg timeout required')
    local cfg = {}
    for k, v in pairs(M.DEFAULTS) do cfg[k] = opts[k] or v end
    local history, still = {}, 0
    -- No math.huge: it is nil in the game's Lua (measured).
    local stats = {path_m = 0, samples = 0}
    local last

    local monitor = {stats = stats}
    function monitor.update(t_ms, s)
        assert(finite(t_ms) and finite(s.x) and finite(s.z), 'Invalid leg sample')
        stats.samples = stats.samples + 1
        if last then stats.path_m = stats.path_m + hypot(s.x - last.x, s.z - last.z) end
        last = {t = t_ms, x = s.x, z = s.z}
        local distance = hypot(s.x - opts.target.x, s.z - opts.target.z)
        stats.distance_m = distance
        if not stats.min_distance_m or distance < stats.min_distance_m then stats.min_distance_m = distance end
        still = s.moving and 0 or still + 1
        history[#history + 1] = last
        while #history > 1 and history[2].t <= t_ms - cfg.stuck_window_ms do table.remove(history, 1) end

        if t_ms < cfg.min_ms then return nil end
        if still >= cfg.stop_samples then
            return distance <= cfg.arrive_m and 'arrived' or 'stopped'
        end
        local oldest = history[1]
        if opts.detect_stuck ~= false and s.moving and t_ms - oldest.t >= cfg.stuck_window_ms
            and hypot(s.x - oldest.x, s.z - oldest.z) < cfg.stuck_move_m then
            return 'stuck'
        end
        if t_ms >= opts.timeout_ms then return 'timeout' end
        return nil
    end
    return monitor
end

return M
