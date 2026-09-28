-- Who reaches whom with missiles, and where to stop (pure).
-- Measured rule (docs/ru/game/units/missile-range.md, 28.09.2026): the first
-- arrow flies when the distance from the MIDDLE of the shooting block to the
-- nearest rank of the target is <= 125-126 m for a 130 m range, whatever the
-- block's depth. The engine's unit_in_range fires earlier and is not used.
--
-- A block = {id, x, z, bearing, front_m, depth_m, range_m (shooters only)};
-- (x, z) is the centre of its FRONT rank, the block extends depth_m behind it
-- (as apps.formation placements).
--
-- The window (battle theory, phase 2): our army moved forward by `advance`
-- metres along `bearing`; the best advance is where none of their shooters
-- reaches any of our blocks and as many of our shooters as possible reach
-- one of theirs.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {
    fire_short_m = 4,     -- first arrow at range - 4 (126 m of 130): the far end measured
    own_margin_m = 1,     -- ours count as reaching only this much inside (125 m)
    safe_margin_m = 2,    -- theirs must stay this much beyond their first arrow (128 m)
    prefer_m = 5,         -- how deep into the window to stand, at most its middle
    max_advance_m = 1000, -- search no further than this
    max_back_m = 300,     -- and this far back: the window stays put once we stand in it
}

local function params_of(params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    return p
end
M.params = params_of

local function axes(bearing)
    local b = math.rad(bearing)
    return math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
end

-- Middle of a block.
function M.middle(block)
    local fx, fz = axes(block.bearing)
    return {x = block.x - fx * block.depth_m / 2, z = block.z - fz * block.depth_m / 2}
end

-- Distance from a point to the nearest point of a block (0 inside).
function M.to_block(point, block)
    local fx, fz, rx, rz = axes(block.bearing)
    local dx, dz = point.x - block.x, point.z - block.z
    local across, back = dx * rx + dz * rz, -(dx * fx + dz * fz)
    local a = math.max(math.abs(across) - block.front_m / 2, 0)
    local b = math.max(-back, back - block.depth_m, 0)
    return math.sqrt(a * a + b * b)
end

-- Distance at which a shooter's first arrow flies.
function M.first_arrow_m(shooter, params)
    return shooter.range_m - params_of(params).fire_short_m
end

-- Does the shooter reach the target? Returns bool, distance (middle to nearest rank).
function M.reaches(shooter, target, params)
    local d = M.to_block(M.middle(shooter), target)
    return d <= M.first_arrow_m(shooter, params), d
end

-- The advances a in [min_a, max_a] where f(a) <= limit, for f convex in a
-- (the distance from a point moving on a line to a fixed block is): {from, to} or nil.
local function interval(f, limit, min_a, max_a)
    -- Minimum of a convex function by ternary search.
    local lo, hi = min_a, max_a
    for _ = 1, 60 do
        local m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if f(m1) <= f(m2) then hi = m2 else lo = m1 end
    end
    local best = (lo + hi) / 2
    if f(best) > limit then return nil end
    local function edge(inside, outside)
        for _ = 1, 50 do
            local mid = (inside + outside) / 2
            if f(mid) <= limit then inside = mid else outside = mid end
        end
        return inside
    end
    local from = f(min_a) <= limit and min_a or edge(best, min_a)
    local to = f(max_a) <= limit and max_a or edge(best, max_a)
    return {from = from, to = to}
end

local function is_shooter(u)
    return finite(u.range_m) and u.range_m > 0
end

-- own, enemy: lists of blocks; bearing: the way our army moves.
-- Advances are counted from where we stand and may be negative (the window
-- behind us: we stand too close). advance_m is nil only when nobody shoots. Returns {advance_m, window = {from, to} or
-- nil, reached = shooters of ours reaching at advance_m, shooters = ours in
-- total, safe_to = the advance where their first shooter would reach us (nil:
-- never), under_fire_now,
-- reason = 'window' | 'no_enemy_shooters' | 'no_window' | 'no_shooters'}.
function M.window(own, enemy, bearing, params)
    local p = params_of(params)
    assert(finite(bearing), 'Bearing required')
    local fx, fz = axes(bearing)
    local min_a, max_a = -p.max_back_m, p.max_advance_m

    -- Where their shooters start to reach any of our blocks.
    local safe_to
    for _, s in ipairs(enemy) do
        if is_shooter(s) then
            local m = M.middle(s)
            local limit = M.first_arrow_m(s, p) + p.safe_margin_m
            for _, u in ipairs(own) do
                -- Our block moved by a = their middle moved back by a.
                local r = interval(function(a)
                    return M.to_block({x = m.x - fx * a, z = m.z - fz * a}, u)
                end, limit, min_a, max_a)
                if r and (not safe_to or r.from < safe_to) then safe_to = r.from end
            end
        end
    end

    -- From which advance each of our shooters reaches one of their blocks.
    local starts, shooters = {}, 0
    for _, s in ipairs(own) do
        if is_shooter(s) then
            shooters = shooters + 1
            local m = M.middle(s)
            local limit = M.first_arrow_m(s, p) - p.own_margin_m
            local first
            for _, t in ipairs(enemy) do
                local r = interval(function(a)
                    return M.to_block({x = m.x + fx * a, z = m.z + fz * a}, t)
                end, limit, min_a, max_a)
                if r and (not first or r.from < first) then first = r.from end
            end
            if first then starts[#starts + 1] = first end
        end
    end
    table.sort(starts)

    local result = {shooters = shooters, safe_to = safe_to, under_fire_now = safe_to ~= nil and safe_to <= 0}
    local limit = safe_to or max_a
    -- The most of our shooters that reach before theirs do.
    local reached, from = 0, nil
    for i, a in ipairs(starts) do
        if a < limit then reached, from = i, a end
    end
    if shooters == 0 then
        result.reason, result.reached = 'no_shooters', 0
        -- Nothing of ours shoots: just out of their reach; nil when nobody shoots.
        result.advance_m = safe_to and safe_to - 1e-3 or nil
        return result
    end
    if reached == 0 then
        -- No window: stand just out of their reach.
        result.reason, result.reached = 'no_window', 0
        result.advance_m = safe_to and safe_to - 1e-3 or 0
        return result
    end
    if not safe_to then
        -- Nothing of theirs shoots: stand prefer_m inside the reach of our
        -- first shooters; closer only brings their infantry nearer.
        result.reason = 'no_enemy_shooters'
        result.window = {from = starts[1]}
        result.advance_m = starts[1] + p.prefer_m
        result.reached = 0
        for _, a in ipairs(starts) do
            if a <= result.advance_m then result.reached = result.reached + 1 end
        end
        return result
    end
    result.reached = reached
    result.window = {from = from, to = limit}
    result.reason = 'window'
    result.advance_m = math.min(from + p.prefer_m, (from + limit) / 2)
    return result
end

return M
