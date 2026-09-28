-- Stepwise approach and the one-manoeuvre-at-a-time rule (pure).
-- User rules (27.09.2026):
--   * while a manoeuvre is under way no other orders are given (no command spam);
--   * alignment comes before approach;
--   * approach in steps of 50 m, the whole formation keeping its shape, until
--     the front lines are 20 m short of the distance at which the first
--     shooters of EITHER side reach the other side's front.
-- Each step is checked on the stand mask first (apps.mask): the formation
-- must fit where it goes (choose_advance); an obstacle on the way is walked
-- around by the engine. When nothing fits straight on, the same step is tried
-- shifted aside (choose_aside, 28.09.2026: a small rock at a flank stopped the
-- army 200 m short of the window in the game). Only when no place fits either
-- way does the approach stop ('blocked') and the level above decide.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {step_m = 50, margin_m = 20, arrive_tolerance_m = 2, aside_m = {15, 30, 45, 60}}

local function params_of(params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    return p
end

-- How far a side's shooters reach past its own front line. shooters =
-- {{along, range_m}} in the battlefield frame; toward = +1 for our side (it
-- faces +along), -1 for theirs. A shooter behind its front loses that distance.
function M.reach_past_front(shooters, front_along, toward)
    local best = 0
    for _, s in ipairs(shooters) do
        assert(finite(s.along) and finite(s.range_m), 'Shooter needs along and range_m')
        local behind = (front_along - s.along) * toward
        best = math.max(best, s.range_m - math.max(0, behind))
    end
    return best
end

-- The gap between the front lines to stop at: margin short of the first reach.
function M.stop_gap(own_reach, enemy_reach, params)
    return math.max(own_reach, enemy_reach) + params_of(params).margin_m
end

-- The next step from the current gap: {action = 'arrived'} or {action = 'step', advance_m}.
function M.next_step(gap, stop_gap, params)
    local p = params_of(params)
    if gap <= stop_gap + p.arrive_tolerance_m then return {action = 'arrived', gap_m = gap, stop_gap_m = stop_gap} end
    return {action = 'step', advance_m = math.min(p.step_m, gap - stop_gap), gap_m = gap, stop_gap_m = stop_gap}
end

-- Where the next step ends (user rule, 27.09.2026): an obstacle on the way is
-- not a reason to stop — the engine walks the units around it (with its own
-- queues); our job is a place where the WHOLE formation fits. Try the planned
-- advance; if the formation does not fit there, look further along the axis,
-- every search_m, for the first advance where it fits, up to search_limit.
-- Simple variant without the enemy army (user, 27.09.2026): the place past an
-- obstacle may lie beyond the stop line (max_advance) — flagged, not refused;
-- search_limit keeps it margin short of their front. fits_at(advance) -> bool;
-- lane_free_at(advance) -> bool (for the log).
-- Returns {ok, advance_m, detour (lane not straight), searched_m, beyond_stop_line}
-- or {ok = false, reason = 'no_place'}.
function M.choose_advance(advance, max_advance, fits_at, lane_free_at, search_m, search_limit)
    search_m = search_m or 3
    search_limit = search_limit or max_advance
    local a = advance
    while a <= search_limit + 1e-6 do
        if fits_at(a) then
            return {ok = true, advance_m = a, detour = not lane_free_at(a), searched_m = a - advance,
                beyond_stop_line = a > max_advance + 1e-6}
        end
        a = a + search_m
    end
    return {ok = false, reason = 'no_place'}
end

-- Nothing fits straight on: the same step with the whole formation shifted
-- across by each of `shifts` metres (smaller first, right then left), as
-- choose_advance. fits_at(advance, shift), lane_free_at(advance, shift).
-- Returns choose_advance's result plus aside_m (+ right, - left), or no_place.
function M.choose_aside(advance, max_advance, fits_at, lane_free_at, shifts, search_m, search_limit)
    for _, s in ipairs(shifts or M.DEFAULTS.aside_m) do
        for _, shift in ipairs({s, -s}) do
            local r = M.choose_advance(advance, max_advance, function(a) return fits_at(a, shift) end,
                function(a) return lane_free_at(a, shift) end, search_m, search_limit)
            if r.ok then
                r.aside_m = shift
                return r
            end
        end
    end
    return {ok = false, reason = 'no_place'}
end

-- The formation carried along a step as a whole: every placement moved `advance`
-- metres along `bearing` and `aside` metres across (+ right). The approach keeps
-- the formation's shape (user rule); planning it anew at every step changed the
-- widths (30 -> 60 -> 80 m in the game, 28.09.2026) and spread the wall onto rocks.
function M.carry(placements, bearing, advance, aside)
    local b = math.rad(bearing)
    local fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
    local out = {}
    for i, p in ipairs(placements) do
        local q = {}
        for k, v in pairs(p) do q[k] = v end
        q.x = p.x + fx * advance + rx * (aside or 0)
        q.z = p.z + fz * advance + rz * (aside or 0)
        out[i] = q
    end
    return out
end

-- The commander: one manoeuvre at a time, alignment first.
-- decide(input) -> 'wait' | 'align' | 'approach' | 'hold' | 'blocked', reason.
-- input = {align = alignment check, governor = its decision, step = next_step,
--          path = {ok, reason} for the step (mask lane and fits),
--          under_fire = their shooters reach us where we stand (apps.reach)}.
-- Under fire nothing is started: the approach is over, the battle phases
-- decide (28.09.2026: an alignment started under fire lasted 6 minutes and
-- the army was beaten meanwhile).
function M.new_commander()
    local c = {current = nil, log = {}}
    function c.start(kind, now_ms, info)
        assert(not c.current, 'A manoeuvre is already under way: ' .. tostring(c.current and c.current.kind))
        c.current = {kind = kind, started_ms = now_ms, info = info}
    end
    -- Called when the units stopped (or the manoeuvre timed out).
    function c.finish(now_ms, reason)
        if c.current then
            c.log[#c.log + 1] = {kind = c.current.kind, started_ms = c.current.started_ms, ended_ms = now_ms,
                reason = reason}
        end
        c.current = nil
    end
    function c.decide(input)
        if c.current then return 'wait', 'manoeuvre_under_way' end
        if input.under_fire then return 'hold', 'under_fire' end
        if input.align and input.align.needed then
            if input.governor == 'align' then return 'align', 'off_line' end
            if input.governor ~= 'given_up' then return 'wait', 'alignment_' .. tostring(input.governor) end
        end
        if input.step.action == 'arrived' then return 'hold', 'at_stop_line' end
        if not (input.path and input.path.ok) then return 'blocked', input.path and input.path.reason or 'no_path' end
        return 'approach', 'step'
    end
    return c
end

return M
