-- The trunk of the tactical level (pure): phase 2, the approach to the window,
-- with its branches (apps.tree: align, safe_detour, under_fire_stop). The same
-- decisions in battle and in the simulation (5.1, docs/ru/architecture/
-- ai-design.md): the game's entry and tools/sim only build the view of the
-- battle, carry out the intent and tell when a manoeuvre is over.
--
-- view = {now_ms, field (apps.battlefield), current = {anchor, bearing} (where
--         we stand), enemy_main (apps.vision), placements (our planned
--         rectangles), own_blocks, enemy_blocks (apps.reach), mask (filled
--         apps.mask, optional: without it every place fits), goal (march to a
--         point: the enemy army is not considered), align_params (apps.alignment,
--         optional: line = 'centres' | 'enemy_facing', tolerances)}
-- intent = {decision = 'wait' | 'align' | 'approach' | 'hold' | 'blocked',
--           reason, target = {anchor, bearing} (align, approach), detour,
--           window, stop_gap_m, step, path, align = check,
--           placements (approach: the formation carried along the step as a whole)}
local alignment = require('apps.alignment.services')
local approach = require('apps.approach.services')
local reach = require('apps.reach.services')
local mask_services = require('apps.mask.services')
local tree = require('apps.tree.services')

local M = {}

-- A trunk for one battle: the commander (one manoeuvre at a time) and the
-- alignment governor keep their state between decisions.
function M.new(t)
    return {tree = t, commander = approach.new_commander(), governor = alignment.new_governor()}
end

function M.on(s, name)
    return tree.on(s.tree, name)
end

-- The stop and the step: the window (apps.reach); marching to a goal point, or
-- nobody shooting: 20 m short of their front.
local function stop_of(view)
    local window
    if not view.goal then window = reach.window(view.own_blocks, view.enemy_blocks, view.current.bearing) end
    if window and window.advance_m then return view.field.gap_m - window.advance_m, window end
    return approach.stop_gap(0, 0), window
end

-- Where the next step ends on the mask (apps.approach.choose_advance; aside
-- when nothing fits straight on: choose_aside).
local function path_of(s, view, step, stop, window)
    local field, m = view.field, view.mask
    local cb = math.rad(view.current.bearing)
    local rx, rz = math.cos(cb), -math.sin(cb)
    local function fits_at(a, shift)
        if not m then return true end
        shift = shift or 0
        for _, p in ipairs(view.placements) do
            local b = math.rad(p.bearing)
            local moved = {x = p.x + math.sin(b) * a + rx * shift, z = p.z + math.cos(b) * a + rz * shift,
                bearing = p.bearing, front_m = p.front_m, depth_m = p.depth_m}
            if not mask_services.fits(m, moved).ok then return false end
        end
        return true
    end
    local function lane_free_at(a, shift)
        if not m then return true end
        return mask_services.lane(m, shift or 0, field.own.width_m, field.own.front_m, field.own.front_m + a).free
    end
    -- Past an obstacle: up to 20 m short of their front and (branch
    -- safe_detour) never into their shooters' reach.
    local limit = field.gap_m - approach.DEFAULTS.margin_m
    if M.on(s, 'safe_detour') and window and window.safe_to then limit = math.min(limit, window.safe_to) end
    local path = approach.choose_advance(step.advance_m, field.gap_m - stop, fits_at, lane_free_at, 3, limit)
    if not path.ok and m then
        path = approach.choose_aside(step.advance_m, field.gap_m - stop, fits_at, lane_free_at, nil, 3, limit)
    end
    return path
end

-- One decision; starts the manoeuvre it orders (the caller carries it out and
-- calls finish when the units stop).
function M.decide(s, view)
    local stop, window = stop_of(view)
    local step = approach.next_step(view.field.gap_m, stop)
    local check, gov
    if M.on(s, 'align') then
        check = alignment.check(view.field, view.current, view.enemy_main, view.align_params)
        gov = s.governor.decide(view.now_ms, check, false)
    end
    local path = {ok = false, reason = 'no_step'}
    if step.action == 'step' then path = path_of(s, view, step, stop, window) end
    local under_fire = M.on(s, 'under_fire_stop') and window ~= nil and window.under_fire_now == true
    local decision, reason = s.commander.decide({align = check, governor = gov, step = step, path = path,
        under_fire = under_fire})
    local intent = {decision = decision, reason = reason, window = window, stop_gap_m = stop, step = step,
        path = step.action == 'step' and path or nil, align = check, governor = gov}
    if decision == 'approach' then
        local b = math.rad(view.current.bearing)
        local a, aside = path.advance_m, path.aside_m or 0
        intent.target = {anchor = {x = view.current.anchor.x + math.sin(b) * a + math.cos(b) * aside,
            z = view.current.anchor.z + math.cos(b) * a - math.sin(b) * aside}, bearing = view.current.bearing}
        intent.placements = approach.carry(view.placements, view.current.bearing, a, aside)
        intent.detour = path.detour
        s.commander.start('approach', view.now_ms, nil)
    elseif decision == 'align' then
        s.governor.record_order(view.now_ms, check)
        intent.target = alignment.target(view.field, view.current, view.enemy_main, view.align_params)
        s.commander.start('align', view.now_ms, nil)
    end
    return intent
end

-- The manoeuvre under way is over (the units stopped, time ran out, or it was broken off).
function M.finish(s, now_ms, reason)
    s.commander.finish(now_ms, reason)
end

function M.busy(s)
    return s.commander.current ~= nil
end

-- Break the manoeuvre off? under_fire = a unit of ours under missile attack
-- (branch under_fire_stop; without it one action at a time even under fire).
function M.interrupt(s, under_fire)
    return M.on(s, 'under_fire_stop') and under_fire == true
end

return M
