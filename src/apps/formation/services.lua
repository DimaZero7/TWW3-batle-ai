-- Pure formation layouts (no engine). A layout places units by their ROLE
-- (given by apps.strategy); it knows nothing about why a strategy was chosen.
--   line_and_blocks: 'wall' units in a line in front, 'arc' units in
--   near-square blocks behind it, 'lord' level with the line on a flank.
-- Unit sizes come from the roster (data/roster): the measured front and depth
-- for each ordered width; only measured widths are used.
--
-- Frame: bearing 0 = +Z, 90 = +X. forward = (sin b, cos b), right = (cos b, -sin b).
-- A unit's ordered point is the centre of its FRONT rank (measured 27.09.2026);
-- "back" is the distance from the wall's front line backwards.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {
    wall_archer_gap_m = 3,   -- wall's rear rank to the first archer rank
    row_gap_m = 2,           -- between archer rows
    unit_gap_m = 1,          -- between neighbours in one line
    lord_gap_m = 4,          -- wall flank to the lord
    lord_side = 1,           -- 1 = right flank, -1 = left; used when no enemy lord is seen
    min_wall_depth_m = 5,    -- a thinner wall is not a wall (about 3 ranks)
    min_reach_m = 80,        -- every archer must shoot this far past the wall
    -- The thicker the wall the better (like a phalanx): depth decides, reach
    -- only breaks ties (user rule, 27.09.2026). 1 m of depth = 100 m of reach.
    wall_depth_weight = 100,
    max_archer_rows = 3,
    -- Archers stand in near-square blocks with room to turn left or right
    -- without covering each other, and may stick out a little past the wall
    -- (user rule, 27.09.2026).
    archer_max_aspect = 1.3, -- max(front/depth, depth/front)
    archer_overhang_m = 10,  -- per side past the wall's ends
    max_trace = 12,
    fit_options = 6,         -- on the map: the best options tried at one place
}

-- Where to look when the formation does not fit where it was asked to stand
-- (user, 28.09.2026: on a rock the engine sets units crooked): first sideways
-- along the front, then back from the enemy, then turned a little.
-- Turns are whole steps of the engine's facing grid (360/128 deg, apps.orders.facing).
M.FIT = {step_m = 6, max_side_m = 90, max_back_m = 60, back_weight = 1.5,
    turns_deg = {5.625, -5.625, 11.25, -11.25}}

-- Bearing in degrees from point a to point b.
function M.facing(a, b)
    local deg = math.deg(math.atan2(b.x - a.x, b.z - a.z))
    if deg < 0 then deg = deg + 360 end
    return deg
end

local function shape(unit, ordered)
    for _, s in ipairs(unit.shapes or {}) do
        if s.ordered_m == ordered then return s end
    end
    return nil
end

-- Widths measured for every unit of the list.
local function common_widths(units)
    local widths = {}
    if #units == 0 then return widths end
    for _, s in ipairs(units[1].shapes or {}) do
        local all = true
        for _, u in ipairs(units) do
            if not shape(u, s.ordered_m) then all = false; break end
        end
        if all then widths[#widths + 1] = s.ordered_m end
    end
    return widths
end

-- Units side by side, centred: returns the total front and each unit's centre offset.
local function line(units, width, gap)
    local total, fronts = 0, {}
    for i, u in ipairs(units) do
        fronts[i] = shape(u, width).front_m
        total = total + fronts[i] + (i > 1 and gap or 0)
    end
    local offsets, cursor = {}, -total / 2
    for i = 1, #units do
        offsets[i] = cursor + fronts[i] / 2
        cursor = cursor + fronts[i] + gap
    end
    return total, offsets
end

-- Room to turn: a block turning in place sweeps a circle of its half diagonal,
-- so neighbours need (diagonal - front) side by side and (diagonal - depth) between rows.
local function turn_gaps(archers, width, p)
    local side, row, aspect = p.unit_gap_m, p.row_gap_m, 1
    for _, u in ipairs(archers) do
        local s = shape(u, width)
        local diagonal = math.sqrt(s.front_m ^ 2 + s.depth_m ^ 2)
        side = math.max(side, diagonal - s.front_m)
        row = math.max(row, diagonal - s.depth_m)
        aspect = math.max(aspect, s.front_m / s.depth_m, s.depth_m / s.front_m)
    end
    return side, row, aspect
end

local function evaluate(walls, archers, wall_width, archer_width, rows, p)
    local option = {wall_width = wall_width, archer_width = archer_width, rows = rows, failed = {}}
    local side_gap, row_gap, aspect = p.unit_gap_m, p.row_gap_m, 1
    if #archers > 0 then side_gap, row_gap, aspect = turn_gaps(archers, archer_width, p) end
    option.archer_gap_m, option.archer_row_gap_m, option.archer_aspect = side_gap, row_gap, aspect
    local wall_front = line(walls, wall_width, p.unit_gap_m)
    local wall_depth = 0
    for _, u in ipairs(walls) do wall_depth = math.max(wall_depth, shape(u, wall_width).depth_m) end
    option.wall_front_m, option.wall_depth_m = wall_front, wall_depth
    local per_row = math.ceil(#archers / rows)
    local back, reach, widest = wall_depth + p.wall_archer_gap_m, nil, 0
    option.row_back_m = {}
    for r = 1, rows do
        local members = {}
        for i = (r - 1) * per_row + 1, math.min(r * per_row, #archers) do members[#members + 1] = archers[i] end
        if #members > 0 then
            local front = line(members, archer_width, side_gap)
            widest = math.max(widest, front)
            local depth = 0
            for _, u in ipairs(members) do
                local s = shape(u, archer_width)
                depth = math.max(depth, s.depth_m)
                -- From the unit's middle to the wall's front line.
                local left = u.range_m - (back + s.depth_m / 2)
                if not reach or left < reach then reach = left end
            end
            option.row_back_m[r] = back
            back = back + depth + row_gap
        end
    end
    option.archer_front_m, option.min_reach_m = widest, reach or 0
    if widest > wall_front + 2 * p.archer_overhang_m then
        option.failed[#option.failed + 1] = 'archers_wider_than_wall'
    end
    if aspect > p.archer_max_aspect then option.failed[#option.failed + 1] = 'archers_not_square' end
    if wall_depth < p.min_wall_depth_m then option.failed[#option.failed + 1] = 'wall_too_thin' end
    if #archers > 0 and option.min_reach_m < p.min_reach_m then option.failed[#option.failed + 1] = 'reach_too_short' end
    option.score = option.min_reach_m + p.wall_depth_weight * wall_depth
    return option
end

local function better(a, b)
    if (#a.failed == 0) ~= (#b.failed == 0) then return #a.failed == 0 end
    if #a.failed ~= #b.failed then return #a.failed < #b.failed end
    return a.score > b.score
end

local function place(anchor, bearing, along, back)
    local b = math.rad(bearing)
    local fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
    return anchor.x + rx * along - fx * back, anchor.z + rz * along - fz * back
end

-- input: {anchor = {x, z}, bearing, units = {{id, role, range_m, shapes}},
--         enemy_lord = {x, z} (optional, only if visible)}
-- The lord stands on the flank nearer to the enemy lord (he is to bind him).
-- params: overrides of M.DEFAULTS.
-- Returns {status = 'ok' | 'infeasible' | 'no_wall', choice, placements, options (best first), unplaced}.
local function line_and_blocks(input, params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    assert(input and input.anchor and finite(input.anchor.x) and finite(input.anchor.z), 'Anchor required')
    assert(finite(input.bearing), 'Bearing required')
    local walls, archers, lords, unplaced = {}, {}, {}, {}
    for _, u in ipairs(input.units) do
        local role = u.role
        assert(type(role) == 'string', 'Unit ' .. tostring(u.id) .. ' has no role')
        if role == 'wall' then walls[#walls + 1] = u
        elseif role == 'arc' then archers[#archers + 1] = u
        elseif role == 'lord' then lords[#lords + 1] = u
        else unplaced[#unplaced + 1] = u.id end
    end
    local result = {status = 'ok', placements = {}, options = {}, unplaced = unplaced,
        roles = {wall = #walls, arc = #archers, lord = #lords}}
    if #walls == 0 then result.status = 'no_wall' return result end

    local wall_widths = common_widths(walls)
    local archer_widths = #archers > 0 and common_widths(archers) or {0}
    local options = {}
    for _, ww in ipairs(wall_widths) do
        for _, aw in ipairs(archer_widths) do
            for rows = 1, (#archers > 0 and math.min(p.max_archer_rows, #archers) or 1) do
                -- Rows that would hold the same units per row as one row less add nothing.
                local n = #archers
                if rows == 1 or math.ceil(n / rows) ~= math.ceil(n / (rows - 1)) then
                    options[#options + 1] = evaluate(walls, archers, ww, aw, math.ceil(n / math.ceil(math.max(n, 1) / rows)), p)
                end
            end
        end
    end
    assert(#options > 0, 'No measured widths for the wall')
    table.sort(options, better)
    for i = 1, math.min(#options, p.max_trace) do result.options[i] = options[i] end

    local side = p.lord_side
    if input.enemy_lord and finite(input.enemy_lord.x) and finite(input.enemy_lord.z) then
        local b = math.rad(input.bearing)
        local along = math.cos(b) * (input.enemy_lord.x - input.anchor.x) - math.sin(b) * (input.enemy_lord.z - input.anchor.z)
        side = along < 0 and -1 or 1
        result.lord_side_reason = 'enemy_lord'
    else
        result.lord_side_reason = 'default'
    end
    result.lord_side = side

    -- Every unit's place for one option.
    local function build(best)
        local placements = {}
        local function add(u, role, width, along, back, row)
            local s = shape(u, width) or {front_m = 1, depth_m = 1}
            local x, z = place(input.anchor, input.bearing, along, back)
            placements[#placements + 1] = {id = u.id, role = role, row = row, x = x, z = z,
                bearing = input.bearing, width = width, front_m = s.front_m, depth_m = s.depth_m,
                along_m = along, back_m = back}
        end
        local _, wall_offsets = line(walls, best.wall_width, p.unit_gap_m)
        for i, u in ipairs(walls) do add(u, 'wall', best.wall_width, wall_offsets[i], 0, 0) end
        local per_row = #archers > 0 and math.ceil(#archers / best.rows) or 0
        for r = 1, best.rows do
            local members = {}
            for i = (r - 1) * per_row + 1, math.min(r * per_row, #archers) do members[#members + 1] = archers[i] end
            if #members > 0 then
                local _, offsets = line(members, best.archer_width, best.archer_gap_m)
                for i, u in ipairs(members) do add(u, 'arc', best.archer_width, offsets[i], best.row_back_m[r], r) end
            end
        end
        for i, u in ipairs(lords) do
            local along = side * (best.wall_front_m / 2 + p.lord_gap_m + (i - 1) * p.lord_gap_m)
            add(u, 'lord', nil, along, 0, 0)
        end
        return placements
    end

    local best, index = options[1], 1
    result.placements = build(best)
    -- On the map (input.fits(placement) -> bool, e.g. the stand mask): the
    -- best option whose every unit fits; only options that keep the layout's
    -- rules (square archers, a thick enough wall, reach) are tried.
    if input.fits then
        result.fit = {ok = false, tried = 0}
        for i, o in ipairs(options) do
            if result.fit.tried >= p.fit_options then break end
            if #o.failed == 0 or i == 1 then
                result.fit.tried = result.fit.tried + 1
                local placements = build(o)
                local all = true
                for _, q in ipairs(placements) do
                    if not input.fits(q) then all = false break end
                end
                if all then
                    best, index, result.placements = o, i, placements
                    result.fit.ok, result.fit.option = true, i
                    result.fit.how = i == 1 and 'as_planned' or 'width'
                    break
                end
            end
        end
    end
    result.choice = best
    if #best.failed > 0 then result.status = 'infeasible' end
    return result
end

M.LAYOUTS = {line_and_blocks = line_and_blocks}

-- Places units with the named layout; result.layout names it.
function M.plan(layout, input, params)
    local fn = M.LAYOUTS[layout]
    assert(fn, 'Unknown formation layout: ' .. tostring(layout))
    local result = fn(input, params)
    result.layout = layout
    return result
end

-- Places units on the map: input.fits(placement) -> bool. Order of what is
-- given up (user, 28.09.2026): another width at the same place; the whole
-- formation moved (sideways first, then back, nearest first) facing the same
-- way; then turned by 5 or 10 degrees. result.fit = {ok, how = 'as_planned' |
-- 'width' | 'shift' | 'turn' | 'none', shift = {side_m, back_m}, turn_deg, places_tried}.
function M.fit(layout, input, params, search)
    assert(input.fits, 'fit needs input.fits')
    local s = {}
    for k, v in pairs(M.FIT) do
        if search and search[k] ~= nil then s[k] = search[k] else s[k] = v end
    end
    local base = M.plan(layout, input, params)
    if base.fit.ok or base.status == 'no_wall' then return base end
    local shifts = {}
    local side = 0
    while side <= s.max_side_m + 1e-6 do
        local back = 0
        while back <= s.max_back_m + 1e-6 do
            for _, sign in ipairs(side > 0 and {1, -1} or {1}) do
                if side > 0 or back > 0 then
                    shifts[#shifts + 1] = {side_m = sign * side, back_m = back,
                        cost = math.sqrt(side ^ 2 + (s.back_weight * back) ^ 2)}
                end
            end
            back = back + s.step_m
        end
        side = side + s.step_m
    end
    table.sort(shifts, function(a, b) return a.cost < b.cost end)
    local tried = 0
    local function at(shift, turn)
        tried = tried + 1
        local b = math.rad(input.bearing)
        local fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
        local moved = {}
        for k, v in pairs(input) do moved[k] = v end
        moved.anchor = {x = input.anchor.x + rx * shift.side_m - fx * shift.back_m,
            z = input.anchor.z + rz * shift.side_m - fz * shift.back_m}
        moved.bearing = (input.bearing + turn) % 360
        local r = M.plan(layout, moved, params)
        if r.fit.ok then
            r.fit.how = turn ~= 0 and 'turn' or 'shift'
            r.fit.shift, r.fit.turn_deg, r.fit.places_tried = {side_m = shift.side_m, back_m = shift.back_m}, turn, tried
            r.anchor, r.bearing = moved.anchor, moved.bearing
            return r
        end
        return nil
    end
    for _, shift in ipairs(shifts) do
        local r = at(shift, 0)
        if r then return r end
    end
    for _, turn in ipairs(s.turns_deg) do
        local r = at({side_m = 0, back_m = 0}, turn)
        if r then return r end
        for _, shift in ipairs(shifts) do
            r = at(shift, turn)
            if r then return r end
        end
    end
    base.fit.how, base.fit.places_tried = 'none', tried
    return base
end

-- Turn a block in place. The engine's relative rotate pivots on the FRONT
-- rank centre (measured 27.09.2026), so a 90 deg turn shifts the block by half
-- its front sideways and forwards. Keeping the middle instead: the new order
-- point is the middle plus half the depth along the new facing.
-- middle = {x, z} (unit:position() is the middle of the soldiers).
function M.turn_in_place(middle, bearing, delta_deg, depth_m)
    assert(finite(middle.x) and finite(middle.z) and finite(bearing) and finite(delta_deg) and finite(depth_m),
        'Invalid turn')
    local new = (bearing + delta_deg) % 360
    local b = math.rad(new)
    return {x = middle.x + math.sin(b) * depth_m / 2, z = middle.z + math.cos(b) * depth_m / 2, bearing = new}
end

-- Pairs of placed units whose rectangles overlap (same bearing: intervals along and back).
function M.overlaps(placements)
    local found = {}
    for i = 1, #placements do
        for j = i + 1, #placements do
            local a, b = placements[i], placements[j]
            local along = math.min(a.along_m + a.front_m / 2, b.along_m + b.front_m / 2)
                - math.max(a.along_m - a.front_m / 2, b.along_m - b.front_m / 2)
            local back = math.min(a.back_m + a.depth_m, b.back_m + b.depth_m) - math.max(a.back_m, b.back_m)
            if along > 0.01 and back > 0.01 then found[#found + 1] = {a.id, b.id} end
        end
    end
    return found
end

return M
