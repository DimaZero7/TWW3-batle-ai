-- The battlefield between the two main groups (pure). It only computes; the
-- decisions that use it come later (next task: aligning our army).
--
-- The axis runs from the centre of our main group to the centre of theirs, so
-- the picture is symmetric about it. Along the axis: our back and front lines,
-- the gap between the front lines, their front and back lines. Across it:
-- how far each group reaches to the left and right. The battlefield is the
-- rectangle from our back line to their back line, as wide as the wider
-- reach on either side plus a margin for manoeuvre (40 m, user's decision; 200 m was too much).
--
-- Frame: bearing 0 = +Z, 90 = +X; forward = (sin b, cos b), right = (cos b, -sin b).
-- "along" is metres from our centre towards theirs, "across" metres to the right.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {margin_m = 40}

-- Centre of flat {x1, z1, ...} points.
function M.middle(points)
    local sx, sz, n = 0, 0, #points / 2
    for i = 1, #points, 2 do sx, sz = sx + points[i], sz + points[i + 1] end
    return {x = sx / n, z = sz / n}
end

local function axes(bearing)
    local b = math.rad(bearing)
    return math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
end

-- World point -> {along, across} in the battlefield frame.
function M.to_frame(field, p)
    local fx, fz, rx, rz = axes(field.bearing)
    local dx, dz = p.x - field.origin.x, p.z - field.origin.z
    return {along = dx * fx + dz * fz, across = dx * rx + dz * rz}
end

-- {along, across} -> world point.
function M.to_world(field, along, across)
    local fx, fz, rx, rz = axes(field.bearing)
    return {x = field.origin.x + fx * along + rx * across, z = field.origin.z + fz * along + rz * across}
end

local function reach(field, points)
    local r
    for i = 1, #points, 2 do
        local f = M.to_frame(field, {x = points[i], z = points[i + 1]})
        if not r then
            r = {min_along = f.along, max_along = f.along, left = f.across, right = f.across}
        else
            r.min_along, r.max_along = math.min(r.min_along, f.along), math.max(r.max_along, f.along)
            r.left, r.right = math.min(r.left, f.across), math.max(r.right, f.across)
        end
    end
    r.width_m = r.right - r.left
    return r
end

-- input = {own = {points, centre?}, enemy = {points, centre?}}: all soldiers of
-- each side's main group (flat {x1, z1, ...}, metres) and, optionally, the
-- group centre from apps.vision (strength-weighted); else the plain middle.
-- Returns {status = 'ok' | 'contact', origin, bearing, centres_m, gap_m,
--          own = {back_m, front_m, left_m, right_m, width_m},
--          enemy = {front_m, back_m, left_m, right_m, width_m},
--          half_width_m, margin_m, corners = {4 world points}}.
function M.frame(input, params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    for _, side in ipairs({'own', 'enemy'}) do
        local s = input[side]
        assert(s and type(s.points) == 'table' and #s.points >= 2 and #s.points % 2 == 0,
            'Main group points required: ' .. side)
    end
    local own_c = input.own.centre or M.middle(input.own.points)
    local enemy_c = input.enemy.centre or M.middle(input.enemy.points)
    local dx, dz = enemy_c.x - own_c.x, enemy_c.z - own_c.z
    local distance = math.sqrt(dx * dx + dz * dz)
    assert(finite(distance) and distance > 0.01, 'The two main groups have the same centre')
    local bearing = math.deg(math.atan2(dx, dz))
    if bearing < 0 then bearing = bearing + 360 end
    local field = {origin = {x = own_c.x, z = own_c.z}, bearing = bearing, centres_m = distance,
        margin_m = p.margin_m}
    local own, enemy = reach(field, input.own.points), reach(field, input.enemy.points)
    field.own = {back_m = own.min_along, front_m = own.max_along, left_m = own.left, right_m = own.right,
        width_m = own.width_m}
    field.enemy = {front_m = enemy.min_along, back_m = enemy.max_along, left_m = enemy.left,
        right_m = enemy.right, width_m = enemy.width_m}
    field.gap_m = field.enemy.front_m - field.own.front_m
    field.status = field.gap_m > 0 and 'ok' or 'contact'
    -- Symmetric about the axis: the wider reach on either side, plus the margin.
    local half = math.max(-own.left, own.right, -enemy.left, enemy.right) + p.margin_m
    field.half_width_m = half
    local back, front = math.min(own.min_along, enemy.min_along), math.max(own.max_along, enemy.max_along)
    field.corners = {M.to_world(field, back, -half), M.to_world(field, back, half),
        M.to_world(field, front, half), M.to_world(field, front, -half)}
    return field
end

return M
