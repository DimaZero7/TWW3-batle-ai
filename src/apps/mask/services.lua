-- "Can we stand here?" mask over the battlefield (pure). A grid in the
-- battlefield frame (along the axis x across it), 3 m cells (user's decision);
-- each cell: can stand or not. A cell is standable when the area is clear and
-- our infantry can reach it (engine checks through a reader, see fill; water
-- nobody can cross fails the reach check). Forest and water are not treated
-- specially yet (user's decision). The map is visible to the player, so
-- knowing it is fair.
-- Used by the simple approach: does a unit fit there, is a lane free.
local value = require('apps.core.value')
local battlefield = require('apps.battlefield.services')

local M = {}
local finite = value.finite

M.DEFAULTS = {step_m = 3, pad_m = 10}
-- Cell codes in encode(): stand, blocked, unknown.
M.CODES = {stand = '.', blocked = '#', unknown = '?'}

-- The grid over the battlefield rectangle: from the rearmost back line to the
-- farthest one along the axis (plus pad_m at both ends: formations reach a bit
-- past their outermost soldiers), the full width across it.
function M.grid(field, step, pad)
    step = step or M.DEFAULTS.step_m
    pad = pad or M.DEFAULTS.pad_m
    assert(finite(step) and step > 0, 'Step must be positive')
    local along0 = math.min(field.own.back_m, field.enemy.front_m) - pad
    local along1 = math.max(field.enemy.back_m, field.own.front_m) + pad
    local rows = math.ceil((along1 - along0) / step)
    local cols = math.ceil(2 * field.half_width_m / step)
    return {step = step, along0 = along0, across0 = -field.half_width_m, rows = rows, cols = cols,
        count = rows * cols, frame = {origin = field.origin, bearing = field.bearing}}
end

-- Cell i (1-based, row-major: rows along the axis) -> centre {along, across, x, z}.
function M.cell(g, i)
    local r, c = math.floor((i - 1) / g.cols), (i - 1) % g.cols
    local along, across = g.along0 + (r + 0.5) * g.step, g.across0 + (c + 0.5) * g.step
    local p = battlefield.to_world(g.frame, along, across)
    return {along = along, across = across, x = p.x, z = p.z}
end

function M.index(g, along, across)
    local r = math.floor((along - g.along0) / g.step)
    local c = math.floor((across - g.across0) / g.step)
    if r < 0 or c < 0 or r >= g.rows or c >= g.cols then return nil end
    return r * g.cols + c + 1
end

function M.new(g)
    return {grid = g, stand = {}, known = 0}
end

-- reading = {clear, reachable}.
function M.record(mask, i, reading)
    if mask.stand[i] == nil then mask.known = mask.known + 1 end
    mask.stand[i] = reading.clear == true and reading.reachable == true
end

-- Fills cells [first, first + count) with reader(x, z) -> clear, reachable.
-- The battle reads the engine in batches (apps.mask.adapter); the simulation
-- reads a captured map. Returns the next index and whether the mask is complete.
function M.fill(mask, reader, first, count)
    first = first or 1
    local last = math.min(mask.grid.count, first + (count or mask.grid.count) - 1)
    for i = first, last do
        local c = M.cell(mask.grid, i)
        local clear, reachable = reader(c.x, c.z)
        M.record(mask, i, {clear = clear, reachable = reachable})
    end
    return last + 1, last >= mask.grid.count
end

-- true / false; nil when outside the grid or not read yet.
function M.stand_at(mask, along, across)
    local i = M.index(mask.grid, along, across)
    if not i then return nil end
    return mask.stand[i]
end

-- Does a unit fit at a placement {x, z, bearing, front_m, depth_m} (the
-- order point is the centre of the front rank)? Samples every half cell
-- inside its rectangle. Returns {ok, blocked, unknown, samples}.
function M.fits(mask, rect)
    local s = mask.grid.step / 2
    local b = math.rad(rect.bearing)
    local fx, fz, rx, rz = math.sin(b), math.cos(b), math.cos(b), -math.sin(b)
    local r = {blocked = 0, unknown = 0, samples = 0}
    local a = -rect.front_m / 2
    while a <= rect.front_m / 2 + 1e-6 do
        local back = 0
        while back <= rect.depth_m + 1e-6 do
            local p = {x = rect.x + rx * a - fx * back, z = rect.z + rz * a - fz * back}
            local f = battlefield.to_frame(mask.grid.frame, p)
            local stand = M.stand_at(mask, f.along, f.across)
            r.samples = r.samples + 1
            if stand == nil then r.unknown = r.unknown + 1
            elseif not stand then r.blocked = r.blocked + 1 end
            back = back + s
        end
        a = a + s
    end
    r.ok = r.blocked == 0 and r.unknown == 0
    return r
end

-- A lane along the axis, width_m wide around `across`, from from_along to
-- to_along: {free, first_blocked_along}. Unknown cells count as blocked:
-- the approach must not walk into what it has not checked.
function M.lane(mask, across, width_m, from_along, to_along)
    local g = mask.grid
    local dir = to_along >= from_along and 1 or -1
    local along = from_along
    while (to_along - along) * dir >= 0 do
        local x = across - width_m / 2
        while x <= across + width_m / 2 + 1e-6 do
            if M.stand_at(mask, along, x) ~= true then return {free = false, first_blocked_along = along} end
            x = x + g.step / 2
        end
        along = along + dir * g.step / 2
    end
    return {free = true}
end

function M.summary(mask)
    local s = {cells = mask.grid.count, known = mask.known, stand = 0, blocked = 0}
    for i = 1, mask.grid.count do
        if mask.stand[i] == true then s.stand = s.stand + 1 elseif mask.stand[i] == false then s.blocked = s.blocked + 1 end
    end
    return s
end

-- One character per cell, rows along the axis joined by '/', for telemetry.
function M.encode(mask)
    local g, rows = mask.grid, {}
    for r = 0, g.rows - 1 do
        local chars = {}
        for c = 1, g.cols do
            local i = r * g.cols + c
            local stand = mask.stand[i]
            local ch
            if stand == nil then ch = M.CODES.unknown
            elseif not stand then ch = M.CODES.blocked
            else ch = M.CODES.stand end
            chars[c] = ch
        end
        rows[#rows + 1] = table.concat(chars)
    end
    return table.concat(rows, '/')
end

return M
