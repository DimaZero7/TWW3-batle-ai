-- Queues past an obstacle (pure). User's idea (27.09.2026): units that an
-- obstacle keeps from their places walk around it in a queue — the front ones
-- first, the rear ones right behind, nobody cutting in; a closed side sends the
-- whole flow round the other side. The logistics must be as efficient as
-- possible: the goal is the earliest time the LAST unit stands in its place.
--
-- Everything is in the battlefield frame (apps.battlefield): the formation
-- moves towards +along; "across" is to the right. Points are the front-rank
-- centres (the order point); headings are degrees from the axis (+ = right).
--
-- The model:
--   * the obstacle band (band): blocked cells of the stand mask (apps.mask)
--     between us and the places, grown by 5 m (user's rule); the free strips
--     across it are the sides; the field edge closes a side;
--   * participants (user's rule): lines along the movement from every soldier;
--     a unit whose strip touches the grown obstacle must go round it; the
--     others keep their straight way (but queue if a detour holds their strip);
--   * the gate: the near edge of the band. A unit going through takes a strip
--     of the gate as wide as its front for as long as it takes to walk its depth
--     (plus a gap); two strips may not overlap at the same time. One walking
--     pace for all our infantry (roster, 1.5 m/s): after the gate nobody catches up;
--   * a row that fits the side's gap goes through side by side, pressed together;
--     what does not fit follows right behind (the queue); nobody crosses
--     another's way: participants keep their left-to-right order over the
--     sides (every such split is tried) and units that go through abreast keep it too;
--   * units are planned front row first (user: the rear ones must not get in
--     before the front ones), in a row the one nearer its side first; each
--     takes the place across the gap and the width (own, or narrower from the
--     roster at its measured reform time) that puts it in its place the earliest.
-- The group's command stays one manoeuvre ("get there"); inside it every unit
-- has its own route (at most: gate in, gate out, straight into the place) and a
-- release rule (new_dispatch): start when the units ahead have walked far enough.
-- verify() replays the plan in space and time and counts where two units would
-- go deep into each other (the check, not the planner).
local value = require('apps.core.value')
local battlefield = require('apps.battlefield.services')

local M = {}
local finite = value.finite

M.DEFAULTS = {
    tolerance_m = 5,     -- the obstacle grows by this much (user's rule)
    gap_m = 0.4,         -- clearance between two units (the formation keeps about 1 m)
    queue_gap_m = 2,     -- between a unit and the one it follows through the gate
    moving_buffer_m = 1, -- extra room ahead of and behind a walking unit (verify)
    soft_m = 3,          -- a touch shallower than this is not a conflict (verify)
    narrower = 1,        -- narrower roster widths tried in a detour
    settle_m = 5,        -- the last stretch into the place is straight: depth + this
    max_splits = 64,     -- splits of the participants over the sides tried
    top_splits = 4,      -- the best ones by the gate planned in full
    row_m = 3,           -- units whose fronts are this close along are one row
    speed_mps = 1.5,     -- walking pace of our infantry (roster)
    dt_s = 1,            -- time step of verify
    lead_m = 5,          -- the next leg is ordered this far before a gate (no stop)
    patience_s = 30,     -- a unit waits at most this much past its planned release
    reform_settle_s = 3, -- a reform in place is over when it stands (after this) or reform_s + this
    crowd_m = 1.0,       -- soldiers of two units closer than this: crowding
}

local function params_of(params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do p[k] = v end
    for k, v in pairs(params or {}) do p[k] = v end
    return p
end
M.params = params_of

local function dist(a, b)
    local da, dc = a.along - b.along, a.across - b.across
    return math.sqrt(da * da + dc * dc)
end

local function wrap(deg)
    deg = deg % 360
    if deg > 180 then deg = deg - 360 end
    return deg
end

-- Plan input from two formation plans (apps.formation placements: id, x, z,
-- bearing, front_m, depth_m, width): where every unit stands now and where it
-- goes, in the battlefield frame. shapes[id] = the unit's roster widths.
function M.units_from(field, before, after, shapes)
    local was = {}
    for _, q in ipairs(before) do was[q.id] = q end
    local out = {}
    for _, q in ipairs(after) do
        local b = was[q.id]
        if b then
            local list = {}
            for _, s in ipairs((shapes and shapes[q.id]) or {}) do
                if s.front_m and s.depth_m then
                    list[#list + 1] = {ordered_m = s.ordered_m, front_m = s.front_m, depth_m = s.depth_m,
                        reform_s = s.reform_s}
                end
            end
            out[#out + 1] = {id = q.id, place = q.id, key = q.key or q.role, start = battlefield.to_frame(field, b),
                target = battlefield.to_frame(field, q),
                heading0 = wrap(b.bearing - field.bearing), heading1 = wrap(q.bearing - field.bearing),
                start_front_m = b.front_m, start_depth_m = b.depth_m,
                front_m = q.front_m, depth_m = q.depth_m, width = q.width, bearing = q.bearing, shapes = list}
        end
    end
    return out
end

-- Where the units go: {from, to} along (their starts' fronts to their places'
-- fronts) and {left, right} across (their strips at the start and the place).
function M.extent(units)
    local e
    for _, u in ipairs(units) do
        local l = math.min(u.start.across, u.target.across) - u.front_m / 2
        local r = math.max(u.start.across, u.target.across) + u.front_m / 2
        if not e then
            e = {from = u.start.along, to = u.target.along, left = l, right = r}
        else
            e.from, e.to = math.min(e.from, u.start.along), math.max(e.to, u.target.along)
            e.left, e.right = math.min(e.left, l), math.max(e.right, r)
        end
    end
    return e
end

-- The obstacle band between from_along and to_along (and, across, between
-- left and right: the units' strips) on the stand mask
-- (apps.mask): {along0, along1, blocked = {{left, right}} (grown by the
-- tolerance), intervals = {{left, right}} free across the whole band}, or nil
-- when nothing is in the way. Cells not read yet count as blocked.
function M.band(mask, from_along, to_along, params, left, right)
    local p = params_of(params)
    local g = mask.grid
    local lo, hi
    -- Only what is in the way counts: across within the units' own strips
    -- (not grown: the map edge running beside a flank unit is not in its way,
    -- game 27.09.2026). The map edge still closes a side below, and the
    -- tolerance grows the obstacle for the rule who goes round.
    left = left or -1e9
    right = right or 1e9
    for r = 0, g.rows - 1 do
        local along = g.along0 + (r + 0.5) * g.step
        if along > from_along and along < to_along then
            for c = 1, g.cols do
                local across = g.across0 + (c - 0.5) * g.step
                if across > left and across < right and mask.stand[r * g.cols + c] ~= true then
                    lo, hi = math.min(lo or along, along), math.max(hi or along, along)
                end
            end
        end
    end
    if not lo then return nil end
    local half = g.step / 2
    local band = {along0 = lo - half - p.tolerance_m, along1 = hi + half + p.tolerance_m}
    -- Columns with a blocked cell anywhere over the band.
    local spans = {}
    for c = 1, g.cols do
        local closed = false
        for r = 0, g.rows - 1 do
            local along = g.along0 + (r + 0.5) * g.step
            if along + half > band.along0 and along - half < band.along1 and mask.stand[r * g.cols + c] ~= true then
                closed = true
                break
            end
        end
        if closed then
            local left = g.across0 + (c - 1) * g.step - p.tolerance_m
            local right = g.across0 + c * g.step + p.tolerance_m
            local last = spans[#spans]
            if last and left <= last[2] then last[2] = math.max(last[2], right) else spans[#spans + 1] = {left, right} end
        end
    end
    band.blocked = spans
    local edge0, edge1 = g.across0, g.across0 + g.cols * g.step
    local intervals, at = {}, edge0
    for _, s in ipairs(spans) do
        if s[1] > at then intervals[#intervals + 1] = {at, math.min(s[1], edge1)} end
        at = math.max(at, s[2])
    end
    if at < edge1 then intervals[#intervals + 1] = {at, edge1} end
    band.intervals = intervals
    return band
end

local function interval_of(band, left, right)
    for k, iv in ipairs(band.intervals) do
        if left >= iv[1] - 1e-6 and right <= iv[2] + 1e-6 then return k end
    end
    return nil
end

-- The free intervals right beside the blocked spans a unit's strip touches:
-- the one to the left of the first span and the one to the right of the last.
local function sides_of(band, left, right)
    local first, last
    for _, s in ipairs(band.blocked) do
        if s[2] > left and s[1] < right then
            first = first or s
            last = s
        end
    end
    local out = {}
    for k, iv in ipairs(band.intervals) do
        if not first or math.abs(iv[2] - first[1]) < 1e-6 or math.abs(iv[1] - last[2]) < 1e-6 then
            out[#out + 1] = {k = k, toward = (not first or math.abs(iv[2] - first[1]) < 1e-6) and -1 or 1}
        elseif iv[1] >= first[2] - 1e-6 and iv[2] <= last[1] + 1e-6 then
            -- A gap between two obstacles right ahead of the unit.
            out[#out + 1] = {k = k, toward = (iv[1] + iv[2]) / 2 < (left + right) / 2 and -1 or 1}
        end
    end
    return out
end

-- An oriented rectangle: front point, heading (deg from the axis), front and
-- depth; grown by half the gap on every side (plus a buffer along a walk).
local function rect(front, heading, w, d, p, moving)
    local h = math.rad(heading)
    local ua, uc = math.cos(h), math.sin(h)
    local hd = d / 2 + p.gap_m / 2 + (moving and p.moving_buffer_m or 0)
    local hw = w / 2 + p.gap_m / 2
    return {a = front.along - ua * d / 2, c = front.across - uc * d / 2, ua = ua, uc = uc, hd = hd, hw = hw,
        r = math.sqrt(hd * hd + hw * hw)}
end

-- How deep two rectangles go into each other (0 when apart): the least
-- overlap over the four separating axes.
local function depth_of(x, y)
    local da, dc = y.a - x.a, y.c - x.c
    local rr = x.r + y.r
    if da * da + dc * dc >= rr * rr then return 0 end
    local least
    local axes = {x.ua, x.uc, -x.uc, x.ua, y.ua, y.uc, -y.uc, y.ua}
    for i = 1, 7, 2 do
        local sa, sc = axes[i], axes[i + 1]
        local rx = x.hd * math.abs(x.ua * sa + x.uc * sc) + x.hw * math.abs(-x.uc * sa + x.ua * sc)
        local ry = y.hd * math.abs(y.ua * sa + y.uc * sc) + y.hw * math.abs(-y.uc * sa + y.ua * sc)
        local o = rx + ry - math.abs(da * sa + dc * sc)
        if o <= 0 then return 0 end
        if not least or o < least then least = o end
    end
    return least
end
M.depth_of = depth_of

-- Two units get in each other's way when they go deeper than soft_m into each
-- other: a light touch (a block wheeling next to its neighbour) the engine
-- sorts out by itself; walking through another unit it does not.
local function overlap(x, y, soft)
    return depth_of(x, y) > (soft or 0)
end
M.overlap = overlap

-- The walk of an option: one rectangle per time step from the moment the unit
-- starts walking (the same for every start delay, so it is made once).
local function walk_frames(u, option, p)
    if option.frames then return option.frames, option.walk_s end
    local v, dt = p.speed_mps, p.dt_s
    local legs, at, total = {}, u.start, 0
    for i, pt in ipairs(option.route) do
        local len = dist(at, pt)
        local w, d = option.front_m, option.depth_m
        if pt.final then w, d = u.front_m, u.depth_m end
        -- The block keeps facing forward while it walks (every order faces it
        -- forward): it slides, it does not swing its corners round.
        local heading = pt.final and u.heading1 or u.heading0
        legs[i] = {from = at, to = pt, len = len, s0 = total, heading = heading, w = w, d = d}
        total = total + len
        at = pt
    end
    local frames, li = {}, 1
    for k = 0, math.floor(total / v / dt) do
        local s = k * dt * v
        while li < #legs and s > legs[li].s0 + legs[li].len do li = li + 1 end
        local l = legs[li]
        local f = l.len > 1e-6 and math.max(0, math.min(1, (s - l.s0) / l.len)) or 1
        frames[k] = rect({along = l.from.along + (l.to.along - l.from.along) * f,
            across = l.from.across + (l.to.across - l.from.across) * f}, l.heading, l.w, l.d, p, true)
    end
    option.frames, option.walk_s = frames, total / v
    return frames, option.walk_s
end

-- The timeline of one option: stands at the start until `delay` (+ reform),
-- walks the route, stands in its place for good.
local function timeline(u, option, delay, p)
    local frames, walk_s = walk_frames(u, option, p)
    local t0 = delay + option.reform_s
    option.start_rect = option.start_rect
        or rect(u.start, u.heading0, u.start_front_m or u.front_m, u.start_depth_m or u.depth_m, p, false)
    option.end_rect = option.end_rect or rect(u.target, u.heading1, u.front_m, u.depth_m, p, false)
    return {t0 = t0, finish = t0 + walk_s, frames = frames, start = option.start_rect, final = option.end_rect,
        dt = p.dt_s}
end

local function rect_at(tl, t)
    if t < tl.t0 then return tl.start end
    if t >= tl.finish then return tl.final end
    return tl.frames[math.floor((t - tl.t0) / tl.dt)] or tl.final
end

-- The route of an option: gate in and gate out at `across` (a detour), then the place.
local function route_of(u, band, across, shape, settle_m)
    local route = {}
    if band and shape then
        if (shape.reform_s or 0) > 0 then
            -- Narrow in place first (user, 27.09.2026): the engine takes a new
            -- width only where it is sent, so a block sent narrower to the gap
            -- would stop and reform right at the rock with the queue behind it.
            route[#route + 1] = {along = u.start.along, across = u.start.across, width = shape.width,
                heading_deg = u.heading0, reform = true, reform_s = shape.reform_s}
        end
        route[#route + 1] = {along = band.along0, across = across, width = shape.width, heading_deg = 0}
        -- Out of the gap only when the rear is past the band too (or, for a
        -- place right behind the obstacle, level with the place).
        if u.target.along > band.along1 then
            local out = math.min(band.along1 + shape.depth_m, u.target.along)
            route[#route + 1] = {along = out, across = across, width = shape.width, heading_deg = 0}
        end
        -- The last stretch straight into the place: a block that comes in at
        -- an angle sweeps into the units already standing beside it.
        local last = route[#route]
        local into = u.depth_m + (settle_m or 5)
        if last.along < u.target.along - into - 1 then
            route[#route + 1] = {along = u.target.along - into, across = u.target.across, width = u.width,
                heading_deg = u.heading1}
        end
    end
    route[#route + 1] = {along = u.target.along, across = u.target.across, width = u.width, final = true}
    return route
end

-- The widths a unit may take through an interval `room` metres wide: its own
-- when it fits — then nothing else (game, 27.09.2026: a narrowed block stands
-- reforming before the gap and again in its place, which looked like stalls
-- and cost more than it saved); only a block wider than the gap narrows, to
-- the widest roster widths that fit.
local function widths(u, room, count)
    local out = {}
    if u.front_m <= room then
        out[1] = {width = u.width, front_m = u.front_m, depth_m = u.depth_m, reform_s = 0}
        return out
    end
    local narrower = {}
    for _, s in ipairs(u.shapes or {}) do
        if s.front_m < u.front_m - 1e-6 and s.front_m <= room then narrower[#narrower + 1] = s end
    end
    table.sort(narrower, function(a, b) return a.front_m > b.front_m end)
    for i = 1, math.min(count, #narrower) do
        local s = narrower[i]
        out[#out + 1] = {width = s.ordered_m, front_m = s.front_m, depth_m = s.depth_m, reform_s = s.reform_s or 0}
    end
    return out
end

-- The option (route and shapes) a planned unit walks, for timelines.
local function option_of(u, row)
    return {route = row.route, reform_s = row.reform_s or 0,
        front_m = row.slot and row.slot.front_m or u.front_m, depth_m = row.slot and row.slot.depth_m or u.depth_m}
end

-- units = {{id, start = {along, across}, target = {along, across}, heading0,
--   heading1, start_front_m?, start_depth_m?, front_m, depth_m, width (ordered),
--   shapes = {{ordered_m, front_m, depth_m, reform_s}}}} (units_from).
-- Returns {band, makespan_s, total_s, order = {ids by gate time}, split,
--   splits_tried, units = {[id] = {kind = 'aside' | 'straight' | 'detour',
--   slot?, side?, route, route_m, release = {at_s, after = {{id, walked_m}}},
--   arrive_s?, enter_s?, finish_s}}}.
local function plan_once(units, band, p, fixed_split)
    local v = p.speed_mps
    local items, crossing, participants = {}, {}, {}
    for _, u in ipairs(units) do
        assert(u.id and u.start and u.target and finite(u.front_m) and finite(u.depth_m),
            'Unit needs id, start, target, front_m, depth_m')
        u.heading0, u.heading1 = u.heading0 or 0, u.heading1 or 0
        local it = {u = u, id = u.id, kind = 'aside'}
        local left, right = u.start.across - u.front_m / 2, u.start.across + u.front_m / 2
        if band and u.start.along <= band.along0 and u.target.along > band.along0 then
            it.natural = interval_of(band, left, right)
            if it.natural then
                it.kind = 'straight'
            else
                it.kind = 'detour'
                it.sides = sides_of(band, left, right)
                participants[#participants + 1] = it
            end
            crossing[#crossing + 1] = it
        end
        items[#items + 1] = it
    end
    table.sort(participants, function(x, y) return x.u.start.across < y.u.start.across end)

    local function gate_distance(it, split)
        if it.kind == 'straight' then return 0 end
        local iv = band.intervals[split[it.id]]
        return math.max(0, iv[1] - (it.u.start.across - it.u.front_m / 2), (it.u.start.across + it.u.front_m / 2) - iv[2])
    end

    -- Earliest time >= a when the strip at `across` (front w, held d seconds)
    -- is free and nobody abreast would be crossed.
    local function earliest(reserved, it, side, across, w, d, a)
        local function blocks(r, t)
            local same_lane = math.abs(across - r.across) < (w + r.front_m) / 2 + p.gap_m
            -- One lane is a queue: nobody overtakes the one planned before it.
            if same_lane then return t < r.enter_s + r.hold_s - 1e-6 end
            if not (t < r.enter_s + r.hold_s - 1e-6 and r.enter_s < t + d - 1e-6) then return false end
            -- Abreast in the same gap: the left-to-right order must hold.
            if r.side == side then
                local was = it.u.start.across - r.start_across
                local now = across - r.across
                if was * now < 0 then return true end
            end
            return false
        end
        local times = {a}
        for _, r in ipairs(reserved) do
            if r.enter_s + r.hold_s > a then times[#times + 1] = r.enter_s + r.hold_s end
        end
        table.sort(times)
        for _, t in ipairs(times) do
            local free = true
            for _, r in ipairs(reserved) do
                if blocks(r, t) then free = false break end
            end
            if free then return t end
        end
        return times[#times]
    end

    local function candidate(it, side, across, shape, reserved)
        local u = it.u
        local gate_in = {along = band.along0, across = across}
        local a = dist(u.start, gate_in) / v + shape.reform_s
        local d = (shape.depth_m + p.queue_gap_m) / v
        local e = earliest(reserved, it, side, across, shape.front_m, d, a)
        local route
        if it.kind == 'straight' then
            route = route_of(u, nil)
        else
            route = route_of(u, band, across, shape, p.settle_m)
        end
        local len, at = 0, u.start
        for _, pt in ipairs(route) do len, at = len + dist(at, pt), pt end
        -- Walking the whole route after waiting (e - a) at the start; a
        -- narrowed block reforms back to its own width in its place.
        local finish = (e - a) + 2 * shape.reform_s + len / v
        return {across = across, side = side, front_m = shape.front_m, depth_m = shape.depth_m, width = shape.width,
            reform_s = shape.reform_s, arrive_s = a, enter_s = e, hold_s = d, finish_s = finish, route = route,
            route_m = len, start_across = u.start.across}
    end

    -- Places across a gap: its edges, nearest to the unit's own place, and
    -- right beside every strip already taken.
    -- Units of one side in different lanes keep their left-to-right order
    -- (in one lane they queue): nobody crosses another's way before or after the gap.
    local function in_order(side, x, w, natural, reserved)
        for _, r in ipairs(reserved) do
            if r.side == side and math.abs(x - r.across) >= (w + r.front_m) / 2 + p.gap_m - 1e-6
                and (x - r.across) * (natural - r.start_across) < 0 then
                return false
            end
        end
        return true
    end
    local function places(iv, side, w, natural, reserved)
        local lo, hi = iv[1] + w / 2, iv[2] - w / 2
        if lo > hi + 1e-6 then return {} end
        local all = {lo, hi, math.max(lo, math.min(hi, natural))}
        for _, r in ipairs(reserved) do
            for _, x in ipairs({r.across - (r.front_m + w) / 2 - p.gap_m, r.across + (r.front_m + w) / 2 + p.gap_m, r.across}) do
                if x >= lo - 1e-6 and x <= hi + 1e-6 then all[#all + 1] = x end
            end
        end
        local out = {}
        for _, x in ipairs(all) do
            if in_order(side, x, w, natural, reserved) then out[#out + 1] = x end
        end
        return out
    end

    local function schedule(split)
        local order = {}
        for _, it in ipairs(crossing) do order[#order + 1] = {it = it, g = gate_distance(it, split)} end
        -- Front row first; in a row the one nearer its side (straight ones first).
        table.sort(order, function(x, y)
            local ax, ay = x.it.u.start.along, y.it.u.start.along
            if math.abs(ax - ay) > p.row_m then return ax > ay end
            if math.abs(x.g - y.g) > 1e-6 then return x.g < y.g end
            return tostring(x.it.id) < tostring(y.it.id)
        end)
        local reserved, chosen, makespan, total = {}, {}, 0, 0
        for _, o in ipairs(order) do
            local it, best = o.it, nil
            local u = it.u
            if it.kind == 'straight' then
                best = candidate(it, it.natural, u.start.across,
                    {width = u.width, front_m = u.front_m, depth_m = u.depth_m, reform_s = 0}, reserved)
            else
                local side = split[it.id]
                local iv = band.intervals[side]
                for _, shape in ipairs(widths(u, iv[2] - iv[1], p.narrower)) do
                    for _, x in ipairs(places(iv, side, shape.front_m, u.start.across, reserved)) do
                        local c = candidate(it, side, x, shape, reserved)
                        c.cost = c.finish_s + 1e-3 * math.abs(x - u.start.across)
                        if not best or c.cost < best.cost then best = c end
                    end
                end
                if not best then return nil end
            end
            best.id = it.id
            reserved[#reserved + 1] = best
            chosen[it.id] = best
            makespan, total = math.max(makespan, best.finish_s), total + best.finish_s
        end
        return {chosen = chosen, reserved = reserved, makespan = makespan, total = total}
    end

    -- Every split of the participants over their sides that keeps their order across.
    local best, tried, split, found = nil, 0, {}, {}
    local function walk(i, min_side)
        if tried >= p.max_splits then return end
        if i > #participants then
            tried = tried + 1
            local s2 = schedule(split)
            if s2 then
                local copy = {}
                for k, v2 in pairs(split) do copy[k] = v2 end
                s2.split = copy
                found[#found + 1] = {split = copy, makespan = s2.makespan, total = s2.total}
                if not best or s2.makespan < best.makespan - 1e-6
                    or (math.abs(s2.makespan - best.makespan) <= 1e-6 and s2.total < best.total - 1e-6) then
                    best = s2
                end
            end
            return
        end
        local it = participants[i]
        for _, sd in ipairs(it.sides) do
            if sd.k >= min_side then
                split[it.id] = sd.k
                walk(i + 1, sd.k)
            end
        end
        split[it.id] = nil
    end
    if fixed_split and #crossing > 0 then
        best = schedule(fixed_split)
        if best then best.split = fixed_split end
        tried = 1
    elseif #crossing > 0 then
        walk(1, 0)
    end
    assert(#crossing == 0 or best, 'No way past the obstacle')
    table.sort(found, function(x, y)
        if math.abs(x.makespan - y.makespan) > 1e-6 then return x.makespan < y.makespan end
        return x.total < y.total
    end)

    local result = {band = band, units = {}, order = {}, makespan_s = 0, total_s = 0, splits_tried = tried,
        split = best and best.split or {}, candidates = found}
    for _, it in ipairs(items) do
        local u = it.u
        local row = {kind = it.kind}
        local c = best and best.chosen[it.id]
        if c then
            if it.kind == 'detour' then
                row.slot = {across = c.across, width = c.width, front_m = c.front_m, depth_m = c.depth_m,
                    reform_s = c.reform_s}
                row.side = c.side
            end
            row.natural = it.natural
            row.route, row.route_m, row.reform_s = c.route, c.route_m, c.reform_s
            row.arrive_s, row.enter_s, row.finish_s = c.arrive_s, c.enter_s, c.finish_s
            row.release = {at_s = c.enter_s - c.arrive_s, after = {}}
        else
            row.route = route_of(u, nil)
            row.route_m, row.reform_s = dist(u.start, u.target), 0
            row.finish_s = row.route_m / v
            row.release = {at_s = 0, after = {}}
        end
        result.units[it.id] = row
        result.makespan_s = math.max(result.makespan_s, row.finish_s)
        result.total_s = result.total_s + row.finish_s
    end
    -- Release rule: a unit held back starts when every unit ahead of it at the
    -- gate that kept it waiting has walked as far as it would have by then in
    -- the plan (one pace for all, so the rule does not depend on the real speed).
    if best then
        for _, r in ipairs(best.reserved) do result.order[#result.order + 1] = r end
        table.sort(result.order, function(x, y) return x.enter_s < y.enter_s end)
        for i, r in ipairs(result.order) do
            result.order[i] = r.id
            local row = result.units[r.id]
            if row.release.at_s > 1e-6 then
                for _, k in ipairs(best.reserved) do
                    local rk = result.units[k.id]
                    local ahead = k.id ~= r.id and k.enter_s < r.enter_s and k.enter_s + k.hold_s > r.arrive_s - 1e-6
                    if ahead and rk.release.at_s < row.release.at_s then
                        local walked = math.min(rk.route_m, v * (row.release.at_s - rk.release.at_s - rk.reform_s))
                        if walked > 0 then row.release.after[#row.release.after + 1] = {id = k.id, walked_m = walked} end
                    end
                end
                table.sort(row.release.after, function(x, y) return tostring(x.id) < tostring(y.id) end)
            end
        end
    end
    return result
end

-- Places of identical units are interchangeable (the same kind, width and
-- depth, in one row of the formation): the formation stays the same. In one
-- queue (a lane through a side) the unit that comes out first takes the place
-- farthest from where it comes out, so nobody stands in the way of the ones
-- behind it (the far places fill first). Returns true when a place changed hands.
local function swap_places(units, plan, p)
    local groups, changed = {}, false
    for _, u in ipairs(units) do
        local row = plan.units[u.id]
        -- Only a queue (one lane of a detour) comes out in an order that fights
        -- the order of the places; a unit going straight keeps its own place.
        if row and row.enter_s and u.key and row.slot then
            local key = table.concat({tostring(row.side), string.format('%.0f', row.slot.across * 2), tostring(u.key),
                string.format('%.1f|%.1f', u.front_m, u.depth_m)}, '|')
            local g = groups[key]
            if not g then g = {}; groups[key] = g end
            g[#g + 1] = {u = u, row = row, exit = row.slot and row.slot.across or u.start.across}
        end
    end
    -- One row of the formation: places whose fronts are within row_m of the next.
    local rows = {}
    for _, g in pairs(groups) do
        table.sort(g, function(a, b) return a.u.target.along < b.u.target.along end)
        local current = {g[1]}
        for i = 2, #g do
            if g[i].u.target.along - g[i - 1].u.target.along > p.row_m then
                rows[#rows + 1] = current
                current = {}
            end
            current[#current + 1] = g[i]
        end
        rows[#rows + 1] = current
    end
    for _, g in ipairs(rows) do
        if #g > 1 then
            local exit = 0
            for _, e in ipairs(g) do exit = exit + e.exit end
            exit = exit / #g
            local places = {}
            for _, e in ipairs(g) do
                places[#places + 1] = {target = e.u.target, heading1 = e.u.heading1, place = e.u.place, width = e.u.width}
            end
            table.sort(g, function(a, b)
                if math.abs(a.row.enter_s - b.row.enter_s) > 1e-6 then return a.row.enter_s < b.row.enter_s end
                return tostring(a.u.id) < tostring(b.u.id)
            end)
            table.sort(places, function(a, b)
                local da, db = math.abs(a.target.across - exit), math.abs(b.target.across - exit)
                if math.abs(da - db) > 1e-6 then return da > db end
                return tostring(a.place) < tostring(b.place)
            end)
            for i, e in ipairs(g) do
                local q = places[i]
                if q.place ~= e.u.place then changed = true end
                e.u.target, e.u.heading1, e.u.place, e.u.width = q.target, q.heading1, q.place, q.width
            end
        end
    end
    return changed
end

-- The best few splits by the gate are planned in full (places swapped, timed
-- in space and time); the one with the earliest last arrival wins.
function M.plan(units, band, params)
    local p = params_of(params)
    local orig = {}
    for _, u in ipairs(units) do
        orig[u.id] = {target = u.target, heading1 = u.heading1, place = u.place, width = u.width}
    end
    local function restore(assign)
        for _, u in ipairs(units) do
            local a = assign[u.id]
            u.target, u.heading1, u.place, u.width = a.target, a.heading1, a.place, a.width
        end
    end
    local first = plan_once(units, band, p)
    local splits = {}
    for i = 1, math.min(p.top_splits, #first.candidates) do splits[i] = first.candidates[i].split end
    if #splits == 0 then splits[1] = first.split end
    local best, best_assign
    for _, split in ipairs(splits) do
        restore(orig)
        local r = plan_once(units, band, p, split)
        if p.swap ~= false and swap_places(units, r, p) then
            r = plan_once(units, band, p, split)
            r.swapped = true
        end
        if not best or r.makespan_s < best.makespan_s - 1e-6
            or (math.abs(r.makespan_s - best.makespan_s) <= 1e-6 and r.total_s < best.total_s - 1e-6) then
            best = r
            best_assign = {}
            for _, u in ipairs(units) do
                best_assign[u.id] = {target = u.target, heading1 = u.heading1, place = u.place, width = u.width}
            end
        end
    end
    restore(best_assign)
    for _, u in ipairs(units) do
        local row = best.units[u.id]
        row.place, row.target = u.place, u.target
    end
    best.splits_tried, best.candidates = first.splits_tried, nil
    best.full_plans = #splits
    return best
end

-- The check: the plan replayed in space and time (every unit stands until its
-- release, walks its route at one pace, stands in its place). Counts the
-- seconds two units go deeper than soft_m into each other.
-- Returns {pairs = {'a|b' = seconds}, spans = {'a|b' = {from_s, to_s}}, seconds, worst_m}.
function M.verify(units, plan, params)
    local p = params_of(params)
    local tls, ids, last = {}, {}, 0
    for _, u in ipairs(units) do
        local row = plan.units[u.id]
        if row then
            local tl = timeline(u, option_of(u, row), row.release.at_s, p)
            tls[#tls + 1] = tl
            ids[#ids + 1] = u.id
            last = math.max(last, tl.finish)
        end
    end
    local out = {pairs = {}, spans = {}, seconds = 0, worst_m = 0}
    local t = 0
    while t <= last + p.dt_s do
        local any = false
        for i = 1, #tls do
            local ri = rect_at(tls[i], t)
            for j = i + 1, #tls do
                local dpt = depth_of(ri, rect_at(tls[j], t))
                if dpt > p.soft_m then
                    local a, b = tostring(ids[i]), tostring(ids[j])
                    local key = a < b and (a .. '|' .. b) or (b .. '|' .. a)
                    out.pairs[key] = (out.pairs[key] or 0) + p.dt_s
                    local span = out.spans[key] or {from_s = t}
                    span.to_s = t
                    out.spans[key] = span
                    out.worst_m = math.max(out.worst_m, dpt)
                    any = true
                end
            end
        end
        if any then out.seconds = out.seconds + p.dt_s end
        t = t + p.dt_s
    end
    return out
end

-- Carries the plan out, tick by tick. positions[id] = {along, across} of a
-- unit's centre (the engine's unit position). update(now_s, positions) returns
-- the orders to give now: {{id, along, across, width, heading_deg (from the
-- axis; nil for the final place), final, leg}}. A unit gets its first order on
-- release and the next one lead_m before it reaches a gate — never a stop.
function M.new_dispatch(plan, depths, params)
    local p = params_of(params)
    local units = {}
    for id, row in pairs(plan.units) do
        local depth = (row.slot and row.slot.depth_m) or depths[id] or 0
        units[id] = {row = row, leg = 0, depth = depth}
    end
    local d = {started_s = nil, log = {}}

    local function remaining(it, pos)
        local route, leg = it.row.route, math.max(it.leg, 1)
        local r = math.max(0, dist(pos, route[leg]) - it.depth / 2)
        for i = leg + 1, #route do r = r + dist(route[i - 1], route[i]) end
        return r
    end
    local function walked(it, pos)
        if it.leg == 0 or not pos then return 0 end
        return math.max(0, it.row.route_m - remaining(it, pos))
    end
    local function order(it, id, out)
        local pt = it.row.route[it.leg]
        out[#out + 1] = {id = id, along = pt.along, across = pt.across, width = pt.width, heading_deg = pt.heading_deg,
            final = pt.final == true, leg = it.leg}
    end

    function d.update(now_s, positions)
        d.started_s = d.started_s or now_s
        local t = now_s - d.started_s
        local out = {}
        -- Walked distances before anyone moves on this tick.
        local done_m = {}
        for id, it in pairs(units) do done_m[id] = walked(it, positions[id]) end
        for id, it in pairs(units) do
            if it.leg == 0 then
                local ready = true
                for _, a in ipairs(it.row.release.after) do
                    local k = units[a.id]
                    -- Arrived counts as walked all the way (a unit stops a bit short).
                    local k_done = k and k.leg == #k.row.route and k.arrived
                    if not k or k.leg == 0 or (done_m[a.id] < a.walked_m - 1e-6 and not k_done) then ready = false end
                end
                if t < it.row.release.at_s - 1e-6 and #it.row.release.after == 0 then ready = false end
                local forced = not ready and t >= it.row.release.at_s + p.patience_s
                if ready or forced then
                    it.leg, it.released_s, it.forced = 1, t, forced or nil
                    d.log[#d.log + 1] = {id = id, t_s = t, event = forced and 'forced' or 'released'}
                    order(it, id, out)
                end
            elseif positions[id] then
                local pt = it.row.route[it.leg]
                local pos = positions[id]
                if pt.reform then
                    local since = t - (it.leg_s or it.released_s)
                    if (since >= p.reform_settle_s and pos.still) or since >= (pt.reform_s or 0) + p.reform_settle_s then
                        it.leg, it.leg_s = it.leg + 1, t
                        d.log[#d.log + 1] = {id = id, t_s = t, event = 'leg', leg = it.leg}
                        order(it, id, out)
                    end
                elseif it.leg < #it.row.route then
                    local nxt = it.row.route[it.leg + 1]
                    local da, dc = nxt.along - pt.along, nxt.across - pt.across
                    local passed = (pos.along - pt.along) * da + (pos.across - pt.across) * dc >= 0
                    if passed or dist(pos, pt) - it.depth / 2 <= p.lead_m then
                        it.leg, it.leg_s = it.leg + 1, t
                        d.log[#d.log + 1] = {id = id, t_s = t, event = 'leg', leg = it.leg}
                        order(it, id, out)
                    end
                elseif not it.arrived and positions[id].still and dist(pos, pt) <= it.depth + p.lead_m then
                    it.arrived = true
                end
            end
        end
        table.sort(out, function(a, b) return tostring(a.id) < tostring(b.id) end)
        return out
    end

    -- Every unit has had its last order.
    function d.done()
        for _, it in pairs(units) do
            if it.leg < #it.row.route then return false end
        end
        return true
    end
    function d.state(id) return units[id] end
    return d
end

-- Crowding: soldiers that have a soldier of ANOTHER unit closer than crowd_m
-- (the formation keeps at least 1 m between units). units = {{id, points =
-- flat {x1, z1, ...}}}. Returns {soldiers, pairs = {'a|b' = soldiers}}.
function M.crowding(units, params)
    local cell = params_of(params).crowd_m
    local grid = {}
    local function key(i, j) return i .. ':' .. j end
    for _, u in ipairs(units) do
        local pts = u.points
        for i = 1, #pts, 2 do
            local k = key(math.floor(pts[i] / cell), math.floor(pts[i + 1] / cell))
            local list = grid[k]
            if not list then list = {}; grid[k] = list end
            list[#list + 1] = {u.id, pts[i], pts[i + 1]}
        end
    end
    local out = {soldiers = 0, pairs = {}}
    local limit = cell * cell
    for _, u in ipairs(units) do
        local pts = u.points
        for i = 1, #pts, 2 do
            local x, z = pts[i], pts[i + 1]
            local ci, cj = math.floor(x / cell), math.floor(z / cell)
            local other
            for di = -1, 1 do
                for dj = -1, 1 do
                    local list = grid[key(ci + di, cj + dj)]
                    if list then
                        for _, s in ipairs(list) do
                            if s[1] ~= u.id then
                                local dx, dz = s[2] - x, s[3] - z
                                if dx * dx + dz * dz < limit then other = s[1] break end
                            end
                        end
                    end
                    if other then break end
                end
                if other then break end
            end
            if other then
                out.soldiers = out.soldiers + 1
                local a, b = tostring(u.id), tostring(other)
                local pair = a < b and (a .. '|' .. b) or (b .. '|' .. a)
                out.pairs[pair] = (out.pairs[pair] or 0) + 1
            end
        end
    end
    return out
end

return M
