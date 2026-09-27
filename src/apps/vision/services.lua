-- How we see the battle (pure): units of each side are joined into groups; the
-- strongest group of a side is its main army. Both sides are seen, so later
-- decisions can work group against group: our units in full, the enemy ONLY as
-- our side sees it (apps.observation). Strength comes from apps.assessment.
--
-- Two units are linked when the gap between the edges of their formations
-- (closest soldiers) is at most link_gap_m; links chain (A-B, B-C => one group).
-- A lone unit is a group too. 30 m: the game AI in defence keeps 2-6 m between
-- neighbours in a line and ~20 m between lines, 21.5 m at most over six army
-- layouts (research/analysis/enemy-layout, 27.09.2026), plus margin (user rule).
local value = require('apps.core.value')

local M = {}
local finite = value.finite

M.DEFAULTS = {link_gap_m = 30}

-- points: flat {x1, z1, x2, z2, ...} in metres.
local function bounds(points)
    local b = {min_x = points[1], max_x = points[1], min_z = points[2], max_z = points[2]}
    for i = 3, #points, 2 do
        local x, z = points[i], points[i + 1]
        if x < b.min_x then b.min_x = x elseif x > b.max_x then b.max_x = x end
        if z < b.min_z then b.min_z = z elseif z > b.max_z then b.max_z = z end
    end
    return b
end

local function middle(points)
    local sx, sz, n = 0, 0, #points / 2
    for i = 1, #points, 2 do sx, sz = sx + points[i], sz + points[i + 1] end
    return {x = sx / n, z = sz / n}
end

-- Gap between the formations of two units: the closest pair of soldiers.
-- limit: stop early once a gap at or below it is found (enough to link).
function M.gap(a, b, limit)
    local ba, bb = a.bounds, b.bounds
    local dx = math.max(0, ba.min_x - bb.max_x, bb.min_x - ba.max_x)
    local dz = math.max(0, ba.min_z - bb.max_z, bb.min_z - ba.max_z)
    local box = math.sqrt(dx * dx + dz * dz)
    if limit and box > limit then return box end
    local best
    local pa, pb = a.points, b.points
    for i = 1, #pa, 2 do
        local x, z = pa[i], pa[i + 1]
        for j = 1, #pb, 2 do
            local ex, ez = x - pb[j], z - pb[j + 1]
            local d = ex * ex + ez * ez
            if not best or d < best then best = d end
        end
        if limit and best <= limit * limit then return math.sqrt(best) end
    end
    return math.sqrt(best)
end

-- units: {{id, points = {x1, z1, ...} (metres), strength, bearing?}}, visible enemies only.
-- bearing (optional): where the unit faces; the player sees it, so it is fair.
-- total_strength (optional): strength of the whole enemy army (its roster is
-- known before the battle), for the share of it we see.
-- Returns {groups = {{ids, strength, centre, facing?, bounds, radius_m}} strongest first,
--          main = groups[1] or nil, seen_share}.
function M.groups(units, params, total_strength)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    local items = {}
    for _, u in ipairs(units) do
        assert(type(u.points) == 'table' and #u.points >= 2 and #u.points % 2 == 0,
            'Unit ' .. tostring(u.id) .. ' has no soldier points')
        assert(finite(u.strength) and u.strength >= 0, 'Unit ' .. tostring(u.id) .. ' has no strength')
        items[#items + 1] = {id = u.id, points = u.points, strength = u.strength, bearing = u.bearing,
            bounds = bounds(u.points), middle = middle(u.points)}
    end
    -- Chain linking (union-find over all pairs).
    local parent = {}
    for i = 1, #items do parent[i] = i end
    local function root(i)
        while parent[i] ~= i do parent[i] = parent[parent[i]]; i = parent[i] end
        return i
    end
    for i = 1, #items do
        for j = i + 1, #items do
            if root(i) ~= root(j) and M.gap(items[i], items[j], p.link_gap_m) <= p.link_gap_m then
                parent[root(i)] = root(j)
            end
        end
    end
    local by_root, groups = {}, {}
    for i, it in ipairs(items) do
        local r = root(i)
        if not by_root[r] then
            by_root[r] = {ids = {}, strength = 0, members = {}}
            groups[#groups + 1] = by_root[r]
        end
        local h = by_root[r]
        h.ids[#h.ids + 1] = it.id
        h.members[#h.members + 1] = it
        h.strength = h.strength + it.strength
    end
    local seen = 0
    for _, h in ipairs(groups) do
        -- Centre: unit middles weighted by strength (by count if all are zero).
        local sx, sz, w = 0, 0, 0
        for _, it in ipairs(h.members) do
            local weight = h.strength > 0 and it.strength or 1
            sx, sz, w = sx + it.middle.x * weight, sz + it.middle.z * weight, w + weight
        end
        h.centre = {x = sx / w, z = sz / w}
        -- Facing: mean direction of the units' bearings, weighted like the centre.
        local fs, fc = 0, 0
        for _, it in ipairs(h.members) do
            if finite(it.bearing) then
                local weight = h.strength > 0 and it.strength or 1
                fs, fc = fs + math.sin(math.rad(it.bearing)) * weight, fc + math.cos(math.rad(it.bearing)) * weight
            end
        end
        if fs ~= 0 or fc ~= 0 then h.facing = math.deg(math.atan2(fs, fc)) % 360 end
        local b, radius = nil, 0
        for _, it in ipairs(h.members) do
            local ub = it.bounds
            if not b then
                b = {min_x = ub.min_x, max_x = ub.max_x, min_z = ub.min_z, max_z = ub.max_z}
            else
                b.min_x, b.max_x = math.min(b.min_x, ub.min_x), math.max(b.max_x, ub.max_x)
                b.min_z, b.max_z = math.min(b.min_z, ub.min_z), math.max(b.max_z, ub.max_z)
            end
            for k = 1, #it.points, 2 do
                local dx, dz = it.points[k] - h.centre.x, it.points[k + 1] - h.centre.z
                radius = math.max(radius, math.sqrt(dx * dx + dz * dz))
            end
        end
        h.bounds, h.radius_m, h.members = b, radius, nil
        seen = seen + h.strength
    end
    table.sort(groups, function(a, b)
        if a.strength ~= b.strength then return a.strength > b.strength end
        return #a.ids > #b.ids
    end)
    local share = nil
    if finite(total_strength) and total_strength > 0 then share = seen / total_strength end
    return {groups = groups, main = groups[1], seen_share = share, link_gap_m = p.link_gap_m}
end

-- The picture of the battle: our groups and the enemy's, with the same rule.
-- input = {own = units, enemy = visible enemy units, enemy_total = strength of
-- the whole enemy army (optional)}; units as in M.groups.
function M.picture(input, params)
    assert(type(input) == 'table' and type(input.own) == 'table' and type(input.enemy) == 'table',
        'Both sides required')
    local own = M.groups(input.own, params)
    local total = 0
    for _, u in ipairs(input.own) do total = total + u.strength end
    if total > 0 then own.seen_share = 1 end
    return {own = own, enemy = M.groups(input.enemy, params, input.enemy_total)}
end

return M
