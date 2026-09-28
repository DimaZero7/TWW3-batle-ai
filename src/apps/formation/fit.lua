-- The formation on the map (pure, no engine): apps.formation.services plans a
-- layout; here the same layout is moved or turned until every unit fits
-- (input.fits(placement) -> bool, e.g. the stand mask, apps.mask).
local M = {}

-- Where to look when the formation does not fit where it was asked to stand
-- (user, 28.09.2026: on a rock the engine sets units crooked): first sideways
-- along the front, then back from the enemy, then turned a little.
-- Turns are whole steps of the engine's facing grid (360/128 deg, apps.orders.facing).
M.FIT = {step_m = 6, max_side_m = 90, max_back_m = 60, back_weight = 1.5,
    turns_deg = {5.625, -5.625, 11.25, -11.25}}

-- Places units on the map: input.fits(placement) -> bool. Order of what is
-- given up (user, 28.09.2026): another width at the same place; the whole
-- formation moved (sideways first, then back, nearest first) facing the same
-- way; then turned by 5 or 10 degrees. result.fit = {ok, how = 'as_planned' |
-- 'width' | 'shift' | 'turn' | 'none', shift = {side_m, back_m}, turn_deg, places_tried}.
function M.fit(plan, layout, input, params, search)
    assert(input.fits, 'fit needs input.fits')
    local s = {}
    for k, v in pairs(M.FIT) do
        if search and search[k] ~= nil then s[k] = search[k] else s[k] = v end
    end
    local base = plan(layout, input, params)
    if base.status == 'no_wall' or base.fit.ok then return base end
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
        local r = plan(layout, moved, params)
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

return M
