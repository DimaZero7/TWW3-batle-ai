-- Unit kinds and formation width bounds shared by deployment and orders.
-- Width bounds are engineering input limits, not measured game limits.
local M = {}

M.KINDS = {lord = true, spearmen = true, archers = true, infantry = true}

M.WIDTH = {
    lord = {min = 3, max = 8, semantics = 'single_entity_no_formation_span'},
    formation = {min = 20, max = 40},
}

function M.is_lord(kind)
    return kind == 'lord'
end

-- Returns min, max width in metres for a roster kind.
function M.width_bounds(kind)
    local bounds = M.is_lord(kind) and M.WIDTH.lord or M.WIDTH.formation
    return bounds.min, bounds.max
end

function M.valid_width(kind, width)
    local low, high = M.width_bounds(kind)
    return type(width) == 'number' and width == width and width >= low and width <= high
end

function M.can_shoot(unit_state)
    return unit_state.kind == 'archers' and type(unit_state.ammo) == 'number'
        and unit_state.ammo == unit_state.ammo and unit_state.ammo - unit_state.ammo == 0
        and unit_state.ammo > 0
end

return M
