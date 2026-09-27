-- Engine callbacks for deployment.services: place, measure, pair distance.
-- Built only from calls used in the measured deployment runs
-- (teleport_to_location, position/ordered_position, bearing, ordered_width,
-- unit_distance). Not re-verified in game after the move to this project.
local value = require('apps.core.value')
local orders = require('apps.orders.adapter')

local M = {}

-- registry.unit(unit_id), registry.controller(unit_id), vector(x, z).
function M.place_callback(registry, vector)
    return function(_, p)
        orders.teleport(registry.controller(p.unit_id), vector(p.x, p.z), p.facing_deg, p.width_m)
    end
end

local function plain(read)
    return read.status == 'known' and read.value or nil
end

function M.measure_callback(registry)
    return function(unit_id)
        local u = registry.unit(unit_id)
        return {
            position = plain(value.vector(function() return u:position() end)),
            ordered_position = plain(value.vector(function() return u:ordered_position() end)),
            bearing_deg = plain(value.read(function() return u:bearing() end, 'number')),
            ordered_bearing_deg = plain(value.read(function() return u:ordered_bearing() end, 'number')),
            ordered_width_m = plain(value.read(function() return u:ordered_width() end, 'number')),
        }
    end
end

function M.pair_distance_callback(registry)
    return function(a, b)
        return registry.unit(a):unit_distance(registry.unit(b))
    end
end

return M
