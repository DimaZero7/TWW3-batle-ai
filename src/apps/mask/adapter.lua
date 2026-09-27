-- Engine reader for the "can we stand here?" mask (apps.mask.services).
-- Per cell: is_area_clear over the cell and can_reach_position of our
-- infantry unit. Read in batches from a timer: can_reach_position is the
-- slow part (docs/ru/apps/mask.md).
local map = require('apps.map.adapter')
local services = require('apps.mask.services')

local M = {}

-- reader(x, z) -> clear, reachable for services.fill.
function M.reader(bm, vector_type, unit, step)
    return function(x, z)
        local p = map.vector(vector_type, x, bm:get_terrain_height(x, z), z)
        local clear = bm:is_area_clear(p, 0, step, step, false)
        local ok, reachable = pcall(function() return unit:can_reach_position(p) end)
        return clear == true, ok and reachable == true
    end
end

-- Reads the next batch into mask. Returns the next index and done.
function M.read(bm, vector_type, unit, mask, first, count)
    return services.fill(mask, M.reader(bm, vector_type, unit, mask.grid.step), first, count)
end

return M
