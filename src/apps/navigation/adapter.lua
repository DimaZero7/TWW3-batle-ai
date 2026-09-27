-- Read-only point reachability for explicit units, after deployment.
-- A reachable point does not prove that the straight line to it is clear.
local map_adapter = require('apps.map.adapter')

local M = {}

-- cells: {ix, iz, x, z, height, inside_radar}; units: engine units.
-- Returns rows with reachable[i] per unit; cells outside the radar are skipped.
function M.read_cells(manager, vector_type, units, cells)
    assert(manager:get_current_phase_name() == 'Deployed',
        'Reachability capture requires Deployed phase')
    assert(#units > 0, 'At least one unit is required')
    local rows = {}
    for _, cell in ipairs(cells) do
        local row = {ix = cell.ix, iz = cell.iz, inside_radar = cell.inside_radar, reachable = {}}
        if cell.inside_radar then
            local p = map_adapter.vector(vector_type, cell.x, cell.height, cell.z)
            for i, unit in ipairs(units) do
                local reachable = unit:can_reach_position(p)
                assert(type(reachable) == 'boolean', 'Reachability did not return a boolean')
                row.reachable[i] = reachable
            end
        end
        rows[#rows + 1] = row
    end
    return rows
end

return M
