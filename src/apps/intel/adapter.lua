-- Trusted-side visibility reads.
-- Only the current visible state of an enemy is reliable; see
-- docs/en/apps/intel.md.
local value = require('apps.core.value')
local services = require('apps.intel.services')

local M = {}
local finite = value.finite

-- true / false, or nil when the engine call fails or returns a non-boolean.
function M.query(unit, observer_alliance)
    local ok, visible = pcall(function() return unit:is_visible_to_alliance(observer_alliance) end)
    if not ok or type(visible) ~= 'boolean' then return nil end
    return visible
end

local function read_position(unit)
    local ok, p = pcall(function()
        local pos = unit:position()
        local x, y, z = pos:get_x(), pos:get_y(), pos:get_z()
        assert(finite(x) and finite(y) and finite(z), 'invalid position')
        return {x = x, y = y, z = z}
    end)
    if ok then return p end
    return nil
end

-- Position is read only for a visible unit: a hidden enemy is never touched.
function M.observe(unit, observer_alliance, public_id, now_ms, memory)
    local visible = M.query(unit, observer_alliance)
    local position = visible and read_position(unit) or nil
    return services.record(memory, public_id, visible, position, now_ms)
end

return M
