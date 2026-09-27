-- Read-only map access: radar, terrain cells, buildings and CCO structures.
-- No unit orders, files or timers. Geometry lives in map.services.
local value = require('apps.core.value')
local services = require('apps.map.services')

local M = {}
local finite, integer = value.finite, value.integer

-- context is the game's `common` object (CCO access).
function M.read_frame(context)
    return services.build_frame(function(x, z)
        return context.get_context_value(string.format(
            'BattleRadarPosition(ToVector4(%.9f,0,%.9f,0))', x, z))
    end)
end

-- vector_type is the game's `battle_vector` class.
function M.vector(vector_type, x, y, z)
    local p = vector_type:new()
    p:set_x(x)
    p:set_y(y)
    p:set_z(z)
    return p
end

-- Reads grid cells [start, start + limit): height, clear area and ground type.
-- Returns cells, next index, done.
function M.read_batch(manager, vector_type, grid, start, limit)
    assert(integer(start) and start >= 0 and start <= grid.count, 'Invalid start index')
    assert(integer(limit) and limit > 0, 'Invalid batch size')
    local cells = {}
    local next_index = math.min(start + limit, grid.count)
    for id = start, next_index - 1 do
        local ix, iz = services.cell_index(grid, id)
        local x, z = services.cell_center(grid, ix, iz)
        local height = manager:get_terrain_height(x, z)
        assert(finite(height), 'Invalid terrain height')
        local p = M.vector(vector_type, x, height, z)
        local clear = manager:is_area_clear(p, 0, grid.step, grid.step, false)
        assert(type(clear) == 'boolean', 'Area query did not return a boolean')
        local ground = manager:ground_type(p)
        assert(type(ground) == 'string', 'Ground query did not return a string')
        cells[#cells + 1] = {ix = ix, iz = iz, x = x, z = z, height = height, clear = clear,
            ground = ground, inside_radar = services.inside_radar(grid, x, z)}
    end
    return cells, next_index, next_index == grid.count
end

local function check_page(start, limit)
    assert(integer(start) and start >= 0, 'Invalid start')
    assert(integer(limit) and limit > 0, 'Invalid limit')
end

-- Native building inventory. Positions are origins/centres, not footprints.
-- Native indices are 1-based. Measured scope: docs/en/game/map/objects.md.
function M.read_buildings(manager, start, limit)
    check_page(start, limit)
    local all = manager:buildings()
    local count = all:count()
    assert(start <= count, 'Start outside building list')
    local next_index = math.min(start + limit, count)
    local rows = {}
    for index = start + 1, next_index do
        local b = all:item(index)
        local p = b:position()
        local c = b:central_position()
        rows[#rows + 1] = {index = index, name = b:name(), category = b:category(),
            x = p:get_x(), y = p:get_y(), z = p:get_z(),
            center_x = c:get_x(), center_y = c:get_y(), center_z = c:get_z(),
            orientation = b:orientation(), health = b:health(),
            alliance_owner_id = b:alliance_owner_id(), has_gate = b:has_gate(),
            is_fort_wall = b:is_fort_wall(), is_fort_tower = b:is_fort_tower(),
            is_selectable = b:is_selectable()}
    end
    return rows, next_index, next_index == count, count
end

local STRUCTURE_FIELDS = {'Name', 'CategoryType', 'IsBridge', 'IsDestroyed', 'IsDestructable',
    'HasMissileWeapon', 'CanUpdateAbilities', 'LocalEffectText', 'GlobalEffectText',
    'SpecialAbilitiesList.Size'}

-- CCO exposes a separate, smaller, 0-based list. Do not equate its indices
-- with native indices. Names and effect texts are localized UI text, not ids.
function M.read_structure_contexts(common, start, limit)
    check_page(start, limit)
    local count = common.get_context_value('BattleRoot.BuildingsList.Size')
    assert(integer(count) and count >= 0, 'Invalid CCO count')
    assert(start <= count, 'Start outside CCO building list')
    local next_index = math.min(start + limit, count)
    local rows = {}
    for index = start, next_index - 1 do
        local base = 'BattleRoot.BuildingsList.At(' .. index .. ').'
        local row = {index = index}
        for _, key in ipairs(STRUCTURE_FIELDS) do
            local field = common.get_context_value(base .. key)
            assert(field ~= nil, 'Missing CCO field: ' .. key)
            row[key] = field
        end
        row.x, row.y, row.z, row.w = common.get_context_value(base .. 'Position')
        assert(type(row.x) == 'number' and type(row.y) == 'number' and type(row.z) == 'number',
            'Missing CCO position')
        rows[#rows + 1] = row
    end
    return rows, next_index, next_index == count, count
end

return M
