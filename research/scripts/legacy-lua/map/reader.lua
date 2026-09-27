-- Read-only map adapter. No unit orders, files, timers or global side effects.
-- Contract and measured limits: docs/map/README.md.
local M = {}
local function finite(n)
    return type(n) == 'number' and n == n and math.abs(n) < 1e100
end
local function integer(n)
    return finite(n) and n == math.floor(n)
end

function M.read_frame(context)
    local function query(x, z)
        local u, v = context.get_context_value(string.format(
            'BattleRadarPosition(ToVector4(%.9f,0,%.9f,0))', x, z))
        assert(finite(u) and finite(v), 'Radar query did not return two finite numbers')
        return u, v
    end
    local u0, v0 = query(0, 0)
    local ux, vx = query(100, 0)
    local uz, vz = query(0, 100)
    local a, b, c, d = (ux-u0)/100, (uz-u0)/100, (vx-v0)/100, (vz-v0)/100
    local det = a*d-b*c
    assert(finite(det) and math.abs(det) > 1e-12, 'Degenerate radar transform')
    -- Only the unrotated orientation measured in our two maps is supported.
    assert(a > 0 and d < 0 and math.abs(b) <= math.abs(a)*1e-6
        and math.abs(c) <= math.abs(d)*1e-6, 'Untested radar orientation')
    local frame = {kind='radar_frame', movement_boundary_verified=false,
        a=a, b=b, c=c, d=d, u0=u0, v0=v0, corners={}}
    for i, uv in ipairs({{0,0},{1,0},{1,1},{0,1}}) do
        local u, v = uv[1]-u0, uv[2]-v0
        local x, z = (d*u-b*v)/det, (-c*u+a*v)/det
        local actual_u, actual_v = query(x, z)
        assert(math.abs(actual_u-uv[1]) < 1e-5 and math.abs(actual_v-uv[2]) < 1e-5,
            'Radar corner validation failed')
        frame.corners[i] = {x=x, z=z, u=actual_u, v=actual_v}
        frame.min_x = frame.min_x and math.min(frame.min_x,x) or x
        frame.max_x = frame.max_x and math.max(frame.max_x,x) or x
        frame.min_z = frame.min_z and math.min(frame.min_z,z) or z
        frame.max_z = frame.max_z and math.max(frame.max_z,z) or z
    end
    frame.width = frame.max_x-frame.min_x
    frame.depth = frame.max_z-frame.min_z
    frame.center_x = (frame.min_x+frame.max_x)/2
    frame.center_z = (frame.min_z+frame.max_z)/2
    return frame
end

function M.world_to_radar(frame, x, z)
    assert(finite(x) and finite(z), 'Invalid world coordinate')
    return frame.a*x+frame.b*z+frame.u0, frame.c*x+frame.d*z+frame.v0
end

function M.new_grid(frame, step)
    assert(finite(step) and step > 0, 'Step must be positive')
    assert(finite(frame.min_x) and finite(frame.max_x) and frame.max_x > frame.min_x
        and finite(frame.min_z) and finite(frame.max_z) and frame.max_z > frame.min_z,
        'Invalid frame dimensions')
    local columns = math.ceil((frame.max_x-frame.min_x)/step)
    local rows = math.ceil((frame.max_z-frame.min_z)/step)
    return {min_x=frame.min_x, min_z=frame.min_z, radar_max_x=frame.max_x,
        radar_max_z=frame.max_z, step=step, columns=columns, rows=rows,
        count=columns*rows, query_max_x=frame.min_x+columns*step,
        query_max_z=frame.min_z+rows*step}
end

function M.cell_center(grid, ix, iz)
    assert(integer(ix) and ix >= 0 and ix < grid.columns
        and integer(iz) and iz >= 0 and iz < grid.rows, 'Cell index outside grid')
    return grid.min_x+(ix+0.5)*grid.step, grid.min_z+(iz+0.5)*grid.step
end

function M.world_to_cell(grid, x, z)
    assert(finite(x) and finite(z), 'Invalid world coordinate')
    -- Half-open indexing interval; points on the upper edge are not clamped.
    if x < grid.min_x or z < grid.min_z
        or x >= grid.radar_max_x or z >= grid.radar_max_z then return nil end
    return math.floor((x-grid.min_x)/grid.step), math.floor((z-grid.min_z)/grid.step)
end

function M.read_batch(manager, vector_type, grid, start, limit)
    assert(integer(start) and start >= 0 and start <= grid.count, 'Invalid start index')
    assert(integer(limit) and limit > 0, 'Invalid batch size')
    local cells = {}
    local next_index = math.min(start+limit,grid.count)
    for id=start,next_index-1 do
        local iz=math.floor(id/grid.columns)
        local ix=id-iz*grid.columns
        local x,z=M.cell_center(grid,ix,iz)
        local height=manager:get_terrain_height(x,z)
        assert(finite(height), 'Invalid terrain height')
        local p=vector_type:new();p:set_x(x);p:set_y(height);p:set_z(z)
        local clear=manager:is_area_clear(p,0,grid.step,grid.step,false)
        assert(type(clear)=='boolean', 'Area query did not return a boolean')
        local ground=manager:ground_type(p)
        assert(type(ground)=='string', 'Ground query did not return a string')
        cells[#cells+1]={ix=ix,iz=iz,x=x,z=z,height=height,clear=clear,ground=ground,
            inside_radar=x<=grid.radar_max_x and z<=grid.radar_max_z}
    end
    return cells,next_index,next_index==grid.count
end

return M
