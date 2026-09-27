-- Entry: diagnostic map capture. Separate from the AI entries.
-- Writes tww3_bai_map_capture_{events.jsonl, grid.csv, ready.xml} into the
-- game directory. With config.features it also reads native/CCO objects and
-- per-cell reachability for the first unit of each side after deployment.
-- Scenario: scenarios/map_capture.xml. Launcher: tools/launcher/launch.ps1.
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local map = require('apps.map.adapter')
local map_services = require('apps.map.services')
local navigation = require('apps.navigation.adapter')
local orders = require('apps.orders.adapter')

local M = {}

M.PREFIX = 'tww3_bai_map_capture_'
local BATCH_CELLS = 4096
local BATCH_REACH = 1024

-- config: step (metres, one of 1/2/3/5), features (boolean).
function M.main(bm, config, globals)
    if _G.tww3_bai_map_capture then return end
    _G.tww3_bai_map_capture = true
    local common = globals.common
    local vector_type = globals.battle_vector
    local prefix = M.PREFIX
    local grid_file = prefix .. 'grid.csv'
    local stopped = false

    -- Capture rows historically use `wall` for the UTC timestamp.
    local emit = telemetry.sink(prefix .. 'events.jsonl', function(row)
        row.wall, row.wall_iso = row.wall_iso, nil
    end)
    local function guarded(fn, stage)
        return errors.guard(fn, function(err)
            stopped = true
            emit('probe_error', {stage = stage, message = err})
        end, function() return stopped end)
    end

    local function complete()
        emit('probe_done', {phase = bm:get_current_phase_name(), speed = bm:current_battle_speed()})
    end

    -- Features stage 3: reachability per grid cell for both sides' first unit.
    local function navigation_stage(units)
        for side, u in ipairs(units) do
            local p = u:position()
            emit('navigation_unit', {side = side, unit_type = u:type(), x = p:get_x(), y = p:get_y(),
                z = p:get_z(), phase = bm:get_current_phase_name()})
        end
        local f = assert(io.open(grid_file, 'r'))
        local header = f:read('*l')
        local rows = {}
        for line in f:lines() do rows[#rows + 1] = line end
        f:close()
        telemetry.write_file(grid_file, header .. ',reach_side_1,reach_side_2\n')
        local next_i, start, yes = 1, os.clock(), {0, 0}
        local function batch()
            local cells = {}
            local last = math.min(next_i + BATCH_REACH - 1, #rows)
            for i = next_i, last do
                local ix, iz, x, z, h, _, _, inside = rows[i]:match(
                    '^(%d+),(%d+),([^,]+),([^,]+),([^,]+),(%d+),"([^"]+)",(%d+)$')
                assert(ix, 'Malformed surface CSV')
                cells[#cells + 1] = {ix = tonumber(ix), iz = tonumber(iz), x = tonumber(x), z = tonumber(z),
                    height = tonumber(h), inside_radar = inside == '1'}
            end
            local result = navigation.read_cells(bm, vector_type, units, cells)
            local lines = {}
            for j, row in ipairs(result) do
                local values = {-1, -1}
                if row.inside_radar then
                    for side = 1, 2 do
                        values[side] = row.reachable[side] and 1 or 0
                        yes[side] = yes[side] + values[side]
                    end
                end
                lines[#lines + 1] = rows[next_i + j - 1] .. ',' .. values[1] .. ',' .. values[2] .. '\n'
            end
            telemetry.append(grid_file, table.concat(lines))
            next_i = last + 1
            if next_i > #rows then
                emit('navigation_done', {cells = #rows, reachable_side_1 = yes[1], reachable_side_2 = yes[2],
                    clock_elapsed_s = os.clock() - start, phase = bm:get_current_phase_name(),
                    speed = bm:current_battle_speed()})
                complete()
            else
                bm:real_callback(guarded(batch, 'features'), 20, prefix .. 'navigation_' .. next_i)
            end
        end
        batch()
    end

    -- Features stage 2: CCO structure list.
    local function contexts_stage(units)
        local next_index = 0
        local function batch()
            local rows, following, done, count = map.read_structure_contexts(common, next_index, 64)
            for _, row in ipairs(rows) do emit('structure_context', row) end
            next_index = following
            if done then
                emit('structure_contexts_done', {count = count})
                navigation_stage(units)
            else
                bm:real_callback(guarded(batch, 'features'), 20, prefix .. 'contexts_' .. next_index)
            end
        end
        batch()
    end

    -- Features stage 1: native building inventory.
    local function inventory_stage(units)
        emit('feature_conditions', {version = common.game_version(), phase = bm:get_current_phase_name(),
            speed = bm:current_battle_speed(), is_siege = common.get_context_value('BattleRoot.IsSiege'),
            battle_type = common.get_context_value('BattleRoot.BattleTypeState')})
        local next_index = 0
        local function batch()
            local rows, following, done, count = map.read_buildings(bm, next_index, 256)
            for _, row in ipairs(rows) do emit('building', row) end
            next_index = following
            if done then
                emit('buildings_done', {count = count})
                contexts_stage(units)
            else
                bm:real_callback(guarded(batch, 'features'), 20, prefix .. 'objects_' .. next_index)
            end
        end
        batch()
    end

    -- Takes control so units stay put, deploys at x20, then pauses and scans.
    local function run_features()
        local units = {}
        for side = 1, 2 do
            local army = bm:alliances():item(side):armies():item(1)
            units[side] = army:units():item(1)
            local uc = orders.take_control(army, units[side])
            orders.set_fire_at_will(uc, false)
            orders.halt(uc)
        end
        bm:register_phase_change_callback('Deployed', guarded(function()
            bm:real_callback(guarded(function()
                bm:modify_battle_speed(0)
                -- Speed changes apply on a later engine tick.
                bm:real_callback(guarded(function() inventory_stage(units) end, 'features'),
                    250, prefix .. 'features_paused')
            end, 'features'), 250, prefix .. 'features_deployed')
        end, 'features'))
        bm:modify_battle_speed(20)
        bm:end_current_battle_phase()
    end

    local function initialize()
        assert(bm:get_current_phase_name() == 'Deployment', 'Expected static Deployment phase')
        bm:modify_battle_speed(0)
        bm:output_battle_xml(prefix .. 'ready.xml')
        local frame = map.read_frame(common)
        emit('frame', {min_x = frame.min_x, max_x = frame.max_x, min_z = frame.min_z, max_z = frame.max_z,
            width = frame.width, depth = frame.depth, kind = frame.kind, movement_boundary_verified = false})
        for i, c in ipairs(frame.corners) do emit('corner', {id = i, x = c.x, z = c.z, u = c.u, v = c.v}) end
        local grid = map_services.new_grid(frame, config.step)
        emit('grid_begin', grid)
        telemetry.write_file(grid_file, 'ix,iz,x,z,height,clear,ground,inside_radar\n')
        local next_id, batches, blocked = 0, 0, 0
        local start = type(os.clock) == 'function' and os.clock() or nil
        local function batch()
            local cells, following, done = map.read_batch(bm, vector_type, grid, next_id, BATCH_CELLS)
            local lines = {}
            for _, c in ipairs(cells) do
                if not c.clear then blocked = blocked + 1 end
                local ground = '"' .. c.ground:gsub('"', '""') .. '"'
                lines[#lines + 1] = string.format('%d,%d,%.9f,%.9f,%.6f,%d,%s,%d\n',
                    c.ix, c.iz, c.x, c.z, c.height, c.clear and 1 or 0, ground, c.inside_radar and 1 or 0)
            end
            telemetry.append(grid_file, table.concat(lines))
            next_id = following
            batches = batches + 1
            if done then
                emit('grid_done', {cells = next_id, blocked = blocked, batches = batches,
                    clock_elapsed_s = start and os.clock() - start or 'unavailable'})
                if config.features then run_features() else complete() end
            else
                bm:real_callback(guarded(batch, 'grid'), 20, prefix .. 'batch_' .. batches)
            end
        end
        batch()
    end

    emit('script_loaded')
    bm:real_callback(guarded(initialize, 'grid'), 3000, prefix .. 'init')
end

return M
