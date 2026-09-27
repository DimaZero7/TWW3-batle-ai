-- Entry: the player commands side 1 by hand; the script only records.
-- Scenario: scenarios/manual_hamlet.xml (six Empire spearmen near the hamlet).
-- The script never gives orders to the player's units. The far enemy general
-- is held by script. Every tick, for every own unit: movement and every
-- soldier's position; an order change starts a navigation.services monitor,
-- so the log shows arrived / stopped / stuck for each player order.
-- Ends when the player ends the battle, on the real-time deadline, or when
-- nobody takes damage for stall_ms of game time.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local map = require('apps.map.adapter')
local formation = require('apps.units.formation_adapter')
local navigation = require('apps.navigation.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_manual_record_tick'
M.HELD_UNIT = 'far_general'
-- A new order: the ordered point moved more than this, or the width changed.
M.ORDER_MOVE_M = 1
M.ORDER_WIDTH_M = 0.5
M.ORDER_TIMEOUT_MS = 600000

-- config: build, tick_ms, deadline_s, stall_ms. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_manual_record then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0, own = {}}
    _G.tww3_bai_manual_record = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'manual_record'
        row.policy = 'player'
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function fail(err)
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        pcall(emit, 'error', {message = tostring(err)})
        pcall(flush)
    end
    local function guarded(fn)
        return errors.guard(fn, fail, function() return state.finished end)
    end

    local function cco(u, key)
        return common.get_context_value('CcoBattleUnit', tostring(u:unique_ui_id()), key)
    end

    local function finish(status, end_battle)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status, ticks = state.ticks})
        flush()
        if end_battle then bm:end_battle() end
    end

    -- An order is visible only as a change of the ordered point or width.
    local function watch_orders(it, m, now)
        if not m.ordered_x then return end
        local last = it.order
        local changed = not last
            or math.sqrt((m.ordered_x - last.x) ^ 2 + (m.ordered_z - last.z) ^ 2) > M.ORDER_MOVE_M
            or math.abs((m.ordered_width or 0) - (last.width or 0)) > M.ORDER_WIDTH_M
        if not changed then return end
        it.order = {x = m.ordered_x, z = m.ordered_z, width = m.ordered_width, bearing = m.ordered_bearing, ms = now}
        if last then
            emit('order_seen', {unit = it.name, x = m.ordered_x, z = m.ordered_z, width = m.ordered_width,
                bearing = m.ordered_bearing, from_x = m.x, from_z = m.z})
            it.monitor = navigation.new_leg_monitor({target = {x = m.ordered_x, z = m.ordered_z},
                timeout_ms = M.ORDER_TIMEOUT_MS})
        end
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        local now = bm:time_elapsed_ms()
        if state.stall.update(now, battle.health_signature(state.all_units)) then
            finish('stalled', true)
            return
        end
        local rows = {}
        for _, it in ipairs(state.own) do
            local m = formation.motion(it.unit)
            local row = {unit = it.name, motion = m}
            local men = formation.soldiers(cco, it.unit)
            if men.status == 'ok' then row.soldiers_dm = men.xz_dm else row.soldiers_unavailable = men.reason end
            rows[#rows + 1] = row
            watch_orders(it, m, now)
            if it.monitor then
                local reason = it.monitor.update(now - it.order.ms, {x = m.x, z = m.z, moving = m.is_moving == true})
                if reason then
                    local s = it.monitor.stats
                    emit('order_end', {unit = it.name, reason = reason, t_ms = now - it.order.ms, x = m.x, z = m.z,
                        path_m = s.path_m, distance_m = s.distance_m, min_distance_m = s.min_distance_m})
                    it.monitor = nil
                end
            end
        end
        emit('own_sample', {units = rows})
        flush()
    end

    local function start()
        if state.active or state.finished then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:change_victory_countdown_limit(-1)
        -- The engine redeploys the non-player army when deployment ends.
        orders.teleport(state.held_uc, map.vector(vector_type, state.held_spawn.x,
            bm:get_terrain_height(state.held_spawn.x, state.held_spawn.z), state.held_spawn.z), 270, 5)
        orders.halt(state.held_uc)
        emit('start', {tick_ms = config.tick_ms, deadline_s = config.deadline_s, stall_ms = config.stall_ms})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline', true)
        end), 'tww3_bai_manual_record_deadline')
        bm:add_infotext('BAI: RECORDING YOUR ORDERS')
        state.active = true
        flush()
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
    end

    local function initialise()
        emit('loaded')
        local reason = battle.unsupported_reason(bm)
        if reason then
            emit('skipped', {reason = reason})
            state.finished = true
            return
        end
        assert(common and vector_type, 'common and battle_vector globals required')
        local sides = battle.read_sides(bm)
        for _, u in ipairs(sides[1].units) do state.own[#state.own + 1] = {name = u:name(), unit = u} end
        state.held = battle.find_by_name(sides[2], M.HELD_UNIT)
        assert(#state.own > 0 and state.held, 'scenario units missing')
        state.all_units = {state.held}
        for _, it in ipairs(state.own) do state.all_units[#state.all_units + 1] = it.unit end
        local p = state.held:position()
        state.held_spawn = {x = p:get_x(), z = p:get_z()}
        -- Only the enemy general is taken over; the player's units stay the player's.
        state.held_uc = orders.take_control(sides[2].army, state.held)
        orders.set_fire_at_will(state.held_uc, false)
        orders.halt(state.held_uc)
        state.batch = clock.batch_stamp() .. '-manual-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        local names = {}
        for _, it in ipairs(state.own) do names[#names + 1] = it.name end
        emit('ready', {units = names})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('Complete', guarded(function() finish('battle_ended', false) end))
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
