-- Entry: roster capture. For every unit of side 1 (scenario built by
-- tools/roster.py from config/roster/capture.json):
--   unit_card    — card stats, details and native profile, read in deployment;
--   shape_result — for every width in config.widths, units with more than one
--                  man reform in place at their own slot (all at once) and the
--                  final soldier positions and reform time are logged.
-- The far enemy general is held by script. tools/roster.py turns the log into
-- data/roster/<unit key>.json.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local map = require('apps.map.adapter')
local card = require('apps.units.card_adapter')
local formation = require('apps.units.formation_adapter')
local navigation = require('apps.navigation.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_roster_capture_tick'
M.HELD_UNIT = 'far_general'

-- config: build, speed, tick_ms, deadline_s, stall_ms, settle_ms, widths,
-- baseline_width, shape_timeout_s, units = {{script_name, key, slot = {x, z}}}.
function M.main(bm, config, globals)
    if _G.tww3_bai_roster_capture then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        units = {}, width_index = 0}
    _G.tww3_bai_roster_capture = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'roster_capture'
        row.policy = 'roster_capture'
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
    local function vec(x, z)
        return map.vector(vector_type, x, bm:get_terrain_height(x, z), z)
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status, widths_done = state.width_index, ticks = state.ticks})
        for _, it in ipairs(state.units) do orders.halt(it.uc) end
        orders.halt(state.held_uc)
        flush()
        bm:end_battle()
    end

    -- One width step: every shaped unit back to its slot at the baseline width.
    local function begin_width(index)
        state.width_index = index
        for _, it in ipairs(state.shaped) do
            orders.teleport(it.uc, vec(it.slot.x, it.slot.z), 0, config.baseline_width)
            orders.halt(it.uc)
            it.monitor, it.done, it.last_moving_ms = nil, false, nil
        end
        state.phase, state.phase_ms = 'settle', bm:time_elapsed_ms()
    end

    local function order_width()
        local width = config.widths[state.width_index]
        for _, it in ipairs(state.shaped) do
            orders.move_formation(it.uc, vec(it.slot.x, it.slot.z), 0, width, false)
            it.monitor = navigation.new_leg_monitor({target = it.slot, timeout_ms = config.shape_timeout_s * 1000,
                detect_stuck = false})
        end
        state.phase, state.phase_ms = 'shape', bm:time_elapsed_ms()
        emit('shape_ordered', {width = width, units = #state.shaped})
    end

    local function track()
        local t_ms = bm:time_elapsed_ms() - state.phase_ms
        local pending = 0
        for _, it in ipairs(state.shaped) do
            if not it.done then
                local m = formation.motion(it.unit)
                if m.is_moving then it.last_moving_ms = t_ms end
                local reason = it.monitor.update(t_ms, {x = m.x, z = m.z, moving = m.is_moving == true})
                if reason then
                    it.done = true
                    local men = formation.soldiers(cco, it.unit)
                    emit('shape_result', {script_name = it.name, key = it.key, width = config.widths[state.width_index],
                        reason = reason, t_ms = t_ms, last_moving_ms = it.last_moving_ms, motion = m,
                        soldiers_dm = men.xz_dm, soldiers_unavailable = men.reason})
                else
                    pending = pending + 1
                end
            end
        end
        if pending == 0 then
            if state.width_index < #config.widths then begin_width(state.width_index + 1) else finish('completed') end
        end
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled')
            return
        end
        if state.phase == 'settle' then
            if bm:time_elapsed_ms() - state.phase_ms >= config.settle_ms then order_width() end
        elseif state.phase == 'shape' then
            track()
        end
        flush()
    end

    local function start()
        if state.active or state.finished then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_roster_capture_speed')
        bm:change_victory_countdown_limit(-1)
        orders.teleport(state.held_uc, vec(state.held_spawn.x, state.held_spawn.z), 270, 5)
        orders.halt(state.held_uc)
        emit('start', {speed = config.speed, deadline_s = config.deadline_s, stall_ms = config.stall_ms,
            widths = config.widths, shaped = #state.shaped})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_roster_capture_deadline')
        bm:add_infotext('BAI: ROSTER CAPTURE')
        state.active = true
        if #state.shaped == 0 then finish('completed') return end
        begin_width(1)
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
        state.held = battle.find_by_name(sides[2], M.HELD_UNIT)
        assert(state.held, 'scenario unit missing: ' .. M.HELD_UNIT)
        local p = state.held:position()
        state.held_spawn = {x = p:get_x(), z = p:get_z()}
        state.held_uc = orders.take_control(sides[2].army, state.held)
        orders.set_fire_at_will(state.held_uc, false)
        orders.halt(state.held_uc)
        state.all_units, state.shaped = {state.held}, {}
        for _, spec in ipairs(config.units) do
            local u = battle.find_by_name(sides[1], spec.script_name)
            assert(u, 'scenario unit missing: ' .. spec.script_name)
            local it = {name = spec.script_name, key = spec.key, slot = spec.slot, unit = u}
            it.uc = orders.take_control(sides[1].army, u)
            orders.set_fire_at_will(it.uc, false)
            orders.halt(it.uc)
            state.units[#state.units + 1] = it
            state.all_units[#state.all_units + 1] = u
            if u:initial_number_of_men() > 1 then state.shaped[#state.shaped + 1] = it end
        end
        state.batch = clock.batch_stamp() .. '-roster-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {units = #state.units, game_version = common.game_version()})
        for _, it in ipairs(state.units) do
            emit('unit_card', {script_name = it.name, key = it.key, profile = card.profile(it.unit),
                stats = card.stats(cco, it.unit), details = card.details(cco, it.unit)})
        end
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_roster_capture_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
