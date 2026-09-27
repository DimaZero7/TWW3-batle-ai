-- Entry: movement probe for the obstacle task (docs/ru/architecture/tasks/obstacles.md).
-- One own unit runs a plan of legs from config.plan (config/move-plans/*.json):
--   shape:    reform in place at another width (formation geometry, reform time);
--   traverse: one plain engine order to a far target (how the engine itself
--             goes around obstacles).
-- Each leg: teleport to the start, settle, one goto_location_angle_width, then
-- every tick the unit's movement and every soldier's position are logged
-- until navigation.services says arrived / stopped / stuck / timeout.
-- Scenario: scenarios/move_probe.xml. The far enemy general is held by script.
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
local TIMER = 'tww3_bai_move_probe_tick'
M.PROBE_UNIT = 'probe_spears'
M.HELD_UNIT = 'far_general'

-- config: build, speed, tick_ms, deadline_s, stall_ms, settle_ms,
-- plan {name, monitor, legs}. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_move_probe then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap',
        leg_index = 0, ticks = 0}
    _G.tww3_bai_move_probe = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local legs = config.plan.legs

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'move_probe'
        row.policy = 'move_probe'
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

    local function snapshot()
        local row = {motion = formation.motion(state.probe)}
        local men = formation.soldiers(cco, state.probe)
        if men.status == 'ok' then row.soldiers_dm = men.xz_dm else row.soldiers_unavailable = men.reason end
        return row
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', legs_done = state.leg_index - (state.leg and 1 or 0),
            legs = #legs, ticks = state.ticks})
        orders.halt(state.probe_uc)
        orders.halt(state.held_uc)
        bm:end_battle()
    end

    local function begin_leg(index)
        local leg = legs[index]
        state.leg_index, state.leg = index, leg
        local s = leg.start
        orders.teleport(state.probe_uc, vec(s.x, s.z), s.facing, s.width)
        orders.halt(state.probe_uc)
        state.phase, state.phase_ms = 'settle', bm:time_elapsed_ms()
        emit('leg_begin', {leg = leg.name, index = index, kind = leg.kind})
    end

    local function order_leg()
        local leg = state.leg
        local t = leg.target
        local before = snapshot()
        orders.move_formation(state.probe_uc, vec(t.x, t.z), t.facing, t.width, leg.run)
        state.phase, state.phase_ms = 'track', bm:time_elapsed_ms()
        local opts = {target = t, timeout_ms = leg.timeout_s * 1000, detect_stuck = leg.kind ~= 'shape'}
        for k, v in pairs(config.plan.monitor or {}) do opts[k] = v end
        state.monitor = navigation.new_leg_monitor(opts)
        before.leg = leg.name
        emit('leg_ordered', before)
    end

    local function track()
        local t_ms = bm:time_elapsed_ms() - state.phase_ms
        local row = snapshot()
        row.leg, row.t_ms = state.leg.name, t_ms
        emit('move_sample', row)
        local m = row.motion
        local reason = state.monitor.update(t_ms, {x = m.x, z = m.z, moving = m.is_moving == true})
        if reason then
            local s = state.monitor.stats
            emit('leg_end', {leg = state.leg.name, reason = reason, t_ms = t_ms, path_m = s.path_m,
                distance_m = s.distance_m, min_distance_m = s.min_distance_m, samples = s.samples})
            state.leg = nil
            if state.leg_index < #legs then begin_leg(state.leg_index + 1) else finish('completed') end
        end
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        -- No damage for stall_ms of game time: end the battle (nobody fights here,
        -- so the build sets stall_ms above the plan length).
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled')
            flush()
            return
        end
        if state.phase == 'settle' then
            if bm:time_elapsed_ms() - state.phase_ms >= config.settle_ms then order_leg() end
        elseif state.phase == 'track' then
            track()
        end
        flush()
    end

    local function start()
        if state.active then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_move_probe_speed')
        bm:change_victory_countdown_limit(-1)
        -- The engine redeploys the non-player army when deployment ends.
        orders.teleport(state.held_uc, vec(state.held_spawn.x, state.held_spawn.z), 270, 5)
        orders.halt(state.held_uc)
        emit('start', {speed = config.speed, tick_ms = config.tick_ms, deadline_s = config.deadline_s,
            stall_ms = config.stall_ms, plan = config.plan})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
            flush()
        end), 'tww3_bai_move_probe_deadline')
        bm:add_infotext('BAI: MOVE PROBE')
        state.active = true
        begin_leg(1)
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
        assert(type(legs) == 'table' and #legs > 0, 'Move plan has no legs')
        local sides = battle.read_sides(bm, 1)
        state.probe = battle.find_by_name(sides[1], M.PROBE_UNIT)
        state.held = battle.find_by_name(sides[2], M.HELD_UNIT)
        assert(state.probe and state.held, 'scenario units missing')
        state.all_units = {state.probe, state.held}
        local p = state.held:position()
        state.held_spawn = {x = p:get_x(), z = p:get_z()}
        -- Take control before deployment ends so the engine AI cannot move them.
        state.probe_uc = orders.take_control(sides[1].army, state.probe)
        state.held_uc = orders.take_control(sides[2].army, state.held)
        for _, uc in ipairs({state.probe_uc, state.held_uc}) do
            orders.set_fire_at_will(uc, false)
            orders.halt(uc)
        end
        state.batch = clock.batch_stamp() .. '-move-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {units = 2, legs = #legs, plan = config.plan.name,
            slow_speed = state.probe:slow_speed(), fast_speed = state.probe:fast_speed(),
            men = state.probe:number_of_men_alive(),
            -- Number precision of the game's Lua: doubles print 1/3 as 0.33333333333333.
            number_check = {third = 1 / 3, big = 16777217, huge = math.huge ~= nil}})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_move_probe_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
