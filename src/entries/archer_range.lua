-- Entry: when do archers start shooting, and does the depth of their block
-- matter? (user, 28.09.2026: in a deep, column-like block the enemy is in range
-- of the front ranks but the unit does not shoot — as if the rear ranks keep
-- their own range.)
-- Lanes 200 m apart; in each an archer unit at its own width (so its own
-- depth) and in front of it one enemy unit held by script. All script
-- controlled; the far generals are held.
--   mode 'fire_at_will': archers stand, fire at will on; the target is moved
--     step_m closer every step_ticks ticks from start_d to end_d (metres from
--     the archers' front rank to the target's front rank).
--   mode 'attack': the target stands at start_d; every archer is ordered to
--     attack its target (it walks into range by itself).
-- Every tick, for every lane: ammo left, the engine's firing flag and
-- unit_in_range, the archers' front and rear ranks and the target's nearest
-- rank along the lane (from the soldiers) -> 'range_sample'. Research only.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local map = require('apps.map.adapter')
local unit_motion = require('apps.units.formation_adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_archer_range_tick'

-- config: build, speed, tick_ms, deadline_s, stall_ms, mode, start_d, end_d,
-- step_m, step_ticks, after_ticks, z0, lanes = {{archer, target, width, x}},
-- held = {script names}. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_archer_range then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0, lanes = {}}
    _G.tww3_bai_archer_range = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local bearing = facing.snap(0)
    local back_bearing = facing.snap(180)
    local fx, fz = math.sin(math.rad(bearing)), math.cos(math.rad(bearing))

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'archer_range'
        row.policy = 'archer_range'
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
    local function read(fn)
        local ok, v = pcall(fn)
        if ok then return v end
        return nil
    end

    -- Soldiers along the lane: {min, max} of (point - lane origin) . forward.
    local function extent(u, lane)
        local men = unit_motion.soldiers(cco, u)
        if men.status ~= 'ok' then return nil end
        local lo, hi
        local p = men.xz_dm
        for i = 1, #p, 2 do
            local a = (p[i] / 10 - lane.x) * fx + (p[i + 1] / 10 - config.z0) * fz
            lo, hi = math.min(lo or a, a), math.max(hi or a, a)
        end
        return {lo, hi}
    end

    local function place_target(lane, d)
        -- The target's front rank faces back towards the archers, d ahead of theirs.
        orders.teleport(lane.tuc, vec(lane.x + fx * d, config.z0 + fz * d), back_bearing, 30)
        orders.halt(lane.tuc)
        lane.d = d
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', ticks = state.ticks})
        flush()
        bm:end_battle()
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        local rows, all_fired = {}, true
        for _, lane in ipairs(state.lanes) do
            local ammo = read(function() return lane.archer:ammo_left() end)
            lane.first_ammo = lane.first_ammo or ammo
            local fired = ammo and lane.first_ammo and ammo < lane.first_ammo
            if fired and not lane.fired_at then lane.fired_at = state.ticks end
            if not lane.fired_at then all_fired = false end
            local a, t = extent(lane.archer, lane), extent(lane.target, lane)
            rows[#rows + 1] = {lane = lane.i, width = lane.width, d_m = lane.d, ammo = ammo,
                firing = read(function() return cco(lane.archer, 'IsFiringMissiles') end),
                in_range = read(function() return lane.archer:unit_in_range(lane.target) end),
                moving = lane.archer:is_moving(), archer_front = a and a[2], archer_rear = a and a[1],
                target_near = t and t[1], range = read(function() return lane.archer:missile_range() end)}
        end
        emit('range_sample', {tick = state.ticks, lanes = rows})
        if config.mode == 'fire_at_will' and state.ticks % config.step_ticks == 0 then
            for _, lane in ipairs(state.lanes) do
                if not lane.fired_at and lane.d - config.step_m >= config.end_d then place_target(lane, lane.d - config.step_m) end
            end
        end
        local low = true
        for _, lane in ipairs(state.lanes) do
            if lane.d - config.step_m >= config.end_d and not lane.fired_at then low = false end
        end
        if all_fired or low then
            state.after = (state.after or 0) + 1
            if state.after >= config.after_ticks then finish(all_fired and 'completed' or 'end_distance') end
        end
        flush()
    end

    local function start()
        if state.active or state.finished then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_archer_range_speed')
        bm:change_victory_countdown_limit(-1)
        for _, lane in ipairs(state.lanes) do
            orders.teleport(lane.auc, vec(lane.x, config.z0), bearing, lane.width)
            orders.halt(lane.auc)
            place_target(lane, config.start_d)
        end
        emit('start', {mode = config.mode, bearing = bearing, lanes = config.lanes, start_d = config.start_d,
            end_d = config.end_d, step_m = config.step_m, step_ticks = config.step_ticks})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_archer_range_deadline')
        bm:add_infotext('BAI: ARCHER RANGE')
        -- Teleports apply on a later tick; let the blocks form before anyone shoots.
        bm:callback(guarded(function()
            for _, lane in ipairs(state.lanes) do
                if config.mode == 'attack' then
                    orders.attack_ranged(lane.auc, lane.target)
                else
                    orders.set_fire_at_will(lane.auc, true)
                end
            end
            state.active = true
            bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        end), 3000, 'tww3_bai_archer_range_go')
        flush()
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
        state.all_units = {}
        local function take(side, name)
            local u = battle.find_by_name(sides[side], name)
            assert(u, 'scenario unit missing: ' .. name)
            local uc = orders.take_control(sides[side].army, u)
            orders.set_fire_at_will(uc, false)
            orders.halt(uc)
            state.all_units[#state.all_units + 1] = u
            return u, uc
        end
        for i, l in ipairs(config.lanes) do
            local lane = {i = i, x = l.x, width = l.width}
            lane.archer, lane.auc = take(1, l.archer)
            lane.target, lane.tuc = take(2, l.target)
            state.lanes[i] = lane
        end
        for _, h in ipairs(config.held) do take(h.side, h.name) end
        state.batch = clock.batch_stamp() .. '-range-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {lanes = #state.lanes, mode = config.mode})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_archer_range_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
