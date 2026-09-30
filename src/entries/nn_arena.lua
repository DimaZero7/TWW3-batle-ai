-- Entry: the arena battle recorder (scenario tools/nn/scenario.py, config/nn/arena.json).
-- Records whole battles of the game's AI against itself as data for training a
-- neural network later. Side 2 is always the game's own AI; it attacks or
-- defends as the battle file sets it (the alliance that wins on timeout defends).
-- Side 1 (ours) is handed to CA's script AI planner:
--   own_ai = 'attack' — the planner attacks the enemy force (re-issued every
--            15 s, as generated battles do);
--   own_ai = 'defend' — the planner defends where the army stands;
--   own_ai = 'hold'   — no planner, no orders: our units stand where they are
--            (a still target for the game's AI; measurements, 30.09.2026).
-- Each side may have its own army (tools/nn/scenario.py: named arenas).
-- Every tick 'nn_sample' records every unit of both sides (full view: trusted
-- research telemetry); 'nn_final' is the last such record.
-- A unit that routs and rallies is given back to the planner with its last
-- order ('rejoined'): without this it stood idle to the end (30.09.2026). A
-- unit idle under fire is kicked ('idle_kick': stage 1 back to the planner,
-- stage 2 a planner of its own that attacks).
-- The battle ends on the game's outcome, a stall, the timeout or the deadline: 'result'.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local planner = require('apps.orders.planner_adapter')
local map = require('apps.map.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_nn_arena_tick'
M.REISSUE_MS = 15000

local function try(fn, ...)
    local ok, v = pcall(fn, ...)
    if ok then return v end
    return nil
end

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- config: build, speed, tick_ms, deadline_s, stall_ms, timeout_ms, own_ai ('attack' | 'defend' | 'hold'),
-- units = {own = [...], enemy = [...]} (script names, slots), defend_radius_m.
function M.main(bm, config, globals)
    if _G.tww3_bai_nn_arena then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        sides = {{}, {}}, by_uid = {}}
    _G.tww3_bai_nn_arena = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local started_ms, started_wall, last_reissue = 0, 0, 0

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'nn_arena'
        row.policy = 'nn_arena_' .. tostring(config.own_ai)
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function cleanup()
        state.active = false
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_process('tww3_bai_nn_arena_start') end)
    end
    local function fail(err)
        cleanup()
        state.finished = true
        pcall(emit, 'error', {message = tostring(err)})
        pcall(flush)
        pcall(function() bm:out('[TWW3 BAI] STOP: ' .. tostring(err)) end)
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
    local function name_of(u)
        if not u then return '' end
        local uid = try(function() return tostring(u:unique_ui_id()) end)
        local it = uid and state.by_uid[uid]
        return it and it.name or '?'
    end

    -- One unit, full state (research telemetry: both sides).
    local function sample(it)
        local u = it.unit
        local p = try(function() return u:position() end)
        local o = try(function() return u:ordered_position() end)
        return {n = it.name, x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(try(function() return u:bearing() end), 0),
            men = try(function() return u:number_of_men_alive() end),
            hp = round(try(function() return u:unary_hitpoints() end), 4),
            mp = round(try(cco, u, 'MoralePercent'), 3), ms = try(cco, u, 'MoraleState'),
            r = try(function() return u:is_routing() end), s = try(function() return u:is_shattered() end),
            w = try(function() return u:is_wavering() end), m = try(function() return u:is_in_melee() end),
            mv = try(function() return u:is_moving() end), f = try(function() return u:is_moving_fast() end),
            a = try(function() return u:ammo_left() end), fire = try(cco, u, 'IsFiringMissiles'),
            t = name_of(try(function() return u:current_target() end)),
            fat = try(function() return u:fatigue_state() end), k = try(cco, u, 'NumKills'),
            ox = o and round(o:get_x()), oz = o and round(o:get_z()),
            lf = try(function() return u:is_left_flank_threatened() end),
            rf = try(function() return u:is_right_flank_threatened() end),
            bf = try(function() return u:is_rear_flank_threatened() end)}
    end

    local function snapshot(event)
        local rows = {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do
                local row = sample(it)
                row.side = side
                rows[#rows + 1] = row
            end
        end
        emit(event, {t = bm:time_elapsed_ms() - started_ms, units = rows})
    end

    local function side_summary()
        local out = {}
        for side = 1, 2 do
            local men, hp, standing = 0, 0, 0
            for _, it in ipairs(state.sides[side]) do
                local u = it.unit
                local m = try(function() return u:number_of_men_alive() end) or 0
                men = men + m
                hp = hp + (try(function() return u:unary_hitpoints() end) or 0)
                if m > 0 and not try(function() return u:is_routing() end)
                    and not try(function() return u:is_shattered() end) then standing = standing + 1 end
            end
            out['side_' .. side .. '_men'], out['side_' .. side .. '_hp'] = men, round(hp, 3)
            out['side_' .. side .. '_standing_units'] = standing
        end
        return out
    end

    local function finish(status, winner)
        if state.finished then return end
        if state.cancel_deadline then state.cancel_deadline() end
        snapshot('nn_final')
        local row = side_summary()
        row.status, row.winner = status, winner or 0
        row.duration_model_ms = bm:time_elapsed_ms() - started_ms
        row.duration_wall_s = clock.elapsed_wall_seconds(started_wall)
        row.rejoined = state.planner and state.planner.rejoined or 0
        row.idle_kicks = state.planner and state.planner.kicks or 0
        emit('result', row)
        cleanup()
        state.finished = true
        flush()
    end

    -- Our units in the planner: rallied units back with its last order, idle ones under fire kicked
    -- (apps.orders.planner_adapter). Not in 'hold': nobody leads our side then.
    local function lead(now)
        local rallied = state.planner.check_rallies()
        if #rallied > 0 then
            local names = {}
            for k, i in ipairs(rallied) do names[k] = state.sides[1][i].name end
            emit('rejoined', {t = now - started_ms, units = names})
        end
        for _, k in ipairs(state.planner.check_idle(now)) do
            emit('idle_kick', {t = now - started_ms, unit = state.sides[1][k.unit].name, stage = k.stage})
        end
    end

    local function tick()
        if not state.active then return end
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled', 0)
            bm:force_battle_end(0, 'stalled', true)
            return
        end
        if bm:battle_outcome_decided() then
            local winner = bm:victorious_alliance()
            if winner ~= 0 then finish('completed', winner) return end
        end
        local now = bm:time_elapsed_ms()
        if now - started_ms >= config.timeout_ms then
            finish('timeout', 0)
            bm:force_battle_end(0, 'timeout', true)
            return
        end
        if state.planner then lead(now) end
        if config.own_ai == 'attack' and now - last_reissue >= M.REISSUE_MS then
            last_reissue = now
            state.planner.attack()
        end
        state.ticks = state.ticks + 1
        snapshot('nn_sample')
        flush()
    end

    local function hand_over()
        local own_units, enemy_units = {}, {}
        for _, it in ipairs(state.sides[1]) do own_units[#own_units + 1] = it.unit end
        for _, it in ipairs(state.sides[2]) do enemy_units[#enemy_units + 1] = it.unit end
        if config.own_ai == 'hold' then
            emit('own_ai', {mode = 'hold', own_ai = config.own_ai})
            return
        end
        state.planner = planner.hand_over(bm, 'tww3_bai_nn_own', state.own_alliance, own_units, enemy_units)
        if config.own_ai == 'defend' then
            local sx, sz = 0, 0
            for _, u in ipairs(own_units) do
                local p = u:position()
                sx, sz = sx + p:get_x(), sz + p:get_z()
            end
            state.planner.defend(vec(sx / #own_units, sz / #own_units), config.defend_radius_m)
        else
            state.planner.attack()
        end
        emit('own_ai', {mode = state.planner.mode, own_ai = config.own_ai})
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end)
        hand_over()
        last_reissue = bm:time_elapsed_ms()
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        emit('start', {speed = config.speed, timeout_ms = config.timeout_ms, deadline_s = config.deadline_s})
        snapshot('nn_sample')
        flush()
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline', 0)
            bm:force_battle_end(0, 'deadline', true)
        end), 'tww3_bai_nn_arena_deadline')
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
    end

    local function initialise()
        emit('loaded')
        local reason = battle.unsupported_reason(bm)
        if reason then
            emit('skipped', {reason = reason})
            state.finished = true
            flush()
            return
        end
        assert(common and vector_type, 'common and battle_vector globals required')
        assert(config.own_ai == 'attack' or config.own_ai == 'defend' or config.own_ai == 'hold',
            'own_ai must be attack, defend or hold')
        local sides = battle.read_sides(bm)
        state.own_alliance = sides[1].alliance
        local ok_roles, roles = pcall(battle.read_roles, bm)
        state.all_units = {}
        for side, key in ipairs({'own', 'enemy'}) do
            for _, spec in ipairs(config.units[key]) do
                local u = battle.find_by_name(sides[side], spec.script_name)
                assert(u, 'scenario unit missing: ' .. spec.script_name)
                local it = {name = spec.script_name, slot = spec.slot, key = spec.key, unit = u, side = side}
                state.sides[side][#state.sides[side] + 1] = it
                state.by_uid[tostring(u:unique_ui_id())] = it
                state.all_units[#state.all_units + 1] = u
            end
        end
        state.batch = clock.batch_stamp() .. '-nn-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {own = #state.sides[1], enemy = #state.sides[2], own_ai = config.own_ai,
            roles = ok_roles and roles or tostring(roles)})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(tick))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active then
                tick()
                if not state.finished then finish('incomplete', 0) end
            end
        end))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_nn_arena_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
