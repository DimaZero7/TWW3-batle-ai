-- Entry: both armies fought by the game's own AI; our code only observes.
-- A side already run by the general battle AI is left untouched. A side the
-- engine marks as player-controlled is handed to CA's script AI planner with
-- "attack the enemy force" (orders.planner_adapter), re-issued periodically
-- like generated battles do. Scenario: scenarios/ai_vs_ai.xml.
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local planner = require('apps.orders.planner_adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_ai_vs_ai_tick'
local REISSUE_MS = 15000
local SNAPSHOT_EVERY = 5

-- config: build, speed, timeout_ms, tick_ms.
function M.main(bm, config)
    if _G.tww3_bai_ai_vs_ai then return end
    local state = {active = false, finished = false, sides = {}, batch = '', run_id = 'bootstrap', ticks = 0}
    _G.tww3_bai_ai_vs_ai = state
    local started_ms, started_wall, last_reissue = 0, 0, 0

    local emit = telemetry.sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'ai_vs_ai'
        row.policy = 'vanilla_ai'
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function cleanup()
        state.active = false
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_process('tww3_bai_ai_vs_ai_start') end)
    end

    local function fail(err)
        cleanup()
        state.finished = true
        pcall(emit, 'error', {message = tostring(err)})
        pcall(function() bm:out('[TWW3 BAI] STOP: ' .. tostring(err)) end)
    end

    local function guarded(fn)
        return errors.guard(fn, fail, function() return state.finished end)
    end

    local function unit_row(side, u)
        local p = u:position()
        return {side = side, unit_name = u:name(), unit_type = u:type(), x = p:get_x(), z = p:get_z(),
            men = u:number_of_men_alive(), hp = u:unary_hitpoints(), routing = u:is_routing(),
            shattered = u:is_shattered(), in_melee = u:is_in_melee(), moving = u:is_moving(),
            script_controlled = u:is_script_controlled()}
    end

    local function snapshot(event)
        for _, info in ipairs(state.sides) do
            for _, u in ipairs(info.units) do emit(event, unit_row(info.side, u)) end
        end
    end

    local function side_summary()
        local out = {}
        for _, info in ipairs(state.sides) do
            local men, standing = 0, 0
            for _, u in ipairs(info.units) do
                men = men + u:number_of_men_alive()
                if u:number_of_men_alive() > 0 and not u:is_routing() and not u:is_shattered() then
                    standing = standing + 1
                end
            end
            out['side_' .. info.side .. '_men'] = men
            out['side_' .. info.side .. '_standing_units'] = standing
        end
        return out
    end

    local function finish(status, winner)
        if state.finished then return end
        snapshot('final_unit')
        local row = side_summary()
        row.status, row.winner = status, winner or 0
        row.duration_model_ms = bm:time_elapsed_ms() - started_ms
        row.duration_wall_s = clock.elapsed_wall_seconds(started_wall)
        emit('result', row)
        cleanup()
        state.finished = true
        bm:out('[TWW3 BAI] ai_vs_ai ' .. status .. '; winner=' .. tostring(winner or 0))
    end

    local function tick()
        if not state.active then return end
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
        if now - last_reissue >= REISSUE_MS then
            last_reissue = now
            for _, info in ipairs(state.sides) do
                if info.planner then info.planner.attack() end
            end
        end
        state.ticks = state.ticks + 1
        if state.ticks % SNAPSHOT_EVERY == 0 then
            snapshot('snapshot')
            emit('progress', side_summary())
        end
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        bm:modify_battle_speed(config.speed)
        for _, info in ipairs(state.sides) do
            if info.player_controlled then
                local enemy = state.sides[3 - info.side]
                info.planner = planner.hand_over(bm, 'bai_side_' .. info.side, info.alliance, info.units, enemy.units)
                info.planner.attack()
            end
            emit('ai_assigned', {side = info.side,
                controller = info.planner and info.planner.mode or 'general_battle_ai'})
        end
        last_reissue = bm:time_elapsed_ms()
        emit('start', {speed = config.speed, timeout_ms = config.timeout_ms})
        bm:add_infotext('BAI: VANILLA AI vs VANILLA AI (observer)')
        state.active = true
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
        local sides = battle.read_sides(bm)
        for side = 1, 2 do
            local info = sides[side]
            local ok, player = pcall(function() return info.army:is_player_controlled() end)
            info.player_controlled = ok and player == true
            info.player_controlled_status = ok and 'known' or 'unknown'
            state.sides[side] = info
        end
        state.batch = clock.batch_stamp() .. '-aivai-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {side_1_units = #sides[1].units, side_2_units = #sides[2].units,
            side_1_player_controlled = state.sides[1].player_controlled,
            side_2_player_controlled = state.sides[2].player_controlled,
            player_controlled_status = state.sides[1].player_controlled_status})
        snapshot('initial_unit')
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(tick))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active then
                tick()
                if not state.finished then finish('incomplete', 0) end
            end
        end))
        local function deployment()
            emit('deployment')
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_ai_vs_ai_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
