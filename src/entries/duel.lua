-- Entry: one unit per side, script-controlled melee duel with automatic
-- rematches. Scenario: scenarios/ranged_melee.xml (or a custom 1v1 lord duel).
-- Wires apps together; decisions come from an ai policy, orders from
-- orders.adapter, events go to tww3_bai_events.jsonl.
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local ai = require('apps.ai.services')
local ai_contract = require('apps.ai.contract')
local bundled_policy = require('apps.ai.policies.forced_melee')

local M = {}

local FILES = {
    log = 'tww3_bai_events.jsonl',
    pending = 'tww3_bai_pending.txt',
    sequence = 'tww3_bai_sequence.txt',
    policy = 'tww3_bai_policy.lua',
}
local TIMER = 'tww3_bai_duel_tick'
-- "переиграть эту битву" in the Russian UI: the rematch confirmation text.
-- Other game languages are not supported yet.
local REMATCH_PROMPT = '\208\191\208\181\209\128\208\181\208\184\208\179\209\128\208\176\209\130\209\140'
    .. '\032\209\141\209\130\209\131\032\208\177\208\184\209\130\208\178\209\131'

-- config: build, runs, speed, timeout_ms, tick_ms.
function M.main(bm, config)
    if _G.tww3_bai_duel then return end
    local state = {active = false, finished = false, units = {}, run = 1, batch = '', run_id = 'bootstrap'}
    _G.tww3_bai_duel = state
    local policy = bundled_policy
    local started_ms, started_wall = 0, 0
    local speed_changed = false
    local scenario, automated = 'custom_duel', false

    local emit = telemetry.sink(FILES.log, function(row)
        row.schema = 1
        row.build, row.policy = config.build, policy.version
        row.run_id, row.batch, row.run = state.run_id, state.batch, state.run
        row.scenario = scenario or 'custom_duel'
        row.model_ms, row.wall_time = bm:time_elapsed_ms(), os.time()
    end)

    local function cleanup(restore_speed)
        state.active = false
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_real_callback('tww3_bai_duel_restart') end)
        pcall(function() bm:remove_real_callback('tww3_bai_duel_confirm') end)
        pcall(function() bm:remove_process('tww3_bai_duel_start') end)
        for _, item in ipairs(state.units) do
            if item.controller then orders.release(item.controller) end
        end
        if speed_changed and restore_speed ~= false then
            if state.stop_speed_guard then state.stop_speed_guard() end
            pcall(function() bm:modify_battle_speed(state.original_speed) end)
            speed_changed = false
        end
    end

    local function fail(err)
        cleanup()
        state.finished = true
        pcall(telemetry.write_file, FILES.pending, '')
        pcall(emit, 'error', {message = tostring(err)})
        pcall(function() bm:out('[TWW3 BAI] STOP: ' .. tostring(err)) end)
        -- A diagnostic failure must be visible, not silently resemble vanilla AI.
        pcall(function() bm:add_infotext('BAI ERROR: ' .. tostring(err):match('^[^\n]+')) end)
    end

    local function guarded(fn)
        return errors.guard(fn, fail, function() return state.finished end)
    end

    local function snapshot(item, event)
        local u = item.unit
        local p = u:position()
        emit(event or 'snapshot', {
            side = item.side, unit_type = u:type(), x = p:get_x(), z = p:get_z(),
            hp = u:unary_hitpoints(), men = u:number_of_men_alive(),
            routing = u:is_routing(), shattered = u:is_shattered(),
            in_melee = u:is_in_melee(), script_controlled = u:is_script_controlled(),
            ammo = u:ammo_left(), starting_ammo = u:starting_ammo(),
            fire_at_will = u:is_behaviour_active('fire_at_will'),
            skirmish = u:is_behaviour_active('skirmish'),
        })
    end

    -- Presses the results screen's rematch button through the native UI API
    -- (no OS mouse). The generic RestartBattle CCO command did not restart
    -- direct XML battles.
    local function restart_when_ready_loop()
        local attempts = 0
        local function restart_when_ready()
            local ok, err = pcall(function()
                attempts = attempts + 1
                local popup = find_uicomponent(core:get_ui_root(), 'in_battle_results_popup')
                local button = popup and find_uicomponent(popup, 'button_rematch')
                if attempts == 1 or attempts % 20 == 0 then
                    emit('restart_wait', {attempt = attempts, phase = bm:get_current_phase_name(),
                        popup_found = popup ~= nil and popup ~= false,
                        button_found = button ~= nil and button ~= false,
                        popup_visible = popup and popup:Visible() or false,
                        button_state = button and button:CurrentState() or 'missing'})
                end
                if popup and popup:Visible() and button and button:Visible()
                    and button:CurrentState() ~= 'inactive' then
                    emit('restart_requested', {method = 'results_rematch_callback'})
                    button:SimulateLClick()
                    local confirm_attempts = 0
                    local function confirm_rematch()
                        local ok_confirm, confirm_error = pcall(function()
                            confirm_attempts = confirm_attempts + 1
                            local dialog = find_uicomponent(core:get_ui_root(), 'dialogue_box')
                            local label = dialog and find_uicomponent(dialog, 'DY_text')
                            local yes = dialog and find_uicomponent(dialog, 'both_group', 'button_tick')
                            if dialog and dialog:Visible() and label and yes and yes:Visible() then
                                local text = label:GetStateText()
                                -- WH3 replaces string.find with a native UTF-8 variant.
                                -- Use the original Lua function for an exact byte search.
                                local find_plain = string.find_lua or string.find
                                assert(find_plain(text, REMATCH_PROMPT, 1, true),
                                    'unexpected confirmation after rematch: ' .. text)
                                emit('restart_confirmed', {prompt = text})
                                yes:SimulateLClick()
                            else
                                assert(confirm_attempts < 20, 'rematch confirmation unavailable after 10 seconds')
                                bm:real_callback(confirm_rematch, 500, 'tww3_bai_duel_confirm')
                            end
                        end)
                        if not ok_confirm then fail(confirm_error) end
                    end
                    bm:real_callback(confirm_rematch, 500, 'tww3_bai_duel_confirm')
                else
                    assert(attempts < 60, 'results rematch button unavailable after 30 seconds')
                    bm:real_callback(restart_when_ready, 500, 'tww3_bai_duel_restart')
                end
            end)
            if not ok then fail(err) end
        end
        bm:real_callback(restart_when_ready, 500, 'tww3_bai_duel_restart')
    end

    local function matchup()
        return state.units[1].unit:type() .. ':' .. state.units[2].unit:type()
    end

    local function finish(status, winner)
        if state.finished then return end
        for _, item in ipairs(state.units) do snapshot(item, 'final_unit') end
        emit('result', {status = status, winner = winner or 0,
            duration_model_ms = bm:time_elapsed_ms() - started_ms,
            duration_wall_s = clock.elapsed_wall_seconds(started_wall)})
        cleanup(not automated)
        state.finished = true
        bm:out('[TWW3 BAI] ' .. status .. '; winner=' .. tostring(winner or 0) .. '; ' .. state.run_id)
        -- Restart only after a final engine result; never retry errors/timeouts in a loop.
        if status == 'completed' and state.run < config.runs then
            telemetry.write_file(FILES.pending, table.concat(
                {config.build, state.batch, state.run + 1, matchup()}, '\t'))
            bm:end_battle()
            restart_when_ready_loop()
        else
            telemetry.write_file(FILES.pending, '')
        end
    end

    local function check_result()
        if not bm:battle_outcome_decided() then return false end
        local winner = bm:victorious_alliance()
        if winner == 0 then return false end
        finish('completed', winner)
        return true
    end

    local function read_unit(u)
        return {routing = u:is_routing(), shattered = u:is_shattered(), men = u:number_of_men_alive(),
            in_melee = u:is_in_melee(), idle = u:is_idle()}
    end

    local function tick()
        if not state.active then return end
        if check_result() then return end
        local now = bm:time_elapsed_ms()
        if now - started_ms >= config.timeout_ms then
            finish('timeout', 0)
            -- A forced draw is a timeout in our telemetry, never a natural draw.
            bm:force_battle_end(0, 'timeout', true)
            return
        end
        -- Engine orders may become visible only on a subsequent model tick.
        for _, item in ipairs(state.units) do
            if not item.control_confirmed then
                if not item.unit:is_script_controlled() then
                    assert(now - started_ms < 2000, 'engine did not grant script control for side ' .. item.side)
                    return
                end
                item.control_confirmed = true
                emit('control_acquired', {side = item.side, unit_type = item.unit:type(),
                    attack_mode = 'melee', engine_speed = bm:current_battle_speed()})
            end
        end
        for index, item in ipairs(state.units) do
            local target = state.units[3 - index].unit
            local observation = ai.duel_observation(read_unit(item.unit),
                {valid = target:is_valid_target(), visible = target:is_visible_to_alliance(item.alliance)},
                item, now)
            local action, reason = ai.decide_duel(policy, observation)
            if action == 'attack' then
                orders.attack_melee(item.controller, target)
                item.ordered, item.last_order_ms = true, now
            end
            if action == 'attack' or reason ~= item.last_reason then
                emit('decision', {side = item.side, action = action, reason = reason,
                    target_side = 3 - index, attack_mode = 'melee'})
                item.last_reason = reason
            end
            snapshot(item)
        end
    end

    -- File I/O + loadstring, not require: reread the policy for every battle.
    -- The external file runs with full Lua access; only load files you trust.
    local function load_external_policy()
        local external = telemetry.read_file(FILES.policy)
        if not external then return nil end
        local chunk, err = loadstring(external, '@' .. FILES.policy)
        assert(chunk, err)
        return ai_contract.check_duel_policy(chunk())
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        local external = load_external_policy()
        if external then policy = external end
        emit('start', {policy_source = external and 'external' or 'bundled',
            speed = config.speed, max_runs = config.runs})
        state.original_speed = bm:current_battle_speed()
        speed_changed = true
        bm:modify_battle_speed(config.speed)
        state.stop_speed_guard = battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end)
        for _, item in ipairs(state.units) do
            item.controller = orders.take_control(item.army, item.unit)
            orders.prepare_melee(item.controller, item.unit)
            emit('control_requested', {side = item.side})
        end
        bm:add_infotext('BAI ACTIVE: SCRIPT CONTROL / FORCED MELEE')
        state.active = true
        tick()
        if state.active then bm:repeat_callback(guarded(tick), config.tick_ms, TIMER) end
    end

    local function initialise()
        emit('loaded')
        local reason = battle.unsupported_reason(bm)
        if reason then
            emit('skipped', {reason = reason})
            state.finished = true
            return
        end
        local sides = battle.read_sides(bm, 1)
        for side = 1, 2 do
            local info = sides[side]
            state.units[side] = {side = side, unit = info.units[1], army = info.army, alliance = info.alliance}
        end
        -- Engine script names survive Lua environment boundaries; a scenario global does not.
        automated = state.units[1].unit:name() == 'bai_ranged_a' and state.units[2].unit:name() == 'bai_ranged_b'
        if automated then scenario = 'ranged_melee' end
        for _, item in ipairs(state.units) do
            local unit = item.unit
            if automated then
                assert(unit:type() == 'wh3_main_ksl_inf_kossars_0', 'unexpected unit in the ranged scenario')
            else
                assert(unit:is_commanding_unit() and unit:initial_number_of_men() == 1 and unit:is_infantry(),
                    'custom duel requires one foot lord per side')
            end
        end
        local sequence = telemetry.next_sequence(FILES.sequence)
        state.batch = clock.batch_stamp() .. '-' .. sequence
        local pending = telemetry.read_file(FILES.pending) or ''
        local build, batch, run, pair = pending:match('^([^\t]+)\t([^\t]+)\t(%d+)\t([^\t]+)$')
        if build == config.build and pair == matchup() and tonumber(run) <= config.runs then
            state.batch, state.run = batch, tonumber(run)
        end
        -- Consume before running; a crash must not create an endless retry.
        telemetry.write_file(FILES.pending, '')
        state.run_id = state.batch .. '-r' .. state.run .. '-s' .. sequence
        emit('ready', {unit_type = state.units[1].unit:type(), enemy_unit_type = state.units[2].unit:type(),
            max_runs = config.runs, automatic = automated})
        for _, item in ipairs(state.units) do snapshot(item, 'initial_unit') end
        bm:out('[TWW3 BAI] Ready: duel, run ' .. state.run .. '/' .. config.runs)
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(function()
            if state.active then check_result() end
        end))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active and not check_result() then finish('incomplete', 0) end
        end))
        -- The XML scenario and follow-up runs never require a Start Battle click.
        local function deployment()
            emit('deployment')
            if automated or state.run > 1 then
                bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_duel_start')
            end
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
