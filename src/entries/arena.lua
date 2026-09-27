-- Entry: three simultaneous paired melee encounters on one map load.
-- Experimental isolation: pairs share army morale and are only separated by
-- distance. Scenario: scenarios/triple_melee.xml.
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local value = require('apps.core.value')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local ai = require('apps.ai.services')
local ai_contract = require('apps.ai.contract')
local bundled_policy = require('apps.ai.policies.forced_melee')

local M = {}

local TIMER = 'tww3_bai_arena_tick'
local LOG = 'tww3_bai_events.jsonl'
local PROXIMITY_WARNING_M = 120

-- Pair id -> expected roster and spawn Z. Unit script names: bai_arena_<id>_<side>.
M.CASES = {
    {kind = 'wh3_main_ksl_inf_kossars_0', men = 120, z = -340},
    {kind = 'wh3_main_ksl_inf_tzar_guard_1', men = 100, z = -20},
    {kind = 'wh3_main_ksl_mon_snow_leopard_0', men = 1, z = 300},
}

-- config: build, speed, timeout_ms, tick_ms.
function M.main(bm, config)
    if _G.tww3_bai_arena then return end
    local state = {pairs = {}, units = {}, active = false, finished = false, batch = ''}
    _G.tww3_bai_arena = state
    local policy = bundled_policy
    local started_ms, started_wall = 0, 0

    local sink = telemetry.sink(LOG, function(row)
        row.schema, row.build = 1, config.build
        row.batch, row.policy, row.scenario = state.batch, policy.version, 'triple_melee'
        row.model_ms = bm:time_elapsed_ms()
    end)
    local function emit(event, pair, fields)
        local row = fields or {}
        row.run, row.pair = pair and pair.id or 0, pair and pair.id or 0
        row.run_id = pair and (state.batch .. '-p' .. pair.id) or 'bootstrap'
        return sink(event, row)
    end

    local function cleanup()
        state.active = false
        bm:remove_process(TIMER)
        bm:remove_process('tww3_bai_arena_start')
        for _, item in ipairs(state.units) do
            if item.controller then orders.release(item.controller) end
        end
    end

    local function fail(err)
        state.finished = true
        pcall(cleanup)
        pcall(function() bm:modify_battle_speed(1) end)
        pcall(emit, 'error', nil, {message = tostring(err)})
        pcall(function() bm:out('[TWW3 BAI ARENA] ERROR: ' .. tostring(err)) end)
    end

    local function guarded(fn)
        return errors.guard(fn, fail, function() return state.finished end)
    end

    local function snapshot(pair, item, event)
        local u, p = item.unit, item.unit:position()
        emit(event or 'snapshot', pair, {
            side = item.side, unit_type = u:type(), unit_name = u:name(),
            x = p:get_x(), z = p:get_z(), men = u:number_of_men_alive(), hp = u:unary_hitpoints(),
            ammo = u:ammo_left(), starting_ammo = u:starting_ammo(),
            routing = u:is_routing(), shattered = u:is_shattered(), in_melee = u:is_in_melee(),
            script_controlled = u:is_script_controlled(),
            fire_at_will = u:is_behaviour_active('fire_at_will'),
        })
    end

    local function finish_pair(pair, status, winner, reason)
        if pair.finished then return end
        pair.finished = true
        pair.status = status
        for _, item in ipairs(pair.units) do snapshot(pair, item, 'final_unit') end
        emit('result', pair, {
            status = status, winner = winner, reason = reason, result_basis = 'first_rout_or_death',
            duration_model_ms = bm:time_elapsed_ms() - started_ms,
            duration_wall_s = clock.elapsed_wall_seconds(started_wall),
            isolation = pair.contaminated and 'proximity_warning' or 'distance_only',
            min_other_pair_distance = pair.min_other_distance or -1,
            shared_army_morale = true,
        })
        -- Keep script control: a victorious vanilla AI must not join a neighbour.
        for _, item in ipairs(pair.units) do orders.halt(item.controller) end
    end

    local function finish_arena()
        local completed, warnings = 0, 0
        for _, pair in ipairs(state.pairs) do
            if pair.status == 'completed' then completed = completed + 1 end
            if pair.contaminated then warnings = warnings + 1 end
        end
        state.finished = true
        emit('arena_complete', nil, {status = completed == 3 and 'completed' or 'incomplete',
            completed_pairs = completed, map_loads = 1, proximity_warning_pairs = warnings,
            duration_model_ms = bm:time_elapsed_ms() - started_ms, shared_army_morale = true})
        cleanup()
        bm:end_battle()
    end

    local function defeated(u)
        return u:number_of_men_alive() == 0 or u:is_shattered() or u:is_routing()
    end

    local function proximity()
        for a = 1, #state.units do
            local ua = state.units[a]
            if ua.unit:number_of_men_alive() > 0 then
                local pa = ua.unit:position()
                for b = a + 1, #state.units do
                    local ub = state.units[b]
                    if ua.pair ~= ub.pair and ub.unit:number_of_men_alive() > 0 then
                        local pb = ub.unit:position()
                        local distance = value.distance_xz({x = pa:get_x(), z = pa:get_z()},
                            {x = pb:get_x(), z = pb:get_z()})
                        for _, pair in ipairs({ua.pair, ub.pair}) do
                            if not pair.finished then
                                pair.min_other_distance = math.min(pair.min_other_distance or distance, distance)
                                if distance < PROXIMITY_WARNING_M and not pair.contaminated then
                                    pair.contaminated = true
                                    emit('isolation_warning', pair, {reason = 'other_pair_within_120m',
                                        distance = distance, unit_a = ua.unit:name(), unit_b = ub.unit:name()})
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    local function tick()
        if not state.active then return end
        local now = bm:time_elapsed_ms()
        for _, item in ipairs(state.units) do
            if not item.control_confirmed then
                if not item.unit:is_script_controlled() then
                    assert(now - started_ms < 2000, 'script control not granted: ' .. item.unit:name())
                    return
                end
                item.control_confirmed = true
                local p = item.unit:position()
                assert(math.abs(p:get_z() - item.pair.case.z) < 30, 'arena placement not applied: ' .. item.unit:name())
                emit('control_acquired', item.pair, {side = item.side, engine_speed = bm:current_battle_speed()})
                snapshot(item.pair, item, 'initial_unit')
            end
        end
        proximity()
        local finished = 0
        for _, pair in ipairs(state.pairs) do
            if not pair.finished then
                local a, b = defeated(pair.units[1].unit), defeated(pair.units[2].unit)
                if a or b then
                    finish_pair(pair, 'completed', a and (b and -1 or 2) or 1,
                        a and b and 'both_defeated' or 'rout_or_death')
                elseif now - started_ms >= config.timeout_ms then
                    finish_pair(pair, 'timeout', 0, 'model_timeout')
                end
            end
            if pair.finished then
                finished = finished + 1
                -- Also halt units that rally after a locally recorded first-rout result.
                for _, item in ipairs(pair.units) do orders.halt(item.controller) end
            else
                for side, item in ipairs(pair.units) do
                    local u, target = item.unit, pair.units[3 - side].unit
                    local observation = ai.duel_observation(
                        {routing = u:is_routing(), shattered = u:is_shattered(), men = u:number_of_men_alive(),
                            in_melee = u:is_in_melee(), idle = u:is_idle()},
                        {valid = target:is_valid_target(), visible = target:is_visible_to_alliance(item.alliance)},
                        {ordered = item.ordered, last_order_ms = item.last_order_ms or started_ms}, now)
                    local action, reason = ai.decide_duel(policy, observation)
                    if action == 'attack' then
                        orders.attack_melee(item.controller, target)
                        item.ordered, item.last_order_ms = true, now
                    end
                    if action == 'attack' or reason ~= item.last_reason then
                        emit('decision', pair, {side = side, action = action, reason = reason,
                            target_side = 3 - side, target_name = target:name(), attack_mode = 'melee'})
                        item.last_reason = reason
                    end
                    snapshot(pair, item)
                end
            end
        end
        if finished == 3 then finish_arena() end
    end

    local function start()
        if state.active then return end
        local external = telemetry.read_file('tww3_bai_policy.lua')
        if external then
            local chunk, err = loadstring(external, '@tww3_bai_policy.lua')
            assert(chunk, err)
            policy = ai_contract.check_duel_policy(chunk())
        end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        bm:modify_battle_speed(config.speed)
        bm:change_victory_countdown_limit(-1)
        for _, pair in ipairs(state.pairs) do
            emit('start', pair, {speed = config.speed, policy_source = external and 'external' or 'bundled',
                shared_army_morale = true, result_basis = 'first_rout_or_death'})
            for _, item in ipairs(pair.units) do
                -- Restore the validated XML positions after the engine's deployment transition.
                orders.teleport(item.controller, item.spawn, item.side == 1 and 270 or 90,
                    pair.case.men > 1 and 30 or 5)
            end
        end
        state.active = true
        bm:add_infotext('BAI: 3 PARALLEL DUELS / x20 / ONE MAP')
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
    end

    local function initialise()
        local reason = battle.unsupported_reason(bm)
        if reason then
            emit('skipped', nil, {reason = reason})
            state.finished = true
            return
        end
        for id, case in ipairs(M.CASES) do state.pairs[id] = {id = id, case = case, units = {}} end
        local sides = battle.read_sides(bm, 3)
        for side = 1, 2 do
            local info = sides[side]
            for _, u in ipairs(info.units) do
                local found = false
                for id, pair in ipairs(state.pairs) do
                    if u:name() == 'bai_arena_' .. id .. '_' .. side then
                        assert(not pair.units[side], 'duplicate arena unit')
                        assert(u:type() == pair.case.kind and u:number_of_men_alive() == pair.case.men,
                            'unexpected arena roster')
                        local p = u:position()
                        emit('spawn_check', nil, {unit_name = u:name(), expected_z = pair.case.z,
                            x = p:get_x(), z = p:get_z()})
                        assert(math.abs(p:get_z() - pair.case.z) < 60, 'spawn displaced: ' .. u:name()
                            .. ' z=' .. p:get_z() .. ' expected=' .. pair.case.z)
                        local item = {unit = u, army = info.army, alliance = info.alliance,
                            side = side, pair = pair, spawn = p}
                        pair.units[side] = item
                        state.units[#state.units + 1] = item
                        found = true
                    end
                end
                assert(found, 'unrecognised arena unit: ' .. u:name())
            end
        end
        -- Take ownership before auto-deployment, not only after Deployed.
        for _, item in ipairs(state.units) do
            item.controller = orders.take_control(item.army, item.unit)
            orders.prepare_melee(item.controller, item.unit)
        end
        local sequence = telemetry.next_sequence('tww3_bai_sequence.txt')
        state.batch = clock.batch_stamp() .. '-arena-' .. sequence
        emit('ready', nil, {automatic = true, map_loads = 1, expected_pairs = 3, shared_army_morale = true})
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(tick))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active then
                tick()
                if not state.finished then
                    for _, pair in ipairs(state.pairs) do finish_pair(pair, 'incomplete', 0, 'global_battle_ended') end
                    finish_arena()
                end
            end
        end))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_arena_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
