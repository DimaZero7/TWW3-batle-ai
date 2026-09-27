-- Entry: live test of every documented unit readout.
-- Scenario: scenarios/unit_readout.xml (Moorlands Route, Empire vs Empire).
-- Script-controlled stages make the readouts change: idle, march, ranged,
-- cease fire, melee, halt. Every tick all units are read with:
--   units.state_adapter (own, 68 fields) and its enemy gate (must withhold),
--   units.range_adapter (every unit pair, visibility-gated),
--   intel (visibility + last seen per side) inside the side view,
--   observation.adapter: the view of ONE side (side_view, both sides) and
--   telemetry.sampler_adapter: the full omniscient summary (full_view),
--   so tools/analysis/unit_readout.py can prove hidden units never leak.
-- plus a static profile (attributes, behaviours, abilities, rank) and
-- telemetry.sampler_adapter frames. tools/analysis/unit_readout.py checks
-- the log against the documented expectations.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local value = require('apps.core.value')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local unit_state = require('apps.units.state_adapter')
local intel_memory = require('apps.intel.services')
local sampler = require('apps.telemetry.sampler_adapter')
local map = require('apps.map.adapter')
local observation = require('apps.observation.adapter')
local observation_services = require('apps.observation.services')
local nav_diagnostics = require('apps.navigation.diagnostics_adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_unit_readout_tick'

-- Stage start times, model seconds after Deployed.
M.STAGES = {
    {name = 'idle', at = 0},
    {name = 'march', at = 8},
    {name = 'ranged', at = 30},
    {name = 'cease_fire', at = 75},
    {name = 'melee', at = 85},
    {name = 'scout_a80', at = 140},
    {name = 'scout_a40', at = 148},
    {name = 'scout_a15', at = 156},
    {name = 'scout_b80', at = 164},
    {name = 'scout_b40', at = 172},
    {name = 'scout_b15', at = 180},
    {name = 'halt', at = 188},
    {name = 'done', at = 196},
}
M.ATTRIBUTES = {'hide_forest', 'stalk', 'charge_defense_vs_large', 'charge_reflection', 'encourages'}
M.BEHAVIOURS = {'defend', 'skirmish', 'fire_at_will', 'change_formation_spacing'}
-- Missile units: range readings are built only for them (5 pairs per side).
local SHOOTERS = {a_archers = true, b_archers = true, a_stalkers = true, b_stalkers = true}
M.CCO_PROFILE = {'CharacterRank', 'HasCharacterRank', 'ExperienceLevel', 'ActiveEffectList.Size'}

-- config: build, speed, tick_ms. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_unit_readout then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', stage = 'setup',
        stage_index = 0, ticks = 0, units = {}, by_name = {}, sides = {}, side_units = {}}
    _G.tww3_bai_unit_readout = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local started_ms = 0

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'unit_readout'
        row.policy = 'readout_probe'
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
        row.stage = row.stage or state.stage
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
    local function item(name) return state.by_name[name] end
    local scout, park

    -- Static profile: what each unit is and can do (docs: game/units/states.md).
    local function profile(it)
        local u, row = it.unit, {side = it.side, name = it.name}
        local function safe(key, fn, kind)
            local r = value.read(fn, kind)
            -- Not `a and b or c`: a known false must stay false.
            if r.status == 'known' then row[key] = r.value else row[key] = 'unknown:' .. r.reason end
        end
        safe('type', function() return u:type() end, 'string')
        safe('initial_men', function() return u:initial_number_of_men() end, 'number')
        safe('commanding', function() return u:is_commanding_unit() end, 'boolean')
        safe('missile_range', function() return u:missile_range() end, 'number')
        for _, key in ipairs(M.ATTRIBUTES) do
            safe('attribute_' .. key, function() return u:has_attribute(key) end, 'boolean')
        end
        for _, key in ipairs(M.BEHAVIOURS) do
            safe('can_' .. key, function() return u:can_use_behaviour(key) end, 'boolean')
        end
        for _, key in ipairs(M.CCO_PROFILE) do
            local ok, v = pcall(cco, u, key)
            if not ok then
                row['cco_' .. key] = 'unknown:read_error'
            elseif v == nil then
                row['cco_' .. key] = 'unknown:nil'
            else
                row['cco_' .. key] = v
            end
        end
        for _, kind in ipairs({'owned_non_passive_special_abilities', 'owned_passive_special_abilities'}) do
            local ok, list = pcall(function() return u[kind](u) end)
            if ok and type(list) == 'table' then
                local keys = {}
                for _, k in ipairs(list) do keys[#keys + 1] = tostring(k) end
                row[kind] = keys
            else
                row[kind] = 'unknown'
            end
        end
        emit('unit_profile', row)
    end

    -- One sample per tick, no duplicates:
    --   side_view x2: own units in full + only permitted enemy data (+ range
    --                 for own shooters); the only data an AI side may use;
    --   full_view:    the omniscient summary (ground truth for the report);
    --   enemy_gate:   state_adapter from the enemy's side, every 5th tick.
    local function read_all()
        local now = bm:time_elapsed_ms()
        for side = 1, 2 do
            local info = state.sides[side]
            local view = observation.observe_side({side = side, alliance = info.alliance,
                own = state.side_units[side], enemies = state.side_units[3 - side],
                memory = info.view_memory, cco = cco, now_ms = now, shooters = SHOOTERS})
            emit('side_view', {observer_side = side, view = view,
                counts = observation_services.visibility_counts(view)})
        end
        emit('full_view', sampler.sample(state.sampler, now - started_ms))
        if state.ticks % 5 == 0 then
            for _, it in ipairs(state.units) do
                local enemy_side = state.sides[3 - it.side]
                emit('enemy_gate', {observer_side = enemy_side.side, name = it.name,
                    readings = unit_state.observe(it.unit, {owned = false,
                        observer_alliance = enemy_side.alliance, cco = cco})})
            end
        end
    end

    local function diagnostics()
        for _, it in ipairs(state.units) do
            emit('nav_state', {name = it.name, readings = nav_diagnostics.state(bm, it.unit),
                origins = nav_diagnostics.origins(bm, it.unit, vec)})
        end
    end

    -- Reveal experiment. Long marches across the fighting field were not
    -- reliable (scouts met and fought mid-way), so a scout is teleported to
    -- a fixed distance from the enemy ambush, facing it: the general to the
    -- forest ambush (approach from the south), the archers to the huntsmen
    -- (approach from the north). One side scouts at a time.
    local SCOUTS = {
        a = {{'a_general', 'b_forest', -1, 5}, {'a_archers', 'b_stalkers', 1, 30}},
        b = {{'b_general', 'a_forest', -1, 5}, {'b_archers', 'a_stalkers', 1, 30}},
    }
    function scout(side, distance)
        for _, s in ipairs(SCOUTS[side]) do
            local scout_item, target = item(s[1]), item(s[2])
            local dz = s[3] * distance            -- -1: south of the target, 1: north
            orders.set_guard(scout_item.uc, scout_item.unit, false)
            orders.teleport(scout_item.uc, vec(target.spawn.x, target.spawn.z + dz),
                dz < 0 and 0 or 180, s[4])
            orders.halt(scout_item.uc)
        end
    end
    -- Send a side's scouts back behind their lines, far from every ambush.
    function park(side)
        for i, s in ipairs(SCOUTS[side]) do
            local scout_item = item(s[1])
            orders.teleport(scout_item.uc, vec(side == 'a' and 330 or -330, -40 + i * 40),
                side == 'a' and 270 or 90, s[4])
            orders.halt(scout_item.uc)
        end
    end

    -- Stage actions: only verified order recipes (orders.adapter).
    local ACTIONS = {
        idle = function() end,
        march = function()
            orders.move(item('a_spears').uc, vec(20, -60), false)
            orders.move(item('b_spears').uc, vec(-20, 60), false)
            orders.move(item('a_general').uc, vec(110, 0), true)
            orders.move(item('b_general').uc, vec(-110, 0), false)
        end,
        ranged = function()
            orders.attack_ranged(item('a_archers').uc, item('b_spears').unit)
            orders.attack_ranged(item('b_archers').uc, item('a_spears').unit)
        end,
        cease_fire = function()
            orders.stop_firing(item('a_archers').uc)
            orders.stop_firing(item('b_archers').uc)
        end,
        melee = function()
            orders.attack_melee(item('a_spears').uc, item('b_spears').unit)
            orders.attack_melee(item('b_spears').uc, item('a_spears').unit)
            orders.set_guard(item('a_general').uc, item('a_general').unit, true)
            orders.set_guard(item('b_general').uc, item('b_general').unit, true)
        end,
        scout_a80 = function() scout('a', 80) end,
        scout_a40 = function() scout('a', 40) end,
        scout_a15 = function() scout('a', 15) end,
        scout_b80 = function() park('a'); scout('b', 80) end,
        scout_b40 = function() scout('b', 40) end,
        scout_b15 = function() scout('b', 15) end,
        halt = function()
            for _, it in ipairs(state.units) do orders.halt(it.uc) end
        end,
    }

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', ticks = state.ticks,
            duration_model_ms = bm:time_elapsed_ms() - started_ms})
        -- Keep script control: released units would fight on forever
        -- (victory countdown is disabled for the test).
        for _, it in ipairs(state.units) do orders.halt(it.uc) end
        bm:end_battle()
    end

    local function tick()
        if not state.active then return end
        -- No damage to anyone for stall_ms of game time: end the battle.
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled')
            flush()
            return
        end
        local elapsed = (bm:time_elapsed_ms() - started_ms) / 1000
        local next_stage = M.STAGES[state.stage_index + 1]
        if next_stage and elapsed >= next_stage.at then
            state.stage_index = state.stage_index + 1
            state.stage = next_stage.name
            if state.stage == 'done' then
                read_all()
                diagnostics()
                finish()
                flush()
                return
            end
            emit('stage', {name = state.stage, elapsed_s = elapsed})
            ACTIONS[state.stage]()
            diagnostics()
        end
        state.ticks = state.ticks + 1
        read_all()
        flush()
    end

    local function start()
        if state.active then return end
        started_ms = bm:time_elapsed_ms()
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end)
        bm:change_victory_countdown_limit(-1)
        -- The AI-controlled army is redeployed by the engine when deployment
        -- ends (measured: ambush units moved ~300 m to the centre), so every
        -- unit goes back to its XML spawn point.
        for _, it in ipairs(state.units) do
            orders.teleport(it.uc, vec(it.spawn.x, it.spawn.z), it.side == 1 and 270 or 90,
                it.unit:initial_number_of_men() > 1 and 30 or 5)
            orders.halt(it.uc)
        end
        emit('start', {speed = config.speed, tick_ms = config.tick_ms, deadline_s = config.deadline_s})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
            flush()
        end), 'tww3_bai_unit_readout_deadline')
        bm:add_infotext('BAI: UNIT READOUT TEST')
        flush()
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
        assert(common and vector_type, 'common and battle_vector globals required')
        local sides = battle.read_sides(bm, 5)
        local registry = {}
        for side = 1, 2 do
            local info = sides[side]
            info.view_memory = intel_memory.new_memory()
            state.side_units[side] = {}
            state.sides[side] = info
            registry[side] = {alliance = info.alliance, units = {}}
            for _, u in ipairs(info.units) do
                local it = {side = side, name = u:name(), unit = u}
                state.units[#state.units + 1] = it
                state.by_name[it.name] = it
                table.insert(state.side_units[side], it)
                registry[side].units[#registry[side].units + 1] = {unit = u, roster = {id = it.name}}
            end
        end
        for _, name in ipairs({'a_general', 'a_spears', 'a_archers', 'a_forest', 'a_stalkers',
            'b_general', 'b_spears', 'b_archers', 'b_forest', 'b_stalkers'}) do
            assert(state.by_name[name], 'scenario unit missing: ' .. name)
        end
        state.sampler = sampler.new(registry, cco)
        state.all_units = {}
        for _, it in ipairs(state.units) do state.all_units[#state.all_units + 1] = it.unit end
        -- Take control before deployment ends, so the engine's AI cannot
        -- redeploy the non-player army, and remember the XML spawn points.
        for _, it in ipairs(state.units) do
            local p = it.unit:position()
            it.spawn = {x = p:get_x(), z = p:get_z()}
            it.uc = orders.take_control(state.sides[it.side].army, it.unit)
            orders.set_fire_at_will(it.uc, false)
            if it.unit:can_use_behaviour('skirmish') then
                it.uc:change_behaviour_active('skirmish', false)
            end
            orders.halt(it.uc)
        end
        state.batch = clock.batch_stamp() .. '-readout-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {units = #state.units})
        for _, it in ipairs(state.units) do profile(it) end
        read_all()
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            emit('deployment')
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_unit_readout_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
