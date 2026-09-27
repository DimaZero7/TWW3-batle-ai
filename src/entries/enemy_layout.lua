-- Entry: how the game AI deploys and stands (research, full view).
-- Scenario and config: tools/enemy_layout.py (config/armies/defender_layouts.json).
-- Our army (side 1) is held by script; the enemy army (side 2) is left to
-- the game AI, which redeploys it when deployment ends. For hold_s of game
-- time every tick logs every enemy unit: movement, and whether our side sees it;
-- snapshots at deployment, after it and at the end add every soldier.
-- This is research data for choosing how to find the enemy army; the AI
-- itself will only use what our side sees.
-- Every config.picture_every ticks the AI modules run on real data exactly as
-- the AI will: apps.vision (groups of both sides; the enemy only as our side
-- sees it) and apps.battlefield between the two main groups -> 'battlefield'.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local unit_motion = require('apps.units.formation_adapter')
local assessment = require('apps.assessment.services')
local vision = require('apps.vision.services')
local battlefield = require('apps.battlefield.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_enemy_layout_tick'

-- config: build, speed, tick_ms, deadline_s, stall_ms, hold_s, layout, picture_every,
-- roster ({unit key = roster unit input}: strength of a unit by its type),
-- enemy_mode: 'native' (the game AI as the battle sets it; in our XML battles
-- both sides count as attackers, measured 27.09.2026) or 'defend' (the game AI
-- through CA's script_ai_planner, told to defend where it deployed).
M.DEFEND_RADIUS_M = 80

-- Function names of a table and of its metatable __index (engine objects).
local function api(object)
    local names = {}
    local function add(t)
        if type(t) ~= 'table' then return end
        for k, v in pairs(t) do
            if type(v) == 'function' then names[#names + 1] = tostring(k) end
        end
    end
    pcall(add, object)
    pcall(function() add(getmetatable(object).__index) end)
    table.sort(names)
    return names
end
function M.main(bm, config, globals)
    if _G.tww3_bai_enemy_layout then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        own = {}, enemy = {}}
    _G.tww3_bai_enemy_layout = state
    local common = globals and globals.common

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'enemy_layout'
        row.policy = 'game_ai'
        row.layout = config.layout
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

    local function seen_by_us(u)
        local ok, seen = pcall(function() return u:is_visible_to_alliance(state.our_alliance) end)
        if ok and type(seen) == 'boolean' then return seen end
        return 'unknown'
    end

    local function enemy_rows(with_soldiers)
        local rows = {}
        for _, it in ipairs(state.enemy) do
            local row = {name = it.name, key = it.key, motion = unit_motion.motion(it.unit), seen = seen_by_us(it.unit)}
            if with_soldiers then
                local men = unit_motion.soldiers(cco, it.unit)
                row.soldiers_dm, row.soldiers_unavailable = men.xz_dm, men.reason
            end
            rows[#rows + 1] = row
        end
        return rows
    end

    local function strength(u)
        local spec = config.roster and config.roster[u:type()]
        if spec then return assessment.strength(spec) end
        return u:initial_number_of_men()
    end

    local function points_of(u)
        local men = unit_motion.soldiers(cco, u)
        if men.status ~= 'ok' then return nil end
        local pts = {}
        for i, v in ipairs(men.xz_dm) do pts[i] = v / 10 end
        return pts
    end

    -- The AI's own view: our units in full, the enemy only if our side sees it.
    local function battle_picture()
        local started = os.clock and os.clock() or 0
        local own, enemy, total = {}, {}, 0
        for _, it in ipairs(state.own) do
            local pts = points_of(it.unit)
            if pts then own[#own + 1] = {id = it.name, points = pts, strength = strength(it.unit)} end
        end
        for _, it in ipairs(state.enemy) do
            local s = strength(it.unit)
            total = total + s  -- the enemy roster is known before the battle
            if seen_by_us(it.unit) == true then
                local pts = points_of(it.unit)
                if pts then enemy[#enemy + 1] = {id = it.name, points = pts, strength = s} end
            end
        end
        local picture = vision.picture({own = own, enemy = enemy, enemy_total = total})
        local field
        if picture.own.main and picture.enemy.main then
            local function group_points(units, group)
                local ids, pts = {}, {}
                for _, id in ipairs(group.ids) do ids[id] = true end
                for _, u in ipairs(units) do
                    if ids[u.id] then for _, v in ipairs(u.points) do pts[#pts + 1] = v end end
                end
                return pts
            end
            field = battlefield.frame({own = {points = group_points(own, picture.own.main), centre = picture.own.main.centre},
                enemy = {points = group_points(enemy, picture.enemy.main), centre = picture.enemy.main.centre}})
        end
        local function brief(side)
            local groups = {}
            for i, g in ipairs(side.groups) do
                groups[i] = {ids = g.ids, strength = g.strength, centre = g.centre, radius_m = g.radius_m}
            end
            return {groups = groups, seen_share = side.seen_share}
        end
        emit('battlefield', {t_ms = bm:time_elapsed_ms() - state.start_ms, own = brief(picture.own),
            enemy = brief(picture.enemy), field = field, seen = #enemy, enemy_units = #state.enemy,
            clock_s = os.clock and (os.clock() - started) or 'unavailable'})
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('enemy_snapshot', {stage = 'end', units = enemy_rows(true)})
        local own = {}
        for _, it in ipairs(state.own) do
            local men = unit_motion.soldiers(cco, it.unit)
            own[#own + 1] = {name = it.name, soldiers_dm = men.xz_dm}
        end
        emit('own_snapshot', {stage = 'end', units = own})
        emit('result', {status = status, ticks = state.ticks})
        for _, it in ipairs(state.own) do orders.halt(it.uc) end
        flush()
        bm:end_battle()
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        local now = bm:time_elapsed_ms()
        if state.stall.update(now, battle.health_signature(state.all_units)) then
            finish('stalled')
            return
        end
        emit('enemy_sample', {t_ms = now - state.start_ms, units = enemy_rows(false)})
        if config.picture_every and state.ticks % config.picture_every == 0 then battle_picture() end
        if now - state.start_ms >= config.hold_s * 1000 then finish('completed') return end
        flush()
    end

    -- The game AI keeps the army; we only give it the objective "defend here".
    local function defend(sides)
        local sx, sz = 0, 0
        for _, it in ipairs(state.enemy) do
            local p = it.unit:position()
            sx, sz = sx + p:get_x(), sz + p:get_z()
        end
        local centre = sides[2].units[1]:position()
        centre:set_x(sx / #state.enemy)
        centre:set_z(sz / #state.enemy)
        local report = {script_ai_planner = script_ai_planner and api(script_ai_planner) or 'not_loaded'}
        if script_ai_planner and script_ai_planner.defend_position and bm.get_scriptunit_for_unit then
            local own = {}
            for _, it in ipairs(state.enemy) do own[#own + 1] = bm:get_scriptunit_for_unit(it.unit) end
            state.planner = script_ai_planner:new('tww3_bai_enemy_defend', own)
            state.planner:defend_position(centre, M.DEFEND_RADIUS_M)
            report.used = 'script_ai_planner.defend_position'
        else
            local planner = sides[2].alliance:create_ai_unit_planner()
            report.engine_planner = api(planner)
            if planner.defend_position then
                for _, it in ipairs(state.enemy) do planner:add_units(it.unit) end
                planner:defend_position(centre, M.DEFEND_RADIUS_M)
                state.planner = planner
                report.used = 'engine_planner.defend_position'
            else
                report.used = 'none'
            end
        end
        report.centre = {x = centre:get_x(), z = centre:get_z()}
        report.radius_m = M.DEFEND_RADIUS_M
        emit('enemy_mode', report)
    end

    local function start()
        if state.active or state.finished then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        state.start_ms = bm:time_elapsed_ms()
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_enemy_layout_speed')
        bm:change_victory_countdown_limit(-1)
        emit('start', {speed = config.speed, hold_s = config.hold_s, deadline_s = config.deadline_s})
        emit('enemy_snapshot', {stage = 'deployed', units = enemy_rows(true)})
        if config.enemy_mode == 'defend' then defend(state.sides) end
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_enemy_layout_deadline')
        bm:add_infotext('BAI: ENEMY LAYOUT')
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
        assert(common, 'common global required')
        local sides = battle.read_sides(bm)
        state.sides = sides
        state.our_alliance = sides[1].alliance
        -- Which side the engine thinks attacks (recorded even if it conflicts).
        local roles, evidence = nil, nil
        pcall(function() roles = battle.read_roles(bm, function(e) evidence = e end) end)
        state.all_units = {}
        for _, u in ipairs(sides[1].units) do
            local it = {name = u:name(), unit = u, uc = orders.take_control(sides[1].army, u)}
            orders.set_fire_at_will(it.uc, false)
            orders.halt(it.uc)
            state.own[#state.own + 1] = it
            state.all_units[#state.all_units + 1] = u
        end
        -- The enemy stays under the game AI: no controller, no orders.
        for _, u in ipairs(sides[2].units) do
            state.enemy[#state.enemy + 1] = {name = u:name(), key = u:type(), unit = u}
            state.all_units[#state.all_units + 1] = u
        end
        state.batch = clock.batch_stamp() .. '-layout-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {own = #state.own, enemy = #state.enemy, roles = roles, role_evidence = evidence})
        emit('enemy_snapshot', {stage = 'deployment', units = enemy_rows(true)})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_enemy_layout_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
