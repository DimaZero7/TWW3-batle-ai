-- Entry: in-battle check of the formation planner (apps.formation).
-- Scenario and config: tools/sim/formation.py (build target formation-probe).
-- After deployment:
--   1. the enemy is teleported to its simulated plan and held;
--   2. our army is planned in battle by apps.plan (assessment -> strategy ->
--      formation) from both rosters, facing the visible enemy (or the
--      configured enemy anchor), and teleported there;
--   3. stages: placed -> align -> hold. align: apps.vision (groups of both
--      sides from our side's view) -> apps.battlefield -> apps.alignment; if
--      our army is clearly off (config.own.start_bearing / anchor can start it
--      crooked), the formation is planned again opposite the enemy and the
--      units WALK there by orders; then it is checked again on real soldiers.
--      Every alignment goes through the governor (apps.alignment: tolerance,
--      not while moving, every 20 s at most, give up if it does not help).
--      hold: config.hold_s of game time, no orders at all (no rushing); every
--      5 s the governor is asked on real soldiers and its answer is only
--      logged ('governor'): it must not want to realign again and again;
--      with config.turn_test also: archers turn right/left, engine rotate and ours.
-- stage_snapshot: every own unit's movement and soldiers when a stage ends;
-- hold_sample: every own unit's movement every tick of the hold;
-- turn_sample: archers' soldiers every tick while they turn.
-- tools/analysis/formation_probe.py compares the result with the plan.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local map = require('apps.map.adapter')
local formation = require('apps.formation.services')
local plan_services = require('apps.plan.services')
local unit_motion = require('apps.units.formation_adapter')
local assessment = require('apps.assessment.services')
local vision = require('apps.vision.services')
local battlefield = require('apps.battlefield.services')
local alignment = require('apps.alignment.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_formation_probe_tick'
M.LORD_WIDTH = 5
-- stage name, turn of the archers (degrees), how: 'rotate' = the engine's
-- relative rotate (pivots on the front rank), 'in_place' = our turn around the middle.
M.BASE_STAGES = {{'placed', 0}, {'align', 0, 'align'}, {'hold', 0, 'hold'}}
M.TURN_STAGES = {{'turn_right', 90, 'rotate'}, {'back_from_right', -90, 'rotate'},
    {'turn_left', -90, 'rotate'}, {'back_from_left', 90, 'rotate'},
    {'turn_right_in_place', 90, 'in_place'}, {'back_right_in_place', -90, 'in_place'},
    {'turn_left_in_place', -90, 'in_place'}, {'back_left_in_place', 90, 'in_place'}}

-- config: build, speed, tick_ms, deadline_s, stall_ms, settle_ms, stage_timeout_s, hold_s, turn_test,
-- align_timeout_s, own.start_bearing (optional crooked start),
-- role ('attack' | 'defend'), own = {anchor, units (roster input, id = script name)},
-- enemy = {anchor, units (roster input), placements}, formation_params (overrides, tests only).
function M.main(bm, config, globals)
    if _G.tww3_bai_formation_probe then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        own = {}, enemy = {}, by_name = {}, stage_index = 0, orders_after_placed = 0, orders_alignment = 0,
        stages = {}, governor = alignment.new_governor()}
    for _, s in ipairs(M.BASE_STAGES) do state.stages[#state.stages + 1] = s end
    if config.turn_test then
        for _, s in ipairs(M.TURN_STAGES) do state.stages[#state.stages + 1] = s end
    end
    _G.tww3_bai_formation_probe = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'formation_probe'
        row.policy = 'formation_probe'
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

    local function units_row(filter)
        local rows = {}
        for _, it in ipairs(state.own) do
            if not filter or filter(it) then
                local men = unit_motion.soldiers(cco, it.unit)
                rows[#rows + 1] = {script_name = it.name, role = it.role, motion = unit_motion.motion(it.unit),
                    soldiers_dm = men.xz_dm, soldiers_unavailable = men.reason}
            end
        end
        return rows
    end
    local function is_archer(it) return it.role == 'arc' end

    local function strength_of(id, list)
        for _, u in ipairs(list) do
            if u.id == id then return assessment.strength(u) end
        end
        return 1
    end

    local function points_of(u)
        local men = unit_motion.soldiers(cco, u)
        if men.status ~= 'ok' then return nil end
        local pts = {}
        for i, v in ipairs(men.xz_dm) do pts[i] = v / 10 end
        return pts
    end

    -- The AI's own view: our units in full, the enemy only as our side sees it.
    local function battle_picture()
        local own, enemy, total = {}, {}, 0
        for _, it in ipairs(state.own) do
            local pts = points_of(it.unit)
            if pts then
                own[#own + 1] = {id = it.name, points = pts, strength = strength_of(it.name, config.own.units),
                    bearing = it.unit:bearing()}
            end
        end
        for _, it in ipairs(state.enemy) do
            local s = strength_of(it.name, config.enemy.units)
            total = total + s
            local ok, seen = pcall(function() return it.unit:is_visible_to_alliance(state.alliance) end)
            if ok and seen then
                local pts = points_of(it.unit)
                if pts then enemy[#enemy + 1] = {id = it.name, points = pts, strength = s, bearing = it.unit:bearing()} end
            end
        end
        local picture = vision.picture({own = own, enemy = enemy, enemy_total = total})
        if not (picture.own.main and picture.enemy.main) then return picture, nil end
        local function group_points(units, group)
            local ids, pts = {}, {}
            for _, id in ipairs(group.ids) do ids[id] = true end
            for _, u in ipairs(units) do
                if ids[u.id] then for _, v in ipairs(u.points) do pts[#pts + 1] = v end end
            end
            return pts
        end
        local field = battlefield.frame({
            own = {points = group_points(own, picture.own.main), centre = picture.own.main.centre},
            enemy = {points = group_points(enemy, picture.enemy.main), centre = picture.enemy.main.centre}})
        return picture, field
    end

    local function enemy_of(picture)
        local main = picture.enemy.main
        return {centre = main.centre, facing = main.facing}
    end

    -- Align stage: decide from our side's view and walk there if needed. While
    -- the enemy is not seen there is nothing to align with: try again next tick
    -- (finding an unseen enemy is backlog #11). Returns true once decided.
    local function align()
        local picture, field = battle_picture()
        if not field then
            state.align_waits = (state.align_waits or 0) + 1
            return false
        end
        local current = {anchor = state.placed.anchor, bearing = state.placed.bearing}
        local check = alignment.check(field, current, enemy_of(picture))
        local now = bm:time_elapsed_ms()
        local decision = state.governor.decide(now, check, false)
        local row = {check = check, overhang = alignment.overhang(field), field = field, decision = decision}
        if decision == 'align' then
            state.governor.record_order(now, check)
            local target = alignment.target(field, current, enemy_of(picture))
            local result = plan_services.start({role = config.role, bearing = target.bearing,
                own = {anchor = target.anchor, units = config.own.units},
                enemy = {units = config.enemy.units, lord = state.placed.lord}, formation_params = config.formation_params})
            assert(result.formation, 'No strategy when aligning')
            for _, p in ipairs(result.formation.placements) do
                local it = state.by_name[p.id]
                it.placement = p
                orders.move_formation(it.uc, vec(p.x, p.z), p.bearing, p.width or M.LORD_WIDTH, false)
                state.orders_alignment = state.orders_alignment + 1
            end
            row.target, row.plan = target, result.formation
        end
        row.waited_ticks = state.align_waits or 0
        emit('alignment', row)
        return true
    end

    -- The same check on the real soldiers (our group's centre and facing).
    local function measured_check()
        local picture, field = battle_picture()
        if not field then return nil end
        local own = picture.own.main
        local current = {anchor = own.centre, bearing = own.facing or state.placed.bearing}
        return alignment.check(field, current, enemy_of(picture)), field
    end

    local function check_aligned()
        local check, field = measured_check()
        if not check then return end
        emit('alignment_after', {check = check, overhang = alignment.overhang(field), field = field,
            orders = state.orders_alignment})
    end

    -- During the hold: what the governor would do now (logged only, no orders).
    local function ask_governor(elapsed)
        local check = measured_check()
        if not check then return end
        local moving = false
        for _, it in ipairs(state.own) do
            if it.unit:is_moving() then moving = true end
        end
        emit('governor', {t_ms = elapsed, decision = state.governor.decide(bm:time_elapsed_ms(), check, moving),
            check = check})
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status, stages_done = state.stage_index, stages = #state.stages, ticks = state.ticks,
            orders_after_placed = state.orders_after_placed, orders_alignment = state.orders_alignment})
        for _, it in ipairs(state.own) do orders.halt(it.uc) end
        for _, it in ipairs(state.enemy) do orders.halt(it.uc) end
        flush()
        bm:end_battle()
    end

    local function begin_stage(index)
        state.stage_index, state.stage_ms, state.still = index, bm:time_elapsed_ms(), 0
        local stage = state.stages[index]
        if stage[3] == 'align' then state.align_done = align() end
        if stage[2] ~= 0 then
            for _, it in ipairs(state.own) do
                if is_archer(it) then state.orders_after_placed = state.orders_after_placed + 1 end
                if is_archer(it) and stage[3] == 'rotate' then
                    orders.rotate(it.uc, stage[2], false)
                elseif is_archer(it) then
                    local p = it.unit:position()
                    local t = formation.turn_in_place({x = p:get_x(), z = p:get_z()}, it.unit:bearing(), stage[2],
                        it.placement.depth_m)
                    orders.move_formation(it.uc, vec(t.x, t.z), t.bearing, it.placement.width, false)
                end
            end
        end
        emit('stage', {stage = stage[1], turn_deg = stage[2], how = stage[3]})
    end

    local function tick()
        if not state.active then return end
        state.ticks = state.ticks + 1
        local now = bm:time_elapsed_ms()
        if state.stall.update(now, battle.health_signature(state.all_units)) then
            finish('stalled')
            return
        end
        local stage = state.stages[state.stage_index]
        local elapsed = now - state.stage_ms
        if stage[2] ~= 0 then
            emit('turn_sample', {stage = stage[1], t_ms = elapsed, units = units_row(is_archer)})
        end
        if stage[3] == 'align' and not state.align_done then
            state.align_done = align()
            state.still = 0  -- orders may have just been given
        end
        if stage[3] == 'hold' then
            local rows = {}
            for _, it in ipairs(state.own) do
                rows[#rows + 1] = {script_name = it.name, motion = unit_motion.motion(it.unit)}
            end
            emit('hold_sample', {t_ms = elapsed, orders_after_placed = state.orders_after_placed, units = rows})
            if state.ticks % 5 == 0 then ask_governor(elapsed) end
        end
        local moving = false
        for _, it in ipairs(state.own) do
            if it.unit:is_moving() then moving = true end
        end
        state.still = moving and 0 or state.still + 1
        local done
        if stage[3] == 'hold' then
            done = elapsed >= config.hold_s * 1000
        elseif stage[3] == 'align' then
            -- Walking takes a while; the first ticks may still be standing.
            done = (elapsed >= config.settle_ms and state.still >= 3) or elapsed >= config.align_timeout_s * 1000
        else
            done = (elapsed >= config.settle_ms and state.still >= 2) or elapsed >= config.stage_timeout_s * 1000
        end
        if stage[3] == 'align' and not state.align_done then
            done = elapsed >= config.align_timeout_s * 1000
        end
        if done then
            if stage[3] == 'align' and not state.align_done then
                emit('alignment', {decision = 'enemy_not_seen', waited_ticks = state.align_waits or 0})
            end
            if stage[3] == 'align' then check_aligned() end
            emit('stage_snapshot', {stage = stage[1], t_ms = elapsed, settled = state.still >= 2,
                units = units_row()})
            if state.stage_index < #state.stages then begin_stage(state.stage_index + 1) else finish('completed') end
        end
        flush()
    end

    -- Where the enemy is, from our side's view only.
    local function enemy_view()
        local sx, sz, n, lord = 0, 0, 0, nil
        for _, it in ipairs(state.enemy) do
            local ok, seen = pcall(function() return it.unit:is_visible_to_alliance(state.alliance) end)
            if ok and seen then
                local p = it.unit:position()
                sx, sz, n = sx + p:get_x(), sz + p:get_z(), n + 1
                if it.unit:is_commanding_unit() and it.unit:initial_number_of_men() == 1 then
                    lord = {x = p:get_x(), z = p:get_z()}
                end
            end
        end
        if n == 0 then return config.enemy.anchor, nil, 'configured_anchor', 0 end
        return {x = sx / n, z = sz / n}, lord, 'visible_enemy', n
    end

    local function place_own()
        local centre, lord, source, seen = enemy_view()
        local bearing = config.own.start_bearing or formation.facing(config.own.anchor, centre)
        state.placed = {anchor = config.own.anchor, bearing = bearing, lord = lord}
        -- The enemy roster (unit types) is known before the battle, as to the player.
        local result = plan_services.start({role = config.role, bearing = bearing,
            own = {anchor = config.own.anchor, units = config.own.units},
            enemy = {units = config.enemy.units, lord = lord}, formation_params = config.formation_params})
        emit('plan', {facing_source = source, enemies_seen = seen, enemy_centre = centre, bearing = bearing,
            status = result.status, strategy = result.decision.strategy, features = result.features,
            candidates = result.decision.candidates, plan = result.formation})
        assert(result.formation, 'No strategy for this battle: ' .. tostring(result.status))
        local plan = result.formation
        plan.overlaps = formation.overlaps(plan.placements)
        for _, p in ipairs(plan.placements) do
            local it = state.by_name[p.id]
            it.role, it.placement = p.role, p
            orders.teleport(it.uc, vec(p.x, p.z), p.bearing, p.width or M.LORD_WIDTH)
            orders.halt(it.uc)
        end
    end

    local function start()
        if state.active or state.finished then return end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_formation_probe_speed')
        bm:change_victory_countdown_limit(-1)
        for _, p in ipairs(config.enemy.placements) do
            local it = state.by_name[p.script_name]
            orders.teleport(it.uc, vec(p.x, p.z), p.bearing, p.width)
            orders.halt(it.uc)
        end
        emit('start', {speed = config.speed, deadline_s = config.deadline_s, stall_ms = config.stall_ms})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_formation_probe_deadline')
        bm:add_infotext('BAI: FORMATION PROBE')
        -- The enemy teleport applies on a later tick; plan once it has.
        bm:real_callback(guarded(function()
            place_own()
            state.active = true
            begin_stage(1)
            flush()
            bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        end), 500, 'tww3_bai_formation_probe_place')
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
        state.alliance = sides[1].alliance
        state.all_units = {}
        local function adopt(side, list, name)
            local u = battle.find_by_name(sides[side], name)
            assert(u, 'scenario unit missing: ' .. name)
            local it = {name = name, unit = u, uc = orders.take_control(sides[side].army, u)}
            orders.set_fire_at_will(it.uc, false)
            orders.halt(it.uc)
            list[#list + 1] = it
            state.by_name[name] = it
            state.all_units[#state.all_units + 1] = u
        end
        for _, u in ipairs(config.own.units) do adopt(1, state.own, u.id) end
        for _, p in ipairs(config.enemy.placements) do adopt(2, state.enemy, p.script_name) end
        state.batch = clock.batch_stamp() .. '-formation-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {own = #state.own, enemy = #state.enemy})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_formation_probe_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
