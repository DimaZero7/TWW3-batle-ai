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
--      mask: apps.mask over the battlefield from the engine (is_area_clear +
--      can_reach_position of our infantry), mask_batch cells per tick -> 'mask';
--      approach (config.approach): apps.approach commander in battle — one
--      manoeuvre at a time, alignment first, 50 m steps, a place past an
--      obstacle (the engine walks around it); 'approach_decision' for every
--      decision, 'approach_sample' every 2 ticks while units move,
--      'approach_manoeuvre' with every soldier when a manoeuvre ends.
--      config.goal: march to a point (the enemy army is not considered);
--      config.logistics: a step that walks round an obstacle goes by
--      apps.logistics — who goes which side, in which order and width, and
--      when each unit sets off (a queue); 'logistics_plan' once, then
--      'logistics_order' for every order it gives; the manoeuvre ends when
--      every unit has had its last order and all stand. approach_sample then
--      also carries crowding (soldiers of two units closer than 1 m) and,
--      every 6th tick, every soldier;
--      hold: config.hold_s of game time, no orders at all (no rushing); every
--      5 s the governor is asked on real soldiers and its answer is only
--      logged ('governor'): it must not want to realign again and again;
--      with config.turn_test also: archers turn right/left, engine rotate and ours.
--   config.handover (the player's test): only 'placed', then 'manual' — our AI
--      places the army and releases it to the player at the player's speed;
--      the script only records ('manual_sample' every tick, soldiers every 2nd;
--      'player_order' and 'order_end' for every order the player gives);
--      config.enemy_ai hands the enemy to the game's AI: 'native' as the
--      battle sets it (it attacks), 'defend' — told to defend where it stands.
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
local mask_services = require('apps.mask.services')
local mask_adapter = require('apps.mask.adapter')
local approach = require('apps.approach.services')
local reach = require('apps.reach.services')
local navigation = require('apps.navigation.services')
local logistics = require('apps.logistics.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_formation_probe_tick'
M.LORD_WIDTH = 5
M.DEFEND_RADIUS_M = 80  -- the enemy's defence area in the player's test (as entries.enemy_layout)
-- stage name, turn of the archers (degrees), how: 'rotate' = the engine's
-- relative rotate (pivots on the front rank), 'in_place' = our turn around the middle.
M.BASE_STAGES = {{'placed', 0}, {'align', 0, 'align'}, {'mask', 0, 'mask'}, {'hold', 0, 'hold'}}
M.MASK_BATCH = 500
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
        stages = {}, governor = alignment.new_governor(), commander = approach.new_commander(), holder = {}}
    if config.facing_sweep then
        -- Research: which facing the engine gives for a commanded bearing.
        state.stages = {{'placed', 0}, {'sweep', 0, 'sweep'}}
    elseif config.handover then
        -- The player's test: our AI places the army, then hands it over.
        state.stages = {{'placed', 0}, {'manual', 0, 'manual'}}
    else
        for _, s in ipairs(M.BASE_STAGES) do
            if s[3] == 'hold' and config.approach then state.stages[#state.stages + 1] = {'approach', 0, 'approach'} end
            state.stages[#state.stages + 1] = s
        end
    end
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
        -- A unit with no soldiers left (destroyed, or gone off the field) is not seen.
        if #pts == 0 then return nil end
        return pts
    end

    local function roster_of(id)
        for _, u in ipairs(config.own.units) do
            if u.id == id then return u end
        end
        return {}
    end

    -- Blocks for apps.reach where the units stand now: the middle of the
    -- soldiers (unit:position), the unit's facing, front and depth from our
    -- plan (ours) or from the roster shape nearest to its ordered width
    -- (visible enemies only).
    local function nearest_shape(spec, width)
        local best
        for _, s in ipairs(spec.shapes or {}) do
            if not best or math.abs(s.ordered_m - width) < math.abs(best.ordered_m - width) then best = s end
        end
        return best
    end
    local function reach_blocks()
        local own, enemy = {}, {}
        local function add(list, it, bearing, front, depth, range_m)
            local p = it.unit:position()
            local b = math.rad(bearing)
            list[#list + 1] = {id = it.name, x = p:get_x() + math.sin(b) * depth / 2,
                z = p:get_z() + math.cos(b) * depth / 2, bearing = bearing, front_m = front, depth_m = depth,
                range_m = range_m or 0}
        end
        for _, it in ipairs(state.own) do
            local p = it.placement
            if p then
                local ok, b = pcall(function() return it.unit:bearing() end)
                add(own, it, ok and b or p.bearing, p.front_m, p.depth_m, roster_of(it.name).range_m)
            end
        end
        for _, it in ipairs(state.enemy) do
            local spec
            for _, u in ipairs(config.enemy.units) do if u.id == it.name then spec = u end end
            local ok, seen = pcall(function() return it.unit:is_visible_to_alliance(state.alliance) end)
            if spec and ok and seen then
                local okb, b = pcall(function() return it.unit:bearing() end)
                local okw, w = pcall(function() return it.unit:ordered_width() end)
                local s = nearest_shape(spec, okw and w or 30) or {front_m = 1, depth_m = 1}
                add(enemy, it, okb and b or 0, s.front_m, s.depth_m, spec.range_m)
            end
        end
        return own, enemy
    end

    -- The enemy blocks we see: the formation keeps its window against them.
    local function enemy_blocks()
        local _, enemy = reach_blocks()
        return enemy
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
        if config.goal then
            -- Marching to a point: it stands in for the enemy's main group.
            enemy[1] = {id = 'goal', points = {config.goal.x, config.goal.z}, strength = 1}
            total = 1
        end
        for _, it in ipairs(config.goal and {} or state.enemy) do
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
            local result = plan_services.start({role = config.role, roles = config.roles, bearing = target.bearing,
                own = {anchor = target.anchor, units = config.own.units},
                enemy = {units = config.enemy.units, lord = state.placed.lord, blocks = enemy_blocks()}, formation_params = config.formation_params})
            assert(result.formation, 'No strategy when aligning')
            for _, p in ipairs(result.formation.placements) do
                local it = state.by_name[p.id]
                it.placement = p
                orders.move_formation(it.uc, vec(p.x, p.z), p.bearing, p.width or M.LORD_WIDTH, false)
                state.orders_alignment = state.orders_alignment + 1
            end
            row.target, row.plan = target, result.formation
            state.placed.anchor, state.placed.bearing = target.anchor, target.bearing
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

    -- Mask stage: read the next batch of cells from the engine; true when complete.
    local function mask_step()
        if not state.mask then
            local _, field = battle_picture()
            if not field then return false end
            local reach_unit
            for _, it in ipairs(state.own) do
                if it.role == 'wall' then reach_unit = it.unit; break end
            end
            state.mask = {mask = mask_services.new(mask_services.grid(field)), next = 1, field = field,
                reader = mask_adapter.reader(bm, vector_type, reach_unit or state.own[1].unit, mask_services.DEFAULTS.step_m),
                clock = 0}
        end
        local m = state.mask
        local started = os.clock and os.clock() or 0
        local done
        m.next, done = mask_services.fill(m.mask, m.reader, m.next, config.mask_batch or M.MASK_BATCH)
        m.clock = m.clock + ((os.clock and os.clock() or 0) - started)
        if done then
            local fits = {}
            for _, it in ipairs(state.own) do
                if it.placement then
                    local r = mask_services.fits(m.mask, it.placement)
                    fits[#fits + 1] = {id = it.name, ok = r.ok, blocked = r.blocked, unknown = r.unknown}
                end
            end
            emit('mask', {grid = m.mask.grid, summary = mask_services.summary(m.mask), code = mask_services.encode(m.mask),
                fits = fits, lane = mask_services.lane(m.mask, 0, m.field.own.width_m, m.field.own.front_m,
                    m.field.enemy.front_m), clock_s = m.clock})
        end
        return done
    end

    local function any_moving()
        for _, it in ipairs(state.own) do
            if it.unit:is_moving() then return true end
        end
        return false
    end

    -- Give the whole formation its new place (planned again at anchor/bearing).
    local function order_formation(anchor, bearing)
        local result = plan_services.start({role = config.role, roles = config.roles, bearing = bearing,
            own = {anchor = anchor, units = config.own.units},
            enemy = {units = config.enemy.units, lord = state.placed.lord, blocks = enemy_blocks()}, formation_params = config.formation_params})
        assert(result.formation, 'No strategy when moving')
        for _, p in ipairs(result.formation.placements) do
            -- A place that changed hands in a queue stays with its new holder.
            local it = state.by_name[state.holder[p.id] or p.id]
            it.placement = p
            orders.move_formation(it.uc, vec(p.x, p.z), p.bearing, p.width or M.LORD_WIDTH, false)
            state.orders_approach = (state.orders_approach or 0) + 1
        end
        state.placed.anchor, state.placed.bearing = anchor, bearing
        return result.formation
    end

    -- The logistics dispatcher's orders for this tick (apps.logistics).
    local function logistics_tick(now)
        local lg = state.logistics
        local positions = {}
        for _, it in ipairs(state.own) do
            local p = it.unit:position()
            local f = battlefield.to_frame(lg.field, {x = p:get_x(), z = p:get_z()})
            f.still = not it.unit:is_moving()
            positions[it.name] = f
        end
        for _, o in ipairs(state.dispatch.update(now / 1000, positions)) do
            local it = state.by_name[o.id]
            local row = lg.plan.units[o.id]
            if o.final then
                local q = lg.after_by_id[row.place]
                orders.move_formation(it.uc, vec(q.x, q.z), q.bearing, q.width or M.LORD_WIDTH, false)
            else
                local w = battlefield.to_world(lg.field, o.along, o.across)
                orders.move_formation(it.uc, vec(w.x, w.z), lg.field.bearing + (o.heading_deg or 0),
                    o.width or M.LORD_WIDTH, false)
            end
            state.orders_approach = (state.orders_approach or 0) + 1
            lg.orders = lg.orders + 1
            emit('logistics_order', {id = o.id, leg = o.leg, final = o.final, t_ms = now - lg.started_ms,
                along = o.along, across = o.across, width = o.width})
        end
    end

    -- A step round an obstacle by apps.logistics: plan the queue and start it.
    local function start_logistics(field, m, anchor, bearing, now)
        local before, shapes = {}, {}
        for _, it in ipairs(state.own) do
            local q = it.placement
            if q then
                before[#before + 1] = {id = it.name, x = q.x, z = q.z, bearing = q.bearing, front_m = q.front_m,
                    depth_m = q.depth_m, width = q.width, role = q.role}
            end
        end
        for _, u in ipairs(config.own.units) do shapes[u.id] = u.shapes end
        local result = plan_services.start({role = config.role, roles = config.roles, bearing = bearing,
            own = {anchor = anchor, units = config.own.units},
            enemy = {units = config.enemy.units, lord = state.placed.lord, blocks = enemy_blocks()}, formation_params = config.formation_params})
        assert(result.formation, 'No strategy when moving')
        local after, after_by_id = result.formation.placements, {}
        for _, q in ipairs(after) do after_by_id[q.id] = q end
        local units = logistics.units_from(field, before, after, shapes)
        local e = logistics.extent(units)
        local band = logistics.band(m, e.from, e.to, config.logistics_params, e.left, e.right)
        local started = os.clock and os.clock() or 0
        local plan = logistics.plan(units, band, config.logistics_params)
        local clock_s = (os.clock and os.clock() or 0) - started
        local depths = {}
        for _, u in ipairs(units) do depths[u.id] = u.depth_m end
        state.dispatch = logistics.new_dispatch(plan, depths, config.logistics_params)
        state.logistics = {plan = plan, field = field, after_by_id = after_by_id, started_ms = now, orders = 0}
        -- Places of identical units may have changed hands (and stay so).
        local holder = {}
        for id, row in pairs(plan.units) do
            state.by_name[id].placement = after_by_id[row.place]
            holder[row.place] = id
        end
        state.holder = holder
        state.placed.anchor, state.placed.bearing = anchor, bearing
        local rows = {}
        for id, row in pairs(plan.units) do
            rows[#rows + 1] = {id = id, kind = row.kind, side = row.side, slot = row.slot, place = row.place,
                release = row.release, route = row.route, finish_s = row.finish_s, enter_s = row.enter_s}
        end
        emit('logistics_plan', {band = band, makespan_s = plan.makespan_s,
            splits_tried = plan.splits_tried, split = plan.split, swapped = plan.swapped == true, clock_s = clock_s,
            units = rows, order = plan.order, before = before, after = after, field = field})
        logistics_tick(now)
    end

    local function crowd_now(with_soldiers)
        local list, rows = {}, {}
        for _, it in ipairs(state.own) do
            local men = unit_motion.soldiers(cco, it.unit)
            if men.status == 'ok' then
                local pts = {}
                for i, v in ipairs(men.xz_dm) do pts[i] = v / 10 end
                list[#list + 1] = {id = it.name, points = pts}
                if with_soldiers then rows[it.name] = men.xz_dm end
            end
        end
        local c = logistics.crowding(list)
        return {soldiers = c.soldiers, pairs = c.pairs}, with_soldiers and rows or nil
    end

    -- One tick of the approach commander; true when the stage is over.
    local function approach_tick(now, elapsed)
        local c = state.commander
        if c.current then
            if state.dispatch then logistics_tick(now) end
            state.appr_still = any_moving() and 0 or (state.appr_still or 0) + 1
            if state.ticks % 2 == 0 then
                local rows = {}
                for _, it in ipairs(state.own) do rows[#rows + 1] = {script_name = it.name, motion = unit_motion.motion(it.unit)} end
                -- Reading every soldier is costly (a test measure, not the AI): every 4th tick, all soldiers every 20th.
                local crowd, soldiers, crowd_clock
                if state.ticks % 4 == 0 then
                    local started = os.clock and os.clock() or 0
                    crowd, soldiers = crowd_now(state.ticks % 20 == 0)
                    crowd_clock = (os.clock and os.clock() or 0) - started
                end
                -- Research only (the AI never reads it): where the game's AI goes meanwhile.
                local enemy
                if config.enemy_ai then
                    enemy = {}
                    for _, it in ipairs(state.enemy) do
                        enemy[#enemy + 1] = {script_name = it.name, motion = unit_motion.motion(it.unit)}
                    end
                end
                emit('approach_sample', {kind = c.current.kind, t_ms = now - c.current.started_ms, units = rows,
                    crowd = crowd, soldiers_dm = soldiers, crowd_clock_s = crowd_clock, logistics = state.dispatch ~= nil,
                    enemy = enemy})
            end
            local took = now - c.current.started_ms
            -- Under missile fire the manoeuvre is broken off at once (the one
            -- exception to "one action at a time", 28.09.2026: in battle the army
            -- waited out a 6-minute alignment under fire and was beaten).
            local fired_on
            for _, it in ipairs(state.own) do
                local ok, under = pcall(function() return it.unit:is_under_missile_attack() end)
                if ok and under then fired_on = it.name break end
            end
            if fired_on then
                local kind = c.current.kind
                c.finish(now, 'under_fire')
                for _, it in ipairs(state.own) do orders.halt(it.uc) end
                state.dispatch = nil
                emit('approach_manoeuvre', {kind = kind, t_ms = took, reason = 'under_fire', unit = fired_on,
                    units = units_row()})
                return false
            end
            local timeout = took >= config.manoeuvre_timeout_s * 1000
            local dispatched = not state.dispatch or state.dispatch.done()
            if (took >= config.settle_ms and state.appr_still >= 3 and dispatched) or timeout then
                local kind = c.current.kind
                c.finish(now, timeout and 'timeout' or 'stopped')
                local row = {kind = kind, t_ms = took, reason = timeout and 'timeout' or 'stopped', units = units_row(),
                    logistics = state.dispatch ~= nil}
                if state.dispatch then
                    row.dispatch_log, row.logistics_orders = state.dispatch.log, state.logistics.orders
                    state.dispatch = nil
                end
                emit('approach_manoeuvre', row)
            end
            return false
        end
        local decide_started = os.clock and os.clock() or 0
        local picture, field = battle_picture()
        if not field then return elapsed >= config.approach_timeout_s * 1000 end
        local m = mask_services.new(mask_services.grid(field))
        local reach_unit
        for _, it in ipairs(state.own) do if it.role == 'wall' then reach_unit = it.unit; break end end
        mask_services.fill(m, mask_adapter.reader(bm, vector_type, reach_unit or state.own[1].unit, m.grid.step))
        -- The stop: the window (apps.reach, battle theory phase 2) — ours reach
        -- theirs, theirs do not reach us. Marching to a goal point, or nobody
        -- shooting: 20 m short of their front.
        local window, stop
        if not config.goal then
            local own_blocks, enemy_blocks = reach_blocks()
            window = reach.window(own_blocks, enemy_blocks, state.placed.bearing)
        end
        if window and window.advance_m then
            stop = field.gap_m - window.advance_m
        else
            stop = approach.stop_gap(0, 0)
        end
        local step = approach.next_step(field.gap_m, stop)
        local check = alignment.check(field, {anchor = state.placed.anchor, bearing = state.placed.bearing},
            enemy_of(picture))
        local gov = state.governor.decide(now, check, false)
        local path = {ok = false, reason = 'no_step'}
        if step.action == 'step' then
            local function fits_at(a)
                for _, it in ipairs(state.own) do
                    local p = it.placement
                    if p then
                        local b = math.rad(p.bearing)
                        local moved = {x = p.x + math.sin(b) * a, z = p.z + math.cos(b) * a, bearing = p.bearing,
                            front_m = p.front_m, depth_m = p.depth_m}
                        if not mask_services.fits(m, moved).ok then return false end
                    end
                end
                return true
            end
            local function lane_free_at(a)
                return mask_services.lane(m, 0, field.own.width_m, field.own.front_m, field.own.front_m + a).free
            end
            -- Past an obstacle: never into their shooters' reach.
            local limit = field.gap_m - approach.DEFAULTS.margin_m
            if window and window.safe_to then limit = math.min(limit, window.safe_to) end
            path = approach.choose_advance(step.advance_m, field.gap_m - stop, fits_at, lane_free_at, 3, limit)
        end
        local decision, reason = c.decide({align = check, governor = gov, step = step, path = path,
            under_fire = window and window.under_fire_now})
        local decide_clock = (os.clock and os.clock() or 0) - decide_started
        emit('approach_decision', {decision = decision, reason = reason, gap_m = field.gap_m, stop_gap_m = stop,
            window = window, step = step, path = path,
            align = {angle_off_deg = check.angle_off_deg, offset_m = check.offset_m, needed = check.needed},
            mask = mask_services.summary(m), clock_s = decide_clock})
        if decision == 'approach' then
            local b = math.rad(state.placed.bearing)
            local a = path.advance_m
            local anchor = {x = state.placed.anchor.x + math.sin(b) * a, z = state.placed.anchor.z + math.cos(b) * a}
            if config.logistics and path.detour then
                start_logistics(field, m, anchor, state.placed.bearing, now)
            else
                order_formation(anchor, state.placed.bearing)
            end
            c.start('approach', now, nil)
            state.appr_still = 0
        elseif decision == 'align' then
            local target = alignment.target(field, {anchor = state.placed.anchor, bearing = state.placed.bearing},
                enemy_of(picture))
            state.governor.record_order(now, check)
            order_formation(target.anchor, target.bearing)
            c.start('align', now, nil)
            state.appr_still = 0
        elseif decision == 'hold' or decision == 'blocked' then
            return true
        end
        return elapsed >= config.approach_timeout_s * 1000
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status, stages_done = state.stage_index, stages = #state.stages, ticks = state.ticks,
            orders_after_placed = state.orders_after_placed, orders_alignment = state.orders_alignment,
            orders_approach = state.orders_approach or 0})
        for _, it in ipairs(state.own) do orders.halt(it.uc) end
        for _, it in ipairs(state.enemy) do if it.uc then orders.halt(it.uc) end end
        flush()
        bm:end_battle()
    end

    -- The game's AI keeps the enemy but defends where it stands (as in
    -- entries.enemy_layout: CA's script_ai_planner, else the engine planner).
    local function enemy_defend()
        local sx, sz = 0, 0
        for _, it in ipairs(state.enemy) do
            local p = it.unit:position()
            sx, sz = sx + p:get_x(), sz + p:get_z()
        end
        local centre = vec(sx / #state.enemy, sz / #state.enemy)
        local report = {radius_m = config.defend_radius_m or M.DEFEND_RADIUS_M, centre = {x = sx / #state.enemy, z = sz / #state.enemy}}
        if script_ai_planner and script_ai_planner.defend_position and bm.get_scriptunit_for_unit then
            local list = {}
            for _, it in ipairs(state.enemy) do list[#list + 1] = bm:get_scriptunit_for_unit(it.unit) end
            state.planner = script_ai_planner:new('tww3_bai_enemy_defend', list)
            state.planner:defend_position(centre, config.defend_radius_m or M.DEFEND_RADIUS_M)
            report.used = 'script_ai_planner.defend_position'
        else
            local planner = state.enemy_alliance:create_ai_unit_planner()
            for _, it in ipairs(state.enemy) do planner:add_units(it.unit) end
            planner:defend_position(centre, config.defend_radius_m or M.DEFEND_RADIUS_M)
            state.planner = planner
            report.used = 'engine_planner.defend_position'
        end
        emit('enemy_mode', report)
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
        if stage[3] == 'manual' then
            -- Hand the army over: from now on the script only records.
            for _, it in ipairs(state.own) do orders.release(it.uc) end
            if config.enemy_ai == 'defend' then enemy_defend() end
        end
        emit('stage', {stage = stage[1], turn_deg = stage[2], how = stage[3]})
    end

    -- Handover: a player's order is seen only as a change of the ordered point
    -- or width (as in manual_record); each one gets a navigation leg monitor.
    local function manual_tick(now, elapsed)
        local rows = {}
        for _, it in ipairs(state.own) do
            local m = unit_motion.motion(it.unit)
            local row = {script_name = it.name, role = it.role, motion = m}
            if state.ticks % 2 == 0 then
                local men = unit_motion.soldiers(cco, it.unit)
                row.soldiers_dm = men.xz_dm
            end
            rows[#rows + 1] = row
            if m.ordered_x then
                local last = it.order
                local changed = not last
                    or math.sqrt((m.ordered_x - last.x) ^ 2 + (m.ordered_z - last.z) ^ 2) > 1
                    or math.abs((m.ordered_width or 0) - (last.width or 0)) > 0.5
                if changed then
                    it.order = {x = m.ordered_x, z = m.ordered_z, width = m.ordered_width, ms = now}
                    if last then
                        emit('player_order', {unit = it.name, role = it.role, x = m.ordered_x, z = m.ordered_z,
                            width = m.ordered_width, bearing = m.ordered_bearing, from_x = m.x, from_z = m.z, t_ms = elapsed})
                        it.monitor = navigation.new_leg_monitor({target = {x = m.ordered_x, z = m.ordered_z},
                            timeout_ms = 600000})
                    end
                end
            end
            if it.monitor then
                local reason = it.monitor.update(now - it.order.ms, {x = m.x, z = m.z, moving = m.is_moving == true})
                if reason then
                    local st = it.monitor.stats
                    emit('order_end', {unit = it.name, reason = reason, t_ms = now - it.order.ms, x = m.x, z = m.z,
                        path_m = st.path_m, distance_m = st.distance_m, min_distance_m = st.min_distance_m})
                    it.monitor = nil
                end
            end
        end
        -- Research only (the AI never reads it): where the enemy goes, to tell
        -- whether the game's AI attacks or defends.
        local enemy = {}
        if config.enemy_ai then
            for _, it in ipairs(state.enemy) do
                enemy[#enemy + 1] = {script_name = it.name, motion = unit_motion.motion(it.unit)}
            end
        end
        emit('manual_sample', {t_ms = elapsed, units = rows, enemy = enemy})
    end

    local function tick()
        if not state.active then return end
        local tick_started = os.clock and os.clock() or 0
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
        if stage[3] == 'mask' and not state.mask_done then state.mask_done = mask_step() end
        if stage[3] == 'approach' and not state.approach_done then
            state.approach_done = approach_tick(now, elapsed)
        end
        if stage[3] == 'manual' then manual_tick(now, elapsed) end
        if stage[3] == 'sweep' and not state.sweep_done then
            -- One commanded bearing every config.facing_sweep.ticks ticks, sent
            -- raw (not through apps.orders.facing), read just before the next.
            local sw = config.facing_sweep
            local it = state.by_name[sw.unit]
            state.sweep = state.sweep or {i = 0, wait = 0}
            local s = state.sweep
            if s.i > 0 and s.wait == 0 then
                local men = unit_motion.soldiers(cco, it.unit)
                emit('sweep_sample', {sent = sw.bearings[s.i], engine = it.unit:bearing(), soldiers_dm = men.xz_dm})
            end
            if s.wait == 0 then
                s.i = s.i + 1
                if s.i > #sw.bearings then
                    state.sweep_done = true
                else
                    local p = it.placement
                    it.uc:teleport_to_location(vec(p.x, p.z), sw.bearings[s.i], p.width)
                    s.wait = sw.ticks
                end
            end
            s.wait = s.wait - 1
            if s.wait < 0 then s.wait = 0 end
        end
        if stage[3] == 'hold' then
            local rows = {}
            for _, it in ipairs(state.own) do
                local row = {script_name = it.name, motion = unit_motion.motion(it.unit)}
                if config.fire then
                    local ok_a, ammo = pcall(function() return it.unit:ammo_left() end)
                    local ok_u, under = pcall(function() return it.unit:is_under_missile_attack() end)
                    row.ammo = ok_a and ammo or nil
                    row.under_fire = ok_u and under or nil
                end
                rows[#rows + 1] = row
            end
            -- Research only (the AI never reads it): the enemy's men and where it goes.
            local enemy = {}
            if config.enemy_ai then
                for _, it in ipairs(state.enemy) do
                    enemy[#enemy + 1] = {script_name = it.name, motion = unit_motion.motion(it.unit)}
                end
            end
            emit('hold_sample', {t_ms = elapsed, orders_after_placed = state.orders_after_placed, units = rows,
                enemy = enemy})
            if state.ticks % 5 == 0 then ask_governor(elapsed) end
        end
        local moving = false
        for _, it in ipairs(state.own) do
            if it.unit:is_moving() then moving = true end
        end
        state.still = moving and 0 or state.still + 1
        local done
        if stage[3] == 'manual' then
            done = false  -- until the player ends the battle (or the deadline)
        elseif stage[3] == 'sweep' then
            done = state.sweep_done == true
        elseif stage[3] == 'hold' then
            done = elapsed >= config.hold_s * 1000
        elseif stage[3] == 'approach' then
            done = state.approach_done
        elseif stage[3] == 'mask' then
            done = state.mask_done or elapsed >= config.align_timeout_s * 1000
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
                units = units_row(), tick_clock_max_s = state.tick_clock_max, tick_clock_sum_s = state.tick_clock_sum,
                ticks = state.stage_ticks})
            state.tick_clock_max, state.tick_clock_sum, state.stage_ticks = 0, 0, 0
            if state.stage_index < #state.stages then begin_stage(state.stage_index + 1) else finish('completed') end
        end
        flush()
        -- How long the script takes per tick (the game waits for it).
        local took = (os.clock and os.clock() or 0) - tick_started
        state.tick_clock_max = math.max(state.tick_clock_max or 0, took)
        state.tick_clock_sum = (state.tick_clock_sum or 0) + took
        state.stage_ticks = (state.stage_ticks or 0) + 1
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
        -- On the map (user, 28.09.2026: units put on a rock stood crooked): the
        -- stand mask around our army, read from the engine as in the mask stage;
        -- the formation takes a width and a place where every unit fits.
        local started = os.clock and os.clock() or 0
        local reach_unit = state.own[1].unit
        for _, it in ipairs(state.own) do
            if it.unit:initial_number_of_men() > reach_unit:initial_number_of_men() then reach_unit = it.unit end
        end
        local area = mask_services.new(mask_services.area(plan_services.area(config.own.anchor, bearing)))
        mask_services.fill(area, mask_adapter.reader(bm, vector_type, reach_unit, area.grid.step))
        local mask_clock = (os.clock and os.clock() or 0) - started
        -- The enemy roster (unit types) is known before the battle, as to the player.
        local result = plan_services.start({role = config.role, roles = config.roles, bearing = bearing,
            own = {anchor = config.own.anchor, units = config.own.units},
            enemy = {units = config.enemy.units, lord = lord, blocks = enemy_blocks()}, formation_params = config.formation_params,
            fits = function(q) return mask_services.fits(area, q, true).ok end})
        local fit_clock = (os.clock and os.clock() or 0) - started - mask_clock
        emit('plan', {facing_source = source, enemies_seen = seen, enemy_centre = centre, bearing = bearing,
            status = result.status, strategy = result.decision.strategy, features = result.features,
            candidates = result.decision.candidates, plan = result.formation,
            fit = result.formation and result.formation.fit, mask = mask_services.summary(area),
            mask_clock_s = mask_clock, fit_clock_s = fit_clock})
        assert(result.formation, 'No strategy for this battle: ' .. tostring(result.status))
        local plan = result.formation
        if plan.anchor then state.placed.anchor, state.placed.bearing = plan.anchor, plan.bearing end
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
        if config.speed then
            bm:modify_battle_speed(config.speed)
            battle.speed_guard(bm, config.speed, function(from)
                emit('speed_restored', {from_speed = from, to_speed = config.speed})
            end, 'tww3_bai_formation_probe_speed')
        end
        bm:change_victory_countdown_limit(-1)
        for _, p in ipairs(config.enemy.placements) do
            local it = state.by_name[p.script_name]
            if it.uc then
                orders.teleport(it.uc, vec(p.x, p.z), p.bearing, p.width)
                orders.halt(it.uc)
            end
        end
        emit('start', {speed = config.speed, deadline_s = config.deadline_s, stall_ms = config.stall_ms})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_formation_probe_deadline')
        bm:add_infotext('BAI: FORMATION PROBE')
        -- The enemy teleport applies on a later tick; plan once it has.
        bm:real_callback(guarded(function()
            place_own()
            -- config.fire: our shooters fire at will (the window check, battle theory phase 2).
            if config.fire then
                for _, it in ipairs(state.own) do orders.set_fire_at_will(it.uc, true) end
            end
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
        state.enemy_alliance = sides[2].alliance
        -- Who the game takes for the attacker (deployment areas may decide it).
        local ok_roles, roles = pcall(battle.read_roles, bm)
        emit('roles', {ok = ok_roles, roles = ok_roles and roles or tostring(roles)})
        state.all_units = {}
        -- With the game's AI (the player's test, or config.enemy_ai) the enemy is never taken: it
        -- deploys and fights by itself (as in entries.enemy_layout; taken and
        -- released, it stood idle — 27.09.2026).
        local native_enemy = config.enemy_ai
        local function adopt(side, list, name)
            local u = battle.find_by_name(sides[side], name)
            assert(u, 'scenario unit missing: ' .. name)
            local it = {name = name, unit = u}
            if not (side == 2 and native_enemy) then
                it.uc = orders.take_control(sides[side].army, u)
                orders.set_fire_at_will(it.uc, false)
                orders.halt(it.uc)
            end
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
