-- Entry: the missile probe (tools/nn/missile_probe.py). One shooting indicator, one scenario, the game and the
-- simulator alike: lanes far apart, in each a shooter facing bearing s_b and a target whose centre is d metres
-- away at t_angle degrees off the shooter's facing, the target facing the shooter turned by t_rot degrees
-- (0: its front to the shooter, 90: its flank, 180: its back). Everyone is held by script and fearless.
-- The shooter (lane.mode): 'fire' fire at will, no order; 'attack' an attack order on its target.
-- The target (lane.target_mode): 'stand' halts; 'step' starts at start_d and steps step_m closer every
-- step_s seconds until the shooter's first shot (the range test); 'move' at the go is ordered to a point
-- (move_fwd, move_lat: metres ahead of and across from the shooter, in the shooter's frame), at a run with
-- move_run (a crossing, running at the shooter, running away).
-- lane.reform_men (optional): once the target is down to that many men it is ordered to a point 5 m to its right
-- with its starting width (goto_location_angle_width: the thinned unit re-forms), then stands.
-- lane.friend (optional): a unit of the shooter's side placed friend_fwd m ahead of and friend_lat m right of the
-- shooter, facing as it does, held (the line of fire past friends); its row is sampled as 'f'.
-- Every tick_ms 'probe_sample': per running lane the shooter's ammo, men, firing flag, place, bearing,
-- moving; the target's men, health (CCO HealthValue), unary hit points, place, bearing, moving. A lane ends
-- when the shooter is out of ammo or the target dead (and settle_s more for the projectiles in the air), when
-- the shooter has not shot for idle_s after its first shot, or after max_s; then both halt and the shooter
-- stops firing. The battle ends when every lane is done.
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local map = require('apps.map.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_missile_probe_tick'
M.MODES = {fire = true, attack = true}
M.TARGET_MODES = {stand = true, step = true, move = true}

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- A point `fwd` metres ahead of and `lat` metres right of (x, z) facing bearing b (degrees; 0 = +z). Pure.
function M.frame(x, z, b, fwd, lat)
    local r = math.rad(b)
    local fx, fz = math.sin(r), math.cos(r)
    local rx, rz = math.cos(r), -math.sin(r)
    return x + fx * fwd + rx * lat, z + fz * fwd + rz * lat
end

-- Where a lane puts its two units at distance d: {sx, sz, sb, tx, tz, tb} (centres and bearings). The shooter
-- stands at (x, z) facing s_b; the target's centre is d away at t_angle off that facing, facing the shooter
-- turned by t_rot. Pure.
function M.layout(lane, d)
    local sb = lane.s_b or 0
    local a = sb + (lane.t_angle or 0)
    local r = math.rad(a)
    local tx, tz = lane.x + d * math.sin(r), lane.z + d * math.cos(r)
    return {sx = lane.x, sz = lane.z, sb = sb, tx = tx, tz = tz, tb = (a + 180 + (lane.t_rot or 0)) % 360}
end

-- config: build, speed, tick_ms, deadline_s, settle_ms, lanes = {{name, shooter, target, x, z, s_b, d, t_angle,
-- t_rot, s_width, t_width, mode, target_mode, start_d, step_m, step_s, move_fwd, move_lat, move_run, max_s,
-- idle_s, settle_s}}, park = {{name, x, z, bearing}}. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_missile_probe then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', units = {}, lanes = {}}
    _G.tww3_bai_missile_probe = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'missile_probe'
        row.policy = 'missile_probe'
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
    local function read(fn, ...)
        local ok, v = pcall(fn, ...)
        if ok then return v end
        return nil
    end
    local function vec(x, z)
        return map.vector(vector_type, x, bm:get_terrain_height(x, z), z)
    end
    local function now_ms() return bm:time_elapsed_ms() end

    local function unit_row(u)
        local p = read(function() return u:position() end)
        return {men = read(function() return u:number_of_men_alive() end),
            hp = round(read(cco, u, 'HealthValue'), 0), hpu = round(read(function() return u:unary_hitpoints() end), 4),
            x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(read(function() return u:bearing() end), 0), mv = read(function() return u:is_moving() end)}
    end

    local function place(it, x, z, bearing, width)
        orders.teleport(it.uc, vec(x, z), facing.snap(bearing), width)
        orders.halt(it.uc)
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', winner = 0})
        flush()
        bm:end_battle()
    end

    local function end_lane(lane, why)
        if not lane.running then return end
        lane.running = false
        orders.stop_firing(lane.s.uc)
        orders.halt(lane.t.uc)
        emit('probe_lane_end', {lane = lane.name, why = why, t = now_ms() - lane.t0,
            first_ms = lane.first_ms and lane.first_ms - lane.t0})
    end

    local function tick()
        if not state.active then return end
        local rows, busy, now = {}, false, now_ms()
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local s, t = unit_row(lane.s.unit), unit_row(lane.t.unit)
                s.ammo = read(function() return lane.s.unit:ammo_left() end)
                s.fire = read(cco, lane.s.unit, 'IsFiringMissiles')
                lane.ammo0 = lane.ammo0 or s.ammo
                if s.ammo and lane.last_ammo and s.ammo < lane.last_ammo then
                    lane.last_shot = now
                    if not lane.first_ms then
                        lane.first_ms = now
                        emit('probe_first', {lane = lane.name, t = now - lane.t0, d = lane.d})
                    end
                end
                lane.last_ammo = s.ammo
                -- the range test: the target steps closer until the first shot
                if lane.target_mode == 'step' and not lane.first_ms and now - lane.step_at >= lane.step_s * 1000 then
                    lane.step_at = now
                    lane.d = lane.d - lane.step_m
                    local L = M.layout(lane, lane.d)
                    place(lane.t, L.tx, L.tz, L.tb, lane.t_width)
                end
                if lane.reform_men and not lane.reformed and t.men and t.men <= lane.reform_men and t.x then
                    lane.reformed = now
                    local tb = M.layout(lane, lane.d).tb      -- its placed facing (the place() convention)
                    local rx, rz = M.frame(t.x, t.z, tb, 0, 5)
                    orders.move_formation(lane.t.uc, vec(rx, rz), tb, lane.t_width, false)
                    emit('probe_reform', {lane = lane.name, t = now - lane.t0, men = t.men})
                end
                local row = {lane = lane.name, t = now - lane.t0, d = lane.d, s = s, tg = t}
                if lane.f then row.f = unit_row(lane.f.unit) end
                rows[#rows + 1] = row
                local over = (s.ammo == 0) or (t.men == 0)
                if over and not lane.over_ms then lane.over_ms = now end
                if lane.over_ms and now - lane.over_ms >= lane.settle_s * 1000 then
                    end_lane(lane, s.ammo == 0 and 'ammo' or 'dead')
                elseif lane.last_shot and now - lane.last_shot >= lane.idle_s * 1000 then
                    end_lane(lane, 'idle')
                elseif now - lane.t0 >= lane.max_s * 1000 then
                    end_lane(lane, 'max_s')
                end
                busy = busy or lane.running
            end
        end
        if #rows > 0 then emit('probe_sample', {lanes = rows}) end
        flush()
        if not busy then finish('completed') end
    end

    local function go()
        for _, lane in ipairs(state.lanes) do
            lane.t0, lane.running, lane.step_at = now_ms(), true, now_ms()
            if lane.mode == 'attack' then
                orders.attack_ranged(lane.s.uc, lane.t.unit, false, true)
            else
                orders.set_fire_at_will(lane.s.uc, true)
            end
            if lane.target_mode == 'move' then
                local mx, mz = M.frame(lane.x, lane.z, lane.s_b or 0, lane.move_fwd, lane.move_lat or 0)
                orders.move(lane.t.uc, vec(mx, mz), lane.move_run == true)
            end
        end
        emit('probe_go', {lanes = #state.lanes})
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        flush()
    end

    local function start()
        if state.active or state.finished or state.started then return end
        state.started = true
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_missile_probe_speed')
        pcall(function() bm:change_victory_countdown_limit(-1) end)
        for _, p in ipairs(config.park or {}) do
            local it = state.units[p.name]
            if it then place(it, p.x, p.z, p.bearing, p.width or 5) end
        end
        for _, lane in ipairs(state.lanes) do
            local L = M.layout(lane, lane.d)
            place(lane.s, L.sx, L.sz, L.sb, lane.s_width)
            place(lane.t, L.tx, L.tz, L.tb, lane.t_width)
            if lane.f then
                local fx, fz = M.frame(lane.x, lane.z, L.sb, lane.friend_fwd or 0, lane.friend_lat or 0)
                place(lane.f, fx, fz, L.sb, lane.friend_width or 30)
            end
        end
        emit('start', {speed = config.speed, lanes = config.lanes})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_missile_probe_deadline')
        bm:add_infotext('BAI: MISSILE PROBE')
        bm:callback(guarded(go), config.settle_ms, 'tww3_bai_missile_probe_go')
        flush()
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
        local sides = battle.read_sides(bm)
        for side = 1, 2 do
            for _, u in ipairs(sides[side].units) do
                local name = u:name()
                local uc = orders.take_control(sides[side].army, u)
                orders.set_fire_at_will(uc, false)
                orders.halt(uc)
                pcall(function() uc:morale_behavior_fearless() end)
                state.units[name] = {name = name, unit = u, uc = uc, side = side}
            end
        end
        for i, l in ipairs(config.lanes) do
            assert(M.MODES[l.mode], 'unknown mode ' .. tostring(l.mode))
            assert(M.TARGET_MODES[l.target_mode], 'unknown target mode ' .. tostring(l.target_mode))
            local lane = {}
            for k, v in pairs(l) do lane[k] = v end
            lane.s, lane.t = state.units[l.shooter], state.units[l.target]
            assert(lane.s, 'scenario unit missing: ' .. tostring(l.shooter))
            assert(lane.t, 'scenario unit missing: ' .. tostring(l.target))
            if l.friend then
                lane.f = state.units[l.friend]
                assert(lane.f, 'scenario unit missing: ' .. tostring(l.friend))
            end
            if l.target_mode == 'step' then lane.d = l.start_d end
            state.lanes[i] = lane
        end
        state.batch = clock.batch_stamp() .. '-mprobe-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {lanes = #state.lanes})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_missile_probe_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
