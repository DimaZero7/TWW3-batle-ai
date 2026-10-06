-- Entry: the morale probe (tools/nn/morale_probe.py). One morale term, one scenario, the game and the
-- simulator alike: lanes far apart, in each a few units placed by script (lane.units: role, script
-- name, place, bearing, width) and held (no AI); the units named in config.fearless are fearless (the
-- other side of the term under test keeps its morale). Each lane runs a list of scripted steps
-- (lane.steps), each fired once when its trigger has happened and delay_s more seconds have passed:
--   on = 'go'       the probe's start;
--   on = 'contact'  the first melee of role `of` (is_in_melee);
--   on = 'rout'     the first rout of role `of`;  on = 'rally'  its first rally after a rout;
--   on = 'step'     step number `of` of the lane fired.
-- What a step does (do): 'attack' / 'attack_walk' (unit attacks target in melee, at a run / walk),
-- 'shoot' (unit shoots target), 'move' / 'move_walk' (to lane point x, z), 'halt', 'stop_fire',
-- 'shadow' (each tick unit keeps d metres from role `target`: a move at a run to the point d from
-- it on the line between them, once it is more than 8 m off; until a 'halt' of that unit),
-- 'ability' (unit uses ability `key` on itself: a lord's Stand Your Ground, Rally),
-- 'end' (the lane ends).
-- Every tick_ms 'probe_sample': per running lane every role's place, bearing, men, health, melee,
-- routing / wavering / shattered, CCO MoralePercent and MoraleGreatestEffect (the strongest morale
-- effect's text), PercentHpLostRecently, moving fast, firing, and its active effects (fx: the phase
-- keys of CCO ActiveEffectList, as the bridge's nn_effects). A lane also ends after max_s. The
-- battle ends when every lane is done.
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local map = require('apps.map.adapter')
local services = require('apps.bridge.services')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_morale_probe_tick'
M.TRIGGERS = {go = true, contact = true, rout = true, rally = true, step = true}
M.ACTIONS = {attack = true, attack_walk = true, shoot = true, move = true, move_walk = true, halt = true,
    stop_fire = true, shadow = true, ability = true, ['end'] = true}

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- The point `d` metres from (tx, tz) towards (ux, uz) (pure): where a shadowing unit stands.
function M.shadow_point(ux, uz, tx, tz, d)
    local dx, dz = ux - tx, uz - tz
    local n = math.sqrt(dx * dx + dz * dz)
    if n < 1e-6 then return tx, tz + d end
    return tx + dx / n * d, tz + dz / n * d
end

-- Has step s's trigger happened, and when (ms)? lane.events: {go = ms, contact = {role = ms}, ...,
-- step = {index = ms}} (pure). Returns the trigger's time or nil.
function M.trigger_ms(s, events)
    if s.on == 'go' then return events.go end
    local book = events[s.on]
    return book and book[s.of]
end

-- config: build, speed, tick_ms, deadline_s, settle_ms, fearless = {names}, park = {{name, x, z,
-- bearing, width}}, lanes = {{name, x, z, max_s, units = {{role, name, x, z, b, width}}, steps = {{on,
-- of, delay_s, ['do'], unit, target, x, z, d}}}} (unit places in the lane's frame: x, z added to the
-- lane's). globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_morale_probe then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', units = {}, lanes = {}}
    _G.tww3_bai_morale_probe = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'morale_probe'
        row.policy = 'morale_probe'
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function stop_timers() pcall(function() bm:remove_process(TIMER) end) end
    local function fail(err)
        state.active = false
        state.finished = true
        stop_timers()
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
        local row = {men = read(function() return u:number_of_men_alive() end),
            hp = round(read(function() return u:unary_hitpoints() end), 4),
            m = read(function() return u:is_in_melee() end), x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(read(function() return u:bearing() end), 0),
            fast = read(function() return u:is_moving_fast() end),
            r = read(function() return u:is_routing() end), w = read(function() return u:is_wavering() end),
            s = read(function() return u:is_shattered() end),
            mp = round(read(cco, u, 'MoralePercent'), 3), phr = round(read(cco, u, 'PercentHpLostRecently'), 4),
            fire = read(cco, u, 'IsFiringMissiles')}
        local effect = read(cco, u, 'MoraleGreatestEffect')
        if type(effect) == 'string' and effect ~= '' then row.mge = effect end
        local fx = services.active_effects(function(field) return read(cco, u, field) end)
        if fx and #fx > 0 then row.fx = table.concat(fx, ',') end
        return row
    end

    local function place(it, x, z, bearing, width)
        orders.teleport(it.uc, vec(x, z), facing.snap(bearing), width)
        orders.halt(it.uc)
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        stop_timers()
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', winner = 0})
        flush()
        bm:end_battle()
    end

    local function end_lane(lane, why)
        if not lane.running then return end
        lane.running = false
        for _, it in pairs(lane.roles) do pcall(orders.halt, it.uc) end
        emit('probe_lane_end', {lane = lane.name, why = why, t = now_ms() - lane.events.go})
    end

    local function act(lane, s, now)
        local it = s.unit and lane.roles[s.unit]
        local tg = s.target and lane.roles[s.target]
        local d = s['do']
        if d == 'attack' or d == 'attack_walk' then
            orders.attack_melee(it.uc, tg.unit, d == 'attack_walk')
        elseif d == 'shoot' then
            orders.attack_ranged(it.uc, tg.unit, false, false)
        elseif d == 'move' or d == 'move_walk' then
            orders.move(it.uc, vec(lane.x + s.x, lane.z + s.z), d == 'move')
        elseif d == 'halt' then
            lane.shadows[s.unit] = nil
            orders.halt(it.uc)
        elseif d == 'stop_fire' then
            orders.stop_firing(it.uc)
        elseif d == 'ability' then
            local ok, used = pcall(orders.use_ability_on_self, it.uc, it.unit, s.key)
            emit('probe_ability', {lane = lane.name, unit = s.unit, key = s.key, ok = ok and used == true,
                t = now - lane.events.go})
        elseif d == 'shadow' then
            lane.shadows[s.unit] = {target = s.target, d = s.d}
        elseif d == 'end' then
            end_lane(lane, 'step')
        end
    end

    local function shadow(lane)
        for role, sh in pairs(lane.shadows) do
            local it, tg = lane.roles[role], lane.roles[sh.target]
            local p, q = read(function() return it.unit:position() end), read(function() return tg.unit:position() end)
            if p and q then
                local x, z = M.shadow_point(p:get_x(), p:get_z(), q:get_x(), q:get_z(), sh.d)
                if math.sqrt((p:get_x() - x) ^ 2 + (p:get_z() - z) ^ 2) > 8 then
                    orders.move(it.uc, vec(x, z), true)
                    it.shadow_moving = true
                elseif it.shadow_moving then
                    orders.halt(it.uc)
                    it.shadow_moving = false
                end
            end
        end
    end

    local function tick()
        if not state.active then return end
        local rows, busy, now = {}, false, now_ms()
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local ev = lane.events
                local r = {lane = lane.name, t = now - ev.go, u = {}}
                for role, it in pairs(lane.roles) do
                    local row = unit_row(it.unit)
                    r.u[role] = row
                    if row.m and not ev.contact[role] then
                        ev.contact[role] = now
                        emit('probe_event', {lane = lane.name, kind = 'contact', role = role, t = now - ev.go})
                    end
                    if row.r and not ev.rout[role] then
                        ev.rout[role] = now
                        emit('probe_event', {lane = lane.name, kind = 'rout', role = role, t = now - ev.go})
                    end
                    if ev.rout[role] and row.r == false and not ev.rally[role] then
                        ev.rally[role] = now
                        emit('probe_event', {lane = lane.name, kind = 'rally', role = role, t = now - ev.go})
                    end
                end
                rows[#rows + 1] = r
                for i, s in ipairs(lane.steps) do
                    if lane.running and not ev.step[i] then
                        local at = M.trigger_ms(s, ev)
                        if at and now - at >= (s.delay_s or 0) * 1000 then
                            ev.step[i] = now
                            emit('probe_step', {lane = lane.name, i = i, ['do'] = s['do'], unit = s.unit,
                                target = s.target, t = now - ev.go})
                            act(lane, s, now)
                        end
                    end
                end
                if lane.running then shadow(lane) end
                if lane.running and now - ev.go >= lane.max_s * 1000 then end_lane(lane, 'max_s') end
                busy = busy or lane.running
            end
        end
        if #rows > 0 then emit('probe_sample', {lanes = rows}) end
        flush()
        if not busy then finish('completed') end
    end

    local function go()
        local now = now_ms()
        for _, lane in ipairs(state.lanes) do
            lane.running = true
            lane.events = {go = now, contact = {}, rout = {}, rally = {}, step = {}}
        end
        emit('probe_go', {lanes = #state.lanes})
        state.active = true
        guarded(tick)()
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        flush()
    end

    local function start()
        if state.active or state.finished or state.started then return end
        state.started = true
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_morale_probe_speed')
        pcall(function() bm:change_victory_countdown_limit(-1) end)
        for _, p in ipairs(config.park or {}) do
            local it = state.units[p.name]
            if it then place(it, p.x, p.z, p.bearing, p.width or 5) end
        end
        for _, lane in ipairs(state.lanes) do
            for _, spec in ipairs(lane.units) do
                place(lane.roles[spec.role], lane.x + spec.x, lane.z + spec.z, spec.b, spec.width or 30)
            end
        end
        emit('start', {speed = config.speed, lanes = config.lanes})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_morale_probe_deadline')
        bm:add_infotext('BAI: MORALE PROBE')
        bm:callback(guarded(go), config.settle_ms, 'tww3_bai_morale_probe_go')
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
        local fearless = {}
        for _, name in ipairs(config.fearless or {}) do fearless[name] = true end
        local sides = battle.read_sides(bm)
        for side = 1, 2 do
            for _, u in ipairs(sides[side].units) do
                local name = u:name()
                local uc = orders.take_control(sides[side].army, u)
                orders.set_fire_at_will(uc, false)
                orders.halt(uc)
                if fearless[name] then pcall(function() uc:morale_behavior_fearless() end) end
                state.units[name] = {name = name, unit = u, uc = uc, side = side}
            end
        end
        for i, l in ipairs(config.lanes) do
            local lane = {name = l.name, x = l.x, z = l.z, max_s = l.max_s, units = l.units, steps = l.steps or {},
                roles = {}, shadows = {}}
            for _, spec in ipairs(l.units) do
                lane.roles[spec.role] = state.units[spec.name]
                assert(lane.roles[spec.role], 'scenario unit missing: ' .. tostring(spec.name))
            end
            for _, s in ipairs(lane.steps) do
                assert(M.TRIGGERS[s.on], 'unknown trigger ' .. tostring(s.on))
                assert(M.ACTIONS[s['do']], 'unknown action ' .. tostring(s['do']))
                assert(s['do'] == 'end' or lane.roles[s.unit], 'unknown role ' .. tostring(s.unit))
            end
            state.lanes[i] = lane
        end
        state.batch = clock.batch_stamp() .. '-mprobe-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {lanes = #state.lanes})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_morale_probe_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
