-- Entry: how much does a lord lose when several units surround him? (user, 01.10.2026:
-- damage is dealt by the soldiers that reach and fight, not by the whole unit, so four
-- units around a lord may gain little.) Research probe, both sides held by script.
-- Two lanes far apart; in each a lord stands (ours: the General, theirs: the Warlord) and
-- the other side's infantry attacks him. A trial puts 1-4 attacking units around each lord
-- (front, back, left, right), start_m from him, facing him; after settle_ms they are
-- ordered to attack him and the lord is told to halt (he fights back by himself). A lane's
-- trial ends fight_s after its first contact, after max_s, or when its lord falls below
-- min_hp; then its attackers go back to their park. A trial may run in some lanes only
-- (trial.lanes) and may send the other side's lord in as an attacker (kind 'lord': the
-- lane's rival, who then stands in no lane of his own). Between trials every unit is healed
-- (heal_hitpoints_unary) and its fatigue cut to a tenth (CA's script_unit:refresh); the
-- attacking units rotate so each fights few trials. Everyone is fearless (no routs).
-- Every tick_ms: per lane the lord's and each attacker's health, men, kills, damage dealt
-- (CCO DamageDealt: nil in this build, 01.10.2026), melee flag and place ('swarm_sample'); every soldier_ms the attackers' soldiers within
-- the radii of the lord, counted per unit, and those within near_m as offsets
-- ('swarm_men'; CCO ManList positions: full view, research telemetry).
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local map = require('apps.map.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER, MEN_TIMER = 'tww3_bai_lord_swarm_tick', 'tww3_bai_lord_swarm_men'
-- Where each side of the lord lies, degrees off his facing.
M.SIDES = {front = 0, right = 90, back = 180, left = 270}

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- Soldiers (flat x1, z1, x2, z2 ... in metres) within each radius of (x, z) and the offsets of
-- those within near_m (decimetres). Pure.
function M.count_near(xz, x, z, radii, near_m)
    local counts, near = {}, {}
    for k = 1, #radii do counts[k] = 0 end
    for i = 1, #xz - 1, 2 do
        local dx, dz = xz[i] - x, xz[i + 1] - z
        local d = math.sqrt(dx * dx + dz * dz)
        for k, r in ipairs(radii) do
            if d <= r then counts[k] = counts[k] + 1 end
        end
        if d <= near_m then
            near[#near + 1] = math.floor(dx * 10 + 0.5)
            near[#near + 1] = math.floor(dz * 10 + 0.5)
        end
    end
    return counts, near
end

-- The attackers of a trial in a lane: {{name, kind, side}}; spear units rotate through the
-- pool (rotation = how many spear places were handed out before); kind 'lord' is the other
-- side's lord (lane.rival). Pure.
function M.assign(trial, lane, rotation)
    local out, used = {}, rotation
    for _, place in ipairs(trial.attackers) do
        local name
        if place.kind == 'lord' then
            name = lane.rival
        elseif place.kind == 'ap' then
            name = lane.ap[1]
        else
            name = lane.spears[used % #lane.spears + 1]
            used = used + 1
        end
        out[#out + 1] = {name = name, kind = place.kind, side = place.side}
    end
    return out, used
end

-- Whether a trial runs in a lane (trial.lanes: the names it runs in; none = every lane). Pure.
function M.runs_in(trial, lane)
    if not trial.lanes then return true end
    for _, name in ipairs(trial.lanes) do
        if name == lane.name then return true end
    end
    return false
end

-- config: build, speed, tick_ms, soldier_ms, deadline_s, start_m, settle_ms, fight_s, max_s,
-- min_hp, radii, near_m, lanes = {{name, lord, rival, x, z, bearing, spears = {...}, ap = {...},
-- park = {x, z, bearing}}}, trials = {{name, lanes?, attackers = {{side, kind}}}}, widths = {name = m},
-- depths = {name = m}. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_lord_swarm then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', trial = 0,
        units = {}, lanes = {}, rotation = 0}
    _G.tww3_bai_lord_swarm = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'lord_swarm'
        row.policy = 'lord_swarm'
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function stop_timers()
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_process(MEN_TIMER) end)
    end
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

    local function unit_row(it)
        local u = it.unit
        local p = read(function() return u:position() end)
        return {n = it.name, men = read(function() return u:number_of_men_alive() end),
            hp = round(read(cco, u, 'HealthValue'), 0), hpu = round(read(function() return u:unary_hitpoints() end), 4),
            k = read(cco, u, 'NumKills'), dd = round(read(cco, u, 'DamageDealt'), 0),
            m = read(function() return u:is_in_melee() end), x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(read(function() return u:bearing() end), 0)}
    end

    -- Soldier positions of a unit in metres (flat), or nil.
    local function soldiers(u)
        local size = read(cco, u, 'ManList.Size')
        if type(size) ~= 'number' or size < 0 or size > 256 then return nil end
        local xz = {}
        for i = 0, size - 1 do
            local ok, x, _, z = pcall(cco, u, 'ManList.At(' .. i .. ').Position')
            if ok and type(x) == 'number' and type(z) == 'number' then
                xz[#xz + 1] = x
                xz[#xz + 1] = z
            end
        end
        return xz
    end

    local function place(it, x, z, bearing, width)
        orders.teleport(it.uc, vec(x, z), bearing, width)
        orders.halt(it.uc)
    end

    local function refresh(it)
        local hp = read(function() return it.unit:unary_hitpoints() end)
        if hp and hp < 1 then pcall(function() it.unit:heal_hitpoints_unary(1 - hp) end) end
        pcall(function() it.uc:change_fatigue_amount(0.1) end)
    end

    local function park(lane, it)
        local k = it.park_index or 0
        place(it, lane.park.x + 45 * k, lane.park.z, lane.park.bearing, config.widths[it.name] or 30)
    end

    local function sample()
        local rows = {}
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local r = {lane = lane.name, t = now_ms() - lane.t0, lord = unit_row(lane.lord_it), att = {}}
                for _, a in ipairs(lane.current) do r.att[#r.att + 1] = unit_row(state.units[a.name]) end
                rows[#rows + 1] = r
            end
        end
        return rows
    end

    local function men_sample()
        if not state.active then return end
        local rows = {}
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local p = read(function() return lane.lord_it.unit:position() end)
                if p then
                    local x, z = p:get_x(), p:get_z()
                    local r = {lane = lane.name, t = now_ms() - lane.t0, x = round(x), z = round(z), att = {}}
                    for _, a in ipairs(lane.current) do
                        local xz = soldiers(state.units[a.name].unit)
                        if xz then
                            local counts, near = M.count_near(xz, x, z, config.radii, config.near_m)
                            r.att[#r.att + 1] = {n = a.name, c = counts, near = near}
                        end
                    end
                    rows[#rows + 1] = r
                end
            end
        end
        emit('swarm_men', {trial = state.trial, radii = config.radii, lanes = rows})
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        stop_timers()
        if state.cancel_deadline then state.cancel_deadline() end
        emit('result', {status = status or 'completed', trials = state.trial, winner = 0})
        flush()
        bm:end_battle()
    end

    local begin_trial

    local function end_lane(lane, why)
        if not lane.running then return end
        lane.running = false
        emit('swarm_lane_end', {trial = state.trial, lane = lane.name, why = why,
            contact_ms = lane.contact and lane.contact - lane.t0, t = now_ms() - lane.t0,
            lord = unit_row(lane.lord_it)})
        for _, a in ipairs(lane.current) do park(lane, state.units[a.name]) end
    end

    local function tick()
        if not state.active then return end
        local rows = sample()
        if #rows > 0 then emit('swarm_sample', {trial = state.trial, lanes = rows}) end
        local busy = false
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local now = now_ms()
                local melee = read(function() return lane.lord_it.unit:is_in_melee() end)
                if not lane.contact and melee then lane.contact = now end
                local hp = read(function() return lane.lord_it.unit:unary_hitpoints() end) or 1
                if hp < config.min_hp then
                    end_lane(lane, 'low_hp')
                elseif lane.contact and now - lane.contact >= config.fight_s * 1000 then
                    end_lane(lane, 'fight_s')
                elseif now - lane.t0 >= config.max_s * 1000 then
                    end_lane(lane, 'max_s')
                end
                busy = busy or lane.running
            end
        end
        flush()
        -- Every lane done: the next trial (in_trial is set once its attack is ordered).
        if state.in_trial and not busy then
            state.in_trial = false
            bm:callback(guarded(begin_trial), config.settle_ms, 'tww3_bai_lord_swarm_next')
        end
    end

    -- Places the next trial's units and orders the attack settle_ms later.
    begin_trial = function()
        state.trial = state.trial + 1
        local trial = config.trials[state.trial]
        if not trial then return finish('completed') end
        local rotation = state.rotation
        for _, lane in ipairs(state.lanes) do
            lane.active = M.runs_in(trial, lane)
            local current = {}
            if lane.active then current, state.rotation = M.assign(trial, lane, rotation) end
            lane.current = current
        end
        for _, lane in ipairs(state.lanes) do
            if lane.active then
                refresh(lane.lord_it)
                place(lane.lord_it, lane.x, lane.z, lane.bearing, 5)
                for _, a in ipairs(lane.current) do
                    local it = state.units[a.name]
                    refresh(it)
                    local dir = math.rad(lane.bearing + M.SIDES[a.side])
                    local r = config.start_m + (config.depths[a.name] or 9) / 2
                    place(it, lane.x + r * math.sin(dir), lane.z + r * math.cos(dir),
                        (lane.bearing + M.SIDES[a.side] + 180) % 360, config.widths[a.name] or 30)
                end
            end
        end
        emit('swarm_trial', {trial = state.trial, name = trial.name,
            lanes = (function()
                local out = {}
                for _, lane in ipairs(state.lanes) do
                    if lane.active then out[#out + 1] = {lane = lane.name, attackers = lane.current} end
                end
                return out
            end)()})
        flush()
        bm:callback(guarded(function()
            for _, lane in ipairs(state.lanes) do
                lane.t0, lane.contact, lane.running = now_ms(), nil, lane.active
                if lane.active then
                    orders.halt(lane.lord_it.uc)
                    for _, a in ipairs(lane.current) do
                        orders.attack_melee(state.units[a.name].uc, lane.lord_it.unit)
                    end
                end
            end
            state.in_trial = true
        end), config.settle_ms, 'tww3_bai_lord_swarm_go')
    end

    local function start()
        if state.active or state.finished then return end
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_lord_swarm_speed')
        pcall(function() bm:change_victory_countdown_limit(-1) end)
        for _, lane in ipairs(state.lanes) do
            for _, name in ipairs(lane.spears) do park(lane, state.units[name]) end
            for _, name in ipairs(lane.ap) do park(lane, state.units[name]) end
        end
        emit('start', {speed = config.speed, trials = #config.trials, radii = config.radii})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_lord_swarm_deadline')
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        bm:repeat_callback(guarded(men_sample), config.soldier_ms, MEN_TIMER)
        bm:callback(guarded(begin_trial), config.settle_ms, 'tww3_bai_lord_swarm_first')
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
            local lane = {name = l.name, x = l.x, z = l.z, bearing = facing.snap(l.bearing), park = l.park,
                spears = l.spears, ap = l.ap, rival = l.rival, lord_it = state.units[l.lord], current = {}}
            assert(lane.lord_it, 'scenario unit missing: ' .. tostring(l.lord))
            for k, name in ipairs(l.spears) do
                assert(state.units[name], 'scenario unit missing: ' .. name)
                state.units[name].park_index = k - 1
            end
            for k, name in ipairs(l.ap) do
                assert(state.units[name], 'scenario unit missing: ' .. name)
                state.units[name].park_index = #l.spears + k - 1
            end
            if l.rival then
                assert(state.units[l.rival], 'scenario unit missing: ' .. l.rival)
                state.units[l.rival].park_index = #l.spears + #l.ap
            end
            state.lanes[i] = lane
        end
        state.batch = clock.batch_stamp() .. '-swarm-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {lanes = #state.lanes, trials = #config.trials})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_lord_swarm_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
