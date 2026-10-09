-- Entry: the melee probe (tools/nn/charge_probe.py). One indicator, one scenario, the game and the
-- simulator alike: lanes far apart, in each an attacker and a target placed gap_m apart (front to
-- front), the target facing the attacker (or away: target_mode 'rear'); everyone is held by script
-- and fearless, so the fight is the formula's alone (no routs, no AI choices).
-- The attacker's order (lane.mode):
--   'attack_run'   attack the target at a run (the charge: an attack order with a run-up);
--   'attack_walk'  attack it at a walk (no charge);
--   'move_run'     a move order at a run to a point move_beyond_m behind the target's front;
--   'recharge'     attack_run; recharge_after_s after the first contact a move back_m away at a run,
--                  then (arrived, or recharge_max_s later) attack_run again;
--   'hold'         no order;
--   'withdraw'     attack_run; recharge_after_s after the first contact a move back_m away at a run,
--                  never changed again (the melee exit: is the order kept, dropped after melee_breakoff_secs?);
--   'shoot'        a ranged attack on the target at a walk (fire at will on), never changed;
--   'script'       lane.steps {{at_s, kind = 'face' | 'move', bearing, width, dx, dz, run}}: at at_s after the
--                  go a 'face' (goto_location_angle_width at its place: turn in place to the world bearing) or a
--                  'move' (goto_location to its start + (dx, dz)); the turning tests;
--   'reengage'     attack_run (the target with target_mode 'both' too); part_after_s after the first contact both
--                  get a move part_m straight back at a run (probe_phase 'out'; 'clear' the first tick neither is in
--                  melee), part_s later both attack each other at a run ('back'); the first melee flag after 'clear'
--                  is contact 2; the lane ends after2_s after it (fight_s does not end it);
--   'tire'         no fight first: both units shuttle at a run straight back tire_leg_m and to their places
--                  until both are at least tire_until (a fatigue_state key) or tire_max_s after the go ('tire_end'),
--                  go back to their places facing each other, and once both are there (or ready_max_s later,
--                  'tired') both attack each other at a run; contacts count only from then.
-- lane.a_ability (optional): the attacker uses it on himself a_ability_after_s after the first contact
-- (Foe-Seeker's vigour). lane.men_all_s (optional): the attacker's soldier places every men_ms from the go
-- to men_all_s, wherever the enemy is (how a formation turns).
-- The target (lane.target_mode): 'stand' halts and, with lane.answer, is ordered to attack the
-- attacker at its first contact (both then fight under an attack order); 'hold' halts and is never
-- ordered (braced spears); 'both' attacks the attacker at a run from the start ('both_walk': at a walk); 'rear' halts facing
-- away. lane.lord (optional): a lord placed behind the target (dz m), who uses lane.lord.ability on
-- himself at the lane's first contact (Stand Your Ground) unless the key is empty (the control); with
-- lane.lord.at_m instead once the two units' centres are within at_m before the contact.
-- lane.target2 (optional): a second target unit placed t2_dx m beside the target (+x), facing the same way, halted
-- and never ordered (it fights only what touches it); sampled as 't2'. lane.after (optional) {at_s, kind, dx, dz,
-- walk}: at_s after the first contact the attacker gets one more order (the fresh-order probe): 'attack_t2' (attack
-- the second target), 'attack_same' (the same attack again), 'halt' (the bridge's hold), 'move_near' (a move to its
-- place + (dx, dz)), 'none' (the control); emitted as probe_phase 'after:<kind>'.
-- lane.t2_mode 'attack' (optional): the second target attacks the attacker at a walk at the go (two on one).
-- lane.damage (optional) {t = {method, share}, t2 = ..., a = ...}: once placed (settle_ms before the go) the unit is
-- brought down: method 'reduce' unit:reduce_hitpoints_unary(share), 'kill' unit:kill_number_of_men(floor(men x share))
-- (the engine's own calls); event probe_damage with the men and health before and after.
-- target_mode 'push': at the go the target gets a move order at a walk to a point push_m ahead of its front
-- (through the attacker: the game's own planner's far move point), never changed.
-- lane.t_morale (optional): the target keeps its morale (not fearless) - it can rout; its first rout is emitted as
-- probe_phase 'rout' and lane.at_rout says what the attacker does then: 'halt' (halts where it is) or 'none' (goes
-- on attacking: the chase); the lane ends lane.after_rout_s after the rout. With config.men_after_rout_s the
-- soldiers' places of both units are sampled every men_ms for that long after the rout wherever the enemy is (the
-- routing mob's shape); the sample rows carry r (routing) and sh (shattered) too. lane.rout_at_s (optional): a
-- t_morale target not routed that long after the first contact is routed by script (morale_behavior_rout; emitted as
-- probe_phase 'rout_forced'). lane.extras (optional) {{name, dx, dz, fire}}: units of the attacker's side placed
-- (dx, dz) from the target's centre facing the attacker's way, halted, fearless, never ordered; with fire, fire at will;
-- sampled as the list 'e' in probe_sample (unit rows) and probe_men (soldier places after the rout).
-- A t_morale target's sample row also carries its morale: mp (CCO MoralePercent), ms (MoraleState), mge
-- (MoraleGreatestEffect, the strongest effect's text), fx (CCO ActiveEffectList keys) and w (wavering).
-- lane.morale_rows (optional): both units' sample rows carry those morale fields (fearless or not; the dmgmelee plan).
-- at_rout 'away' (the rallysecure plan): at the rout the attacker is teleported to (x, z + away_dz) facing back and
-- halted (no chase, no enemy near; probe_phase 'away'). The target's first stop of routing after the rout (not
-- shattered) is its rally: probe_phase 'rally' (place, bearing, morale), the target halts and lane.rally_friends
-- {{name, dx, dz, width, px, pz}} - units of the target's side parked at (px, pz) from the start - are teleported to
-- dx m to its right and dz m ahead of its centre (its facing at the rally), facing its way (probe_phase 'friends');
-- sampled as the list 'f'. The lane ends after_rally_s after the rally (then after_rout_s no longer applies).
-- Every tick_ms 'probe_sample': per running lane both units' (and the lord's) men, health
-- (CCO HealthValue), melee flag, place, bearing, moving / moving fast, kills, fatigue, status keys
-- (CCO StatusList: braced, melee...). Every men_ms while the two are within men_near_m of each
-- other and up to men_after_s after a contact: both units' soldier places ('probe_men', decimetres;
-- CCO ManList, full view: research telemetry). A lane ends fight_s after its first contact, after
-- max_s, or when a unit is dead; then both halt. The battle ends when every lane is done.
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
local TIMER, MEN_TIMER = 'tww3_bai_charge_probe_tick', 'tww3_bai_charge_probe_men'
M.MODES = {attack_run = true, attack_walk = true, move_run = true, recharge = true, hold = true, withdraw = true,
    script = true, shoot = true, reengage = true, tire = true}
M.TARGET_MODES = {stand = true, hold = true, both = true, both_walk = true, rear = true, push = true}
M.AFTER_KINDS = {attack_t2 = true, attack_same = true, halt = true, move_near = true, none = true}
M.AT_ROUT = {halt = true, none = true, away = true}
-- The engine's fatigue states, fresh to exhausted (unit:fatigue_state()).
M.FATIGUE = {'threshold_fresh', 'threshold_active', 'threshold_winded', 'threshold_tired', 'threshold_very_tired',
    'threshold_exhausted'}

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- Where a lane puts its two units: {ax, az, ab, tx, tz, tb} (centres and bearings). The target's
-- front line is at z = lane.z; the target faces +z (bearing 0; 'rear': 180), the attacker stands
-- gap_m beyond that front, facing back. Depths are the blocks' (0 for a lone man). Pure.
function M.layout(lane)
    local td, ad = lane.t_depth or 0, lane.a_depth or 0
    local tb = lane.target_mode == 'rear' and 180 or 0
    return {ax = lane.x, az = lane.z + lane.gap_m + ad / 2, ab = 180, tx = lane.x, tz = lane.z - td / 2, tb = tb}
end

-- A point dx m to the right and dz m ahead of (x, z) for a unit facing the world bearing (degrees; 0 = +z, 90 = +x).
-- Pure. Returns x, z.
function M.beside(x, z, bearing, dx, dz)
    local r = math.rad(bearing or 0)
    return x + (dz or 0) * math.sin(r) + (dx or 0) * math.cos(r), z + (dz or 0) * math.cos(r) - (dx or 0) * math.sin(r)
end

-- A fatigue state's index (0 fresh .. 5 exhausted), nil for anything else. Pure.
function M.fatigue_level(state)
    for i, k in ipairs(M.FATIGUE) do
        if k == state then return i - 1 end
    end
    return nil
end

-- The tire shuttle's point for a unit placed at (x, z) facing bearing: odd legs tire_leg_m straight back, even legs
-- its place. Pure. Returns x, z.
function M.shuttle_point(x, z, bearing, leg, leg_m)
    if leg % 2 == 1 then return M.beside(x, z, bearing, 0, -leg_m) end
    return x, z
end

-- The reengage lane's phase after one tick (pure): 'in' -> 'out' part_after_s after the first contact -> 'back'
-- part_s after 'out'. Returns the new phase.
function M.reengage_step(phase, now_ms, contact_ms, out_ms, lane)
    if phase == 'in' and contact_ms and now_ms - contact_ms >= lane.part_after_s * 1000 then return 'out' end
    if phase == 'out' and out_ms and now_ms - out_ms >= lane.part_s * 1000 then return 'back' end
    return phase
end

-- The recharge's phase after one tick (pure): 'in' (attacking) -> 'out' recharge_after_s after the
-- first contact -> 'back' (attacking again) once out of melee and back_m away, or recharge_max_s
-- after leaving. Returns the new phase.
function M.recharge_step(phase, now_ms, contact_ms, out_ms, moved_m, in_melee, lane)
    if phase == 'in' and contact_ms and now_ms - contact_ms >= lane.recharge_after_s * 1000 then
        return 'out'
    end
    if phase == 'out' and lane.mode ~= 'withdraw' and ((not in_melee and moved_m >= lane.back_m - 5)
            or now_ms - out_ms >= lane.recharge_max_s * 1000) then
        return 'back'
    end
    return phase
end

-- config: build, speed, tick_ms, men_ms, men_near_m, men_after_s, deadline_s, settle_ms,
-- lanes = {{name, attacker, target, x, z, gap_m, a_depth, t_depth, a_width, t_width, mode,
-- target_mode, answer, fight_s, max_s, move_beyond_m, recharge_after_s, back_m, recharge_max_s,
-- lord = {name, ability, dz}}}, park = {{name, x, z, bearing}}. globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_charge_probe then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', units = {}, lanes = {}}
    _G.tww3_bai_charge_probe = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'charge_probe'
        row.policy = 'charge_probe'
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

    local function statuses(u)
        local n = read(cco, u, 'StatusList.Size')
        if type(n) ~= 'number' or n < 0 or n > 32 then return nil end
        local keys = {}
        for i = 0, n - 1 do
            local k = read(cco, u, 'StatusList.At(' .. i .. ').Key')
            if type(k) == 'string' then keys[#keys + 1] = k end
        end
        return table.concat(keys, ',')
    end

    local function unit_row(u)
        local p = read(function() return u:position() end)
        return {men = read(function() return u:number_of_men_alive() end),
            hp = round(read(cco, u, 'HealthValue'), 0), hpu = round(read(function() return u:unary_hitpoints() end), 4),
            m = read(function() return u:is_in_melee() end), x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(read(function() return u:bearing() end), 0),
            mv = read(function() return u:is_moving() end), fast = read(function() return u:is_moving_fast() end),
            k = read(cco, u, 'NumKills'), fat = read(function() return u:fatigue_state() end), st = statuses(u),
            uma = read(function() return u:is_under_missile_attack() end), cuma = read(cco, u, 'IsUnderMissileAttack'),
            r = read(function() return u:is_routing() end), sh = read(function() return u:is_shattered() end)}
    end

    -- A unit row with its morale (t_morale targets): MoralePercent, MoraleState, the strongest effect's text, the
    -- active effects' keys, wavering.
    local function morale_row(u)
        local row = unit_row(u)
        row.mp, row.ms = round(read(cco, u, 'MoralePercent'), 3), read(cco, u, 'MoraleState')
        row.w = read(function() return u:is_wavering() end)
        local effect = read(cco, u, 'MoraleGreatestEffect')
        if type(effect) == 'string' and effect ~= '' then row.mge = effect end
        local fx = services.active_effects(function(field) return read(cco, u, field) end)
        if fx and #fx > 0 then row.fx = table.concat(fx, ',') end
        return row
    end

    -- Soldier places of a unit in decimetres (flat x1, z1, x2, z2 ...), or nil.
    local function soldiers(u)
        local size = read(cco, u, 'ManList.Size')
        if type(size) ~= 'number' or size < 0 or size > 256 then return nil end
        local xz = {}
        for i = 0, size - 1 do
            local ok, x, _, z = pcall(cco, u, 'ManList.At(' .. i .. ').Position')
            if ok and type(x) == 'number' and type(z) == 'number' then
                xz[#xz + 1] = math.floor(x * 10 + 0.5)
                xz[#xz + 1] = math.floor(z * 10 + 0.5)
            end
        end
        return xz
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

    local function dist(a, b)
        local p, q = read(function() return a:position() end), read(function() return b:position() end)
        if not (p and q) then return nil end
        return math.sqrt((p:get_x() - q:get_x()) ^ 2 + (p:get_z() - q:get_z()) ^ 2)
    end

    local function beaten(u)
        return (read(function() return u:number_of_men_alive() end) or 0) == 0
    end

    local function end_lane(lane, why)
        if not lane.running then return end
        lane.running = false
        orders.halt(lane.a.uc)
        orders.halt(lane.t.uc)
        if lane.t2 then orders.halt(lane.t2.uc) end
        emit('probe_lane_end', {lane = lane.name, why = why, t = now_ms() - lane.t0,
            contact_ms = lane.contact and lane.contact - lane.t0,
            contact2_ms = lane.contact2 and lane.contact2 - lane.t0})
    end

    local function attack(lane, it, enemy, walk)
        orders.attack_melee(it.uc, enemy.unit, walk)
    end

    local function tick()
        if not state.active then return end
        local rows, busy, now = {}, false, now_ms()
        for _, lane in ipairs(state.lanes) do
            if lane.running then
                local a_m = read(function() return lane.a.unit:is_in_melee() end)
                local t_m = read(function() return lane.t.unit:is_in_melee() end)
                local fighting = lane.mode ~= 'tire' or lane.phase == 'fight'
                if not lane.contact and fighting and (a_m or t_m) then
                    lane.contact = now
                    emit('probe_contact', {lane = lane.name, t = now - lane.t0, n = 1})
                    if lane.answer and lane.target_mode ~= 'hold' and lane.target_mode ~= 'both'
                            and lane.target_mode ~= 'both_walk' then
                        attack(lane, lane.t, lane.a, false)
                    end
                    if lane.lord and lane.lord.ability and lane.lord.ability ~= '' and not lane.lord.at_m then
                        local it = state.units[lane.lord.name]
                        local ok, used = pcall(orders.use_ability_on_self, it.uc, it.unit, lane.lord.ability)
                        emit('probe_ability', {lane = lane.name, key = lane.lord.ability, t = now - lane.t0,
                            status = ok and (used and 'used' or 'not_ready') or 'failed'})
                    end
                end
                if lane.lord and lane.lord.at_m and not lane.lord_cast and not lane.contact then
                    local d = dist(lane.a.unit, lane.t.unit)
                    if d and d <= lane.lord.at_m then
                        lane.lord_cast = true
                        local it = state.units[lane.lord.name]
                        local ok, used = pcall(orders.use_ability_on_self, it.uc, it.unit, lane.lord.ability)
                        emit('probe_ability', {lane = lane.name, key = lane.lord.ability, t = now - lane.t0,
                            d = round(d), status = ok and (used and 'used' or 'not_ready') or 'failed'})
                    end
                end
                if lane.mode == 'recharge' or lane.mode == 'withdraw' then
                    local p = read(function() return lane.a.unit:position() end)
                    local moved = (p and lane.out_from) and
                        math.sqrt((p:get_x() - lane.out_from[1]) ^ 2 + (p:get_z() - lane.out_from[2]) ^ 2) or 0
                    local phase = M.recharge_step(lane.phase, now, lane.contact, lane.out_ms or now, moved, a_m, lane)
                    if phase ~= lane.phase then
                        lane.phase = phase
                        if phase == 'out' and p then
                            lane.out_ms, lane.out_from = now, {p:get_x(), p:get_z()}
                            local away = lane.layout.ab + 180   -- back the way it came
                            local r = math.rad(away)
                            orders.move(lane.a.uc, vec(p:get_x() + lane.back_m * math.sin(r),
                                p:get_z() + lane.back_m * math.cos(r)), true)
                        elseif phase == 'back' then
                            attack(lane, lane.a, lane.t, false)
                        end
                        emit('probe_phase', {lane = lane.name, phase = phase, t = now - lane.t0, moved_m = round(moved)})
                    end
                    if lane.phase == 'back' and not lane.contact2 and a_m then
                        lane.contact2 = now
                        emit('probe_contact', {lane = lane.name, t = now - lane.t0, n = 2})
                    end
                end
                if lane.mode == 'reengage' then
                    local phase = M.reengage_step(lane.phase, now, lane.contact, lane.out_ms, lane)
                    if phase ~= lane.phase then
                        lane.phase = phase
                        if phase == 'out' then
                            lane.out_ms = now
                            local moved = {}
                            for _, it in ipairs({{lane.a, lane.layout.ab}, {lane.t, lane.layout.tb}}) do
                                local p = read(function() return it[1].unit:position() end)
                                if p then
                                    local x, z = M.beside(p:get_x(), p:get_z(), it[2], 0, -lane.part_m)
                                    orders.move(it[1].uc, vec(x, z), true)
                                    moved[#moved + 1] = {round(x), round(z)}
                                end
                            end
                            emit('probe_phase', {lane = lane.name, phase = 'out', t = now - lane.t0, to = moved})
                        elseif phase == 'back' then
                            attack(lane, lane.a, lane.t, false)
                            attack(lane, lane.t, lane.a, lane.target_mode == 'both_walk')
                            emit('probe_phase', {lane = lane.name, phase = 'back', t = now - lane.t0,
                                cleared = lane.clear_ms ~= nil,
                                a_fat = read(function() return lane.a.unit:fatigue_state() end),
                                t_fat = read(function() return lane.t.unit:fatigue_state() end)})
                        end
                    end
                    if lane.phase == 'out' and not lane.clear_ms and not a_m and not t_m then
                        lane.clear_ms = now
                        emit('probe_phase', {lane = lane.name, phase = 'clear', t = now - lane.t0,
                            after_out_s = round((now - lane.out_ms) / 1000)})
                    end
                    if lane.phase == 'back' and lane.clear_ms and not lane.contact2 and (a_m or t_m) then
                        lane.contact2 = now
                        emit('probe_contact', {lane = lane.name, t = now - lane.t0, n = 2})
                    end
                end
                if lane.mode == 'tire' then
                    local L = lane.layout
                    local homes = {a = {L.ax, L.az, L.ab, lane.a_width}, t = {L.tx, L.tz, L.tb, lane.t_width}}
                    local fat = {a = read(function() return lane.a.unit:fatigue_state() end),
                        t = read(function() return lane.t.unit:fatigue_state() end)}
                    local function off(who, x, z)
                        local p = read(function() return lane[who].unit:position() end)
                        if not p then return math.huge end
                        return math.sqrt((p:get_x() - x) ^ 2 + (p:get_z() - z) ^ 2)
                    end
                    local function moving(who) return read(function() return lane[who].unit:is_moving() end) end
                    if lane.phase == 'tire' then
                        for _, who in ipairs({'a', 't'}) do
                            local h = homes[who]
                            local x, z = M.shuttle_point(h[1], h[2], h[3], lane.legs[who], lane.tire_leg_m)
                            -- the next leg once there, or once stopped 5 s into the leg (a formation may stop a few
                            -- metres off its point; standing would rest it)
                            if off(who, x, z) <= 8 or (not moving(who) and now - lane.leg_ms[who] >= 5000) then
                                lane.legs[who], lane.leg_ms[who] = lane.legs[who] + 1, now
                                x, z = M.shuttle_point(h[1], h[2], h[3], lane.legs[who], lane.tire_leg_m)
                                orders.move(lane[who].uc, vec(x, z), true)
                            end
                        end
                        local goal = M.fatigue_level(lane.tire_until)
                        local tired = (M.fatigue_level(fat.a) or -1) >= goal and (M.fatigue_level(fat.t) or -1) >= goal
                        if tired or now - lane.t0 >= lane.tire_max_s * 1000 then
                            lane.phase, lane.return_ms = 'return', now
                            for _, who in ipairs({'a', 't'}) do
                                local h = homes[who]
                                orders.move_formation(lane[who].uc, vec(h[1], h[2]), h[3], h[4], true)
                            end
                            emit('probe_phase', {lane = lane.name, phase = 'tire_end', t = now - lane.t0,
                                why = tired and 'tired' or 'tire_max_s', a_fat = fat.a, t_fat = fat.t,
                                legs = {lane.legs.a, lane.legs.t}})
                        end
                    elseif lane.phase == 'return' then
                        -- home: both stopped, at their places or 5 s after the order
                        local home = true
                        for _, who in ipairs({'a', 't'}) do
                            local h = homes[who]
                            if moving(who) or (off(who, h[1], h[2]) > 8 and now - lane.return_ms < 5000) then
                                home = false
                            end
                        end
                        if home or now - lane.return_ms >= lane.ready_max_s * 1000 then
                            lane.phase = 'fight'
                            attack(lane, lane.a, lane.t, false)
                            attack(lane, lane.t, lane.a, lane.target_mode == 'both_walk')
                            emit('probe_phase', {lane = lane.name, phase = 'tired', t = now - lane.t0,
                                why = home and 'home' or 'ready_max_s', a_fat = fat.a, t_fat = fat.t})
                        end
                    end
                end
                if lane.mode == 'script' then
                    for i, step in ipairs(lane.steps or {}) do
                        if not step.done and now - lane.t0 >= step.at_s * 1000 then
                            step.done = true
                            local p = read(function() return lane.a.unit:position() end)
                            if step.kind == 'face' and p then
                                orders.move_formation(lane.a.uc, vec(p:get_x(), p:get_z()), step.bearing,
                                    step.width or lane.a_width, step.run == true)
                            elseif step.kind == 'move' then
                                local L = lane.layout
                                orders.move(lane.a.uc, vec(L.ax + (step.dx or 0), L.az + (step.dz or 0)), step.run == true)
                            end
                            emit('probe_phase', {lane = lane.name, phase = 'step' .. i .. ':' .. tostring(step.kind),
                                t = now - lane.t0})
                        end
                    end
                end
                local af = lane.after
                if af and lane.contact and not lane.after_done and now - lane.contact >= af.at_s * 1000 then
                    lane.after_done = true
                    local walk = af.walk == true
                    if af.kind == 'attack_t2' then
                        attack(lane, lane.a, lane.t2, walk)
                    elseif af.kind == 'attack_same' then
                        attack(lane, lane.a, lane.t, walk)
                    elseif af.kind == 'halt' then
                        orders.halt(lane.a.uc)
                    elseif af.kind == 'move_near' then
                        local p = read(function() return lane.a.unit:position() end)
                        if p then orders.move(lane.a.uc, vec(p:get_x() + (af.dx or 0), p:get_z() + (af.dz or 0)), not walk) end
                    end
                    emit('probe_phase', {lane = lane.name, phase = 'after:' .. af.kind, t = now - lane.t0})
                end
                if lane.a_ability and lane.contact and not lane.a_ability_done
                        and now - lane.contact >= (lane.a_ability_after_s or 0) * 1000 then
                    lane.a_ability_done = true
                    local ok, used = pcall(orders.use_ability_on_self, lane.a.uc, lane.a.unit, lane.a_ability)
                    emit('probe_ability', {lane = lane.name, who = 'a', key = lane.a_ability, t = now - lane.t0,
                        status = ok and (used and 'used' or 'not_ready') or 'failed'})
                end
                if lane.t_morale and not lane.rout_ms and lane.rout_at_s and lane.contact and not lane.rout_forced
                        and now - lane.contact >= lane.rout_at_s * 1000 then
                    lane.rout_forced = true
                    local ok = pcall(function() lane.t.uc:morale_behavior_rout() end)
                    emit('probe_phase', {lane = lane.name, phase = 'rout_forced', t = now - lane.t0,
                        status = ok and 'done' or 'failed'})
                end
                if lane.t_morale and not lane.rout_ms then
                    local t_r = read(function() return lane.t.unit:is_routing() end)
                    if t_r then
                        lane.rout_ms = now
                        if lane.at_rout == 'halt' then orders.halt(lane.a.uc) end
                        emit('probe_phase', {lane = lane.name, phase = 'rout', t = now - lane.t0,
                            at_rout = lane.at_rout or 'none'})
                        if lane.at_rout == 'away' then
                            local ax, az = lane.x, lane.z + (lane.away_dz or 400)
                            local ok, e = pcall(place, lane.a, ax, az, 180, lane.a_width)
                            emit('probe_phase', {lane = lane.name, phase = 'away', t = now - lane.t0, x = round(ax),
                                z = round(az), status = ok and 'done' or 'failed', error = (not ok) and tostring(e) or nil})
                        end
                    end
                end
                if lane.after_rally_s and lane.rout_ms and not lane.rally_ms
                        and read(function() return lane.t.unit:is_routing() end) == false
                        and not read(function() return lane.t.unit:is_shattered() end) then
                    lane.rally_ms = now
                    orders.halt(lane.t.uc)
                    local p = read(function() return lane.t.unit:position() end)
                    local b = read(function() return lane.t.unit:bearing() end) or 0
                    emit('probe_phase', {lane = lane.name, phase = 'rally', t = now - lane.t0,
                        rout_s = round((now - lane.rout_ms) / 1000), x = p and round(p:get_x()), z = p and round(p:get_z()),
                        b = round(b, 0), mp = round(read(cco, lane.t.unit, 'MoralePercent'), 3)})
                    if p and lane.rally_friends and #lane.rally_friends > 0 then
                        local placed = {}
                        for _, f in ipairs(lane.rally_friends) do
                            local fx, fz = M.beside(p:get_x(), p:get_z(), b, f.dx, f.dz)
                            local ok = pcall(place, state.units[f.name], fx, fz, b, f.width or lane.t_width)
                            placed[#placed + 1] = {name = f.name, x = round(fx), z = round(fz), status = ok and 'done' or 'failed'}
                        end
                        emit('probe_phase', {lane = lane.name, phase = 'friends', t = now - lane.t0, units = placed})
                    end
                end
                local tg_row = (lane.t_morale or lane.morale_rows) and morale_row or unit_row
                local a_row = lane.morale_rows and morale_row or unit_row
                local r = {lane = lane.name, t = now - lane.t0, a = a_row(lane.a.unit), tg = tg_row(lane.t.unit)}
                if lane.rally_friends and #lane.rally_friends > 0 then
                    r.f = {}
                    for _, f in ipairs(lane.rally_friends) do r.f[#r.f + 1] = unit_row(state.units[f.name].unit) end
                end
                if lane.lord then r.l = unit_row(state.units[lane.lord.name].unit) end
                if lane.extras and #lane.extras > 0 then
                    r.e = {}
                    for _, e in ipairs(lane.extras) do r.e[#r.e + 1] = unit_row(state.units[e.name].unit) end
                end
                if lane.t2 then r.t2 = unit_row(lane.t2.unit) end
                rows[#rows + 1] = r
                if beaten(lane.a.unit) or beaten(lane.t.unit) then
                    end_lane(lane, 'dead')
                elseif lane.rally_ms and lane.after_rally_s and now - lane.rally_ms >= lane.after_rally_s * 1000 then
                    end_lane(lane, 'after_rally')
                elseif lane.rout_ms and not (lane.rally_ms and lane.after_rally_s)
                        and now - lane.rout_ms >= (lane.after_rout_s or 45) * 1000 then
                    end_lane(lane, 'after_rout')
                elseif lane.after2_s and lane.contact2 and now - lane.contact2 >= lane.after2_s * 1000 then
                    end_lane(lane, 'after_contact2')
                elseif lane.contact and not lane.after2_s and now - lane.contact >= lane.fight_s * 1000 then
                    end_lane(lane, 'fight_s')
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

    local function men_sample()
        if not state.active then return end
        local rows, now = {}, now_ms()
        for _, lane in ipairs(state.lanes) do
            local last = lane.contact2 or lane.contact
            if lane.running and lane.rout_ms and config.men_after_rout_s then
                -- the routing mob: both units' men for men_after_rout_s after the rout, wherever the enemy is
                if now - lane.rout_ms <= config.men_after_rout_s * 1000 then
                    local row = {lane = lane.name, t = now - lane.t0, a = soldiers(lane.a.unit), tg = soldiers(lane.t.unit)}
                    if lane.extras and #lane.extras > 0 then
                        row.e = {}
                        for _, e in ipairs(lane.extras) do row.e[#row.e + 1] = soldiers(state.units[e.name].unit) end
                    end
                    rows[#rows + 1] = row
                end
            elseif lane.running and lane.men_all_s then
                if now - lane.t0 <= lane.men_all_s * 1000 then
                    rows[#rows + 1] = {lane = lane.name, t = now - lane.t0, a = soldiers(lane.a.unit)}
                end
            elseif lane.running and lane.mode == 'tire' and lane.phase ~= 'fight' then
                -- no soldier places while the pair shuttles (only the fight after it)
            elseif lane.running and (not last or now - last <= config.men_after_s * 1000) then
                local d = dist(lane.a.unit, lane.t.unit)
                if d and d <= config.men_near_m then
                    rows[#rows + 1] = {lane = lane.name, t = now - lane.t0, a = soldiers(lane.a.unit),
                        tg = soldiers(lane.t.unit), t2 = lane.t2 and soldiers(lane.t2.unit) or nil}
                end
            end
        end
        if #rows > 0 then emit('probe_men', {lanes = rows}) end
    end

    local function go()
        for _, lane in ipairs(state.lanes) do
            lane.t0, lane.running, lane.phase = now_ms(), true, 'in'
            if lane.mode == 'tire' then
                -- the shuttle's first leg: straight back, both at a run
                lane.phase, lane.legs, lane.leg_ms = 'tire', {a = 1, t = 1}, {a = lane.t0, t = lane.t0}
                local L = lane.layout
                for who, h in pairs({a = {L.ax, L.az, L.ab}, t = {L.tx, L.tz, L.tb}}) do
                    local x, z = M.shuttle_point(h[1], h[2], h[3], 1, lane.tire_leg_m)
                    orders.move(lane[who].uc, vec(x, z), true)
                end
            elseif lane.target_mode == 'both' or lane.target_mode == 'both_walk' then
                attack(lane, lane.t, lane.a, lane.target_mode == 'both_walk')
            end
            if lane.t2 and lane.t2_mode == 'attack' then attack(lane, lane.t2, lane.a, true) end
            if lane.target_mode == 'push' then
                orders.move(lane.t.uc, vec(lane.layout.tx, lane.z + (lane.push_m or 60)), false)
            end
            local m = lane.mode
            if m == 'attack_run' or m == 'recharge' or m == 'withdraw' or m == 'reengage' then
                attack(lane, lane.a, lane.t, false)
            elseif m == 'attack_walk' then
                attack(lane, lane.a, lane.t, true)
            elseif m == 'shoot' then
                orders.set_fire_at_will(lane.a.uc, true)
                orders.attack_ranged(lane.a.uc, lane.t.unit, false, true)
            elseif m == 'move_run' then
                local L = lane.layout
                orders.move(lane.a.uc, vec(L.tx, lane.z - lane.move_beyond_m), true)
            end
        end
        emit('probe_go', {lanes = #state.lanes})
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        bm:repeat_callback(guarded(men_sample), config.men_ms, MEN_TIMER)
        flush()
    end

    local function start()
        if state.active or state.finished or state.started then return end
        state.started = true
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_charge_probe_speed')
        pcall(function() bm:change_victory_countdown_limit(-1) end)
        for _, p in ipairs(config.park or {}) do
            local it = state.units[p.name]
            if it then place(it, p.x, p.z, p.bearing, p.width or 5) end
        end
        for _, lane in ipairs(state.lanes) do
            local L = lane.layout
            place(lane.a, L.ax, L.az, L.ab, lane.a_width)
            place(lane.t, L.tx, L.tz, L.tb, lane.t_width)
            if lane.t2 then place(lane.t2, L.tx + (lane.t2_dx or 0), L.tz, L.tb, lane.t_width) end
            for who, d in pairs(lane.damage or {}) do
                local it = ({a = lane.a, t = lane.t, t2 = lane.t2})[who]
                if it then
                    local u = it.unit
                    local men0 = read(function() return u:number_of_men_alive() end)
                    local hp0 = read(function() return u:unary_hitpoints() end)
                    local ok, e = pcall(function()
                        if d.method == 'kill' then
                            u:kill_number_of_men(math.floor((men0 or 0) * d.share), false)
                        else
                            u:reduce_hitpoints_unary(d.share, false)
                        end
                    end)
                    emit('probe_damage', {lane = lane.name, who = who, method = d.method, share = d.share,
                        status = ok and 'done' or 'failed', error = (not ok) and tostring(e) or nil, men0 = men0,
                        hp0 = round(hp0, 4), men1 = read(function() return u:number_of_men_alive() end),
                        hp1 = round(read(function() return u:unary_hitpoints() end), 4)})
                end
            end
            if lane.lord then
                place(state.units[lane.lord.name], L.tx, lane.z - (lane.t_depth or 0) - lane.lord.dz, 0, 5)
            end
            for _, e in ipairs(lane.extras or {}) do
                local it = state.units[e.name]
                place(it, L.tx + (e.dx or 0), L.tz + (e.dz or 0), L.ab, lane.a_width)
                if e.fire then orders.set_fire_at_will(it.uc, true) end
            end
            for _, f in ipairs(lane.rally_friends or {}) do
                place(state.units[f.name], f.px, f.pz, 0, f.width or lane.t_width)
            end
        end
        emit('start', {speed = config.speed, lanes = config.lanes})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_charge_probe_deadline')
        bm:add_infotext('BAI: MELEE PROBE')
        bm:callback(guarded(go), config.settle_ms, 'tww3_bai_charge_probe_go')
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
        local keep_morale = {}
        for _, l in ipairs(config.lanes) do
            if l.t_morale then keep_morale[l.target] = true end
        end
        for side = 1, 2 do
            for _, u in ipairs(sides[side].units) do
                local name = u:name()
                local uc = orders.take_control(sides[side].army, u)
                orders.set_fire_at_will(uc, false)
                orders.halt(uc)
                if not keep_morale[name] then pcall(function() uc:morale_behavior_fearless() end) end
                state.units[name] = {name = name, unit = u, uc = uc, side = side}
            end
        end
        for i, l in ipairs(config.lanes) do
            assert(M.MODES[l.mode], 'unknown mode ' .. tostring(l.mode))
            assert(M.TARGET_MODES[l.target_mode], 'unknown target mode ' .. tostring(l.target_mode))
            local lane = {}
            for k, v in pairs(l) do lane[k] = v end
            lane.a, lane.t = state.units[l.attacker], state.units[l.target]
            assert(lane.a, 'scenario unit missing: ' .. tostring(l.attacker))
            assert(lane.t, 'scenario unit missing: ' .. tostring(l.target))
            if l.lord then assert(state.units[l.lord.name], 'scenario unit missing: ' .. tostring(l.lord.name)) end
            if l.target2 then
                lane.t2 = state.units[l.target2]
                assert(lane.t2, 'scenario unit missing: ' .. tostring(l.target2))
            end
            if l.after then assert(M.AFTER_KINDS[l.after.kind], 'unknown after kind ' .. tostring(l.after.kind)) end
            if l.at_rout then assert(M.AT_ROUT[l.at_rout], 'unknown at_rout ' .. tostring(l.at_rout)) end
            if l.mode == 'tire' then
                assert(M.fatigue_level(l.tire_until), 'unknown tire_until ' .. tostring(l.tire_until))
            end
            for _, e in ipairs(l.extras or {}) do
                assert(state.units[e.name], 'scenario unit missing: ' .. tostring(e.name))
            end
            for _, f in ipairs(l.rally_friends or {}) do
                assert(state.units[f.name], 'scenario unit missing: ' .. tostring(f.name))
            end
            lane.layout = M.layout(l)
            state.lanes[i] = lane
        end
        state.batch = clock.batch_stamp() .. '-probe-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {lanes = #state.lanes})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_charge_probe_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
