-- Entry: does an army lose morale when its lord is KILLED, against when he ROUTS? (user, 05.10.2026:
-- a killed lord shocks the army, Skaven most, and Vampire Counts crumble; the 75 recorded lord
-- falls were all routs, and the database has general_died_recently -16 / general_dead -10 and
-- general_fled_recently -16.) Research probe, every unit held by script (tools/nn/lord_fall.py).
-- One army is treated (config.treated_side); the other side is fearless, so its routs never lift
-- the treated army (routing enemies +2.5 each). The treated army: its lord behind its line, two
-- infantry units ordered into melee against two of the enemy's ('fight', within the lord's aura),
-- two standing idle far from every enemy and outside the aura ('idle': the shock alone).
-- At treat_after_ms after the first contact (at the latest treat_latest_ms after the start) the
-- treated lord is killed ('kill': reduce_hitpoints_unary(1), then uc:kill() if he still stands
-- 1 s later), routed ('rout': morale_behavior_rout) or left alone ('none': the control; the same
-- moment is marked). observe_ms later the battle ends.
-- Every tick_ms 'fall_sample': every unit's place, men, health, CCO MoralePercent / MoraleState /
-- MoraleGreatestEffect / IsAlive / PercentCasualtiesRecently / PercentHpLostRecently, routing,
-- shattered, wavering, melee. 'lord_fall' marks the moment (and the lord's row then).
local battle = require('apps.battle.adapter')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local orders = require('apps.orders.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_lord_fall_tick'
M.TREATMENTS = {kill = true, rout = true, none = true}
-- A fight unit out of melee this long gets its attack order again (the engine may drop it).
M.REISSUE_MS = 5000

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- When the treatment is due: contact_ms (first melee, or nil) + after, at the latest start + latest. Pure.
function M.due(now, start, contact, after, latest)
    if contact and now - contact >= after then return true end
    return now - start >= latest
end

-- config: build, speed, tick_ms, deadline_s, treated_side (1 | 2), treatment ('kill' | 'rout' | 'none'),
-- lords = {own, enemy} (script names), fight = {{own, enemy}, ...} (the pairs ordered into melee),
-- idle = {names}, treat_after_ms, treat_latest_ms, observe_ms, factions = {own, enemy}.
-- globals: common, battle_vector.
function M.main(bm, config, globals)
    if _G.tww3_bai_lord_fall then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', units = {}, order = {}}
    _G.tww3_bai_lord_fall = state
    local common = globals and globals.common
    local started = 0

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'lord_fall'
        row.policy = 'lord_fall_' .. tostring(config.treatment)
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
    local function now_ms() return bm:time_elapsed_ms() - started end

    local function row_of(it)
        local u = it.unit
        local p = read(function() return u:position() end)
        local row = {n = it.name, side = it.side, role = it.role,
            x = p and round(p:get_x()), z = p and round(p:get_z()),
            men = read(function() return u:number_of_men_alive() end),
            hp = round(read(function() return u:unary_hitpoints() end), 4),
            mp = round(read(cco, u, 'MoralePercent'), 3), ms = read(cco, u, 'MoraleState'),
            alive = read(cco, u, 'IsAlive'),
            r = read(function() return u:is_routing() end), s = read(function() return u:is_shattered() end),
            w = read(function() return u:is_wavering() end), m = read(function() return u:is_in_melee() end),
            pcr = round(read(cco, u, 'PercentCasualtiesRecently'), 4),
            phr = round(read(cco, u, 'PercentHpLostRecently'), 4)}
        local effect = read(cco, u, 'MoraleGreatestEffect')
        if type(effect) == 'string' and effect ~= '' then row.mge = effect end
        return row
    end

    local function sample()
        local rows = {}
        for _, name in ipairs(state.order) do rows[#rows + 1] = row_of(state.units[name]) end
        return rows
    end

    local function finish(status)
        if state.finished then return end
        state.active = false
        state.finished = true
        pcall(function() bm:remove_process(TIMER) end)
        if state.cancel_deadline then state.cancel_deadline() end
        emit('fall_final', {t = now_ms(), units = sample()})
        emit('result', {status = status or 'completed', winner = 0, treatment = config.treatment,
            treated_side = config.treated_side, treated_ms = state.treated_ms, contact_ms = state.contact_ms,
            lord_dead = state.lord_dead, reissued = state.reissued or 0})
        flush()
        bm:end_battle()
    end

    local function treated_lord()
        return state.units[config.treated_side == 1 and config.lords.own or config.lords.enemy]
    end

    local function treat()
        state.treated_ms = now_ms()
        local lord = treated_lord()
        local before = row_of(lord)
        local method, err = 'none', nil
        if config.treatment == 'kill' then
            method = 'reduce_hitpoints_unary'
            local ok, e = pcall(function() lord.unit:reduce_hitpoints_unary(1, false) end)
            if not ok then err = tostring(e) end
        elseif config.treatment == 'rout' then
            method = 'morale_behavior_rout'
            local ok, e = pcall(function() lord.uc:morale_behavior_rout() end)
            if not ok then err = tostring(e) end
        end
        emit('lord_fall', {t = state.treated_ms, treatment = config.treatment, side = config.treated_side,
            lord = lord.name, method = method, error = err, contact_ms = state.contact_ms, before = before})
    end

    -- kill: the lord still standing 1 s after the damage dies by uc:kill() (CA's script_unit:kill).
    local function confirm_kill(now)
        if config.treatment ~= 'kill' or state.lord_dead ~= nil or now - state.treated_ms < 1000 then return end
        local lord = treated_lord()
        local men = read(function() return lord.unit:number_of_men_alive() end) or 0
        if men > 0 then
            local ok, e = pcall(function() lord.uc:kill() end)
            emit('lord_fall_fallback', {t = now, method = 'uc_kill', ok = ok, error = not ok and tostring(e) or nil})
            state.lord_dead = 'uc_kill'
        else
            state.lord_dead = 'damage'
        end
    end

    local function keep_fighting(now)
        for _, pair in ipairs(config.fight) do
            for k = 1, 2 do
                local it, foe = state.units[pair[k]], state.units[pair[3 - k]]
                local melee = read(function() return it.unit:is_in_melee() end)
                local down = read(function() return it.unit:is_routing() end)
                    or (read(function() return foe.unit:number_of_men_alive() end) or 0) == 0
                if melee or down then
                    it.out_since = nil
                elseif not it.out_since then
                    it.out_since = now
                elseif now - it.out_since >= M.REISSUE_MS then
                    orders.attack_melee(it.uc, foe.unit)
                    it.out_since = now
                    state.reissued = (state.reissued or 0) + 1
                    emit('fall_reissue', {t = now, u = it.name, tg = foe.name})
                end
            end
        end
    end

    local function tick()
        if not state.active then return end
        local now = now_ms()
        emit('fall_sample', {t = now, treated = state.treated_ms ~= nil, units = sample()})
        if not state.contact_ms then
            for _, pair in ipairs(config.fight) do
                for _, name in ipairs(pair) do
                    if read(function() return state.units[name].unit:is_in_melee() end) then state.contact_ms = now end
                end
            end
            if state.contact_ms then emit('contact', {t = now}) end
        end
        if not state.treated_ms then
            if M.due(now, 0, state.contact_ms, config.treat_after_ms, config.treat_latest_ms) then treat() end
        else
            confirm_kill(now)
            if now - state.treated_ms >= config.observe_ms then return finish('completed') end
        end
        keep_fighting(now)
        flush()
    end

    local function start()
        if state.active or state.finished then return end
        started = bm:time_elapsed_ms()
        bm:modify_battle_speed(config.speed)
        battle.speed_guard(bm, config.speed, function(from)
            emit('speed_restored', {from_speed = from, to_speed = config.speed})
        end, 'tww3_bai_lord_fall_speed')
        pcall(function() bm:change_victory_countdown_limit(-1) end)
        for _, pair in ipairs(config.fight) do
            orders.attack_melee(state.units[pair[1]].uc, state.units[pair[2]].unit)
            orders.attack_melee(state.units[pair[2]].uc, state.units[pair[1]].unit)
        end
        emit('start', {speed = config.speed, treatment = config.treatment, treated_side = config.treated_side,
            factions = config.factions})
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline')
        end), 'tww3_bai_lord_fall_deadline')
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
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
        assert(common, 'common global required')
        assert(M.TREATMENTS[config.treatment], 'treatment must be kill, rout or none')
        assert(config.treated_side == 1 or config.treated_side == 2, 'treated_side must be 1 or 2')
        local roles = {}
        roles[config.lords.own], roles[config.lords.enemy] = 'lord', 'lord'
        for _, pair in ipairs(config.fight) do roles[pair[1]], roles[pair[2]] = 'fight', 'fight' end
        for _, name in ipairs(config.idle) do roles[name] = 'idle' end
        local sides = battle.read_sides(bm)
        for side = 1, 2 do
            for _, u in ipairs(sides[side].units) do
                local name = u:name()
                local uc = orders.take_control(sides[side].army, u)
                orders.set_fire_at_will(uc, false)
                orders.halt(uc)
                if side ~= config.treated_side then pcall(function() uc:morale_behavior_fearless() end) end
                state.units[name] = {name = name, unit = u, uc = uc, side = side, role = roles[name] or 'other'}
                state.order[#state.order + 1] = name
            end
        end
        for name in pairs(roles) do assert(state.units[name], 'scenario unit missing: ' .. name) end
        state.batch = clock.batch_stamp() .. '-fall-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {units = #state.order, treatment = config.treatment, treated_side = config.treated_side})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_lord_fall_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
