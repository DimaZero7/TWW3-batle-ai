-- Entry: the arena battle recorder (scenario tools/nn/scenario.py, config/nn/arena.json).
-- Records whole battles of the game's AI against itself as data for training a
-- neural network later. Side 2 is always the game's own AI; it attacks or
-- defends as the battle file sets it (the alliance that wins on timeout defends).
-- Side 1 (ours) is handed to CA's script AI planner:
--   own_ai = 'attack' — the planner attacks the enemy force (re-issued every
--            15 s, as generated battles do);
--   own_ai = 'defend' — the planner defends where the army stands;
--   own_ai = 'hold'   — no planner, no orders: our units stand where they are
--            (a still target for the game's AI; measurements, 30.09.2026);
--   own_ai = 'net'    — the network commands our units: the state goes to the
--            companion outside the game every decide_ms, its orders come back
--            (apps.bridge.adapter; the game's AI attacks);
--   own_ai = 'human'  — a human commands our units (the player's side); the script gives no
--            orders, records as for the others plus apps.telemetry.observer_adapter (ordered
--            bearing and width, idle, visibility, damage flags, effects, abilities, soldiers);
--   own_ai = 'scripted' — every unit of ours under script, one attack order on the nearest enemy
--            (the lord duel's control, tools/nn/lord_duel.py).
-- config.enemy_ai = 'scripted' takes side 2 from the game's AI the same way (the lord duel: the
-- other lord under one plain attack order); absent, side 2 is the game's AI. A scripted unit gets its
-- attack again only when the engine dropped it: not in melee, no target, for M.SCRIPTED_LOST_MS.
-- config.enemy_ai = 'companion' (with own_ai 'net'): side 2 under a script of the simulator in the
-- companion (config.enemy_script, e.g. ai_like; tools/nn/companion/script.py) through a second bridge
-- (apps.bridge.adapter) with its own files: the same state at the same decision, the same rules as
-- ours (routing units the game's, rallies, shooters' duty, stalls), its events 'en_*' for 'nn_*'
-- and its result fields 'en_*'. Its state carries config.units' slots and widths (layout).
-- With own_ai 'human' (tools.build human --enemy-ai net): a human plays side 1 and side 2 is under the companion
-- (config.enemy_script 'net': the network seeing the battle as side 2; or a script) - the enemy's bridge only.
-- config.free_speed = true (the human's battle): the battle starts at config.speed and the human may change the
-- speed and pause it (no speed guard); the bridge's decisions and polls run on battle time, so a pause stops them
-- and the decision interval stays decide_ms of battle time at any speed.
-- config.observe = true adds the human's observer (apps.telemetry.observer_adapter) to any mode, without
-- the soldiers; config.cards = true also records every unit's card on change (nn_card), both sides
-- (the lord against the game's AI: tools/nn/lord_ai.py).
-- Each side may have its own army (tools/nn/scenario.py: named arenas).
-- config.skirmish (with own_ai 'net'): 'off' turns the game's skirmish mode off for every unit a bridge commands
-- (ours, and the enemy's under a script), as the simulator has none (apps.bridge.adapter); absent or 'game': the
-- game's own setting. Every row of a unit that has the mode carries sk = is it on now (unit:is_behaviour_active).
-- Every tick 'nn_sample' records every unit of both sides (full view: trusted
-- research telemetry); 'nn_final' is the last such record.
-- A unit that routs and rallies is given back to the planner with its last
-- order ('rejoined'): without this it stood idle to the end (30.09.2026). A
-- unit idle under fire is kicked ('idle_kick': stage 1 back to the planner,
-- stage 2 a planner of its own that attacks).
-- The battle ends on the game's outcome, a stall, the timeout or the deadline: 'result'.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local planner = require('apps.orders.planner_adapter')
local map = require('apps.map.adapter')
local bridge = require('apps.bridge.adapter')
local observer = require('apps.telemetry.observer_adapter')
local orders = require('apps.orders.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_nn_arena_tick'
local DECIDE, POLL = 'tww3_bai_nn_arena_decide', 'tww3_bai_nn_arena_poll'
M.REISSUE_MS = 15000
M.OWN_AI = {attack = true, defend = true, hold = true, net = true, human = true, scripted = true}
M.ENEMY_AI = {scripted = true, companion = true}
M.SCRIPTED_LOST_MS = 3000
M.SKIRMISH = {game = true, off = true}

local function try(fn, ...)
    local ok, v = pcall(fn, ...)
    if ok then return v end
    return nil
end

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- config: build, speed, tick_ms, deadline_s, stall_ms, timeout_ms, own_ai ('attack' | 'defend' | 'hold' | 'net'),
-- units = {own = [...], enemy = [...]} (script names, slots, keys), defend_radius_m;
-- 'net' also: decide_ms, poll_ms, factions = {own, enemy}, enemy_role; 'human': soldiers_every (ticks);
-- enemy_ai: nil (the game's AI), 'scripted' or 'companion' (enemy_script); scripted_targets: nil (nearest) or
-- 'like' (lord on lord).
function M.main(bm, config, globals)
    if _G.tww3_bai_nn_arena then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        sides = {{}, {}}, by_uid = {}}
    _G.tww3_bai_nn_arena = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local started_ms, started_wall, last_reissue = 0, 0, 0

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'nn_arena'
        row.policy = 'nn_arena_' .. tostring(config.own_ai)
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function cleanup()
        state.active = false
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_process(DECIDE) end)
        pcall(function() bm:remove_process(POLL) end)
        pcall(function() bm:remove_process('tww3_bai_nn_arena_start') end)
    end
    local function fail(err)
        cleanup()
        state.finished = true
        pcall(emit, 'error', {message = tostring(err)})
        pcall(flush)
        pcall(function() bm:out('[TWW3 BAI] STOP: ' .. tostring(err)) end)
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
    local function name_of(u)
        if not u then return '' end
        local uid = try(function() return tostring(u:unique_ui_id()) end)
        local it = uid and state.by_uid[uid]
        return it and it.name or '?'
    end

    -- One unit, full state (research telemetry: both sides).
    local function sample(it)
        local u = it.unit
        local p = try(function() return u:position() end)
        local o = try(function() return u:ordered_position() end)
        local row = {n = it.name, x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(try(function() return u:bearing() end), 0),
            men = try(function() return u:number_of_men_alive() end),
            hp = round(try(function() return u:unary_hitpoints() end), 4),
            mp = round(try(cco, u, 'MoralePercent'), 3), ms = try(cco, u, 'MoraleState'),
            r = try(function() return u:is_routing() end), s = try(function() return u:is_shattered() end),
            w = try(function() return u:is_wavering() end), m = try(function() return u:is_in_melee() end),
            mv = try(function() return u:is_moving() end), f = try(function() return u:is_moving_fast() end),
            a = try(function() return u:ammo_left() end), fire = try(cco, u, 'IsFiringMissiles'),
            t = name_of(try(function() return u:current_target() end)),
            fat = try(function() return u:fatigue_state() end), k = try(cco, u, 'NumKills'),
            ox = o and round(o:get_x()), oz = o and round(o:get_z()),
            lf = try(function() return u:is_left_flank_threatened() end),
            rf = try(function() return u:is_right_flank_threatened() end),
            bf = try(function() return u:is_rear_flank_threatened() end)}
        if it.can_sk then row.sk = orders.skirmish_active(u) end   -- (false must stay false: not an and-or)
        return row
    end

    -- For the simulator's army collapse and morale rules (docs/en/training/measurements.md), in the
    -- recording only (not in the companion's state): sv = unit:strategic_value() (the game's strength
    -- estimate of the unit now), pcr / phr = CCO PercentCasualtiesRecently / PercentHpLostRecently
    -- (men / HP lost in the last 4 s), mge = CCO MoraleGreatestEffect (localised display text of the
    -- effect weighing most on morale now; absent when empty). A field that cannot be read is left out.
    local function morale_fields(it, row)
        local u = it.unit
        row.sv = round(try(function() return u:strategic_value() end), 1)
        row.pcr = round(try(cco, u, 'PercentCasualtiesRecently'), 4)
        row.phr = round(try(cco, u, 'PercentHpLostRecently'), 4)
        local effect = try(cco, u, 'MoraleGreatestEffect')
        if type(effect) == 'string' and effect ~= '' then row.mge = effect end
    end

    local function snapshot(event)
        local rows = {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do
                local row = sample(it)
                row.side = side
                morale_fields(it, row)
                if state.observer then state.observer.decorate(row) end
                rows[#rows + 1] = row
            end
        end
        -- bop: CCO BattleRoot.BalanceOfPowerPercent, the top bar for the player's alliance (bop_side).
        emit(event, {t = bm:time_elapsed_ms() - started_ms, units = rows, bop_side = state.player_side,
            bop = round(try(common and common.get_context_value, 'BattleRoot.BalanceOfPowerPercent'), 4)})
    end

    local function side_summary()
        local out = {}
        for side = 1, 2 do
            local men, hp, standing = 0, 0, 0
            for _, it in ipairs(state.sides[side]) do
                local u = it.unit
                local m = try(function() return u:number_of_men_alive() end) or 0
                men = men + m
                hp = hp + (try(function() return u:unary_hitpoints() end) or 0)
                if m > 0 and not try(function() return u:is_routing() end)
                    and not try(function() return u:is_shattered() end) then standing = standing + 1 end
            end
            out['side_' .. side .. '_men'], out['side_' .. side .. '_hp'] = men, round(hp, 3)
            out['side_' .. side .. '_standing_units'] = standing
        end
        return out
    end

    local function finish(status, winner)
        if state.finished then return end
        if state.cancel_deadline then state.cancel_deadline() end
        snapshot('nn_final')
        local row = side_summary()
        row.status, row.winner = status, winner or 0
        row.duration_model_ms = bm:time_elapsed_ms() - started_ms
        row.duration_wall_s = clock.elapsed_wall_seconds(started_wall)
        row.rejoined = state.planner and state.planner.rejoined or 0
        row.idle_kicks = state.planner and state.planner.kicks or 0
        if state.scripted then row.scripted_orders = state.scripted_orders or 0 end
        if state.net then
            state.net.finish()
            for k, v in pairs(state.net.stats()) do row[k] = v end
        end
        if state.enemy_net then
            state.enemy_net.finish()
            for k, v in pairs(state.enemy_net.stats()) do row['en_' .. k:gsub('^nn_', '')] = v end
        end
        emit('result', row)
        cleanup()
        state.finished = true
        flush()
    end

    -- Our units in the planner: rallied units back with its last order, idle ones under fire kicked
    -- (apps.orders.planner_adapter). Not in 'hold': nobody leads our side then.
    local function lead(now)
        local rallied = state.planner.check_rallies()
        if #rallied > 0 then
            local names = {}
            for k, i in ipairs(rallied) do names[k] = state.sides[1][i].name end
            emit('rejoined', {t = now - started_ms, units = names})
        end
        for _, k in ipairs(state.planner.check_idle(now)) do
            emit('idle_kick', {t = now - started_ms, unit = state.sides[1][k.unit].name, stage = k.stage})
        end
    end

    local scripted_check   -- below, with the scripted side's helpers

    local function tick()
        if not state.active then return end
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled', 0)
            bm:force_battle_end(0, 'stalled', true)
            return
        end
        if bm:battle_outcome_decided() then
            local winner = bm:victorious_alliance()
            if winner ~= 0 then finish('completed', winner) return end
        end
        local now = bm:time_elapsed_ms()
        if now - started_ms >= config.timeout_ms then
            finish('timeout', 0)
            bm:force_battle_end(0, 'timeout', true)
            return
        end
        if state.planner then lead(now) end
        if state.scripted then scripted_check(now) end
        if config.own_ai == 'attack' and now - last_reissue >= M.REISSUE_MS then
            last_reissue = now
            state.planner.attack()
        end
        state.ticks = state.ticks + 1
        snapshot('nn_sample')
        if state.observer then
            state.observer.changes()
            state.observer.soldiers()
        end
        flush()
    end

    -- What the network's companion reads: every unit, full view (its observation hides the rest),
    -- plus the unit key and v = visible to the other side.
    local function net_rows()
        local rows = {}
        for side = 1, 2 do
            local other = state.alliances[3 - side]
            for _, it in ipairs(state.sides[side]) do
                local row = sample(it)
                row.side, row.key = side, it.key
                row.v = try(function() return it.unit:is_visible_to_alliance(other) end)
                rows[#rows + 1] = row
            end
        end
        return rows
    end

    -- The rows of one decision, read once for both bridges (the enemy's under a script).
    local shared = {}
    local function shared_rows()
        local now = bm:time_elapsed_ms()
        if shared.ms ~= now then shared.ms, shared.rows = now, net_rows() end
        return shared.rows
    end

    -- The enemy side under a script in the companion: a second bridge, side 2 its own.
    local function enemy_start()
        local layout = {}
        for _, key in ipairs({'own', 'enemy'}) do
            for _, spec in ipairs(config.units[key]) do
                layout[#layout + 1] = {n = spec.script_name, slot = spec.slot, width = spec.width}
            end
        end
        state.enemy_net = bridge.start({army = state.enemy_army, own = state.sides[2], enemies = state.sides[1],
            vector = vec, rows = shared_rows, cco = cco, skirmish = config.skirmish,
            emit = function(event, fields) emit((event:gsub('^nn_', 'en_')), fields) end,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end,
            model_ms = function() return bm:time_elapsed_ms() end,
            state_file = bridge.ENEMY_STATE_FILE, orders_file = bridge.ENEMY_ORDERS_FILE,
            meta = {batch = state.batch, factions = config.factions, decide_ms = config.decide_ms,
                attacker = config.enemy_role == 'defend' and 1 or 2, control = 2, script = config.enemy_script,
                layout = layout}})
        emit('enemy_ai', {mode = 'companion', enemy_ai = config.enemy_ai, script = config.enemy_script,
            state_file = bridge.ENEMY_STATE_FILE, orders_file = bridge.ENEMY_ORDERS_FILE})
    end

    local function net_start()
        state.net = bridge.start({army = state.own_army, own = state.sides[1], enemies = state.sides[2],
            vector = vec, rows = shared_rows, emit = emit, cco = cco, skirmish = config.skirmish,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end,
            model_ms = function() return bm:time_elapsed_ms() end,
            meta = {batch = state.batch, factions = config.factions, decide_ms = config.decide_ms,
                attacker = config.enemy_role == 'defend' and 1 or 2}})
        emit('own_ai', {mode = 'net', own_ai = config.own_ai, decide_ms = config.decide_ms,
            poll_ms = config.poll_ms, state_file = bridge.STATE_FILE, orders_file = bridge.ORDERS_FILE})
        if config.enemy_ai == 'companion' then enemy_start() end
    end

    -- Both bridges at a decision, ours first (the same state for both).
    local function net_decide()
        if state.net then state.net.decide() end
        if state.enemy_net then state.enemy_net.decide() end
    end
    local function net_poll()
        if state.net then state.net.poll() end
        if state.enemy_net then state.enemy_net.poll() end
    end

    -- A scripted side: each unit under script attacks the nearest standing enemy ('scripted_order');
    -- config.scripted_targets 'like': a lord the nearest enemy lord, any other unit the nearest enemy
    -- that is not a lord (slot 'lord'), else the nearest enemy (the lord duel with escorts).
    local function nearest_enemy(side, it, wanted)
        local p = try(function() return it.unit:position() end)
        local best, best_d
        for _, e in ipairs(state.sides[3 - side]) do
            local q = try(function() return e.unit:position() end)
            local up = (try(function() return e.unit:number_of_men_alive() end) or 0) > 0
                and not try(function() return e.unit:is_shattered() end)
            if p and q and up and (not wanted or wanted(e)) then
                local d = (p:get_x() - q:get_x()) ^ 2 + (p:get_z() - q:get_z()) ^ 2
                if not best_d or d < best_d then best, best_d = e, d end
            end
        end
        return best
    end
    local function scripted_attack(side, it, why)
        local foe
        if config.scripted_targets == 'like' then
            local lord = it.slot == 'lord'
            foe = nearest_enemy(side, it, function(e) return (e.slot == 'lord') == lord end)
        end
        foe = foe or nearest_enemy(side, it)
        if not foe then return end
        orders.attack_melee(it.uc, foe.unit)
        it.lost_since = nil
        state.scripted_orders = (state.scripted_orders or 0) + 1
        emit('scripted_order', {t = bm:time_elapsed_ms() - started_ms, u = it.name, tg = foe.name, why = why})
    end
    local function scripted_start(side)
        local army = side == 1 and state.own_army or state.enemy_army
        for _, it in ipairs(state.sides[side]) do
            it.uc = orders.take_control(army, it.unit)
            scripted_attack(side, it, 'start')
        end
        state.scripted = state.scripted or {}
        state.scripted[#state.scripted + 1] = side
        emit('scripted_side', {side = side})
    end
    scripted_check = function(now)
        for _, side in ipairs(state.scripted or {}) do
            for _, it in ipairs(state.sides[side]) do
                local u = it.unit
                local standing = (try(function() return u:number_of_men_alive() end) or 0) > 0
                    and not try(function() return u:is_routing() end)
                local lost = standing and not try(function() return u:is_in_melee() end)
                    and name_of(try(function() return u:current_target() end)) == ''
                if not lost then
                    it.lost_since = nil
                elseif not it.lost_since then
                    it.lost_since = now
                elseif now - it.lost_since >= M.SCRIPTED_LOST_MS then
                    scripted_attack(side, it, 'lost')
                end
            end
        end
    end

    -- The observer over every unit of both sides (the human's battle, or config.observe).
    local function observe(soldiers_every)
        if state.observer then return end
        local units = {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do units[#units + 1] = it end
        end
        state.observer = observer.new({units = units, alliances = state.alliances, cco = cco, emit = emit,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end,
            soldiers_every = soldiers_every, cards = config.cards == true})
    end

    local function hand_over()
        local own_units, enemy_units = {}, {}
        for _, it in ipairs(state.sides[1]) do own_units[#own_units + 1] = it.unit end
        for _, it in ipairs(state.sides[2]) do enemy_units[#enemy_units + 1] = it.unit end
        if config.own_ai == 'hold' then
            emit('own_ai', {mode = 'hold', own_ai = config.own_ai})
            return
        end
        if config.own_ai == 'human' then
            observe(config.soldiers_every)
            emit('own_ai', {mode = 'human', own_ai = config.own_ai, soldiers_every = config.soldiers_every})
            if config.enemy_ai == 'companion' then enemy_start() end
            return
        end
        if config.own_ai == 'net' then return net_start() end
        if config.own_ai == 'scripted' then
            scripted_start(1)
            emit('own_ai', {mode = 'scripted', own_ai = config.own_ai})
            return
        end
        state.planner = planner.hand_over(bm, 'tww3_bai_nn_own', state.own_alliance, own_units, enemy_units)
        if config.own_ai == 'defend' then
            local sx, sz = 0, 0
            for _, u in ipairs(own_units) do
                local p = u:position()
                sx, sz = sx + p:get_x(), sz + p:get_z()
            end
            state.planner.defend(vec(sx / #own_units, sz / #own_units), config.defend_radius_m)
        else
            state.planner.attack()
        end
        emit('own_ai', {mode = state.planner.mode, own_ai = config.own_ai})
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        bm:modify_battle_speed(config.speed)
        if not config.free_speed then
            battle.speed_guard(bm, config.speed, function(from)
                emit('speed_restored', {from_speed = from, to_speed = config.speed})
            end)
        end
        hand_over()
        if config.enemy_ai == 'scripted' then scripted_start(2) end
        if config.observe then observe(0) end
        if state.observer then state.observer.changes() end
        last_reissue = bm:time_elapsed_ms()
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        emit('start', {speed = config.speed, timeout_ms = config.timeout_ms, deadline_s = config.deadline_s,
            free_speed = config.free_speed == true})
        snapshot('nn_sample')
        flush()
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline', 0)
            bm:force_battle_end(0, 'deadline', true)
        end), 'tww3_bai_nn_arena_deadline')
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        if state.net or state.enemy_net then
            net_decide()
            bm:repeat_callback(guarded(function() if state.active then net_decide() end end),
                config.decide_ms, DECIDE)
            bm:repeat_callback(guarded(function() if state.active then net_poll() end end),
                config.poll_ms, POLL)
        end
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
        assert(M.OWN_AI[config.own_ai], 'own_ai must be attack, defend, hold, net, human or scripted')
        assert(config.enemy_ai == nil or M.ENEMY_AI[config.enemy_ai], 'enemy_ai must be absent, scripted or companion')
        assert(config.enemy_ai ~= 'companion' or config.own_ai == 'net' or config.own_ai == 'human',
            'enemy_ai companion needs own_ai net or human')
        assert(config.skirmish == nil or M.SKIRMISH[config.skirmish], 'skirmish must be absent, game or off')
        local sides = battle.read_sides(bm)
        state.own_alliance, state.own_army, state.enemy_army = sides[1].alliance, sides[1].army, sides[2].army
        state.alliances = {sides[1].alliance, sides[2].alliance}
        -- The side whose balance of power the top bar shows (sides are bm:alliances() in order).
        local player = try(function() return bm:get_player_alliance_num() end)
        if player == 1 or player == 2 then state.player_side = player end
        local ok_roles, roles = pcall(battle.read_roles, bm)
        state.all_units = {}
        for side, key in ipairs({'own', 'enemy'}) do
            for _, spec in ipairs(config.units[key]) do
                local u = battle.find_by_name(sides[side], spec.script_name)
                assert(u, 'scenario unit missing: ' .. spec.script_name)
                local it = {name = spec.script_name, slot = spec.slot, key = spec.key, unit = u, side = side,
                    can_sk = try(function() return u:can_use_behaviour('skirmish') end) == true}
                state.sides[side][#state.sides[side] + 1] = it
                state.by_uid[tostring(u:unique_ui_id())] = it
                state.all_units[#state.all_units + 1] = u
            end
        end
        state.batch = clock.batch_stamp() .. '-nn-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {own = #state.sides[1], enemy = #state.sides[2], own_ai = config.own_ai,
            roles = ok_roles and roles or tostring(roles), player_side = state.player_side,
            player_alliance_cco = try(common.get_context_value, 'BattleRoot.PlayerAllianceContext.Id')})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(tick))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active then
                tick()
                if not state.finished then finish('incomplete', 0) end
            end
        end))
        local function deployment()
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_nn_arena_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
