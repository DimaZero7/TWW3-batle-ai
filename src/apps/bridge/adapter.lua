-- The bridge to the network: our side is commanded by the companion
-- (tools/nn/companion), a program outside the game (docs/en/apps/bridge.md).
--   decide()  every decision tick: writes the state of every unit with the next
--             move number;
--   poll()    often: reads the companion's answer and gives each of our units
--             its order through apps.orders.adapter (only orders that changed;
--             'keep' gives nothing: the order in force goes on);
--   finish()  writes the last state with done = true.
-- A unit that routs loses its order in the engine; the simulator keeps it and the unit goes on
-- with it the moment it rallies. So the bridge keeps the order a unit had when it broke and gives
-- it again at once when the unit stands again (every poll; event nn_rally; an attack on a target
-- that is gone becomes hold, as in the simulator). An answer to a state in which the unit was down
-- (routing, shattered, dead) carries no order for it: the network gives such units HOLD only as a
-- filler, and given to a unit that rallied meanwhile it halted it (gate 02.10.2026, 42 of 102 rallies).
-- A shooter under an attack order that stands idle (its target in melee, out of
-- sight) is released to fire at will and takes its target again later; an attack
-- on a target already in melee within range is given as fire at will at once
-- (services.missile_duty, services.fire_freely, event nn_duty).
-- A held shooter shoots as the simulator's does: once per decision the bridge picks the nearest
-- enemy in range (services.hold_target) and gives it as an attack's target (walk), under the same
-- duty; a held shooter that walks is halted to fire at will again (services.hold_guard; event nn_hold).
-- A unit that stands under an attack or a move the engine dropped (after a fight it was drawn
-- into) is given that order again (services.order_stalled; event nn_stall).
-- Any order restarts a shooter's aim in the game, so a shooter already firing at the unit it is to
-- shoot (services.on_target) is not given it again: an attack (or a resume, a held shooter's pick) on
-- that unit is taken as given (status kept, event nn_aim_kept), a hold keeps the engine's target as
-- the held shooter's pick instead of halting it (nn_hold keep), and a held shooter with no pick takes the
-- unit it is firing at if that qualifies. Once the engine is off that target (another target, no fire
-- for FIRE_RECENT decisions) the order is given for real (nn_duty engine_off). A shooter shooting
-- standing (services.firing) that is to fire at will (its target in melee) is not halted for it
-- (nn_duty free_kept).
-- An idle shooter without a valid engine target is explicitly re-aimed within REAIM_AFTER
-- decisions, even if the enemy is still in melee (nn_reaim). Empty shooters under ATTACK
-- leave ranged duty immediately and receive a melee attack (nn_empty_melee).
-- No answer by the next decision tick: the orders in force stay (event nn_miss).
-- Abilities: an 'ability <unit> <key>' line of the answer is used at once when the unit
-- stands and can_perform_special_ability(key) says yes: perform_special_ability(key, the
-- unit itself) (self-cast; the recipe verified in battle, docs/en/game/units/commands.md).
-- In the game can_perform_special_ability says the lord owns it, ready or not (01.10.2026):
-- readiness is the companion's (its count and the card's active effects).
-- Each request is logged (event nn_ability: used, not_ready, down, unknown_unit, error); the
-- state document carries each ability's last use (abilities_used) and every unit's active
-- effects (row fx, when opts.cco reads the unit's card), so the companion knows the timers.
-- Changes only are logged once per decision: nn_ability_ready (an own unit's active ability:
-- can_perform_special_ability turned true or false) and nn_effects (a unit's active phases).
-- The Lua side writes everything it reads; what a human would not see is hidden
-- by the companion's observation (tools/nn/model/observation.py).
local json = require('apps.core.json')
local orders = require('apps.orders.adapter')
local exchange = require('apps.bridge.exchange_adapter')
local services = require('apps.bridge.services')

local M = {}

M.STATE_FILE = 'tww3_bai_nn_state.json'
M.ORDERS_FILE = 'tww3_bai_nn_orders.txt'
M.KEEP_TIMES = 20   -- moves whose write time is kept to measure the answer's wait

local function read(fn)
    local ok, v = pcall(fn)
    if ok then return v end
    return nil
end

local function clock()
    if type(os.clock) ~= 'function' then return nil end
    return read(os.clock)
end

local function standing(u)
    return (read(function() return u:number_of_men_alive() end) or 0) > 0
        and read(function() return u:is_routing() end) ~= true
        and read(function() return u:is_shattered() end) ~= true
end

local function shooter(u)
    return (read(function() return u:ammo_left() end) or 0) > 0
        and (read(function() return u:missile_range() end) or 0) > 0
end

local function empty_shooter(u)
    return read(function() return u:ammo_left() end) == 0
        and (read(function() return u:missile_range() end) or 0) > 0
end

-- opts: army (our engine army), own / enemies = {{name, unit}}, vector(x, z) -> engine
-- vector, rows() -> every unit's row, meta (put into every state: batch...), emit(event,
-- fields), now_ms() (battle time since the start), model_ms() (engine time); optional
-- cco(unit, field) -> the unit's CcoBattleUnit value (active effects into the rows).
function M.start(opts)
    local handle = {written = 0, applied = 0, answered = 0, missed = 0, given = 0, keeps = 0, bad = 0,
        regiven = 0, released = 0, resumed = 0, write_mode = nil, abilities_used = 0, abilities_refused = 0,
        hold_aims = 0, hold_halts = 0, stalls = 0, aims_kept = 0, reaims = 0, empty_melees = 0}
    -- current: the order in force; lost: the order a unit had when it broke (given again the moment
    -- it rallies); down_at: the last move whose state showed the unit down (its answer has no order
    -- for it); duty: a shooter's state under an attack order (or a held shooter's picked target);
    -- guard: a held unit's {pick, walk, block} (services.hold_target); stall: a unit's count of
    -- decisions standing under an order the engine dropped (services.order_stalled).
    local times, current, lost, duty, enemy_by_name, guard, stall = {}, {}, {}, {}, {}, {}, {}
    local down_at = {}
    local reaim = {}   -- targetless decisions, independent of network target changes / ranged duty
    local seen, seen_rows = {}, {}   -- the rows of the last decision, by unit name and as a list
    local fired = {}   -- the last move whose state showed the unit firing
    local used, own_by_name, unit_by_name = {}, {}, {}   -- used[name][key] = battle ms of the last use
    for _, it in ipairs(opts.enemies) do enemy_by_name[it.name], unit_by_name[it.name] = it, it.unit end
    for _, it in ipairs(opts.own) do own_by_name[it.name], unit_by_name[it.name] = it, it.unit end
    -- Own units' active abilities (as the card lists them) and what was last logged.
    local owned, ready_was, fx_was = {}, {}, {}
    for _, it in ipairs(opts.own) do
        local list = read(function() return it.unit:owned_non_passive_special_abilities() end)
        if type(list) == 'table' and #list > 0 then
            owned[it.name] = {}
            for _, k in ipairs(list) do owned[it.name][#owned[it.name] + 1] = tostring(k) end
        end
    end
    exchange.remove(M.STATE_FILE)
    exchange.remove(M.ORDERS_FILE)
    for _, it in ipairs(opts.own) do
        it.uc = orders.take_control(opts.army, it.unit)
        orders.set_fire_at_will(it.uc, true)
    end

    -- A shooter fires at will instead of holding a target it cannot hit (services.missile_duty).
    local function free_fire(it, target, action)
        local d = duty[it.name]
        orders.halt(it.uc)
        orders.set_fire_at_will(it.uc, true)
        d.free, d.free_for, d.idle = true, 0, 0
        handle.released = handle.released + 1
        opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = action, tg = target})
    end

    local function recent(name)
        return fired[name] ~= nil and handle.written - fired[name] <= services.FIRE_RECENT
    end

    -- Is the shooter firing at `target` already (services.on_target)?
    local function on_target(it, target)
        return services.on_target(seen[it.name], seen[target], recent(it.name))
    end

    -- A shooter takes `target` (an enemy's name): nothing when it is firing at it already (the
    -- engine's aim stands for the order: duty.engine), as fire at will when it stands in melee within
    -- range (services.fire_freely), else as an explicit target. Its duty table goes on.
    local function aim(it, target, run)
        local enemy = enemy_by_name[target]
        if not enemy then return 'unknown_target' end
        duty[it.name] = duty[it.name] or {}
        if on_target(it, target) then
            local d = duty[it.name]
            d.free, d.free_for, d.engine = false, 0, target
            handle.aims_kept = handle.aims_kept + 1
            opts.emit('nn_aim_kept', {t = opts.now_ms(), u = it.name, tg = target})
            return 'kept'
        end
        duty[it.name].engine = nil
        local range = read(function() return it.unit:missile_range() end)
        if services.fire_freely(seen[it.name], seen[target], range) then
            local d = duty[it.name]
            if not d.free and services.firing(seen[it.name], recent(it.name)) then
                -- shooting standing already: fire at will is what it does (a halt would restart its aim)
                d.free, d.free_for, d.idle = true, 0, 0
                handle.aims_kept = handle.aims_kept + 1
                opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = 'free_kept', tg = target})
                return 'kept'
            end
            if not d.free then free_fire(it, target, 'free') end
        else
            orders.attack_ranged(it.uc, enemy.unit, run, true)
            duty[it.name].free, duty[it.name].free_for = false, 0
        end
        return 'given'
    end

    -- A shooter left on the engine's aim (duty.engine) whose engine is off that target now: the
    -- order is given for real.
    local function drifted(it, d, target)
        if d.engine ~= target or d.free then return false end
        local me, row = seen[it.name], seen[target]
        if not (me and not services.down(me) and me.m ~= true) or on_target(it, target) then return false end
        if not (row and not services.down(row)) then return false end
        d.engine = nil
        opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = 'engine_off', tg = target})
        return true
    end

    local function give(it, order)
        local uc = it.uc
        if order.kind ~= 'attack' and order.kind ~= 'hold' then reaim[it.name] = nil end
        if order.kind ~= 'attack' then duty[it.name] = nil end
        guard[it.name] = nil
        stall[it.name] = nil
        if order.kind == 'hold' then
            -- A shooter firing at the unit it would be aimed at under hold goes on (no halt).
            local me = seen[it.name]
            local tg = me and me.t
            if shooter(it.unit) and type(tg) == 'string' and on_target(it, tg)
                and services.hold_target(me, seen_rows, read(function() return it.unit:missile_range() end), tg) == tg then
                guard[it.name] = {pick = tg}
                duty[it.name] = {engine = tg}
                handle.aims_kept = handle.aims_kept + 1
                opts.emit('nn_hold', {t = opts.now_ms(), u = it.name, action = 'keep', tg = tg})
                return 'kept'
            end
            orders.halt(uc)
            orders.set_fire_at_will(uc, true)
            guard[it.name] = {}       -- a held shooter is aimed by watch_shooters (services.hold_target)
        elseif order.kind == 'move' or order.kind == 'withdraw' then
            orders.set_fire_at_will(uc, true)
            orders.move(uc, opts.vector(order.x, order.z), order.kind == 'withdraw' or order.run)
        else
            local enemy = enemy_by_name[order.target]
            if not enemy then return 'unknown_target' end
            if shooter(it.unit) then
                return aim(it, order.target, order.run)
            else
                duty[it.name] = nil
                orders.attack_melee(uc, enemy.unit)
                if empty_shooter(it.unit) then
                    handle.empty_melees = handle.empty_melees + 1
                    opts.emit('nn_empty_melee', {t = opts.now_ms(), u = it.name, tg = order.target})
                end
            end
        end
        return 'given'
    end

    local function effects(rows)
        if not opts.cco then return end
        for _, row in ipairs(rows) do
            local u = unit_by_name[row.n]
            if u then
                row.fx = services.active_effects(function(field)
                    return read(function() return opts.cco(u, field) end)
                end)
                local text = row.fx and table.concat(row.fx, ',') or nil
                if text ~= fx_was[row.n] then
                    fx_was[row.n] = text
                    opts.emit('nn_effects', {t = opts.now_ms(), u = row.n, fx = row.fx or 'unknown'})
                end
            end
        end
    end

    local function readiness()
        for name, keys in pairs(owned) do
            local it = own_by_name[name]
            ready_was[name] = ready_was[name] or {}
            for _, key in ipairs(keys) do
                local r = read(function() return it.unit:can_perform_special_ability(key) end)
                if r ~= ready_was[name][key] then
                    ready_was[name][key] = r
                    opts.emit('nn_ability_ready', {t = opts.now_ms(), u = name, key = key, ready = r})
                end
            end
        end
    end

    local function write(done)
        local rows = opts.rows()
        effects(rows)
        readiness()
        local doc = services.state_document(opts.meta, handle.written, opts.now_ms(), rows, done, used)
        handle.write_mode = exchange.write(M.STATE_FILE, json.encode(doc))
        return rows
    end

    -- Shooters under an attack order: released to fire at will while they cannot shoot
    -- their target, given it again once they can (services.missile_duty).
    local function stand_free(it)
        duty[it.name] = nil
        orders.halt(it.uc)
        orders.set_fire_at_will(it.uc, true)
    end

    -- A held shooter: aimed at the simulator's target (services.hold_target, event nn_hold), under
    -- the same duty as an attack order's target; halted again if it walks (services.hold_guard).
    local function watch_hold(it, g, rows)
        local me = seen[it.name]
        if not (shooter(it.unit) and standing(it.unit)) then
            if g.pick then
                g.pick = nil
                stand_free(it)
            end
            return
        end
        local check = services.hold_guard(g, me, g.pick ~= nil)
        if check == 'wait' then return end
        if check == 'halt' then
            opts.emit('nn_hold', {t = opts.now_ms(), u = it.name, action = 'halt', tg = g.pick})
            g.pick = nil
            stand_free(it)
            handle.hold_halts = handle.hold_halts + 1
            return
        end
        local range = read(function() return it.unit:missile_range() end)
        -- No pick yet: the unit the engine fires at is kept while it qualifies (no new aim).
        local current = g.pick
        if not current and me and type(me.t) == 'string' and on_target(it, me.t) then current = me.t end
        local pick = services.hold_target(me, rows, range, current)
        if pick ~= g.pick then
            g.pick = pick
            opts.emit('nn_hold', {t = opts.now_ms(), u = it.name, action = pick and 'aim' or 'none', tg = pick})
            if pick then
                duty[it.name] = {}
                aim(it, pick, false)
                handle.hold_aims = handle.hold_aims + 1
            else
                stand_free(it)
            end
        elseif pick and duty[it.name] and drifted(it, duty[it.name], pick) then
            aim(it, pick, false)
        elseif pick and duty[it.name] then
            local action = services.missile_duty(duty[it.name], me, seen[pick])
            if action == 'release' then
                free_fire(it, pick, 'release')
            elseif action == 'resume' then
                handle.resumed = handle.resumed + 1
                opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = action, tg = pick})
                aim(it, pick, false)
            end
        end
    end

    local function recover_target(it, order, rows)
        if not (order and (order.kind == 'attack' or order.kind == 'hold') and shooter(it.unit)) then
            reaim[it.name] = nil
            return false
        end
        reaim[it.name] = reaim[it.name] or {}
        local target = services.reaim_target(reaim[it.name], seen[it.name], rows,
            read(function() return it.unit:missile_range() end), order.target, recent(it.name))
        local enemy = target and enemy_by_name[target]
        if not enemy then return false end
        -- Deliberately bypass aim/fire_freely: halt + free fire can leave no engine target
        -- indefinitely. attack_unit supplies a target for the engine to turn toward and shoot.
        orders.attack_ranged(it.uc, enemy.unit, order.kind == 'attack' and order.run, true)
        duty[it.name] = {}
        if order.kind == 'hold' then guard[it.name] = {pick = target} end
        handle.reaims = handle.reaims + 1
        opts.emit('nn_reaim', {t = opts.now_ms(), u = it.name, k = order.kind, tg = target})
        return true
    end

    local function watch_shooters(rows)
        seen, seen_rows = {}, rows
        for _, row in ipairs(rows) do
            seen[row.n] = row
            if row.fire == true then fired[row.n] = handle.written end
        end
        for _, it in ipairs(opts.own) do
            local order, d, g = current[it.name], duty[it.name], guard[it.name]
            if d and order and order.kind == 'attack' and empty_shooter(it.unit) and standing(it.unit) then
                give(it, order)
            elseif recover_target(it, order, rows) then
                -- Recovery takes precedence over free-fire duty and the held walk cooldown.
            elseif d and order and order.kind == 'attack' and drifted(it, d, order.target) then
                give(it, order)
            elseif d and order and order.kind == 'attack' then
                local action = services.missile_duty(d, seen[it.name], seen[order.target])
                if action == 'release' then
                    free_fire(it, order.target, 'release')
                elseif action == 'resume' then
                    handle.resumed = handle.resumed + 1
                    opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = action, tg = order.target})
                    give(it, order)
                end
            elseif g and order and order.kind == 'hold' then
                watch_hold(it, g, rows)
            end
            -- An attack (not a shooter's: its duty above) or a move the engine dropped: given again
            -- (services.order_stalled, event nn_stall).
            if order and (order.kind == 'move' or order.kind == 'withdraw' or (order.kind == 'attack' and not d)) then
                stall[it.name] = stall[it.name] or {}
                local tg = order.kind == 'attack' and seen[order.target] or nil
                if services.order_stalled(stall[it.name], seen[it.name], order, tg) == 'regive' then
                    handle.stalls = handle.stalls + 1
                    opts.emit('nn_stall', {t = opts.now_ms(), u = it.name, k = order.kind, tg = order.target})
                    give(it, order)
                end
            else
                stall[it.name] = nil
            end
        end
    end

    function handle.decide()
        if handle.written > handle.answered then
            handle.missed = handle.missed + 1
            opts.emit('nn_miss', {t = opts.now_ms(), move = handle.written, answered = handle.answered})
        end
        handle.written = handle.written + 1
        times[handle.written] = {model = opts.model_ms(), real = clock()}
        times[handle.written - M.KEEP_TIMES] = nil
        local rows = write(false)
        for _, row in ipairs(rows) do
            if own_by_name[row.n] and services.down(row) then down_at[row.n] = handle.written end
        end
        watch_shooters(rows)
    end

    -- A unit that rallied goes on with the order it had when it broke, as in the simulator (an
    -- attack on a target that is gone holds: the simulator turns it to HOLD).
    local function rally(it)
        local order = lost[it.name]
        lost[it.name] = nil
        if order.kind == 'attack' then
            local enemy = enemy_by_name[order.target]
            if not enemy or (read(function() return enemy.unit:number_of_men_alive() end) or 0) <= 0 then
                order = {kind = 'hold', run = false}
            end
        end
        local status = give(it, order)
        if status ~= 'unknown_target' then current[it.name] = order end
        handle.regiven = handle.regiven + 1
        opts.emit('nn_rally', {t = opts.now_ms(), u = it.name, k = order.kind, x = order.x, z = order.z,
            tg = order.target, run = order.run, status = status})
    end

    -- Every poll: a unit that broke loses its order (the engine dropped it), one that stands again
    -- gets it back at once.
    local function follow_morale()
        for _, it in ipairs(opts.own) do
            if not standing(it.unit) then
                if current[it.name] then lost[it.name] = current[it.name] end
                current[it.name], duty[it.name], guard[it.name], stall[it.name] = nil, nil, nil, nil
                reaim[it.name] = nil
            elseif lost[it.name] and not current[it.name] then
                rally(it)
            end
        end
    end

    -- One ability request of the answer -> its status (module comment).
    local function use_ability(req)
        local it = own_by_name[req.unit]
        if not it then return 'unknown_unit' end
        if not standing(it.unit) then return 'down' end
        if read(function() return it.unit:can_perform_special_ability(req.key) end) ~= true then
            return 'not_ready'
        end
        local ok = pcall(function() it.uc:perform_special_ability(req.key, it.unit) end)
        if not ok then return 'error' end
        used[req.unit] = used[req.unit] or {}
        used[req.unit][req.key] = opts.now_ms()
        return 'used'
    end

    function handle.poll()
        follow_morale()
        local text = exchange.read(M.ORDERS_FILE)
        if not text then return end
        local doc, reason = services.parse_orders(text)
        if not doc then
            handle.bad = handle.bad + 1
            return reason
        end
        if doc.batch ~= opts.meta.batch or doc.move <= handle.answered or doc.move > handle.written then return end
        handle.answered = doc.move
        local sent = times[doc.move] or {}
        local row = {t = opts.now_ms(), move = doc.move, lag = handle.written - doc.move,
            wait_model_ms = sent.model and opts.model_ms() - sent.model,
            wait_real_ms = services.real_ms(sent.real, clock()), think_ms = doc.think_ms,
            orders = {}, kept = 0, keeps = 0, skipped = 0}
        for _, it in ipairs(opts.own) do
            local order = doc.orders[it.name]
            -- down in the state answered (the network's HOLD is a filler), or down now: no order
            local up = standing(it.unit) and not (down_at[it.name] and doc.move <= down_at[it.name])
            if order and order.kind == 'keep' then
                -- The order in force goes on; a unit without one stands as it was taken.
                row.keeps = row.keeps + 1
            elseif order and not up then
                row.skipped = row.skipped + 1
            elseif order and services.changed(current[it.name], order) then
                local status = give(it, order)
                if status ~= 'unknown_target' then current[it.name], lost[it.name] = order, nil end
                if status ~= 'kept' then handle.given = handle.given + 1 end
                row.orders[#row.orders + 1] = {u = it.name, k = order.kind, x = order.x, z = order.z,
                    tg = order.target, run = order.run, status = status}
            elseif order then
                row.kept = row.kept + 1
            end
        end
        handle.keeps = handle.keeps + row.keeps
        handle.applied = handle.applied + 1
        opts.emit('nn_orders', row)
        for _, req in ipairs(doc.abilities or {}) do
            local status = use_ability(req)
            if status == 'used' then
                handle.abilities_used = handle.abilities_used + 1
            else
                handle.abilities_refused = handle.abilities_refused + 1
            end
            opts.emit('nn_ability', {t = opts.now_ms(), move = doc.move, u = req.unit, key = req.key, status = status})
        end
    end

    function handle.finish()
        pcall(write, true)
    end

    function handle.stats()
        return {nn_moves = handle.written, nn_answered = handle.applied, nn_missed = handle.missed,
            nn_orders_given = handle.given, nn_keeps = handle.keeps, nn_bad_files = handle.bad, nn_write_mode = handle.write_mode,
            nn_regiven = handle.regiven, nn_released = handle.released, nn_resumed = handle.resumed,
            nn_abilities_used = handle.abilities_used, nn_abilities_refused = handle.abilities_refused,
            nn_hold_aims = handle.hold_aims, nn_hold_halts = handle.hold_halts, nn_stalls = handle.stalls,
            nn_aims_kept = handle.aims_kept, nn_reaims = handle.reaims, nn_empty_melees = handle.empty_melees}
    end

    return handle
end

return M
