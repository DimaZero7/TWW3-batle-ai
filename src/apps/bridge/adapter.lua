-- The bridge to the network: our side is commanded by the companion
-- (tools/nn/companion), a program outside the game (docs/en/apps/bridge.md).
--   decide()  every decision tick: writes the state of every unit with the next
--             move number;
--   poll()    often: reads the companion's answer and gives each of our units
--             its order through apps.orders.adapter (only orders that changed;
--             'keep' gives nothing: the order in force goes on). A unit that
--             routs loses its order in the engine: the bridge forgets it too, so
--             the same order is given again once the unit rallies;
--   finish()  writes the last state with done = true.
-- A shooter under an attack order that stands idle (its target in melee, out of
-- sight) is released to fire at will and takes its target again later; an attack
-- on a target already in melee within range is given as fire at will at once
-- (services.missile_duty, services.fire_freely, event nn_duty).
-- No answer by the next decision tick: the orders in force stay (event nn_miss).
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

-- opts: army (our engine army), own / enemies = {{name, unit}}, vector(x, z) -> engine
-- vector, rows() -> every unit's row, meta (put into every state: batch...), emit(event,
-- fields), now_ms() (battle time since the start), model_ms() (engine time).
function M.start(opts)
    local handle = {written = 0, applied = 0, answered = 0, missed = 0, given = 0, keeps = 0, bad = 0,
        regiven = 0, released = 0, resumed = 0, write_mode = nil}
    -- current: the order in force; lost: the order a unit had when it broke (given again after
    -- it rallies, also for 'keep'); duty: a shooter's state under an attack order.
    local times, current, lost, duty, enemy_by_name = {}, {}, {}, {}, {}
    local seen = {}   -- the rows of the last decision, by unit name
    for _, it in ipairs(opts.enemies) do enemy_by_name[it.name] = it end
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

    local function give(it, order)
        local uc = it.uc
        if order.kind ~= 'attack' then duty[it.name] = nil end
        if order.kind == 'hold' then
            orders.halt(uc)
            orders.set_fire_at_will(uc, true)
        elseif order.kind == 'move' or order.kind == 'withdraw' then
            orders.set_fire_at_will(uc, true)
            orders.move(uc, opts.vector(order.x, order.z), order.kind == 'withdraw' or order.run)
        else
            local enemy = enemy_by_name[order.target]
            if not enemy then return 'unknown_target' end
            if shooter(it.unit) then
                duty[it.name] = duty[it.name] or {}
                local range = read(function() return it.unit:missile_range() end)
                if services.fire_freely(seen[it.name], seen[order.target], range) then
                    if not duty[it.name].free then free_fire(it, order.target, 'free') end
                else
                    orders.attack_ranged(uc, enemy.unit, order.run, true)
                    duty[it.name].free, duty[it.name].free_for = false, 0
                end
            else
                duty[it.name] = nil
                orders.attack_melee(uc, enemy.unit)
            end
        end
        return 'given'
    end

    local function write(done)
        local rows = opts.rows()
        local doc = services.state_document(opts.meta, handle.written, opts.now_ms(), rows, done)
        handle.write_mode = exchange.write(M.STATE_FILE, json.encode(doc))
        return rows
    end

    -- Shooters under an attack order: released to fire at will while they cannot shoot
    -- their target, given it again once they can (services.missile_duty).
    local function watch_shooters(rows)
        seen = {}
        for _, row in ipairs(rows) do seen[row.n] = row end
        for _, it in ipairs(opts.own) do
            local order, d = current[it.name], duty[it.name]
            if d and order and order.kind == 'attack' then
                local action = services.missile_duty(d, seen[it.name], seen[order.target])
                if action == 'release' then
                    free_fire(it, order.target, 'release')
                elseif action == 'resume' then
                    handle.resumed = handle.resumed + 1
                    opts.emit('nn_duty', {t = opts.now_ms(), u = it.name, action = action, tg = order.target})
                    give(it, order)
                end
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
        watch_shooters(write(false))
    end

    function handle.poll()
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
            local up = standing(it.unit)
            if not up then
                -- The engine drops the order of a routing unit: forget it, give it again after the rally.
                lost[it.name] = current[it.name] or lost[it.name]
                current[it.name], duty[it.name] = nil, nil
            end
            local again = false
            if order and order.kind == 'keep' and up and not current[it.name] and lost[it.name] then
                order, again = lost[it.name], true   -- keep after a rally: the order the unit had
            end
            if order and order.kind == 'keep' then
                -- The order in force goes on; a unit without one stands as it was taken.
                row.keeps = row.keeps + 1
            elseif order and not up then
                row.skipped = row.skipped + 1
            elseif order and services.changed(current[it.name], order) then
                again = again or lost[it.name] ~= nil   -- the first order after a rally
                local status = give(it, order)
                if status == 'given' then current[it.name], lost[it.name] = order, nil end
                handle.given = handle.given + 1
                if again then handle.regiven = handle.regiven + 1 end
                row.orders[#row.orders + 1] = {u = it.name, k = order.kind, x = order.x, z = order.z,
                    tg = order.target, run = order.run, status = status, again = again or nil}
            elseif order then
                row.kept = row.kept + 1
            end
        end
        handle.keeps = handle.keeps + row.keeps
        handle.applied = handle.applied + 1
        opts.emit('nn_orders', row)
    end

    function handle.finish()
        pcall(write, true)
    end

    function handle.stats()
        return {nn_moves = handle.written, nn_answered = handle.applied, nn_missed = handle.missed,
            nn_orders_given = handle.given, nn_keeps = handle.keeps, nn_bad_files = handle.bad, nn_write_mode = handle.write_mode,
            nn_regiven = handle.regiven, nn_released = handle.released, nn_resumed = handle.resumed}
    end

    return handle
end

return M
