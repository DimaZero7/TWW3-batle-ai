-- The bridge to the network: our side is commanded by the companion
-- (tools/nn/companion), a program outside the game (docs/en/apps/bridge.md).
--   decide()  every decision tick: writes the state of every unit with the next
--             move number;
--   poll()    often: reads the companion's answer and gives each of our units
--             its order through apps.orders.adapter (only orders that changed;
--             'keep' gives nothing: the order in force goes on);
--   finish()  writes the last state with done = true.
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
        write_mode = nil}
    local times, current, enemy_by_name = {}, {}, {}
    for _, it in ipairs(opts.enemies) do enemy_by_name[it.name] = it end
    exchange.remove(M.STATE_FILE)
    exchange.remove(M.ORDERS_FILE)
    for _, it in ipairs(opts.own) do
        it.uc = orders.take_control(opts.army, it.unit)
        orders.set_fire_at_will(it.uc, true)
    end

    local function give(it, order)
        local uc = it.uc
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
                orders.attack_ranged(uc, enemy.unit)
            else
                orders.attack_melee(uc, enemy.unit)
            end
        end
        return 'given'
    end

    local function write(done)
        local doc = services.state_document(opts.meta, handle.written, opts.now_ms(), opts.rows(), done)
        handle.write_mode = exchange.write(M.STATE_FILE, json.encode(doc))
    end

    function handle.decide()
        if handle.written > handle.answered then
            handle.missed = handle.missed + 1
            opts.emit('nn_miss', {t = opts.now_ms(), move = handle.written, answered = handle.answered})
        end
        handle.written = handle.written + 1
        times[handle.written] = {model = opts.model_ms(), real = clock()}
        times[handle.written - M.KEEP_TIMES] = nil
        write(false)
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
            if order and order.kind == 'keep' then
                -- The order in force goes on; a unit without one stands as it was taken.
                row.keeps = row.keeps + 1
            elseif order and not standing(it.unit) then
                row.skipped = row.skipped + 1
            elseif order and services.changed(current[it.name], order) then
                local status = give(it, order)
                if status == 'given' then current[it.name] = order end
                handle.given = handle.given + 1
                row.orders[#row.orders + 1] = {u = it.name, k = order.kind, x = order.x, z = order.z,
                    tg = order.target, run = order.run, status = status}
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
            nn_orders_given = handle.given, nn_keeps = handle.keeps, nn_bad_files = handle.bad, nn_write_mode = handle.write_mode}
    end

    return handle
end

return M
