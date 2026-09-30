-- The bridge's rules without the engine: the orders file format, when an order
-- is new enough to be given again, and the state document. Pure.
-- Format and timings: docs/en/apps/bridge.md.
local value = require('apps.core.value')

local M = {}

M.FORMAT = 'tww3_bai_nn_orders 1'
-- keep: no new order, the one in force goes on (the game gives nothing).
M.KINDS = {hold = true, move = true, attack = true, withdraw = true, keep = true}
-- A move point closer than this to the one in force is the same order: giving
-- it again would restart the unit's path for nothing.
M.REPEAT_M = 5

local function words(line)
    local out = {}
    for w in line:gmatch('%S+') do out[#out + 1] = w end
    return out
end

-- One unit line: unit <name> hold | move <x> <z> <run> | withdraw <x> <z> <run>
-- | attack <target name> <run> | keep. run is 1 or 0.
local function unit_order(w)
    local name, kind = w[2], w[3]
    if not name or not M.KINDS[kind] then return nil, 'bad kind: ' .. tostring(kind) end
    local order = {kind = kind, run = false}
    if kind == 'move' or kind == 'withdraw' then
        order.x, order.z = tonumber(w[4]), tonumber(w[5])
        if not (value.finite(order.x) and value.finite(order.z)) then return nil, 'bad point for ' .. name end
        order.run = w[6] == '1'
    elseif kind == 'attack' then
        order.target = w[4]
        if not order.target then return nil, 'attack without a target for ' .. name end
        order.run = w[5] == '1'
    end
    return name, order
end

-- The companion's answer -> {move, batch, think_ms, orders = {name -> order}},
-- or nil and the reason. A file without its last line 'end' is not complete.
function M.parse_orders(text)
    if type(text) ~= 'string' or text == '' then return nil, 'empty' end
    local doc = {orders = {}, count = 0}
    local header, complete = false, false
    for line in text:gmatch('[^\r\n]+') do
        local w = words(line)
        if not header then
            if line ~= M.FORMAT then return nil, 'unknown format' end
            header = true
        elseif w[1] == 'move' then
            doc.move = tonumber(w[2])
        elseif w[1] == 'batch' then
            doc.batch = w[2]
        elseif w[1] == 'think_ms' then
            doc.think_ms = tonumber(w[2])
        elseif w[1] == 'unit' then
            local name, order = unit_order(w)
            if not name then return nil, order end
            doc.orders[name] = order
            doc.count = doc.count + 1
        elseif w[1] == 'end' then
            complete = true
        end
    end
    if not complete then return nil, 'incomplete' end
    if not value.integer(doc.move) or not doc.batch then return nil, 'no move or batch' end
    return doc
end

-- Is `new` a different order from `old` (the one in force)?
function M.changed(old, new)
    if not old then return true end
    if old.kind ~= new.kind or old.run ~= new.run or old.target ~= new.target then return true end
    if new.x and old.x then
        return math.sqrt((new.x - old.x) ^ 2 + (new.z - old.z) ^ 2) > M.REPEAT_M
    end
    return false
end

-- What the companion reads: meta (batch, factions, attacker...), the move
-- number, the battle time and every unit's row.
function M.state_document(meta, move, t_ms, units, done)
    local doc = {move = move, t = t_ms, units = units, done = done == true}
    for k, v in pairs(meta) do doc[k] = v end
    return doc
end

-- Real milliseconds between two os.clock() readings (nil when unknown).
function M.real_ms(from, to)
    if not (value.finite(from) and value.finite(to)) then return nil end
    return math.floor((to - from) * 1000 + 0.5)
end

return M
