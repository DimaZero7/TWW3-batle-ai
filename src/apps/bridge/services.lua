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
-- An ability line: ability <unit name> <ability key>: use it now (once; independent of the
-- unit's order). Files without ability lines are as before.
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

-- The companion's answer -> {move, batch, think_ms, orders = {name -> order},
-- abilities = {{unit, key}...}}, or nil and the reason. A file without its last line 'end'
-- is not complete.
function M.parse_orders(text)
    if type(text) ~= 'string' or text == '' then return nil, 'empty' end
    local doc = {orders = {}, count = 0, abilities = {}}
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
        elseif w[1] == 'ability' then
            if not w[2] or not w[3] then return nil, 'ability without a unit or a key' end
            doc.abilities[#doc.abilities + 1] = {unit = w[2], key = w[3]}
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

-- A shooter's duty under an ATTACK order (the rows of the state, one per decision).
-- In the game a shooter holding an explicit target it cannot hit (the target in
-- melee, out of sight) stands without shooting and picks no other: 722 unit-seconds
-- in one gate battle (01.10.2026). So:
--   * an attack order on a target in melee within the shooter's range is not given
--     as an explicit target: the shooter fires at will (fire_freely);
--   * after RELEASE_AFTER decisions standing idle (not moving, not firing, not in
--     melee, with ammunition) under an explicit target the order is released to fire
--     at will. The count goes on across the network's new attack orders: it changes
--     a shooter's target every ~5 s, a count that restarted with each would never end;
--   * fire at will: the game chooses the target, as its own AI's shooters do. The
--     ordered target is taken again once it is out of melee, after at least FREE_MIN
--     decisions.
M.RELEASE_AFTER = 4
M.FREE_MIN = 10

local function up(row)
    return row ~= nil and (row.men or 0) > 0 and row.r ~= true and row.s ~= true
end

-- Standing, not moving, not in melee, not shooting, with ammunition.
function M.shooter_idle(row)
    return up(row) and row.m ~= true and row.mv ~= true and row.fire ~= true and (row.a or 0) > 0
end

-- Give this attack order as fire at will: the target stands in melee within range
-- (centre to centre, metres: never further than the engine's edge-to-edge range).
function M.fire_freely(me, target, range)
    if not (up(me) and up(target) and target.m == true) then return false end
    if not (value.finite(me.x) and value.finite(me.z) and value.finite(target.x) and value.finite(target.z)) then
        return false
    end
    return math.sqrt((me.x - target.x) ^ 2 + (me.z - target.z) ^ 2) <= (range or 0)
end

-- duty = {idle, free, free_for} (kept per unit between calls; {} to start).
-- Returns 'release', 'resume' or nil.
function M.missile_duty(duty, me, target)
    duty.idle, duty.free_for = duty.idle or 0, duty.free_for or 0
    if duty.free then
        duty.free_for = duty.free_for + 1
        if duty.free_for >= M.FREE_MIN and up(me) and up(target) and target.m ~= true then
            duty.free, duty.free_for, duty.idle = false, 0, 0
            return 'resume'
        end
        return nil
    end
    duty.idle = M.shooter_idle(me) and duty.idle + 1 or 0
    if duty.idle >= M.RELEASE_AFTER then
        duty.free, duty.free_for, duty.idle = true, 0, 0
        return 'release'
    end
    return nil
end

-- What the companion reads: meta (batch, factions, attacker...), the move
-- number, the battle time and every unit's row; used = {unit name -> {ability key ->
-- battle ms of its last use by the bridge}} (the companion counts the timers from it).
function M.state_document(meta, move, t_ms, units, done, used)
    local doc = {move = move, t = t_ms, units = units, done = done == true, abilities_used = used or {}}
    for k, v in pairs(meta) do doc[k] = v end
    return doc
end

-- The phases active on a unit now, as its card shows them (CCO ActiveEffectList: verified in
-- battle, docs/en/game/units/states.md). read(field) -> the unit's CcoBattleUnit value or nil.
-- Returns a list of phase keys ({} when none), or nil when the list cannot be read.
M.MAX_EFFECTS = 16
function M.active_effects(read)
    local n = read('ActiveEffectList.Size')
    if type(n) ~= 'number' then return nil end
    local out = {}
    for i = 0, math.min(n, M.MAX_EFFECTS) - 1 do
        local key = read('ActiveEffectList.At(' .. i .. ').PhaseRecordContext.Key')
        if type(key) == 'string' and key ~= '' then out[#out + 1] = key end
    end
    return out
end

-- Real milliseconds between two os.clock() readings (nil when unknown).
function M.real_ms(from, to)
    if not (value.finite(from) and value.finite(to)) then return nil end
    return math.floor((to - from) * 1000 + 0.5)
end

return M
