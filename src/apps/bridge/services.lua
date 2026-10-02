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

-- A row of a unit that takes no orders: routing, shattered or dead (men read as 0). A reading
-- that failed (nil) is not taken as down.
function M.down(row)
    return row ~= nil and (row.r == true or row.s == true or (type(row.men) == 'number' and row.men <= 0))
end

-- Present and not shattered (a routing unit can still be shot).
local function present(row)
    return row ~= nil and (row.men or 0) > 0 and row.s ~= true
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

-- A shooter under a HOLD order shoots as the simulator's holding shooter does
-- (tools/nn/sim/missile.py choose_target): the nearest standing enemy in range, else the nearest
-- routing one. The game's own fire at will does not: a halted shooter picks a target itself, often
-- one just out of range (in melee), keeps it and stands. Gate 02.10.2026 (it4, 8 battles): standing
-- shooters with an enemy in range by the simulator's measure fired within 10 s in 40 % of the
-- seconds under hold against 98 % under an attack order and 90-97 % for the game's AI (at 0-15 m
-- beyond range between centres: 9 % against 100 %); one slinger unit stood 155 s with full ammunition.
-- So the bridge aims a held shooter itself: hold_target picks the target (centre distance within
-- range + HOLD_REACH_M: the simulator measures range between the formations' edges, the game's AI
-- shoots from up to ~15 m beyond range between centres), and the adapter gives it as an attack
-- order would be given (fire_freely, missile_duty). A held shooter that starts walking under it
-- for HOLD_WALK decisions is halted to fire at will for FREE_MIN decisions: hold never walks.
M.HOLD_REACH_M = 10
M.HOLD_WALK = 2

local function in_reach(me, row, range)
    if not (present(row) and row.side ~= me.side and row.v ~= false) then return nil end
    if not (value.finite(row.x) and value.finite(row.z)) then return nil end
    local d = math.sqrt((me.x - row.x) ^ 2 + (me.z - row.z) ^ 2)
    if d > range + M.HOLD_REACH_M then return nil end
    return d
end

-- me: the shooter's row; rows: every unit's row (the state's); range: its missile range, m;
-- current: the name of the target picked last time (kept while it still qualifies, unless it routs
-- and a standing enemy is in reach). -> the target's name or nil. A unit in melee keeps current.
function M.hold_target(me, rows, range, current)
    if not (up(me) and (me.a or 0) > 0 and (range or 0) > 0 and value.finite(me.x) and value.finite(me.z)) then
        return nil
    end
    if me.m == true then return current end
    local best, best_d, routing, routing_d, keep
    for _, row in ipairs(rows or {}) do
        local d = in_reach(me, row, range)
        if d then
            if row.n == current then keep = row end
            if row.r ~= true then
                if not best_d or d < best_d then best, best_d = row, d end
            elseif not routing_d or d < routing_d then
                routing, routing_d = row, d
            end
        end
    end
    if keep and (keep.r ~= true or not best) then return current end
    local pick = best or routing
    return pick and pick.n or nil
end

-- guard = {walk, block} (kept per held shooter; {} to start), me: its row, aiming: a target is
-- given. -> 'halt' when it walks under the target for HOLD_WALK decisions (then nothing is picked
-- for FREE_MIN decisions), 'wait' while blocked, nil otherwise.
function M.hold_guard(guard, me, aiming)
    guard.walk, guard.block = guard.walk or 0, guard.block or 0
    if guard.block > 0 then
        guard.block = guard.block - 1
        return 'wait'
    end
    guard.walk = (aiming and me ~= nil and me.mv == true and me.m ~= true) and guard.walk + 1 or 0
    if guard.walk >= M.HOLD_WALK then
        guard.walk, guard.block = 0, M.FREE_MIN
        return 'halt'
    end
    return nil
end

-- An order the engine dropped. The engine forgets a unit's attack (or move) order after a fight it
-- was drawn into (the unit stands where the fight ended), while the bridge still holds the order as
-- in force and gives nothing new for the same order: gate 02.10.2026 (it5), our lord under an attack
-- on enemy slingers 80-170 m away stood 92 s after beating off the enemy lord, another 14 s. So a
-- unit under a melee attack whose target is present and further than STALL_M (centre to centre),
-- or under a move whose point is further than STALL_POINT_M, that stands - not moving, not in melee,
-- not shooting, not losing health - for STALL_AFTER decisions in a row is given its order again.
M.STALL_AFTER = 3
M.STALL_M = 20
M.STALL_POINT_M = 15

-- stall = {n, hp} (kept per unit; {} to start), me: the unit's row, order: the order in force,
-- target: the attack target's row (nil for a move). -> 'regive' or nil.
function M.order_stalled(stall, me, order, target)
    local last_hp = stall.hp
    stall.n, stall.hp = stall.n or 0, me and me.hp
    if not (order and up(me) and value.finite(me.x) and value.finite(me.z)) then
        stall.n = 0
        return nil
    end
    local far = false
    if order.kind == 'attack' and present(target) and value.finite(target.x) and value.finite(target.z) then
        far = math.sqrt((me.x - target.x) ^ 2 + (me.z - target.z) ^ 2) > M.STALL_M
    elseif (order.kind == 'move' or order.kind == 'withdraw') and value.finite(order.x) and value.finite(order.z) then
        far = math.sqrt((me.x - order.x) ^ 2 + (me.z - order.z) ^ 2) > M.STALL_POINT_M
    end
    local hurt = value.finite(last_hp) and value.finite(me.hp) and me.hp < last_hp - 1e-4
    if far and me.mv ~= true and me.m ~= true and me.fire ~= true and not hurt then
        stall.n = stall.n + 1
    else
        stall.n = 0
    end
    if stall.n >= M.STALL_AFTER then
        stall.n = 0
        return 'regive'
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
