-- Force assessment at the start of a battle (pure). From both rosters it
-- computes one set of features; strategies only read them
-- (docs/ru/architecture/strategies.md#как-выбирается-стратегия).
--
-- Unit (roster input, tools/sim/formation.py): {id, class, men, commanding,
-- fire ('arc' | 'direct' | nil), range_m, health, armour, melee_attack,
-- melee_defence, missile_damage, ammo, shapes}. Missing card values count as
-- unknown, never as zero strength.
local value = require('apps.core.value')

local M = {}
local finite = value.finite

-- lord: a single commanding entity; shooter: has a missile range;
-- infantry: foot unit without missiles; other: everything else (cavalry…).
function M.category(u)
    if u.men == 1 and u.commanding then return 'lord' end
    if finite(u.range_m) and u.range_m > 0 then return 'shooter' end
    if type(u.class) == 'string' and u.class:sub(1, 3) == 'inf' then return 'infantry' end
    return 'other'
end

-- Melee strength of infantry: health x (attack + defence) / 100.
function M.melee_power(u)
    local health = finite(u.health) and u.health or u.men
    return health * ((u.melee_attack or 0) + (u.melee_defence or 0)) / 100
end

-- Missile strength of a shooter: men x missile damage (x 1 when unknown).
function M.ranged_power(u)
    return u.men * (finite(u.missile_damage) and u.missile_damage or 1)
end

function M.side(units)
    local s = {count = {lord = 0, shooter = 0, infantry = 0, other = 0}, melee_power = 0, ranged_power = 0,
        arc_shooters = 0}
    local armour, weight = 0, 0
    for _, u in ipairs(units) do
        local c = M.category(u)
        s.count[c] = s.count[c] + 1
        if c == 'infantry' then
            s.melee_power = s.melee_power + M.melee_power(u)
            if finite(u.armour) then
                local w = finite(u.health) and u.health or u.men
                armour, weight = armour + u.armour * w, weight + w
            end
        elseif c == 'shooter' then
            s.ranged_power = s.ranged_power + M.ranged_power(u)
            if u.fire == 'arc' then s.arc_shooters = s.arc_shooters + 1 end
        end
    end
    s.arc_share = s.count.shooter > 0 and s.arc_shooters / s.count.shooter or 0
    -- Health-weighted mean armour of the infantry; nil when there is none or it is unknown.
    s.infantry_armour = weight > 0 and armour / weight or nil
    return s
end

local function ratio(a, b)
    if b > 0 then return a / b end
    return nil
end

-- input: {role = 'attack' | 'defend' | nil, own = {units}, enemy = {units}}.
function M.assess(input)
    assert(input and type(input.own) == 'table' and type(input.enemy) == 'table', 'Both armies required')
    local own, enemy = M.side(input.own), M.side(input.enemy)
    return {role = input.role, own = own, enemy = enemy,
        melee_ratio = ratio(own.melee_power, enemy.melee_power),
        ranged_ratio = ratio(own.ranged_power, enemy.ranged_power)}
end

return M
