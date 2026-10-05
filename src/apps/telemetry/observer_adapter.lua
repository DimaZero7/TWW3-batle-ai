-- Extra research telemetry of a battle a human plays (entries.nn_arena, own_ai 'human';
-- docs/en/launch/run.md): everything the game gives about every unit of both sides, beyond the
-- arena's nn_sample row, so the human's orders and the battle can be studied later. Full view:
-- trusted research telemetry, never an AI's input.
--   decorate(row)  adds to a unit's sample row (a field that cannot be read is left out):
--                  ob / ow = ordered bearing (deg) / width (m) of the order in force, idle = the
--                  engine sees no order (unit:is_idle()), v = visible to the other side,
--                  uma = under missile attack, td = taking damage, dir = damage inflicted recently
--                  (CCO IsUnderMissileAttack, IsTakingDamage, DamageInflictedRecently);
--   changes()      emits on change only, as the network's bridge does: nn_effects (a unit's active
--                  phases, CCO ActiveEffectList) and nn_ability_ready (a unit's active ability:
--                  can_perform_special_ability turned true or false - a use shows as true -> false),
--                  both sides, with the side;
--   soldiers()     every soldiers_every-th call: nn_soldiers, every soldier's x, z in decimetres
--                  of every unit (CCO ManList; ~50 ms for 2000 men, so not every second).
local services = require('apps.bridge.services')
local formation = require('apps.units.formation_adapter')
local value = require('apps.core.value')

local M = {}

M.CCO_NUMBERS = {dir = 'DamageInflictedRecently'}
M.CCO_BOOLEANS = {uma = 'IsUnderMissileAttack', td = 'IsTakingDamage'}

local function read(fn)
    local ok, v = pcall(fn)
    if ok then return v end
    return nil
end

local finite = value.finite
local function round(v, k)
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- opts: units = {{name, unit, side}}, alliances = {side 1's, side 2's engine alliance},
-- cco(unit, field), emit(event, fields), now_ms(), soldiers_every (calls; 0 or nil: never).
function M.new(opts)
    local by_name, owned, fx_was, ready_was = {}, {}, {}, {}
    local calls = 0
    for _, it in ipairs(opts.units) do
        by_name[it.name] = it
        local list = read(function() return it.unit:owned_non_passive_special_abilities() end)
        if type(list) == 'table' and #list > 0 then
            owned[it.name] = {}
            for _, k in ipairs(list) do owned[it.name][#owned[it.name] + 1] = tostring(k) end
        end
    end
    local self = {}

    function self.decorate(row)
        local it = by_name[row.n]
        if not it then return row end
        local u = it.unit
        local ob, ow = read(function() return u:ordered_bearing() end), read(function() return u:ordered_width() end)
        if finite(ob) then row.ob = round(ob, 0) end
        if finite(ow) then row.ow = round(ow) end
        local idle = read(function() return u:is_idle() end)
        if type(idle) == 'boolean' then row.idle = idle end
        local other = opts.alliances and opts.alliances[3 - it.side]
        local v = other and read(function() return u:is_visible_to_alliance(other) end)
        if type(v) == 'boolean' then row.v = v end
        for key, field in pairs(M.CCO_NUMBERS) do
            local x = read(function() return opts.cco(u, field) end)
            if finite(x) then row[key] = round(x, 3) end
        end
        for key, field in pairs(M.CCO_BOOLEANS) do
            local x = read(function() return opts.cco(u, field) end)
            if type(x) == 'boolean' then row[key] = x end
        end
        return row
    end

    function self.changes()
        for _, it in ipairs(opts.units) do
            local fx = services.active_effects(function(field)
                return read(function() return opts.cco(it.unit, field) end)
            end)
            local text = fx and table.concat(fx, ',') or nil
            if text ~= fx_was[it.name] then
                fx_was[it.name] = text
                opts.emit('nn_effects', {t = opts.now_ms(), u = it.name, side = it.side, fx = fx or 'unknown'})
            end
            for _, key in ipairs(owned[it.name] or {}) do
                ready_was[it.name] = ready_was[it.name] or {}
                local r = read(function() return it.unit:can_perform_special_ability(key) end)
                if r ~= ready_was[it.name][key] then
                    ready_was[it.name][key] = r
                    opts.emit('nn_ability_ready', {t = opts.now_ms(), u = it.name, side = it.side, key = key, ready = r})
                end
            end
        end
    end

    function self.soldiers()
        local every = opts.soldiers_every or 0
        if every <= 0 then return end
        calls = calls + 1
        if (calls - 1) % every ~= 0 then return end
        local rows = {}
        for _, it in ipairs(opts.units) do
            local men = formation.soldiers(opts.cco, it.unit)
            if men.status == 'ok' then
                rows[#rows + 1] = {n = it.name, side = it.side, xz_dm = men.xz_dm}
            else
                rows[#rows + 1] = {n = it.name, side = it.side, unavailable = men.reason}
            end
        end
        opts.emit('nn_soldiers', {t = opts.now_ms(), units = rows})
    end

    return self
end

return M
