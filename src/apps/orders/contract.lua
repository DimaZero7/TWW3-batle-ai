-- Command shape and validation. Pure.
-- A command is a plain table:
--   {unit_id, action='move', x, z, run?, facing_deg?, width_m?}
--   {unit_id, action='attack', target_id, mode='melee'|'ranged'}
--   {unit_id, action='guard', enabled}
--   {unit_id, action='halt'}
-- facing_deg/width_m on move require an explicit movement contract.
local value = require('apps.core.value')
local units = require('apps.units.contract')

local M = {}
local finite = value.finite

M.MOVEMENT_CONTRACTS = {['formation-move-v1'] = true, ['deployment-march-formation-v1'] = true}

local ACTION_FIELDS = {
    move = {'x', 'z', 'run'},
    attack = {'target_id', 'mode'},
    guard = {'enabled'},
    halt = {},
}

-- Asserts a dense 1..n array with at most limit entries.
function M.check_array(commands, limit)
    assert(type(commands) == 'table' and #commands <= limit, 'command limit exceeded')
    for k in pairs(commands) do
        assert(type(k) == 'number' and k >= 1 and k <= #commands and k % 1 == 0,
            'commands must be array')
    end
    for i = 1, #commands do assert(commands[i] ~= nil, 'sparse command array') end
end

-- own[unit_id] = {alive, position, kind, ammo}; enemies[target_id] = {visibility, position}.
-- At most one motion (move/attack/halt) and one guard change per unit.
function M.validate(commands, own, enemies, limit, contract)
    assert(contract == nil or M.MOVEMENT_CONTRACTS[contract], 'unknown movement contract')
    M.check_array(commands, limit)
    local motion, guard = {}, {}
    for _, c in ipairs(commands) do
        assert(type(c) == 'table', 'command object required')
        local allowed = {unit_id = true, action = true}
        for _, key in ipairs(ACTION_FIELDS[c.action] or {}) do allowed[key] = true end
        if c.action == 'move' and contract then
            allowed.facing_deg = true
            allowed.width_m = true
        end
        for key in pairs(c) do assert(allowed[key], 'unknown command field') end
        local u = own[c.unit_id]
        assert(u and u.alive == true and u.position, 'unit ownership/alive required')
        if c.action == 'guard' then
            assert(type(c.enabled) == 'boolean' and not guard[c.unit_id], 'invalid guard')
            guard[c.unit_id] = true
        else
            assert(not motion[c.unit_id], 'duplicate motion')
            motion[c.unit_id] = true
            if c.action == 'move' then
                assert(finite(c.x) and finite(c.z) and (c.run == nil or type(c.run) == 'boolean'),
                    'invalid move')
                if contract then
                    assert(finite(c.facing_deg) and c.facing_deg >= 0 and c.facing_deg < 360,
                        'invalid move facing')
                    assert(finite(c.width_m) and units.valid_width(u.kind, c.width_m),
                        'invalid move width')
                end
            elseif c.action == 'attack' then
                local target = enemies[c.target_id]
                assert(target and target.visibility == 'visible' and target.position,
                    'target not visible')
                assert(c.mode == 'melee' or (c.mode == 'ranged' and units.can_shoot(u)),
                    'invalid attack mode')
            else
                assert(c.action == 'halt', 'unsupported action')
            end
        end
    end
    return commands
end

return M
