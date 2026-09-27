-- Trusted Lua 5.1 host for policy modules (API v1). No engine objects are
-- passed into policies. Known limits (not a hardened sandbox):
--  * string methods stay reachable through the shared string metatable,
--    e.g. ('x'):rep(n), and C functions count as one instruction;
--  * copy() limits depth and table size but not shared subtables (a DAG),
--    and runs outside the instruction budget.
-- See docs/en/apps/sandbox.md before running policies you did not write.
local value = require('apps.core.value')
local ai = require('apps.ai.contract')
local orders = require('apps.orders.contract')

local M = {}
local finite = value.finite

function M.copy(data, depth, seen)
    depth = depth or 0
    seen = seen or {}
    assert(depth < 24, 'data depth exceeded')
    local kind = type(data)
    if kind == 'nil' or kind == 'boolean' or kind == 'string' then return data end
    if kind == 'number' then assert(finite(data), 'nonfinite data'); return data end
    assert(kind == 'table' and getmetatable(data) == nil and not seen[data], 'plain acyclic data required')
    seen[data] = true
    local result, count = {}, 0
    for k, v in pairs(data) do
        count = count + 1
        assert(count <= 4096, 'data size exceeded')
        assert(type(k) == 'string' or (finite(k) and k % 1 == 0), 'invalid data key')
        result[k] = M.copy(v, depth + 1, seen)
    end
    seen[data] = nil
    return result
end

local SAFE_LIBRARIES = {
    math = {'abs', 'ceil', 'floor', 'max', 'min', 'sqrt', 'sin', 'cos', 'atan', 'atan2',
        'acos', 'asin', 'tan', 'exp', 'log', 'pow', 'fmod', 'pi'},
    string = {'byte', 'char', 'find', 'format', 'gmatch', 'gsub', 'len', 'lower', 'match',
        'sub', 'upper'},
    table = {'concat', 'insert', 'remove', 'sort'},
}

local function environment()
    local env = {pairs = pairs, ipairs = ipairs, next = next, type = type, tonumber = tonumber,
        tostring = tostring, assert = assert, error = error, select = select}
    for lib, names in pairs(SAFE_LIBRARIES) do
        env[lib] = {}
        for _, name in ipairs(names) do env[lib][name] = _G[lib][name] end
    end
    return env
end

-- Runs fn with an instruction budget (debug hook every 1000 instructions).
function M.bounded(fn, ...)
    assert(debug and debug.sethook and debug.gethook, 'instruction guard unavailable')
    local old, mask, count = debug.gethook()
    local budget = 0
    debug.sethook(function()
        budget = budget + 1000
        if budget > 1000000 then error('policy instruction budget') end
    end, '', 1000)
    local ok, result = pcall(fn, ...)
    debug.sethook(old, mask, count)
    assert(ok, result)
    return result
end

-- Loads policy source text (never bytecode) into a restricted environment.
-- Returns {deploy = fn(context), step = fn(observation)}.
function M.load(source, context, profile_name)
    local capacity = ai.profile(profile_name)
    assert(type(source) == 'string' and #source <= 131072 and source:byte(1) ~= 27, 'source text required')
    local chunk, err = loadstring(source, 'policy')
    assert(chunk, err)
    setfenv(chunk, environment())
    local module = M.bounded(chunk)
    assert(type(module) == 'table' and module.api_version == 1 and type(module.create) == 'function'
        and type(module.step) == 'function', 'API v1 module required')
    if context.own_battle_role or module.battle_role_version then
        local role = context.own_battle_role
        assert(module.battle_role_version == 1 and type(role) == 'table' and role.version == 1
            and (role.role == 'attacker' or role.role == 'defender'), 'native own-role v1 context required')
        for key in pairs(role) do assert(key == 'version' or key == 'role', 'unknown own-role field') end
    end
    if context.diagnostic_contract or module.diagnostic_contract then
        assert(orders.MOVEMENT_CONTRACTS[context.diagnostic_contract]
            and module.diagnostic_contract == context.diagnostic_contract, 'explicit movement contract required')
    end
    local placement_v2 = context.deployment_contract ~= nil or module.deployment_version == 2
        or module.deployment_contract ~= nil
    if placement_v2 then
        assert(context.deployment_contract == 'deployment-placement-v2'
            and module.deployment_contract == context.deployment_contract
            and module.deployment_version == 2 and type(module.deploy) == 'function',
            'explicit deployment-placement-v2 capability required')
    elseif capacity.id == 'empire-15-v1' then
        assert(module.deployment_version == 1 and type(module.deploy) == 'function',
            'deployment-v1 capability required')
    end
    local state = M.copy(M.bounded(module.create, M.copy(context)))
    local deployed = false
    return {
        deploy = function(deploy_context)
            assert(not deployed and type(module.deploy) == 'function',
                'deployment unavailable or already consumed')
            if placement_v2 then
                assert(deploy_context.version == 2 and deploy_context.contract == 'deployment-placement-v2',
                    'deployment context version mismatch')
            else
                assert(module.deployment_version == 1, 'deployment-v1 capability required')
            end
            deployed = true
            local plan = M.copy(M.bounded(module.deploy, state, M.copy(deploy_context)))
            orders.check_array(plan, capacity.units_per_side)
            assert(#plan == capacity.units_per_side, 'complete deployment required')
            return plan
        end,
        step = function(observation)
            assert(capacity.id ~= 'empire-15-v1' or deployed, 'deployment required before step')
            local commands = M.copy(M.bounded(module.step, state, M.copy(observation)))
            orders.check_array(commands, capacity.max_command_proposals_per_step)
            return commands
        end,
    }
end

-- Validates a step's commands against the profile's command limit.
function M.validate(commands, own, enemies, profile_name, movement_contract)
    local limit = ai.profile(profile_name).max_command_proposals_per_step
    return orders.validate(commands, own, enemies, limit, movement_contract)
end

return M
