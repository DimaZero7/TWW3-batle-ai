-- Start-of-battle plan (pure): assessment -> strategy -> formation. Only
-- calls the modules in order and passes their data along; each module is
-- tested on its own (docs/ru/apps/plan.md). modules may be replaced in tests.
local assessment = require('apps.assessment.services')
local strategy = require('apps.strategy.services')
local strategy_contract = require('apps.strategy.contract')
local formation = require('apps.formation.services')

local M = {}

M.MODULES = {assessment = assessment, strategy = strategy, contract = strategy_contract, formation = formation}

-- input: {role, bearing, own = {anchor = {x, z}, units}, enemy = {units, lord = {x, z} | nil (only if visible)}}
-- Returns {status, features, decision, formation}; status 'no_strategy' when nothing fits.
function M.start(input, modules)
    local m = modules or M.MODULES
    local features = m.assessment.assess({role = input.role, own = input.own.units, enemy = input.enemy.units})
    local decision = m.strategy.decide(features, input.own.units, m.assessment.category)
    m.contract.check_decision(decision, input.own.units)
    local result = {features = features, decision = decision}
    if decision.strategy == 'none' then
        result.status = 'no_strategy'
        return result
    end
    local units = {}
    for i, u in ipairs(input.own.units) do
        local copy = {}
        for k, v in pairs(u) do copy[k] = v end
        copy.role = decision.roles[u.id]
        units[i] = copy
    end
    -- input.formation_params: overrides for experiments and tests only.
    local params = {}
    for k, v in pairs(decision.formation_params or {}) do params[k] = v end
    for k, v in pairs(input.formation_params or {}) do params[k] = v end
    result.formation = m.formation.plan(decision.layout, {anchor = input.own.anchor, bearing = input.bearing,
        units = units, enemy_lord = input.enemy.lord}, params)
    result.status = result.formation.status
    return result
end

return M
