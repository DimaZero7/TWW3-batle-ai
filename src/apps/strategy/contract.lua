-- Contract between apps.strategy and apps.formation (pure): a decision must
-- name a strategy, and a chosen one must give a layout and a known role to
-- every own unit. Checked in tests and in battle (apps.plan).
local strategy = require('apps.strategy.services')

local M = {}

function M.check_decision(decision, units)
    assert(type(decision) == 'table' and type(decision.strategy) == 'string', 'Decision without a strategy')
    assert(type(decision.candidates) == 'table', 'Decision without the reasons of the choice')
    if decision.strategy == 'none' then return decision end
    assert(type(decision.layout) == 'string', 'Decision without a formation layout')
    assert(type(decision.roles) == 'table', 'Decision without roles')
    for _, u in ipairs(units) do
        local role = decision.roles[u.id]
        assert(strategy.ROLES[role], 'Unit ' .. tostring(u.id) .. ' has no valid role: ' .. tostring(role))
    end
    return decision
end

return M
