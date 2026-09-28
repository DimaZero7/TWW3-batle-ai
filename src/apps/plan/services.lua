-- Start-of-battle plan (pure): assessment -> strategy -> formation. Only
-- calls the modules in order and passes their data along; each module is
-- tested on its own (docs/ru/apps/plan.md). modules may be replaced in tests.
local assessment = require('apps.assessment.services')
local strategy = require('apps.strategy.services')
local strategy_contract = require('apps.strategy.contract')
local formation = require('apps.formation.services')
local facing = require('apps.orders.facing')

local M = {}

M.MODULES = {assessment = assessment, strategy = strategy, contract = strategy_contract, formation = formation}

-- input: {role, bearing, own = {anchor = {x, z}, units}, enemy = {units, lord = {x, z} | nil (only if visible)}}
-- The stand mask to fit a formation on the map at the start (apps.mask.area):
-- around the anchor, wide and deep enough for the formation and the places
-- apps.formation.fit may move it to.
M.AREA = {ahead_m = 30, behind_m = 170, half_width_m = 200}
function M.area(anchor, bearing)
    return {centre = anchor, bearing = bearing, ahead_m = M.AREA.ahead_m, behind_m = M.AREA.behind_m,
        half_width_m = M.AREA.half_width_m}
end

-- Returns {status, features, decision, formation}; status 'no_strategy' when nothing fits.
function M.start(input, modules)
    local m = modules or M.MODULES
    local features = m.assessment.assess({role = input.role, own = input.own.units, enemy = input.enemy.units})
    local decision
    if input.roles then
        -- Tests only: roles given outright (e.g. spearmen in two lines to test
        -- movement), laid out like "wall and arc" without choosing a strategy.
        decision = {strategy = 'test_roles', layout = 'line_and_blocks', roles = input.roles}
    else
        decision = m.strategy.decide(features, input.own.units, m.assessment.category)
        m.contract.check_decision(decision, input.own.units)
    end
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
    -- A facing the engine can hold (apps.orders.facing): off its grid every
    -- block stands turned against the line of centres, a staircase.
    local layout_input = {anchor = input.own.anchor, bearing = facing.snap(input.bearing), units = units,
        enemy_lord = input.enemy.lord, fits = input.fits}
    -- input.fits(placement) -> bool: the map (the stand mask); the formation
    -- then finds a width and a place where every unit fits (apps.formation.fit).
    if input.fits then
        result.formation = m.formation.fit(decision.layout, layout_input, params)
    else
        result.formation = m.formation.plan(decision.layout, layout_input, params)
    end
    result.status = result.formation.status
    return result
end

return M
