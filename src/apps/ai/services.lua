-- Decision helpers. Pure: observations in, decisions out, no engine objects.
local contract = require('apps.ai.contract')

local M = {}

-- Builds the duel observation from plain unit readings.
-- unit/target: {routing, shattered, men, valid, visible, in_melee, idle}.
function M.duel_observation(unit, target, order, now_ms)
    return {
        routing = unit.routing, shattered = unit.shattered, men = unit.men,
        target_valid = target.valid, target_visible = target.visible,
        in_melee = unit.in_melee, idle = unit.idle,
        has_order = order.ordered == true,
        since_order_ms = now_ms - (order.last_order_ms or 0),
    }
end

function M.decide_duel(policy, observation)
    return contract.check_duel_decision(policy.decide(observation))
end

return M
