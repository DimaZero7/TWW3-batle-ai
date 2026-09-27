-- AI capacity profiles and the decision contract. Pure.
local M = {}

-- Trusted capacity selection, never policy-controlled.
local PROFILES = {
    ['empire-7-v1'] = {units_per_side = 7, max_command_proposals_per_step = 14,
        spearmen = 4, archers = 2, speed = 20, window_style = 'Hidden'},
    ['empire-15-v1'] = {units_per_side = 15, max_command_proposals_per_step = 30,
        spearmen = 8, archers = 6, speed = 7, window_style = 'Normal'},
}

M.DEFAULT_PROFILE = 'empire-7-v1'

function M.profile(name, speed)
    name = name or M.DEFAULT_PROFILE
    local s = PROFILES[name]
    assert(s, 'unknown capacity profile')
    if speed == nil then speed = s.speed end
    assert(speed == 20 or (name == 'empire-15-v1' and speed == 7), 'unsupported playback speed')
    return {id = name, units_per_side = s.units_per_side,
        max_command_proposals_per_step = s.max_command_proposals_per_step,
        roster_counts = {lord = 1, spearmen = s.spearmen, archers = s.archers},
        speed = speed, window_style = s.window_style,
        decision_interval_ms = 1000, instructions_per_call = 1000000}
end

-- Simple duel policy contract: policy.decide(observation) -> action, reason.
-- observation: routing, shattered, men, target_valid, target_visible,
-- in_melee, idle, has_order, since_order_ms.
M.DUEL_ACTIONS = {attack = true, wait = true}

function M.check_duel_policy(policy)
    assert(type(policy) == 'table' and type(policy.decide) == 'function'
        and type(policy.version) == 'string', 'invalid duel policy')
    return policy
end

function M.check_duel_decision(action, reason)
    assert(M.DUEL_ACTIONS[action], 'unsupported policy action')
    assert(type(reason) == 'string', 'decision reason required')
    return action, reason
end

return M
