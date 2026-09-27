-- Pure Lua 5.1: no game objects and no random-number consumption.
local policy = { version = "forced-melee-v2" }

function policy.decide(observation)
    if observation.routing or observation.shattered or observation.men == 0 then
        return "wait", "self_unavailable"
    end
    if not observation.target_valid or not observation.target_visible then
        return "wait", "target_unavailable"
    end
    if observation.in_melee then
        return "wait", "already_in_melee"
    end
    if not observation.has_order then
        return "attack", "initial_charge"
    end
    if observation.idle and observation.since_order_ms >= 5000 then
        return "attack", "retry_after_idle"
    end
    return "wait", "keep_current_order"
end

return policy
