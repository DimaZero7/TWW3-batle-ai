-- Diagnostic variant of forced_melee: its visible 10 s delay proves that
-- a reloaded external policy file is executed by the next battle.
local policy = {version = 'forced-melee-v3-delay'}

function policy.decide(o)
    if o.routing or o.shattered or o.men == 0 then return 'wait', 'self_unavailable' end
    if not o.target_valid or not o.target_visible then return 'wait', 'target_unavailable' end
    if o.in_melee then return 'wait', 'already_in_melee' end
    if not o.has_order then
        if o.since_order_ms < 10000 then return 'wait', 'warmup_10s' end
        return 'attack', 'initial_charge'
    end
    if o.idle and o.since_order_ms >= 5000 then return 'attack', 'retry_after_idle' end
    return 'wait', 'keep_current_order'
end

return policy
