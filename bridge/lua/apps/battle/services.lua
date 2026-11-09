-- Pure checks over battle context read by battle.adapter.
local M = {version = 1}

-- Asserts that the scenario's declared roles match the native ones.
function M.verify_roles(roles, expected)
    if expected then
        assert(#expected == 2, 'two declared roles required')
        for i = 1, 2 do
            assert(expected[i] == roles[i].role, 'declared/native battle role mismatch')
        end
    end
    return roles
end

-- Stall detector: a battle is stalled when the health signature of all units
-- (see battle.adapter.health_signature) has not changed for window_ms of
-- MODEL time. update() returns true once stalled.
function M.new_stall_detector(window_ms)
    assert(type(window_ms) == 'number' and window_ms > 0, 'stall window required')
    local last_signature, last_change_ms
    local detector = {window_ms = window_ms}
    function detector.update(now_ms, signature)
        if last_signature == nil or signature ~= last_signature then
            last_signature, last_change_ms = signature, now_ms
            return false
        end
        return now_ms - last_change_ms >= window_ms
    end
    function detector.quiet_ms(now_ms)
        return last_change_ms and now_ms - last_change_ms or 0
    end
    return detector
end

function M.opponent(side)
    assert(side == 1 or side == 2, 'side must be 1 or 2')
    return 3 - side
end

return M
