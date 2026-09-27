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

function M.opponent(side)
    assert(side == 1 or side == 2, 'side must be 1 or 2')
    return 3 - side
end

return M
