-- Pure observer memory: current visibility plus last-seen position.
-- Memory is private to ONE observer and ONE battle; create a new one per battle.
local value = require('apps.core.value')

local M = {}

function M.new_memory()
    return {}
end

local function copy_seen(s)
    if not s then return nil end
    return {x = s.x, y = s.y, z = s.z, seen_ms = s.seen_ms}
end

-- visible: true / false / nil (unknown). position: {x,y,z} or nil when it
-- could not be read. public_id comes from a permitted roster mapping,
-- never from an enemy engine handle.
function M.record(memory, public_id, visible, position, now_ms)
    assert(type(public_id) == 'string' and public_id ~= '', 'public_id required')
    assert(value.finite(now_ms) and now_ms >= 0, 'valid time required')
    assert(type(memory) == 'table', 'observer memory required')
    local out = {id = public_id, sampled_ms = now_ms,
        visibility = visible == nil and 'unknown' or (visible and 'visible' or 'not_visible')}
    if visible then
        if position then
            local seen = {x = position.x, y = position.y, z = position.z, seen_ms = now_ms}
            memory[public_id] = copy_seen(seen)
            out.current = copy_seen(seen)
        else
            out.position_status = 'unavailable'
        end
    end
    out.last_seen = copy_seen(memory[public_id])
    return out
end

return M
