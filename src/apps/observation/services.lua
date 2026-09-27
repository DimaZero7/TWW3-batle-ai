-- Shape and self-check of a side view. Pure.
local M = {}

-- Fields an enemy record may carry: nothing about strength, orders or morale.
M.ENEMY_FIELDS = {id = true, visibility = true, sampled_ms = true, current = true,
    last_seen = true, position_status = true}

function M.new_view(side, now_ms)
    return {schema_version = 1, side = side, time_ms = now_ms, own = {}, enemies = {}, range = {}}
end

-- Asserts the view cannot carry hidden-enemy data:
--  * enemy records only have ENEMY_FIELDS;
--  * a non-visible enemy has no current position;
--  * a range reading against a non-visible enemy is withheld.
-- Returns the view so callers can chain it.
function M.check(view)
    for name, record in pairs(view.enemies) do
        for key in pairs(record) do
            assert(M.ENEMY_FIELDS[key], 'enemy field not allowed in side view: ' .. name .. '.' .. key)
        end
        if record.visibility ~= 'visible' then
            assert(record.current == nil, 'hidden enemy has a current position: ' .. name)
        end
    end
    for pair, reading in pairs(view.range) do
        local target = pair:match('>(.+)$')
        local enemy = view.enemies[target]
        if enemy and enemy.visibility ~= 'visible' then
            assert(reading.access == 'withheld', 'range to a hidden enemy is not withheld: ' .. pair)
        end
    end
    return view
end

-- Counts for a compact progress line: visible / hidden / unknown enemies.
function M.visibility_counts(view)
    local counts = {visible = 0, not_visible = 0, unknown = 0}
    for _, record in pairs(view.enemies) do
        counts[record.visibility] = (counts[record.visibility] or 0) + 1
    end
    return counts
end

return M
