-- Strategy choice (pure): a catalogue of strategies, each with its conditions,
-- score, unit roles and the formation layout it asks for. Descriptions:
-- docs/ru/architecture/strategies.md. Reads only assessment features; knows
-- nothing about geometry (that is apps.formation).
local M = {}

-- Roles shared with apps.formation layouts.
M.ROLES = {wall = true, arc = true, lord = true, other = true}

local function check(results, id, ok, value)
    results[#results + 1] = {id = id, ok = ok == true, value = value}
end

M.CATALOG = {
    {
        key = 'wall_and_arc',
        layout = 'line_and_blocks',
        params = {max_enemy_armour = 50, min_arc_share = 0.5},
        formation = {},
        -- The six conditions of docs/ru/architecture/strategies.md#wall_and_arc.
        conditions = function(f, p)
            local r = {}
            check(r, 'enemy_infantry_stronger', f.enemy.melee_power > f.own.melee_power,
                {own = f.own.melee_power, enemy = f.enemy.melee_power})
            check(r, 'own_infantry_to_hold', f.own.count.infantry >= 1, f.own.count.infantry)
            check(r, 'more_archers', f.own.ranged_power > f.enemy.ranged_power,
                {own = f.own.ranged_power, enemy = f.enemy.ranged_power})
            check(r, 'archers_pierce', f.enemy.infantry_armour ~= nil and f.enemy.infantry_armour <= p.max_enemy_armour,
                f.enemy.infantry_armour)
            check(r, 'arc_fire', f.own.arc_share >= p.min_arc_share, f.own.arc_share)
            check(r, 'we_attack', f.role == 'attack', f.role)
            return r
        end,
        -- Bigger fire advantage, better (draft; to be tuned by battles).
        score = function(f) return f.ranged_ratio or 0 end,
        role = function(category, u)
            if category == 'lord' then return 'lord' end
            if category == 'infantry' then return 'wall' end
            if category == 'shooter' and u.fire == 'arc' then return 'arc' end
            return 'other'
        end,
    },
}

function M.find(key)
    for _, entry in ipairs(M.CATALOG) do
        if entry.key == key then return entry end
    end
    return nil
end

-- Every strategy with its conditions; the best one whose conditions all hold.
function M.select(features, catalog)
    catalog = catalog or M.CATALOG
    local candidates, best = {}, nil
    for _, entry in ipairs(catalog) do
        local results = entry.conditions(features, entry.params)
        local failed = {}
        for _, c in ipairs(results) do
            if not c.ok then failed[#failed + 1] = c.id end
        end
        local candidate = {key = entry.key, conditions = results, failed = failed,
            score = #failed == 0 and entry.score(features) or nil}
        candidates[#candidates + 1] = candidate
        if candidate.score and (not best or candidate.score > best.score) then best = candidate end
    end
    return best and best.key or 'none', candidates
end

-- roles[id] = role for the chosen strategy; category(u) comes from apps.assessment.
function M.assign_roles(entry, units, category)
    local roles = {}
    for _, u in ipairs(units) do roles[u.id] = entry.role(category(u), u) end
    return roles
end

-- decision = {strategy, layout, formation_params, roles, candidates}; strategy = 'none' when nothing fits.
function M.decide(features, units, category, catalog)
    local key, candidates = M.select(features, catalog)
    local decision = {strategy = key, candidates = candidates}
    if key == 'none' then return decision end
    local entry = M.find(key)
    decision.layout, decision.formation_params = entry.layout, entry.formation
    decision.roles = M.assign_roles(entry, units, category)
    return decision
end

return M
