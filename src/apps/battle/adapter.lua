-- Battle context read from the engine: sides, armies, units and roles.
-- The only place that walks bm:alliances(); other apps receive plain lists.
local M = {version = 1}

-- Battle kinds this project does not drive. Order matters for the report.
M.UNSUPPORTED = {'is_from_campaign', 'is_multiplayer', 'is_replay', 'is_quest_battle',
    'is_tutorial', 'is_siege_battle', 'is_ambush_battle'}

-- Returns the first unsupported battle kind, or nil for a plain custom battle.
function M.unsupported_reason(bm)
    for _, method in ipairs(M.UNSUPPORTED) do
        if bm[method](bm) then return method end
    end
    return nil
end

-- Native attacker role per alliance. Never inferred from index or spawn.
-- record(evidence) is optional and receives what the engine returned.
function M.read_roles(bm, record)
    local alliances = bm:alliances()
    assert(alliances:count() == 2, 'native role requires exactly two alliances')
    local roles, raw = {}, {}
    for i = 1, 2 do
        local ok, value = pcall(function() return alliances:item(i):is_attacker() end)
        if not (ok and type(value) == 'boolean') then
            if record then
                record({version = 1, source = 'alliance:is_attacker()', failed_alliance = i,
                    return_type = ok and type(value) or 'error'})
            end
            error('native attacker role unavailable')
        end
        raw[i] = value
        roles[i] = {version = 1, role = value and 'attacker' or 'defender'}
    end
    local evidence = {version = 1, source = 'alliance:is_attacker()', is_attacker = raw}
    if record then record(evidence) end
    assert(raw[1] ~= raw[2], 'conflicting native attacker roles')
    return roles, evidence
end

-- Returns sides[1..2] = {side, alliance, army, units = {unit...}}.
-- expected_units (optional) asserts the exact unit count per side.
function M.read_sides(bm, expected_units)
    local alliances = bm:alliances()
    assert(alliances:count() == 2, 'battle requires exactly two alliances')
    local sides = {}
    for side = 1, 2 do
        local alliance = alliances:item(side)
        local armies = alliance:armies()
        assert(armies:count() == 1, 'battle requires one army per alliance')
        local army = armies:item(1)
        local list = army:units()
        if expected_units then
            assert(list:count() == expected_units,
                'expected ' .. expected_units .. ' units on side ' .. side)
        end
        local units = {}
        for i = 1, list:count() do units[i] = list:item(i) end
        sides[side] = {side = side, alliance = alliance, army = army, units = units}
    end
    return sides
end

-- Finds a unit by its XML script_name on the given side.
function M.find_by_name(side_info, script_name)
    for _, unit in ipairs(side_info.units) do
        if unit:name() == script_name then return unit end
    end
    return nil
end

return M
