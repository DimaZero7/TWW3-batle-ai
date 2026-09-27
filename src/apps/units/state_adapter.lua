-- Trusted read-only sensor of one own unit. Not part of policy API v1.
-- The caller establishes ownership; this does not authenticate an untrusted caller.
-- Field catalogue: data/units/field-catalog.json, docs/en/game/units/state-fields.md.
local value = require('apps.core.value')

local M = {version = 1}
local read, unknown = value.read, value.unknown

local NUMERIC = {'unary_hitpoints', 'number_of_men_alive', 'initial_number_of_men',
    'unary_of_men_alive', 'ammo_left', 'starting_ammo', 'bearing', 'ordered_bearing',
    'ordered_width', 'slow_speed', 'fast_speed'}
local BOOLEANS = {'is_routing', 'is_shattered', 'is_leaving_battle', 'is_moving',
    'is_moving_fast', 'is_in_melee', 'is_hidden', 'is_script_controlled', 'is_wavering',
    'is_under_missile_attack', 'is_idle', 'is_controllable', 'is_left_flank_threatened',
    'is_right_flank_threatened', 'is_rear_flank_threatened'}
local BEHAVIOURS = {'defend', 'skirmish', 'fire_at_will', 'change_formation_spacing'}
local UNIT_GETTERS = {'current_target', 'left_flank_threat', 'right_flank_threat', 'rear_threat'}
local CCO_NUMBERS = {'HealthValue', 'HealthMax', 'HealthPercent', 'PercentHpLostRecently',
    'DamageInflictedRecently', 'PrimaryAmmoPercent', 'MoralePercent', 'MoraleState',
    'FatigueState', 'NumKills', 'NumEntities', 'NumEntitiesInitial'}
local CCO_STRINGS = {'MoraleName', 'MoraleGreatestEffect', 'FatigueName'}
local CCO_BOOLEANS = {'IsFiringMissiles', 'IsRouting', 'IsShattered', 'IsAlive', 'IsWithdrawing',
    'IsAwaitingOrderAfterRally', 'IsOutOfControl', 'IsWavering', 'IsTakingDamage',
    'IsUnderMissileAttack', 'IsInLastStand'}

-- options: owned (true for own units), observer_alliance, cco(unit, key),
-- target_id(unit) -> public id of a visible unit.
function M.observe(unit, options)
    options = options or {}
    local result = {schema_version = 1, sensors = {}}
    if options.observer_alliance then
        result.visibility = read(function()
            return unit:is_visible_to_alliance(options.observer_alliance)
        end, 'boolean')
    else
        result.visibility = unknown('observer_missing')
    end
    -- New enemy sensor permissions require a separate reviewed schema. A visible
    -- enemy is not permission to disclose its orders, morale or threat indicators.
    if options.owned ~= true then
        result.access = 'withheld_own_only'
        return result
    end
    result.access = 'own'
    local s = result.sensors
    for _, key in ipairs(NUMERIC) do
        s['native.' .. key] = read(function() return unit[key](unit) end, 'number')
    end
    for _, key in ipairs(BOOLEANS) do
        s['native.' .. key] = read(function() return unit[key](unit) end, 'boolean')
    end
    s['native.fatigue_state'] = read(function() return unit:fatigue_state() end, 'string')
    s['native.position'] = value.vector(function() return unit:position() end)
    s['native.ordered_position'] = value.vector(function() return unit:ordered_position() end)
    for _, key in ipairs(BEHAVIOURS) do
        s['native.behaviour.' .. key] = read(function()
            return unit:is_behaviour_active(key)
        end, 'boolean')
    end
    -- Unit-returning getters only disclose an opaque id of a visible unit.
    for _, key in ipairs(UNIT_GETTERS) do
        s['native.' .. key] = read(function()
            local target = unit[key](unit)
            if not target then return '' end
            if not options.observer_alliance
                or target:is_visible_to_alliance(options.observer_alliance) ~= true then
                return nil
            end
            if not options.target_id then return nil end
            return options.target_id(target)
        end, 'string')
    end
    local function cco(key)
        if not options.cco then return nil end
        return options.cco(unit, key)
    end
    for _, key in ipairs(CCO_NUMBERS) do s['cco.' .. key] = read(function() return cco(key) end, 'number') end
    for _, key in ipairs(CCO_STRINGS) do s['cco.' .. key] = read(function() return cco(key) end, 'string') end
    for _, key in ipairs(CCO_BOOLEANS) do s['cco.' .. key] = read(function() return cco(key) end, 'boolean') end
    local size = read(function() return cco('StatusList.Size') end, 'number')
    s['cco.StatusList.Size'] = size
    if size.status == 'known' and size.value >= 0 and size.value % 1 == 0 and size.value <= 64 then
        local keys, complete = {}, true
        for i = 0, size.value - 1 do
            local entry = read(function() return cco('StatusList.At(' .. i .. ').Key') end, 'string')
            if entry.status == 'known' then keys[#keys + 1] = entry.value else complete = false end
        end
        s['cco.StatusList.Keys'] = complete and value.known(keys) or unknown('incomplete_status_list')
    else
        s['cco.StatusList.Keys'] = unknown('invalid_status_count')
    end
    -- Enumerated in the diagnostic; this is the card's number, not a claimed
    -- reconstruction of the engine's current panic threshold or MoralePercent.
    local stat_index
    local stat_count = read(function() return cco('UnitDetailsContext.StatList.Size') end, 'number')
    if stat_count.status == 'known' and stat_count.value >= 0 and stat_count.value % 1 == 0
        and stat_count.value <= 100 then
        for i = 0, stat_count.value - 1 do
            local key = read(function()
                return cco('UnitDetailsContext.StatList.At(' .. i .. ').Key')
            end, 'string')
            if key.status == 'known' and key.value == 'stat_morale' then stat_index = i; break end
        end
    end
    for _, field in ipairs({'Value', 'DisplayedValue', 'ValueBase'}) do
        s['cco.card.stat_morale.' .. field] = stat_index and read(function()
            return cco('UnitDetailsContext.StatList.At(' .. stat_index .. ').' .. field)
        end, 'number') or unknown('stat_key_unavailable')
    end
    return result
end

return M
