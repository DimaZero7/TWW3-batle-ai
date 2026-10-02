-- Minimal fake of the WH3 battle manager for entry smoke tests.
-- Implements only what src/entries use; callbacks run when the test pumps them.
local F = {}

local function vec(x, y, z)
    local v = {x = x or 0, y = y or 0, z = z or 0}
    function v:get_x() return self.x end
    function v:get_y() return self.y end
    function v:get_z() return self.z end
    function v:set_x(n) self.x = n end
    function v:set_y(n) self.y = n end
    function v:set_z(n) self.z = n end
    function v:distance(o) return math.sqrt((self.x - o.x) ^ 2 + (self.y - o.y) ^ 2 + (self.z - o.z) ^ 2) end
    return v
end
F.vector_type = {new = function() return vec() end}

local function list(items)
    return {count = function() return #items end, item = function(_, i) return items[i] end}
end

function F.unit(name, kind, x, z)
    local u = {script_name = name, kind = kind, pos = vec(x, 0, z), men = 120, controlled = false,
        routing = false}
    function u:name() return self.script_name end
    function u:type() return self.kind end
    function u:position() return self.pos end
    function u:unary_hitpoints() return self.men / 120 end
    function u:number_of_men_alive() return self.men end
    function u:initial_number_of_men() return 120 end
    function u:is_routing() return self.routing end
    function u:is_shattered() return false end
    function u:is_in_melee() return self.melee == true end
    function u:is_script_controlled() return self.controlled end
    function u:ammo_left() return self.ammo or 0 end
    function u:starting_ammo() return 0 end
    function u:is_behaviour_active() return false end
    function u:can_use_behaviour() return true end
    function u:is_valid_target() return true end
    function u:is_visible_to_alliance() return true end
    function u:is_idle() return true end
    -- Walks while a move or an attack is in force (set by the controller; a test may stop it).
    function u:is_moving() return self.moving == true end
    function u:is_commanding_unit() return false end
    function u:is_infantry() return true end
    function u:can_reach_position() return true end
    function u:unique_ui_id() return 'uid_' .. self.script_name end
    function u:has_attribute() return false end
    function u:is_commanding_unit() return false end
    function u:missile_range() return self.range or 0 end
    -- Abilities a test gives the unit: u.abilities[key] = true (ready) or false (owned, not ready).
    function u:can_perform_special_ability(key) return (self.abilities or {})[key] == true end
    function u:owned_non_passive_special_abilities()
        local keys = {}
        for k in pairs(self.abilities or {}) do keys[#keys + 1] = k end
        table.sort(keys)
        return keys
    end
    function u:owned_passive_special_abilities() return {} end
    function u:unit_distance() return 50 end
    function u:unit_in_range() return false end
    function u:bearing() return 0 end
    function u:ordered_bearing() return 0 end
    function u:ordered_width() return 30 end
    function u:ordered_position() return self.pos end
    function u:is_moving_fast() return false end
    function u:slow_speed() return 4 end
    function u:fast_speed() return 6 end
    return u
end

local function controller(log)
    local uc = {}
    function uc:add_units(u) self.unit = u end
    function uc:take_control() self.unit.controlled = true end
    function uc:release_control() self.unit.controlled = false end
    -- The last free-fire switch and attack_unit's arguments stay on the unit (the log keeps its old lines).
    function uc:fire_at_will(on) self.unit.free_fire = on end
    function uc:change_behaviour_active() end
    function uc:melee() end
    function uc:halt()
        self.unit.moving = false
        log[#log + 1] = 'halt'
    end
    function uc:attack_unit(enemy, primary, run)
        self.unit.attack_args = {target = enemy and enemy:name(), primary = primary, run = run}
        -- a melee unit walks to its target; a shooter (ammunition, range) stands and shoots
        self.unit.moving = not ((self.unit.range or 0) > 0 and (self.unit.ammo or 0) > 0)
        log[#log + 1] = 'attack' .. (enemy and (' ' .. enemy:name()) or '')
    end
    function uc:teleport_to_location() end
    function uc:goto_location(p, run)
        self.unit.moving = true
        log[#log + 1] = string.format('goto %s %g %g %s', self.unit:name(), p:get_x(), p:get_z(), tostring(run))
    end
    -- Arrives at once: entry tests check wiring, not movement.
    function uc:goto_location_angle_width(p) self.unit.pos = vec(p:get_x(), p:get_y(), p:get_z()) end
    function uc:rotate() end
    -- Like the engine: the target is mandatory. The ability is then on recharge.
    function uc:perform_special_ability(key, target)
        assert(target, 'perform_special_ability needs a target')
        if self.unit.abilities then self.unit.abilities[key] = false end
        log[#log + 1] = 'ability ' .. self.unit:name() .. ' ' .. key .. ' on ' .. target:name()
    end
    return uc
end

-- The engine's AI unit planner: records what it is told in log.
local function ai_planner(log)
    local p = {}
    function p:add_units(u) log[#log + 1] = 'add ' .. u:name() end
    function p:remove_units(u) log[#log + 1] = 'remove ' .. u:name() end
    function p:attack_unit(u) log[#log + 1] = 'attack ' .. u:name() end
    function p:defend_position(_, radius) log[#log + 1] = 'defend ' .. radius end
    return p
end

-- sides: {{unit, ...}, {unit, ...}}
function F.manager(sides)
    local bm = {now = 0, phase = 'Deployment', speed = 1, phase_callbacks = {}, queue = {},
        orders = {}, planner_log = {}, outcome = false, winner = 0, ended = false}
    local alliances = {}
    for i, units in ipairs(sides) do
        local army = {units = function() return list(units) end,
            create_unit_controller = function() return controller(bm.orders) end}
        alliances[i] = {armies = function() return list({army}) end,
            is_attacker = function() return i == 1 end,
            create_ai_unit_planner = function() return ai_planner(bm.planner_log) end}
    end
    for _, m in ipairs({'is_from_campaign', 'is_multiplayer', 'is_replay', 'is_quest_battle',
        'is_tutorial', 'is_siege_battle', 'is_ambush_battle'}) do
        bm[m] = function() return false end
    end
    function bm:alliances() return list(alliances) end
    function bm:time_elapsed_ms() return self.now end
    function bm:get_current_phase_name() return self.phase end
    function bm:register_phase_change_callback(phase, fn)
        self.phase_callbacks[phase] = self.phase_callbacks[phase] or {}
        table.insert(self.phase_callbacks[phase], fn)
    end
    function bm:set_phase(phase)
        self.phase = phase
        for _, fn in ipairs(self.phase_callbacks[phase] or {}) do fn() end
    end
    function bm:callback(fn) table.insert(self.queue, fn) end
    -- Short real-time callbacks run on pump(); long ones (deadlines, >= 5 s)
    -- wait until the test calls fire_timers().
    bm.timers = {}
    function bm:real_callback(fn, ms, name)
        if (ms or 0) >= 5000 then self.timers[name or fn] = fn else table.insert(self.queue, fn) end
    end
    function bm:fire_timers()
        local due = self.timers
        self.timers = {}
        for _, fn in pairs(due) do fn() end
    end
    -- Repeating callbacks run on tick() once their interval has passed (at most once a tick).
    bm.repeating, bm.intervals, bm.last_run = {}, {}, {}
    function bm:repeat_callback(fn, ms, name)
        local key = name or fn
        self.repeating[key], self.intervals[key], self.last_run[key] = fn, ms or 0, self.now
    end
    function bm:remove_process(name) self.repeating[name] = nil end
    function bm:remove_real_callback(name) self.timers[name] = nil end
    function bm:end_current_battle_phase() self:set_phase('Deployed') end
    function bm:current_battle_speed() return self.speed end
    function bm:modify_battle_speed(s) self.speed = s end
    function bm:out() end
    function bm:add_infotext() end
    function bm:battle_outcome_decided() return self.outcome end
    function bm:victorious_alliance() return self.winner end
    function bm:force_battle_end() self.ended = true end
    function bm:end_battle() self.ended = true end
    function bm:change_victory_countdown_limit() end
    function bm:output_battle_xml(path) local f = assert(io.open(path, 'w')); f:write('<battle/>'); f:close() end
    function bm:get_terrain_height(x, z) return (x + z) / 100 end
    function bm:is_area_clear() return true end
    function bm:ground_type() return 'grass' end
    function bm:buildings() return list({}) end
    -- Runs queued callbacks until none are left (bounded).
    function bm:pump(limit)
        for _ = 1, limit or 10000 do
            local fn = table.remove(self.queue, 1)
            if not fn then return end
            fn()
        end
        error('callback queue did not drain')
    end
    function bm:tick(ms)
        self.now = self.now + (ms or 1000)
        local names = {}
        for name in pairs(self.repeating) do names[#names + 1] = name end
        table.sort(names, function(a, b) return tostring(a) < tostring(b) end)
        for _, name in ipairs(names) do
            if self.repeating[name] and self.now - self.last_run[name] >= self.intervals[name] then
                self.last_run[name] = self.now
                self.repeating[name]()
            end
        end
    end
    return bm
end

-- Per-unit context values a test sets: F.cco[unique_ui_id][field] (e.g. IsFiringMissiles).
F.cco = {}
-- Root context values a test sets: F.root[path] (e.g. 'BattleRoot.BalanceOfPowerPercent').
F.root = {}

-- Unrotated radar over x in [-100, 100], z in [-100, 100].
F.common = {
    get_context_value = function(key, id, field)
        if key == 'CcoBattleUnit' and F.cco[id] and F.cco[id][field] ~= nil then return F.cco[id][field] end
        if id == nil and F.root[key] ~= nil then return F.root[key] end
        -- Two soldiers per unit; Position returns three numbers like the engine.
        if key == 'CcoBattleUnit' and field == 'ManList.Size' then return 2 end
        -- A two-row unit card.
        if key == 'CcoBattleUnit' and field == 'UnitDetailsContext.StatList.Size' then return 2 end
        if key == 'CcoBattleUnit' and field == 'UnitDetailsContext.StatList.At(0).Key' then return 'stat_armour' end
        if key == 'CcoBattleUnit' and field == 'UnitDetailsContext.StatList.At(1).Key' then return 'stat_morale' end
        if key == 'CcoBattleUnit' and field and field:match('^UnitDetailsContext%.StatList%.At%(%d%)%.Value$') then return 30 end
        if key == 'CcoBattleUnit' and field == 'UnitDetailsContext.Mass' then return 60 end
        if key == 'CcoBattleUnit' and field and field:match('^ManList%.At%(%d+%)%.Position$') then
            return 1.25, 0, -2.5
        end
        local x, z = key:match('ToVector4%(([%-%d%.]+),0,([%-%d%.]+),0%)')
        if x then return (tonumber(x) + 100) / 200, (100 - tonumber(z)) / 200 end
        if key == 'BattleRoot.BuildingsList.Size' then return 0 end
        return nil
    end,
    game_version = function() return 'fake' end,
}

return F
