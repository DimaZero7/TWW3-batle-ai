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
    return v
end
F.vector_type = {new = function() return vec() end}

local function list(items)
    return {count = function() return #items end, item = function(_, i) return items[i] end}
end

function F.unit(name, kind, x, z)
    local u = {script_name = name, kind = kind, pos = vec(x, 0, z), men = 120, controlled = false,
        routing = false, attacked = 0}
    function u:name() return self.script_name end
    function u:type() return self.kind end
    function u:position() return self.pos end
    function u:unary_hitpoints() return self.men / 120 end
    function u:number_of_men_alive() return self.men end
    function u:initial_number_of_men() return 120 end
    function u:is_routing() return self.routing end
    function u:is_shattered() return false end
    function u:is_in_melee() return false end
    function u:is_script_controlled() return self.controlled end
    function u:ammo_left() return 0 end
    function u:starting_ammo() return 0 end
    function u:is_behaviour_active() return false end
    function u:can_use_behaviour() return true end
    function u:is_valid_target() return true end
    function u:is_visible_to_alliance() return true end
    function u:is_idle() return true end
    function u:is_commanding_unit() return false end
    function u:is_infantry() return true end
    function u:can_reach_position() return true end
    return u
end

local function controller(log)
    local uc = {}
    function uc:add_units(u) self.unit = u end
    function uc:take_control() self.unit.controlled = true end
    function uc:release_control() self.unit.controlled = false end
    function uc:fire_at_will() end
    function uc:change_behaviour_active() end
    function uc:melee() end
    function uc:halt() log[#log + 1] = 'halt' end
    function uc:attack_unit(target) self.unit.attacked = self.unit.attacked + 1; log[#log + 1] = 'attack' end
    function uc:teleport_to_location() end
    return uc
end

-- sides: {{unit, ...}, {unit, ...}}
function F.manager(sides)
    local bm = {now = 0, phase = 'Deployment', speed = 1, phase_callbacks = {}, queue = {},
        orders = {}, outcome = false, winner = 0, ended = false}
    local alliances = {}
    for i, units in ipairs(sides) do
        local army = {units = function() return list(units) end,
            create_unit_controller = function() return controller(bm.orders) end}
        alliances[i] = {armies = function() return list({army}) end,
            is_attacker = function() return i == 1 end}
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
    function bm:real_callback(fn) table.insert(self.queue, fn) end
    function bm:repeat_callback(fn) self.repeating = fn end
    function bm:remove_process() self.repeating = nil end
    function bm:remove_real_callback() end
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
        if self.repeating then self.repeating() end
    end
    return bm
end

-- Unrotated radar over x in [-100, 100], z in [-100, 100].
F.common = {
    get_context_value = function(key)
        local x, z = key:match('ToVector4%(([%-%d%.]+),0,([%-%d%.]+),0%)')
        if x then return (tonumber(x) + 100) / 200, (100 - tonumber(z)) / 200 end
        if key == 'BattleRoot.BuildingsList.Size' then return 0 end
        return nil
    end,
    game_version = function() return 'fake' end,
}

return F
