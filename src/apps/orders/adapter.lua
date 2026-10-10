-- Engine calls that give orders. Only recipes verified in battle are used:
-- docs/en/game/units/commands.md. An accepted call is not proof of execution;
-- check the result through units.state_adapter.
local facing = require('apps.orders.facing')

local M = {}

-- Script control of one unit through its own controller.
function M.take_control(army, unit)
    local uc = army:create_unit_controller()
    uc:add_units(unit)
    uc:take_control()
    return uc
end

function M.release(uc)
    pcall(function() uc:release_control() end)
end

-- Typical setup for scripted melee tests: no free fire, no skirmish.
function M.prepare_melee(uc, unit)
    uc:fire_at_will(false)
    if unit:can_use_behaviour('skirmish') then
        uc:change_behaviour_active('skirmish', false)
    end
    uc:melee(true)
end

function M.move(uc, position, run)
    uc:goto_location(position, run == true)
end

-- Arrive with a given front (degrees) and formation width (metres).
-- A lord accepts the width but one entity never forms ranks. The engine holds
-- facings on a grid (apps.orders.facing): the order goes for the grid facing
-- nearest to facing_deg, exactly.
function M.move_formation(uc, position, facing_deg, width_m, run)
    uc:goto_location_angle_width(position, facing.command(facing_deg), width_m, run == true)
end

-- Relative rotation.
function M.rotate(uc, degrees, run)
    uc:rotate(degrees, run == true)
end

-- Units decelerate; there is no instant stop.
function M.halt(uc)
    uc:halt()
end

-- At a run (the charge) unless walk is true (attack_unit's third argument, as attack_ranged).
function M.attack_melee(uc, enemy, walk)
    uc:melee(true)
    uc:attack_unit(enemy, false, walk ~= true)
end

-- Explicit fire at a target works even with free fire disabled.
-- attack_unit(target, primary weapon, run) — the arguments as CA's own script
-- library uses them (lib_battle_patrol_manager, lib_battle_script_unit):
-- run = false walks to the range (1.0-1.6 m/s for archers, 01.10.2026).
-- free_fire keeps fire at will on, so the unit is not left without a target
-- once the ordered one dies.
function M.attack_ranged(uc, enemy, run, free_fire)
    uc:melee(false)
    uc:fire_at_will(free_fire == true)
    uc:attack_unit(enemy, true, run == true)
end

function M.set_fire_at_will(uc, enabled)
    uc:fire_at_will(enabled == true)
end

-- fire_at_will(false) alone does NOT cancel an explicit fire order.
function M.stop_firing(uc)
    uc:halt()
    uc:fire_at_will(false)
end

-- 'defend' stops pursuit; it is not entrenchment and not braced state.
function M.set_guard(uc, unit, enabled)
    if not unit:can_use_behaviour('defend') then return false end
    uc:change_behaviour_active('defend', enabled == true)
    return true
end

-- Skirmish mode (the unit steps back from approaching enemies by itself, over its order): on or off.
-- false when the unit has no such mode (can_use_behaviour). CA's script library sets it the same way
-- (script_unit:change_behaviour_active('skirmish', ...)).
function M.set_skirmish(uc, unit, enabled)
    if not unit:can_use_behaviour('skirmish') then return false end
    uc:change_behaviour_active('skirmish', enabled == true)
    return true
end

-- Is skirmish mode on now (unit:is_behaviour_active)? nil when the unit has no such mode or it cannot be read.
function M.skirmish_active(unit)
    local ok, can = pcall(function() return unit:can_use_behaviour('skirmish') end)
    if not (ok and can) then return nil end
    local ok2, on = pcall(function() return unit:is_behaviour_active('skirmish') end)
    if ok2 and type(on) == 'boolean' then return on end
    return nil
end

-- Requires <can_withdraw>true</can_withdraw> in the scenario XML;
-- withdraw() without an argument errors in this build.
function M.withdraw(uc)
    uc:withdraw(true)
end

-- Lord self-target ability. The target argument is mandatory.
function M.use_ability_on_self(uc, unit, key)
    if not unit:can_perform_special_ability(key) then return false end
    uc:perform_special_ability(key, unit)
    return true
end

-- Diagnostic/deployment placement; applied on a following engine tick.
function M.teleport(uc, position, bearing_deg, width_m)
    uc:teleport_to_location(position, facing.command(bearing_deg), width_m)
end

return M
