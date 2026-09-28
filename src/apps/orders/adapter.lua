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

function M.attack_melee(uc, enemy)
    uc:melee(true)
    uc:attack_unit(enemy, false, true)
end

-- Explicit fire at a target works even with free fire disabled.
function M.attack_ranged(uc, enemy)
    uc:melee(false)
    uc:fire_at_will(false)
    uc:attack_unit(enemy, true, false)
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

-- Applies one validated command (orders.contract). ctx supplies engine
-- lookups: controller(unit_id), unit(unit_id), enemy(target_id), vector(x, z).
function M.apply(command, ctx)
    local uc = ctx.controller(command.unit_id)
    if command.action == 'move' then
        local p = ctx.vector(command.x, command.z)
        if command.facing_deg then
            M.move_formation(uc, p, command.facing_deg, command.width_m, command.run)
        else
            M.move(uc, p, command.run)
        end
    elseif command.action == 'attack' then
        local enemy = ctx.enemy(command.target_id)
        if command.mode == 'ranged' then M.attack_ranged(uc, enemy) else M.attack_melee(uc, enemy) end
    elseif command.action == 'guard' then
        M.set_guard(uc, ctx.unit(command.unit_id), command.enabled)
    elseif command.action == 'halt' then
        M.halt(uc)
    else
        error('unsupported action: ' .. tostring(command.action))
    end
end

return M
