-- Missile damage (pure): how many hit points a shooting block takes off its
-- target per second, and a step of a missile exchange for the simulation.
--
-- Measured 28.09.2026 (archer-range --range-mode damage, three runs, the
-- distances moved between lanes; docs/ru/game/units/missile-damage.md): Empire
-- archers (90 men, 20 arrows each) at a fearless Empire spearmen unit (120 men,
-- 8280 HP, 30 m wide, facing them) standing still:
--   * every man looses about 0.1 arrow a second (9 a second for the unit);
--   * hit points per arrow = c(distance) * h^0.3, h = the target's share of
--     hit points left (a thinned block is hit less often);
--   * c: 13.4 at 70 m, 11.6 at 90 m, about 10 at 105-120 m (from the archers'
--     middle to the target's front rank);
--   * half the unit is gone in 33-67 s, all of it in 108-187 s.
-- Other shooters and targets are not measured yet: their numbers scale by the
-- card's missile damage (19 for the measured archers) and are marked 'scaled'.
local value = require('apps.core.value')
local reach = require('apps.reach.services')

local M = {}
local finite = value.finite

M.MEASURED = {
    shooter = 'wh2_dlc13_emp_inf_archers_0', target = 'wh_main_emp_inf_spearmen_0',
    missile_damage = 19,
    arrows_per_man_s = 0.1,
    strength_power = 0.3,
    -- {distance from the shooters' middle to the target's front rank, HP per arrow at full strength}
    hp_per_arrow = {{70, 13.4}, {90, 11.6}, {110, 10.0}},
}

-- HP per arrow at a distance, at full strength of the target (flat outside the measured span).
function M.hp_per_arrow(distance_m, missile_damage)
    local t = M.MEASURED.hp_per_arrow
    local v
    if distance_m <= t[1][1] then v = t[1][2]
    elseif distance_m >= t[#t][1] then v = t[#t][2]
    else
        for i = 1, #t - 1 do
            local a, b = t[i], t[i + 1]
            if distance_m <= b[1] then
                v = a[2] + (b[2] - a[2]) * (distance_m - a[1]) / (b[1] - a[1])
                break
            end
        end
    end
    if finite(missile_damage) and missile_damage > 0 then
        v = v * missile_damage / M.MEASURED.missile_damage
    end
    return v
end

-- Whether the pair is the measured one.
function M.calibrated(shooter, target)
    return shooter.key == M.MEASURED.shooter and target.key == M.MEASURED.target
end

-- HP a second the shooter takes off the target at distance_m. A unit =
-- {key, men, hp, hp_max, ammo, missile_damage}.
function M.rate(shooter, target, distance_m)
    if not (shooter.men and shooter.men > 0 and (shooter.ammo or 0) > 0 and target.hp and target.hp > 0) then
        return 0
    end
    local h = target.hp / target.hp_max
    return shooter.men * M.MEASURED.arrows_per_man_s
        * M.hp_per_arrow(distance_m, shooter.missile_damage) * h ^ M.MEASURED.strength_power
end

-- The nearest block of the other side the shooter reaches, and the distance.
function M.target_of(shooter, others, params)
    local best, best_d
    for _, t in ipairs(others) do
        if (t.hp or 0) > 0 then
            local ok, d = reach.reaches(shooter, t, params)
            if ok and (not best_d or d < best_d) then best, best_d = t, d end
        end
    end
    return best, best_d
end

-- One step of a missile exchange: every shooter with arrows fires at the
-- nearest block of the other side it reaches (as fire at will does). Damage is
-- applied after everyone has shot. units: blocks (apps.reach) with side,
-- men, men_max, hp, hp_max, ammo, range_m, key, missile_damage. Changes them in
-- place; returns {{shooter, target, distance_m, hp, arrows, calibrated}}.
function M.step(units, dt, params)
    assert(finite(dt) and dt > 0, 'dt required')
    local shots, damage = {}, {}
    for _, s in ipairs(units) do
        if finite(s.range_m) and s.range_m > 0 and (s.men or 0) > 0 and (s.ammo or 0) > 0 then
            local others = {}
            for _, u in ipairs(units) do
                if u.side ~= s.side then others[#others + 1] = u end
            end
            local t, d = M.target_of(s, others, params)
            if t then
                local arrows = math.min(s.ammo, s.men * M.MEASURED.arrows_per_man_s * dt)
                local hp = M.rate(s, t, d) * dt * arrows / (s.men * M.MEASURED.arrows_per_man_s * dt)
                damage[t] = (damage[t] or 0) + hp
                s.ammo = s.ammo - arrows
                shots[#shots + 1] = {shooter = s.id, target = t.id, distance_m = d, hp = hp, arrows = arrows,
                    calibrated = M.calibrated(s, t)}
            end
        end
    end
    for t, hp in pairs(damage) do
        t.hp = math.max(0, t.hp - hp)
        t.men = math.ceil(t.hp / (t.hp_max / t.men_max) - 1e-9)
    end
    return shots
end

return M
