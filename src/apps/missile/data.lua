-- Measured missile damage (data only; the logic is apps.missile.services).
-- Source: docs/ru/game/units/missile-damage.md — archer-range --range-mode
-- damage, 28.09.2026, three runs with the distances moved between lanes.
-- status: 'measured' for this pair; other shooters and targets are scaled by
-- the card's missile damage ('scaled').
return {
    shooter = 'wh2_dlc13_emp_inf_archers_0', target = 'wh_main_emp_inf_spearmen_0',
    missile_damage = 19,
    arrows_per_man_s = 0.1,
    strength_power = 0.3,
    -- {distance from the shooters' middle to the target's front rank, HP per arrow at full strength}
    hp_per_arrow = {{70, 13.4}, {90, 11.6}, {110, 10.0}},
}
