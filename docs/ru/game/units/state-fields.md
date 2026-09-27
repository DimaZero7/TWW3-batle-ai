# Проверенные поля состояния

[Общий справочник](state-sensors.md) · [English](../../../en/game/units/state-fields.md)

68 полей. Диапазоны относятся только к трём сохранённым опытам; они не являются пределами игры. «Только false» подтверждает чтение, но не переход в true. Число чтений объединяет три своих типа; полные детали доступны в JSON.

`JSON` (локальный архив: `research/evidence/units/unit-state-20260926/field-catalog.json`)

| Поле | Тип | Наблюдалось | Успешных чтений |
|---|---|---|---:|
| `cco.DamageInflictedRecently` | number | 0 … 831 | 7926 |
| `cco.FatigueName` | string | строки; см. JSON | 7926 |
| `cco.FatigueState` | number | 0 … 4 | 7926 |
| `cco.HealthMax` | number | 4068 … 8280 | 7926 |
| `cco.HealthPercent` | number/number | 0.001771337 … 1 | 7926 |
| `cco.HealthValue` | number | 0 … 8280 | 7926 |
| `cco.IsAlive` | boolean | false, true | 7926 |
| `cco.IsAwaitingOrderAfterRally` | boolean | false | 7926 |
| `cco.IsFiringMissiles` | boolean | false, true | 7926 |
| `cco.IsInLastStand` | boolean | false | 7926 |
| `cco.IsOutOfControl` | boolean | false | 7926 |
| `cco.IsRouting` | boolean | false, true | 7926 |
| `cco.IsShattered` | boolean | false, true | 7926 |
| `cco.IsTakingDamage` | boolean | false, true | 7926 |
| `cco.IsUnderMissileAttack` | boolean | false | 7926 |
| `cco.IsWavering` | boolean | false, true | 7926 |
| `cco.IsWithdrawing` | boolean | false, true | 7926 |
| `cco.MoraleGreatestEffect` | string | строки; см. JSON | 7926 |
| `cco.MoraleName` | string | строки; см. JSON | 7926 |
| `cco.MoralePercent` | number/number | -1.78 … 1.46 | 7926 |
| `cco.MoraleState` | number | 1 … 7 | 7926 |
| `cco.NumEntities` | number | 0 … 120 | 7926 |
| `cco.NumEntitiesInitial` | number | 1 … 120 | 7926 |
| `cco.NumKills` | number | 0 … 68 | 7926 |
| `cco.PercentHpLostRecently` | number/number | 0 … 0.1922705 | 7926 |
| `cco.PrimaryAmmoPercent` | number/number | 0 … 1 | 7926 |
| `cco.StatusList.Keys` | string[] (Lua) | 11 ключей | 7926 |
| `cco.StatusList.Size` | number | 0 … 4 | 7926 |
| `cco.card.stat_morale.DisplayedValue` | number | -55 … 75 | 1176 |
| `cco.card.stat_morale.Value` | number | -72 … 75 | 1176 |
| `cco.card.stat_morale.ValueBase` | number | -72 … 75 | 1176 |
| `native.ammo_left` | number | 0 … 1800 | 7926 |
| `native.bearing` | number/number | 0 … 359.9945 | 7926 |
| `native.behaviour.change_formation_spacing` | boolean | false | 7926 |
| `native.behaviour.defend` | boolean | false, true | 7926 |
| `native.behaviour.fire_at_will` | boolean | false, true | 7926 |
| `native.behaviour.skirmish` | boolean | false | 7926 |
| `native.current_target` | string | пусто / ID видимого | 7926 |
| `native.fast_speed` | number/number | 3 … 3.4 | 7926 |
| `native.fatigue_state` | string | строки; см. JSON | 7926 |
| `native.initial_number_of_men` | number | 1 … 120 | 7926 |
| `native.is_controllable` | boolean | false, true | 7926 |
| `native.is_hidden` | boolean | false | 7926 |
| `native.is_idle` | boolean | false, true | 7926 |
| `native.is_in_melee` | boolean | false, true | 7926 |
| `native.is_leaving_battle` | boolean | false, true | 7926 |
| `native.is_left_flank_threatened` | boolean | false, true | 7926 |
| `native.is_moving` | boolean | false, true | 7926 |
| `native.is_moving_fast` | boolean | false, true | 7926 |
| `native.is_rear_flank_threatened` | boolean | false, true | 7926 |
| `native.is_right_flank_threatened` | boolean | false, true | 7926 |
| `native.is_routing` | boolean | false, true | 7926 |
| `native.is_script_controlled` | boolean | true | 7926 |
| `native.is_shattered` | boolean | false, true | 7926 |
| `native.is_under_missile_attack` | boolean | false | 7926 |
| `native.is_wavering` | boolean | false, true | 7926 |
| `native.left_flank_threat` | string | пусто / ID видимого | 1176 |
| `native.number_of_men_alive` | number | 0 … 120 | 7926 |
| `native.ordered_bearing` | number/number | 0 … 359.8242 | 7926 |
| `native.ordered_position` | {x,y,z} | X/Y/Z | 7926 |
| `native.ordered_width` | number/number | 1.4 … 50 | 7926 |
| `native.position` | {x,y,z} | X/Y/Z | 7926 |
| `native.rear_threat` | string | пусто / ID видимого | 1176 |
| `native.right_flank_threat` | string | пусто / ID видимого | 1176 |
| `native.slow_speed` | number | 1.5 … 1.5 | 7926 |
| `native.starting_ammo` | number | 0 … 1800 | 7926 |
| `native.unary_hitpoints` | number/number | 0 … 1 | 7926 |
| `native.unary_of_men_alive` | number/number | 0 … 1 | 7926 |

Ошибки типа трёх методов угрозы в первой версии оболочки исключены из этой таблицы, но сохранены в исходных логах. Поля карточки морали встроены в модуль только для двух последних опытов; основной опыт сохраняет их отдельно как card_stat. Поля, не достигшие true (обстрел своих, ожидание приказа после возвращения, потеря контроля и другие), нельзя считать полноценно проверенными механиками.
