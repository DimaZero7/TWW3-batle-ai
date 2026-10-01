# orders

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/orders.md)

Приказы отрядам: какими вызовами движка они отдаются, как отдать отряды штатному
ИИ игры и какие направления держит движок. Код: `src/apps/orders/`.

## adapter — проверенные вызовы

Только рецепты из [команд, проверенных в бою](../game/units/commands.md):

| Функция | Вызовы движка |
|---|---|
| `take_control(army, unit)` | `create_unit_controller`, `add_units`, `take_control` |
| `release(uc)` | `release_control` |
| `prepare_melee(uc, unit)` | `fire_at_will(false)`, отключить `skirmish`, `melee(true)` |
| `move`, `move_formation`, `rotate`, `halt` | `goto_location`, `goto_location_angle_width`, `rotate`, `halt` |
| `attack_melee(uc, enemy)` | `melee(true)`, `attack_unit(enemy, false, true)` |
| `attack_ranged(uc, enemy, run, free_fire)` | `melee(false)`, `fire_at_will(free_fire)`, `attack_unit(enemy, true, run)`; `attack_unit(цель, основное оружие, бег)`: аргументы так, как их передаёт библиотека скриптов CA |
| `set_fire_at_will(uc, on)` | `fire_at_will(on)` |
| `stop_firing(uc)` | `halt()` + `fire_at_will(false)` |
| `set_guard(uc, unit, on)` | `change_behaviour_active('defend', on)` |
| `withdraw(uc)` | `withdraw(true)` |
| `use_ability_on_self(uc, unit, key)` | `perform_special_ability(key, unit)` |
| `teleport(uc, p, bearing, width)` | `teleport_to_location` |

## planner_adapter — отдать отряды штатному ИИ

`hand_over(bm, name, alliance, units, enemies)` передаёт отряды тактическому
ИИ игры вместо наших приказов. Использует `script_ai_planner` из библиотеки
CA (`lib_battle_script_ai_planner.lua`) — тот же механизм, которым
генерируемые бои ведут армии ИИ. Он оборачивает
`alliance:create_ai_unit_planner()`; без библиотеки адаптер обращается к
этому планировщику напрямую. Возвращает `{mode, attack(), defend(центр, радиус),
check_rallies(), check_idle(мс), release()}`: `attack()` — «атаковать вражескую армию»,
повторяется точкой входа. Им пользуются [ai_vs_ai](entries.md#ai_vs_ai) и
[nn_arena](entries.md#nn_arena).

**Что планировщик CA сам не делает, и как мы это чиним** (записи арены, 30.09.2026):

- `check_rallies()` — отряд, который побежал и собрался, планировщик больше не ведёт: без
  этого он стоял без дела до конца боя (ИИ игры свои собравшиеся отряды снова ведёт в бой).
  Такой отряд вынимается из планировщика, возвращается в него, и через 0,6 с повторяется
  последний приказ.
- `check_idle(мс)` — отряд стоит без дела под стрелами: не идёт, не в рукопашной, нет цели,
  теряет здоровье. У наших отрядов так было 15–75% времени, когда их обстреливали, у отрядов
  ИИ игры — 0–3%. Через 6 с такого простоя — назад в планировщик с приказом; если через 6 с он
  снова стоит под обстрелом — свой планировщик с приказом атаковать ближайшего врага. Не чаще
  раза в 15 с на отряд. Проверено тестами `tests/apps/orders/test_planner_adapter.py`.

### Штатный ИИ с задачей (27.09.2026)

У `script_ai_planner` в бою есть, среди прочего: `attack_force`, `attack_unit`,
`defend_force`, `defend_position`, `move_to_position`, `move_to_force`,
`rush_position`, `set_patrol_defend_radius`, `release`. Проверен
`defend_position(позиция, радиус)`: штатный ИИ держит место
([как он стоит](../../../research/analysis/enemy-layout/README.md)).

**Кто нападает, а кто защищается, задаёт файл боя** (28.09.2026): защитник — тот
союз, что побеждает по таймауту, `<timeout_winning_alliance_index>` в
`<battle_description>`. Без этого тега движок считает нападающими обе стороны и
штатный ИИ всегда атакует; с ним `is_attacker()` даёт нападающего и защитника, а
штатный ИИ-защитник стоит у себя (8 минут игры, ни одного движения). Зоны
расстановки на роли не влияют (проверены половины карты и бой без зон).
`tools/nn/scenario.py` пишет этот тег в сценарий арены: защитник — ИИ игры, когда наша
сторона атакует, и наша сторона, когда она обороняется.
Все варианты и замеры — [нападающий и защитник](../game/battle-roles.md).

## facing — направления, которые держит движок

`apps.orders.facing` (чистый): движок держит направление отряда в 64 секторах
по 5,625° и ставит его в середину сектора
([замер](../game/units/commands.md#направление-отряда-64-сектора)). `snap(b)` —
направление, которое движок удержит; `command(f)` — что послать; `teleport` и
`move_formation` посылают его сами.

## Что важно знать

- Принятый вызов — не доказательство выполнения. Результат проверяем по
  состоянию отряда ([units](units.md)).
- `fire_at_will(false)` **не отменяет** явный приказ стрелять: нужен ещё `halt()`.
- `defend` прекращает преследование, это не «окопаться».
- `withdraw` работает, только если в XML разрешён отход.
- Мгновенной остановки нет; отряд завершает движение в нескольких метрах
  от точки приказа.
