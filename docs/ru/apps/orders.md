# orders

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/orders.md)

Команды отрядам: какие бывают, как проверяются и какими вызовами движка
выполняются. Код: `src/apps/orders/`.

## contract — форма команды

| Действие | Поля |
|---|---|
| `move` | `unit_id`, `x`, `z`, `run?`; с контрактом движения ещё `facing_deg`, `width_m` |
| `attack` | `unit_id`, `target_id`, `mode = 'melee' \| 'ranged'` |
| `guard` | `unit_id`, `enabled` |
| `halt` | `unit_id` |

`validate(commands, own, enemies, limit, movement_contract?)` отклоняет
весь набор, если хотя бы одна команда неверна:

- плотный массив не длиннее `limit`, никаких лишних полей;
- отряд свой и жив; на отряд не больше одного движения (`move`/`attack`/`halt`)
  и одного `guard`;
- цель атаки **видима**; `ranged` — только лучники с боезапасом;
- `facing_deg` / `width_m` — только с контрактом `formation-move-v1` или
  `deployment-march-formation-v1`, ширина в рамках `units.contract`.

## adapter — проверенные вызовы

Только рецепты из [команд, проверенных в бою](../game/units/commands.md):

| Функция | Вызовы движка |
|---|---|
| `take_control(army, unit)` | `create_unit_controller`, `add_units`, `take_control` |
| `prepare_melee(uc, unit)` | `fire_at_will(false)`, отключить `skirmish`, `melee(true)` |
| `move`, `move_formation`, `rotate`, `halt` | `goto_location`, `goto_location_angle_width`, `rotate`, `halt` |
| `attack_melee(uc, enemy)` | `melee(true)`, `attack_unit(enemy, false, true)` |
| `attack_ranged(uc, enemy)` | `melee(false)`, `fire_at_will(false)`, `attack_unit(enemy, true, false)` |
| `stop_firing(uc)` | `halt()` + `fire_at_will(false)` |
| `set_guard(uc, unit, on)` | `change_behaviour_active('defend', on)` |
| `withdraw(uc)` | `withdraw(true)` |
| `use_ability_on_self(uc, unit, key)` | `perform_special_ability(key, unit)` |
| `teleport(uc, p, bearing, width)` | `teleport_to_location` |
| `apply(command, ctx)` | Выполняет одну проверенную команду |

## planner_adapter — отдать отряды штатному ИИ

`hand_over(bm, name, alliance, units, enemies)` передаёт отряды тактическому
ИИ игры вместо наших приказов. Использует `script_ai_planner` из библиотеки
CA (`lib_battle_script_ai_planner.lua`) — тот же механизм, которым
генерируемые бои ведут армии ИИ. Он оборачивает
`alliance:create_ai_unit_planner()`; без библиотеки адаптер обращается к
этому планировщику напрямую. Возвращает `{mode, attack(), release()}`:
`attack()` — «атаковать вражескую армию», повторяется точкой входа.

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
`tools/sim/formation.py` пишет этот тег всегда: защитник — сторона с ролью `defend`.

## facing — направления, которые держит движок

`apps.orders.facing` (чистый): движок держит направление отряда в 64 секторах
по 5,625° и ставит его в середину сектора
([замер](../game/units/commands.md#направление-отряда-64-сектора)). `snap(b)` —
направление, которое движок удержит (на нём планируется строй: `apps.plan`
приводит к нему `bearing`); `command(f)` — что послать; `teleport` и
`move_formation` посылают его сами.

## Что важно знать

- Принятый вызов — не доказательство выполнения. Результат проверяем по
  состоянию отряда ([units](units.md)).
- `fire_at_will(false)` **не отменяет** явный приказ стрелять: нужен ещё `halt()`.
- `defend` прекращает преследование, это не «окопаться».
- `withdraw` работает, только если в XML разрешён отход.
- Мгновенной остановки нет; отряд завершает движение в нескольких метрах
  от точки приказа.
