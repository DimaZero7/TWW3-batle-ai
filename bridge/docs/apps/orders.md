# orders

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md)

Приказы отрядам: какими вызовами движка они отдаются и какие направления держит движок.
Код: `bridge/lua/apps/orders/`.

## adapter — проверенные вызовы

Только рецепты команд, проверенных в бою:

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
| `set_skirmish(uc, unit, on)` | `change_behaviour_active('skirmish', on)` — режим застрельщиков (отряд сам отходит от подходящего врага, перебивая приказ); `false`, если режима у отряда нет (`can_use_behaviour`) |
| `skirmish_active(unit)` | `is_behaviour_active('skirmish')`: включён ли режим сейчас; `nil`, если режима нет |
| `withdraw(uc)` | `withdraw(true)` |
| `use_ability_on_self(uc, unit, key)` | `perform_special_ability(key, unit)` |
| `teleport(uc, p, bearing, width)` | `teleport_to_location` |

## facing — направления, которые держит движок

`apps.orders.facing` (чистый): движок держит направление отряда в 64 секторах
по 5,625° и ставит его в середину сектора (замер в бою). `snap(b)` —
направление, которое движок удержит; `command(f)` — что послать; `teleport` и
`move_formation` посылают его сами.

<a id="planner_adapter"></a>

## planner_adapter — отряды ИИ игры

`hand_over(bm, name, alliance, units, enemies)` отдаёт отряды тактическому ИИ игры вместо наших приказов — так
генерированные бои CA ведут армию ИИ: `script_ai_planner:new` над скрипт-отрядами (`bm:get_scriptunit_for_unit`),
приказ `attack_force(враги)` (`attack()`) или `defend_position(центр, радиус)` (`defend()`); без библиотеки —
планировщик движка `alliance:create_ai_unit_planner()` с `attack_unit` ближайшего врага. `check_rallies()` (каждый
тик) возвращает планировщику собравшийся после бегства отряд и повторяет последний приказ (иначе он стоит до
конца боя), `check_idle(now_ms)` — отряд, стоящий без цели под обстрелом дольше 6 с: сперва назад в планировщик,
потом в свой планировщик «атаковать ближайшего». Это не общий боевой ИИ стороны 2: планировщик только исполняет
заданную задачу (атаковать / держать место), сам армию не расставляет и не обороняется по файлу боя. Входит в
[nn_arena](entries.md#nn_arena) с `own_ai = 'ai'`.

## Нападающий и защитник

**Кто нападает, а кто защищается, задаёт файл боя** (28.09.2026): защитник — тот
союз, что побеждает по таймауту, `<timeout_winning_alliance_index>` в
`<battle_description>`. Без этого тега движок считает нападающими обе стороны и
ИИ игры всегда атакует; с ним `is_attacker()` даёт нападающего и защитника, а
ИИ игры-защитник стоит у себя (8 минут игры, ни одного движения). Зоны
расстановки на роли не влияют (проверены половины карты и бой без зон).
`sim/arena.py` пишет этот тег в файл боя арены: защитник — ИИ игры, когда наша
сторона атакует (`--own-role attack`), и наша сторона, когда она обороняется. Против атакующего планировщика
(`own_ai = 'ai'`, только пехота этапа 0, 13.11) ИИ-защитник не стоял: шёл навстречу с 5–8 с боя, 113–224 м до
касания (`game/probe/evidence/aivai_game.json`).

## Что важно знать

- Принятый вызов — не доказательство выполнения. Результат проверяем по
  состоянию отряда (строка `nn_sample`, [точки входа](entries.md#nn_arena)).
- `fire_at_will(false)` **не отменяет** явный приказ стрелять: нужен ещё `halt()`.
- `defend` прекращает преследование, это не «окопаться».
- `withdraw` работает, только если в XML разрешён отход.
- Мгновенной остановки нет; отряд завершает движение в нескольких метрах
  от точки приказа.
