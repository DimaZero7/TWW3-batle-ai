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

## Что важно знать

- Принятый вызов — не доказательство выполнения. Результат проверяем по
  состоянию отряда ([units](units.md)).
- `fire_at_will(false)` **не отменяет** явный приказ стрелять: нужен ещё `halt()`.
- `defend` прекращает преследование, это не «окопаться».
- `withdraw` работает, только если в XML разрешён отход.
- Мгновенной остановки нет; отряд завершает движение в нескольких метрах
  от точки приказа.
