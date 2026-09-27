# units

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/units.md)

Состояние своих отрядов, дальность стрельбы и общие правила ширины строя.
Код: `src/apps/units/`. Что проверено в игре — в [знаниях об отрядах](../game/units/README.md).

## state_adapter — сенсор своего отряда

```lua
local state = require('apps.units.state_adapter')
local r = state.observe(unit, {
    owned = true,                    -- только свои отряды
    observer_alliance = my_alliance,
    cco = function(u, key) ... end,  -- чтение CcoBattleUnit
    target_id = function(u) ... end, -- публичный id видимого отряда
})
-- r.sensors['native.number_of_men_alive'] = {status='known', value=87}
```

- 68 полей: здоровье, численность, боезапас, скорость, движение, бой,
  бегство, фланги, поведение, мораль и усталость из CCO, статусы.
  Каталог — [поля состояния](../game/units/state-fields.md) и
  `data/units/field-catalog.json`.
- Для чужого отряда (`owned ~= true`) возвращается только видимость:
  `access = 'withheld_own_only'`, других чтений нет.
- Цель и угрозы флангов отдаются только как id **видимого** отряда.

## range_adapter — дальность стрельбы

`observe(source, target?, context)` — дальность движка и карточки,
`unit_in_range`, расстояние между отрядами. Сначала проверяются оба конца:
союзник или видимый враг. Иначе `access = 'withheld'` и **ни одного** чтения
дальности, позиции или CCO. Подробно — [дальность стрельбы](../game/units/missile-range.md).

## contract

| Функция | Что делает |
|---|---|
| `width_bounds(kind)` | Лорд 3–8 м, строй 20–40 м. Это инженерные рамки ввода, не измеренные пределы игры |
| `valid_width(kind, width)` | Ширина в рамках |
| `can_shoot(unit)` | Лучники с положительным боезапасом |

Эти рамки раньше были продублированы в `deployment` и `policy_host`.
