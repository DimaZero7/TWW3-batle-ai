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

## card_adapter — карточка и профиль отряда

| Функция | Что делает |
|---|---|
| `stats(cco, unit)` | Вся карточка `UnitDetailsContext.StatList` по порядку: `{key, Value, DisplayedValue, ValueBase}` |
| `details(cco, unit)` | `UnitDetailsContext.Mass`, `Name` и поля отряда `HealthMax`, `NumEntitiesInitial` |
| `profile(unit)` | Тип, командир, класс, вид отряда (пехота, конница…), бойцов, скорости, дальность, стрелы, способности |

Неизвестное пишется как `'unknown:<причина>'`. Используется прогоном
[сбора ростера](../game/units/roster.md).

## formation_adapter — движение и бойцы своего отряда

| Функция | Что делает |
|---|---|
| `motion(unit)` | Позиция, заданная позиция, направление, заданные направление и ширина, число бойцов, флаги движения. Нечитаемое пропускается; без позиции — ошибка |
| `soldiers(cco, unit)` | CCO `ManList.At(i).Position` каждого бойца → `{status = 'ok', count, xz_dm = {x1, z1, ...}}` в целых дециметрах или `{status = 'unavailable', reason}` |

Только для своих отрядов: позиции бойцов врага выдали бы скрытых.

## contract

| Функция | Что делает |
|---|---|
| `width_bounds(kind)` | Лорд 3–8 м, строй 20–40 м. Это инженерные рамки ввода, не измеренные пределы игры |
| `valid_width(kind, width)` | Ширина в рамках |
| `can_shoot(unit)` | Лучники с положительным боезапасом |

Эти рамки раньше были продублированы в `deployment` и `policy_host`.
