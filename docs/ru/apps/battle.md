# battle

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/battle.md)

Контекст боя: стороны, армии, отряды и роли. Единственное место, которое
обходит `bm:alliances()`; остальные приложения получают готовые списки.
Код: `src/apps/battle/`.

## adapter

| Функция | Что делает |
|---|---|
| `unsupported_reason(bm)` | Первый неподдерживаемый тип боя (`is_from_campaign`, `is_multiplayer`, `is_replay`, `is_quest_battle`, `is_tutorial`, `is_siege_battle`, `is_ambush_battle`) или `nil` |
| `read_sides(bm, expected_units?)` | `{side, alliance, army, units}` для двух сторон; проверяет одну армию на сторону и, если задано, число отрядов |
| `find_by_name(side, script_name)` | Отряд по `script_name` из XML |
| `speed_guard(bm, speed, on_restore?)` | Держит заданную скорость до фазы `Complete`: движок сбрасывает её, когда исход решён. Возвращает `stop()` |
| `read_roles(bm, record?)` | Атакующий/защитник по `alliance:is_attacker()`; никогда не выводится из индекса или места появления |

## services

| Функция | Что делает |
|---|---|
| `verify_roles(roles, expected)` | Сверяет роли из сценария с ролями движка |
| `opponent(side)` | `1 → 2`, `2 → 1` |

## Пример

```lua
local battle = require('apps.battle.adapter')

local reason = battle.unsupported_reason(bm)
if reason then return end
local sides = battle.read_sides(bm, 1)   -- ровно один отряд на сторону
local enemy = sides[2].units[1]
```

Имена отрядов из XML (`script_name`) переживают границы Lua-окружений,
поэтому точки входа узнают свой сценарий по ним, а не по глобальным
переменным.
