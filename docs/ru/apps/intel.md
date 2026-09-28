# intel

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/intel.md)

Что сторона знает о врагах: видим ли он сейчас и где его видели последний
раз. Код: `src/apps/intel/`.

## Правило

Надёжно только **текущее видимое** состояние врага. Для скрывшегося врага
допустима последняя известная позиция со временем наблюдения. Скрытое
текущее состояние не читается: у невидимого отряда даже не вызывается
`position()`.

## adapter

| Функция | Что делает |
|---|---|
| `query(unit, alliance)` | `true` / `false`, либо `nil`, если движок не ответил |
| `observe(unit, alliance, public_id, now_ms, memory)` | Видимость + позиция видимого + память |

## services (без движка)

| Функция | Что делает |
|---|---|
| `new_memory()` | Память одного наблюдателя на один бой |
| `record(memory, id, visible, position, now_ms)` | Обновляет память, возвращает наблюдение |

```lua
local intel = require('apps.intel.adapter')
local memory = require('apps.intel.services').new_memory()

local seen = intel.observe(enemy, my_alliance, 'enemy-1', now_ms, memory)
-- seen.visibility = 'visible' | 'not_visible' | 'unknown'
-- seen.current = {x, y, z, seen_ms}      только если видим
-- seen.last_seen = {x, y, z, seen_ms}    последняя известная позиция
```

`public_id` берётся из разрешённого сопоставления состава, а не из
объекта движка врага. Память создаётся заново для каждого боя и каждой
стороны.
