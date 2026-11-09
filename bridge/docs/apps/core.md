# core

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md)

Общие утилиты без движка и без зависимостей от других приложений.
Код: `bridge/lua/apps/core/`.

| Модуль | Функции |
|---|---|
| `value` | `finite(n)`, `integer(n)`, `known(v)`, `unknown(reason)`, `read(fn, kind, invalid_type_reason?)`, `vector(fn)`, `distance_xz(a, b)` |
| `json` | `encode(data)` — ключи отсортированы, `NaN`/`inf` → `null`; `string(text)` |
| `errors` | `raise(code, message)`, `check(cond, code, message)`, `code(message)`, `guard(fn, on_error, is_stopped?)` |
| `clock` | `wall_seconds()`, `elapsed_wall_seconds(start)`, `iso_utc()`, `batch_stamp()` |

## Пример

```lua
local value = require('apps.core.value')
local json = require('apps.core.json')

local ammo = value.read(function() return unit:ammo_left() end, 'number')
if ammo.status == 'known' and ammo.value == 0 then ... end

print(json.encode({event = 'snapshot', hp = 0.5}))  -- {"event":"snapshot","hp":0.5}
```

## Почему так

- `read` — одно чтение для всех адаптеров. Причина ошибки по умолчанию — `type_<тип>`; свою можно передать
  третьим аргументом (`invalid_type_reason`).
- `wall_seconds` считает секунды от начала суток: секунды эпохи теряют
  точность в числах Lua внутри WH3.
- Подробнее о правилах: [ошибки и неизвестные значения](../architecture/error_handling.md).
