# Ошибки и неизвестные значения

[← Назад](README.md) · [Документация](../README.md) · [English](../../en/architecture/error_handling.md)

В бою ошибка скрипта не должна пройти молча: иначе бой пойдёт дальше без
нашего скрипта, а запись или замер окажутся неверными. Поэтому в проекте три правила.

## 1. Чтение из движка никогда не падает: `known` / `unknown`

Адаптеры читают значения через `core.value.read`:

```lua
local value = require('apps.core.value')

local hp = value.read(function() return unit:unary_hitpoints() end, 'number')
-- {status = 'known', value = 0.8}
-- или {status = 'unknown', reason = 'read_error' | 'nil' | 'type_string' | 'nonfinite'}
```

- `0` и `false` — известные значения, не «пусто».
- `unknown` не означает, что механики нет. Это только «сейчас не прочитали».
- Координаты копируются в обычную таблицу `{x, y, z}` (`value.vector`):
  объект движка наружу не отдаётся.

## 2. Неверные данные — сразу ошибка

Сервисы и адаптеры проверяют входные данные через `assert` с понятным текстом:
`'declared/native battle role mismatch'`, `'battle requires one army per alliance'`,
`'Invalid terrain height'`. Бой с неверными данными не продолжается «как-нибудь»:
точка входа останавливается с ошибкой.

Для новых ошибок с устойчивым кодом есть `core.errors`:

```lua
local errors = require('apps.core.errors')
errors.raise('ROUTE_BLOCKED', 'no reachable cell near target')  -- "[ROUTE_BLOCKED] ..."
errors.code(message)  -- 'ROUTE_BLOCKED'
```

## 3. Колбэки движка защищены и останавливаются после первой ошибки

Всё, что вызывает движок (`register_phase_change_callback`, `repeat_callback`,
`real_callback`), оборачивается в `errors.guard`:

```lua
local function guarded(fn)
    return errors.guard(fn, fail, function() return state.finished end)
end
bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
```

`fail` один раз пишет событие `error` с трассировкой, снимает таймеры и
возвращает управление отрядами. Launcher видит событие `error` / `probe_error`
и завершает запуск со статусом `lua_error`.

## Где смотреть ошибку

1. `build/<цель>/runs/<время>/events.jsonl` — событие `error` с трассировкой.
2. `status.json` рядом — итог запуска (`lua_error`, `timeout`, `crash_report`).
3. Отчёт о падении игры копируется в ту же папку, если появился.
