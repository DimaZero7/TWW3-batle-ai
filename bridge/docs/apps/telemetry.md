# telemetry

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md)

Запись того, что произошло в бою. Код: `bridge/lua/apps/telemetry/`.

## adapter — файлы

| Функция | Что делает |
|---|---|
| `sink(path, stamp?)` | Возвращает `emit(event, fields)`: строка JSONL с `event`, `wall_iso` и полями из `stamp(row)` |
| `buffered_sink(path, stamp?)` | Те же строки, но копятся в памяти; `flush()` пишет их одним открытием файла. Для частых записей: сбрасывать раз в тик и в конце боя |
| `read_file`, `write_file`, `append` | Небольшие файлы состояния в папке игры |
| `next_sequence(path)` | Счётчик на диске: уникальные номера серий без случайных чисел игры |

`sink` открывает файл на каждую строку заново: при падении игры в журнале не
теряются строки из буфера.

```lua
local telemetry = require('apps.telemetry.adapter')
local emit = telemetry.sink('tww3_bai_events.jsonl', function(row)
    row.build, row.model_ms = config.build, bm:time_elapsed_ms()
end)
emit('result', {status = 'completed', winner = 1})
```

<a id="observer_adapter"></a>

## observer_adapter — всё о бое человека

Дополнительная запись боя, который играет человек ([nn_arena](entries.md#nn_arena), режим `human`): всё, что игра
даёт о каждом отряде обеих сторон, сверх строки `nn_sample`. Полная сводка, не вход для ИИ.

| Функция | Что делает |
|---|---|
| `new(opts)` | `units` (имя, отряд, сторона), `alliances`, `cco`, `emit`, `now_ms`, `soldiers_every` |
| `decorate(row)` | В строку отряда: `ob`, `ow` — направление и ширина строя заданного приказа; `idle` — у отряда нет приказа; `v` — виден другой стороне; `uma`, `td`, `dir` — под обстрелом, получает урон, нанесённый урон за последнее время (CCO) |
| `changes()` | При изменении: `nn_effects` (активные эффекты отряда) и `nn_ability_ready` (готовность способности; использование — true → false), обе стороны |
| `soldiers()` | Каждый `soldiers_every`-й вызов: `nn_soldiers` — место каждого бойца, дм |

## Обработка вне игры

- [bridge/launcher/read_jsonl.ps1](../../launcher/read_jsonl.ps1) —
  чтение журнала по мере записи (использует launcher).
- Результаты запуска — [запуск боя](../launch/run.md#результаты).
