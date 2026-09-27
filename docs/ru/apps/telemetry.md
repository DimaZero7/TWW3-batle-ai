# telemetry

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/telemetry.md)

Запись того, что произошло в бою. Код: `src/apps/telemetry/`.

## adapter — файлы

| Функция | Что делает |
|---|---|
| `sink(path, stamp?)` | Возвращает `emit(event, fields)`: строка JSONL с `event`, `wall_iso` и полями из `stamp(row)` |
| `read_file`, `write_file`, `append` | Небольшие файлы состояния в папке игры |
| `next_sequence(path)` | Счётчик на диске: уникальные номера серий без случайных чисел игры |

Файл открывается на каждую строку заново: при падении игры в журнале не
теряются строки из буфера.

```lua
local telemetry = require('apps.telemetry.adapter')
local emit = telemetry.sink('tww3_bai_events.jsonl', function(row)
    row.build, row.model_ms = config.build, bm:time_elapsed_ms()
end)
emit('decision', {side = 1, action = 'attack', reason = 'initial_charge'})
```

## sampler_adapter — послебоевые снимки

Снимок обеих сторон для анализа после боя: позиции, видимость для каждой
стороны, здоровье и мораль из CCO, боезапас и его расход, цель, заданная
позиция, расстояния между атакующими и получившими урон.

> **Никогда не вход для политики.** Снимок читает и скрытых врагов.

| Функция | Что делает |
|---|---|
| `new(sides, cco)` | Реестр `unit → roster id`, запасной поиск по `unique_ui_id` |
| `sample(state, time_ms)` | Кадр телеметрии |
| `entity_hit(state, event, time_ms)`, `drain_hits(state)` | Попадания из игрового события |

## Обработка вне игры

- [tools/telemetry/read_jsonl.ps1](../../../tools/telemetry/read_jsonl.ps1) —
  чтение журнала по мере записи (использует launcher).
- Результаты запуска — [запуск боя](../launch/run.md#результаты).
