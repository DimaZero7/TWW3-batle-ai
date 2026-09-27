# Сборка pack

[Документация](../README.md) · [Запуск](run.md) · [English](../../en/launch/build.md)

Сборка не трогает игру: она только создаёт файлы в `build/<цель>/`.

```bash
.venv/Scripts/python -m tools.build duel --runs 3 --speed 20 --timeout 300
.venv/Scripts/python -m tools.build arena
.venv/Scripts/python -m tools.build map-capture --step 3 --features
```

## Цели

| Цель | Точка входа | Сценарий | Pack |
|---|---|---|---|
| `duel` | `entries.duel` | `scenarios/ranged_melee.xml` | `tww3_bai_duel.pack` |
| `arena` | `entries.arena` | `scenarios/triple_melee.xml` | `tww3_bai_arena.pack` |
| `ai-vs-ai` | `entries.ai_vs_ai` | `scenarios/ai_vs_ai.xml` | `tww3_bai_ai_vs_ai.pack` |
| `unit-readout` | `entries.unit_readout` | `scenarios/unit_readout.xml` | `tww3_bai_unit_readout.pack` |
| `map-capture` | `entries.map_capture` | `scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |

## Параметры

| Параметр | Цели | Значение |
|---|---|---|
| `--runs 1..10` | duel | Боёв подряд в одном процессе игры (автопереигровка) |
| `--speed 1/3/10/20` | duel, arena, ai-vs-ai, unit-readout | Скорость боя |
| `--timeout 30..1800` | duel, arena, ai-vs-ai | Лимит модельного времени боя, секунды |
| `--tick-ms` | duel, arena | Период решений, мс (по умолчанию 1000) |
| `--step 1/2/3/5` | map-capture | Размер клетки сетки, м |
| `--features` | map-capture | После расстановки прочитать объекты и достижимость клеток |

## Что лежит в `build/<цель>/`

- `<pack>.pack` — архив для папки `data` игры;
- `<скрипт>.lua` — собранный скрипт, удобно читать при отладке;
- `manifest.json` — `build` (короткий SHA-256 всех входов), конфигурация,
  список модулей, SHA-256 pack, путь сценария, `syntax_checked`.

Значение `build` попадает в каждую строку телеметрии, поэтому по журналу
всегда видно, какой именно код играл.

## Как устроен бандлер

`tools/pack/bundle.py` начинает с точки входа и рекурсивно находит все
`require('...')`. Каждый модуль оборачивается в функцию-фабрику, а вместо
глобального `require` ей передаётся свой загрузчик:

```lua
__modules["apps.core.value"] = function(require)
    -- исходник src/apps/core/value.lua
end
...
if bm then
    __require("entries.duel").main(bm, {runs = 3, ...}, {common = common, battle_vector = battle_vector})
end
```

Поэтому модули пишутся как обычный Lua и так же загружаются в тестах.
Циклический `require` — ошибка сборки.

## Содержимое pack

```text
script\battle\mod\<скрипт>.lua          наш скрипт; игра загружает его в каждом бою
script\battle\<папка>\scenario.lua      load_script_libraries()
script\battle\<папка>\<сценарий>.xml    XML боя
```

Формат — PFH5 без сжатия (`tools/pack/pfh5.py`, по описанию RPFM). Записи
отсортированы, одинаковые входы дают одинаковые байты. Обязательный мод
`true_sight.pack` из `config/mod-dependencies.json` всегда попадает в заголовок pack.

Поскольку скрипт из `script\battle\mod\` загружается в любом бою, **не
добавляйте тестовый pack в список модов кампании**. Launcher ставит его
только на время запуска.
