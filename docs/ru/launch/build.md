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
| `move-probe` | `entries.move_probe` | `scenarios/move_probe.xml` | `tww3_bai_move_probe.pack` |
| `manual` | `entries.manual_record` | `scenarios/manual_hamlet.xml` | `tww3_bai_manual.pack` |
| `formation-probe` | `entries.formation_probe` | `scenarios/formation_probe.xml` (из `config/armies/`) | `tww3_bai_formation_probe.pack` |
| `archer-range` | `entries.archer_range` | `scenarios/archer_range.xml` (`tools/archer_range.py`) | `tww3_bai_archer_range.pack` |
| `enemy-layout` | `entries.enemy_layout` | `scenarios/enemy_layout.xml` (из `config/armies/defender_layouts.json`) | `tww3_bai_enemy_layout.pack` |
| `roster-capture` | `entries.roster_capture` | `scenarios/roster_capture.xml` (из `config/roster/capture.json`) | `tww3_bai_roster_capture.pack` |
| `map-capture` | `entries.map_capture` | `scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |

## Параметры

| Параметр | Цели | Значение |
|---|---|---|
| `--runs 1..10` | duel | Боёв подряд в одном процессе игры (автопереигровка) |
| `--speed 1/3/10/20` | duel, arena, ai-vs-ai, unit-readout, move-probe | Скорость боя |
| `--timeout 30..1800` | duel, arena, ai-vs-ai | Лимит модельного времени боя, секунды |
| `--tick-ms` | duel, arena | Период решений, мс (по умолчанию 1000) |
| `--step 1/2/3/5` | map-capture | Размер клетки сетки, м |
| `--features` | map-capture | После расстановки прочитать объекты и достижимость клеток |
| `--army` | formation-probe | Армия из `config/armies/<имя>.json` (по умолчанию `first_attack`) |
| `--layout`, `--enemy-mode` | enemy-layout | Состав врага; штатный ИИ как есть или с задачей «обороняй» (по умолчанию) |
| `--turn-test` | formation-probe | Добавить повороты лучников вправо и влево |
| `--handover`, `--enemy-ai native\|defend`, `--fast`, `--plain` | formation-probe | Тестовый бой игрока: наш ИИ расставляет, дальше игрок; враг — штатный ИИ; своя скорость или `--speed`; только файл боя без скрипта |
| `--facing-sweep` | formation-probe | Исследование: какие направления держит движок |
| `--range-mode fire_at_will\|attack` | archer-range | Когда лучники начинают стрелять при разной глубине блока |
| `--plan` | move-probe | План из `config/move-plans/<имя>.json` (по умолчанию `hamlet`) |
| `--scenario` | все | Другой файл из `scenarios/` вместо сценария цели |
| `--window MIN_X MAX_X MIN_Z MAX_Z` | map-capture | Снять только этот участок карты |
| `--deadline` | все, кроме map-capture | Лимит реального времени на бой, с (по умолчанию из длины сценария и скорости) |
| `--stall-minutes` | все, кроме map-capture | Завершить бой, если никто не получает урон столько минут игры (10). В move-probe не меньше длины плана + 2 мин |

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
