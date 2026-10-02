# Сборка pack

[← Назад](README.md) · [Документация](../README.md) · [Запуск](run.md) · [English](../../en/launch/build.md)

Сборка не трогает игру: она только создаёт файлы в `build/<цель>/`.

```bash
.venv/Scripts/python -m tools.build nn-arena --own-ai attack --timeout 900
.venv/Scripts/python -m tools.build ai-vs-ai --speed 3
.venv/Scripts/python -m tools.build map-capture --step 3 --features
```

## Цели

| Цель | Точка входа | Сценарий | Pack |
|---|---|---|---|
| `ai-vs-ai` | `entries.ai_vs_ai` | `scenarios/ai_vs_ai.xml` | `tww3_bai_ai_vs_ai.pack` |
| `unit-readout` | `entries.unit_readout` | `scenarios/unit_readout.xml` | `tww3_bai_unit_readout.pack` |
| `move-probe` | `entries.move_probe` | `scenarios/move_probe.xml` | `tww3_bai_move_probe.pack` |
| `manual` | `entries.manual_record` | `scenarios/manual_hamlet.xml` | `tww3_bai_manual.pack` |
| `lord-swarm` | `entries.lord_swarm` | `scenarios/lord_swarm.xml` (`tools/nn/lord_swarm.py`) | `tww3_bai_lord_swarm.pack` |
| `archer-range` | `entries.archer_range` | `scenarios/archer_range.xml` (`tools/archer_range.py`) | `tww3_bai_archer_range.pack` |
| `enemy-layout` | `entries.enemy_layout` | `scenarios/enemy_layout.xml` (из `config/armies/defender_layouts.json`) | `tww3_bai_enemy_layout.pack` |
| `roster-capture` | `entries.roster_capture` | `scenarios/roster_capture.xml` (из `config/roster/capture.json`) | `tww3_bai_roster_capture.pack` |
| `nn-arena` | `entries.nn_arena` | `scenarios/nn_arena.xml` (`tools/nn/scenario.py`) | `tww3_bai_nn_arena.pack` |
| `map-capture` | `entries.map_capture` | `scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |

Что делает каждая точка входа — [точки входа](../apps/entries.md).

## Параметры

| Параметр | Цели | Значение |
|---|---|---|
| `--speed 1/3/10/20` | все, кроме map-capture и manual | Скорость боя (по умолчанию 20) |
| `--timeout 30..3600` | ai-vs-ai, nn-arena | Лимит модельного времени боя, с (600); из него считается срок боя. Бои арены 30.09.2026 записаны с 900, [проверка сети](gate.md) — с 3600, как в симуляторе |
| `--tick-ms` | все, кроме map-capture | Период тика, мс (1000) |
| `--step 1/2/3/5` | map-capture | Размер клетки сетки, м |
| `--features` | map-capture | После расстановки прочитать объекты и достижимость клеток |
| `--window MIN_X MAX_X MIN_Z MAX_Z` | map-capture | Снять только этот участок карты |
| `--own-ai attack\|defend` | nn-arena | Наша сторона под планировщиком CA атакует (ИИ игры обороняется) или обороняется (ИИ игры атакует) |
| `--own-ai net` | nn-arena | Нашей стороной командует сеть в помощнике ([смотреть бой сети](watch.md)) |
| `--own-role attack\|defend` | nn-arena `net` | Сеть атакует (ИИ игры обороняется и побеждает, когда время вышло) или обороняется (по умолчанию: ИИ игры атакует) |
| `--army-seed N` | nn-arena | Бой из [генератора армий](../training/armies.md): `generate.battle(N)`, у стороны лорд и 0–19 отрядов. Без `--own-ai` нашей стороной командует сеть. Файл боя — `build/nn-arena/random_<N>.xml`, `scenarios/` не трогается. В манифест — `army` (зерно, train/eval, бюджет, шаблоны, бойцы) |
| `--army-swap` | nn-arena `--army-seed` | Армии зерна меняются местами: нашей стороне — армия противника из генератора, ИИ игры — наша (`random_<N>_swap.xml`, в манифесте `army.swap` — true): второй бой пары [проверки в игре](gate.md#какие-бои) |
| `--decide-ms 250..5000` | nn-arena `net` | Время боя между двумя решениями сети, мс (1000) |
| `--layout`, `--enemy-mode native\|defend` | enemy-layout | Состав врага; штатный ИИ как есть или с задачей «обороняй» (по умолчанию) |
| `--range-mode fire_at_will\|attack\|damage` | archer-range | Когда лучники начинают стрелять при разной глубине блока; `damage` — урон по бесстрашной цели на 70–120 м |
| `--damage-rotate N` | archer-range `damage` | Сдвинуть дистанции на N полос (та же дистанция на другом грунте) |
| `--swarm infantry\|lords\|all`, `--repeats N` | lord-swarm | Пехота вокруг каждого лорда, другой лорд (с отрядами) на нём или всё вместе; каждая раскладка N раз (2). Замеры каждые 0,2 с (`--tick-ms`) |
| `--plan` | move-probe | План из `config/move-plans/<имя>.json` (по умолчанию `hamlet`) |
| `--capture` | roster-capture | Другой список отрядов вместо `config/roster/capture.json` |
| `--scenario` | все | Другой файл из `scenarios/` вместо сценария цели |
| `--deadline` | все, кроме map-capture | Лимит реального времени на бой, с (по умолчанию из длины сценария и скорости) |
| `--stall-minutes` | все, кроме map-capture | Завершить бой, если никто не получает урон столько минут игры (10). В move-probe, roster-capture, archer-range и enemy-layout — не меньше длины сценария + 2 мин |

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
    __require("entries.nn_arena").main(bm, {own_ai = "attack", ...}, {common = common, battle_vector = battle_vector})
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
