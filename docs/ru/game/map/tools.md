# Lua-модуль карты и инструменты запуска

[← Назад](README.md) · [Инструкция по карте](README.md) · [English](../../../en/game/map/tools.md)

## Компоненты

| Файл | Назначение |
|---|---|
| [src/apps/map/adapter.lua](../../../../src/apps/map/adapter.lua) | Рамка мини-карты (`read_frame`), чтение клеток порциями (`read_batch`), пакетное чтение объектов и сооружений CCO (`read_buildings`, `read_structure_contexts`); [проверенные рамки](objects.md) |
| [src/apps/map/services.lua](../../../../src/apps/map/services.lua) | Геометрия без движка: рамка, сетка и координаты клеток (`new_grid`, `world_to_radar`, `world_to_cell`, `cell_center`) |
| [src/apps/navigation/adapter.lua](../../../../src/apps/navigation/adapter.lua) | Доступность точек выбранным отрядам после расстановки (`read_cells`); [ограничения](passages.md) |
| [slopes.py](../../../../tools/analysis/slopes.py) | Уклоны и соседние перепады из сохранённых высот |
| [passages.py](../../../../tools/analysis/passages.py) | Поиск соединений мелководья по сохранённой доступности |
| [src/entries/map_capture.lua](../../../../src/entries/map_capture.lua) | Точка входа: статический опыт, пауза, отложенные вызовы, CSV и JSONL; режим `--features` — начало боя под контролем скрипта, объекты/CCO и доступность всех точек |
| [heightmap.py](../../../../tools/analysis/heightmap.py) | Таблица высот и PNG из CSV без игры; [инструкция](heights.md) |
| [build.py](../../../../tools/build.py) | Отдельный pack из модулей карты, точки входа и фиксированного сценария |
| [launch.ps1](../../../../tools/launcher/launch.ps1) | Проверка/установка pack, запуск процесса, ожидание результата, сохранение, закрытие своего процесса и удаление своих служебных файлов из игры |
| [map_capture.xml](../../../../scenarios/map_capture.xml) | Штатный полевой Кислев; два минимальных противостоящих отряда нужны для загрузки боя |

В модулях карты и доступности нет команд отрядам, таймеров, файлов и
глобального запуска. Это локальные Lua-модули, не сетевой сервис. Жизненным
циклом управляет точка входа `entries.map_capture`; в сборку другой цели модули
попадают, только если её точка входа их подключает.

## Интерфейс модуля

Сборщик (`tools/build.py`) собирает модули из `src/` в один скрипт через
функции-обёртки. Непроверенный игровой путь `require` не предполагается.

```lua
local map = require('apps.map.adapter')
local map_services = require('apps.map.services')
local navigation = require('apps.navigation.adapter')

local frame = map.read_frame(common)
local grid = map_services.new_grid(frame, 5)
local cells, next_index, done = map.read_batch(bm, battle_vector, grid, 0, 4096)
local u, v = map_services.world_to_radar(frame, 100, 0)
local ix, iz = map_services.world_to_cell(grid, 100, 0)
local x, z = map_services.cell_center(grid, ix, iz)
local rows = navigation.read_cells(bm, battle_vector, units, cells)  -- после расстановки
```

- `frame`: коэффициенты, проверенные углы, `min_x/max_x/min_z/max_z`, размеры,
  центр, `kind="radar_frame"`, `movement_boundary_verified=false`.
- `grid`: начало, `step`, столбцы/строки/число клеток, `radar_max_x/z` и отдельные `query_max_x/z`.
- `cells`: `ix, iz, x, z, height, clear, ground, inside_radar`. Последний флаг
  проверяет центр, а не полное попадание квадрата внутрь рамки.
- `next_index`: линейный индекс следующей порции от нуля. Повторять до `done`.
  Промежутки между порциями организует вызывающий код; большой цикл не запускать блокирующе в бою.

Ошибочные типы, нечисловые/бесконечные ответы, непроверенная ориентация радара
или несовпадение углов вызывают ошибку. Запускатель сохраняет `probe_error`,
а не превращает ошибку в зелёную клетку или успешный результат. `math.huge`
не используется: в проверенном игровом Lua его не было.

## Запуск из корня репозитория

Нужны установленная WH3, Python проекта (`.venv` с зависимостями из
`requirements-dev.txt`), PowerShell и доступ на запись в папку игры. Путь к игре —
в `config/default.json` или `config/local.json`.

```bash
.venv/Scripts/python -m pytest tests/apps/map
```

```bash
.venv/Scripts/python -m tools.build map-capture --step 5
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target map-capture
```

Шаги сетки: `1`, `2`, `3`, `5`; `--window MIN_X MAX_X MIN_Z MAX_Z` снимает только
участок карты. Игра должна быть закрыта. Сбор статический, на паузе; управление
мышью не требуется.

Результат: `build/map-capture/runs/<ГГГГММДД-ччммсс>/`: manifest сборки, сведения
о процессе, события, XML боя, CSV сетки и статус с итогом очистки
([запуск боя](../../launch/run.md)).

**Launcher закрывает свой процесс после сохранения.** При ошибке или тайм-ауте
также завершается только свой проверенный процесс. Запущенная игра или файлы
прошлого запуска в папке игры блокируют новый запуск: их нужно проверить и убрать
вручную. Оставлять игру открытой (`-KeepGameOpen`) только для явно нужной
пользователю ручной проверки.

## Степень проверки

Семь тестов Lua 5.1 проверяют сохранённые точки радара двух карт, ошибочные
ответы, углы, четыре размера сетки, крайние индексы и чтение порциями. Они
не симулируют землю и навигацию WH3.

Выделенный модуль и сборщик затем проверены в игре на Кислеве, шаг 5 м:
**42 025 из 42 025 клеток совпали с предыдущим замером** по индексам, высоте
до шести знаков CSV, ответам ограничений и типу земли. Центры также совпали.
`Доказательства` (локальный архив: `research/evidence/map/module-check-20260925/comparison.json`).
Исходный опыт запускал все четыре шага; отдельная живая проверка выделенного
модуля — только 5 м. Не расширять этот вывод на другие карты или активный бой.

В первом опыте модуля `finish.ps1` вызван вручную. Автоматический вызов очистки
и закрытие при ошибках добавлены после предпочтения пользователя. В истории
проверки сохраняем это различие, не называем все ветки проверенными в игре.

## Полный сбор полевых данных

Флаг `--features` добавляет обычный список объектов, [сооружения и мосты CCO](bridges.md), доступность точек для первого отряда каждой из двух армий. Использовать сохранённые диагностические сценарии с двумя отрядами. Запускатель берёт их под управление, запрещает стрельбу, останавливает, завершает расстановку и ставит бой на паузу. Без флага остаётся прежний сбор поверхности во время расстановки.

Повтор проверенного сбора участка с мостом (XML того замера лежит в локальном
архиве):

```bash
.venv/Scripts/python -m tools.build map-capture --step 3 --features --scenario ../research/evidence/map/field-20260925/bridge-navigation/map_probe.xml
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target map-capture
```

`--scenario` берёт путь от папки `scenarios/` и подставляет XML, а не выбирает
карту по русскому названию. Исходный сценарий, собранный Lua и хеши сохраняются в папке запуска. Это отдельный диагностический бой.

В CSV добавлены `reach_side_1`, `reach_side_2`: 1 — да, 0 — нет, −1 — вне рамки, не запрашивалось. В JSONL — `building`, `structure_context`, `navigation_unit`, `feature_conditions`, количества и время. Записи отрядов показывают, к кому относятся столбцы. Файлы читать как **UTF-8**, в том числе в Windows: имена CCO могут быть русскими.

Для поиска бродов указать новые столбцы:

```bash
.venv/Scripts/python -m tools.analysis.passages --csv build/map-capture/runs/<время>/tww3_bai_map_capture_grid.csv --step 3 --reach-columns reach_side_1 reach_side_2 --output build/map-research/passages
```

Заменить папку запуска на настоящую. Это поиск кандидатов; наличие моста не гарантирует наличие брода.

Повтор в игре совпал с исследовательским запуском по **всем 116 964 строкам**, включая поверхность и доступность, **2 696 объектам** и **одной записи CCO**. Запросы доступности с записью заняли **2,430 с** по часам Lua, без загрузки. Весь запуск — примерно **36,7 с**, затем игра автоматически закрылась. `Проверка` (локальный архив: `research/evidence/map/field-20260925/feature-runner/validation.json`) · `Состояние запуска` (локальный архив: `research/evidence/map/field-20260925/feature-runner/status.json`).
