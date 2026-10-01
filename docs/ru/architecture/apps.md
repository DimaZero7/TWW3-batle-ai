# Список приложений

[← Назад](README.md) · [Документация](../README.md) · [Устройство проекта](overview.md) · [English](../../en/architecture/apps.md)

Код основы: приложения в игре, точки входа и инструменты вне игры. Большая часть
перенесена из исследовательского набора `lua-knowledge-kit`: поведение модулей
сохранено, 12 исходных тестов сенсоров перенесены без изменений проверок и проходят.
Старые исходники лежат в [research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).

**Сброс 30.09.2026** (решение автора проекта): наш алгоритмический ИИ — модули стратегии,
тактики, строя, дерево ИИ, политики прежних точек входа `duel` и `arena`, песочница политик, расстановка по плану,
симулятор боя и проигрыватель — удалён вместе с документами и тестами. Прежняя работа — в
истории Git (последний коммит до сброса — `242c3b1`).

## Приложения в игре (`src/apps`)

| Приложение | Файлы | Что делает | Откуда перенесено |
|---|---|---|---|
| [core](../apps/core.md) | `value`, `json`, `errors`, `clock` | Безопасное чтение значений, JSON для телеметрии, коды ошибок, защита колбэков | Функции, продублированные в старых скриптах kit (`state`, `range`, `visibility`, `post_battle_telemetry`, `capture`) |
| [battle](../apps/battle.md) | `adapter`, `services` | Стороны, армии, отряды, неподдерживаемые типы боя, роли атакующего/защитника, скорость, срок и застой боя | `runtime/battle_role.lua` |
| [map](../apps/map.md) | `adapter`, `services` | Рамка радара, сетка, высота, грунт, проходимость клетки, здания и сооружения CCO | `map/reader.lua`, `map/objects.lua` |
| [navigation](../apps/navigation.md) | `adapter`, `diagnostics_adapter`, `services` | Достижимость клеток для отряда, диагностика `can_reach_position`, конец захода (`arrived`/`stopped`/`stuck`/`timeout`) | `map/reachability.lua`, `runtime/reach_observation.lua` |
| [units](../apps/units.md) | `state_adapter`, `range_adapter`, `card_adapter`, `formation_adapter` | 68 полей состояния своего отряда, дальность стрельбы, карточка и профиль отряда, движение и бойцы | `units/state.lua`, `units/range.lua` |
| [intel](../apps/intel.md) | `adapter`, `services` | Видимость врага, память последней известной позиции | `visibility/reader.lua` |
| [observation](../apps/observation.md) | `adapter`, `services` | Сводка одной стороны: свои целиком, о врагах только разрешённое; самопроверка на утечки | новое |
| [orders](../apps/orders.md) | `adapter`, `planner_adapter`, `facing` | Проверенные в бою вызовы приказов, передача отрядов планировщику CA, 64 направления движка | [проверенные команды](../game/units/commands.md) |
| [telemetry](../apps/telemetry.md) | `adapter`, `sampler_adapter` | JSONL-события, файлы состояния, послебоевые снимки обеих сторон | `runtime/post_battle_telemetry.lua` |
| [bridge](../apps/bridge.md) | `adapter`, `exchange_adapter`, `services` | Мост к сети: файл состояния помощнику, его приказы нашим отрядам | новое |

## Точки входа (`src/entries`)

Каждая точка входа измеряет игру или записывает бой; своих решений в бою у неё нет (в `nn_arena`
с `--own-ai net` нашей стороной командует сеть, а не точка входа).

| Точка входа | Цель сборки | Сценарий | Что делает |
|---|---|---|---|
| [ai_vs_ai](../apps/entries.md#ai_vs_ai) | `ai-vs-ai` | `ai_vs_ai.xml` | Армия ИИ — под общим боевым ИИ игры, армия игрока — под планировщиком CA; решений о бое наш код не принимает: повторяет задачу планировщика, возвращает в бой собравшиеся и простаивающие отряды и записывает бой |
| [unit_readout](../apps/entries.md#unit_readout) | `unit-readout` | `unit_readout.xml` | Проверка в бою всех задокументированных показателей отрядов и видимости |
| [move_probe](../apps/entries.md#move_probe) | `move-probe` | `move_probe.xml` | Один отряд по плану заходов: строй и обычные приказы, позиции бойцов |
| [roster_capture](../apps/entries.md#roster_capture) | `roster-capture` | `roster_capture.xml` | Карточки и строй отрядов для ростера |
| [archer_range](../apps/entries.md#archer_range) | `archer-range` | `archer_range.xml` | Когда лучники начинают стрелять при разной глубине блока, урон стрел по дистанции |
| [lord_swarm](../apps/entries.md#lord_swarm) | `lord-swarm` | `lord_swarm.xml` | Стоящий лорд под атакой 1–4 отрядов пехоты (и другого лорда): его потери и бойцы вокруг |
| [enemy_layout](../apps/entries.md#enemy_layout) | `enemy-layout` | `enemy_layout.xml` | Как штатный ИИ расставляется и стоит при разных составах |
| [manual_record](../apps/entries.md#manual) | `manual` | `manual_hamlet.xml` | Ручной бой игрока с записью |
| [nn_arena](../apps/entries.md#nn_arena) | `nn-arena` | `nn_arena.xml` | Запись боёв ИИ игры на арене: каждый отряд обеих сторон раз в секунду — данные для обучения; с `--own-ai net` нашей стороной командует сеть |
| [map_capture](../apps/entries.md#map_capture) | `map-capture` | `map_capture.xml` | Сетка карты, объекты и достижимость в CSV/JSONL |

## Инструменты вне игры (`tools`)

| Инструмент | Что делает |
|---|---|
| `tools/build.py`, `tools/pack/` | Бандлер модулей, запись PFH5, manifest ([сборка](../launch/build.md)) |
| `tools/launcher/launch.ps1` | Установка, запуск на честной сложности, ожидание, очистка ([запуск](../launch/run.md)) |
| `tools/launcher/watch.ps1` | Бой, где нашей стороной командует сеть: сборка, помощник в контейнере, launcher ([смотреть бой сети](../launch/watch.md)) |
| `tools/telemetry/read_jsonl.ps1` | Чтение JSONL, пока игра пишет |
| `tools/config.py`, `tools/lua_runtime.py` | Пути и настройки машины; Lua 5.1 в Python для тестов |
| `tools/roster.py`, `tools/readouts.py` | Ростер отрядов из прогонов; каталог показателей и профили сбора |
| `tools/archer_range.py`, `tools/enemy_layout.py` | Сценарии и настройки зондов `archer-range` и `enemy-layout` |
| `tools/analysis/` | Карта высот, склоны, проходы, проходимость, препятствия; разборы прогонов `move-probe`, `unit-readout`, `archer-range`, `enemy-layout` |
| `tools/nn/` | Данные для обучения: правила из базы игры (`gamedb.py`), [паспорта отрядов](../training/units.md) (`dbtables.py`, `units.py`), сценарий арены и именованных арен (`scenario.py`), записи арены как массивы (`gamedata.py`), [замеры](../training/measurements.md) для симулятора (`measure.py`), запуск в контейнере с PyTorch (`dock.sh`), помощник, который командует нашей стороной в игре (`companion/`, [bridge](../apps/bridge.md)) — [подробнее](../training/README.md) |
| `tools/docs/` | Оглавления хабов документации (`python -m tools.docs.index_doc`) |
| `tools/architecture.py` | Уровень каждого приложения (сейчас все — основа) — для тестов архитектуры и документации |

Launcher и сборка выросли из старых `map-capture/launch.ps1`, `finish.ps1` и
`build.py` (пути исправлены, логика упаковки та же); они лежат в
[research/scripts/legacy-launch](../../../research/scripts/legacy-launch/).

## Что не перенесено в рабочий код

- Расстановка `deployment.lua` (v1) и `deployment_v2.lua` — только в архиве
  [research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).
- Разовые исследовательские скрипты (`map-analysis`, `unit-state`, `unit-range`,
  `unit-actions`) — в [research/scripts](../../../research/scripts/). Они
  ссылаются на старую структуру и нужны для воспроизведения архивных опытов.
- Турнир, агенты, очереди заданий — исключены ещё в kit.
