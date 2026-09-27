# Список приложений

[Документация](../README.md) · [Устройство проекта](overview.md) · [English](../../en/architecture/apps.md)

Код перенесён из исследовательского набора `lua-knowledge-kit`. Поведение
модулей сохранено: 12 исходных тестов сенсоров перенесены без изменений
проверок и проходят. Старые исходники лежат в
[research/scripts/legacy-lua](../../../research/scripts/legacy-lua/).

## Приложения в игре (`src/apps`)

| Приложение | Файлы | Что делает | Откуда перенесено |
|---|---|---|---|
| [core](../apps/core.md) | `value`, `json`, `errors`, `clock` | Безопасное чтение значений, JSON для телеметрии, коды ошибок, защита колбэков | Функции, продублированные в `state`, `range`, `visibility`, `post_battle_telemetry`, `harness`, `arena`, `capture` |
| [battle](../apps/battle.md) | `adapter`, `services` | Стороны, армии, отряды, неподдерживаемые типы боя, роли атакующего/защитника | `runtime/battle_role.lua`, начало `harness.lua` |
| [map](../apps/map.md) | `adapter`, `services` | Рамка радара, сетка, высота, грунт, проходимость клетки, здания и сооружения CCO | `map/reader.lua`, `map/objects.lua` |
| [navigation](../apps/navigation.md) | `adapter`, `diagnostics_adapter`, `services` | Достижимость клеток для отряда, диагностика `can_reach_position` | `map/reachability.lua`, `runtime/reach_observation.lua` |
| [assessment](../apps/assessment.md) | `services` | Оценка сил по составу армий: признаки для стратегий | новое |
| [strategy](../apps/strategy.md) | `services`, `contract` | Список стратегий, условия, роли, выбор с журналом | новое |
| [formation](../apps/formation.md) | `services` | Расстановки строя по ролям: `line_and_blocks` по данным ростера | новое |
| [plan](../apps/plan.md) | `services` | Цепочка начала боя: оценка → стратегия → строй | новое |
| [vision](../apps/vision.md) | `services` | Группы видимых врагов, основная армия — самая сильная группа | новое |
| [battlefield](../apps/battlefield.md) | `services` | Поле боя между основными группами: ось, передние линии, запас 40 м | новое |
| [units](../apps/units.md) | `state_adapter`, `range_adapter`, `card_adapter`, `formation_adapter`, `contract` | 68 полей состояния своего отряда, дальность стрельбы, карточка и профиль отряда, движение и бойцы, границы ширины строя | `units/state.lua`, `units/range.lua` |
| [intel](../apps/intel.md) | `adapter`, `services` | Видимость врага, память последней известной позиции | `visibility/reader.lua` |
| [orders](../apps/orders.md) | `contract`, `adapter`, `planner_adapter` | Форма и проверка команд, проверенные в бою вызовы приказов | `validate` из `policy_host.lua`, вызовы из `harness`/`arena` и `commands.md` |
| [deployment](../apps/deployment.md) | `contract`, `services`, `adapter` | Контракт расстановки v2, транзакция «собрать → проверить → применить → сверить» | `runtime/deployment_v2.lua` |
| [ai](../apps/ai.md) | `contract`, `services`, `policies/` | Профили армий, контракт решения, политики дуэли | `profile` из `policy_host.lua`, `policy.lua`, `delayed_melee.lua` |
| [observation](../apps/observation.md) | `adapter`, `services` | Сводка одной стороны: свои целиком, о врагах только разрешённое; самопроверка на утечки | новое |
| [sandbox](../apps/sandbox.md) | `services` | Загрузка чужих политик в урезанное окружение с лимитом инструкций | `policy_host.lua` |
| [telemetry](../apps/telemetry.md) | `adapter`, `sampler_adapter` | JSONL-события, файлы состояния, послебоевые снимки обеих сторон | `emit` из harness, `runtime/post_battle_telemetry.lua` |

## Точки входа (`src/entries`)

| Точка входа | Цель сборки | Сценарий | Что делает |
|---|---|---|---|
| [duel](../apps/entries.md#duel) | `duel` | `ranged_melee.xml` | Один отряд на сторону, принудительная рукопашная, серия боёв с автопереигровкой |
| [arena](../apps/entries.md#arena) | `arena` | `triple_melee.xml` | Три пары одновременно на одной карте |
| [ai_vs_ai](../apps/entries.md#ai_vs_ai) | `ai-vs-ai` | `ai_vs_ai.xml` | Обе армии под штатным ИИ игры, наш код только наблюдает |
| [unit_readout](../apps/entries.md#unit_readout) | `unit-readout` | `unit_readout.xml` | Проверка в бою всех задокументированных показателей отрядов |
| [move_probe](../apps/entries.md#move_probe) | `move-probe` | `move_probe.xml` | Один отряд по плану заходов: строй и обычные приказы, позиции бойцов |
| [roster_capture](../apps/entries.md#roster_capture) | `roster-capture` | `roster_capture.xml` | Карточки и строй отрядов для ростера |
| [formation_probe](../apps/entries.md#formation_probe) | `formation-probe` | `formation_probe.xml` | Строй армии в бою по плану `apps.formation`, повороты лучников |
| [enemy_layout](../apps/entries.md#enemy_layout) | `enemy-layout` | `enemy_layout.xml` | Как стоит штатный ИИ в обороне при разных составах |
| [manual_record](../apps/entries.md#manual) | `manual` | `manual_hamlet.xml` | Ручной бой игрока с записью |
| [map_capture](../apps/entries.md#map_capture) | `map-capture` | `map_capture.xml` | Сетка карты, объекты и достижимость в CSV/JSONL |

## Инструменты вне игры (`tools`)

| Инструмент | Что делает | Откуда |
|---|---|---|
| `tools/build.py`, `tools/pack/` | Бандлер модулей, запись PFH5, manifest | `game-launch/build.py`, `map-capture/build.py` (пути исправлены, логика упаковки та же) |
| `tools/launcher/launch.ps1` | Установка, запуск, ожидание, очистка | Логика `map-capture/launch.ps1` + `finish.ps1`; старые скрипты начинались с `throw` |
| `tools/telemetry/read_jsonl.ps1` | Чтение JSONL, пока игра пишет | `telemetry/telemetry.ps1` |
| `tools/analysis/` | Карта высот, склоны, проходы | `map-capture/heightmap.py`, `slopes.py`, `passages.py` |

## Что не перенесено в рабочий код

- `deployment.lua` (v1) — только в архиве; актуален `deployment-placement-v2`.
- Разовые исследовательские скрипты (`map-analysis`, `unit-state`, `unit-range`,
  `unit-actions`) — в [research/scripts](../../../research/scripts/). Они
  ссылаются на старую структуру и нужны для воспроизведения архивных опытов.
- Турнир, агенты, очереди заданий — исключены ещё в kit.
