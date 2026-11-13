# Список приложений

[← Назад](README.md) · [Документация](../README.md) · [Устройство моста](overview.md)

Код моста: приложения в игре, точки входа и инструменты вне игры.

## Приложения в игре (`bridge/lua/apps`)

| Приложение | Файлы | Что делает |
|---|---|---|
| [core](../apps/core.md) | `value`, `json`, `errors`, `clock` | Безопасное чтение значений, JSON для телеметрии, коды ошибок, защита колбэков |
| [battle](../apps/battle.md) | `adapter`, `services`, `winds`, `live_label`, `enemy_cards`, `box_pick`, `screen_proj`, `label_chat`, `label_overlay` | Стороны, армии, отряды, неподдерживаемые типы боя, роли атакующего/защитника, скорость, срок и застой боя; ветра магии; разметка вживую |
| [map](../apps/map.md) | `adapter`, `services` | Рамка радара, сетка, высота, грунт, проходимость клетки, здания и сооружения CCO |
| [navigation](../apps/navigation.md) | `adapter`, `services` | Достижимость клеток для отряда, конец захода (`arrived`/`stopped`/`stuck`/`timeout`) |
| [units](../apps/units.md) | `card_adapter`, `formation_adapter` | Карточка и профиль отряда, движение и бойцы |
| [orders](../apps/orders.md) | `adapter`, `facing`, `planner_adapter` | Проверенные в бою вызовы приказов, 64 направления движка, отряды под планировщиком ИИ игры |
| [telemetry](../apps/telemetry.md) | `adapter`, `observer_adapter` | JSONL-события, файлы состояния, запись боя человека |
| [bridge](../apps/bridge.md) | `adapter`, `exchange_adapter`, `services`, `deploy_adapter` | Мост к помощнику: файл состояния, его приказы нашим отрядам, расстановка стратегом |
| [mem](../apps/mem.md) | `projectiles`, `entities`, `hits`, `morale`, `order` | Только пробы (`--mem`): снаряды, бойцы, попадания, мораль и строй из памяти игры через memreader Plus |

## Точки входа (`bridge/lua/entries`)

Каждая точка входа измеряет игру, записывает бой или передаёт его помощнику; своих решений в бою у неё нет (в
`nn_arena` с `--own-ai net` нашей стороной командует цепочка в помощнике, а не точка входа).

| Точка входа | Цель сборки | Сценарий | Что делает |
|---|---|---|---|
| [nn_arena](../apps/entries.md#nn_arena) | `nn-arena`, `human` | файл боя арены (`sim/arena.py`) | Бой цепочки против ИИ игры (мост к помощнику), бой ИИ против ИИ (`own_ai 'ai'`) или бой человека; каждый отряд обеих сторон раз в секунду в записи |
| [charge_probe](../apps/entries.md#charge_probe) | `charge-probe` | файл боя плана | Одна механика рукопашной в полосах, игра и двойник симулятора одинаково |
| [missile_probe](../apps/entries.md#missile_probe) | `missile-probe` | файл боя плана | Одна механика стрельбы в полосах |
| [morale_probe](../apps/entries.md#morale_probe) | `morale-probe` | файл боя плана | Одно слагаемое морали в полосах |
| [map_capture](../apps/entries.md#map_capture) | `map-capture` | `map_capture.xml` | Сетка карты, объекты и достижимость в CSV/JSONL |
| [move_probe](../apps/entries.md#move_probe) | `move-probe` | `move_probe.xml` | Один отряд по плану заходов: строй и обычные приказы, позиции бойцов |

## Инструменты вне игры (`bridge/`)

| Инструмент | Что делает |
|---|---|
| `bridge/build.py`, `bridge/pack/` | Бандлер модулей, запись PFH5, правила игры в пробах (`kvtable.py`), manifest ([сборка](../launch/build.md)) |
| `bridge/mem_sites.py`, `bridge/config/memreader.json` | Закрепление memreader Plus и места памяти для `--mem` ([memreader](../apps/memreader.md)) |
| `bridge/launcher/launch.ps1` | Установка, запуск на честной сложности, ожидание, очистка ([запуск](../launch/run.md)) |
| `bridge/launcher/watch.ps1` | Бой, где нашей стороной командует помощник: сборка, помощник в контейнере, launcher ([смотреть бой сети](../launch/watch.md)) |
| `bridge/launcher/{preferences,installed,event_log,cleanup,read_jsonl}.ps1` | Настройки игры на запуск (сложность, графика), список установленного, журнал событий, уборка, чтение JSONL, пока игра пишет |
| `bridge/companion/` | Помощник: стратег (`deploy.py`), цепочка (`chain.py`), входы уровней (`unit_input`, `tactic_input`, `operative_input`, `strategist_input`), файлы обмена (`exchange.py`), цикл (`loop.py`) ([bridge](../apps/bridge.md)) |
| `bridge/config.py`, `bridge/config/` | Пути и настройки машины (`default.json` в git, `local.json` — своя машина), обязательный мод, планы `move-probe` |
