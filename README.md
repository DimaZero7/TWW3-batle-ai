# TWW3 Battle AI

![Moorlands Route: measured heights, ground and objects](data/maps/moorlands-route/overview.png)

**Русский** · [English](#english)

Основа для боевого ИИ **Total War: WARHAMMER III**: измеренная механика боя,
инструменты для работы с игрой из Lua-скриптов и нейросеть для боя, которая учится с нуля
в нашем симуляторе и проверяется в игре против ИИ игры.

Прежний алгоритмический ИИ проекта (стратегия, тактика, строй, дерево ИИ) удалён по
решению автора проекта; он в истории Git (последний коммит с ним — `242c3b1`).

## Что есть в проекте

- **Механика игры** — что проверено в WH3: карта, отряды, мораль, рукопашная,
  стрельба, скорость хода, сложность, как воюет штатный ИИ — [знания об игре](docs/ru/game/README.md).
- **Инфраструктура** — сборка pack, launcher с честной сложностью, чтение карты,
  карточки, состояние, видимость и дальность отрядов, формы строя, проверенные
  приказы, контекст боя, телеметрия, правила наблюдения и зонды, которые измеряют
  игру или записывают бои — [модули кода](docs/ru/apps/README.md).
- **Данные для обучения** — правила из базы данных игры, запись боёв ИИ игры на
  арене и загрузчик этих записей — [данные для обучения нейросети](docs/ru/training/README.md).
- **Нейросеть для боя** — симулятор боя, вход и устройство сети, обучение PPO, мост к игре и
  проверка в игре — [обучение](docs/ru/training/training.md), [симулятор](docs/ru/training/simulator.md),
  [проверка в игре](docs/ru/launch/gate.md).

Код разбит на приложения: логика — в сервисах, обращения к движку — только в
адаптерах. Тесты проверяют Lua 5.1 без запуска игры.

## Быстрый старт

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m tools.build nn-arena --own-ai attack --timeout 900
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target nn-arena
```

Для боя нужны установленная игра, путь к ней в `config/local.json` и подписка на
мод True Sight — [установка и настройка](docs/ru/environment/setup.md).

## Документация

- **Данные для обучения нейросети** — [арена, записи боёв, правила игры](docs/ru/training/README.md)
- **Механика игры** — [что проверено в WH3](docs/ru/game/README.md) · [каталог показателей](docs/ru/game/readouts.md) · [атлас](docs/ru/game/atlas.md)
- **Код** — [модули `src/apps`](docs/ru/apps/README.md) · [точки входа](docs/ru/apps/entries.md)
- **Архитектура** — [устройство проекта и слои](docs/ru/architecture/overview.md) · [список приложений](docs/ru/architecture/apps.md) · [ошибки и неизвестные значения](docs/ru/architecture/error_handling.md) · [как вести документацию](docs/ru/architecture/documentation.md)
- **Окружение и запуск** — [установка](docs/ru/environment/setup.md) · [сборка pack](docs/ru/launch/build.md) · [запуск боя](docs/ru/launch/run.md)
- **Тестирование** — [тесты без игры](docs/ru/testing/tests.md)
- **Исследования** — [архив исследований и доказательств](docs/ru/research/README.md)

Полное оглавление: [docs/ru/README.md](docs/ru/README.md) · English: [docs/en/README.md](docs/en/README.md).

---

<a id="english"></a>

## English

The base for a battle AI for **Total War: WARHAMMER III**: measured battle
mechanics, tools to work with the game from Lua scripts, and a battle network trained
from scratch in our simulator and checked in the game against the game's AI.

The project's earlier algorithmic AI (strategy, tactics, formation, the AI tree) was removed
by the project owner's decision; it is in the Git history (the last commit with it is
`242c3b1`).

### What the project has

- **Game mechanics** — what is verified in WH3: the map, units, morale, melee,
  shooting, pace, difficulty, how the game's own AI fights — [game knowledge](docs/en/game/README.md).
- **Infrastructure** — pack build, a launcher with fair difficulty, map reading,
  unit cards, state, visibility and range, formation shapes, verified orders,
  battle context, telemetry, observation rules, and probes that measure the game
  or record battles — [code modules](docs/en/apps/README.md).
- **Training data** — rules from the game's database, recording the game's AI
  battles in the arena and a loader for those records — [data for training the network](docs/en/training/README.md).
- **The battle network** — the battle simulator, the network's inputs and model, PPO training,
  the bridge to the game and the in-game check — [training](docs/en/training/training.md),
  [simulator](docs/en/training/simulator.md), [in-game check](docs/en/launch/gate.md).

Code is split into apps; logic lives in services and engine calls only in
adapters. Tests run Lua 5.1 without the game.

### Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m tools.build nn-arena --own-ai attack --timeout 900
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target nn-arena
```

A battle needs the game installed, its path in `config/local.json` and a
subscription to the True Sight mod — [setup and configuration](docs/en/environment/setup.md).

### Documentation

- **Data for training the network** — [the arena, recorded battles, game rules](docs/en/training/README.md)
- **Game mechanics** — [what is verified in WH3](docs/en/game/README.md) · [readout catalogue](docs/en/game/readouts.md) · [atlas](docs/en/game/atlas.md)
- **Code** — [`src/apps` modules](docs/en/apps/README.md) · [entry points](docs/en/apps/entries.md)
- **Architecture** — [project structure and layers](docs/en/architecture/overview.md) · [list of apps](docs/en/architecture/apps.md) · [errors and unknown values](docs/en/architecture/error_handling.md) · [how to keep the docs](docs/en/architecture/documentation.md)
- **Environment and launch** — [setup](docs/en/environment/setup.md) · [building a pack](docs/en/launch/build.md) · [running a battle](docs/en/launch/run.md)
- **Testing** — [tests without the game](docs/en/testing/tests.md)
- **Research** — [research and evidence archive](docs/en/research/README.md)

Full index: [docs/en/README.md](docs/en/README.md) · Русский: [docs/ru/README.md](docs/ru/README.md).
