# TWW3 Battle AI

![Moorlands Route: measured heights, ground and objects](data/maps/moorlands-route/overview.png)

**Русский** · [English](#english)

Собственный боевой ИИ для **Total War: WARHAMMER III** на Lua-скриптах.
Код разбит на приложения, логика — в сервисах, обращения к движку — только в
адаптерах. Тесты проверяют Lua 5.1 без запуска игры.

## Быстрый старт

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m tools.build duel --runs 3
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target duel
```

## Документация

- **Архитектура**
  - [Устройство проекта и слои](docs/ru/architecture/overview.md)
  - [Список приложений](docs/ru/architecture/apps.md)
  - [Ошибки и неизвестные значения](docs/ru/architecture/error_handling.md)
- **Окружение**
  - [Установка и настройка](docs/ru/environment/setup.md)
- **Запуск**
  - [Сборка pack](docs/ru/launch/build.md)
  - [Запуск боя и результаты](docs/ru/launch/run.md)
- **Тестирование**
  - [Тесты без игры](docs/ru/testing/tests.md)
- **Приложения** — [core](docs/ru/apps/core.md) · [battle](docs/ru/apps/battle.md) ·
  [map](docs/ru/apps/map.md) · [navigation](docs/ru/apps/navigation.md) ·
  [units](docs/ru/apps/units.md) · [intel](docs/ru/apps/intel.md) ·
  [orders](docs/ru/apps/orders.md) · [deployment](docs/ru/apps/deployment.md) ·
  [ai](docs/ru/apps/ai.md) · [observation](docs/ru/apps/observation.md) · [sandbox](docs/ru/apps/sandbox.md) ·
  [telemetry](docs/ru/apps/telemetry.md) · [точки входа](docs/ru/apps/entries.md)
- **Знания об игре**
  - [Что проверено в WH3](docs/ru/game/README.md) — карта, отряды, каталог карт
  - [Каталог показателей](docs/ru/game/readouts.md) — что собираем и что ещё можно
  - [Визуальный атлас](docs/ru/game/atlas.md)
- **Исследования**
  - [Архив исследований и доказательств](docs/ru/research/README.md)

Полное оглавление: [docs/ru/README.md](docs/ru/README.md).

---

<a id="english"></a>

## English

Our own battle AI for **Total War: WARHAMMER III**, written as Lua scripts.
Code is split into apps; logic lives in services and engine calls only in
adapters. Tests run Lua 5.1 without the game.

### Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m tools.build duel --runs 3
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target duel
```

### Documentation

- **Architecture**
  - [Project structure and layers](docs/en/architecture/overview.md)
  - [List of apps](docs/en/architecture/apps.md)
  - [Errors and unknown values](docs/en/architecture/error_handling.md)
- **Environment**
  - [Setup and configuration](docs/en/environment/setup.md)
- **Launch**
  - [Building a pack](docs/en/launch/build.md)
  - [Running a battle and results](docs/en/launch/run.md)
- **Testing**
  - [Tests without the game](docs/en/testing/tests.md)
- **Apps** — [core](docs/en/apps/core.md) · [battle](docs/en/apps/battle.md) ·
  [map](docs/en/apps/map.md) · [navigation](docs/en/apps/navigation.md) ·
  [units](docs/en/apps/units.md) · [intel](docs/en/apps/intel.md) ·
  [orders](docs/en/apps/orders.md) · [deployment](docs/en/apps/deployment.md) ·
  [ai](docs/en/apps/ai.md) · [observation](docs/en/apps/observation.md) · [sandbox](docs/en/apps/sandbox.md) ·
  [telemetry](docs/en/apps/telemetry.md) · [entries](docs/en/apps/entries.md)
- **Game knowledge**
  - [What is verified in WH3](docs/en/game/README.md) — map, units, map catalogue
  - [Readout catalogue](docs/en/game/readouts.md) — what we collect and what else is available
  - [Visual atlas](docs/en/game/atlas.md)
- **Research**
  - [Research and evidence archive](docs/en/research/README.md)

Full index: [docs/en/README.md](docs/en/README.md).
