# Устройство проекта и слои

[Документация](../README.md) · [English](../../en/architecture/overview.md)

Проект устроен по образцу photo-fixing: код разбит на **приложения** по
предметным областям, логика живёт в **сервисах**, а точки входа только
связывают их между собой. Главное отличие: код работает внутри игры
(Lua 5.1 в WH3), поэтому обращения к движку собраны в отдельном слое —
**адаптерах**.

## Папки

```text
src/
├── apps/              приложения: core, battle, map, navigation, units, intel,
│                      orders, deployment, ai, sandbox, telemetry
└── entries/           точки входа: duel, arena, map_capture
scenarios/             XML-сценарии боёв
tools/                 Python/PowerShell вне игры
├── build.py           сборка pack: python -m tools.build <цель>
├── pack/              формат PFH5 и бандлер Lua-модулей
├── launcher/          установка pack, запуск WH3, ожидание, очистка
├── telemetry/         чтение JSONL, пока игра пишет
└── analysis/          карты высот, склоны, проходы по сохранённой сетке
tests/                 pytest + lupa: тот же Lua 5.1 без игры
config/                default.json в Git, local.json — настройки машины
data/                  компактные проверенные данные по картам и отрядам
docs/ru, docs/en       документация на двух языках
research/              архив исследований: доказательства, зонды, старые скрипты
```

## Слои приложения

Каждое приложение — папка `src/apps/<имя>/` с файлами по ролям:

| Файл | Аналог в photo-fixing | Правило |
|---|---|---|
| `*adapter.lua` | `models.py` | **Единственное место**, где вызываются `bm`, объекты отрядов, `common` (CCO). Возвращает обычные таблицы |
| `services.lua` | `services.py` | Чистая логика над обычными таблицами. Без движка, без файлов, без таймеров |
| `contract.lua` | `schemas.py` | Форма данных и её проверка: команды, план расстановки, профили |
| `policies/` | — | Только в `ai`: сменные стратегии |

Не у каждого приложения есть все слои. Например, в `units` только адаптеры и
контракт, а в `intel` — адаптер и сервис памяти.

**Точки входа** (`src/entries/`) — аналог `router.py`: подписываются на фазы
боя, запускают тик, вызывают адаптеры и сервисы и пишут телеметрию. Логики
решений в них нет.

## Правила зависимостей

```text
entries     →  любые apps
sandbox     →  core, ai.contract, orders.contract
deployment  →  core, units.contract, orders.adapter (только в своём адаптере)
orders      →  core, units.contract
navigation  →  core, map.adapter (построение вектора)
ai, battle, intel, map, telemetry, units  →  core (и свой contract)
core        →  ничего
```

Схема проверяется тестом `tests/tools/test_build.py`: бандлер падает на
циклическом `require`.

- `core` ничего не импортирует из приложений. В photo-fixing `common` знал о
  `auth` и `routes` — здесь так делать нельзя.
- Сервисы не импортируют адаптеры. Если сервису нужен движок, его передают
  колбэком (см. `deployment.services.verify(measure, pair_distance)`).
- Модули подключаются обычным `require('apps.map.services')`. Сборщик кладёт
  все модули в один скрипт и подставляет свой загрузчик
  ([сборка](../launch/build.md)).

## Поток данных в бою

```text
XML-сценарий  →  WH3 загружает бой и наш скрипт из pack
  →  entry: проверка типа боя (battle), чтение сторон
  →  фаза Deployment: расстановка (deployment)
  →  фаза Deployed: захват управления (orders), тик раз в tick_ms:
        units / intel / map  →  наблюдение
        ai                   →  решение
        orders.contract      →  проверка команд
        orders.adapter       →  приказы движку
        telemetry            →  events.jsonl
  →  результат, очистка колбэков, переигровка или конец
  →  launcher копирует результаты в build/<цель>/runs/<время>/
```

## Как добавить приложение

1. Создать `src/apps/<имя>/` и разделить код: вызовы движка — в `adapter.lua`,
   логику — в `services.lua`, форму данных — в `contract.lua`.
2. Написать тесты в `tests/apps/<имя>/`. Сервисы тестируются напрямую,
   адаптеры — на поддельных объектах движка.
3. Описать приложение в `docs/ru/apps/<имя>.md` и `docs/en/apps/<имя>.md`,
   добавить строку в [список приложений](apps.md) и в оглавления.
