# Сборка pack

[← Назад](README.md) · [Документация](../README.md) · [Запуск](run.md)

Сборка не трогает игру: она только создаёт файлы в `build/<цель>/` (вне git).

```bash
python -m bridge.build nn-arena --own-ai net --deploy strategist --army-seed N --armies v3 --map ksl_plains_01 --speed 10
python -m bridge.build charge-probe --probe-plan charge --probe-battle 1 --mem
python -m bridge.build map-capture --step 3 --features
```

## Цели

| Цель | Точка входа | Сценарий | Pack |
|---|---|---|---|
| `nn-arena` | `entries.nn_arena` (`own_ai` `net`) | `build/nn-arena/<арена>.xml` (`sim/arena.py`; `bridge/scenarios/nn_arena.xml` не трогается) | `tww3_bai_nn_arena.pack` |
| `human` | `entries.nn_arena` (`own_ai` `human`) | бой арены в `build/human/` ([бой, который играет человек](run.md#бой-который-играет-человек)) | `tww3_bai_human.pack` |
| `charge-probe` | `entries.charge_probe` | файл боя плана (`game/probe/charge_probe.py`) | `tww3_bai_charge_probe.pack` |
| `missile-probe` | `entries.missile_probe` | файл боя плана (`game/probe/missile_probe.py`) | `tww3_bai_missile_probe.pack` |
| `morale-probe` | `entries.morale_probe` | файл боя плана (`game/probe/morale_probe.py`) | `tww3_bai_morale_probe.pack` |
| `map-capture` | `entries.map_capture` | `bridge/scenarios/map_capture.xml` | `tww3_bai_map_capture.pack` |
| `move-probe` | `entries.move_probe` | `bridge/scenarios/move_probe.xml` | `tww3_bai_move_probe.pack` |

Что делает каждая точка входа — [точки входа](../apps/entries.md).

## Параметры

| Параметр | Цели | Значение |
|---|---|---|
| `--speed 1/3/10/20` | все, кроме map-capture | Скорость боя (по умолчанию 20; у `human` — 1). Бой цепочки — ×10: на ×20 три сети не успевают за секунду боя |
| `--timeout 30..3600` | nn-arena, human | Лимит модельного времени боя, с (600; у `human` — 3600); из него считается срок боя |
| `--tick-ms` | все, кроме map-capture | Период тика, мс (1000) |
| `--deadline` | все, кроме map-capture | Лимит реального времени на бой, с (по умолчанию из длины сценария и скорости) |
| `--stall-minutes` | все, кроме map-capture | Завершить бой, если никто не получает урон столько минут игры (10). В пробах и move-probe — не меньше длины сценария + 2 мин |
| `--scenario` | все | Другой файл из `bridge/scenarios/` вместо сценария цели |
| `--own-ai net` | nn-arena | Нашей стороной командует помощник ([смотреть бой сети](watch.md)); по умолчанию |
| `--own-role attack\|defend` | nn-arena, human | Наша сторона атакует (ИИ игры обороняется и побеждает, когда время вышло) или обороняется (по умолчанию: ИИ игры атакует) |
| `--deploy script\|strategist` | nn-arena | Кто ставит нашу сторону в расстановке: места файла боя (по умолчанию, фаза кончается через 1 с) или стратег помощника (`companion --strategist`, [расстановка стратегом](../apps/bridge.md#расстановка-стратегом)) |
| `--decide-ms 250..5000` | nn-arena | Время боя между двумя решениями, мс (1000) |
| `--army-seed N` | nn-arena, human | Бой генератора армий (`sim/armies`, `generate.battle(N)`): у стороны лорд и 0–19 отрядов. Файл боя — `build/nn-arena/random_<N>.xml`. В манифест — `army` (зерно, train/eval, бюджет, шаблоны, бойцы) |
| `--armies v1\|v2\|v3` | nn-arena `--army-seed` | Схема генератора (`sim/armies/generate.py`): `v3` — схема обучения |
| `--army-swap` | nn-arena `--army-seed`, `--army-from-run` | Армии зерна меняются местами: нашей стороне — армия противника из генератора, ИИ игры — наша (`random_<N>_swap.xml`, в манифесте `army.swap` — true) |
| `--army-from-run ПАПКА` | nn-arena, human | Армии, расстановка, зазор и радиус обороны старого запуска (`manifest.json` папки прогона) вместо генератора. Численность отрядов — из пулов (`sim/config/pools.json`), сумма сверяется с `army.men` манифеста (не сходится — ошибка). `--army-swap` меняет стороны этого запуска местами. Файл — `<арена>_run<id>[_swap].xml`, в манифесте `army` старого запуска и `army.from_run`. Не вместе с `--army-seed` (`game/arena/army_from_run.py`) |
| `--army-replace OLD=NEW` | nn-arena `--army-from-run` | Каждый отряд ключа OLD становится отрядом ключа NEW на том же месте; можно повторять |
| `--map ИМЯ` | nn-arena, human | Бой на снятой карте `sim/maps/ИМЯ/` (`meta.json`): её файл боя, ландшафт и зоны расстановки (`sim/maps/__init__.py`); с `--army-seed` / `--army-from-run`. Путь ландшафта идёт в состояние моста (`map`) |
| `--skirmish off\|game` | nn-arena | Режим застрельщиков отрядов под мостом: выключен у всех (по умолчанию — как у сети в симуляторе) или как его ставит игра ([bridge](../apps/bridge.md#режим-застрельщиков)) |
| `--label-own`, `--label-enemy` | nn-arena, human | Подпись над армией в игре (имя генерала в сценарии) — свой текст вместо того, кто управляет стороной |
| `--free-speed` | human | Бой начинается на `--speed`, человек может менять скорость и ставить паузу (скорость не держится; решения и опросы — по времени боя) |
| `--soldiers-every N` | human | Место каждого бойца каждые N тиков (по умолчанию 5; 0 — никогда); чтение всех бойцов ~50 мс |
| `--live-label` | human | Разметка вживую: карточки врага, группы человека и состояние каждую секунду ([battle](../apps/battle.md#разметка-вживую), `game/label/live.py`) |
| `--live-test` | human `--live-label` | Команды проверки скриптом из `tww3_bai_live_cmd.txt` в папке игры |
| `--extra-mod WORKSHOP_ID` | human | Ещё один мод мастерской на этот бой (пак из папки мастерской, закреплён по SHA-256 сейчас; launcher копирует его в `data/` и убирает после) — только свои моды человека для его боя, никогда не в бои сети |
| `--probe-plan`, `--probe-battle N` | charge-probe | План [зонда рукопашной](../apps/entries.md#charge_probe) (`PLANS` в `game/probe/charge_probe.py`) и номер его боя (с 1). Замер каждые 0,5 с |
| `--mprobe-plan`, `--mprobe-battle N` | missile-probe | План [пробы стрельбы](../apps/entries.md#missile_probe) и номер боя |
| `--morale-plan`, `--morale-battle N` | morale-probe | План [пробы морали](../apps/entries.md#morale_probe) и номер боя |
| `--mem` | charge-, missile-, morale-probe | memreader Plus рядом с нашим паком: снаряды в полёте, бойцы, мораль из памяти игры ([memreader](../apps/memreader.md)); места — из `sites_file` в `bridge/config/memreader.json` |
| `--kv [SECTION.]KEY=VALUE\|xN` | charge-, missile-, morale-probe | Строка таблиц правил игры (`_kv_morale` / `_kv_rules` / `_kv_fatigue`: секции `morale` / `battle` / `fatigue`) — VALUE или N × значение игры, таблицей своего пака пробы (`bridge/pack/kvtable.py`); можно повторять. Никогда для боя, который пишет данные: правило не игры |
| `--step 1/2/3/5` | map-capture | Размер клетки сетки, м |
| `--nodes` | map-capture | Опрос в кратных шагу точках (узлы высот движка), а не в центрах клеток |
| `--features` | map-capture | После расстановки прочитать объекты и достижимость клеток |
| `--window MIN_X MAX_X MIN_Z MAX_Z` | map-capture | Снять только этот участок карты |
| `--plan` | move-probe | План из `bridge/config/move-plans/<имя>.json` (по умолчанию `hamlet`) |

## Что лежит в `build/<цель>/`

- `<pack>.pack` — архив для папки `data` игры;
- `<скрипт>.lua` — собранный скрипт, удобно читать при отладке;
- `manifest.json` — `build` (короткий SHA-256 всех входов), конфигурация,
  список модулей, SHA-256 pack, путь сценария, `syntax_checked`.

Значение `build` попадает в каждую строку телеметрии, поэтому по журналу
всегда видно, какой именно код играл.

## Как устроен бандлер

`bridge/pack/bundle.py` начинает с точки входа и рекурсивно находит все
`require('...')` (модули ищутся от `bridge/lua/`). Каждый модуль оборачивается в функцию-фабрику, а вместо
глобального `require` ей передаётся свой загрузчик:

```lua
__modules["apps.core.value"] = function(require)
    -- исходник bridge/lua/apps/core/value.lua
end
...
if bm then
    __require("entries.nn_arena").main(bm, {own_ai = "net", ...}, {common = common, battle_vector = battle_vector})
end
```

Поэтому модули пишутся как обычный Lua и так же загружаются в тестах.
Циклический `require` — ошибка сборки. В пак попадают только модули, до которых дотягиваются `require` от точки входа.

## Содержимое pack

```text
script\battle\mod\<скрипт>.lua          наш скрипт; игра загружает его в каждом бою
script\battle\<папка>\scenario.lua      load_script_libraries()
script\battle\<папка>\<сценарий>.xml    XML боя
```

Формат — PFH5 без сжатия (`bridge/pack/pfh5.py`, по описанию RPFM). Записи
отсортированы, одинаковые входы дают одинаковые байты. Обязательный мод
`true_sight.pack` из `bridge/config/mod-dependencies.json` всегда попадает в заголовок pack. С `--kv` в пак идут свои
таблицы правил, с `--live-label` — картинки разметки.

Поскольку скрипт из `script\battle\mod\` загружается в любом бою, **не
добавляйте тестовый pack в список модов кампании**. Launcher ставит его
только на время запуска.
