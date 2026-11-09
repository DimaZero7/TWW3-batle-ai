# bridge — мост между сетью и игрой

Мод на Lua внутри Total War: WARHAMMER III и помощник в Docker. Мост раз в секунду боя пишет в папку игры документ
(каждый отряд обеих сторон), берёт ответ помощника и отдаёт приказы своим отрядам. Сеть видит в игре то же, что в
симуляторе: договор и таблица сверки — [contract/](../contract/README.md). Полная документация — [docs/](docs/README.md).

| часть | что |
|---|---|
| `lua/apps/*` | модули мода: `bridge` (обмен, правила приказов, расстановка), `orders` (приказы движку), `battle` (бой, живая разметка), `map`, `navigation`, `units`, `telemetry`, `core`; `mem` — память игры через memreader Plus, только пробы |
| `lua/entries/*` | 6 входов: `nn_arena` (бой цепочки, бой человека), `charge/missile/morale_probe` (пробы механик), `map_capture` (снять карту), `move_probe` (путь) |
| `build.py`, `pack/` | `python -m bridge.build <цель>`: Lua в один скрипт и `.pack` в `build/<цель>/` (цели `nn-arena human charge-probe missile-probe morale-probe map-capture move-probe`) |
| `launcher/` | `launch.ps1`: ставит пак в `data/` игры, «Нормальная», графика (`BAI_GRAPHICS`), запуск, уборка; `watch.ps1` — бой с помощником |
| `companion/` | помощник (образ `snake-ai-trainer`): `--chain` (оперативник → тактики → сеть отряда), `--strategist` (расстановка); сети — через `net.<уровень>.ckpt` |
| `config.py`, `config/` | папка игры (`local.json` вне git поверх `default.json`), обязательный мод True Sight, memreader, планы `move-probe` |
| `scenarios/` | файлы боя: `nn_arena.xml` (арена `sim/arena.py`), `map_capture.xml`, `move_probe.xml`, `lord_swarm.xml` (основа проб) |

Бой цепочки:

    python -m bridge.build nn-arena --army-seed S --armies v3 --map ksl_plains_01 --deploy strategist --own-ai net --speed 10 --timeout 1200
    powershell -File bridge/launcher/watch.ps1 -NoBuild -Strategist default -StrategistTask hold -Chain

Оперативник цепочки по умолчанию — формула (`-ChainOp formula`): сети оперативника на 40 групп пока нет (run8 — на 6).

Файлы обмена в папке игры: `tww3_bai_nn_deploy.json` → `tww3_bai_nn_plan.txt` (расстановка), `tww3_bai_nn_state.json`
→ `tww3_bai_nn_orders.txt` (каждая секунда); запись через `.tmp`, приказы кончаются строкой `end`.

Закреплено по SHA-256: True Sight (`config/mod-dependencies.json`), memreader Plus (`config/memreader.json`, места —
`reverse/mem/sites_exe1103.json`). Лаунчер не запускает игру при `state/HUMAN_PLAYING` (кроме цели `human`) и при чужом
`state/probe_game.lock` (держатель передаёт `BAI_LOCK_HELD`).

Тест (дымовой): `python -m pytest -q bridge/tests` — на хосте сборка пака и все входы Lua; в контейнере ещё помощник
цепочки отвечает на записанный документ боя М1 (`tests/data/state_m1.json`). Нельзя: бои против ИИ игры не на
«Нормальной»; memreader в бою сети; новое поле входа без строки в `contract/parity.md`.
