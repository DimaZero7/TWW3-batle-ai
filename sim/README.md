# sim — симулятор боя TWW3 на PyTorch

Много боёв сразу тензорами [B, N]: один отряд — строка (люди, здоровье, место, фронт, мораль, усталость, боезапас).
Шаг 0,5 с. На нём учатся уровни сети (`net`), с ним пробы игры (`game`) сверяют игру («двойник»).

**Запуск.** Бой в коде: `st = scenario.build(армии, params.load())`, затем `battle.step(st, orders, P)` по шагам
(`battle.stepper(device)` — со сборкой `torch.compile` на видеокарте). Армии — `armies.generate.battle(seed)` →
`scenario.from_arena(...)`; противник — `scripts.human / nearest / hold_shoot` (учёба; `ai_game` — только экзамен этапа); счёт боя — `reward.measure`.
Тест (дымовой: бой идёт и повторяется один в один):
- хост без torch: `../TWW3-batle-ai/.venv/Scripts/python -m pytest -q sim/tests` (только проверка импорта);
- контейнер: `docker run --rm -v <проект>:/repo -w /repo -e PYTHONPATH=/repo snake-ai-trainer python -m pytest -q sim/tests`
  (образ `snake-ai-trainer`: Python 3.12, torch 2.13 + CUDA 13.2, g++, pytest; Dockerfile нет).

**Числа — только из игры, у каждого рядом `_why`: откуда и почему.**
- `config/units.json`, `abilities.json`, `effects.json`, `game_rules.json` — база игры (`db.pack`); пересобрать:
  `py -3.14 -m sim.passports.units --units k1,k2`, `... .abilities`, `python -m sim.passports.effects`,
  `py -3.14 -m sim.passports.gamedb` (путь к игре — общие настройки `bridge.config`).
- `config/sim.json` — то, чего в базе нет: замеры проб, разбор программы игры (Ghidra, memreader Plus), сообщество.
- `config/pools.json`, `army_templates.json`, `arena.json` — армии и арена; `data/` — паспорта проб, карточки отрядов.
- `maps/<имя>/` — снятые карты (высоты, лес, стволы, «нельзя встать»), снимаются один раз в игре; путь — `sim.maps.MAPS`.

**Доказательства.** Ссылки `_why` ведут в `sim/evidence/<папка>/` — разборы и выводы проб (md, в git).
Сырые прогоны (json, npz, записи боёв) в git не входят: ссылки на них — `tmp/old-build/<папка>/...`.
