# contract — договор на стыках sim ↔ net ↔ bridge

Правило: всё, что сеть получает в симуляторе, в игре мост даёт так же — то же поле, та же формула, та же нормировка.
Новый вход, эффект или отряд — в одном задании: симулятор (`sim`) + мост (`bridge/lua`, `bridge/companion`) + строка
в [parity.md](parity.md) + проверка здесь. Владелец правки — тот, кто меняет вход.

**Кто строит вход.** Мост пишет сырые числа игры; нормировка — только в помощнике, теми же функциями сети:

| уровень | симулятор | игра (помощник) | вход сети |
|---|---|---|---|
| отряд | `net/unit/view.raw(st)` | `bridge/companion/unit_input.game_raw(doc)` | `view.RAW` → строка, карта зоны, паспорт |
| тактик | `net/tactic/raw.Seer` | `tactic_input.TacticSeer` (+ `fx`, `tr`, `hid`, кнопки) | `tactic/raw.RAW` → `tactic/inputs.build` |
| оперативник | `tactic/raw` + `operative/frame.Axes` | `operative_input.OperativeSeer` | `operative/inputs.build` |
| стратег | `strategist/cases.Pre` | `strategist_input.StrategistSeer` (документ расстановки) | `strategist/inputs.build` |

Нормировка: `men / men0`, `ammo / ammo0` — из паспорта `sim.params.static(key)`; `hp`, `mp` — доли игры; `fat` —
уровень 0–5 (`threshold_fresh…exhausted`); `x, z` — метры карты; `b` — градусы; строй (`front, depth, fill`) —
`sim.geometry.dims` от ширины `ow` и бойцов; карта — снятая один раз `sim/maps/<имя>/` по полю `map` документа.

**Документ состояния** `tww3_bai_nn_state.json` (мост → помощник, раз в `decide_ms` = 1000 мс боя): `batch, move, t`
(мс), `done, factions, attacker, decide_ms, map, abilities_used, abilities_count, winds {cur, rate}` и строка на
каждый отряд обеих сторон: `n side key x z b men hp mp ms r s w m mv f fire a t fat k ox oz lf rf bf ow tr hid v fx
[sk]`. Мост пишет полный вид; что не видит человек (`v` false), прячет помощник.

**Приказы** `tww3_bai_nn_orders.txt` (помощник → мост, опрос 100 мс): `tww3_bai_nn_orders 1`, `move N`, `batch B`,
`mode chain`, `think_ms X`, по строке на свой отряд — `unit <n> hold|keep`, `unit <n> move|withdraw x z run`,
`unit <n> attack <цель> run`, хвост `face <град> width <м>` у hold/move/withdraw; `ability <n> <ключ> [<цель>]`;
последняя строка `end` (без неё мост не берёт ничего). Отряд, который не берёт приказов (мёртв, бежит), строки не
получает. Разбор: Lua `bridge/lua/apps/bridge/services.lua parse_orders`, Python `bridge/companion/exchange.parse_orders`.
Расстановка: `tww3_bai_nn_deploy.json` → `tww3_bai_nn_plan.txt` (`place <unit> x z b width cx cz`, `end`).

**Известные расхождения** (ждут решения пользователя, подробно — [parity.md](parity.md)): `x, z` (мелко); `vx, vz,
f` — в игре сдвиг центра за 1 с, в симуляторе мгновенная скорость; `field / off_map`; «нельзя встать» во входе нет;
поводок цепочки раз в 1 с против каждого шага 0,1 с. **lf / rf / bf** (угроза флангу): игра — сбоку от 62 м, сзади от
47 м, куда бы ни смотрел враг; симулятор — 45 м и только враг лицом (`sim/battle.py threat_flags`); совпадение по
дорожкам 0,20–1,0. Во вход новых уровней не идут, но правило морали симулятора «фланги прикрыты» их читает — сверка
ждёт. Режим застрельщиков: у сети в симуляторе выключен — мост выключает его у своих отрядов по умолчанию
(`--skirmish off`; `game` — как ставит игра).

**Тесты** (нужны torch и net — контейнер):
`docker run --rm -v <проект>:/repo -w /repo -e PYTHONPATH=/repo snake-ai-trainer python -m pytest -q contract`.
Каждый берёт бой симулятора, делает из него документы моста (`test_tactic_bridge_parity_torch.doc_of_sim`) и
сравнивает вход и приказы помощника с симулятором: `unit` (сырые столбцы, вид на карте), `tactic` (сырое состояние,
строки, окна), `operative` (вход армии), `strategist` (вход до боя, места расстановки), `chain` (цепочка в игре
= цепочка симулятора, формат приказов).
