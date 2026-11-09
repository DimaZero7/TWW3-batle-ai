# Моды, исследования сообщества и инструменты — что даёт точности симулятора

Поиск 12.10.2026 (WebSearch/WebFetch). Страницы Steam-руководств (sharedfiles) отдавали 429, twcenter — 403,
fandom — 402: по ним брались выдержки поиска (помечено «выдержка»). Сверено с `config/nn/game_rules.json`,
`tools/nn/gamedb.py`, `tools/nn/dbtables.py`, `docs/ru/game/mechanics/*.md`, `build/holes/report.md`,
`build/holes/wavering.md`, разделом «Чего нет» в `docs/ru/training/simulator.md`.

Главный вывод: почти все «скрытые параметры» из присланной заметки — это строки `_kv_rules_tables`, которые мы
уже читаем. Нового по-настоящему три вещи: (1) единица времени таймеров морали, вероятно, 0,1 с — это прямо бьёт
по «бегств меньше, чем в игре»; (2) настоящие имена ~80 безымянных столбцов базы через схемы RPFM; (3) пробы с
подменённым значением kv в своём моде — способ узнать смысл ключа из самой игры.

## Сводная таблица (по пользе на цену, лучшие сверху)

| # | Находка | Польза для наших расхождений | Цена | Проба в игре |
|---|---|---|---|---|
| 1 | Таймеры морали в тиках 0,1 с: `waver_base_timeout` 25 = 2,5 с; колеблющийся, получивший потери, бежит по истечении таймера | бегств 0,82 против 1,19; ячейки «wavering» сим/игра 0,52–0,74 и «shaken» 27–33 (сим бежит прямо из «потрясён») | анализ записей ~1–2 ч агента, правило в `morale.py` ~полдня | не нужна сначала (есть записи); потом 2 боя |
| 2 | Шок морали: ≥25 % потерь за 4 с при морали ≤13 очков → короткое бегство | бегств мало; бегущие теряют меньше; всплески после натиска/залпа | анализ записей ~1 ч | не нужна сначала |
| 3 | Схемы RPFM (`rpfm-schemas`, `schema_wh3.ron`, MIT) — настоящие имена столбцов | убрать догадки в `dbtables.py` (`INFERRED`, f05…f57); числа очереди (burst_size/burst_shot_delay крысиного пулемёта), collision_attack_* и масса для крысоволков | скачать файл ~>10 МБ (нужно разрешение) или поставить RPFM; сопоставить ~1–2 ч | нет |
| 4 | Проба «подмена одного kv в своём моде» | узнаём смысл и единицы ключа из игры: `waver_base_timeout`, `broken_finish_base_timeout`, `charge_timeout`, `melee_attack_interval` оружия | пакет уже собираем (`tools/pack/pfh5.py`, `gamedb.mod_rules`); 1 бой-проба на ключ | да, 1–2 боя на ключ |
| 5 | Формулы столкновений и сбивания с ног (руководство «Total war: Warhammer — Some notes», WH2) | крысоволки/гигантские крысы скоро: натиск зверей сбивает людей, сбитые не бьют 1–2,7 с | правило на уровне отряда ~1 день; нужна проба | да (проба wolfrats уже есть) |
| 6 | Анимации (AssetEditor, rmv2_anim_blender, .animpack/.frg/.anm.meta) — длина удара по типу оружия | темп ударов новых отрядов без проб; «двуручные быстрее, алебарды медленнее» | высокая (бинарные ресурсы, GUI-инструменты) | для проверки — да |
| 7 | Matched combat (`matched_combat_*`) | возможная причина разного темпа убийств «строй на строй» и по одиночкам | только анализ, смысл неточный | возможно |
| 8 | Инструменты «гонять бои быстрее» | ничего нового: ускорение `bm:modify_battle_speed` и полосы уже используем | — | — |

## 1. Таймеры морали в тиках 0,1 с (самое ценное)

Что найдено:
- fandom «Leadership» (выдержка поиска; её же пересказывает screenrant): **«мораль ниже 0 в течение 2,5 с или
  дольше — отряд бежит»**; колебание ниже 16. `waver_base_timeout` = 25 → ровно 2,5 с, если тик 0,1 с.
- Учебник по морали Empire: Total War (twcenter, тот же движок; выдержка): «отряд, дошедший до колебания,
  колеблется `waver_base_timeout`; **если за это время он понёс потери, он бежит, когда таймер истечёт**»; иначе
  может вернуться к «стойкий».
- Тема twcenter «On Morale» (выдержка): `broken_finish_base_timeout` 600 в Empire — «похоже, десятые доли
  секунды, то есть 1 минута».
- Наше собственное: пики длины бегства до сбора на **18–19 и 36–37 с** (`broken_finish_base_timeout` 180 = 18 с при
  тике 0,1 с) и усталость, которая сходится только при 10 тиках/с (`docs/ru/game/mechanics/fatigue.md`).

Что у нас сейчас: `docs/ru/game/mechanics/morale.md` читает `waver_base_timeout` как 25 **с** и не моделирует;
`tools/nn/sim/morale.py` начинает бегство сразу, как мораль ≤ 0 (строка `start = ... (M <= 0)`), без таймера.

Почему это может закрыть дыру: `build/holes/wavering.md` — в симуляторе колеблющиеся бегут в 0,52–0,74 раза реже
игры, а «потрясённые» (ещё не колеблются) бегут в 27–33 раза чаще. Оба знака совпадают с недостающим правилом:
в игре нельзя побежать прямо из «потрясён» за одно окно (нужно ≥2,5 с ниже 0), а колеблющийся под потерями бежит
по таймеру даже без падения ниже 0.

Осторожно: тот же тик дал бы `post_rally_no_rout_timer` 10 = 1 с (у нас 10 с, и повторных бегств и так больше игры)
и `morale_shock_rout_timer` 14 = 1,4 с (наши документы пишут «ужас: бежит 14 с» со ссылкой на fandom). То есть
единицы могут быть разными у разных ключей — решают записи.

Проверка без игры (на записях, которые уже есть): (а) время от первой секунды MoralePercent ≤ 0 до флага бегства;
(б) от начала колебания до бегства — отдельно для тех, кто терял здоровье в первые 2,5 с колебания, и тех, кто нет;
(в) есть ли бегства без MoralePercent ≤ 0 (у колеблющихся под потерями). Если (а) ≈ 2,5 с, а (в) находится —
правило в `morale.py` (таймер колебания 2,5 с + «потери в колебании → бегство по таймеру»), сверка по
`build/holes` (каналы rout) и replay 72 боёв. Записи пишутся раз в секунду — 2,5 с различимы.

## 2. Шок морали

- kv: `recent_casualties_shock_threshold` 25, `morale_shock_terror_morale_threshold_short/long` 13,
  `morale_shock_rout_timer_short/long` 14, `morale_shock_rout_immunity_timer` 85.
- Сообщество (twcenter «"Morale Shock" explained», Empire/Shogun 2, выдержка): 20 % потерь за 4 с (в WH3 — 25) →
  «шок морали», резкий штраф, отряд легко бежит. В WH3 те же ключи описывают «ужас» (≤13 очков → бегство).
- У нас: в `morale.md` «Шок морали — не моделируется».
- Проверка на записях: окна, где отряд потерял ≥25 % здоровья за 4 с; доля бегств в следующие 1–2 с при морали
  ≤13 очков и выше 13. Если шок есть — он добавляет бегства после натиска и залпов в упор (наша дыра «бегств
  меньше»). Цена — час анализа.

## 3. Схемы RPFM — настоящие имена столбцов

- `github.com/Frodo45127/rpfm-schemas` — `schema_wh3.ron`: версии всех таблиц WH3 с именами полей (имена из
  Assembly Kit CA). Файл больше 10 МБ (WebFetch не берёт), лицензия MIT. Есть `rpfm_cli` для выгрузки таблиц.
- У нас в `tools/nn/dbtables.py` безымянны: `battle_entities` f12, f18–f57 (~40 полей), `projectiles` ~30 полей
  (f12…f57), `melee_weapons` f05–f19; ещё список `INFERRED` — имена, данные нами по значениям.
- Подтверждено, что такие поля существуют (сообщество): `burst_size` и `burst_shot_delay` (очередь: крысиный
  пулемёт, варпогнемёты), `marksmanship_bonus`, `collision_attack_max_targets` и `..._cooldown`,
  `weapon_length`, `splash_attack_*`; у `battle_entities` — `charge_distance_commence_run` / `adopt_charge_pose`
  (наши имена из `INFERRED` верны по смыслу).
- Польза: числа вместо замеров для новых отрядов (очередь пулемёта — мы её мерили пробой: 18 выстрелов за 7–8 с),
  проверка догадок, поля, о которых не знали. Цена: скачать файл (разрешение пользователя), скрипт сопоставления
  версий таблиц — 1–2 ч. Проба не нужна.

## 4. Проба «подмена kv в своём моде»

- Моды из списка (Realistic Physics, Realistic Charge, Hecleas Animation Overhaul) делают ровно это: правят строки
  `_kv_rules_tables`. Игра читает строки модов поверх db.pack — наш `gamedb.mod_rules` это уже учитывает.
- Идея: пакет-проба, где один ключ изменён в 10 раз (например `waver_base_timeout` 25 → 250,
  `broken_finish_base_timeout` 180 → 360), и короткая проба морали: если пики сбора сдвигаются с 18 на 36 с —
  единица доказана из игры. Так же — `charge_timeout`, смысл `melee_attack_interval` оружия, `matched_combat_*`.
- Цена: пакет собираем сами (`tools/pack/pfh5.py`), 1–2 боя на ключ. Это правка ресурсов игры, не её программы.
  Важно: пакет-пробу снимать после проб (иначе испортит обычные бои; `gamedb` читает моды).

## 5. Столкновения и сбивание с ног (для крысоволков и гигантских крыс)

Руководство Steam «Total war: Warhammer — Some notes» (sharedfiles 1114878285, WH2; выдержка) и «All the
mechanics CA didn't explain to you» (тема Steam WH2, прочитана):
- шанс отбросить = 10 % × (масса атакующего / масса цели), сбить с ног = 5 % × отношение; при натиске ×2; при
  отношении ниже 0,35 / 0,5 отбросить / сбить нельзя. Это ровно наши kv: `melee_on_damage_effect_knockback_chance`
  0,1, `..._knockdown_chance` 0,05, `melee_on_damage_effect_knock_chance_charging_bonus` 2,
  `interrupt_mass_ratio_clamp_knockback` 0,35, `..._knockdown` 0,5 — совпадение подтверждает чтение;
- сила столкновения = (скорость 1 + скорость 2) × (масса 1 / масса 2); урон = сила^0,8 (`collision_damage_normaliser`)
  × 0,6 (`collision_damage_modifier`), предел 70 (`collision_damage_maximum`), 70 % бронебойный
  (`collision_damage_armour_penetration_ratio`), не чаще раза в 4 с (`collision_damage_immunity_timer`; в WH1 было 3);
- стоящий лицом к врагу отряд (упор) получает +600 массы (`bracing_immovable_by_small_entity_mass` 600);
- сбитый встаёт через 1,0–2,7 с (`knockdown_getting_up_delay_min/max`) и в это время не бьёт.
- Крысоволк (fandom): масса 150, скорость 95, натиск 30. Наши пробы: бег 9,6 м/с, натиск 11,5–12,3.
- Для модели «отряд — один объект»: доля бойцов фронта цели, сбитых при ударе натиском (по формуле шанса и
  отношению масс), минус их удары на 1–2,7 с, плюс урон столкновения по ним. Цена ~1 день; число сбитых нужно
  сверить пробой wolfrats (есть). Сейчас у нас в `sim.json` charge: «collision damage — не моделируется, все модели
  малы» — с крысоволками это перестаёт быть правдой.

## 6. Анимации: темп ударов по типу оружия

- Что известно: `melee_attack_interval` у почти всех оружий 4–5 с; спорят, «минимум между ударами» (Alvieres)
  или «поправка к длине анимации, чтобы темп класса был одинаков» (Spellbound1875, форум CA). Анимация удара
  длится 2,9–6 с, у большинства ~3,8 с (руководство kv WH3, выдержка). Тема Steam WH2: быстрее всех бьют с двумя
  оружиями, потом двуручные, потом меч/топор/копьё со щитом, медленнее всех алебарды. Удар без касания (цель
  отброшена до конца анимации) урона не наносит — так монстры теряли ~30 % урона (форум CA, ледяной змей).
- Где лежит: `land_units.man_animation` → набор анимаций → `.animpack` (`.bin` + фрагменты `.frg`: «слот» действия
  → анимация) → `.anim` (длина) и `.anm.meta` (метки: позиция удара, эффекты). Инструменты: The Asset Editor
  (donkeyProgramming, MIT; редактор animpack и anm.meta), `rmv2_anim_blender_import_export` (Python, читает
  `.anim` v4–v8), RPFM (извлечь пакеты); гайд Steam «Creating Custom Unit Animation Sets for WH3» и
  tw-modding «Animation and You» — общие слова, без имён полей длительности.
- У нас: темп = `p / (p × interval + miss_s)` (`tools/nn/sim/melee.py`), подогнан пробами по парам. Для новых
  отрядов (крысоогры, звери) длина анимации дала бы стартовое число до пробы. Цена высокая (бинарные форматы,
  GUI), польза средняя: пробы всё равно нужны. Отложить.

## 7. Matched combat (парные анимации)

- kv: `generic_matched_combat_percentage` 60, `matched_combat_formed_vs_formed_percentage` 40,
  `..._formed_charge_` 20, `..._formed_vs_free_` 60, `matched_combat_percentage` 50,
  `matched_combat_min_hitpoints_to_initiate` 15, `..._whilst_active` 5.
- Hecleas Animation Overhaul ставит все проценты 100 и пороги 2 / 1; игроки пишут, что бои «становятся намного
  длиннее» — значит, парные схватки медленнее обычных ударов. Отсюда гипотеза: темп убийств зависит от того,
  строй против строя (40 %) или против «свободных» (лорд, рассыпанный строй — 60 %). Подтверждений с числами нет.
- Польза: возможно, часть разницы «лорд против отряда» и рукопашной у стрелков. Цена: только через пробу с
  подменой kv (п. 4). Низкий приоритет.

## 8. Инструменты «гонять бои быстрее» и прочее

- Безголового режима игры или открытого симулятора боя у сообщества нет. TWW3 Brawl (`tww3-org/tww3-brawl`) —
  калькулятор на данных twwstats, считает только здоровье отряда, боя не моделирует. Автобой — грубая формула по
  характеристикам, к нашему вопросу не относится.
- Ускорение боя: мы уже ставим `bm:modify_battle_speed(10)` и гоняем несколько пар за бой — лучше у моддеров нет.
- Скриптовый интерфейс отряда (chadvandy, WH3 `battle_unit`): `is_wavering`, `is_routing`, `is_shattered`,
  угрозы флангов, `strategic_value`, `fatigue_state`, `number_of_enemies_killed` — всё это мы, судя по записям,
  уже берём. Нового полезного нет.
- Патчи: 8.0 (20.05.2026), 8.1 (09.07.2026), 9.0 (24.09.2026): общих правок рукопашной, стрельбы, морали нет; у
  отрядов Империи и скавенов нашего набора — нет. 9.0: «отступающие в режиме застрельщиков теперь могут впасть в
  буйство» — не наш случай. Наш db.pack 9.0.1 — актуален.
- honga.net (Royal Military Academy) — карточки с массой и скрытыми числами, удобна для сверки.

## Проверка заметки другой нейросети

| Утверждение | Итог |
|---|---|
| Realistic Physics (GunPawDa): макс. урон столкновения 70, иммунитет 4,0 с, восстановление скорости 0,3, игнор 2,0 с, порог HP 50 | Значения верны — это строки kv_rules, они уже в `game_rules.json` (`collision_damage_maximum`, `collision_damage_immunity_timer`, `collision_restitution_coefficient`, `collision_disable_delay_*`, `hitpoint_threshold_to_fall_through_collision`). Но «порог HP 50» — про падение с обрыва/моста (описание мода), не про сбивание с ног. Мод удалён из мастерской |
| Realistic Charge: knocked_flying 10,5 → 20 | 10,5 совпадает с нашим kv; страница не открылась (429) |
| Enhanced Cavalry Breakthrough | не проверял: кавалерии у нас нет |
| Better Combat Animations: короче атаки — выше урон | мод удалён; описание: «быстрее следующий удар» — согласуется с тем, что темп ограничен анимацией |
| Hecleas Animation Overhaul | верно, правит `matched_combat_*` (см. п. 7) |
| Ridercs35 «Hidden Mechanics and Stats: Collision Attacks» (reddit) | не найдено. Суть (атаки столкновения — настоящие атаки с броском атака/защита и уроном оружия, взаимоисключают брызги) подтверждают тема Steam WH2 и форум CA |
| Asset Editor, rmv2_anim_blender, RPFM / rpfm-schemas, гайд по наборам анимаций | существуют, см. п. 3 и 6 |

## Источники

- Realistic Physics: https://steamcommunity.com/sharedfiles/filedetails/?id=3632674226
- Realistic Charge: https://steamcommunity.com/sharedfiles/filedetails/?id=2859816151
- Hecleas Animation Overhaul: https://steamcommunity.com/sharedfiles/filedetails/?id=3357971543
- Better Combat Animations: https://steamcommunity.com/sharedfiles/filedetails/?id=3284296309
- KV_Rules and so will... three? (WH3): https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
- Combat Mechanics: _KV_rules (WH1): https://steamcommunity.com/sharedfiles/filedetails/?id=719168880
- Total war: Warhammer — Some notes: https://steamcommunity.com/sharedfiles/filedetails?id=1114878285
- All the mechanics CA didn't explain: https://steamcommunity.com/app/594570/discussions/0/2999925875999283361/
- Attack interval (форум CA): https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/2773-modifiable-unit-stats-missing-scalar-melee-attack-interval-amp-scalar-missile-bonus-versus-large
- Collision attacks (форум CA): https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/115-collision-attacks-are-an-underutilized-mechanic
- Ice Wyrm animations: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/3415-ice-wyrm-animations-significantly-impair-melee-performance?page=1
- Splash: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
- Мораль Empire (twcenter): https://www.twcenter.net/threads/how-works-the-morale-system-in-empire-total-war.403791/
- On Morale (twcenter): https://www.twcenter.net/threads/on-morale.404713/
- Morale Shock explained (twcenter): https://www.twcenter.net/threads/morale-shock-explained.463698/
- fandom Leadership: https://totalwarwarhammer.fandom.com/wiki/Leadership
- fandom Wolf Rats: https://totalwarwarhammer.fandom.com/wiki/Wolf_Rats
- rpfm-schemas: https://github.com/Frodo45127/rpfm-schemas
- The Asset Editor: https://github.com/donkeyProgramming/TheAssetEditor
- rmv2_anim_blender_import_export: https://github.com/robert-d-schultz/rmv2_anim_blender_import_export
- Animation and You: https://tw-modding.com/wiki/Tutorial:Animation_and_You
- Missiles and You: https://tw-modding.com/wiki/Tutorial:Missiles_and_You
- Creating Custom Unit Animation Sets for WH3: https://steamcommunity.com/sharedfiles/filedetails/?id=3655765373
- WH3 battle_unit (скрипты): https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html
- TWW3 Brawl: https://github.com/tww3-org/tww3-brawl
- Патчи 8.0 / 8.1 / 9.0: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98 , /blogs/101 , /blogs/110-total-war-warhammer-iii-update-9-0-release-notes
