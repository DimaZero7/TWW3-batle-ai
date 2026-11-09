# Новые отряды: карточки из базы игры

Источник: база игры v9.0.1 (`data/db.pack`), паспорта собраны штатным `py -3.14 -m tools.nn.units --units … --out build/newunits/units_new.json`
(те же таблицы, что у наших 17 отрядов). Названия — русская локализация игры (`local_ru.pack`, `land_units_onscreen_name_*`,
выгрузка `build/newunits/loc_ru.json`). Карточек из боя (`data/roster`) ни у одного из них нет — сверки с карточкой нет.

## Какие ключи

| Просили | Ключ | Название в игре | Статус |
|---|---|---|---|
| Пистольеры | `wh_main_emp_inf_handgunners` | «Пистольеры» (600, 90 бойцов) | **новый**; это аркебузиры (Handgunners), не конница. Конница `wh_main_emp_cav_pistoliers_1` в игре — «Пистолетчики», не берём |
| Арбалетчики | `wh_main_emp_inf_crossbowmen` | «Арбалетчики» | новый |
| Клановые крысы (мечи, без щита) | `wh2_main_skv_inf_clanrats_0` | «Кланокрысы» | новый |
| Клановые крысы с копьями (щиты) | `wh2_main_skv_inf_clanrat_spearmen_1` | «Кланокрысы с копьями (щиты)» | новый |
| Штормокрысы (алебарды) | `wh2_main_skv_inf_stormvermin_0` | «Штормокрысы (алебарды)» | **уже есть**: паспорт в `config/nn/units.json` и строка «Штрм» реестра (из пробы толпы на лорда); нет только в пуле `pools.json` |
| Штормокрысы (мечи и щиты) | `wh2_main_skv_inf_stormvermin_1` | «Штормокрысы (мечи и щиты)» | новый |
| Бегущие ночью | `wh2_main_skv_inf_night_runners_0` | «Бегущие ночью» — **метательные звёзды** | новый (недостающий вариант) |
| Бегущие ночью (пращи) | `wh2_main_skv_inf_night_runners_1` | «Бегущие ночью (пращи)» | **уже есть** (паспорт, пул, реестр «Бег») |
| Бегущие по канавам | `wh2_main_skv_inf_gutter_runners_0` | «Бегущие по канавам» (звёзды, без яда) | **отложен** (Stalk) |
| Бегущие по канавам с пращами | `wh2_main_skv_inf_gutter_runner_slingers_0` | «Бегущие по канавам с пращами» (без яда) | **отложен** (Stalk) |

Яд есть только у вариантов `_1` («(яд)»: `gutter_runners_1`, `gutter_runner_slingers_1`, 750) — у простых `_0` яда нет.

## Карточки

| Показатель | Пистольеры | Арбалетчики | Кланокрысы | Кланокрысы с копьями (щиты) | Штормокрысы (алебарды) | Штормокрысы (мечи и щиты) | Бегущие ночью | Бегущие ночью (пращи) | Бегущие по канавам | Бегущие по канавам с пращами |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ключ | `wh_main_emp_inf_handgunners` | `wh_main_emp_inf_crossbowmen` | `wh2_main_skv_inf_clanrats_0` | `wh2_main_skv_inf_clanrat_spearmen_1` | `wh2_main_skv_inf_stormvermin_0` | `wh2_main_skv_inf_stormvermin_1` | `wh2_main_skv_inf_night_runners_0` | `wh2_main_skv_inf_night_runners_1` | `wh2_main_skv_inf_gutter_runners_0` | `wh2_main_skv_inf_gutter_runner_slingers_0` |
| Каста / категория | missile_infantry / inf_ranged | missile_infantry / inf_ranged | melee_infantry / inf_melee | melee_infantry / inf_melee | melee_infantry / inf_melee | melee_infantry / inf_melee | missile_infantry / inf_ranged | missile_infantry / inf_ranged | missile_infantry / inf_ranged | missile_infantry / inf_ranged |
| Цена (сетевая) | 600 | 475 | 300 | 375 | 950 | 850 | 450 | 450 | 700 | 700 |
| Бойцов | 90 | 90 | 160 | 160 | 160 | 160 | 120 | 120 | 120 | 120 |
| HP бойца / отряда | 69 / 6210 | 69 / 6210 | 60 / 9600 | 60 / 9600 | 62 / 9920 | 62 / 9920 | 56 / 6720 | 56 / 6720 | 58 / 6960 | 58 / 6960 |
| Броня (материал) | 20 (leather) | 20 (leather) | 25 (leather) | 25 (leather) | 90 (heavy_metal) | 90 (heavy_metal) | 10 (body) | 10 (body) | 20 (body) | 20 (body) |
| Щит (блок снарядов) | нет | нет | нет | 35 % | нет | 35 % | нет | нет | нет | нет |
| Атака / защита | 16 / 17 | 14 / 17 | 24 / 18 | 18 / 28 | 32 / 34 | 38 / 36 | 24 / 14 | 24 / 14 | 26 / 20 | 26 / 20 |
| Натиск | 4 | 4 | 10 | 6 | 10 | 18 | 12 | 10 | 14 | 10 |
| Оружие: урон + бронебойный | wh_main_emp_sword_short: 21 + 3 | wh_main_emp_sword_short: 21 + 3 | wh2_main_skv_sword: 20 + 6 | wh2_main_skv_spear: 18 + 5 | wh2_main_skv_halberd: 9 + 23 | wh2_main_skv_sword_stormvermin: 25 + 9 | wh2_main_skv_sword: 20 + 6 | wh2_main_skv_sword: 20 + 6 | wh2_main_skv_claw_blade: 21 + 7 | wh2_main_skv_claw_blade: 21 + 7 |
| Бонус против больших / пехоты | 0 / 0 | 0 / 0 | 0 / 0 | 9 / 0 | 17 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Интервал удара, с | 4.3 | 4.3 | 4.3 | 4.3 | 4.6 | 4.6 | 4.3 | 4.3 | 4.3 | 4.3 |
| Лидерство | 50 | 50 | 45 | 45 | 70 | 70 | 48 | 48 | 54 | 54 |
| Физ. стойкость | — | — | — | — | — | — | 20 % | 20 % | 20 % | 20 % |
| Масса; размер | 90.0; small | 90.0; small | 100.0; small | 100.0; small | 150.0; small | 150.0; small | 90.0; small | 90.0; small | 90.0; small | 90.0; small |
| Шаг / бег / натиск, м/с | 1.5 / 3.3 / 4.0 | 1.5 / 3.3 / 4.0 | 1.5 / 4.2 / 4.8 | 1.5 / 4.2 / 4.8 | 1.5 / 3.8 / 4.5 | 1.5 / 3.8 / 4.5 | 1.5 / 5.4 / 6.0 | 1.5 / 5.4 / 6.0 | 1.5 / 5.4 / 6.0 | 1.5 / 5.4 / 6.0 |
| Разгон / торм.; поворот °/с | 3.5 / 5.0; 180.0 | 3.5 / 5.0; 180.0 | 2.0 / 4.0; 120.0 | 2.0 / 4.0; 120.0 | 2.0 / 5.0; 120.0 | 2.0 / 5.0; 120.0 | 5.0 / 6.0; 120.0 | 5.0 / 6.0; 120.0 | 5.0 / 6.0; 120.0 | 5.0 / 6.0; 120.0 |
| Шеренг; шаг строя h×v (шаблон) | 3; 1.4×1.7 (`wh_main_infantry_ranged`) | 4; 1.4×1.7 (`wh_main_infantry_ranged`) | 8; 1.6×1.7 (`wh_main_infantry_rabble`) | 8; 1.6×1.7 (`wh_main_infantry_rabble`) | 8; 1.8×1.9 (`wh2_main_infantry_medium_rabble`) | 8; 1.8×1.9 (`wh2_main_infantry_medium_rabble`) | 5; 1.9×1.9 (`wh2_main_inf_chaotic_formation`, рыхлый) | 5; 2.2×2.8 (`wh2_main_inf_chaotic_formation_loose`, рыхлый) | 5; 1.9×1.9 (`wh2_main_inf_chaotic_formation`, рыхлый) | 5; 2.2×2.8 (`wh2_main_inf_chaotic_formation_loose`, рыхлый) |
| Застрельщики / упор | да / да | да / да | нет / да | нет / да | нет / да | нет / да | да / да | да / да | да / да | да / да |
| Скрытность (hiding_scalar) | 1.0 | 1.0 | 0.8 | 0.8 | 1.0 | 1.0 | 0.9 | 0.9 | 0.8 | 0.8 |
| Атрибуты | hide_forest | hide_forest | hide_forest | charge_defense_vs_large, charge_reflection, hide_forest | charge_defense_vs_large, charge_reflection, hide_forest | hide_forest | guerrilla_deploy, hide_forest, mounted_fire_move | guerrilla_deploy, hide_forest | guerrilla_deploy, mounted_fire_move, stalk | guerrilla_deploy, stalk |
| Способности | — | — | scurry_away, strength_in_numbers | scurry_away, strength_in_numbers | scurry_away, strength_in_numbers | scurry_away, strength_in_numbers | scurry_away, strength_in_numbers | scurry_away, strength_in_numbers | scurry_away, snare_net, strength_in_numbers | scurry_away, snare_net, strength_in_numbers |
| Боевой потенциал ближ. / дальн. | 120 / 480 | 100 / 400 | 300 / 0 | 375 / 0 | 1050 / 0 | 950 / 0 | 150 / 350 | 150 / 350 | 200 / 500 | 200 / 500 |
| Снаряд (категория) | `wh_emp_rifle_bullet` (musket) | `wh_emp_crossbow_bolt` (arrow) | — | — | — | — | `wh2_main_skv_throwing_star` (musket) | `wh2_main_skv_sling_night_runners` (sling) | `wh2_main_skv_throwing_star_gutter_runners` (musket) | `wh2_main_skv_sling_gutter_runners` (sling) |
| Траектория | low — настильно, нужна чистая линия | dual_low_fixed — навесом поверх своих | — | — | — | — | low — настильно, нужна чистая линия | dual_low_fixed — навесом поверх своих | low — настильно, нужна чистая линия | dual_low_fixed — навесом поверх своих |
| Дальность, м | 145 | 160 | — | — | — | — | 70 | 140 | 70 | 140 |
| Боезапас на бойца | 22 | 22 | — | — | — | — | 10 | 22 | 12 | 24 |
| Перезарядка (база), с | 13.0 | 13.0 | — | — | — | — | 7.0 | 8.0 | 7.0 | 8.0 |
| Урон + бронебойный | 5 + 17 | 18 + 6 | — | — | — | — | 10 + 2 | 10 + 1 | 11 + 3 | 12 + 2 |
| Снарядов за выстрел / за залп | 1 / 1 | 1 / 1 | — | — | — | — | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Меткость отряда + снаряда | 10 + 10 | 10 + 10 | — | — | — | — | 10 + 10 | 10 + 15 | 15 + 10 | 15 + 15 |
| Калибровка: м² на м | 1.8 на 100 | 5.1 на 120 | — | — | — | — | 1.8 на 40 | 4.0 на 85 | 1.8 на 50 | 3.8 на 85 |
| Скорость снаряда, м/с; макс. угол | 100; 88° | 45; 88° | — | — | — | — | 48; 88° | 50; 88° | 48; 88° | 50; 88° |
| Дуга огня (каждая сторона) | ±30° | ±30° | — | — | — | — | ±180° | ±30° | ±180° | ±30° |
| Пробивание; взрыв; мин. дальность | very_low; нет; 0 | very_low; нет; 0 | — | — | — | — | very_low; нет; 0 | very_low; нет; 0 | very_low; нет; 0 | very_low; нет; 0 |

Пояснения:

- **Аркебузиры («Пистольеры»)**: бронебойный выстрел 5 + **17** (у ополчения 12 + 2), дальность 145, перезарядка 13, 3 шеренги,
  стреляют только стоя (нет `mounted_fire_move`), настильно (`low`) — нужна чистая линия мимо своих. Броня 20 (кожа), как у
  лучников. В рукопашной 16 / 17 — слабы. Оружие в базе `wh_main_emp_rifle` (вид анимации `repeater_gun`).
- **Арбалетчики**: стрела (`arrow`), навесом, дальность **160** (длиннее всех наших), 18 + 6, перезарядка 13, 22 выстрела.
- **Бегущие ночью (звёзды)**: категория снаряда `musket` (как пистоль), настильно, 70 м, перезарядка 7, 10 выстрелов, дуга
  огня **360°** (сущность `wh2_main_skv_infantry_fast_360`: fire_arc_close 360), **стрельба на ходу**. «Сила снаряда» 17 на
  карточке игры (honga.net / fandom «Night Runners») = (10 + 2) × 10 / 7 — сходится с базой.
- **Пистолетчики (конница, не берём)**: для справки — штатный сборщик паспорта строит конницу неверно: здоровье, масса и
  скорость берутся с всадника, а надо с лошади (`wh_main_cavalry_fast_blood`: бег 9,0, натиск 11,3, масса 500); 2 выстрела в залп.

## Полный слепок навыков и атрибутов

| Отряд | Атрибуты (`unit_attributes_to_groups_junctions`) | Способности (`land_units_to_unit_abilites_junctions`) | Оружие: бонусы, фазы удара | Снаряд: особенности |
|---|---|---|---|---|
| Аркебузиры | hide_forest | — | меч `wh_main_emp_sword_short` 21+3, без бонусов, без фазы | `wh_emp_rifle_bullet`: musket, low, 145 м, 5+17, 13 с, 1 снаряд, пробивание very_low, без взрыва и эффекта касания |
| Арбалетчики | hide_forest | — | тот же меч | `wh_emp_crossbow_bolt`: arrow, dual_low_fixed, 160 м, 18+6, 13 с, very_low, без эффекта |
| Кланокрысы | hide_forest | Strength in Numbers, Scurry Away! | `wh2_main_skv_sword` 20+6 | — |
| Кланокрысы с копьями (щиты) | charge_defense_vs_large, charge_reflection, hide_forest | те же | `wh2_main_skv_spear` 18+5, против больших +9 | — |
| Штормокрысы (алебарды) | charge_defense_vs_large, charge_reflection, hide_forest | те же | `wh2_main_skv_halberd` 9+23, против больших +17 | — |
| Штормокрысы (мечи и щиты) | hide_forest | те же | `wh2_main_skv_sword_stormvermin` 25+9 | — |
| Бегущие ночью (звёзды) | guerrilla_deploy, hide_forest, mounted_fire_move | те же | `wh2_main_skv_sword` 20+6 | `wh2_main_skv_throwing_star`: musket, low, 70 м, 10+2, 7 с, very_low, без эффекта; дуга 360° |
| Бегущие ночью (пращи) | guerrilla_deploy, hide_forest | те же | меч 20+6 | праща: sling, dual_low_fixed, 140 м, 10+1, 8 с |
| *Бегущие по канавам (отложен)* | guerrilla_deploy, mounted_fire_move, **stalk** | те же + **Snare-Net** | `wh2_main_skv_claw_blade` 21+7 (у `_1` — фаза яда) | звёзды 70 м, 11+3, дуга 360° |
| *Бегущие по канавам с пращами (отложен)* | guerrilla_deploy, **stalk** | те же + **Snare-Net** | claw blade 21+7 | праща 140 м, 12+2 |

Способности (паспорта `config/nn/abilities.json`; Snare-Net — `build/newunits/abilities_new.json`): Strength in Numbers — пока
здоровье ≥ 50 % начального: защита +8, лидерство +6, скорость ×0,9; Scurry Away! — колеблется или бежит: скорость ×1,1;
Snare-Net (только Бегущие по канавам) — пассивная аура на **врагов** в 35 м: скорость и скорость натиска ×0,9 (база; fandom
«Snare-Net»: −10 % / −10 %, 35 м). Яд (`wh_main_unit_contact_poison`, только `_1`): скорость, урон оружия и снарядов ×0,85
(база; fandom «Poison!» −15 %); время действия в нашей расшифровке базы не найдено.

Скрытность по базе: **Stalk** — только у Бегущих по канавам (оба). У Бегущих ночью (оба) Stalk нет: `hide_forest` (прятаться
только в лесу) и `hiding_scalar` 0,9 — `hide_forest` есть у всех 17 наших отрядов, на ровной карте без леса не действует.
Отложить нужно только Бегущих по канавам.

## Шаблоны армий игры (генератор `cdir_military_generator_*`)

Группы из `cdir_military_generator_unit_qualities` (`build/newunits/army_templates_merged.json`):

| Отряд | Группа генератора | Ступень | Семейство |
|---|---|---|---|
| Аркебузиры | `..._ranged_infantry_main_light_backline` | tier_2_step_2 | ranged_infantry |
| Арбалетчики | `..._ranged_infantry_main_light_backline` | tier_1_step_5 | ranged_infantry |
| Кланокрысы | `..._melee_infantry_main_frontline_swords` | tier_1_step_4 | melee_infantry_trash |
| Кланокрысы с копьями (щиты) | `..._melee_infantry_main_frontline_spears` | tier_1_step_4 | melee_infantry_trash |
| Штормокрысы (алебарды) | `..._melee_infantry_main_frontline_spears_high_quality` | tier_3_step_1 | melee_infantry_trash |
| Штормокрысы (мечи и щиты) | `..._melee_infantry_main_frontline_swords_high_quality` | tier_2_step_5 | melee_infantry_trash |
| Бегущие ночью (звёзды) | `..._ranged_infantry_main_light_skirmisher` | tier_1_step_5 | ranged_infantry |
| *Бегущие по канавам* | `..._ranged_infantry_main_light_skirmisher` | tier_2_step_3 | ranged_infantry |
| *… с пращами* | `..._ranged_infantry_main_light_backline` | tier_2_step_3 | ranged_infantry |

Все встречаются в шаблонах своей фракции (через семейство). Доли в наших шаблонных армиях с новым набором (правило
`pools.family_shares`: доля шаблона по семейству, поровну на отряды семейства; расчёт `build/newunits/templ.py`):

| Шаблон | Каждый из 5 ближнего боя Империи | Каждый из 4 стрелков Империи (луки, ополчение, арбалеты, аркебузы) |
|---|---:|---:|
| WH_Empire_land / _2 / _3 / _4 | 12 / 13 / 12 / 11 % | 10 / 8 / 10 / 11 % |

| Шаблон | Каждый из 8 ближнего боя скавенов | Каждый из 3 стрелков скавенов (рабы-пращники, бегущие ночью × 2) |
|---|---:|---:|
| WH_Skaven_land / _2 / _3 / _4 / _5 | 8 / 9 / 8 / 10 / 8 % | 13 / 10 / 11 / 8 / 13 % |

Замечание (предложение, не делалось): шаблоны игры дают штормокрысам свои группы «high_quality» (у скавенов по 10–15 %), а
наше правило сливает всё ближнее в одно семейство и делит поровну — штормокрыса (950) получает ту же долю, что раб (125);
дорогое против дешёвого решает бюджет. Потолки бюджета `budget_max` (скавены 6700, Империя 7725) остаются.
