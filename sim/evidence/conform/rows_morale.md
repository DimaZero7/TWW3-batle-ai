# Сверка: мораль, усталость, опыт (06.10.2026)

Источники в ячейках: ключ базы = `config/nn/game_rules.json` (`_kv_morale_tables` → `morale.*`, `_kv_fatigue_tables` →
`fatigue.*`, `_kv_rules` → `battle.*`), таблицы базы, прочитанные сегодня (`build/conform/morale/dump_tables.py`), и URL.
«ЗАПИСИ» = наш новый разбор 217 честных записей (CCO `MoraleGreatestEffect` — подпись самой сильной поправки морали,
каждую секунду): `build/conform/morale/mge_runs.json`, `mge_win.json`, `rout_delay*.json`.

**Три новые находки по записям (не было в аудитах):**
1. **«Натиск» в игре есть** — самая сильная поправка в 3 444 эпизодах (31 тыс. отрядо-секунд), медиана 6 с, 90 % — до 12 с,
   58 % времени ещё до касания. Наш вывод «+15 не видно» (`morale.why`) опровергнут. → L9.
2. **«Одерживают верх» получают стрелки вне рукопашной** — 55 444 отрядо-секунды (это 70 % всех секунд этой подписи).
   Значит, игра считает перевес и по урону от стрельбы, а у нас он есть только в рукопашной. → L4.
3. **«Атакованы с фланга / с тыла» держатся секунды, а не один тик**: медиана 5 / 4 с, 90 % — до 26 / 28 с (в 86–90 % случаев
   это рукопашная). У нас поправка действует 0,5 с. → L16.

## Мораль (L)

| ID | Механика / параметр | Правило игры (формула + числа) | Источник | У нас | Статус | Влияние | Что сделать |
|---|---|---|---|---|---|---|---|
| L1 | Крах армии: пороги | Когда сила врага ≥ 2,6 × своей **и** своя ≤ 0,22 от начальной, все отряды получают −120 (бегут все, кроме несломимых) | `army_destruction_enemy_strength_ratio` 2,6, `_alliance_strength_ratio` 0,22, `ume_concerned_army_destruction` −120; Steam: «примерно 3 : 1 по стоимости» https://steamcommunity.com/app/594570/discussions/0/2999926461247374251/ | morale.py:30-63, 181-182; `collapse.on` = true | MATCHES | высокое: так кончаются бои | — |
| L2 | Крах армии: как считается сила | Сила = стоимость × здоровье, стрелки без снарядов стоят «значительно меньше»; точной формулы нет | Steam (ссылка L1); https://steamcommunity.com/app/1142710/discussions/0/591775773438794807/ (снаряды и ветра магии входят в силу); форум CA https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/5270-balance-of-power-balance-and-bugfix-mod | `collapse.strength` = strategic, `count` = alive, `hp_power` = 1, `lord_value_scale` = 1,65, `routing_weight` = 0,5, `ammo_slope` 14, `ammo_midpoint` 0,25, `ammo_empty` 0,32, `ammo_empty_direct` 0,57 | FITTED | высокое | Числа подобраны по `BalanceOfPowerPercent` (ошибка 0,005) — прикрывают неизвестную формулу игры. Оставить; заменить, только если найдётся `strategic_value` |
| L3 | Сбор бегущих | Таймер «бегство → сбор» 180 с + 10 с × ранг; смысл не раскрыт (моддер: «время до конца бегства»). Нет сбора, пока враг близко (~10–15 м) или под обстрелом | `broken_finish_base_timeout` 180; twcenter https://www.twcenter.net/threads/some-help-for-a-morale-mod-im-making.754432/ (выдержка поиска); Steam https://steamcommunity.com/app/594570/discussions/0/3113646913564430894/ | morale.py:201-225: +`rally_rate` 2 очка/с, пока нет стоящего врага ближе `rally_free_m` 90 м и нет краха; сбор при `rally_mp` 0,23 | FITTED | высокое: повторные бегства 0,40 против 0,07 у игры через 10 с | Правило игры не найдено. 180 с не используем. Вместе с L28 и L31 (`rout_floor_mp`) подбирать как одно правило |
| L4 | Выигрывает / проигрывает схватку | +3 / +6 / +8 и −3 / −8; как считается перевес — в базе нет. **ЗАПИСИ:** «Одерживают верх» у стрелков вне рукопашной 55 444 отрядо-с (против 3 045 в рукопашной) | `winning_combat_slightly/…/_significantly` 3 / 6 / 8, `losing_combat` −3, `_significantly` −8; `build/conform/morale/mge_win.json` | morale.py:116-126, 148-153: только в рукопашной, отношение урона (нанесено+1)/(получено+1) | DIFFERS | высокое: стрелки в игре держат +3…+8, у нас 0 | Считать перевес и по урону от стрельбы (не только в рукопашной); затем заново подобрать `combat_ratio` |
| L5 | Пороги перевеса `combat_ratio` | В базе только очки, порогов нет | ключи L4 | `combat_ratio.slightly` 1,5, `.yes` 2,5, `.significantly` 4 | FITTED | высокое | Подобрать по подписям «Одерживают верх» / «Несут потери» в записях (они есть каждую секунду) вместо сверки морали |
| L6 | Окно перевеса `recent_s` | Не найдено | — | `morale.recent_s` = 30 (затухание урона нанесено/получено; battle.py:216-219) | FITTED | среднее | Подбирать вместе с L5 |
| L7 | Недавние потери | −6 / −12 / −20 / −44 / −80 при 6 / 10 / 15 / 33 / 50 % здоровья за **последние 4 с** | `recent_casualties_penalty_*`; screenrant «последние четыре секунды» https://screenrant.com/total-war-warhammer-3-unit-stats-guide/ | morale.py:142-144; `casualties_window` = sliding, `casualties_s` = 4; `recent_of_current` = false (доля от начального здоровья) | MATCHES | высокое | Основа доли (от начала или от текущего) не найдена — проверить тестом T4 |
| L8 | Потери за бой | −2 … −74 за 10 … 90 % потерянного здоровья (ступени) | `total_casualties_penalty_10…90`, `use_hitpoints_instead_of_casualties_prop…` 1 | morale.py:140-141 (ступени) | MATCHES | высокое | Ступени или плавно — не найдено; оставить ступени |
| L9 | **Натиск: +15 морали** | +15 с начала натиска, пока действует бонус натиска (~10 с); `charge_timeout` 60. **ЗАПИСИ:** «Натиск» — самая сильная поправка, медиана 6 с, 90 % до 12 с, 58 % до касания | `charge_bonus` 15, `charge_timeout` 60; fandom Leadership (выдержка) https://totalwarwarhammer.fandom.com/wiki/Leadership; Steam https://steamcommunity.com/app/594570/discussions/0/1480982338959739400/; `mge_runs.json` | нет (`morale.why`: «+15 не видно») | MISSING | среднее: натиск у нас постоянно (ИИ и сеть) | Добавить +15 с начала натиска до конца бонуса натиска (~10–12 с); заново проверить L5 и `strong_enemy_points` (L19) |
| L10 | Сглаживание морали | Каждый тик мораль идёт к цели на 15 % разрыва, не меньше 1 очка | `percent_update_per_tick` 0,15, `minimium_increment_update_per_tick` 1 | morale.py:194-199, тик 0,5 с (замер) | MATCHES | высокое | — |
| L11 | Начальная мораль | Лидерство карточки + поправки | `land_units.morale` | паспорт; scenario.py:69 начинает с лидерства и сдвигается к цели; `faction_bonus` 0 / 0 | MATCHES | среднее | `faction_bonus` убрать (0 у всех, заменён L25) |
| L12 | Бегство при морали < 0 | Сайты: «ниже 0 в течение 2,5 с». **ЗАПИСИ:** 47 % из 5 780 бегств наступают < 1 с после первой выборки ≤ 0 (2 719 — без единой выборки ≤ 0) — задержки 2,5 с нет | screenrant (ссылка L7); `rout_delay2.json` | morale.py:228-231: сразу при ≤ 0 | MATCHES | среднее | Задержку не вводить |
| L13 | Аура лорда | +4 до 70 м, спад до 0 к 105 м | `general_aura_radius` 70, `general_inspire_effect_amount_max/min` 4, `inspiration_radius_max_effect_range_modifier` 1,5 | battle.py:272-282; `aura_fade` = true | MATCHES | среднее | — |
| L14 | Лорд сам себе аура | Не найдено; замер: лорд стоит на (лидерство + 4 + 5) / лидерство = 1,129 / 1,083 | наш замер `morale.lord_why` | `lord_own_aura` = false | MATCHES | среднее | — |
| L15 | Лорд в рукопашной | Как у отряда (L4). **ЗАПИСИ:** у лордов в рукопашной «Одерживают верх» 2 567 с, «Несут потери» 1 567 с | ключи L4; `mge_win.json` | `single_combat` = "losing": всегда −3 | FITTED | среднее: гибель и бегство лорда | Подписи противоречат «всегда −3»: проверить, не «Ближний бой» ли это (L42); пересмотреть после L4 |
| L16 | Удар во фланг / тыл | −6 / −14 «при первом касании». **ЗАПИСИ:** подпись держится медиану 5 / 4 с, 90 % до 26 / 28 с | `was_attacked_in_flank` −6, `_rear` −14; fandom Flanking (выдержка) https://totalwarwarhammer.fandom.com/wiki/Flanking; `mge_runs.json` | `attacked_event` = true: на один тик 0,5 с при первом ударе с худшей стороны (battle.py:222-227) | DIFFERS | среднее | Измерить по записям, сколько держится подпись после первого удара и сколько очков теряется; длительность сделать параметром |
| L17 | Старые очки флангов | — (подменяли −6 / −14) | — | `attacked_flank` −1, `attacked_rear` −2 (работают только при `attacked_event` = false) | FITTED | нет: выключены | Удалить ключи |
| L18 | Открытые фланги | −3 один, −6 несколько; `open_flanks_effect_range` 120 м | `ume_concerned_flanks_exposed_single` −3, `_multiple` −6; fandom (L9) | morale.py:169-171 по флагам lf/rf/bf | MATCHES | среднее | Очки как в игре (замер −3,9 / −8,1) |
| L19 | Сильный враг рядом | −3 … −24 по «боевой силе» врага 4 … 32, до 70 м; как считается сила — не найдено. ЗАПИСИ: подпись «Рядом враги, превосходящие в силе и скорости» — всего 478 отрядо-с | `enemy_morale_penalty_value_min/max` −3 / −24, `_combat_power_min/max` 4 / 32, `enemy_effect_range` 70 | battle.py:300-301: −3, если в 70 м стоит враг «дороже» (стоимость × здоровье); `strong_enemy_points` −3 | FITTED | среднее: −3 почти всё время в рукопашной | Замер 0 вне рукопашной; −3 оставлен ради пары копейщики — клановые крысы, вероятно, вместо L9 / L4. После L4 и L9 проверить с 0 |
| L20 | Фланги прикрыты | +5, если рядом свои (до 120 м). ЗАПИСИ: самая сильная поправка в 70 тыс. отрядо-с | `ume_encouraged_flanks_secure` 5, `neighbour_effect_range` 120; fandom (L9) | battle.py:290: стоящий свой ближе 120 м | MATCHES | среднее | — |
| L21 | Гибель лорда | −16 на 45 с, потом −10 до конца. ЗАПИСИ: «Лорд недавно погиб» медиана 42 с, до 45 с | `ume_concerned_general_died_recently` −16, `_general_dead` −10; Steam https://steamcommunity.com/app/594570/discussions/0/1679190184074939292/ | `lord_fall.killed` = true, `recent_s` 45 | MATCHES | среднее | — |
| L22 | Лорд ушёл с карты | −16 «недавно бежал», ~120 с | `ume_concerned_general_fled_recently` −16; замер `lord_fall.why` (2 случая) | `lord_fall.fled` = true, `fled_s` 120 | MATCHES | низкое | Уточнить 120 с по подписи «Лорд недавно бежал» (21 эпизод в записях; срез у нас 61 с — пересчитать без среза) |
| L23 | Лорд бежит, но на поле | Только потеря ауры (замер) | замер `lord_fall.why` | `lord_fall.routed` 0 | MATCHES | среднее | — |
| L24 | Лорд-нежить рассыпается | Считается погибшим через ~8 с бегства | замер `lord_fall.why` | `lord_fall.rout_death_s` {vampire_counts: 8} | MATCHES | нет в наших боях | — |
| L25 | «Сила в числе» (скавены) | +6 лидерства, +8 защиты, скорость ×0,9, пока здоровья > 50 % | база: `wh2_main_unit_passive_strength_in_numbers` (`config/nn/effects.json`) | врождённый эффект; `faction_bonus` 0 | MATCHES | среднее | — |
| L26 | Бегущие свои рядом | −3 за каждого (≤ 4), до 100 м **спереди / с фланга** (дальность тыла в базе не задана) | `routing_friends_effect_weighting` 3, `max_routing_friends_to_consider` 4, `routing_unit_effect_distance_front/_flank` 100 | battle.py:294-296: в любом направлении | DIFFERS | среднее | Мелочь: проверить, считаются ли бегущие сзади |
| L27 | Бегущие враги рядом | +2,5 за каждого (≤ 5), до 100 м | `routing_enemies_effect_weighting` 2,5, `max_routing_enemies_to_consider` 5 | battle.py:297 | MATCHES | среднее | — |
| L28 | Расходные | Бегство расходного пугает только расходных | атрибут `expendable`; fandom Attributes https://totalwarwarhammer.fandom.com/wiki/Attributes (из docs) | `expendable_scares_expendable` = true | MATCHES | среднее (рабы) | — |
| L29 | Под обстрелом | −5, держится 15 с после последнего попадания (замер) | `ume_concerned_attacked_by_projectile` −5 | `under_fire_s` 15; battle.py:220-221 (свой огонь не считается: `ume_concerned_under_friendly_fire` 0) | MATCHES | среднее | — |
| L30 | Усталость → мораль | Очень устал −2, измотан −6, устал 0 | `ume_concerned_very_tired` −2, `_exhausted` −6, `_tired` 0 | morale.py:175-176 | MATCHES | среднее | — |
| L31 | Мораль в начале бегства | Не найдено (полоса «бежит» от 0 до −50) | `ums_broken_threshold_upper/lower` 0 / −50 | `rout_floor_mp` −0,3 (мораль опускается до −0,3 × лидерство) | FITTED | среднее: время сбора | Подбирать вместе с L3 |
| L32 | Разгром после 3 бегств | Третье бегство разбивает | `shatter_after_rout_count` 3; 1d6chan https://1d6chan.miraheze.org/wiki/Total_War_Warhammer/Tactics | morale.py:232 | MATCHES | среднее | — |
| L33 | Разгром при обвале морали | Резкое падение ниже −50 разбивает | `ums_broken_threshold_lower` −50; fandom Leadership (выдержка) | `collapse.shatter_below_broken` = true, только при крахе армии | DIFFERS | низкое | Проверить по записям разгромы ниже −50 без краха |
| L34 | Разгром по потерям | Разгром после 1-го / 2-го бегства при потерях > 5 % / 10 % (описание испорчено) | `shatter_after_first/second_rout_if_casulties_higher_than` 0,05 / 0,1 | нет | MISSING | низкое (аудит Opus: в записях не видно) | — |
| L35 | После сбора 10 с без бегства | 10 с | `post_rally_no_rout_timer` 10 | morale.py:228 | MATCHES | среднее | — |
| L36 | Сбор без лорда | Разрешён | `only_units_with_living_commanding_general_may_enter_non_shattered_rout_enabled` 0 | не зависит от лорда | MATCHES | низкое | — |
| L37 | Правило сбора «target» | — (наша выключенная проба) | `morale.rally_why` | `rally_rule` "fixed", `rally_gap_mp` 0,17, `rally_target_min_mp` 0 | FITTED | нет: выключено | Удалить или оставить для L3 |
| L38 | Колебание | Ниже 16 очков | `ums_wavering_threshold_upper` 16; screenrant (L7) | morale.py:242 | MATCHES | среднее | — |
| L39 | Таймер колебания | 25 с «до перехода в бегство или стойкость» | `waver_base_timeout` 25; twcenter (L3) | нет | MISSING | низкое (Opus: пика на 25 с в 5 992 эпизодах нет) | — |
| L40 | Колеблющийся бросает приказ | «Когда отряд колеблется, текущий приказ сбрасывается» | fandom Leadership (выдержка поиска) | колебание приказ не трогает | MISSING | среднее: сеть отдаёт приказы колеблющимся | Проверить в записях: сбрасывается ли приказ при колебании |
| L41 | Состояния для входа сети | Рвётся в бой 0,9–1,1×, уверен 0,65–1,0×, стойкий от 30 очков, потрясён 12–32 | `ums_*_threshold_*` | morale.py:245-249: «рвётся» с 1,0×, без запаса (гистерезиса) | DIFFERS | низкое: только вход сети | «Рвётся» с 0,9× |
| L42 | «Ближний бой» (красная подпись) | Не найдено, что это (медиана 1 с) | `mge_runs.json` | — | MISSING | низкое | Разобрать вместе с L4 |
| L43 | Шок морали | 25 % потерь за 4 с | `recent_casualties_shock_threshold` 25 | нет | MISSING | низкое (редко у пехоты) | — |
| L44 | Скорость бегства | Числа нет; «Удирай!» ×1,1 при колебании и бегстве | база `wh2_main_unit_passive_scurry_away`; замер `morale.why` | `rout_speed` 0,985 × бег с усталостью | MATCHES | среднее | Можно 1,0 (замер 0,98–1,0) |
| L45 | «Держать строй!» | +4 лидерства и +5 защиты своим до 35 м | база `wh_main_lord_passive_hold_the_line` | эффект способности | MATCHES | среднее | — |
| L46 | «Сплотиться!» / «Стоять насмерть!» по бегущим | +16 лидерства, 35 м, 14 с; «бегущих может собрать лорд» | база `wh_main_character_abilities_rally`; fandom https://totalwarwarhammer.fandom.com/wiki/Rally! (выдержка) | +16 идёт в цель, но мораль бегущего цель не слушает (morale.py:204) — бегущих не собирает | DIFFERS | среднее: ИИ применяет «Сплотиться!» | Тест в игре: «Сплотиться!» рядом с бегущим — собирается ли сразу |
| L47 | Несломимые | Никогда не теряют лидерство | атрибут `unbreakable` | morale.py:210 | MATCHES | низкое (флагелланты в пуле) | — |
| L48 | Сложность «Нормальная» | Игроку 0; ИИ: в WH2 −4 / 0 / +4 / +10, в WH3 — множители 0,4 / 0,8 («максимум +8 лидерства»); на нормальной 0 | `difficulty_modifier_player_normal` 0, `_ai_extra_multiplier_low/high` 0,4 / 0,8; `_kv_rules` `difficulty_mod_ai_normal_multiplier` 0,5 (смысл не найден); Steam https://steamcommunity.com/app/1142710/discussions/0/591775773438794807/; https://steamcommunity.com/app/1142710/discussions/0/764058427367407619/ | 0 обеим сторонам | MATCHES | нет | Тест T15: проверить, что ИИ на «Нормальной» стоит так же, как сеть |
| L49 | Страх / ужас | Страх −8 до 20 м; ужас: бегство 14 с при морали ≤ 13 | `ume_concerned_unit_frightened` −8, `fear_effect_range` 20, `terror_effect_range` 5, `morale_shock_*` | нет | MISSING | нет в наших боях | — |
| L50 | На холме | +10 «выше всех врагов» | `ume_encouraged_on_the_hill` 10 | нет (карта плоская) | MISSING | низкое (рельеф карт арены не изучен) | — |
| L51 | Артиллерия | −8 / −10, близкий промах 12 м | `ume_concerned_attacked_by_artillery` −8, `_damaged_by_artillery` −10 | нет | MISSING | нет в наших боях | — |
| L52 | Прочие поправки | Гибель капитана −2, ночь −5, засада −30, паника −50, «Воодушевлён» +30, кони / слоны, укрепления, воодушевление отрядом +4 | ключи `ume_*`, `unit_inspire_effect_amount` 4 | нет | MISSING | нет в наших боях | — |
| L53 | Угроза флангу: геометрия | Как игра ставит флаги `is_*_flank_threatened` — не найдено; база: 120 м | `open_flanks_effect_range` 120 | `threat.calibration`: `on` true, `range_m` 45, `front_deg` 60, `rear_deg` 150, `facing_deg` 60; `threat.range_m` 60 (старое, при `on` false) | FITTED | среднее | Подобрано по флагам игры. В `threat.why` устарело «fatigue remains OFF»; `threat.range_m` 60 удалить |

## Усталость (F)

Таблица штрафов перечитана из `db.pack` сегодня: `unit_fatigue_effects_tables`, 21 строка, совпадает с `fatigue.effects` до цифры.

| ID | Механика / параметр | Правило игры | Источник | У нас | Статус | Влияние | Что сделать |
|---|---|---|---|---|---|---|---|
| F1 | Тик | 10 тиков/с; рукопашная 190 очков/с → измотан за ~142 с | Steam https://steamcommunity.com/app/1142710/discussions/0/3806155895393804028/ | `calibration.on` true, `per_second` 10 | MATCHES | высокое | — |
| F2 | Пороги уровней | 2800 / 6600 / 12600 / 18000 / 27000, максимум 30000 | `threshold_*`; Reddit (из аудита Sol) https://www.reddit.com/r/totalwar/comments/1h6k19w/ | fatigue.py:77-78 | MATCHES | высокое | — |
| F3 | Штрафы уровней | Активный: атака ×0,95; запыхался: скорость 0,95, атака 0,95, бронебойный 0,9; устал: 0,9 / 0,85 / натиск 0,9 / 0,9; очень устал: 0,85 / 0,75 / броня 0,9 / натиск 0,75 / 0,9; измотан: 0,85 / 0,7 / защита 0,9 / броня 0,75 / натиск 0,7 / 0,9 / перезарядка 0,9 | `unit_fatigue_effects_tables` (db.pack 9.0.1); тема моддера https://steamcommunity.com/workshop/filedetails/discussion/3308073985/612032174956240488 | `fatigue.effects`; fatigue.py:87-105 | MATCHES | высокое | Таблица переписана руками: извлекать её `gamedb.py`, чтобы обновлялась с патчем |
| F4 | Рукопашная отряда | +19 за тик каждому в рукопашной | `combat` 19 | только с приказом атаки, 13,7 за тик; без приказа — шаг или отдых | FITTED | высокое | 13,7 прикрывает «не все бойцы бьются» (правило игры не найдено) |
| F5 | Рукопашная лорда | +19; натиск +34 | `combat` 19, `charging` 34 | `single_combat` 15, `single_charge_s` 2 | FITTED | среднее | Подобрано под «устал» через 60–62 с |
| F6 | Натиск отряда | +34 за тик, пока «натиск» | `charging` 34 | `charge_now > 0` (до 13 с после касания) и только с приказом атаки (battle.py:314, fatigue.py:67-70): ~4 400 очков за натиск | DIFFERS | среднее | Неизвестно, длится ли «натиск» усталости весь бонус. ЗАПИСИ: подпись «Натиск» медиана 6 с — проверить рост усталости после касания |
| F7 | Стрельба | +18 | `shooting` 18 | `calibration.shooting` 7,5 | FITTED | среднее | Подобрано по 204 записям |
| F8 | Отдых | Без дела −18, в готовности −7 | `idle` −18, `ready` −7 | −18, если нет стоящего врага ближе `ready_enemy_m` 80 м, иначе −7 | FITTED | высокое | 80 м прикрывает неизвестное правило «боевой стойки» |
| F9 | Бег | +4 (пехота) | `running` 4 | по флагу бега приказа (`move` "order") | MATCHES | среднее | — |
| F10 | Шаг | −1 | `walking` −1 | −1 | MATCHES | среднее | Устарело в docs README (строка 7: «шаг 3,4») |
| F11 | Бегущие | Бег +4 | `running` 4 | `routing` "running" (замер 5,1) | MATCHES | низкое | — |
| F12 | Перезарядка измотанных | `stat_reloading` ×0,9 — множитель навыка перезарядки (у наших `reload` 0?) | `unit_fatigue_effects_tables` | время перезарядки ÷ 0,9 | DIFFERS | низкое | Не найдено, как навык перезарядки меняет время; тест T13 |
| F13 | Начало боя | Свежие (замер: 3 814 чтений — «свежий») | замер docs/ru/game/mechanics/fatigue.md | 0 очков | MATCHES | низкое | — |
| F14 | Старая модель | — | — | `fatigue.per_second` 5 (при `calibration.on` false) | FITTED | нет: выключена | Удалить вместе с веткой кода |
| F15 | Безупречная бодрость | Не устаёт | атрибут `fatigue_immune`; Steam https://steamcommunity.com/app/1142710/discussions/0/3842178984949278930/ | fatigue.py:72-73 | MATCHES | нет в наших боях | — |
| F16 | Подъём в гору | Стоимость движения +50 / +100 / +150 % на уклонах | `gradient_*_movement_multiplier`; CA https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation | нет | MISSING | низкое | — |
| F17 | Дождь, снег, конница, артиллерия, лестницы, работа | −10 / +6 / +12 / +10000 / +20 | ключи `fatigue.*` | нет | MISSING | нет в наших боях | — |

## Опыт (X)

В наших боях опыт = 0 у всех: XML арены `<unit_experience level="0"/>`, лорды `CharacterRank` 1, `ExperienceLevel` 0
(`build/lord-ai`). Симулятор опыта не знает — для ранга 0 это верно.

| ID | Механика / параметр | Правило игры | Источник | У нас | Статус | Влияние | Что сделать |
|---|---|---|---|---|---|---|---|
| X1 | Лидерство за ранг | +1,06 × ранг с округлением → +1 … +8, ранг 9: +10 | `unit_experience_bonuses_tables` `stat_morale` (0; 0; 1,06); fandom https://totalwarwarhammer.fandom.com/wiki/Experience_and_rank (выдержка); Steam https://steamcommunity.com/app/1142710/discussions/0/3806157163938255596/ | нет | MISSING | нет в наших боях | — |
| X2 | Атака / защита в рукопашной | Ранг 9: +9 / +8 (таблица fandom: +1…+9 / +1…+8); в базе a 0,6, b 0,12 — формула не найдена | `stat_melee_attack/defence` (0; 0,6; 0,12); ссылки X1 | нет | MISSING | нет в наших боях | — |
| X3 | Меткость | b = 3 за ранг (?) | `stat_accuracy` (0; 0; 3) | нет | MISSING | нет в наших боях | — |
| X4 | Перезарядка | флаг 1, b = 2; fandom: −0,2 … за ранг | `stat_reloading` (1; 0; 2); fandom X1 | нет | MISSING | нет в наших боях | — |
| X5 | Сбор быстрее с рангом | +10 с к таймеру сбора за ранг | `broken_finish_timer_experience_bonus` 10 | нет | MISSING | нет в наших боях | — |
| X6 | Опыт за бой | 700 … 22 000 очков на ранги 1–9 (кампания) | `unit_experience_thresholds_tables` | нет | MISSING | нет | — |

## Полнота: каждый ключ sim.json → строка

`morale.`: faction_bonus L11 · recent_s L6 · casualties_s, casualties_window, recent_of_current L7 · extended_s L8 ·
under_fire_s L29 · expendable_scares_expendable L28 · strong_enemy_points L19 · lord_own_aura L14 · single_combat L15 ·
attacked_event L16 · aura_fade L13 · collapse.{on, count, hp_power, strength, lord_value_scale, routing_weight, ammo_empty,
ammo_empty_direct, ammo_slope, ammo_midpoint} L2 · collapse.shatter_below_broken L33 · combat_ratio.* L5 · rally_rule,
rally_gap_mp, rally_target_min_mp L37 · rally_mp, rally_rate, rally_free_m L3 · rout_floor_mp L31 · rout_speed L44 ·
attacked_flank, attacked_rear L17 · lord_fall.{killed, recent_s} L21 · lord_fall.{fled, fled_s} L22 · lord_fall.routed L23 ·
lord_fall.rout_death_s L24. `fatigue.`: per_second F14 · calibration.{on, per_second} F1 · calibration.melee,
multi_combat F4 · calibration.{single_combat, single_charge_s} F5 · calibration.move F9 · calibration.shooting F7 ·
calibration.ready_enemy_m F8 · calibration.routing F11 · effects.* F3 (+F12). `threat.`: range_m, calibration.* L53.
Ключи-пояснения (`why`, `lord_why`, `windows_why`, `expendable_why`, `strong_enemy_why`, `rally_why`, `collapse.why`,
`lord_fall.why`, `calibration.why`) — текст, не параметры.

Итого 76 строк: MATCHES 35, FITTED 15, DIFFERS 8, MISSING 18 (мораль 27 / 10 / 6 / 10, усталость 8 / 5 / 2 / 2,
опыт 0 / 0 / 0 / 6). Из MISSING 10 — «нет в наших боях» (X1–X6, L49, L51, L52, F17).

## Тесты на уровне формулы

| # | Что задать | Что должно получиться |
|---|---|---|
| T1 | Отряд, лидерство 60, цель морали падает на 16 очков | через 1 / 2 / 5 / 10 с: −4,4 / −7,6 / −12,9 / −15,4 |
| T2 | Лорд и свой отряд на 70 / 87,5 / 105 м | аура +4 / +2 / 0; сам лорд — 0 |
| T3 | Отряд далеко от врагов, лорд убит | цель −16 на 45 с, потом −10 до конца |
| T4 | Отряд теряет 10 % здоровья одним залпом | недавние −12 ровно 4 с; длительные −4 на 60 с; за бой −2 навсегда (основа доли — тест в игре: тот же залп по отряду с 50 % здоровья) |
| T5 | Последнее попадание снаряда в t = 0 | −5 до t = 15 с, с 15 с — 0 |
| T6 | Первый удар во фланг / в тыл | сейчас: −6 / −14 на 0,5 с (≈ −1 / −2,1 очка). Игра (ЗАПИСИ): подпись держится 4–5 с — сравнить падение морали за 5 с |
| T7 | 5 бегущих своих в 100 м; бегущий расходный рядом с обычным и с расходным | −12 (предел 4); обычному 0, расходному −3 |
| T8 | Натиск в игре: 1 отряд копейщиков атакует с разбега стоящий отряд | подпись «Натиск» с начала натиска, ~10–12 с; мораль выше на ~+15 по сравнению с той же атакой шагом |
| T9 | Лучники стреляют по отряду, сами не под огнём | игра: «Одерживают верх» (+3…+8); симулятор сейчас 0 |
| T10 | Своя сила 0,21 от начала, враг / своя 2,7; затем 0,23 | −120 всем; при 0,23 — ничего |
| T11 | Бегство при лидерстве 60, врагов ближе 90 м нет | мораль с −18, +2 очка/с, сбор при 13,8 через ~16 с; третье бегство — разгром |
| T12 | Отряд в рукопашной с приказом атаки, без натиска | 137 очков/с: «активный» через 20,4 с, «измотан» через 197 с; одиночка 150/с — 180 с; правило игры +19 — 142 с |
| T13 | Измотанные лучники | атака ×0,7, защита ×0,9, броня ×0,75, скорость ×0,85, мораль −6; перезарядка — замерить в игре (×1/0,9 или без изменения) |
| T14 | Скавены при 51 % и 49 % здоровья | +6 лидерства / 0; скорость ×0,9 / ×1 |
| T15 | Спокойные отряды ИИ и сети одного типа на «Нормальной» | одинаковая `MoralePercent` (нет бонуса сложности) |
| T16 | «Сплотиться!» рядом с бегущим отрядом (игра) | собирается ли сразу; у нас — нет |

## Что не нашлось в сети

- Как игра считает «выигрывает / проигрывает» схватку (отношение чего и за какое окно) — только очки.
- Что такое «боевая сила» для «сильного врага рядом» (4…32 → −3…−24).
- Смысл `broken_finish_base_timeout` 180 с в WH3 и правило сбора (скорость, условия).
- Длительность «атакован во фланг / в тыл» и точная длительность морали натиска в WH3 (только «~10 с» у fandom).
- Формула бонуса ранга для атаки / защиты (a 0,6, b 0,12), меткости и перезарядки; смысл флага 1 у перезарядки.
- Что делают `difficulty_mod_ai_normal_multiplier` 0,5 и `difficulty_modifier_ai_extra_multiplier_low/high` 0,4 / 0,8.
- Геометрия флагов «фланг под угрозой» в игре; считаются ли бегущие сзади.
- Когда идёт «натиск» для усталости (+34): только разбег или весь бонус.
- Ступени или плавный переход в таблицах потерь.

Недоступно: fandom (402 — только выдержки поиска), twcenter (403), Steam часть времени 429.

## Устаревшее в документах (на заметку, не правил)

- `docs/ru/game/mechanics/README.md`, таблица расхождений: строка 5 («30 с подгонка» — теперь 4 с), 6 («не моделируется» —
  теперь 60 с), 7 («шаг 3,4» — теперь −1).
- `docs/ru/game/mechanics/morale.md`: открытые фланги «60 м» — теперь 45 м; «натиск не моделируется. Пробел» — верно,
  но обоснование «+15 не видно» опровергнуто подписями (L9).
- `sim.json` `threat.why`: «fatigue remains OFF» — калибровка усталости включена.
