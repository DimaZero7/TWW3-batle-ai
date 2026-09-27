# Каталог показателей

[Знания об игре](README.md) · [English](../../en/game/readouts.md)

Всё, что можно прочитать об отрядах и бое: и то, что мы уже собираем, и то, что есть в игре, но ещё не проверено. Для каждого прогона выбираем профиль — список того, что собирать ([config/readouts](../../../config/readouts/)). Страница создана из [data/readouts/catalog.json](../../../data/readouts/catalog.json) командой `python -m tools.readouts docs`; правьте JSON, а не эту страницу.

**Статус:** ✅ собираем и проверено в игре · ⬜ есть в API игры, не проверено. **Доступ для ИИ:** только свои · свои и видимые враги · общее для всех · только полная сводка (никогда не вход ИИ).

Всего 139: собираем 89, ещё не проверено 50.

## Что это за отряд

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `unit.type` | Тип отряда (ключ из базы игры) | ✅ собираем | свои и видимые враги |
| `unit.name` | Имя отряда из сценария | ✅ собираем | свои и видимые враги |
| `unit.id` | Уникальный номер отряда в бою | ✅ собираем | только полная сводка |
| `unit.commanding` | Это генерал армии | ✅ собираем | свои и видимые враги |
| `unit.class` | Класс отряда (пехота, кавалерия…) | ✅ собираем | только свои |
| `unit.kind_flags` | Вид: пехота, конница, копейщики против конницы, колесница, звери, слоны, артиллерия, боевая машина | ✅ собираем | только свои |
| `unit.is_character` | Герой или генерал (персонаж) | ⬜ есть в игре | только свои |
| `unit.renown` | Полк славы (особый отряд) | ⬜ есть в игре | только свои |
| `unit.spawned` | Появился во время боя (призван способностью) | ⬜ есть в игре | только свои |
| `unit.attributes` | Особые свойства: прячется в лесу, прячется везде, защита от натиска, отражение натиска, воодушевляет | ✅ собираем | только свои |
| `unit.behaviours_available` | Какие режимы доступны: оборона, перестрелка, свободный огонь, плотность строя | ✅ собираем | только свои |
| `unit.card_stats` | Все характеристики с карточки: атака, защита, броня, натиск, урон и т. д. | ✅ собираем | только свои |
| `unit.card_morale` | Дисциплина (мораль) с карточки | ✅ собираем | только свои |
| `unit.mass` | Масса отряда (сила удара при натиске) | ✅ собираем | только свои |
| `unit.experience_level` | Уровень опыта | ✅ собираем | только свои |
| `unit.experience_progress` | Сколько осталось до следующего уровня опыта | ⬜ есть в игре | только свои |
| `unit.character_rank` | Уровень героя | ✅ собираем | только свои |

## Численность и сила

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.number_of_men_alive` | Сколько бойцов живо | ✅ собираем | только свои |
| `native.initial_number_of_men` | Сколько бойцов было в начале | ✅ собираем | только свои |
| `native.unary_of_men_alive` | Доля живых бойцов | ✅ собираем | только свои |
| `cco.NumEntities` | Сколько бойцов (по интерфейсу) | ✅ собираем | только свои |
| `cco.NumEntitiesInitial` | Сколько бойцов было (по интерфейсу) | ✅ собираем | только свои |
| `unit.strategic_value` | Оценка силы отряда от игры (одно число) | ⬜ есть в игре | только свои |
| `cco.PercentCasualtiesRecently` | Доля бойцов, погибших за последние 4 секунды | ⬜ есть в игре | только свои |

## Здоровье и щиты

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.unary_hitpoints` | Доля здоровья от начального | ✅ собираем | только свои |
| `cco.HealthValue` | Здоровье (число) | ✅ собираем | только свои |
| `cco.HealthMax` | Максимум здоровья | ✅ собираем | только свои |
| `cco.HealthPercent` | Здоровье в процентах (как в интерфейсе) | ✅ собираем | только свои |
| `cco.PercentHpLostRecently` | Сколько здоровья потеряно за 4 секунды | ✅ собираем | только свои |
| `cco.IsTakingDamage` | Получает урон прямо сейчас | ✅ собираем | только свои |
| `health.delayed` | «Отложенное» здоровье (полоска интерфейса) | ⬜ есть в игре | только свои |
| `health.barrier` | Магический щит: сколько осталось, заряжается ли, когда восстановится | ⬜ есть в игре | только свои |
| `health.healing` | Лечение: сила и сколько можно восстановить | ⬜ есть в игре | только свои |
| `health.wounded` | Ранен (особое состояние) | ⬜ есть в игре | только свои |
| `health.invulnerable` | Неуязвим | ⬜ есть в игре | только свои |

## Мораль

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.is_wavering` | Колеблется (на грани бегства) | ✅ собираем | только свои |
| `native.is_routing` | Бежит | ✅ собираем | только свои |
| `native.is_shattered` | Рассеян (уже не вернётся) | ✅ собираем | только свои |
| `cco.MoralePercent` | Мораль в процентах | ✅ собираем | только свои |
| `cco.MoraleState` | Состояние морали (число) | ✅ собираем | только свои |
| `cco.MoraleName` | Состояние морали (текст) | ✅ собираем | только свои |
| `cco.MoraleGreatestEffect` | Что сильнее всего влияет на мораль | ✅ собираем | только свои |
| `cco.IsRouting` | Бежит (по интерфейсу) | ✅ собираем | только свои |
| `cco.IsShattered` | Рассеян (по интерфейсу) | ✅ собираем | только свои |
| `cco.IsWavering` | Колеблется (по интерфейсу) | ✅ собираем | только свои |
| `cco.IsInLastStand` | Последний рубеж (некуда бежать) | ✅ собираем | только свои |
| `cco.IsOutOfControl` | Вне контроля | ✅ собираем | только свои |
| `cco.IsAwaitingOrderAfterRally` | Вернулся после бегства и ждёт приказа | ✅ собираем | только свои |
| `morale.terrified` | В ужасе от страшного врага | ⬜ есть в игре | только свои |
| `morale.rampaging` | В неистовстве (не слушается, лезет в драку) | ⬜ есть в игре | только свои |
| `morale.undead` | Нежить/демоны: рассыпается, нестабилен | ⬜ есть в игре | только свои |

## Усталость

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.fatigue_state` | Усталость (состояние) | ✅ собираем | только свои |
| `cco.FatigueState` | Усталость (число) | ✅ собираем | только свои |
| `cco.FatigueName` | Усталость (текст) | ✅ собираем | только свои |

## Движение и положение

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.position` | Где стоит отряд | ✅ собираем | свои и видимые враги |
| `native.ordered_position` | Куда отряду приказано идти | ✅ собираем | только свои |
| `native.bearing` | Куда повёрнут отряд | ✅ собираем | только свои |
| `native.ordered_bearing` | Куда приказано повернуться | ✅ собираем | только свои |
| `native.ordered_width` | Приказанная ширина строя | ✅ собираем | только свои |
| `native.is_moving` | Двигается | ✅ собираем | только свои |
| `native.is_moving_fast` | Бежит бегом | ✅ собираем | только свои |
| `native.is_idle` | Стоит без приказа | ✅ собираем | только свои |
| `native.is_leaving_battle` | Покидает поле | ✅ собираем | только свои |
| `native.slow_speed` | Скорость шагом | ✅ собираем | только свои |
| `native.fast_speed` | Скорость бегом | ✅ собираем | только свои |
| `cco.IsWithdrawing` | Отступает с поля | ✅ собираем | только свои |
| `move.walk_run` | Идёт шагом или бежит (по интерфейсу) | ⬜ есть в игре | только свои |
| `move.flying` | Умеет летать, летит сейчас, можно ли переключить | ⬜ есть в игре | только свои |
| `move.officer_position` | Где стоит командир отряда | ⬜ есть в игре | только свои |
| `move.soldier_positions` | Где стоит каждый боец (реальная форма строя) | ⬜ есть в игре | только свои |
| `move.buildings_walls` | В здании, на стене, лезет по лестнице | ⬜ есть в игре | только свои |
| `move.radar_position` | Положение на мини-карте | ⬜ есть в игре | только свои |
| `nav.can_reach` | Может ли отряд дойти до точки | ✅ собираем | только свои |
| `nav.deploy_state` | Расставляется / расставлен | ✅ собираем | только свои |

## Приказы и управление

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.is_script_controlled` | Под управлением скрипта | ✅ собираем | только свои |
| `native.is_controllable` | Можно отдавать приказы | ✅ собираем | только свои |
| `native.behaviours_active` | Какие режимы включены: оборона, перестрелка, свободный огонь, плотность | ✅ собираем | только свои |
| `orders.controller` | Кто управляет отрядом: ИИ или игрок | ⬜ есть в игре | только свои |
| `army.player_controlled` | Армия под управлением игрока | ✅ собираем | общее для всех |

## Бой: кто кого бьёт

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.is_in_melee` | В рукопашной | ✅ собираем | только свои |
| `native.current_target` | Кого атакует (только если цель видна) | ✅ собираем | только свои |
| `native.flank_threats` | Угроза с левого/правого фланга и с тыла, и кто угрожает | ✅ собираем | только свои |
| `cco.DamageInflictedRecently` | Урон, нанесённый за 4 секунды | ✅ собираем | только свои |
| `cco.NumKills` | Сколько врагов убил | ✅ собираем | только свои |
| `combat.kills_native` | Сколько врагов убил (по движку) | ⬜ есть в игре | только свои |
| `combat.threat_of_unit` | Насколько опасен для отряда конкретный враг | ⬜ есть в игре | только свои |
| `combat.execution` | Может ли вражеский герой добить отряд одним ударом | ⬜ есть в игре | только свои |
| `combat.capturing` | Захватывает точку, сила захвата | ⬜ есть в игре | только свои |
| `combat.pair_distance` | Расстояние между двумя отрядами (с учётом размеров) | ✅ собираем | только свои |

## Стрельба и боезапас

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `native.ammo_left` | Сколько осталось стрел/патронов | ✅ собираем | только свои |
| `native.starting_ammo` | Сколько боеприпасов было | ✅ собираем | только свои |
| `cco.PrimaryAmmoPercent` | Боезапас в процентах | ✅ собираем | только свои |
| `native.is_under_missile_attack` | Под обстрелом | ✅ собираем | только свои |
| `cco.IsFiringMissiles` | Стреляет сейчас | ✅ собираем | только свои |
| `range.missile_range` | Дальность стрельбы (движок и карточка) | ✅ собираем | только свои |
| `range.in_range` | Цель в радиусе стрельбы | ✅ собираем | только свои |
| `ranged.reload` | Перезарядка: сколько осталось до выстрела | ⬜ есть в игре | только свои |
| `ranged.projectile` | Каким снарядом стреляет сейчас | ⬜ есть в игре | только свои |
| `ranged.fire_at_will` | Включён свободный огонь (по интерфейсу) | ⬜ есть в игре | только свои |
| `ranged.secondary_ammo` | Второй вид боеприпасов, бесконечные боеприпасы | ⬜ есть в игре | только свои |

## Способности и магия

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `abilities.owned` | Какие способности есть у отряда | ✅ собираем | только свои |
| `abilities.can_perform` | Можно ли применить способность сейчас | ✅ собираем | только свои |
| `abilities.state` | Состояние способностей: перезарядка, осталось применений, активна, радиус, мана, шанс сбоя | ⬜ есть в игре | только свои |
| `abilities.effects_count` | Сколько эффектов действует на отряд | ✅ собираем | только свои |
| `abilities.effects` | Какие эффекты действуют (свои усиления, вражеские заклинания) и сколько им осталось | ⬜ есть в игре | только свои |
| `cco.StatusList` | Значки состояний отряда: готов к натиску, скрыт, стреляет, в бою… | ✅ собираем | только свои |
| `magic.winds` | Запас магии армии и скорость восстановления | ⬜ есть в игре | только свои |
| `magic.can_use` | Отряд умеет колдовать | ⬜ есть в игре | только свои |

## Видимость и скрытность

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `vis.visible_to_side` | Видит ли сторона отряд | ✅ собираем | свои и видимые враги |
| `vis.last_seen` | Где противника видели последний раз и когда | ✅ собираем | свои и видимые враги |
| `native.is_hidden` | Отряд скрыт (лес, высокая трава, скрытность) | ✅ собираем | только свои |
| `vis.unspottable` | Невидим по своей природе | ⬜ есть в игре | только свои |
| `vis.guerrilla` | Может расставиться за пределами своей зоны | ⬜ есть в игре | только свои |

## Армия и стороны

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `army.role` | Атакующий или обороняющийся | ✅ собираем | общее для всех |
| `army.counts` | Армия: сколько отрядов и бойцов осталось, сколько погибло, подкрепления | ⬜ есть в игре | только свои |
| `army.general_alive` | Жив ли генерал армии | ⬜ есть в игре | только свои |
| `army.score` | Очки армии | ⬜ есть в игре | только свои |
| `battle.balance_of_power` | Баланс сил сторон (полоска вверху экрана) | ⬜ есть в игре | общее для всех |
| `alliance.timeout_winner` | Кто победит, если время выйдет | ⬜ есть в игре | только свои |

## Поле боя

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `battle.phase` | Фаза боя: расстановка, бой, итоги | ✅ собираем | общее для всех |
| `battle.time` | Время боя | ✅ собираем | общее для всех |
| `battle.time_left` | Сколько времени осталось до лимита | ⬜ есть в игре | общее для всех |
| `battle.outcome` | Исход решён, кто победил | ✅ собираем | общее для всех |
| `battle.speed` | Скорость игры | ✅ собираем | общее для всех |
| `battle.kind` | Тип боя: кампания, мультиплеер, осада, засада… | ✅ собираем | общее для всех |
| `battle.boundaries` | Граница поля, за которую нельзя выходить | ⬜ есть в игре | общее для всех |
| `battle.weather` | Погода, климат, время года | ⬜ есть в игре | общее для всех |
| `battle.capture_points` | Точки захвата: чья, прогресс, кто оспаривает | ⬜ есть в игре | общее для всех |
| `battle.reinforcements` | Подкрепления и их валюта | ⬜ есть в игре | только свои |

## События боя

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `event.entity_hit` | Попадание снаряда: в кого, артиллерия ли | ⬜ есть в игре | только полная сводка |
| `event.orders` | Отданные приказы: атака кого, движение, способность, смена снаряда, отход, стоп | ⬜ есть в игре | только свои |
| `event.routs` | Отряд побежал, генерал побежал | ⬜ есть в игре | только свои |
| `event.left_battlefield` | Отряд ушёл с поля | ⬜ есть в игре | только свои |
| `event.armies_engaging` | Армии сошлись в бою | ⬜ есть в игре | только свои |

## Карта

| ID | Что это | Статус | Доступ для ИИ |
|---|---|---|---|
| `map.radar_frame` | Рамка мини-карты (размер поля) | ✅ собираем | общее для всех |
| `map.height` | Высота земли в точке | ✅ собираем | общее для всех |
| `map.ground` | Тип земли: трава, лес, вода… | ✅ собираем | общее для всех |
| `map.clear` | Свободна ли площадка | ✅ собираем | общее для всех |
| `map.buildings` | Здания и сооружения: где, какие, здоровье, мост ли | ✅ собираем | общее для всех |

<details><summary>Технические вызовы</summary>

| ID | API | Модуль |
|---|---|---|
| `unit.type` | `unit:type()` | entries.unit_readout (profile) |
| `unit.name` | `unit:name()` | apps.battle.adapter |
| `unit.id` | `unit:unique_ui_id()` | apps.telemetry.sampler_adapter |
| `unit.commanding` | `unit:is_commanding_unit()` | entries.unit_readout (profile) |
| `unit.class` | `unit:unit_class()` | apps.units.card_adapter |
| `unit.kind_flags` | `unit:is_infantry/is_cavalry/is_pikemen/is_anti_cavalry_infantry/is_lancers/is_chariot/is_war_beasts/is_elephants/is_artillery/is_war_machine()` | apps.units.card_adapter |
| `unit.is_character` | `CCO IsCharacter / IsGeneral` | — |
| `unit.renown` | `CCO IsRenown` | — |
| `unit.spawned` | `CCO IsSpawnedUnit` | — |
| `unit.attributes` | `unit:has_attribute(key)` | entries.unit_readout (profile) |
| `unit.behaviours_available` | `unit:can_use_behaviour(key)` | entries.unit_readout (profile) |
| `unit.card_stats` | `CCO UnitDetailsContext.StatList / BaseStatValueFromKey` | apps.units.card_adapter |
| `unit.card_morale` | `CCO UnitDetailsContext.StatList stat_morale` | apps.units.state_adapter |
| `unit.mass` | `CCO UnitDetailsContext.Mass` | apps.units.card_adapter |
| `unit.experience_level` | `CCO ExperienceLevel` | entries.unit_readout (profile) |
| `unit.experience_progress` | `CCO ExperiencePercent` | — |
| `unit.character_rank` | `CCO CharacterRank / HasCharacterRank` | entries.unit_readout (profile) |
| `native.number_of_men_alive` | `unit:number_of_men_alive()` | apps.units.state_adapter |
| `native.initial_number_of_men` | `unit:initial_number_of_men()` | apps.units.state_adapter |
| `native.unary_of_men_alive` | `unit:unary_of_men_alive()` | apps.units.state_adapter |
| `cco.NumEntities` | `CCO NumEntities` | apps.units.state_adapter |
| `cco.NumEntitiesInitial` | `CCO NumEntitiesInitial` | apps.units.state_adapter |
| `unit.strategic_value` | `unit:strategic_value()` | — |
| `cco.PercentCasualtiesRecently` | `CCO PercentCasualtiesRecently` | — |
| `native.unary_hitpoints` | `unit:unary_hitpoints()` | apps.units.state_adapter |
| `cco.HealthValue` | `CCO HealthValue` | apps.units.state_adapter |
| `cco.HealthMax` | `CCO HealthMax` | apps.units.state_adapter |
| `cco.HealthPercent` | `CCO HealthPercent` | apps.units.state_adapter |
| `cco.PercentHpLostRecently` | `CCO PercentHpLostRecently` | apps.units.state_adapter |
| `cco.IsTakingDamage` | `CCO IsTakingDamage` | apps.units.state_adapter |
| `health.delayed` | `CCO DelayedHealthPercent` | — |
| `health.barrier` | `CCO BarrierHp / BarrierMaxHp / BarrierHpPercent / IsBarrierCharging / BarrierSecsUntilRecharge` | — |
| `health.healing` | `CCO HealingPower / HealthReplenishRateBase / MaxHealthPercentCanReplenish` | — |
| `health.wounded` | `CCO IsWounded` | — |
| `health.invulnerable` | `unit:is_invulnerable() / CCO IsInvincible` | — |
| `native.is_wavering` | `unit:is_wavering()` | apps.units.state_adapter |
| `native.is_routing` | `unit:is_routing()` | apps.units.state_adapter |
| `native.is_shattered` | `unit:is_shattered()` | apps.units.state_adapter |
| `cco.MoralePercent` | `CCO MoralePercent` | apps.units.state_adapter |
| `cco.MoraleState` | `CCO MoraleState` | apps.units.state_adapter |
| `cco.MoraleName` | `CCO MoraleName` | apps.units.state_adapter |
| `cco.MoraleGreatestEffect` | `CCO MoraleGreatestEffect` | apps.units.state_adapter |
| `cco.IsRouting` | `CCO IsRouting` | apps.units.state_adapter |
| `cco.IsShattered` | `CCO IsShattered` | apps.units.state_adapter |
| `cco.IsWavering` | `CCO IsWavering` | apps.units.state_adapter |
| `cco.IsInLastStand` | `CCO IsInLastStand` | apps.units.state_adapter |
| `cco.IsOutOfControl` | `CCO IsOutOfControl` | apps.units.state_adapter |
| `cco.IsAwaitingOrderAfterRally` | `CCO IsAwaitingOrderAfterRally` | apps.units.state_adapter |
| `morale.terrified` | `CCO IsTerrified` | — |
| `morale.rampaging` | `unit:is_rampaging() / CCO IsForceRampage` | — |
| `morale.undead` | `unit:is_crumbling() / is_unstable() / CCO IsUndead` | — |
| `native.fatigue_state` | `unit:fatigue_state()` | apps.units.state_adapter |
| `cco.FatigueState` | `CCO FatigueState` | apps.units.state_adapter |
| `cco.FatigueName` | `CCO FatigueName` | apps.units.state_adapter |
| `native.position` | `unit:position()` | apps.units.state_adapter |
| `native.ordered_position` | `unit:ordered_position()` | apps.units.state_adapter |
| `native.bearing` | `unit:bearing()` | apps.units.state_adapter |
| `native.ordered_bearing` | `unit:ordered_bearing()` | apps.units.state_adapter |
| `native.ordered_width` | `unit:ordered_width()` | apps.units.state_adapter |
| `native.is_moving` | `unit:is_moving()` | apps.units.state_adapter |
| `native.is_moving_fast` | `unit:is_moving_fast()` | apps.units.state_adapter |
| `native.is_idle` | `unit:is_idle()` | apps.units.state_adapter |
| `native.is_leaving_battle` | `unit:is_leaving_battle()` | apps.units.state_adapter |
| `native.slow_speed` | `unit:slow_speed()` | apps.units.state_adapter |
| `native.fast_speed` | `unit:fast_speed()` | apps.units.state_adapter |
| `cco.IsWithdrawing` | `CCO IsWithdrawing` | apps.units.state_adapter |
| `move.walk_run` | `CCO IsWalking / IsRunning` | — |
| `move.flying` | `unit:can_fly() / is_currently_flying() / CCO CanToggleFlying` | — |
| `move.officer_position` | `unit:position_of_officer()` | — |
| `move.soldier_positions` | `CCO ManList / EntityList -> Position` | — |
| `move.buildings_walls` | `unit:is_currently_garrisoned() / is_on_top_of_wall() / is_climbing_ladder()` | — |
| `move.radar_position` | `CCO RadarPosition` | — |
| `nav.can_reach` | `unit:can_reach_position(p)` | apps.navigation.adapter |
| `nav.deploy_state` | `unit:is_deploying() / is_deployed()` | apps.navigation.diagnostics_adapter |
| `native.is_script_controlled` | `unit:is_script_controlled()` | apps.units.state_adapter |
| `native.is_controllable` | `unit:is_controllable()` | apps.units.state_adapter |
| `native.behaviours_active` | `unit:is_behaviour_active(key)` | apps.units.state_adapter |
| `orders.controller` | `unit:is_ai_controlled() / is_player_controlled()` | — |
| `army.player_controlled` | `army:is_player_controlled()` | entries.ai_vs_ai |
| `native.is_in_melee` | `unit:is_in_melee()` | apps.units.state_adapter |
| `native.current_target` | `unit:current_target()` | apps.units.state_adapter |
| `native.flank_threats` | `unit:is_*_flank_threatened() / left_flank_threat() / right_flank_threat() / rear_threat()` | apps.units.state_adapter |
| `cco.DamageInflictedRecently` | `CCO DamageInflictedRecently` | apps.units.state_adapter |
| `cco.NumKills` | `CCO NumKills` | apps.units.state_adapter |
| `combat.kills_native` | `unit:number_of_enemies_killed()` | — |
| `combat.threat_of_unit` | `CCO ThreatPercentOfUnit(unit)` | — |
| `combat.execution` | `CCO CanBeExecutedByNearbyEnemy / CanNearlyBeExecutedByNearbyEnemy` | — |
| `combat.capturing` | `CCO IsCapturing / CapturePower` | — |
| `combat.pair_distance` | `unit:unit_distance(other)` | apps.units.range_adapter |
| `native.ammo_left` | `unit:ammo_left()` | apps.units.state_adapter |
| `native.starting_ammo` | `unit:starting_ammo()` | apps.units.state_adapter |
| `cco.PrimaryAmmoPercent` | `CCO PrimaryAmmoPercent` | apps.units.state_adapter |
| `native.is_under_missile_attack` | `unit:is_under_missile_attack() / CCO IsUnderMissileAttack` | apps.units.state_adapter |
| `cco.IsFiringMissiles` | `CCO IsFiringMissiles` | apps.units.state_adapter |
| `range.missile_range` | `unit:missile_range() / CCO scalar_missile_range` | apps.units.range_adapter |
| `range.in_range` | `unit:unit_in_range(target)` | apps.units.range_adapter |
| `ranged.reload` | `CCO ReloadPercentMax; per soldier IsReloading / ReloadPercent / ReloadRemainingTime` | — |
| `ranged.projectile` | `CCO ActiveProjectileContext` | — |
| `ranged.fire_at_will` | `CCO IsFiringAtWill` | — |
| `ranged.secondary_ammo` | `CCO SecondaryAmmoPercent / HasInfinitePrimaryAmmo / HasInfiniteSecondaryAmmo` | — |
| `abilities.owned` | `unit:owned_passive_special_abilities() / owned_non_passive_special_abilities()` | entries.unit_readout (profile) |
| `abilities.can_perform` | `unit:can_perform_special_ability(key)` | research (unit-state) |
| `abilities.state` | `CCO BattleAbilityList` | — |
| `abilities.effects_count` | `CCO ActiveEffectList.Size` | entries.unit_readout (profile) |
| `abilities.effects` | `CCO ActiveEffectList -> Name / TimeRemainingSeconds` | — |
| `cco.StatusList` | `CCO StatusList` | apps.units.state_adapter |
| `magic.winds` | `CCO WindsOfMagicPoolContext / army:winds_of_magic_current()` | — |
| `magic.can_use` | `unit:can_use_magic()` | — |
| `vis.visible_to_side` | `unit:is_visible_to_alliance(a)` | apps.intel.adapter |
| `vis.last_seen` | `apps.intel.services memory` | apps.intel.services |
| `native.is_hidden` | `unit:is_hidden()` | apps.units.state_adapter |
| `vis.unspottable` | `CCO IsUnspottable` | — |
| `vis.guerrilla` | `CCO CanGuerrillaDeploy / CanGuerrillaDeployCurrently` | — |
| `army.role` | `alliance:is_attacker()` | apps.battle.adapter |
| `army.counts` | `CCO CcoBattleArmy NumUnits / NumEntities / NumMenDied / NumReinforcements` | — |
| `army.general_alive` | `army:is_commander_alive()` | — |
| `army.score` | `CCO CcoBattleArmy Score` | — |
| `battle.balance_of_power` | `CCO BattleRoot.BalanceOfPowerPercent` | — |
| `alliance.timeout_winner` | `CCO CcoBattleAlliance WillWinOnTimeout` | — |
| `battle.phase` | `bm:get_current_phase_name()` | apps.battle / entries |
| `battle.time` | `bm:time_elapsed_ms()` | entries |
| `battle.time_left` | `bm:remaining_conflict_time()` | — |
| `battle.outcome` | `bm:battle_outcome_decided() / victorious_alliance()` | entries |
| `battle.speed` | `bm:current_battle_speed()` | apps.battle.adapter |
| `battle.kind` | `bm:is_from_campaign() … / battle_type()` | apps.battle.adapter |
| `battle.boundaries` | `bm:battle_boundaries()` | — |
| `battle.weather` | `CCO BattleRoot WeatherRecordContext / ClimateRecordContext / SeasonRecordContext` | — |
| `battle.capture_points` | `CCO BattleRoot.CapturePointList` | — |
| `battle.reinforcements` | `CCO BattleRoot reinforcement pool / currency` | — |
| `map.radar_frame` | `BattleRadarPosition` | apps.map.adapter |
| `map.height` | `bm:get_terrain_height(x, z)` | apps.map.adapter |
| `map.ground` | `bm:ground_type(p)` | apps.map.adapter |
| `map.clear` | `bm:is_area_clear(p)` | apps.map.adapter |
| `map.buildings` | `bm:buildings() / CCO BuildingsList` | apps.map.adapter |
| `event.entity_hit` | `command handler 'Entity Hit'` | apps.telemetry.sampler_adapter (parser only) |
| `event.orders` | `command handler 'Attack Unit' / 'Move' / 'Special Ability' / …` | — |
| `event.routs` | `BattleUnitRouts / BattleCommandingUnitRouts` | — |
| `event.left_battlefield` | `command handler 'Unit Left Battlefield'` | — |
| `event.armies_engaging` | `ScriptEventBattleArmiesEngaging` | — |

</details>
