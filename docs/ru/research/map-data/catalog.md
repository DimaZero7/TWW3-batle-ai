# Данные карты Total War: WARHAMMER III

Дата исследования: 25.09.2026. Область: только поле боя и его окружение.
Отряды, их характеристики, приказы, обучение и выбор входов нейросети сюда не входят.

Это каталог источников данных, а не утверждённая схема нейросети. Первичное
исследование проведено без запуска. Позднее выполнена отдельная
[проверка MP Crossroads в игре](crossroads-live-check.md): высота, тип поверхности
и проверка ограничений участка сработали в 25 точках. Экспорт XML границ не дал.
Затем [проверены большие области до 8 × 8 км](crossroads-large-area-check.md):
отрицательные ответы получены, но координаты края из них не установлены.
В [сетке из 6400 клеток](crossroads-grid-check.md) замкнутая рамка ограничений
не обнаружена. Повторение высот за областью около X/Z = ±1000 м — наблюдаемая
особенность данных, пока не подтверждённая граница поля боя.
В [интернет-исследовании](map-edges-internet-research.md) найден новый
кандидат: `BattleRadarPosition` и позиции отрядов на радаре. Наличие описано
в документации WH3; восстановление рамки таким способом ещё не испытано.
Позднее [опыт с мини-картой](crossroads-radar-check.md) подтвердил работу
`BattleRadarPosition` и рамку 1536 × 1536 м в одном бою. Совпадение этой рамки
с пределами приказов движения пока не проверено.
На [штатном ландшафте Кислева](kislev-plains-check.md) получена другая рамка —
1024 × 1024 м — и сетка из 900 клеток. Запросы поверхности возвращают данные и за
рамкой мини-карты. Ручной проезд возле четырёх сторон согласуется с рамкой:
1088 позиций внутри, ближайшие приближения к сторонам — 3.73–20.68 м.
Это не погрешность метода и не доказательство точного предела движения.
Проезд через 23 клетки с отрицательным ответом для полного квадрата также
подтвердил, что нельзя считать всю такую клетку непроходимой.
Далее [сравнены сетки 5, 3, 2 и 1 м](kislev-scales-check.md) в одном запуске.
Метровая сетка из 1 048 576 клеток прочитана за 7.319 с по игровому `os.clock`;
двухметровая — за 1.921 с. На этой карте ответы двухметровой и метровой сеток
согласуются по цвету на 99.796% площади; это не мера точности навигации.

## Как читать степень подтверждения

| Метка | Что установлено |
|---|---|
| **API** | Метод и его результат описаны в документации WH3. Нашего испытания в движке ещё нет. |
| **СКРИПТ** | Реализация или использование найдены в установленном `data_script.pack`. Это статическая проверка, не вызов API в бою. |
| **CCO** | Поле документировано в системе контекстов игры. Нужен доступ к соответствующему объекту; значение и преобразование в Lua ещё не проверены. |
| **ФАЙЛ** | Структура существует в формате ресурсов или запись обнаружена в архиве. Это не автоматическая выгрузка текущей карты через Lua. |
| **РАСЧЁТ** | Производная величина, которую считаем сами из наблюдений; при выборке точек результат приблизительный. |
| **НЕ ПОДТВЕРЖДЕНО** | Надёжного способа получения в исследованных интерфейсах не найдено. Не используем как гарантированный вход. |

Метки в таблицах ниже отражают первоначальную проверку документации и файлов.
Результаты живых вызовов вынесены в связанный выше отчёт; они не подтверждают
все остальные возможности каталога. «Есть в документации» и «получено в бою» —
разные уровни доказательства. Отсутствие найденного метода не доказывает отсутствие
любого недокументированного способа.

## 1. Координаты и единицы

| Данные | Игровой вид | Единицы / смысл | Основание |
|---|---|---|---|
| Положение в пространстве | `battle_vector`, компоненты `get_x()`, `get_y()`, `get_z()` → `number` | Метры; X/Z — горизонтальная плоскость, Y — высота относительно водной плоскости | API |
| Расстояние между точками | `distance(other)` → `number` | Метры, три измерения | API |
| Горизонтальное расстояние | `distance_xz(other)` → `number` | Метры, высота исключена | API |
| Положение из CCO | `Vector4` | Тип контекстной системы; четвёртый компонент не объявляем высотой или размером. Преобразование отдельно проверить | CCO |

Координаты допускают дробные числа. Метр — единица измерения, а не минимальная
клетка. Точность физики, минимальный шаг координат, разрешение навигации и сетки
рельефа не установлены. `float32` в файле карты не означает, что все Lua-значения
и все вычисления движка имеют тот же формат.

Источник: [Vector](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_vector.html).

## 2. Поверхность, рельеф и ограничения перемещения

| Что можно запросить | Точный источник / аргументы | Результат | Подтверждение и предел |
|---|---|---|---|
| Высота земли в точке | `bm:get_terrain_height(x, z)` | `number`, метры | API + СКРИПТ. В документации второй горизонтальный аргумент назван `y`; игровые библиотеки передают сюда Z |
| Тип поверхности в точке | `bm:ground_type(position)` | `string`, ключ типа поверхности | API + СКРИПТ. Ключ сопоставляется с `ground_types`; это не геометрия всех объектов в точке |
| Отсутствие ограничений поиска пути внутри прямоугольника | `bm:is_area_clear(position, bearing, width, depth, is_naval)` | `boolean` | API. Центр, угол в градусах, ширина/глубина в метрах, `is_naval` по умолчанию false |
| Доля выбранного типа поверхности вдоль линии | `ground_type_proportion_along_line(a, b, ground_type, samples)` | `number`, процент совпавших проб | API + СКРИПТ; выборочная оценка, не точная длина участка |
| Доля типа поверхности в повёрнутом прямоугольнике | `ground_type_proportion_in_bounding_box(centre, bearing, width, depth, ground_type, width_samples, depth_samples)` | `number`, заявленный процент | API + СКРИПТ; угол здесь **в радианах**. Есть ошибка реализации, см. ниже |
| Точка на заданной высоте над землёй | `v_to_ground(...)` | `battle_vector` | СКРИПТ; обёртка над запросом высоты, не дополнительный источник геометрии |

Запрос высоты документирован прежде всего для камеры и маркеров. Он возвращает
одно число, а не несколько поверхностей по вертикали: считать его полным
описанием мостов, стен, туннелей или навесов нельзя. Поведение вне поля не описано.

`is_area_clear` не возвращает причину запрета, контур препятствия или путь. Документация
не гарантирует, что им можно точно определить белую границу карты. Также это не
проверка отсутствия других отрядов, простреливаемости или безопасности точки.

**Лес подтверждён конкретнее, чем остальные категории:** штатный
`script/battle/scripted_tours/battle_fundamentals_tour.lua:297` передаёт ключ
`forest` функции оценки поверхности. Полный перечень действующих ключей в этом
исследовании не выгружен; названия для воды, грязи, снега и прочего не придумываем.
Тип `forest` не сообщает число деревьев, их стволы или фактическое перекрытие обзора.

**Ошибка библиотечной оценки площади:** установленная реализация суммирует
`width_samples` значений, но делит на `depth_samples`. При разных значениях двух
параметров итог не является корректным процентом. При одинаковых счётчиках этого
искажения нет. Проверено чтением функции, не игровым опытом; исправление не внесено.

Источники: [Battle: запросы местности](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle.html#section:battle:Querying),
[библиотечные функции поверхности](https://chadvandy.github.io/tw_modding_resources/WH3/battle/global.html#section:global:Ground%20Types),
`data_script.pack` → `script/_lib/lib_battle_misc.lua:1612–1694` и
`script/_lib/lib_convex_area.lua:136–157`. Результат статической проверки сохранён
в `static-evidence.json` (локальный архив: `research/evidence/map-data/static-evidence.json`).

## 3. Границы и идентификация карты

| Источник | Реально существующие данные | Вид | Что ещё не доказано |
|---|---|---|---|
| `battle` / `battle_manager` | Прямой getter полного прямоугольника игрового поля не найден | — | Автоматическое получение четырёх границ произвольной битвы |
| `CcoBattleRecord` | `PlayableAreaWidth`, `PlayableAreaHeight` | `Float` | Это запись карты, а не подтверждённые текущие пределы; положение прямоугольника, единицы полей и переопределения требуют проверки |
| `CcoBattleRecord` | `Key`, `Map`, `Specification`, `CatchmentName`, `EnvironmentPath` | `String` | Как однозначно связать запись с произвольной кампанийной битвой |
| `CcoCustomBattleMap` | `BattleRecordContext`, `CampaignBattlePresetRecordContext`, `CampaignMapRecordContext` | CCO-ссылки | Контекст меню пользовательского боя не равен объекту текущей кампанийной карты |
| `CcoCustomBattleMap` | `ClickgenCoords`, `ClickgenCoordsRaw` | `Vector4` | Это координаты выбора карты; не границы в боевых метрах |
| `CcoCustomBattleMap` | `Name`, `ClickgenProvinceName`, `ClickgenRegionName`, `ClickgenGroundTypeName` | `UniString` | Названия не содержат геометрию |
| `CcoBattleRoot` | `BattleNameText`, `BattleTypeState`, `SetupContext` | `UniString`, `String`, CCO | Имя боя не гарантирует уникальный путь к карте |
| Battle XML | Путь `battle_map_definition/name`, `tile_map_position`, `tile_upgrade`; иногда `playable_area` | XML: строки, числа, атрибуты | Полнота данных текущего экспорта и итоговая область после загрузки |
| BMD | `playable_area.area`, `has_been_set`, `valid_location_flags` | Прямоугольник, `bool`, флаги | Выбор активного слоя/тайла, преобразование координат, динамические переопределения |

В BMD-прямоугольнике RPFM читает **четыре `f32`: `min_x`, `min_y`, `max_x`,
`max_y`**. Здесь сохранены имена полей формата; файловый `y` нельзя автоматически
считать высотой Y из `battle_vector`. `has_been_set` важен: наличие структуры
само по себе не означает явно заданные границы.

Есть реальный пример в установленном архиве:
`script/benchmarks/empire_vs_greenskins/battle_benchmark.xml:363` содержит
`playable_area` с `dimension=700`, `centre_x=-130`, `centre_y=0`. Это параметры
**другого штатного сценария**, не измерение нашей карты и не доказательство
экспорта границ любого боя. Не интерпретируем `dimension` как сторону или полусторону
без отдельной проверки формата.

**Ответ о точных границах:** подтверждено существование полей в данных карты.
Универсальная цепочка «текущий бой → активная область → точные мировые границы в Lua»
пока не подтверждена. Числовые границы не записываем в утверждённые входы.

Отдельно проверены ресурсы `mp_crossroads_flat`: размеры в DB равны нулю,
а у прямоугольников верхнего BMD и слоя `catchment_03` выключен флаг
`has_been_set`. Подробности и пределы этой проверки:
[результат исследования MP Crossroads](crossroads-boundaries.md).

Источники: [CCO](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleRecord),
[PlayableArea в RPFM](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd/playable_area/mod.rs),
[Rectangle в RPFM](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd/common/mod.rs),
[редактор: Playable Area](https://wiki.totalwar.com/w/TWWAKT_Creating_Zones).
Последняя статья относится к старому Warhammer: её числовые правила не перенесены на WH3.

## 4. Здания, стены, ворота и мелкие объекты

Доступ: `bm:buildings()` → коллекция `battle_buildings`; `count()` → `number`,
`item(index)` → `battle_building`. Название класса не гарантирует, что это только
крупные здания: документация упоминает и мелкие предметы. Полнота охвата всей
видимой декорации не установлена.

| Поле объекта | Вид | Смысл |
|---|---|---|
| `position()` | `battle_vector` | Опорная точка модели; может не совпадать с центром |
| `central_position()` | `battle_vector` | Центр |
| `orientation()` | `number`, градусы | Ориентация |
| `name()` | `string` | Имя |
| `category()` | `string` | Ключ `battlefield_building_categories` |
| `health()` | `number`, 0…1 | Состояние здания |
| `has_gate()`, `is_fort_wall()`, `is_fort_tower()`, `is_selectable()` | `boolean` | Признаки объекта |
| `next()`, `previous()` | `battle_building` либо `nil` | Соседние звенья крепостной стены |
| `alliance_owner_id()` | `number` | Владелец объекта; −1 при отсутствии |

Основание: API. Нет документированного получения полного контура здания,
его габаритов и коллизионной сетки этими методами.

Источники: [Buildings](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_buildings.html),
[Building](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_building.html).

Дополнительный канал **CCO**, через `CcoBattleRoot.BuildingsList` → `CcoBattleBuilding`:

| Поля | Вид |
|---|---|
| `Position`, `CentralPosition` | `Vector4` |
| `IsBridge`, `IsGate`, `IsUnbreachableWall`, `IsIncidental`, `IsToggleable` | `Bool` |
| `IsDestroyed`, `IsDestructable`, `IsFlammable`, `IsOnFire`, `IsEnabled` | `Bool` |
| `CurrentHitpoints`, `MaxHitpoints`, `GateCurrentHitpoints`, `GateMaxHitpoints` | `Float` |
| `DamagePercent`, `GateDamagePercent`, `PercentOnFire` | `Float`; шкалу процентов проверить |
| `CategoryType`, `Name` | `String`, `UniString` |
| `CaptureLocationContext` | `CcoBattleCapturePoint` |

Это состояние объектов карты. Оружие, способности и боевые показатели зданий здесь
не каталогизируются. Источник: [CCO Building](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleBuilding).

## 5. Точки захвата и связанные зоны

Доступ: `bm:capture_location_manager()` → менеджер;
`count()` / `item(index)` → количество и `battle_capture_location`.

| Поле | Вид | Содержание |
|---|---|---|
| `position()` | `battle_vector` | Положение |
| `unique_id()` | `number` | Идентификатор |
| `script_id()` | `string` | Заданный редактором ID; может быть пустым |
| `type()` | `string` | Ключ `capture_location_types` |
| `is_enabled()`, `is_locked()`, `contributes_to_victory()` | `boolean` | Доступность и назначение |
| `linked_buildings()` | `battle_buildings` | Связанные объекты |
| `is_held()`, `is_contested()` | `boolean` | Состояние контроля объекта |
| `holding_alliance_id()`, `contesting_alliance_id()` | `number`, для отсутствующего оспаривания возможен пустой результат | Владелец/оспаривающая сторона; отсутствие владельца описано как 0 |

Основание: API. Список отрядов у точки не входит в этот каталог.
Источник: [Capture Location](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_capture_location.html).

**CCO:** `CcoBattleRoot.CapturePointList` → `CcoBattleCapturePoint` дополнительно
объявляет `Radius` (`Float`), `FlagPosition` (`Vector4`),
`LinkedParentCapturePointContext`, `LinkedChildCapturePointsList`,
`BuildingList`, `ToggleableSlotsList`. Это позволяет исследовать размер и связи
точек. Единицы `Radius`, его соответствие реально захватываемой области и
преобразование списков в Lua ещё не проверены; точный полигон этим не получен.
Источник: [CCO Capture Point](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleCapturePoint).

## 6. Переключаемые преграды и места строительства

Доступ: `bm:toggle_system()`; `toggle_slot_count()` / `toggle_slot(index)`,
`map_barrier_count()` / `map_barrier(index)`. Объекты можно искать и по script ID.

| Объект / данные | Вид | Основание / ограничение |
|---|---|---|
| `toggle_slot:position()` | `battle_vector` | API |
| `unique_id()`, `unique_ui_id()` | `number` | API |
| `script_id()`, `slot_type()` | `string` | API; тип связан с `toggle_system_types` |
| `has_map_barrier()`, `is_held_by_alliance()` | `boolean` | API |
| `map_barrier()`, `holding_alliance()` | Ссылка на объект либо пустой результат | API |
| `map_barrier:enabled()` | `boolean` | API; состояние преграды |
| `map_barrier:record_key()` | `string` | API; ключ записи |
| `has_toggle_slot()`, `toggle_slot()` | `boolean`, объект/пусто | API; связь со слотом |
| `map_barrier:position(...)` | **Контракт противоречив** | Текст говорит о чтении позиции, сигнатура принимает вектор и объявляет `nil`. Не считаем проверенным getter |

`Map Barrier` — отдельный переключаемый объект, а не внешняя граница всего поля.
Контур преграды и её размеры через перечисленные методы не получены.
Источник: [Toggle System](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_toggle_system.html).

**CCO `CcoBattleToggleableSlot`:** `WorldPosition` (`Vector4`), `IsBuilding`,
`IsMapBarrier`, `IsDestroyed`, `IsAnyEnabled` (`Bool`), `EnabledIndex` (`Int`),
`IsConstructionInProgress`, `IsDestructionInProgress` (`Bool`),
`ConstructionTimeRemainingSeconds` (`Float`, секунды), `ParentCapturePointContext`
и `BuildablesList` (ссылки). Для `WorldPosition` описание упоминает Vector3 при
объявленном Vector4 — формат преобразования нужно проверить.
Источник: [CCO Toggleable Slot](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleToggleableSlot).

## 7. Зоны появления, линии входа и выход из засады

| Источник | Данные / вид | Предел |
|---|---|---|
| `bm:reinforcements():spawn_zone_count()` / `spawn_zone(index)` | Количество → `number`; объект `battle_spawn_zone` | API |
| `spawn_zone:unique_id()`, `position()` | `number`, `battle_vector` | Авторская позиция зоны, не её полный контур |
| `has_reinforcement_line()`, `reinforcement_line()` | `boolean`, ссылка на линию | Геометрия линии этим не возвращается |
| `reinforcement_line:script_id()` | `string` | Идентификатор, не концевые точки |
| `attacker_reinforcement_lines_count()` / `attacker_reinforcement_line(index)`; аналогично `defender_…` | Число и объекты линий | API; координат концов в проверенном интерфейсе нет |
| `CcoBattleRoot.SpawnZoneList` → `CcoBattleSpawnZone` | `Position: Vector4`, `UniqueID: Int`, `IsVanguardOnly: Bool` | CCO |
| `CcoBattleRoot.HasAmbushWithdrawArea`, `AmbushWithdrawAreaPosition` | `Bool`, `Vector4` | CCO; положение выхода, не полигон |

Доступность зоны для стороны можно запросить отдельно, но это уже условие боя,
а не геометрия. Не восстанавливаем внешний периметр карты по точкам появления.

Источники: [Reinforcements](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_reinforcements.html),
[CCO Root](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleRoot).

## 8. Климат, погода, сезон и визуальное окружение

**CCO:** у `CcoBattleRoot` есть `ClimateRecordContext`, `WeatherRecordContext`,
`SeasonRecordContext`, `ClimateWeatherDescriptionRecordContext` — ссылки на
записи соответствующих справочников текущего боя.
В описаниях многих вложенных полей тип результата указан `Void`. Не заменяем
его выдуманным числом или строкой; точный Lua-контракт таких полей не установлен.

В XML нашего сценария фактически присутствуют `time_of_day`, `season`,
`precipitation_type`, `prevailing_wind`, `environment_key`, `sea_surface_name`.
Это параметры сценария, а не доказанные getters текущей погоды из любого боя.
Наличие дождя/тумана в окружении не даёт автоматически числовой карты видимости.

Источник: [CCO Root](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleRoot),
`scenarios/ranged_melee.xml` в нашем репозитории.

## 9. Данные ресурсов карты вне игрового Lua

Формат BMD разбирается библиотекой RPFM. Ниже — реально объявленные структуры,
не обещание готового считывателя в нашем моде. Поля зависят от версии файла.

| Содержание | Структура / вид после разбора | Что это даёт и чего не гарантирует |
|---|---|---|
| Игровая область | `PlayableArea`, `Rectangle`, `bool`, флаги | Авторские границы; связь с активным боем ещё не установлена |
| Здания и дальние здания | `battlefield_building_list`, `battlefield_building_list_far` | Списки экземпляров; состояние разрушения в бою нужно читать отдельно |
| Контуры поверхности / воды / иных областей | `terrain_outlines`, `water_outlines`, `non_terrain_outlines` | Списки `Outline2d`, состоящих из `Point2d` с двумя `f32` |
| Области перемещения | `go_outlines` | Авторские данные; не доказанная финальная навигационная сетка |
| Упрощённые контуры зданий | `lite_building_outlines` | Отдельная структура; точность коллизий не установлена |
| Зоны расстановки | `deployment_list` | Авторские определения; не обязательно итоговые зоны кампанийного боя |
| Линии, шаблоны зон, точки захвата | `ef_line_list`, `zones_template_list`, `capture_location_set` | Структурированные списки объектов |
| Подсказки штатному ИИ | `ai_hints` | Авторская разметка, а не универсальная оценка выгодности местности |
| Слои и экземпляры составных объектов | `bmd_catchment_area_list`, `prefab_instance_list` | Нужно учитывать активные варианты и преобразования |
| Места строительства | `toggleable_buildings_slot_list` | Авторские слоты, состояние во время боя отдельно |
| Деревья и трава | BMD-ссылки + `BmdVegetation.tree_list`, `grass_list` | Списки размещения, отдельный ресурс; не готовые маски видимости и проходимости |
| Декорации | `prop_list`, `composite_scene_list`, `custom_material_mesh_list` | Размещения/ссылки; не подтверждённая коллизионная геометрия |
| Рельефные трафареты и декали | `terrain_stencil_triangle_list`, `terrain_stencil_blend_triangle_list`, `terrain_decal_list` | Эти структуры не равны полной карте высот |
| Свет, эффекты, звук, ограничения камеры | Списки lights/probes/particles/sound shapes, `camera_zones` | Данные окружения, не внешняя игровая граница |

Позиции в структурах могут быть `Point3d` (три `f32`), области — `Cube`
(шесть `f32`), преобразования — матрицы. Перенос из локальных координат экземпляра
в координаты боя нельзя пропускать. Не предполагаем, что один BMD содержит весь
итоговый ландшафт без зависимостей, тайлов и наложений.

Источники: [BMD](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd/mod.rs),
[общие геометрические типы](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd/common/mod.rs),
[BMD Vegetation](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd_vegetation/mod.rs).

В индексах установленных архивов для выбранной нами `wh3_main_macro_kho_wastes_01`
действительно найдены `battle_locations_map.xml/.bin`, `tile_map.bmd/.index/.tiles`,
`full_lf_logic_map.compressed_map`, DDS-маски, `environments.csv` и environment override.
В другом архиве найдены девять записей тайлов `.bin`. **Подтверждены имена файлов;
их содержимое и геометрия этим не декодированы.** Нельзя назначать формат числовых
данных по одному расширению или имени файла.

## 10. Вспомогательные источники и выгрузка

| Способ | Реальный результат | Ограничение |
|---|---|---|
| `bm:output_battle_xml(filename)` | Записывает XML текущего battle setup; возвращает `boolean` успеха | API. Наличие финальных границ в таком экспорте ещё не установлено |
| `bm:camera():position()` / `target()` | `battle_vector` | Позиция камеры и точка взгляда, не границы и не карта высот |
| `CcoBattleRoot.CursorContextContext` → `CcoBattleCursorContext` | `HasIntersections: Bool`, `GroundIntersectPosition`, `FirstIntersectPosition`: `Vector4`; `GroundTypeRecordContext`, `BuildingContext`: ссылки | Зависит от положения курсора. Это не произвольный raycast и не используемый нами способ сканирования |
| `common.get_context_value(...)` / объект `cco` | Значение функции контекста / ссылка | Канал доступа к CCO; не даёт отсутствующие поля автоматически |
| `CcoCustomBattleMap.DevInfo`, `OpenTerrainHTML` | Отладочная строка / действие открытия HTML | Помечено debug; доступность в пользовательской сборке не подтверждена, не основа сбора |

Пиксельные размеры UI/миникарты не равны мировым метрам. Чтение CCO — обращение
скрипта к данным, а не Computer Use. Управлять курсором для исследования не пробовали.

Источники: [Battle](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle.html),
[Camera](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_camera.html),
[Common](https://chadvandy.github.io/tw_modding_resources/WH3/battle/common.html),
[CCO Cursor](https://chadvandy.github.io/tw_modding_resources/WH3/cco/documentation.html#CcoBattleCursorContext).

## 11. Чего сейчас нельзя считать доступными точными данными

| Желаемая информация | Статус |
|---|---|
| Итоговые мировые границы любого загруженного боя | Не подтверждена полная цепочка получения |
| Вся карта одним Lua-запросом | Метод не найден |
| Полная heightmap с её исходным разрешением | Есть точечная высота; экспорт полной сетки не найден |
| Все полигоны навигационной сетки и стоимость переходов | Экспорт не найден |
| Полная коллизионная геометрия скал, зданий и деревьев | В проверенном Lua API не найдена |
| Точные полигоны лесов и водоёмов в мировых координатах текущего боя | Есть тип поверхности и файловые структуры; готовый runtime-экспорт не подтверждён |
| Глубина воды и бродов в произвольной точке | Getter не найден; не выводим только из высоты земли |
| Уклон и нормаль земли готовым запросом | Getter не найден; можно лишь вычислять по соседним высотам |
| Свободен ли произвольный отрезок для обзора/выстрела | Универсальный запрос между двумя точками не найден |
| Воздушный объём, верхняя граница полёта, препятствия на каждой высоте | Не подтверждено |
| Итоговые полигоны начальной/передовой расстановки в любом бою | Есть данные setup/ресурсов, общего подтверждённого runtime-getter нет |
| Все поверхности по вертикали: земля, мост, стена, крыша | Один запрос высоты такой структуры не возвращает |
| Точная актуальная карта пожаров растительности | Есть признаки пожара зданий; общей маски не найдено |

Созданный скриптом `convex_area` — наш многоугольник из заданных точек. Он не
извлекает полигоны из движка. Объекты capture point, spawn zone и map barrier
тоже не заменяют границы всего поля.

Источник: [Convex Areas](https://chadvandy.github.io/tw_modding_resources/WH3/battle/convex_area.html).

## 12. Что можем вычислить, но не получаем готовым

Из точечных запросов можно построить свою выборочную сетку высот, типов поверхности
и проверок областей. Из неё — оценку уклонов, перепадов высот и доли леса. Шаг сетки
выбираем сами; это потеря деталей между пробами, а не точная копия внутренней карты.
Центр и расстояния до края можно считать только после получения настоящих границ.

Формат будущей записи наблюдения должен сохранять: источник, координаты запроса,
исходное значение, единицы и статус проверки. Для выборки дополнительно нужны шаг
и область покрытия. Отсутствующее значение не заменяем нулём: нулевая высота или
пустая строка могут иметь собственный смысл. Конкретную схему нейросети здесь не утверждаем.

## 13. Что проверено без запуска

- Прочитана документация Battle, Vector, Buildings, Capture Location, Toggle System,
  Reinforcements, Camera, Common, Global и карта относящихся к местности CCO.
- Из установленного `data_script.pack` просмотрены 2258 текстовых файлов, подходящих
  по расширениям, без ошибок распаковки. Поиск выполнялся по терминам границ,
  поверхности, высоты и экспорта setup. Это не аудит всех нативных функций движка.
- Найдены штатные вызовы высоты, реализация выборки поверхности, использование `forest`
  и XML с явно заданной `playable_area`.
- Проверены индексы `terrain*.pack` без архивов текстур/кампании; обнаружены ресурсы
  нашей карты. Они не использованы как доказательство её числовых размеров.
- Прочитан код формата BMD/PlayableArea в RPFM. Это доказательство описанной структуры,
  а не испытание её извлечения из нашей карты.

Воспроизводимые метаданные: `static-evidence.json` (локальный архив: `research/evidence/map-data/static-evidence.json`).
Черновики и загруженные справочники находятся в игнорируемом `tmp/map-research/`.
Этот документ сохраняет итог исследования и ограничения; папка утверждённых входов
не расширена новыми игровыми признаками.
