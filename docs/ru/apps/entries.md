# Точки входа

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/entries.md)

Точка входа связывает приложения в конкретный бой: подписывается на фазы,
запускает тик, вызывает адаптеры и сервисы, пишет телеметрию. Логики решений
в ней нет. Код: `src/entries/`. Каждая точка входа — модуль с функцией
`main(bm, config, globals)`, её вызывает собранный скрипт.

<a id="duel"></a>

## duel — дуэль с автопереигровкой

Сборка: `python -m tools.build duel --runs 3`. Сценарий `ranged_melee.xml`:
по одному отряду коссаров, скрипт-имена `bai_ranged_a` / `bai_ranged_b`.
Если имена другие, дуэль допускает ручной бой «один пеший лорд на сторону».

1. `loaded` → проверка типа боя → чтение сторон → `ready`.
2. Фаза `Deployment`: через 1 с автоматически завершить расстановку.
3. Фаза `Deployed`: перечитать `tww3_bai_policy.lua` (если есть), выставить
   скорость, взять управление, `prepare_melee`.
4. Тик: результат боя → тайм-аут → подтверждение управления → решение
   политики для каждой стороны → приказ → `decision` и `snapshot`.
5. `result`: `completed` (победитель от движка), `timeout` (принудительная
   ничья, не «естественная»), `incomplete`.
6. Если боёв в серии ещё осталось: записать `tww3_bai_pending.txt`,
   завершить бой, нажать «переиграть» через UI API и подтвердить.

События: `loaded`, `ready`, `initial_unit`, `deployment`, `start`,
`control_requested`, `control_acquired`, `decision`, `snapshot`,
`final_unit`, `result`, `restart_*`, `error`, `skipped`.

<a id="arena"></a>

## arena — три пары на одной карте

Сборка: `python -m tools.build arena`. Сценарий `triple_melee.xml`: пары
`bai_arena_<пара>_<сторона>` — коссары 120, царская гвардия 100, снежный
леопард 1. Управление берётся ещё до расстановки; после неё отряды
возвращаются на точки из XML. Пара завершается при первом бегстве или
гибели. Если отряд другой пары ближе 120 м — `isolation_warning`.
Мораль армии общая для всех пар; это отмечено в каждом результате.

<a id="ai_vs_ai"></a>

## ai_vs_ai — бой двух штатных ИИ

Сборка: `python -m tools.build ai-vs-ai --speed 3`. Сценарий `ai_vs_ai.xml`:
по 4 отряда Кислева на сторону (2 коссара, царская гвардия, крылатые уланы).
Наш код не отдаёт ни одного приказа отрядам.

- Армия, которую движок считает армией ИИ, остаётся **общему боевому ИИ**.
- Армия, которую движок считает армией игрока (`army:is_player_controlled()`),
  передаётся штатному планировщику ИИ ([planner_adapter](orders.md)) с задачей
  «атаковать вражескую армию»; задача повторяется каждые 15 с.
- Кто кем управляет — событие `ai_assigned` (`general_battle_ai` / `script_ai_planner`).
- Каждые 5 тиков: `snapshot` по отрядам и `progress` (бойцы и стоящие отряды
  по сторонам); в конце `final_unit` и `result`.

Первый запуск 27.09.2026 (ещё без True Sight): загрузка ~30 с, обе армии
сошлись сами; к 237-й секунде у стороны 1 оставалось 258 бойцов и 4 отряда,
у стороны 2 — 147 и один отряд (игру закрыли до `result`).

<a id="unit_readout"></a>

## unit_readout — все показатели отрядов и видимость

Сборка: `python -m tools.build unit-readout --speed 20`. Сценарий
`unit_readout.xml` на карте **The Moorlands Route** (`catchment_03`): Империя
против Империи, у каждой стороны генерал, копейщики, лучники, засада
копейщиков в северном лесу и охотники со Stalk на открытой траве.

Этапы (игровые секунды): покой 0 → марш 8 → стрельба 30 → прекращение огня 75 →
рукопашная 85 → разведка стороны 1 на 80/40/15 м (140/148/156) → разведка
стороны 2 (164/172/180) → остановка 188 → конец 196. Каждую секунду:

| Событие | Что это |
|---|---|
| `side_view` ×2 | [Сводка стороны](observation.md): свои целиком, о врагах только разрешённое, дальность своих стрелков |
| `full_view` | Полная сводка (эталон, видит скрытых) |
| `enemy_gate` (раз в 5 с) | Сенсор состояния со стороны противника: только видимость |
| `unit_profile` (один раз) | Тип, бойцы, дальность, атрибуты (в т. ч. `stalk`), режимы, способности, ранг |

Строки копятся в памяти и пишутся одним открытием файла за тик (~1 100 строк
за бой вместо ~16 000 в первой версии, которая тормозила игру и launcher).

Отчёты:

```bash
.venv/Scripts/python -m tools.analysis.unit_readout build/unit-readout/runs/<время>
.venv/Scripts/python -m tools.analysis.unit_readout build/unit-readout/runs/<время> --view side --side 1
```

Результат 27.09.2026 (`20260927-142521`, True Sight, ×20, 196 с игры ≈ 10 с реального
времени): полная сводка **168 / 168**, сводка стороны 1 **14 / 14**, стороны 2
**14 / 14**. Замеры скрытности — [видимость](../game/units/visibility.md).

<a id="move_probe"></a>

## move_probe — движение одного отряда

Сборка: `python -m tools.build move-probe --plan hamlet`. Сценарий
`move_probe.xml` на **The Moorlands Route** (`catchment_03`): отряд
`probe_spears` (копейщики Империи) и далёкий генерал `far_general`, которого
держит скрипт. Задача — [обход препятствий](../architecture/tasks/obstacles.md).

План `config/move-plans/<имя>.json` — список заходов:

- `shape` — перестроиться на месте в другую ширину;
- `traverse` — один обычный приказ в дальнюю точку.

Заход: телепорт в старт, 2 с на успокоение, один
`goto_location_angle_width`, затем каждую секунду `move_sample` (движение
отряда и позиции всех бойцов в дециметрах) до `leg_end` с причиной
`arrived`, `stopped`, `stuck` или `timeout` (`apps.navigation.services`).

Разбор: `tools/analysis/move_probe.py`; итоги — [хутор](../../../research/analysis/hamlet/README.md).

<a id="roster_capture"></a>

## roster_capture — сбор ростера

Сборка: `python -m tools.build roster-capture` — сценарий `roster_capture.xml`
строится из `config/roster/capture.json`. В расстановке для каждого отряда
пишется `unit_card` (карточка, профиль); затем все отряды больше чем из одного
бойца одновременно перестраиваются на своих местах в каждую ширину из списка —
`shape_result` с позициями бойцов и временем. Итог — `tools/roster.py update`,
см. [ростер отрядов](../game/units/roster.md).

<a id="formation_probe"></a>

## formation_probe — строй в бою

Сборка: `python -m tools.build formation-probe --army first_attack` — сценарий и
данные из `config/armies/` и ростера (`tools/sim/formation.py`). После
расстановки враг ставится по симуляции и стоит; наш строй считает в бою
`apps.formation` по видимым врагам и ставит телепортом.
После расстановки армия стоит `hold_s` (60 с) без приказов — проверка «не метаться»
(`hold_sample` каждый тик, вердикт `stability` в разборе). Повороты лучников — только
с `--turn-test`: вправо и влево движковым `rotate` и нашим поворотом на месте.
`plan`, `stage_snapshot` (все бойцы на каждом этапе), `turn_sample` (лучники
во время поворота). Разбор — `tools/analysis/formation_probe.py`.

<a id="manual"></a>

## manual_record — ручной бой с записью

Сборка: `python -m tools.build manual --deadline 3600 --stall-minutes 30`.
Сценарий `manual_hamlet.xml`: 6 отрядов копейщиков под управлением игрока у
хутора. Скрипт не командует ими, только пишет: каждую секунду `own_sample`
(движение и бойцы всех отрядов), при смене заданной точки — `order_seen`,
конец приказа — `order_end` (`arrived`, `stopped`, `stuck`, `timeout`).
Бой заканчивает игрок; иначе — через час или после 30 минут игры без урона.

<a id="map_capture"></a>

## map_capture — сбор карты

Сборка: `python -m tools.build map-capture --step 3 [--features]`.
Сценарий `map_capture.xml`, файлы с префиксом `tww3_bai_map_capture_`.

1. Через 3 с после загрузки: пауза, `ready.xml`, рамка радара (`frame`,
   `corner`), сетка (`grid_begin`).
2. Пачки по 4096 клеток → `grid.csv`:
   `ix,iz,x,z,height,clear,ground,inside_radar`.
3. С `--features`: взять оба отряда под контроль, завершить расстановку на
   ×20, поставить паузу, затем здания (`building`), список CCO
   (`structure_context`) и достижимость клеток для обеих сторон
   (`reach_side_1`, `reach_side_2` в CSV).
4. `probe_done` или `probe_error`.

Обработка сетки без игры — [tools/analysis](../../../tools/analysis/):
`heightmap.py`, `slopes.py`, `passages.py`. Методика — [карта](../game/map/README.md).

## Общее

- **Бой всегда заканчивается сам.** Три предохранителя:
  срок в реальном времени `deadline_s` (`battle.deadline`, результат `deadline`);
  застой — никто не получает урона `stall_ms` игрового времени, по умолчанию
  10 минут (`battle.services.new_stall_detector`, результат `stalled`);
  лимит launcher = 240 с на загрузку + срок каждого боя. Параметры сборки:
  `--deadline`, `--stall-minutes`.

- **Скорость.** Когда исход решён и отряды бегут, движок сам сбрасывает
  скорость до ×1. `battle.speed_guard` каждые 0,5 с возвращает заданную
  скорость до фазы `Complete` и пишет `speed_restored` (проверено 27.09.2026
  на ×20: сброс 1 → 20 сразу после `result`). Пауза (скорость 0) не трогается.

- Защита от повторной загрузки: глобальный флаг (`tww3_bai_duel`,
  `tww3_bai_arena`, `tww3_bai_map_capture`).
- Все колбэки — через `errors.guard` ([ошибки](../architecture/error_handling.md)).
- Проверка без игры — `tests/entries/test_entries.py` на поддельном `bm`.
