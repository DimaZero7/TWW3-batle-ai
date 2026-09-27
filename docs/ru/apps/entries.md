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

- Защита от повторной загрузки: глобальный флаг (`tww3_bai_duel`,
  `tww3_bai_arena`, `tww3_bai_map_capture`).
- Все колбэки — через `errors.guard` ([ошибки](../architecture/error_handling.md)).
- Проверка без игры — `tests/entries/test_entries.py` на поддельном `bm`.
