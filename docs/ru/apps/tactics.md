# tactics

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/tactics.md)

**Узлы дерева:** [Сближение до окна](../tree/approach.md) · [Обход не под стрелками](../tree/safe_detour.md) · [Стоп под обстрелом](../tree/under_fire_stop.md)

Ствол тактического уровня — фаза 2 (сближение до окна) с ветками `align`,
`safe_detour`, `under_fire_stop` ([tree](tree.md)). **Одни и те же решения в игре и в
симуляции**: точка входа в игре и `tools/sim/formation.py` только строят картину
боя, выполняют намерение и сообщают, что манёвр закончен. Код: `src/apps/tactics/`.

**Картина боя** `view`: `now_ms`, `field` ([battlefield](battlefield.md)), `current =
{anchor, bearing}` (где стоим), `enemy_main` ([vision](vision.md)), `placements` (наши
прямоугольники), `own_blocks`, `enemy_blocks` ([reach](reach.md)), `mask` (заполненная
[маска](mask.md); без неё всё помещается), `goal` (поход к точке, враг не учитывается).

**Намерение**: `decision` (`wait`, `align`, `approach`, `hold`, `blocked`), `reason`,
`target = {anchor, bearing}` для `align` и `approach`, `detour`, `window`,
`stop_gap_m`, `step`, `path`, `align` (проверка выравнивания).

| Функция | Что делает |
|---|---|
| `new(tree)` | Ствол на один бой: командир ([approach](approach.md)) и регулятор выравнивания |
| `decide(s, view)` | Окно → рубеж → шаг → выравнивание (ветка `align`) → место за препятствием (ветка `safe_detour`) → командир (ветка `under_fire_stop`); начинает манёвр, который приказывает |
| `finish(s, now, reason)` | Манёвр закончен (встали, вышло время, прерван) |
| `busy(s)` | Идёт ли манёвр |
| `interrupt(s, under_fire)` | Прервать ли манёвр под обстрелом (ветка `under_fire_stop`) |
| `on(s, name)` | Включён ли узел дерева |

**Проверено**: эталоны симулятора (15 армий) совпали с решениями до переделки;
тесты веток; в игре (`window_game`, 28.09.2026) — ошибок нет, стоим в окне, 59 с:
мы 0, они −96 бойцов.
