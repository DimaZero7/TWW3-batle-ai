# navigation

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/navigation.md)

Достижимость точек для конкретных отрядов. Код: `src/apps/navigation/`.
Как искали броды и проходы — в [проходах](../game/map/passages.md) и
[мостах](../game/map/bridges.md).

## adapter

`read_cells(bm, battle_vector, units, cells)` — для каждой клетки в рамке
радара вызывает `unit:can_reach_position(p)` у каждого отряда. Работает
только в фазе `Deployed`. Возвращает `{ix, iz, inside_radar, reachable = {bool, ...}}`.

## diagnostics_adapter

Диагностика вокруг `can_reach_position`, только для своих отрядов; в
решения не попадает.

| Функция | Что делает |
|---|---|
| `state(bm, unit)` | Фаза, позиция, заданная позиция, направление, ширина, движение, управление, развёртывание |
| `gate(bm, unit, p, native_return)` | Что спросили, что ответил движок, в каком состоянии был отряд |
| `origins(bm, unit, make_vector)` | Достижимость собственной позиции отряда и той же X/Z с высотой рельефа |

Ответы — `{status = 'ok' | 'unavailable' | 'unsupported', ...}`.

## Ограничения

- `true` для точки не доказывает, что свободна прямая к ней.
- Ответ относится к конкретному отряду и его текущему положению:
  кавалерия и пехота могут отличаться.
- Проходы и броды по сетке — кандидаты. Движение через них проверяется
  отдельно ([tools/analysis/passages.py](../../../tools/analysis/passages.py)).
