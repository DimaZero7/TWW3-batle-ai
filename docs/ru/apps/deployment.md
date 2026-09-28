# deployment

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/deployment.md)

Начальная расстановка по контракту `deployment-placement-v2`.
Код: `src/apps/deployment/`. Замеры — в [расстановке](../game/units/deployment.md).

## План

Массив `{unit_id, x, z, facing_deg, width_m}` — ровно по одной записи на
каждый свой отряд. Ширина не превращается в «круг резервирования»:
занимаемая площадь по ширине не угадывается.

## contract

| Функция | Что делает |
|---|---|
| `zone(z)` | Проверяет зону из XML: центр, ортонормальные оси `axis_u`/`axis_v`, половины размеров |
| `inside(p, zone, tolerance?)` | Точка внутри прямоугольной зоны |
| `context(roster, zone, copy, 'deployment-placement-v2')` | Контекст для политики: состав, зона, пределы проверки |
| `validate(plan, context)` | Полный план, свои отряды без повторов, конечные числа, `facing_deg` 0..360, ширина, точка в зоне |

## services — транзакция

1. `prepare(policies, contexts, get_phase, copy)` — обе стороны отдают планы,
   оба проверяются **до** любого применения.
2. `apply(plans, get_phase, place)` — размещение; перед каждым шагом
   проверяется, что фаза всё ещё `Deployment`.
3. `verify(plan, context, measure, pair_distance, get_phase, previous?)` —
   сверка по замеру движка. Возвращает отчёт `{valid, stable, errors, units, pairs}`.
   Коды ошибок: `ordered_anchor_mismatch` (> 0,1 м), `native_reference_outside_zone`,
   `native_position_unstable` (> 0,25 м между замерами), `native_contact_unresolved`
   (пары ближе 1 м) и др. Повторять замер раз в секунду, пока не будет два
   стабильных подряд или 10 с.

## adapter

Колбэки для `services` из проверенных вызовов: `place_callback` (телепорт),
`measure_callback` (позиция, заданная позиция, направление, ширина),
`pair_distance_callback` (`unit_distance`).

> Адаптер собран из вызовов, использованных в замерах расстановки, но в этом
> проекте в игре ещё не проверялся.

Прежний контракт v1 с кругами резервирования — только в
[архиве](../../../research/scripts/legacy-lua/runtime/deployment.lua).
