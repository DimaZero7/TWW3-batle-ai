# strategy

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/strategy.md)

Выбор стратегии. Чистый код: список стратегий, у каждой — условия, балл, роли
отрядов и расстановка строя, которую она просит. Читает только признаки
[оценки сил](assessment.md); о геометрии ничего не знает. Описания стратегий —
[список стратегий](../architecture/strategies.md). Код: `src/apps/strategy/`.

| Функция | Что делает |
|---|---|
| `CATALOG` | Список стратегий. Сейчас одна: `wall_and_arc` — шесть условий, расстановка `line_and_blocks` |
| `select(features)` | Проверяет условия всех стратегий; из подходящих — лучший балл. Возвращает ключ (или `none`) и **все** кандидаты с условиями — это журнал выбора |
| `assign_roles(entry, units, category)` | Роли отрядов для стратегии: `wall`, `arc`, `lord`, `other` |
| `decide(features, units, category)` | Решение: стратегия, расстановка, её настройки, роли, кандидаты |
| `contract.check_decision(d, units)` | Договор со строем: у выбранной стратегии есть расстановка и у каждого отряда — известная роль |

Условия `wall_and_arc`: `enemy_infantry_stronger`, `own_infantry_to_hold`,
`more_archers`, `archers_pierce` (средняя броня вражеской пехоты ≤ 50),
`arc_fire` (навесом бьют не меньше половины стрелков), `we_attack`.
