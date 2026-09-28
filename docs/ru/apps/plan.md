# plan

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/plan.md)

**Узлы дерева:** [Расстановка](../tree/deploy.md)

План начала боя: цепочка [оценка сил](assessment.md) → [стратегия](strategy.md) →
[строй](formation.md). Только вызывает модули по порядку и передаёт данные
дальше; решения принимают модули. Код: `src/apps/plan/`.

`start(input, modules?)`, где `input = {role, bearing, own = {anchor, units},
enemy = {units, lord?, blocks?}, tree?, formation_params?, fits?}`. `tree` — переключатели
[дерева](tree.md): ветка `map_fit` решает, передавать ли строю `fits` (карту), ветка
`formation_window` — видимых врагов `enemy.blocks` (окно при выборе стены). Возвращает `{status, features,
decision, formation}`; если ни одна стратегия не подходит — `status = 'no_strategy'`
и строй не строится (в бою это ошибка, а не догадка). `formation_params` —
только для опытов и тестов. `modules` подменяются в тестах шпионами
(`tests/apps/plan/`).

Одна и та же цепочка работает в бою (`entries.formation_probe`) и в симуляции
(`tools/sim/formation.py`).
