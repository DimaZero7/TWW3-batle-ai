# Обход не под стрелками

[← Назад](README.md) · [Документация](../README.md) › [Дерево ИИ](README.md) › Обход не под стрелками · [English](../../en/tree/safe_detour.md)

Ветка сближения: если строй не помещается в конце скачка, место за препятствием
ищется **только там, куда их стрелки не достают**.

## Где в дереве

<!-- generated:tree:here:safe_detour -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Основа<br/>18 модулей"]:::done
  deploy["1 · Расстановка<br/><i>Стратегический</i>"]:::done
  approach["2 · Сближение до окна<br/><i>Тактический</i>"]:::done
  safe_detour["Обход не под стрелками<br/>safe_detour"]:::done_b_h
  base --> deploy
  deploy --> approach
  safe_detour -.- approach
```
<!-- /generated -->

## Карточка

<!-- generated:tree:card:safe_detour -->
| | |
|---|---|
| Что делает | Место за препятствием ищется только вне досягаемости их стрелков |
| Когда начинается | строй не помещается на месте скачка |
| Без него | место ищется до 20 м от их фронта |
| Переключатель | `safe_detour` (вкл) |
| Вид, уровень | ветка, Тактический |
| Растёт из | [Сближение до окна](approach.md) |
| Модули | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md) |
| Статус | готово |
<!-- /generated -->

## Как работает

```mermaid
flowchart LR
  s["строй не помещается<br/>в конце скачка"] --> search["искать дальше по оси<br/>каждые 3 м"]
  search --> lim{"не дальше:<br/>их досягаемости<br/>и 20 м до их фронта"}
  lim -- нашлось --> step["скачок туда<br/>(обходит движок)"]
  lim -- нет --> stop["стоп: места нет"]
```

Без ветки поиск идёт до 20 м от их фронта — и место за препятствием может оказаться
прямо под их лучниками.

## Пример

Скала на пути, близко к их строю (`rock_attack`):

| С веткой: стоп перед скалой | Без ветки: встали за скалой, под их лучниками |
|---|---|
| ![С веткой](../../assets/tree/rock_attack.png) | ![Без ветки](../../assets/tree/rock_attack_no_safe_detour.png) |

## Проверки

- ветка выключена = старое поведение (за скалой): `tests/tools/test_tree_branches.py`;
- эталоны `rock_attack`, `rock_attack_wide`.

Модули: [tactics](../apps/tactics.md) · [approach](../apps/approach.md) · [reach](../apps/reach.md).
