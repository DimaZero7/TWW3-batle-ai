# Стоп под обстрелом

[← Назад](README.md) · [Документация](../README.md) › [Дерево ИИ](README.md) › Стоп под обстрелом · [English](../../en/tree/under_fire_stop.md)

Ветка сближения: единственное исключение из «одного действия за раз». Если наш
отряд под обстрелом, манёвр прерывается и ничего нового не начинается — дальше
решают фазы боя.

## Где в дереве

<!-- generated:tree:here:under_fire_stop -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Основа<br/>18 модулей"]:::done
  deploy["1 · Расстановка<br/><i>Стратегический</i>"]:::done
  approach["2 · Сближение до окна<br/><i>Тактический</i>"]:::done
  under_fire_stop["Стоп под обстрелом<br/>under_fire_stop"]:::done_b_h
  base --> deploy
  deploy --> approach
  under_fire_stop -.- approach
```
<!-- /generated -->

## Карточка

<!-- generated:tree:card:under_fire_stop -->
| | |
|---|---|
| Что делает | Под обстрелом манёвр прерывается и ничего нового не начинается |
| Когда начинается | наш отряд под обстрелом |
| Без него | «одно действие за раз» и под обстрелом |
| Переключатель | `under_fire_stop` (вкл) |
| Вид, уровень | ветка, Тактический |
| Растёт из | [Сближение до окна](approach.md) |
| Модули | [tactics](../apps/tactics.md), [approach](../apps/approach.md) |
| Статус | готово |
<!-- /generated -->

## Как работает

```mermaid
sequenceDiagram
  participant G as игра
  participant T as tactics
  G->>T: скачок идёт
  Note over G: отряд под обстрелом
  G->>T: interrupt(под обстрелом)
  T-->>G: да — прервать
  G->>G: стоп всем отрядам
  G->>T: новая картина боя
  T-->>G: стоять (under_fire)
```

## Пример

Без ветки (первый бой 28.09.2026): выравнивание началось под обстрелом и шло 6
минут — скрипт ждал, армию разбили. С веткой: манёвр прерван сразу, 59 с на месте —
мы −26, они −75.

| | Без ветки | С веткой |
|---|---|---|
| Манёвр под обстрелом | ждали конца 6 мин | прерван сразу |
| Итог | армия разбита | мы −26, они −75 за 59 с |

## Проверки

- `tests/apps/approach/` — командир под обстрелом ничего не начинает;
  `tests/apps/tactics/` — прерывание только с веткой;
- ветка выключена = «одно действие за раз»: `tests/tools/test_tree_branches.py`.

Модули: [tactics](../apps/tactics.md) · [approach](../apps/approach.md).
