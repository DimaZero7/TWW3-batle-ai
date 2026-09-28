# Логистика у препятствия

[← Назад](README.md) · [Документация](../README.md) › [Дерево ИИ](README.md) › Логистика · [English](../../en/tree/logistics.md)

Ветка сближения: когда скачок идёт в обход препятствия, отряды проходят его
очередями, а не толпой. По умолчанию **выключена** — обход делает движок.

## Где в дереве

<!-- generated:tree:here:logistics -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_b_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4,stroke-width:4px
  base["Основа<br/>18 модулей"]:::done
  deploy["1 · Расстановка<br/><i>Стратегический</i>"]:::done
  approach["2 · Сближение до окна<br/><i>Тактический</i>"]:::done
  logistics["Логистика у препятствия<br/>logistics"]:::done_b_h
  base --> deploy
  deploy --> approach
  logistics -.- approach
```
<!-- /generated -->

## Карточка

<!-- generated:tree:card:logistics -->
| | |
|---|---|
| Что делает | Очереди у препятствия: кто обходит, какой стороной, в каком порядке и ширине |
| Когда начинается | скачок идёт в обход препятствия |
| Без него | обход препятствия делает движок |
| Переключатель | `logistics` (выкл) |
| Вид, уровень | ветка, Тактический |
| Растёт из | [Сближение до окна](approach.md) |
| Модули | [logistics](../apps/logistics.md), [mask](../apps/mask.md) |
| Статус | готово |
<!-- /generated -->

## Как работает

```mermaid
flowchart LR
  d["скачок в обход<br/>препятствия"] --> band["полоса: препятствие + 5 м,<br/>кто её пересекает"]
  band --> side["стороны: кто обходит<br/>слева, кто справа"]
  side --> q["очередь в каждой полосе,<br/>без обгонов"]
  q --> n{"отряд шире<br/>прохода?"}
  n -- да --> narrow["сузиться на месте"]
  n -- нет --> go["идти, когда передний<br/>отошёл"]
  narrow --> go
```

## Пример

Длинная стена у скалы (16 отрядов): очереди с обеих сторон, без давки.

![Логистика у скалы](../../../research/analysis/logistics/rock_march_wide/walk_1.png)

В движении, с очередью и все сразу рядом: `python -m tools.viewer logistics rock_march_wide` — [проигрыватель боя](../launch/viewer.md#обход-препятствия-очередями).

## Проверки

- `tests/apps/logistics/` — полоса, стороны, очереди, сужение;
- игра 27.09.2026: 5 местностей, в том числе скавены; задача доработки — № 18.

Модули: [logistics](../apps/logistics.md) · [mask](../apps/mask.md).
