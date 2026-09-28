# Фаза 2 · Сближение до окна

[← Назад](README.md) · [Документация](../README.md) › [Дерево ИИ](README.md) › Сближение до окна · [English](../../en/tree/approach.md)

Подойти скачками и встать в **окне**: наш первый эшелон достаёт их пехоту, их
лучники нашу стену — нет. Решения принимает ствол `tactics` — один и тот же в
игре и в симуляции.

## Где в дереве

<!-- generated:tree:here:approach -->
```mermaid
flowchart BT
  classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef done_h fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-width:4px
  classDef done_b fill:#E1F5EE,stroke:#0F6E56,color:#04342C,stroke-dasharray:5 4
  classDef started fill:#FAEEDA,stroke:#854F0B,color:#412402
  base["Основа<br/>18 модулей"]:::done
  deploy["1 · Расстановка<br/><i>Стратегический</i>"]:::done
  approach["2 · Сближение до окна<br/><i>Тактический</i>"]:::done_h
  align["Выравнивание<br/>align"]:::done_b
  logistics["Логистика у препятствия<br/>logistics"]:::done_b
  safe_detour["Обход не под стрелками<br/>safe_detour"]:::done_b
  under_fire_stop["Стоп под обстрелом<br/>under_fire_stop"]:::done_b
  fire["3 · Перестрелка из окна<br/><i>Боевой</i>"]:::started
  base --> deploy
  deploy --> approach
  align -.- approach
  logistics -.- approach
  safe_detour -.- approach
  under_fire_stop -.- approach
  approach --> fire
```
<!-- /generated -->

## Карточка

<!-- generated:tree:card:approach -->
| | |
|---|---|
| Что делает | Скачками по 50 м к окну: наши лучники достают их, их — нас нет; одно действие за раз; строй переносится целиком, мешает скала — шаг вбок |
| Когда начинается | армия расставлена, враг виден |
| Без него | армия стоит, где расставлена |
| Переключатель | `approach` (вкл) |
| Вид, уровень | фаза, Тактический |
| Растёт из | [Расстановка](deploy.md) |
| Модули | [tactics](../apps/tactics.md), [approach](../apps/approach.md), [reach](../apps/reach.md), [battlefield](../apps/battlefield.md), [vision](../apps/vision.md), [mask](../apps/mask.md) |
| Статус | готово |
<!-- /generated -->

## Как работает

Одно решение (каждый раз, когда прошлый манёвр кончился):

```mermaid
flowchart TB
  v["картина боя: наши и их блоки,<br/>поле, маска"] --> w["reach: окно<br/>(по каждой паре блоков)"]
  w --> s["рубеж и шаг ≤ 50 м"]
  s --> f{"под обстрелом?"}
  f -- да --> hold1["стоять<br/>(ветка «стоп под обстрелом»)"]
  f -- нет --> al{"перекос > 10°<br/>или вбок > 15 м?"}
  al -- да --> align["выровняться<br/>(ветка «выравнивание»)"]
  al -- нет --> arr{"уже на рубеже?"}
  arr -- да --> hold2["стоять: фаза 3"]
  arr -- нет --> path{"строй помещается<br/>в конце шага?"}
  path -- да --> step["скачок"]
  path -- нет --> past["место за препятствием<br/>(ветка «обход не под стрелками»)"]
  past -- нашлось --> step
  past -- нет --> blocked["стоп: места нет"]
```

Кто что делает во время скачка:

```mermaid
sequenceDiagram
  participant G as игра / симулятор
  participant T as tactics (ствол)
  G->>T: картина боя
  T-->>G: намерение: скачок на 50 м
  G->>G: приказы, армия идёт
  G->>T: отряды встали (finish)
  G->>T: новая картина боя
  T-->>G: стоять: на рубеже
```

**Окно** — где встать (от переднего края нашей стены до их фронта):

```mermaid
flowchart LR
  far["дальше 104 м:<br/>никто не достаёт"] --> win["96–104 м: ОКНО<br/>мы достаём, они нет"] --> close["ближе 96 м:<br/>их лучники бьют стену"]
  style win fill:#E1F5EE,stroke:#0F6E56
  style close fill:#FAECE7,stroke:#993C1D
```

## Пример

Симуляция тестовых армий в поле (враг стоит как штатный ИИ в тестовой битве):
8 скачков, стоим на 100,7 м между фронтами, окно 96,7–104,7 м; 60 с перестрелки —
мы 0, они −254.

![Сближение до окна](../../assets/tree/window_open.png)

В игре (x20): окно по настоящим блокам — наши достают со 100 м, их лучники с 95 м;
в окне за 59 с — мы 0, они −96…−149. Штатный ИИ не стоит на месте: при нашем
подходе он перестраивается, через ≈ 100 с его копейщики идут в атаку — это уже
фазы 3–5.

Посмотреть этот бой в движении, игру и симуляцию рядом: [проигрыватель боя](../launch/viewer.md).

## Проверки

- `tests/apps/reach/`, `tests/apps/approach/`, `tests/apps/tactics/`;
- эталоны симулятора (15 армий) — одни решения до и после переноса в `tactics`;
- ветка выключена = армия стоит, где расставлена: `tests/tools/test_tree_branches.py`;
- игра: `window_game` (4 боя 28.09.2026).

Модули: [tactics](../apps/tactics.md) · [approach](../apps/approach.md) · [reach](../apps/reach.md) · [battlefield](../apps/battlefield.md) · [vision](../apps/vision.md) · [mask](../apps/mask.md).
