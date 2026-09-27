# map

[Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/map.md)

Чтение карты: рамка мини-карты (радара), сетка клеток, высота, грунт,
проходимость клетки, здания и сооружения. Код: `src/apps/map/`.
Методы и измеренные пределы — в [знаниях о карте](../game/map/README.md).

## services (без движка)

| Функция | Что делает |
|---|---|
| `build_frame(query)` | Рамка радара по функции `query(x, z) → u, v`. Проверяет углы; повёрнутый радар отвергается как непроверенный |
| `world_to_radar(frame, x, z)` | Мировые координаты → координаты радара 0..1 |
| `new_grid(frame, step)` | Сетка с шагом `step` м: `columns`, `rows`, `count` |
| `cell_center(grid, ix, iz)`, `cell_index(grid, id)` | Центр клетки; номер → `ix, iz` |
| `world_to_cell(grid, x, z)` | Клетка точки или `nil` за рамкой (верхняя граница не входит) |
| `inside_radar(grid, x, z)` | Лежит ли центр клетки в рамке |

## adapter

| Функция | Что делает |
|---|---|
| `read_frame(common)` | `build_frame` по запросам `BattleRadarPosition(...)` |
| `vector(battle_vector, x, y, z)` | Вектор движка |
| `read_batch(bm, battle_vector, grid, start, limit)` | Клетки: `height`, `clear`, `ground`, `inside_radar`; возвращает `cells, next, done` |
| `read_buildings(bm, start, limit)` | Здания движка (индексы с 1): позиция и центр, категория, здоровье, ворота/стены/башни |
| `read_structure_contexts(common, start, limit)` | Отдельный список CCO (индексы с 0): мост, разрушаемость, эффекты |

## Пример

```lua
local map = require('apps.map.adapter')
local geometry = require('apps.map.services')

local frame = map.read_frame(common)
local grid = geometry.new_grid(frame, 3)
local cells, next_id, done = map.read_batch(bm, battle_vector, grid, 0, 4096)
```

## Ограничения

- Рамка радара — **предполагаемая** граница, не проверенная граница движения
  (`movement_boundary_verified = false`).
- Позиции зданий — точки начала/центра, не контур.
- Индексы CCO и движка — разные списки; не сопоставлять их по номеру.
- Названия и тексты эффектов CCO локализованы: это не идентификаторы.
- Большие сетки читаются пачками через `bm:real_callback`, см.
  [map_capture](entries.md#map_capture).
