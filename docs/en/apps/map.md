# map

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/map.md)

Map reading: the minimap (radar) frame, cell grid, height, ground, cell
clearance, buildings and structures. Code: `src/apps/map/`. Methods and
measured limits are in the [map knowledge](../game/map/README.md).

## services (no engine)

| Function | What it does |
|---|---|
| `build_frame(query)` | Radar frame from `query(x, z) → u, v`. Validates corners; a rotated radar is rejected as untested |
| `world_to_radar(frame, x, z)` | World coordinates → radar coordinates 0..1 |
| `new_grid(frame, step)` | Grid with `step` m cells: `columns`, `rows`, `count` |
| `cell_center(grid, ix, iz)`, `cell_index(grid, id)` | Cell centre; id → `ix, iz` |
| `world_to_cell(grid, x, z)` | Cell of a point or `nil` outside the frame (upper edge excluded) |
| `inside_radar(grid, x, z)` | Whether a cell centre lies inside the frame |

## adapter

| Function | What it does |
|---|---|
| `read_frame(common)` | `build_frame` over `BattleRadarPosition(...)` queries |
| `vector(battle_vector, x, y, z)` | Engine vector |
| `read_batch(bm, battle_vector, grid, start, limit)` | Cells: `height`, `clear`, `ground`, `inside_radar`; returns `cells, next, done` |
| `read_buildings(bm, start, limit)` | Engine buildings (1-based): origin and centre, category, health, gate/wall/tower |
| `read_structure_contexts(common, start, limit)` | Separate CCO list (0-based): bridge, destructibility, effects |

## Example

```lua
local map = require('apps.map.adapter')
local geometry = require('apps.map.services')

local frame = map.read_frame(common)
local grid = geometry.new_grid(frame, 3)
local cells, next_id, done = map.read_batch(bm, battle_vector, grid, 0, 4096)
```

## Limits

- The radar frame is an **assumed** boundary, not a verified movement
  boundary (`movement_boundary_verified = false`).
- Building positions are origin/centre points, not footprints.
- CCO and engine indices are different lists; never match them by number.
- CCO names and effect texts are localized, not identifiers.
- Large grids are read in batches via `bm:real_callback`, see
  [map_capture](entries.md#map_capture).
