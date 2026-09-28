# mask

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/mask.md)

"Can we stand here?" mask over the [battlefield](battlefield.md): a grid in the
battlefield frame, 3 m cells (user's decision). A cell is standable when the area
is clear (`is_area_clear`) and our infantry can reach it (`can_reach_position`).
Forest and water are not treated specially yet. For the simple approach.
Code: `src/apps/mask/`.

| Function | What it does |
|---|---|
| `grid(field, step?, pad?)` | The grid over the field (+10 m at both ends along the axis) |
| `fill(mask, reader, first?, count?)` | Fill cells from `reader(x, z) → clear, reachable` (engine in batches / captured map) |
| `fits(mask, placement)` | Does a unit fit at a placement |
| `lane(mask, across, width, from, to)` | Is a lane along the axis free; first blocked point (unread cells block) |
| `stand_at`, `summary`, `encode` | Point answer, counts, telemetry string |

Checked by tests, in the simulation (`hamlet_attack`: the hamlet is 241 blocked
cells of 6120; the lane of our front clips its corner) and in battle: 249 blocked
cells, **99.8%** agreement with the captured map, the whole mask read in **0.03 s**
of CPU ([comparison](../../../research/analysis/mask/README.md), in Russian).

Watch it in motion: [battle viewer](../launch/viewer.md), layer “mask” (preset `debug`).
