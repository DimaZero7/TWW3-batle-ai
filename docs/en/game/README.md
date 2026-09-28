# Game knowledge

[← Back](../README.md) · [Documentation](../README.md) · [Русский](../../ru/game/README.md)

What is verified in Total War: WARHAMMER III (v9.0.0, build 50218.4334952,
September 2026) and how it was measured. Only game facts and methods live
here; the code that uses them is described in [apps](../architecture/apps.md).

| Section | Contents |
|---|---|
| [Map](map/README.md) | Coordinates, radar frame, grid, heights, forest, water and ground, objects, passages, bridges, slopes, tools, evidence |
| [Map catalogue](maps/README.md) | Cards of specific maps: [Moorlands Route](maps/moorlands-route.md) |
| [Units](units/README.md) | State indicators, fields, states, missile range, commands, deployment, evidence |
| [Unit catalogue](units/catalog/README.md) | [Empire](units/catalog/empire.md): identifiers, cost, static stats |
| [Readout catalogue](readouts.md) | Everything readable about units and the battle: tracked and not yet; run profiles |
| [Visual atlas](atlas.md) | The key research images |

## Data available through Lua

- **Map**: point height, world coordinates and grid, ground type, water and
  forest, static objects, point reachability for a specific unit.
- **Own units**: type, position, men, health, movement, combat, routing,
  ammo, range, orders and some formation parameters, CCO card fields.
- **Enemy**: only the current visible state and the last known position.
- **Commands**: move, run, attack a visible target, halt, guard, formation
  and width, some abilities.

Unavailable or incomplete: the exact internal morale reserve, fresh physical
bounds of every entity, hidden state of an invisible enemy, execution
guaranteed merely by an accepted order, a universal passability map without
a unit-specific check.

## Old module names in these pages

The pages were moved from the research set and sometimes use old module
names. Mapping:

| In the text | Now |
|---|---|
| `src/map/reader.lua`, `reader.read_frame`, `reader.cell_center` | [map](../apps/map.md): `map.adapter.read_frame`, `map.services.cell_center` |
| `src/map/objects.lua` | [map](../apps/map.md): `read_buildings`, `read_structure_contexts` |
| `src/map/reachability.lua` | [navigation](../apps/navigation.md): `read_cells` |
| `src/units/state.lua`, `src/units/range.lua` | [units](../apps/units.md): `state_adapter`, `range_adapter` |
| `src/visibility/reader.lua` | [intel](../apps/intel.md) |
| `tools/map-capture/capture.lua`, `features.lua` | [map_capture](../apps/entries.md#map_capture) |
| `tools/map-capture/build.py`, `launch.ps1`, `finish.ps1` | `python -m tools.build map-capture`, `tools/launcher/launch.ps1` |
| `tools/map-capture/heightmap.py`, `slopes.py`, `passages.py` | `tools/analysis/` |
| `evidence/...` next to a page | `research/evidence/...` |

Links inside the pages already point to the new locations. Links to the
excluded tournament material are kept as plain text.
