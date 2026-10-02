# Game knowledge

[← Back](../README.md) · [Documentation](../README.md) › Game knowledge · [Русский](../../ru/game/README.md)

What is verified in Total War: WARHAMMER III and how it was measured: v9.0.0,
build 50218 (25–26.09.2026) and v9.0.1, build 50381 (from 27.09.2026). Only game
facts and methods live here; the code that uses them is described in
[apps](../architecture/apps.md).

<!-- generated:docs:index -->
- **[Map information: working guide](map/README.md)** — This is the current instruction for collecting **map information only**
- **[Battle map catalogue](maps/README.md)** — Cards for specific official battle maps
- **[Battle mechanics](mechanics/README.md)** — A reference of how Total War: WARHAMMER III land battles work, collected from public sources on 02.10.2026: CA's blogs and patch notes, the game's key-value tab…
- **[Units](units/README.md)** — What is verified about units in the game: what can be read, which orders work and how the battle mechanics work
- [Visual atlas](atlas.md) — Pictures of the maps we measured: the official preview and our map of heights, ground and objects
- [Attacker and defender in a battle from a scenario file](battle-roles.md) — Who attacks and who defends in a battle from our own scenario file, and how to set it
- [The game's database](database.md) — The battle rules live in the game's database: morale, fatigue, hit chance, leadership, arrows
- [Battle difficulty](difficulty.md) — **All battles against the game's AI are at Normal difficulty** (the user's decision, 30.09.2026)
- [The game's own battle AI](game-ai.md) — How the game's own AI fights
- [Readout catalogue](readouts.md) — Everything readable about units and the battle: what we already collect and what the game offers but we have not tested yet
<!-- /generated -->

**Battle mechanics** are in the units section: [morale](units/morale.md) · [melee](units/melee.md) · [missile damage](units/missile-damage.md) · [pace and fatigue](units/pace.md); the unit catalogue — [Empire](units/catalog/empire.md).

**How the game's battle mechanics work** according to CA, the database and the community (a reference with sources, confidence and the conflicts with our simulator): [battle mechanics](mechanics/README.md).

How this knowledge feeds network training: [data for training](../training/README.md).

## Data available through Lua

- **Map**: point height, world coordinates and grid, ground type, water and
  forest, static objects, point reachability for a specific unit.
- **Own units**: type, position, men, health, movement, combat, routing,
  ammo, range, orders and some formation parameters, CCO card fields,
  `MoralePercent` — morale as a share of leadership.
- **Enemy**: only the current visible state and the last known position.
- **Commands**: move, run, attack a visible target, halt, guard, formation
  and width, some abilities.

Unavailable or incomplete: the morale points themselves (we compute them from
`MoralePercent` and the database rules — [morale](units/morale.md)), fresh
physical bounds of every entity, hidden state of an invisible enemy, execution
guaranteed merely by an accepted order, a universal passability map without a
unit-specific check.

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

Links inside the pages already point to the new locations.
