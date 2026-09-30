# Lua map reader and capture tools

[← Back](README.md) · [Map guide](README.md) · [Русский](../../../ru/game/map/tools.md)

## Components

| File | Responsibility |
|---|---|
| [src/apps/map/adapter.lua](../../../../src/apps/map/adapter.lua) | Read the radar frame (`read_frame`), sample cells in batches (`read_batch`), read native objects and CCO structures in batches (`read_buildings`, `read_structure_contexts`); [verified scope](objects.md) |
| [src/apps/map/services.lua](../../../../src/apps/map/services.lua) | Geometry without the engine: frame, grid and cell coordinates (`new_grid`, `world_to_radar`, `world_to_cell`, `cell_center`) |
| [src/apps/navigation/adapter.lua](../../../../src/apps/navigation/adapter.lua) | Query supplied units for sampled points after deployment (`read_cells`); [limits](passages.md) |
| [slopes.py](../../../../tools/analysis/slopes.py) | Derive geometric slopes and neighbour-height differences offline |
| [passages.py](../../../../tools/analysis/passages.py) | Find shallow-water connection candidates in saved reachability grids |
| [src/entries/map_capture.lua](../../../../src/entries/map_capture.lua) | Entry point: static test runner, pause, callbacks, CSV and JSONL output; the `--features` mode adds a controlled start, native/CCO objects and full point-reachability capture |
| [heightmap.py](../../../../tools/analysis/heightmap.py) | Offline CSV-to-height-matrix and PNG conversion; [instructions](heights.md) |
| [build.py](../../../../tools/build.py) | Bundle the map modules, the entry and a fixed test scenario into an isolated pack |
| [launch.ps1](../../../../tools/launcher/launch.ps1) | Verify/install the pack, start one process, watch completion, preserve output, close the owned process and remove its private game-side files |
| [map_capture.xml](../../../../scenarios/map_capture.xml) | Vanilla Kislev classic test setup; two minimal opposing units needed to load the battle |

The map and navigation modules have no unit commands, timers, files or global
initialization. They are local Lua modules, not a network service. The
`entries.map_capture` entry owns the execution lifecycle; the modules enter another
target's build only if that target's entry requires them.

## Module interface

The builder (`tools/build.py`) bundles the modules from `src/` into one script
through wrapper functions. No unverified game `require` path is assumed.

```lua
local map = require('apps.map.adapter')
local map_services = require('apps.map.services')
local navigation = require('apps.navigation.adapter')

local frame = map.read_frame(common)
local grid = map_services.new_grid(frame, 5)
local cells, next_index, done = map.read_batch(bm, battle_vector, grid, 0, 4096)
local u, v = map_services.world_to_radar(frame, 100, 0)
local ix, iz = map_services.world_to_cell(grid, 100, 0)
local x, z = map_services.cell_center(grid, ix, iz)
local rows = navigation.read_cells(bm, battle_vector, units, cells)  -- after deployment
```

- `frame`: coefficients, validated corners, `min_x/max_x/min_z/max_z`, width/depth,
  centre, `kind="radar_frame"`, `movement_boundary_verified=false`.
- `grid`: origin, `step`, columns/rows/count, `radar_max_x/z`, separate `query_max_x/z`.
- `cells`: `ix, iz, x, z, height, clear, ground, inside_radar`. The last flag tests
  the centre against the frame, not whether the entire square fits inside it.
- `next_index`: zero-based linear index for the following batch. Repeat until `done`.
  Schedule between batches in the caller; do not run a huge blocking loop in combat.

Invalid types, nonfinite results, unsupported radar orientation and corner mismatch
raise errors. The runner catches errors into `probe_error`; they are never converted
into a green cell or successful capture. No use of `math.huge` is required: it was
missing in the measured game Lua environment.

## Run from the repository root

Requirements: installed WH3, the project's Python (`.venv` with the dependencies
from `requirements-dev.txt`), PowerShell, and write access to the game folder. The
game path is in `config/default.json` or `config/local.json`.

```bash
.venv/Scripts/python -m pytest tests/apps/map
```

```bash
.venv/Scripts/python -m tools.build map-capture --step 5
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target map-capture
```

Grid steps: `1`, `2`, `3`, `5`; `--window MIN_X MAX_X MIN_Z MAX_Z` captures only a
part of the map. The game must be closed. Capture is static and paused; no mouse
automation is needed.

Output: `build/map-capture/runs/<YYYYMMDD-hhmmss>/`: the build manifest, launch
identity, events, exported battle XML, grid CSV and status with the cleanup result
([running a battle](../../launch/run.md)).

**The launcher closes its own game process after saving.** Error/timeout paths also
stop only its identified process. A running game or files of a previous run in the
game folder block a new launch: inspect and remove them by hand. Leave a game open
(`-KeepGameOpen`) only when the user explicitly needs manual control.

## Complete field-feature capture

The optional `--features` mode adds native objects, [structure/bridge contexts](bridges.md), and point-reachability columns for the first unit of each of two opposing armies. Use the preserved two-unit diagnostic scenarios. This runner takes control of both units, disables firing, halts them, advances Deployment, and pauses after Deployed. The default surface-only mode remains in Deployment.

Reproduce the validated bridge-location capture (its XML is in the local archive):

```bash
.venv/Scripts/python -m tools.build map-capture --step 3 --features --scenario ../research/evidence/map/field-20260925/bridge-navigation/map_probe.xml
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target map-capture
```

`--scenario` takes a path relative to `scenarios/` and changes the bundled XML; it does not select a map by localized menu name. Source XML, bundled Lua and hashes are preserved in the run directory. Keep custom captures separate from a normal campaign or manual battle.

Additional CSV columns: `reach_side_1`, `reach_side_2` — 1 true, 0 false, −1 not queried outside the frame. JSONL contains `building`, `structure_context`, `navigation_unit`, `feature_conditions`, counts and timings. The unit events establish which units the columns refer to. Read the text as **UTF-8**, including on Windows; localized CCO names are not ASCII.

For the passage tool, select the new columns explicitly:

```bash
.venv/Scripts/python -m tools.analysis.passages --csv build/map-capture/runs/<time>/tww3_bai_map_capture_grid.csv --step 3 --reach-columns reach_side_1 reach_side_2 --output build/map-research/passages
```

This identifies candidates only; a capture containing a bridge need not contain a ford candidate. In the live regression, **116,964 rows**, including all surface and reachability values, **2,696 native records**, and **one CCO record** matched the research capture. Navigation sampling and logging took **2.430 s** by Lua `os.clock()`; loading was separate. The whole run completed in approximately **36.7 s**, then the owned game process closed. `Validation` (local archive: `research/evidence/map/field-20260925/feature-runner/validation.json`) · `Run status` (local archive: `research/evidence/map/field-20260925/feature-runner/status.json`).

## Verification status

Seven Lua 5.1 tests cover saved radar points from both maps, invalid queries,
corner checks, four grid sizes, edge indexing and batched sampling. These tests
do not simulate WH3 terrain or navigation.

The extracted reader and capture runner were then exercised live on Kislev at 5 m:
**42,025/42,025 cells matched the earlier capture** in indices, height to six CSV
decimals, restriction flags and ground type; world centres also matched.
`Evidence` (local archive: `research/evidence/map/module-check-20260925/comparison.json`).
The original experiment ran all four steps live; the extracted module's dedicated
live regression covered 5 m. Do not silently extend either claim to other maps or combat.

That first module run used the same `finish.ps1` cleanup manually; automatic calling
and error cleanup were added following the user's closure preference. Preserve that
distinction in validation history instead of labelling every path live-tested.
