# Lua map reader and capture tools

[← Back](README.md) · [Map guide](README.md) · [Русский](../../../ru/game/map/tools.md)

## Components

| File | Responsibility |
|---|---|
| [src/map/reader.lua](../../../../src/apps/map/) | Read the radar frame, define grid coordinates, sample cells in batches |
| [src/map/objects.lua](../../../../src/apps/map/) | Read native building/prop origins and state in batches; [verified scope](objects.md) |
| [src/map/reachability.lua](../../../../src/apps/navigation/) | Query supplied units for sampled points after deployment; [limits](passages.md) |
| [slopes.py](../../../../tools/analysis/slopes.py) | Derive geometric slopes and neighbour-height differences offline |
| [passages.py](../../../../tools/analysis/passages.py) | Find shallow-water connection candidates in saved reachability grids |
| [capture.lua](../../../../src/entries/map_capture.lua) | Static test runner, pause, callbacks, CSV and JSONL output |
| [features.lua](../../../../src/entries/map_capture.lua) | Optional controlled start, native/CCO objects and full point-reachability capture |
| [heightmap.py](../../../../tools/analysis/heightmap.py) | Offline CSV-to-height-matrix and PNG conversion; [instructions](heights.md) |
| [build.py](../../../../tools/build.py) | Bundle reader + runner + fixed test scenario into an isolated pack |
| [launch.ps1](../../../../tools/launcher/launch.ps1) | Verify/install the pack, start one process, watch completion, preserve output, close automatically |
| [finish.ps1](../../../../tools/launcher/launch.ps1) | Verify saved files and process identity, close the owned process, remove its private game-side files |
| [map_capture.xml](../../../../scenarios/map_capture.xml) | Vanilla Kislev classic test setup; two minimal opposing units needed to load the battle |

The reader has no unit commands, timers, files, global initialization or AI policy.
It is a local Lua service/module, not a network service. The test runner owns the
execution lifecycle. Nothing is installed into the normal duel pack automatically.

## Module interface

The builder bundles `reader.lua` as a local returned table, using the same wrapper
pattern as the existing harness. No unverified game `require` path is assumed.

```lua
local frame = reader.read_frame(common)
local grid = reader.new_grid(frame, 5)
local cells, next_index, done = reader.read_batch(bm, battle_vector, grid, 0, 4096)
local u, v = reader.world_to_radar(frame, 100, 0)
local ix, iz = reader.world_to_cell(grid, 100, 0)
local x, z = reader.cell_center(grid, ix, iz)
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

Requirements: installed WH3, Python with the project's local Lupa/Lua 5.1 dependency,
PowerShell, and write access to the game folder. The current launch scripts use
this machine's Steam installation path; inspect it before use on another machine.

```powershell
python -m pip install --only-binary=:all: --target .tools/python -r requirements-dev.txt
python -m unittest discover -s tests -p test_map_reader.py -v
python tools/map-capture/build.py --step 5
& ./tools/map-capture/launch.ps1
```

Supported capture steps: `1`, `2`, `3`, `5`. Game must be closed. The launcher's
graphics helper backs up preferences, applies minimum graphics and Ultra unit size.
Capture is static and paused: x20 is the manual-battle preference, not a way to
accelerate these read-only queries. No mouse automation or Computer Use is needed.

Output: `build/map-capture/run-YYYYMMDD-HHMMSS/`, including source snapshots,
manifest and source hashes, launch identity, exported XML, JSONL events, CSV,
status and cleanup record. The builder does not replace `build/tww3_bai_duel.pack`.

**Automatic captures close their own game process after saving.** Error/timeout
paths also stop only their identified process and preserve unfinished files for
diagnosis. They do not erase unverified partial output. A running game or existing
private files blocks a new launch. For a completed run whose cleanup was interrupted:

```powershell
& ./tools/map-capture/finish.ps1 -Run ./build/map-capture/run-YYYYMMDD-HHMMSS
```

Replace the placeholder with the actual run directory. Do not run the normal duel
launch command when intending to capture a map. Leave a game open only when the
user explicitly needs manual control.

## Complete field-feature capture

The optional `--features` mode adds native objects, [structure/bridge contexts](bridges.md), and point-reachability columns for the first unit of each of two opposing armies. Use the preserved two-unit diagnostic scenarios. This runner takes control of both units, disables firing, halts them, advances Deployment, and pauses after Deployed. The default surface-only mode remains in Deployment.

Reproduce the validated bridge-location capture:

```powershell
python tools/map-capture/build.py --step 3 --scenario docs/map/evidence/field-20260925/bridge-navigation/map_probe.xml --features
& ./tools/map-capture/launch.ps1
```

`--scenario` changes the bundled XML; it does not select a map by localized menu name. Source XML, bundled Lua and hashes are preserved in the run directory. Keep custom captures separate from a normal campaign or manual battle.

Additional CSV columns: `reach_side_1`, `reach_side_2` — 1 true, 0 false, −1 not queried outside the frame. JSONL contains `building`, `structure_context`, `navigation_unit`, `feature_conditions`, counts and timings. The unit events establish which units the columns refer to. Read the text as **UTF-8**, including on Windows; localized CCO names are not ASCII.

For the passage tool, select the new columns explicitly:

```powershell
python tools/map-capture/passages.py --csv build/map-capture/run-YYYYMMDD-HHMMSS/tww3_bai_map_capture_grid.csv --step 3 --reach-columns reach_side_1 reach_side_2 --output tmp/map-research/passages
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
