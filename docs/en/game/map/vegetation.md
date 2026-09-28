# Forest zones and tree placements

[← Back](README.md) · [Map guide](README.md) · [Русский](../../../ru/game/map/vegetation.md)

Experiment: 2026-09-25, `run-20260925-224603`. Official Kislev terrain, `wh3_main_macro_ksl_plains_01`, existing `scenarios/map_capture.xml`. The game was launched once and automatically closed after capture; game-side diagnostic files were removed. No combat or unit-effect test was performed.

This section records the completed measurement and its limits. Forest sampling works through Lua; individual placements here were extracted from resources for this fixed scenario. This does not establish a general-purpose live tree API.

## Two matrices

- `forest-3m.csv.gz` (local archive: `research/evidence/map/forest-trees-20260925/forest-3m.csv.gz`) / `.npy`: **1** = `bm:ground_type(vector)` returned `forest`, **0** = another ground type, **-1** = unsampled inside-frame edge strip. Query Y is `bm:get_terrain_height(x,z)`. Each result describes a cell-center sample, not every point in its square. Yellow therefore means nonforest, not necessarily grass or traversable ground.
- `trees-3m.csv.gz` (local archive: `research/evidence/map/forest-trees-20260925/trees-3m.csv.gz`) / `.npy`: number of selected tree-model placement origins in each cell, extracted from native terrain resources. Zero means no selected placement origins; it is not proof that the cell is free of foliage, trunks or collision. These are static positions, not live tree state.
- `tree-points.csv.gz` (local archive: `research/evidence/map/forest-trees-20260925/tree-points.csv.gz`): individual resource placements, model, position, scale, raw rotation byte and flags. Rotation-byte units and flags were not interpreted.
- [forest-3m.png](../../../../research/evidence/map/forest-trees-20260925/forest-3m.png), [trees-3m.png](../../../../research/evidence/map/forest-trees-20260925/trees-3m.png), [comparison-3m.png](../../../../research/evidence/map/forest-trees-20260925/comparison-3m.png): colored matrices with a common coordinate frame.
- `summary.json` (local archive: `research/evidence/map/forest-trees-20260925/summary.json`): counts, hashes, validation, timing and limitations.

Arrays are `[iz,ix]`, 342 rows × 342 columns. Row 0 is at the south / minimum Z. Cell edges start at X/Z = -512 m, step 3 m; the last row and column are clipped to +512 m (1 m wide). The regular last-center query is at +512.5 m, outside the radar frame, so its forest value is deliberately marked unknown. The frame was measured with `BattleRadarPosition`; no new movement-wall measurement was performed.

## Tree source and decoding

Native source: `tile074.pack`, `terrain/tiles/battle/ksl_plains/kislev_plains_01/tile_22/ksl.tree_list.bin`.

Decompressed SHA-256: `dfddc5930c1ce89112d4ce16a6a44d2052dbbc733c0f0369341338fcaf5ae6dc`.

The decoder follows [RPFM's TreeList v4 reader](https://github.com/Frodo45127/rpfm/blob/master/rpfm_lib/src/files/bmd_vegetation/tree_list/v4.rs): `FASTBIN0`, u16 version, u32 group count, then u16-sized model names and u32 instance counts, followed by `f32 x,y,z; u8 rotation; f32 scale; u8 flags`. All 79 groups / 129,885 instances were consumed and re-encoded byte-for-byte. The project decoder snapshot is archived with the evidence; the proprietary native terrain binary is not copied into documentation.

The list also contains other vegetation and stones. Selection uses the birch, Kislev pine, oak, spruce and rich-spruce model families, excluding names containing shrub, log or branch. Broken-tree and birch-totem variants are included. `model-selection.json` (local archive: `research/evidence/map/forest-trees-20260925/model-selection.json`) records every inclusion/exclusion and count. No assumption that every retained model blocks units is made.

## Coordinate validation and results

For this scenario only: battle X = file X − 3072, battle Z = file Z − 3072, battle Y = file Y + 100. The horizontal transform was inferred by matching source placements to the previously captured 1 m terrain heights and comparing alternative offsets/orientations. It is not a universal map transform.

Fresh Lua terrain-height queries at all **5,168** selected positions produced median absolute vertical difference **0.121 m**, p95 **0.494 m**, maximum **1.771 m**; **99.75%** are within 1 m. This supports the terrain alignment; it does not independently establish live tree identity, trunk position or full coverage of every placement system.

- **14,380** forest center samples; zero changed classifications versus the previous 3 m capture.
- **5,168** selected tree placements in **5,155** cells; maximum 2 origins per cell.
- **3,722** tree origins are on `forest` ground; **1,446** are on other ground. Forest classification and tree placement are different layers.
- Grid capture clock interval: **0.964 s**, including the existing height, ground, area-clear reads and logging. Tree-position validation and logging: **0.745 s**. These are Lua `os.clock()` intervals, not independent wall-clock benchmarks or forest-only timing.
- Full launch through capture: **88.08 s**. `status.json` (local archive: `research/evidence/map/forest-trees-20260925/status.json`) and `cleanup.json` (local archive: `research/evidence/map/forest-trees-20260925/cleanup.json`) in the run directory confirm closure.

Unmeasured: trunk/crown shapes, collision radii, effects on units, tree destruction, live enumeration, performance on other maps and automatic transform discovery. Do not use this matrix as a collision or movement mask.

## Reproducing and checking the matrices

The evidence preserves the exact Lua probe and reader, input/exported XML, raw grid and point-validation events, model selection, source hash, resulting matrices and launch/cleanup records. Archived Python snapshots describe the original experiment, including its local extraction inputs; they are not portable launch commands. No new production tree service is claimed.

From the saved raw grid: assign 1 for `ground == "forest"`, 0 otherwise, and −1 when `inside_radar == 0`. From saved tree points: for each point inside `[−512,512)` in X/Z, increment `trees[floor((z+512)/3), floor((x+512)/3)]`. These operations reproduce the matrices without launching the game. Forest values measure the center only; tree values count origins, not covered area.

The existing capture launcher automatically closes its owned game. Repeating the tree experiment requires the archived probe and matching local game resources, not just rebuilding the standard surface-only probe.


## Saved evidence

![Forest zones and tree placements, 3 m](../../../../research/evidence/map/forest-trees-20260925/comparison-3m.png)

`tww3_bai_map_capture_grid.csv.gz` (local archive: `research/evidence/map/forest-trees-20260925/tww3_bai_map_capture_grid.csv.gz`) · `tww3_bai_map_capture_events.jsonl.gz` (local archive: `research/evidence/map/forest-trees-20260925/tww3_bai_map_capture_events.jsonl.gz`) · `capture.lua` (local archive: `research/evidence/map/forest-trees-20260925/capture.lua`) · `reader.lua` (local archive: `research/evidence/map/forest-trees-20260925/reader.lua`) · `map_probe.xml` (local archive: `research/evidence/map/forest-trees-20260925/map_probe.xml`) · `tww3_bai_map_capture_ready.xml` (local archive: `research/evidence/map/forest-trees-20260925/tww3_bai_map_capture_ready.xml`) · `model-selection.json` (local archive: `research/evidence/map/forest-trees-20260925/model-selection.json`) · `provenance.json` (local archive: `research/evidence/map/forest-trees-20260925/provenance.json`) · `manifest.json` (local archive: `research/evidence/map/forest-trees-20260925/manifest.json`) · `launch.json` (local archive: `research/evidence/map/forest-trees-20260925/launch.json`) · `status.json` (local archive: `research/evidence/map/forest-trees-20260925/status.json`) · `cleanup.json` (local archive: `research/evidence/map/forest-trees-20260925/cleanup.json`) · `hashes.json` (local archive: `research/evidence/map/forest-trees-20260925/hashes.json`)

`Decoder/build snapshot` (local archive: `research/evidence/map/forest-trees-20260925/source-snapshots/build_probe.py`) · `Analysis snapshot` (local archive: `research/evidence/map/forest-trees-20260925/source-snapshots/analyze.py`)
