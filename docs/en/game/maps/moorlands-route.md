# The Moorlands Route — Empire grasslands

[← Back](README.md) · [Map catalogue](README.md) · [Русский](../../../ru/game/maps/moorlands-route.md)

The project's first field map: its terrain and objects were captured, and units, visibility and deployment were checked on it. It is not a verified first campaign encounter of Karl Franz or another lord. The installed database identifies an official Creative Assembly classic land battle. The name follows the official preview resource; localized display text has not been extracted.

![Official menu preview](../../../../research/evidence/maps/moorlands-route/preview.png)

The preview shows grassy hills, scattered trees and a road. It is illustrative menu artwork, not a measured overhead map or proof of current traversal/visibility.

## Verified database identity

Extracted from the installed vanilla `db.pack` on 26 September 2026; decoding and re-encoding the source table produced identical bytes. `Full selected record and source hash` (local archive: `research/evidence/maps/moorlands-route/database.json`) · `Preview provenance` (local archive: `research/evidence/maps/moorlands-route/preview-source.json`).

| Field | Value |
|---|---|
| Battle key | `wh3_dlc21_emp_grasslands_01_03` |
| Terrain resource | `terrain\battles\wh3_dlc21_macro_emp_grasslands_01\` |
| Selected area | `catchment_03` |
| Tile upgrade | Not set |
| Culture / author | `wh_main_emp_empire` / Creative Assembly |
| Battle type | `classic`; `is_naval=false` |
| Released / single player / multiplayer | All `true` |
| Large settlement / 15 m walls / underground | All `false` |
| Key buildings | `has_key_buildings=false`; does not prove absence of physical scenery |
| Team-size fields | `4` / `4`; not the number of units in our scenario |
| Matchmaking | `false` |
| Custom battle script / environment override | Not set |
| Environment audio | `empire` |
| Database width / height | Both `0`; no usable size measurement |

The `dlc21` resource prefix is part of the official game key; this selection uses installed vanilla resources, with no third-party map mod.

## Geometry and reuse

Live capture on 26 September 2026: requested `catchment_03`; the game's XML confirmed the terrain resource and resolved tile position `(0.453125, 0.40625)`. The export omits the catchment label, so identity rests on the requested XML, manifest and launch provenance; no independent catchment-resource decoding is claimed. Build: `v9.0.0 Build 50218.4334952 (modded)`; the diagnostic pack causes the modded label.

| Measurement | Result |
|---|---|
| Radar frame | X/Z from −512 to +512 m; 1024 × 1024 m |
| Grid | 342 × 342, step 3 m; 116,964 unique points |
| Inside / outside centres | 116,281 / 683; outside matrix values are unknown (`-1` / `NaN`) |
| Terrain Y inside frame | 516.491–624.861 m; span 108.370 m |
| Native ground counts inside | Grass 79,379; forest 30,225; mud 3,687; sharp stones 2,990 |
| Water ground types | Zero sampled centres; small unsampled water features are not excluded |
| Restricted area-query samples | 2,268 inside; not a navigation mask |
| Native objects | 904 origins, 887 inside; not collision outlines or every tree |
| Capture cost | Surface: 0.876 s by Lua `os.clock`; successful launch/capture: 35.54 s wall time |

![Measured heights, ground and objects](../../../../research/evidence/maps/moorlands-route/capture-20260926/overview.png)

`Summary and limits` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/summary.json`) · `Raw terrain grid` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/terrain-grid.csv`) · `Height matrix` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/height.csv`) · `Forest matrix` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/forest.csv`) · `Ground codes` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/ground-code.csv`) and `legend` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/ground-legend.json`) · `Water mask` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/water-ground.csv`) · `Objects` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/objects.json`) · `Hashes` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/hashes.json`) · `Closure evidence` (local archive: `research/evidence/maps/moorlands-route/capture-20260926/cleanup.json`).

Array order is `[iz, ix]`, with increasing Z by row and X by column. Cell centres are `x=-512+(ix+0.5)*3`, `z=-512+(iz+0.5)*3`. The last row/column centres lie outside the radar rectangle; their small clipped inside portions remain unmeasured. CSV and NPY variants are preserved. This is a terrain-only export: no army positions or opponent data.

One successful capture followed a startup crash; removing the XML BOM preceded success but does not prove the crash cause. The owned game was closed and installed probe files removed. Slopes are derivable from heights but have not been published for this capture. Traversal by the assigned Empire units and active structure bonuses remain untested.

Use the tested [frame](../map/boundaries.md), [sampling](../map/sampling.md), [height](../map/heights.md), [vegetation](../map/vegetation.md) and [object](../map/objects.md) methods. A radar frame remains distinct from proven movement walls; point samples do not describe every part of a cell. Per-unit reachability requires a separate query.

Deployment, weather and battle conditions come from each experiment's scenario file; the preview does not tell them.

## The copy in the repository

`data/maps/moorlands-route/` is the part of the capture kept in Git: `summary.json`, `objects.json`, `ground-legend.json`, pictures and `grid-3m.npz` — the 3 m grid matrices, 342 × 342, order `[iz, ix]`, origin `min_x = min_z = −512`, `step = 3`:

| Array | Contents |
|---|---|
| `clear` | `is_area_clear`: 1 clear, 0 blocked (2,268 cells), −1 outside the frame |
| `ground` | Ground code per `ground-legend.json`: 0 forest, 1 grass, 2 mud, 3 sharp stones; −1 outside the frame |
| `height` | Height, m, `float16`; `NaN` outside the frame |
| `inside` | 1 when the cell centre is inside the radar frame |

The file was made from the capture's local archive; no tool in `tools/` recreates it.

Analyses of this grid (in Russian): [map obstacles](../../../../research/analysis/moorlands-obstacles/README.md) and [the hamlet](../../../../research/analysis/hamlet/README.md).
