# Physical objects and structure effects

[Map guide](README.md) · [Русский](../../../ru/game/map/objects.md) · [Reachability](passages.md)

## Verified object inventory

`bm:buildings()` exposes more than houses. In the measured field battles it also returned boulders, cliff blocks, fences and other props. The native category is coarse: many rocks are labelled `house`. Preserve the exact name/category instead of interpreting the category as a real-world object class.

| Loaded terrain | Objects returned | Origins inside radar frame |
|---|---:|---:|
| Cathay farmland test | 1,543 | 1,391 |
| Wood Elf forest test | 3,700 | 3,680 |
| Cathay bridge-position test | 2,696 | 2,510 |
| Badlands river field test | 159 | 156 |

The list includes objects outside the radar rectangle. These counts do not establish that every decorative object, tree, projectile obstruction or physical shape is included.

The [read-only module](../../../../src/apps/map/) returns native index, name, category, origin X/Y/Z, central X/Y/Z, orientation, health, owner, gate/wall/tower/selectable flags. The origin can differ from the centre. Neither is a footprint, bounding box or collision polygon.

```lua
local rows, next_index, done, total = objects.read_buildings(bm, 0, 256)
-- Pass next_index as the start of the following batch until done.
```

Start indices are zero-based; returned native building indices are one-based. The module issues no orders and makes no world changes. It was compared live against direct native calls: **1,543/1,543 records matched**, including all captured fields. `Validation and hashes` (local archive: `research/evidence/map/field-20260925/module-validation/validation.json`).

`Cathay inventory` (local archive: `research/evidence/map/field-20260925/cathay-navigation/buildings.json`) · `Wood Elf inventory` (local archive: `research/evidence/map/field-20260925/wood-elf-scan/buildings.json`) · `Captured source` (local archive: `research/evidence/map/field-20260925/module-validation/capture.lua`)

## What this says about obstacles

Use three separate layers:

1. Object names, positions and state from the inventory.
2. `is_area_clear` for a specified rectangle — an area restriction query, not the identity of the blocker.
3. A unit's `can_reach_position` — point reachability after deployment, not a straight-line collision test.

The native building names were joined to installed `battlefield_buildings` records. The saved subset includes `collision_3d`, `height_map_mesh`, `ground_type`, health and destruction settings. Those are verified static configuration values, not collision meshes or proof of an object's current effect on a particular unit. `Matched records and source hashes` (local archive: `research/evidence/map/field-20260925/cathay-navigation/matching-db-records.json`).

On the Cathay grid, **1,226** points were reachable despite the containing 3 m area query being restricted. A rectangle overlapping a blocked edge can fail while its centre is accessible. Do not merge the two boolean layers or equate a failed area query with an entirely impassable square.

## Buildings with active bonuses or penalties

The installed `battlefield_buildings_to_special_abilities` table contains 296 assignments and can be joined by the returned building name. The joined objects in the initial Cathay and Wood Elf captures had **zero assignments in this table**. No active structure bonus/penalty was demonstrated in these tests.

`Assignments and byte-for-byte decode proof` (local archive: `research/evidence/map/field-20260925/static-records/battlefield_buildings_to_special_abilities_tables.json`).

This negative result applies to these captures and this data source. It does not establish absence of all scripted effects, quest mechanics or field structures elsewhere. Siege/settlement structures and their auras were excluded from this work. Do not create bonus circles from a building name or `ability_radius` without verifying its ability and activation state.

The CCO `BattleRoot.BuildingsList.Size` query returned 0 in the original Cathay and Wood Elf captures, despite their nonempty native lists. A later Cathay location returned one bridge, with verified `IsBridge` and empty ability/effect fields. Both lists are now exposed separately; see the [bridge check and interface](bridges.md). CCO is not a complete prop inventory.

On the official `chokepoint_badlands_river` field landscape, two returned names (`grn_cheifs_hut`, `grn_idol_02`) matched `wh3_main_siege_defender` assignments. Yet CCO returned zero structures and zero capture points in `land_normal`, `IsSiege=false`. This is a concrete case where a static assignment is **not evidence of an active field-battle bonus**. No settlement was loaded. `Records and context` (local archive: `research/evidence/map/field-20260925/badlands-river/summary.json`).

Reference: [CA building API mirror](https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_building.html). Measured CCO result: `events` (local archive: `research/evidence/map/field-20260925/module-validation/tww3_bai_map_capture_events.jsonl.gz`). Collision shapes, obstacle causes, projectile blocking and active building auras remain unverified.
