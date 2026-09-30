# MP Crossroads (flat) — an empty flat map

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Map catalogue](README.md) › Crossroads flat · [Русский](../../../ru/game/maps/crossroads-flat.md)

A flat field with no obstacles in the middle. The arena's battles are fought on
it ([data for training](../../training/README.md)): nothing but the armies
themselves affects the battle.

## Which map

| Field | Value |
|---|---|
| Map key in the database | `wh_mp_crossroads_flat`, type `classic` |
| Terrain resource | `terrain\battles\mp_crossroads_flat\` |
| Catchment | `catchment_03` |
| Width / height in the database | both 0 — the map size cannot be taken from the database |
| Minimap frame | X −768…+768 m, Z −800…+736 m (1536 × 1536 m) |

The ordinary `wh_mp_crossroads` points at another resource, `mp_crossroads` — it
is not the same map. Key and resource: [boundaries study](../../../ru/research/map-data/crossroads-boundaries.md)
(in Russian); frame: [minimap check](../../../ru/research/map-data/crossroads-radar-check.md)
(in Russian).

## Flat and clear

A query grid of 25.09.2026: at the centre of every 100 × 100 m cell, the ground
height and `is_area_clear` for the whole cell
([dense grid](../../../ru/research/map-data/crossroads-grid-check.md), in Russian;
data `research/evidence/map-data/crossroads-grid-evidence/grid.csv`).

| Area around the centre | Cells | Height | Clear |
|---|---:|---|---|
| ±600 m | 144 | 577.31 m in all | all 144 |
| ±700 m | 196 | 577.31–577.36 m | 194; two cells at (550; −650) and (650; −650) are not clear, their central 10 × 10 m are |

The ground in the middle varies (within ±500 m: grass 87 cells, sand 8, sharp
stones 4, mud 1); whether it affects movement was not checked.

The arena's armies stand 175–235 m from the centre, the deployment zones are
300 × 300 m squares centred at ±285 m (up to 435 m from the centre): all inside
the flat, clear ±600 m square.

## How it was found

On 30.09.2026 the `terrain\battles\*` resources in the game's packs were listed.
`mp_crossroads_flat` has an almost empty logic map — 526 bytes, against 95 KB
for the ordinary `mp_crossroads`.

Other flat or test map candidates in `terrainb.pack` (not checked):
`mp_plains`, `wh3_ksl_terraintest_01`, `qbwh2_dlc17_amtest`,
`test_tze_macrotile_01`.

## What not to do

The card saves searching, but an AI must not hard-code this map: read the
ground in battle, and train so that it works on other maps too.
