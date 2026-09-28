# Coordinates and metres

[← Back](README.md) · [Map guide](README.md) · [Русский](../../../ru/game/map/coordinates.md)

Battle vectors use **X/Z for the horizontal plane and Y for height**. Coordinates
accept fractional values. Treat the battle distances and rectangle sizes used by
these APIs as metres; the interface documents widths/distances in metres.
This does not establish the native resolution of the terrain or navigation engine.

Our grid is not a built-in game grid. A 5 m cell is a square we choose to query.
A 1 m sampling step does not imply metre-accurate passability or a native 1 m voxel.

## Cell coordinates

For origin `(min_x,min_z)`, step `s`, zero-based column `ix` and row `iz`:

```text
cell centre: x = min_x + (ix + 0.5) * s
             z = min_z + (iz + 0.5) * s
columns = ceil((max_x - min_x) / s)
rows    = ceil((max_z - min_z) / s)
linear index = iz * columns + ix
```

`reader.world_to_cell` checks the half-open interval `[min,max)` before computing
`floor((coordinate-min)/s)`. It returns `nil` outside, including the exact upper
edge, rather than clamping a point into the last cell. This is an indexing convention,
not a claim that the physical boundary itself is forbidden.

The radar maximum and the last query square's maximum are stored separately.
On a 1024 m side, 5 m queries reach 1025 m from the origin and 3 m queries reach
1026 m. Clip drawings to the radar frame; the API still queried full edge squares.

## Height

Full method, matrix layout and reproducible illustration: [terrain heights](heights.md).

```lua
local y = bm:get_terrain_height(x, z)
local p = battle_vector:new()
p:set_x(x); p:set_y(y); p:set_z(z)
```

Use the local sampled height for `ground_type` and `is_area_clear`. Do not copy one
height across a hilly map. One returned ground height does not describe bridges,
caves, overhangs, buildings or traversable airspace. Flying-unit handling is outside
the tested map workflow.

Do not assume `(0,0)` is the map centre. Crossroads gave centre `(0,-32)`; the
tested Kislev frame gave `(0,0)`. Read the frame again in every loaded scenario.
