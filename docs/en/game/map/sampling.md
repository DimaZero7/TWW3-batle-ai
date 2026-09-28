# Surface sampling and resolution

[← Back](README.md) · [Map guide](README.md) · [Русский](../../../ru/game/map/sampling.md)

For each cell, query its centre at the local terrain height:

```lua
local height = bm:get_terrain_height(x, z)
local p = battle_vector:new()
p:set_x(x); p:set_y(height); p:set_z(z)
local clear = bm:is_area_clear(p, 0, step, step, false)
local ground = bm:ground_type(p)
```

These are the exact call forms exercised in our captures. `height` is a number,
`ground` a string and `clear` a boolean. Tested ground values include `grass`,
`forest`, `mud`, `rock`, `sharp_stones`, `snow` and, on Crossroads, `sand`.
Do not treat this observed list as exhaustive.

## Meaning of a cell

Height and ground type are **centre samples**. The area query concerns a full
square of the chosen size. A false result identifies a restriction somewhere in
that square, without returning its reason or outline. It is not a guarantee that
the entire square is blocked. True is not a guarantee that a particular formation
can reach it, shoot through it or travel there safely.

The previous 50 m route crossed 23 cells whose full-square query returned false.
This is why we refined the grid instead of using large red cells as solid obstacles.
Surface queries also return data beyond the radar frame, so they cannot identify
the boundary by themselves.

## Actual measurements: one loaded Kislev map

| Step | Cells | Collection, Lua `os.clock` | Raw CSV | gzip CSV |
|---|---:|---:|---:|---:|
| 5 m | 42,025 | 0.267 s | 0.96 MB | 0.31 MB |
| 3 m | 116,964 | 0.770 s | 2.73 MB | 0.86 MB |
| 2 m | 262,144 | 1.921 s | 6.18 MB | 1.83 MB |
| 1 m | 1,048,576 | 7.319 s | 24.99 MB | 7.59 MB |

One sequential pass per step, paused Deployment, 4,096 cells per batch. Timing
includes API work, CSV formatting/writes and callback gaps; excludes game loading,
later compression and drawing. It is not a frame-time benchmark during combat.
All four captures together took about 11 s; process launch through copied results
took 97.886 s. `os.time` differences were unusable (zero); we do not report them as cost.

The 2 m grid agrees with the 1 m area flags on 99.796% of the radar area and took
3.81 times less collection time. This is agreement between query datasets, not
99.8% navigation accuracy. Compared to 1 m, changed area is 0.781% at 5 m,
0.393% at 3 m and 0.204% at 2 m. The main observed gain is refinement of contours.

Use **2 m as the next prototype default**, not as an approved neural input.
One-time 1 m capture is feasible; continuously re-querying a million cells is not
supported by this test. These timing figures belong to the original measurement
runner, not a guaranteed performance contract for every subsequent module version.

![Resolution comparison](../../../../research/evidence/map-data/kislev-scales-evidence/comparison.png)

Archived figure: green = area query true; red = false; blue = outside the radar
frame; yellow = the old route. The yellow line connects recorded positions whose
largest gap was 24.987 m. Finer cells do not increase the accuracy of that old route.

See the [same enlarged fragment](../../../../research/evidence/map-data/kislev-scales-evidence/comparison-zoom.png)
and [full results](../../../ru/research/map-data/kislev-scales-check.md).
