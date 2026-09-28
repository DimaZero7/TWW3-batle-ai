# navigation

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/navigation.md)

Point reachability for specific units. Code: `src/apps/navigation/`. How
fords and passages were found: [passages](../game/map/passages.md) and
[bridges](../game/map/bridges.md).

## adapter

`read_cells(bm, battle_vector, units, cells)` — for every cell inside the
radar frame calls `unit:can_reach_position(p)` for each unit. Works only in
the `Deployed` phase. Returns `{ix, iz, inside_radar, reachable = {bool, ...}}`.

## diagnostics_adapter

Diagnostics around `can_reach_position`, own units only; never feeds
decisions.

| Function | What it does |
|---|---|
| `state(bm, unit)` | Phase, position, ordered position, bearing, width, movement, control, deployment |
| `gate(bm, unit, p, native_return)` | What was asked, what the engine returned, the unit state |
| `origins(bm, unit, make_vector)` | Reachability of the unit's own position and of the same X/Z at terrain height |

Results are `{status = 'ok' | 'unavailable' | 'unsupported', ...}`.

## services

Pure bookkeeping of one movement order (no engine).

`new_leg_monitor({target = {x, z}, timeout_ms, detect_stuck?, arrive_m?, stop_samples?, min_ms?, stuck_window_ms?, stuck_move_m?})`
→ `monitor.update(t_ms, {x, z, moving})` returns `nil` while the order runs,
or the reason it ended:

| Reason | Condition (defaults) |
|---|---|
| `arrived` | 3 samples in a row not moving, centre ≤ 5 m from the target |
| `stopped` | 3 samples in a row not moving, centre farther than 5 m |
| `stuck` | moving, but the centre moved less than 2 m in 10 s |
| `timeout` | the order's time ran out |

Stops in the first 3 s after the order are ignored (reform, start).
`detect_stuck = false` is for a reform in place. `monitor.stats`: path,
distance to the target, closest distance, sample count.

The centre is `unit:position()`, the middle of the unit. The order point is
the front rank ([commands](../game/units/commands.md)), so a narrow formation
ends with `stopped` half its depth from the target.

## Limits

- `true` for a point does not prove the straight line to it is clear.
- The engine silently drops an order to a point with `false`: the unit stays put.
- An answer belongs to a specific unit and its current position: cavalry
  and infantry can differ.
- Grid passages and fords are candidates. Movement through them is verified
  separately ([tools/analysis/passages.py](../../../tools/analysis/passages.py)).
