# navigation

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/navigation.md)

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

## Limits

- `true` for a point does not prove the straight line to it is clear.
- An answer belongs to a specific unit and its current position: cavalry
  and infantry can differ.
- Grid passages and fords are candidates. Movement through them is verified
  separately ([tools/analysis/passages.py](../../../tools/analysis/passages.py)).
