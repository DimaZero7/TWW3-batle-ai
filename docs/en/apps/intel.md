# intel

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/intel.md)

What a side knows about enemies: whether one is visible now and where it
was last seen. Code: `src/apps/intel/`.

## Rule

Only the **current visible** enemy state is reliable. For an enemy that is
hidden, the last known position with its observation time is allowed. Hidden
current state is never read: `position()` is not even called for an
invisible unit.

## adapter

| Function | What it does |
|---|---|
| `query(unit, alliance)` | `true` / `false`, or `nil` when the engine did not answer |
| `observe(unit, alliance, public_id, now_ms, memory)` | Visibility + position of a visible unit + memory |

## services (no engine)

| Function | What it does |
|---|---|
| `new_memory()` | Memory of one observer for one battle |
| `record(memory, id, visible, position, now_ms)` | Updates memory, returns the observation |

```lua
local intel = require('apps.intel.adapter')
local memory = require('apps.intel.services').new_memory()

local seen = intel.observe(enemy, my_alliance, 'enemy-1', now_ms, memory)
-- seen.visibility = 'visible' | 'not_visible' | 'unknown'
-- seen.current = {x, y, z, seen_ms}      only when visible
-- seen.last_seen = {x, y, z, seen_ms}    last known position
```

`public_id` comes from a permitted roster mapping, never from an enemy
engine handle. Create a new memory per battle and per side.
