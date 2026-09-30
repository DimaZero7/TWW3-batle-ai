# Errors and unknown values

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/architecture/error_handling.md)

In battle, a script error must never pass silently: the battle would go on
without our script and the record or measurement would be wrong. The project
follows three rules.

## 1. Engine reads never raise: `known` / `unknown`

Adapters read values through `core.value.read`:

```lua
local value = require('apps.core.value')

local hp = value.read(function() return unit:unary_hitpoints() end, 'number')
-- {status = 'known', value = 0.8}
-- or {status = 'unknown', reason = 'read_error' | 'nil' | 'type_string' | 'nonfinite'}
```

- `0` and `false` are known values, not "empty".
- `unknown` does not mean the mechanic is absent, only "not read right now".
- Coordinates are copied into a plain `{x, y, z}` table (`value.vector`);
  engine objects never leave the adapter.

## 2. Wrong data is an immediate error

Services and adapters check their input with `assert` and a clear message:
`'declared/native battle role mismatch'`, `'battle requires one army per alliance'`,
`'Invalid terrain height'`. A battle with wrong data does not go on "somehow":
the entry stops with an error.

For new errors with a stable code use `core.errors`:

```lua
local errors = require('apps.core.errors')
errors.raise('ROUTE_BLOCKED', 'no reachable cell near target')  -- "[ROUTE_BLOCKED] ..."
errors.code(message)  -- 'ROUTE_BLOCKED'
```

## 3. Engine callbacks are guarded and stop after the first error

Everything the engine calls (`register_phase_change_callback`,
`repeat_callback`, `real_callback`) is wrapped in `errors.guard`:

```lua
local function guarded(fn)
    return errors.guard(fn, fail, function() return state.finished end)
end
bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
```

`fail` writes one `error` event with a traceback, removes timers and releases
unit control. The launcher sees `error` / `probe_error` and ends the run with
status `lua_error`.

## Where to look

1. `build/<target>/runs/<time>/events.jsonl` — the `error` event with a traceback.
2. `status.json` next to it — the run outcome (`lua_error`, `timeout`, `crash_report`).
3. A game crash report is copied to the same folder when one appears.
