# core

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/core.md)

Shared helpers with no engine access and no dependency on other apps.
Code: `src/apps/core/`.

| Module | Functions |
|---|---|
| `value` | `finite(n)`, `integer(n)`, `known(v)`, `unknown(reason)`, `read(fn, kind, invalid_type_reason?)`, `vector(fn)`, `distance_xz(a, b)` |
| `json` | `encode(data)` — sorted keys, `NaN`/`inf` → `null`; `string(text)` |
| `errors` | `raise(code, message)`, `check(cond, code, message)`, `code(message)`, `guard(fn, on_error, is_stopped?)` |
| `clock` | `wall_seconds()`, `elapsed_wall_seconds(start)`, `iso_utc()`, `batch_stamp()` |

## Example

```lua
local value = require('apps.core.value')
local json = require('apps.core.json')

local ammo = value.read(function() return unit:ammo_left() end, 'number')
if ammo.status == 'known' and ammo.value == 0 then ... end

print(json.encode({event = 'snapshot', hp = 0.5}))  -- {"event":"snapshot","hp":0.5}
```

## Why

- `read` used to be repeated in four modules with different failure reasons.
  The default reason is `type_<type>`; the range sensor passes
  `'invalid_type'` to keep its previous format.
- `wall_seconds` counts seconds since midnight: epoch seconds lose precision
  in WH3's Lua numbers.
- The rules in detail: [errors and unknown values](../architecture/error_handling.md).
