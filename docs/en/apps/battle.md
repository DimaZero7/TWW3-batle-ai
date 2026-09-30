# battle

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/battle.md)

Battle context: sides, armies, units and roles. The only place that walks
`bm:alliances()`; other apps receive plain lists. Code: `src/apps/battle/`.

## adapter

| Function | What it does |
|---|---|
| `unsupported_reason(bm)` | The first unsupported battle kind (`is_from_campaign`, `is_multiplayer`, `is_replay`, `is_quest_battle`, `is_tutorial`, `is_siege_battle`, `is_ambush_battle`) or `nil` |
| `read_sides(bm, expected_units?)` | `{side, alliance, army, units}` for both sides; asserts one army per side and, if given, the unit count |
| `find_by_name(side, script_name)` | A unit by its XML `script_name` |
| `speed_guard(bm, speed, on_restore?)` | Keeps the requested speed until `Complete`: the engine lowers it once the outcome is decided. Returns `stop()` |
| `read_roles(bm, record?)` | Attacker/defender from `alliance:is_attacker()`; never inferred from index or spawn |
| `deadline(bm, real_ms, on_expire, name?)` | A wall-clock safety net: calls `on_expire` once after `real_ms` of real time, whatever the battle speed or phase, so a test battle never runs forever. Returns `cancel()` |
| `health_signature(units)` | Men alive and hit points of the units as one string `"men:hp×10000"`. A string, not a number: the game's Lua numbers are single precision, and a large sum would drop small damage. An unreadable unit counts 0 |

## services

| Function | What it does |
|---|---|
| `verify_roles(roles, expected)` | Compares scenario roles with engine roles |
| `new_stall_detector(window_ms)` | Stall detector: the battle is stalled when the `health_signature` of all units has not changed for `window_ms` of model time. `update(now_ms, signature)` returns `true` once stalled; `quiet_ms(now_ms)` is how long nothing has changed |
| `opponent(side)` | `1 → 2`, `2 → 1` |

## Example

```lua
local battle = require('apps.battle.adapter')

local reason = battle.unsupported_reason(bm)
if reason then return end
local sides = battle.read_sides(bm, 1)   -- exactly one unit per side
local enemy = sides[2].units[1]
```

XML unit names (`script_name`) survive Lua environment boundaries, so
entries recognise their scenario by them rather than by global variables.
