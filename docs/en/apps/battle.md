# battle

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/battle.md)

Battle context: sides, armies, units and roles. The only place that walks
`bm:alliances()`; other apps receive plain lists. Code: `src/apps/battle/`.

## adapter

| Function | What it does |
|---|---|
| `unsupported_reason(bm)` | The first unsupported battle kind (`is_from_campaign`, `is_multiplayer`, `is_replay`, `is_quest_battle`, `is_tutorial`, `is_siege_battle`, `is_ambush_battle`) or `nil` |
| `read_sides(bm, expected_units?)` | `{side, alliance, army, units}` for both sides; asserts one army per side and, if given, the unit count |
| `find_by_name(side, script_name)` | A unit by its XML `script_name` |
| `read_roles(bm, record?)` | Attacker/defender from `alliance:is_attacker()`; never inferred from index or spawn |

## services

| Function | What it does |
|---|---|
| `verify_roles(roles, expected)` | Compares scenario roles with engine roles |
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
