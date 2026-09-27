# units

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/units.md)

Own unit state, missile range and shared formation width rules.
Code: `src/apps/units/`. What is verified in game: [unit knowledge](../game/units/README.md).

## state_adapter — own unit sensor

```lua
local state = require('apps.units.state_adapter')
local r = state.observe(unit, {
    owned = true,                    -- own units only
    observer_alliance = my_alliance,
    cco = function(u, key) ... end,  -- CcoBattleUnit read
    target_id = function(u) ... end, -- public id of a visible unit
})
-- r.sensors['native.number_of_men_alive'] = {status='known', value=87}
```

- 68 fields: health, men, ammo, speed, movement, combat, routing, flanks,
  behaviours, CCO morale and fatigue, statuses. Catalogue:
  [state fields](../game/units/state-fields.md) and `data/units/field-catalog.json`.
- For a unit that is not own (`owned ~= true`) only visibility is returned:
  `access = 'withheld_own_only'`, nothing else is read.
- Target and flank threats are disclosed only as the id of a **visible** unit.

## range_adapter — missile range

`observe(source, target?, context)` — engine and card range,
`unit_in_range`, unit distance. Both endpoints are gated first: ally or
visible enemy. Otherwise `access = 'withheld'` and **no** range, position or
CCO read happens. Details: [missile range](../game/units/missile-range.md).

## card_adapter — unit card and profile

| Function | What it does |
|---|---|
| `stats(cco, unit)` | The whole `UnitDetailsContext.StatList` in card order: `{key, Value, DisplayedValue, ValueBase}` |
| `details(cco, unit)` | `UnitDetailsContext.Mass`, `Name` and unit fields `HealthMax`, `NumEntitiesInitial` |
| `profile(unit)` | Type, commander, class, kind flags, men, speeds, range, ammunition, abilities |

Unknown values are written as `'unknown:<reason>'`. Used by the
[roster capture](../game/units/roster.md).

## formation_adapter — own unit movement and soldiers

| Function | What it does |
|---|---|
| `motion(unit)` | Position, ordered position, bearing, ordered bearing and width, men alive, movement flags. Unreadable values are left out; no position is an error |
| `soldiers(cco, unit)` | CCO `ManList.At(i).Position` of every soldier → `{status = 'ok', count, xz_dm = {x1, z1, ...}}` in whole decimetres, or `{status = 'unavailable', reason}` |

Own units only: enemy soldier positions would reveal hidden men.

## contract

| Function | What it does |
|---|---|
| `width_bounds(kind)` | Lord 3–8 m, formation 20–40 m. Engineering input limits, not measured game limits |
| `valid_width(kind, width)` | Width within the bounds |
| `can_shoot(unit)` | Archers with positive ammo |

These bounds used to be duplicated in `deployment` and `policy_host`.
