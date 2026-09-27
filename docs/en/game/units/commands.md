# Commands tested in battle

[Units](README.md) · [Русский](../../../ru/game/units/commands.md) · [Evidence](evidence.md)

Create one controller per unit so that commands do not accidentally affect an entire group:

```lua
local uc = army:create_unit_controller()
uc:add_units(unit)
uc:take_control()
```

This removes the unit from player/native-AI control. All experiments used explicit script control. Position vectors use battle X/Y/Z; obtain ground Y with `bm:get_terrain_height(x,z)`.

| Intent | Tested call / recipe | Observed result |
|---|---|---|
| Walk / run to a point | `uc:goto_location(p, false)` / `true` | All three moved; walking and running states differed; units arrived and stopped |
| Position, facing, width | `uc:goto_location_angle_width(p, 90, width, false)` | Infantry accepted 15 m and 50 m formation orders; individual entity coordinates also confirm rearrangement |
| Turn | `uc:rotate(90, false)` | Relative +90° change, including wraparound from 270° to 0° |
| Stop | `uc:halt()` | Units decelerated and settled; stopping is not instantaneous |
| Melee attack | `uc:melee(true); uc:attack_unit(enemy, false, true)` | All three entered melee; health fell; Archers did not spend arrows |
| Leave melee / reposition | `uc:goto_location(p, true)` to a distant point | All three disengaged; some continued taking damage while separating |
| Targeted shooting | `uc:melee(false); uc:fire_at_will(false); uc:attack_unit(enemy, true, false)` | Archers fired at the ordered target with free fire disabled |
| Free fire | `uc:fire_at_will(true)` | Idle Archers selected a target and spent arrows |
| Stop shooting | `uc:halt(); uc:fire_at_will(false)` | Ammo stabilized after a short completion tail of already-started shots |
| Guard | `uc:change_behaviour_active('defend', enabled)` | All three support it; enabled units stopped pursuing a routed target |
| Rest | `uc:halt()` and wait | Fatigue recovered naturally; there is no separate rest spell |
| Withdraw | `uc:withdraw(true)` | Works in the explicit withdrawal-enabled scenario; read withdrawal/leave states rather than trusting successful return |

For the General, a formation-width parameter is accepted but **one entity does not form ranks**. Ordered position and reported unit position need not coincide exactly; infantry position readings ended several metres away from the requested anchor. Do not require floating-point equality to recognize completion.

## Shooting: two separate orders

`fire_at_will(false)` alone did not cancel targeted shooting. Two completed baseline runs spent another **153 / 138 arrows** after this toggle. Adding `halt()` stopped new sustained shooting; **0–4 further arrows** were observed across the stop stages. Already released projectiles also remain in flight.

`ammo_left()` returned **1800** for 90 intact Archers with 20 shots each. This is a total ammunition count in this setup, not “20 volleys remaining.” The runtime `missile_range()` was **130 m**. Range alone does not guarantee a clear shot; these tests used a clear, nearby target.

The tested recipe deliberately supplies both `melee(...)` and the `primary` argument of `attack_unit`. It does not claim that every combination of these switches is interchangeable.

## Guard and bracing are separate

With guard enabled, units moved only **0.01–1.66 m** after their target was forced to rout. With guard disabled, the same three types moved **78.05–91.39 m** and retained their attack targets. Enemy routing was forced solely as a controlled test stimulus.

`defend` is not a command to dig trenches or add armour. The `braced` state appeared with guard disabled too. See [states](states.md).

## Withdrawal and abilities: current API details

The tested withdrawal fixture explicitly uses `<can_withdraw>true</can_withdraw>` in both armies. Its General is declared with `<general><name>Unit action diagnostic</name><star_rating level="1"/></general>`. Earlier fixtures without these declarations accepted `withdraw(true/false)` but did not depart; the declarations were changed together, so the experiment does not isolate each one's individual causal effect.

A zero-argument `withdraw()` raises a native error in this build. The installed `lib_battle_script_unit.lua` wrapper describes and passes the boolean `should_run`. Do not copy the zero-argument signature from the older controller reference.

In the XML general setup, the working self-target recipe was:

```lua
if unit:can_perform_special_ability(key) then
    uc:perform_special_ability(key, unit)
end
```

Both Stand Your Ground and Foe Seeker produced their corresponding live effects. Omitting the target raised an error in this build. Availability must be read from the particular unit; this is not a claim about the skill tree of a campaign rank-1 lord.

Skirmish was disabled in the fixtures. It is an optional native automation, not a required primitive for our algorithm. Unsupported formation-spacing behavior was not invoked. No “raise shield”, “dig in” or instant restoration command is included.
