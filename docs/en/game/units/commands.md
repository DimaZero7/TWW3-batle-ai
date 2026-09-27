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

## Order point and unreachable points

Measured on 27.09.2026 with Empire spearmen, 120 men
([hamlet](../../../../research/analysis/hamlet/README.md), in Russian):

- The point of `goto_location_angle_width` is the **centre of the front rank**.
  `unit:position()` is the middle of the soldiers. At 10 m width the middle
  stands ≈ 15 m behind the order point, so arrival is checked on the front rank
  or corrected by half the formation depth.
- **An order to an unreachable point is silently dropped.** The call does not
  fail, `ordered_position` keeps its old value and the unit stays `is_idle`.
  Check `unit:can_reach_position(p)` before the order and that
  `ordered_position` changed after it.
- Widths of 5–60 m are accepted as given (`ordered_width` equals the order).
  The real front is 1–2 m narrower. Narrowing from 30 to 10 m took 22 s; to 8
  and 5 m, more than 25 s.

## Turning in place

Measured on 27.09.2026 with 19 × 18 m archer blocks:

- `uc:rotate(90)` turns the unit **around the centre of its front rank**, not
  around its middle. A square block shifts ≈ 12 m sideways and forwards on a 90°
  turn. Archers standing 3 m behind their infantry ran into it (0.7–1.0 m
  between soldiers of different units). The turn took 14–17 s.
- Turning in place: `goto_location_angle_width` to "middle + half the depth
  along the new facing" with the same width (`apps.formation.services.turn_in_place`):
  1.4–1.7 m shift, 2–3 m to the infantry, 7–8 s. Computed from the current middle
  each time, the shift adds up (3.9 m after four turns); use the unit's remembered slot.

## Unit position is not the soldiers' middle

During a minute of standing without orders (27.09.2026) `unit:position()` of a
spearmen wall jumped 2.1 m and of archers 0.9 m, while the soldiers' middle from
`ManList` moved at most 0.3 m and `is_moving` was never `true`. To tell whether a
unit stands still, and for exact geometry, use soldier positions.

## Numbers in the game's Lua

- Numbers are **single precision**: `1/3` → `0.33333334`, `16777217` → `16777216`.
  Integers above 2^24 lose precision; sums of large numbers lose low digits.
- `math.huge` is `nil`. Use `nil` for "no value yet".

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
