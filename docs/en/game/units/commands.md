# Commands tested in battle

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/commands.md) · [Evidence](evidence.md)

Which unit orders from Lua we checked in battle and how the engine carries them out.

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
- **A place on rock puts the unit crooked.** When a unit's place (an order or
  a deployment spot) falls on rock, the engine places the unit itself: shifted
  up to 18 m and facing 172° and 28° instead of 90° (28.09.2026, The Moorlands
  Route). So check the place against the map before the order.
- Widths of 5–60 m are accepted as given (`ordered_width` equals the order).
  The real front is 1–2 m narrower. Narrowing from 30 to 10 m took 22 s; to 8
  and 5 m, more than 25 s.

## Turning in place

Measured on 27.09.2026 with 19 × 18 m archer blocks, The Moorlands Route:

- `uc:rotate(90)` turns the unit **around the centre of its front rank**, not
  around its middle. A square block shifts ≈ 12 m sideways and forwards on a 90°
  turn. Archers standing 3 m behind their infantry ran into it (0.7–1.0 m
  between soldiers of different units). The turn took 14–17 s.
- Turning in place: `goto_location_angle_width` to "middle + half the depth
  along the new facing" with the same width: 1.4–1.7 m shift, 2–3 m to the infantry, 7–8 s. Computed from the current middle
  each time, the shift adds up (3.9 m after four turns); use the unit's remembered slot.

## Unit position is not the soldiers' middle

During a minute of standing without orders (27.09.2026) `unit:position()` of a
spearmen wall jumped 2.1 m and of archers 0.9 m, while the soldiers' middle from
`ManList` moved at most 0.3 m and `is_moving` was never `true`. To tell whether a
unit stands still, and for exact geometry, use soldier positions.

## Unit facing: 64 sectors

Measured on 28.09.2026: one spearmen unit was teleported with every facing round
the circle 1° apart, and 0.1° apart around 90°, and `unit:bearing()` and the
soldiers were read. The engine holds a unit's facing in **64 sectors of 5.625°**
and puts the unit in the **middle** of its sector: everything from 0 to 5.62° →
2.81°, from 84.38° to 89.99° → 87.19°, from 90 to 95.62° → 92.81°. Right after the
order the soldiers face exactly as sent; within a few seconds they turn to the
middle of the sector. A formation planned for another facing ends up as a
**staircase**: the centres on one line, each block turned up to 2.8° from it. So
take the facing at the middle of the sector straight away
(`apps.orders.facing.snap`) and send that in the order. All 27 earlier battles
behaved the same way.

## How a unit walks on an order

Measured on 27–28.09.2026 on The Moorlands Route, x20: plain move orders
(`goto_location_angle_width`) to armies of 13–16 Empire and Skaven units, unit
centres every 2 s.

- A walking unit faces where it goes: the deviation is a median 3–4°, within 37°
  in 90% of samples. It turns at up to 15–20° a second (90% of turns).
- In a general march it walks at ≈ 1.3 m/s (median over 2 s segments). A straight
  move of 4–50 m takes ≈ 9.7 s + distance / 1.37 m/s (53 moves): a few seconds go
  on starting and dressing the formation in place. Speeds by unit type: [pace](pace.md).
- The width in a move order changes only at the destination: the unit walks with
  its old width and re-forms there. To pass in a narrow formation, narrow before
  leaving.
- A long wall of 16 units, each unit ordered to its own place behind a rock: the
  engine **split the wall**
  (the left units went round the rock on the west, the right ones on the east) and
  **assembled it again** behind the rock. It took 198 s; the farthest unit walked
  254 m instead of 170 m straight.
- Round an obstacle the engine lets units press into each other and separates them
  itself afterwards: at the peak 10–45% of the army's men stand closer than 1 m to
  men of another unit (rock 35%, spearmen in two
  rows 45%, Skaven 25%, a 72 m gap between rocks 20%, a rock at the map edge 10%).
  Units jostle and slow down. The same happened by the hamlet in a manual battle:
  the jam comes from a crowd of own units ([hamlet](../../../../research/analysis/hamlet/README.md), in Russian).

**Going round an obstacle, engine only.** Each unit gets one plain order to its
place behind the obstacle, and the engine chooses the path. `formation-probe`
runs of 27.09.2026, The Moorlands Route, x20. Time: until the last unit stops.
"Longer than straight": the median over units of the unit centre's path divided
by the straight line from start to end. "Closest to the rock": how near the unit
centre came to an impassable cell.

| Terrain | Units | Time | Longer than straight | Closest to the rock |
|---|---:|---:|---:|---:|
| a rock | 16 | 208 s | 1.20 | 6.7 m |
| a rock at the map edge, one side closed | 16 | 129 s | — | — |
| a 72 m gap between two rocks | 16 | 139 s | 1.07 | 6.7 m |
| spearmen only, in two rows | 13 | 213 s | 1.28 | 4.2 m |
| Skaven: clanrats of 160 and slaves of 180 men | 13 | 215 s | 1.17 | 10.8 m |

In short: the path is 7–28 % longer than the straight line, and unit centres
come within 4–11 m of the rock. At the map edge only the time was measured.
Source: the analyses `research/analysis/walker/` and
`research/analysis/logistics/`; they were removed with our AI and remain in the
Git history (commit 242c3b1).

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

Skirmish was disabled in the fixtures: it is an optional native automation of the unit. Unsupported formation-spacing behavior was not invoked. No “raise shield”, “dig in” or instant restoration command is included.
