# Visibility and hiding

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/visibility.md)

When the enemy sees a hidden unit, how an ambush hides again and how the ground
hides the enemy from us.

Hiding was verified in game on 2026-09-27 on The Moorlands Route (`catchment_03`), with
True Sight, ×20. Run `20260927-142521`, mode [unit_readout](../../apps/entries.md#unit_readout).

## Setup

| Unit | Hiding | Where |
|---|---|---|
| Empire Spearmen (`wh_main_emp_inf_spearmen_0`) | `hide_forest` | Northern forest, 100% forest within 24 m, ~350 m from the enemy |
| Huntsmen (`wh2_dlc13_emp_inf_huntsmen_0`) | `stalk` — hide anywhere (game DB) | Open grass, no forest within 24 m, ~340 m from the enemy |

Both sides are symmetric. A scout (general to the forest ambush, archers to the
huntsmen) was teleported to 80, 40 and 15 m from the ambush, 8 s per step.

## Result

| Ambush | Before scouting | 80 m | 40 m | 15 m | Scout gone |
|---|---|---|---|---|---|
| In forest (`hide_forest`) | hidden | hidden | **visible** | visible | hidden again |
| Huntsmen on grass (`stalk`) | hidden | **visible** | visible | visible | hidden again |

Identical for both sides. The first sample after a teleport may still be the
previous one: the engine applies a teleport on a following tick.

## What it means for a side's view

- A hidden unit is invisible and our sensors do not reveal it: the side view
  has neither its position nor range to it ([observation rules](../../apps/observation.md)).
- Reveal happens only up close: a forest ambush holds until 40–80 m; stalk on
  open grass is weaker and is seen at 80 m.
- After the scout leaves, the unit hides again. The side view keeps only the
  **last known position** with its time — all the side can rely on.
- `native.is_hidden` and the `hidden` status in `StatusList` are `true` for own
  hidden units.

## The ground hides the enemy

Measured on 27.09.2026 on The Moorlands Route with the formation probe (removed
in the reset of 30.09.2026):

- The armies stood with a ridge between them 3.5–9 m above the line of sight
  (7–9 m from another, turned position). The enemy was **never visible for the
  whole battle**: the engine hides it behind the hill.
- When the armies were placed where the ground between them is at least 2 m
  below the line of sight (on the 3 m [map grid](../map/sampling.md)), the enemy
  became visible.
- For another pair of places (behind a rock) the grid shows a ridge 3–15 m above
  the line of sight: the enemy would not be visible from there either (a
  calculation, not a battle).

The eye height used and how much ridge is enough to hide a unit were not
measured.

## Engine behaviour: the AI army is redeployed

The army the engine treats as AI (not the player's) is repositioned by the
game AI when deployment ends: the forest ambush moved ~300 m to the centre, the
huntsmen too. The player's side stayed put. Tests therefore take control of
every unit during deployment and teleport them back to their XML points after it.

One measurement per distance; the exact reveal threshold between 40 and 80 m
(forest) and beyond 80 m (stalk) was not measured.
