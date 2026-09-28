# formation

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/formation.md)

**Tree nodes:** [Deployment](../tree/deploy.md) · [Formation on the map](../tree/map_fit.md) · [Formation keeps the window](../tree/formation_window.md)

Army formation by the roles a [strategy](strategy.md) assigned; it does not know why the strategy was chosen. Pure code without the engine: the same in battle
and in the simulation (`tools/sim/formation.py`). Code: `src/apps/formation/`.

## services

| Function | What it does |
|---|---|
| `plan(layout, input, params)` | Place units with `layout`; now one: `line_and_blocks` (`wall` line in front, `arc` blocks behind, `lord` in the centre behind the wall, `other` not placed) |
| `facing(a, b)` | Bearing from point `a` to point `b`, degrees (0 = +Z, 90 = +X) |
| `overlaps(placements)` | Pairs of units whose rectangles overlap |
| `lord_routes(placements, choice, params)` | The lord's way from the centre to each flank of the wall: length and whether it is clear |
| `turn_in_place(middle, bearing, delta, depth)` | Order point and facing that turn a unit around its middle; the engine's `rotate` pivots on the front rank and shifts the unit ([commands](../game/units/commands.md#turning-in-place)) |

Input: `anchor` (centre of the wall's front line), `bearing`, `units` from the
[roster](../game/units/roster.md) with **measured** front and depth per width.
It tries every measured wall width, archer width and number of archer rows (up
to 3). Hard rules: archers in near-square blocks with room to turn in place between them (gap = diagonal − front, rows: diagonal − depth), no wider than the wall plus 10 m per side, the wall at least
`min_wall_depth_m` deep, every archer reaching at least `min_reach_m` past the
wall. Among valid options: **the thinnest wall no wider than `max_wall_front_m`** (180 m); the score is
the reach past the wall (the thinner the wall, the nearer the archers to the front). When every wall is
wider, the narrowest. Every option with its score is returned in `options`.

**The thinner the better** (user, 28.09.2026, task 29; before it was "the thicker the better"): a thick
wall pushed the archers back and they did not reach the enemy from the window. In the simulation the
optimum is 30 m spearmen (a 9.3 m wall); thinner walls grow to 240-475 m and do not fit between rocks.
The window against the enemy we see (>= 5 m) is looked for only among walls within the width limit.
On the map: `fit` (`src/apps/formation/fit.lua`).

The lord stands in the centre ([battle theory](../../ru/architecture/battle-theory.md), phase 1):
his job is to bind the enemy lord, who may come on either flank. The first archer row
splits into two halves (the extra block on the right) with a `lord_passage_m` (10 m)
passage on the axis; the lord stands in it level with the row's middle. `lord_routes`:
back into the corridor behind the first row and along it to the wall's end. Checked in
the simulation for every mix of 2 to 20 units and in battle (passage 11.7 m by the
soldiers, the lord 0.1 m from the plan).

Output: `status` (`ok`, `infeasible`, `no_wall`), `choice` and `placements`
(order point = front-rank centre, bearing, width, front, depth, role, row).
Defaults: the thinnest wall up to 180 m, every archer at least 80 m past the wall. Parameter meaning:
see the [Russian page](../../ru/apps/formation.md).

Watch it in motion: [battle viewer](../launch/viewer.md), layer “lord routes” (preset `modules`).
