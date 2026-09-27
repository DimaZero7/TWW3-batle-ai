# formation

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/formation.md)

Army formation by the roles a [strategy](strategy.md) assigned; it does not know why the strategy was chosen. Pure code without the engine: the same in battle
and in the simulation (`tools/sim/formation.py`). Code: `src/apps/formation/`.

## services

| Function | What it does |
|---|---|
| `plan(layout, input, params)` | Place units with `layout`; now one: `line_and_blocks` (`wall` line in front, `arc` blocks behind, `lord` on a flank, `other` not placed) |
| `facing(a, b)` | Bearing from point `a` to point `b`, degrees (0 = +Z, 90 = +X) |
| `overlaps(placements)` | Pairs of units whose rectangles overlap |
| `turn_in_place(middle, bearing, delta, depth)` | Order point and facing that turn a unit around its middle; the engine's `rotate` pivots on the front rank and shifts the unit ([commands](../game/units/commands.md#turning-in-place)) |

Input: `anchor` (centre of the wall's front line), `bearing`, `units` from the
[roster](../game/units/roster.md) with **measured** front and depth per width.
It tries every measured wall width, archer width and number of archer rows (up
to 3). Hard rules: archers in near-square blocks with room to turn in place between them (gap = diagonal − front, rows: diagonal − depth), no wider than the wall plus 10 m per side, the wall at least
`min_wall_depth_m` deep, every archer reaching at least `min_reach_m` past the
wall. Among valid options the best score wins: reach + `wall_depth_weight` ×
wall depth. Every option with its score is returned in `options`.

The lord stands level with the wall on its flank; if the enemy lord is visible
(`input.enemy_lord`), on the flank nearer to him, since our lord's job is to bind him.

Output: `status` (`ok`, `infeasible`, `no_wall`), `choice` and `placements`
(order point = front-rank centre, bearing, width, front, depth, role, row).
Defaults: the thicker the wall the better (depth weight 100, reach only breaks
ties), every archer at least 80 m past the wall. Parameter meaning: see the [Russian page](../../ru/apps/formation.md).
The map is not taken into account yet.
