# approach

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/approach.md)

**Tree nodes:** [Approach to the window](../tree/approach.md)

Simple approach and the one-action-at-a-time rule. Pure. Code: `src/apps/approach/`.

User rules: no command spam (no new orders while a manoeuvre is under way);
alignment before approach; steps of 50 m, the whole formation keeping its shape,
the last one shorter; stop 20 m short of the gap at which the first shooters of
either side reach the other side's front; before each step the [mask](mask.md)
must show a place where the whole formation fits: an obstacle on the way is
walked around by the engine; if the 50 m target does not fit, the first place
past the obstacle is taken (`choose_advance`; in the simple variant without the
enemy army it may lie beyond the stop line, flagged). Stop (`blocked`) only when
there is no place at all.

| Function | What it does |
|---|---|
| `reach_past_front(shooters, front, toward)` | How far a side's shooters reach past its own front |
| `stop_gap(own_reach, enemy_reach)` | Stop line: the larger reach + 20 m |
| `next_step(gap, stop_gap)` | `arrived` or `step` (≤ 50 m) |
| `new_commander()` | `start`, `finish`, `decide` → `wait` / `align` / `approach` / `hold` / `blocked` |

Checked in the simulation: open field 50+50+50+21 m → hold at 129 m; a rock on
the way → 50 m, then 161 m (173 m for a 16-unit wall) to the first place past the
rock. In battle (a 16-unit wall marching to a point past the rock, the enemy not
considered): align → 50 m → 170 m past the rock → 11 m → hold 18 m from the goal;
the engine split the wall at the rock, walked both halves around it and formed it
again; no unit stuck (longest queue wait 10 s); stand-still PASS.
