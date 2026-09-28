# vision

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/vision.md)

How we see the enemy. Pure: it **only finds groups** of visible enemy units, in
(near) real time, and does nothing else. Its output feeds the next forks of the
decision tree. Code: `src/apps/vision/`.

Input: only what our side sees — visible enemy units with soldier positions and
strength (`apps.assessment.strength`).

| Function | What it does |
|---|---|
Each group also has `facing` (mean bearing of its units, weighted like the centre) when bearings are given.

| `gap(a, b, limit?)` | Gap between two formations: the closest soldiers; stops early at `limit` |
| `picture({own, enemy, enemy_total?})` | The battle picture: groups of **both** sides by one rule (ours in full, the enemy as seen), for group-against-group decisions |
| `groups(units, params?, total_strength?)` | Units are linked when the edge gap ≤ `link_gap_m`, links chain, a lone unit is a group; groups strongest first, `main` = the enemy's main army; `seen_share` if the total is given |

`link_gap_m` = 30 m: the game AI in defence keeps 2–6 m within a line and ≈ 20 m
between lines, 21.5 m at most over six layouts; 30 m adds margin (user's
decision). 20 units take ≈ 2 ms. The simulation (`tools/sim/formation.py`) runs the drawn enemy through it
(soldiers laid out evenly in each unit rectangle) and marks the groups and the main army. Tests: `tests/apps/vision/` with six real
game AI layouts in `tests/cases/vision/`.

Watch it in motion: [battle viewer](../launch/viewer.md), layer “groups” (preset `modules`).
