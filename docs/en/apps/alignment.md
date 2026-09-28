# alignment

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/alignment.md)

**Tree nodes:** [Alignment](../tree/align.md)

Aligning our army opposite the enemy's main group. Pure. **Alignment only, no
approach:** our centre goes onto the line their army looks along (their centre
and facing from [vision](vision.md)), our front faces them, the distance stays.
Without their facing the [battlefield](battlefield.md) axis is used. Realign only
when clearly off: more than 10° or 15 m sideways.

| Function | What it does |
|---|---|
| `check(field, current, enemy?, params?)` | How far off we are: `angle_off_deg`, `offset_m`, `needed`, `reasons`, `source` (`centres` / `front` / `enemy_facing` / `field`) |
| `target(field, current, enemy?, params?)` | Where to stand: with `front` and `centres` where we are, only turning; with `enemy_facing` and `field` a point on the line at the same distance |

**The line (`params.line`, since 28.09.2026, task 26):** `front` (default) — a turn in place only:
farther than `near_m` (270 m) towards their centre, tolerance `far_angle_deg` (20 deg); nearer, square
to their front, 10 deg. `centres` — always towards their centre. `enemy_facing` — as before, onto the line
their army looks along (a turn and a shift). Without their group or its facing — the field's axis, turning
and shifting (a march to a point). Why: the game's AI turns to face our army by itself, and aligning with
where it looked chased it ([tree](../tree/align.md#how-we-align-since-28092026-task-26)).
| `overhang(field)` | How far their front hangs over our flanks, left and right |

Against endless realignment (user's rule): the tolerance above, plus a governor
(`new_governor`): not while units still move, at most every 20 s, and give up
after three alignments in a row that did not reduce the error.

Checked by tests, in the simulation (`first_attack_crooked`: 25° and 54 m off →
0° and 0 m) and in battle: 23° and 66 m off → 6 walking orders, 49 s → 1° and 3 m,
every unit within ≈ 1 m of the new plan; a minute of standing with the governor
answering "aligned", no orders, stand-still PASS. With the enemy unseen (a ridge
hid it) the AI waits instead of aligning.

Watch it in motion: [battle viewer](../launch/viewer.md), layer “alignment” (preset `modules`).
