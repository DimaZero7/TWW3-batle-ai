# alignment

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/alignment.md)

Aligning our army opposite the enemy's main group. Pure. **Alignment only, no
approach:** our centre goes onto the line their army looks along (their centre
and facing from [vision](vision.md)), our front faces them, the distance stays.
Without their facing the [battlefield](battlefield.md) axis is used. Realign only
when clearly off: more than 10° or 15 m sideways.

| Function | What it does |
|---|---|
| `check(field, current, enemy?, params?)` | How far off we are: `angle_off_deg`, `offset_m`, `needed`, `reasons`, `source` |
| `target(field, current, enemy?)` | Where to stand: point on the line at the same distance, and facing |
| `overhang(field)` | How far their front hangs over our flanks, left and right |

Against endless realignment (user's rule): the tolerance above, plus a governor
(`new_governor`): not while units still move, at most every 20 s, and give up
after three alignments in a row that did not reduce the error.

Checked by tests, in the simulation (`first_attack_crooked`: 25° and 54 m off →
0° and 0 m) and in battle: 23° and 66 m off → 6 walking orders, 49 s → 1° and 3 m,
every unit within ≈ 1 m of the new plan; a minute of standing with the governor
answering "aligned", no orders, stand-still PASS. With the enemy unseen (a ridge
hid it) the AI waits instead of aligning.
