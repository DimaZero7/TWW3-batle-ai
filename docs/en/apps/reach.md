# reach

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/reach.md)

**Tree nodes:** [Formation keeps the window](../tree/formation_window.md) · [Approach to the window](../tree/approach.md)

Who reaches whom with missiles, and where to stand — the "window" (battle theory, phase 2).
Pure code; it does not move units ([approach](approach.md) does). Code: `src/apps/reach/`.

Measured rule: the first arrow flies when the distance from the **middle** of the shooting
block to the target's nearest rank is <= 125-126 m for a 130 m range, whatever the depth.

| Function | What it does |
|---|---|
| `middle(block)`, `to_block(point, block)` | Middle of a block; distance to its nearest point |
| `first_arrow_m(shooter)`, `reaches(shooter, target)` | The first-arrow distance; whether the shooter reaches the target |
| `window(own, enemy, bearing)` | Our army moved by `a` along `bearing`: `safe_to` — where their first shooter reaches us; the stop where most of our shooters reach theirs before that (`prefer_m` into the window, at most its middle). Advances are counted from where we stand and may be negative, so the stop does not creep. `reason`: `window`, `no_enemy_shooters`, `no_window`, `no_shooters` |

Checked in the simulation (28.09.2026): the test army against the game AI's lines (window
91-103.7 m), no enemy shooters, outranged, already under fire, the stop holding after
arrival, 40 random armies with the enemy line turned up to ±8°.
