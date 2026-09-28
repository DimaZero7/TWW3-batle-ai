# tactics

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/tactics.md)

**Tree nodes:** [Approach to the window](../tree/approach.md) · [Detour out of their reach](../tree/safe_detour.md) · [Stop under fire](../tree/under_fire_stop.md)

The trunk of the tactical level — phase 2, the approach to the window, with the branches `align`,
`safe_detour`, `under_fire_stop` ([tree](tree.md)). The same decisions in battle and in the
simulation: the game's entry and `tools/sim/formation.py` only build the view of the battle, carry
out the intent and say when a manoeuvre is over. `new(tree)`, `decide(s, view)` -> intent
(`wait`, `align`, `approach`, `hold`, `blocked`, target, window, path), `finish`, `busy`,
`interrupt`. Code: `src/apps/tactics/`.
