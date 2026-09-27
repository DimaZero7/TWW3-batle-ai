# assessment

[Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/assessment.md)

Force assessment at the start of a battle. Pure: from both armies (the
[roster](../game/units/roster.md)) it computes one set of features; strategies
only read them. Code: `src/apps/assessment/`.

| Function | What it does |
|---|---|
| `category(u)` | `lord`, `shooter`, `infantry`, `other` |
| `melee_power(u)` | Infantry strength: health × (attack + defence) / 100 |
| `ranged_power(u)` | Shooter strength: men × missile damage |
| `side(units)` | Counts by kind, infantry and shooter strength, arc-fire share, mean infantry armour |
| `assess({role, own, enemy})` | Both sides and `melee_ratio`, `ranged_ratio` |

The strength formulas are a first draft. An unknown card value is never zero:
men stand in for health, 1 for damage.
