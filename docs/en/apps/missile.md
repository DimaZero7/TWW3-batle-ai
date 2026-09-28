# missile

[← Back](README.md) · [Documentation](../README.md) · [Apps](../architecture/apps.md) · [Русский](../../ru/apps/missile.md)

Missile damage: HP a second a shooting block takes off its target, and a step of a missile
exchange for the simulation. Pure code, the same model for both sides. Code: `src/apps/missile/`.

Model from the [damage measurement](../game/units/missile-damage.md) (Empire archers on Empire
spearmen): 0.1 arrow a second per man; HP per arrow = c(distance) · h^0.3 (h — the target's
share of HP left); c = 13.4 at 70 m, 11.6 at 90 m, 10.0 from 110 m. Other shooters and
targets are scaled by the card's missile damage and marked uncalibrated.

| Function | What it does |
|---|---|
| `hp_per_arrow(d, missile_damage)`, `rate(shooter, target, d)` | HP per arrow; HP a second |
| `target_of(shooter, others)` | The nearest block of the other side it reaches (fire at will) |
| `step(units, dt)` | Everyone with arrows shoots its target; damage applied after all shots |

Checked: one archer unit on spearmen at 70, 105 and 120 m loses half and all of the target in
the measured times (±20 %). Not modelled: morale, shields, flank and rear, height, movement, melee.
