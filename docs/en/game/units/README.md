# Verified unit controls

[English](README.md) | [Русский](../../../ru/game/units/README.md) · [Project](../../../../README.md)

Live experiments, 26 September 2026, WH3 **v9.0.0, build 50218.4334952**.
This is a practical interface for a classical tactical algorithm. It is not an AI implementation.

| Guide | Contents |
|---|---|
| [Common state sensors](state-sensors.md) | Health, entities, ammunition, morale, fatigue, control and status readouts; 68 fields with live evidence |
| [Missile attack range](missile-range.md) | Native/card range, pair checks, allied units and visibility-first filtering |
| [Commands](commands.md) | Movement, facing, formation width, melee, shooting, guard, disengagement and withdrawal |
| [Starting deployment](deployment.md) | Historical measurements and current [deployment-placement-v2](../../apps/deployment.md), without reservation circles |
| [Visibility and hiding](visibility.md) | Forest and stalk: reveal distance, re-hiding, last-seen memory |
| [States and passive effects](states.md) | Braced, fatigue, forest hiding, general support and ability availability |
| [Experiments and evidence](evidence.md) | Conditions, measured comparisons, raw logs, limitations and reproduction |
| [Unit catalogue by faction](catalog/README.md) | Short unit cards, exact keys and links to detailed information |

Test units: **120 shieldless Spearmen**, **90 Archers**, **one foot General of the Empire**. Unit experience is zero, Ultra unit size, requested battle speed ×20, minimum graphics. Only the diagnostic script pack is loaded; unit DB records are unchanged. The final general setup is explicitly commanding, character rank 1, unit experience 0; these are separate runtime fields.

**An XML battle's default abilities are not a campaign skill build.** The rank-1 XML general exposes two active abilities and Hold the Line. This proves their availability in this scenario, not their availability to a newly recruited campaign lord. Always read the actual unit's abilities before ordering one.

The guide contains measured behavior and explicit limits. Unsupported hypotheses remain in ignored `tmp/unit-actions/`. Games started for these experiments are closed after capture.
