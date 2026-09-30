# Pace: running, walking and fatigue

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Units](README.md) › Pace and fatigue · [Русский](../../../ru/game/units/pace.md)

How fast units walk and run in battle and how fatigue builds up. Sources:

- arena battle recordings: the first 13 fair arena battles on 30.09.2026 (`20260930-130637` …
  `-132200`, Normal difficulty), 7522 s of records; 5 more were recorded after
  the analysis ([data for training](../../training/README.md));
- unit cards read in battle ([roster](roster.md), `data/roster/`);
- the fatigue table `_kv_fatigue_tables` from the game's database (30.09.2026,
  [the game's database](../database.md), `config/nn/game_rules.json`).

Game v9.0.1, build 50381.

## Speed

Card: `fast_speed` and `slow_speed` of the unit's profile; these are the
reference numbers. Recordings: the 75th percentile of speed over the seconds a
unit was moving (the `mv` flag), routing excluded; speed is the distance moved
between neighbouring one-second records.

| Unit | Running, card | Running, recorded | Walking, card |
|---|---:|---:|---:|
| Empire General | 3.4 m/s | 3.4 m/s | 1.5 m/s |
| Spearmen | 3.0 m/s | 3.0 m/s | 1.5 m/s |
| Archers | 3.3 m/s | 3.3 m/s | 1.5 m/s |

- **Walking in the recordings** is about 1.4–1.6 m/s. We do not split it by
  unit: the lord has only 28 seconds of walking, and for the archers the
  estimate depends a lot on which seconds count as walking.
- **Running in the recordings** depends on how the seconds are chosen. By the
  "running" flag (`f`, routing excluded) it comes out lower: the general
  3.2 m/s, spearmen and archers 3.0 m/s — those seconds include speeding up and
  stopping.

Example: spearmen run the 350 m between the arena's armies in about 115 s
(3.05 m/s) and walk them in about 230 s.

## Fatigue

Numbers from `_kv_fatigue_tables`: how many fatigue points each activity adds
or takes away (minus is rest).

| Activity | Points |
|---|---:|
| Charging | +34 |
| Melee | +19 |
| Shooting | +18 |
| Running | +4 |
| Walking | −1 |
| Standing ready | −7 |
| Standing idle | −18 |

For slopes the table has multipliers 50 / 100 / 150 (shallow / steep / very
steep); how they apply was not checked.

Thresholds (points):

| State | Fresh | Active | Winded | Tired | Very tired | Exhausted | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| From | 0 | 2800 | 6600 | 12600 | 18000 | 27000 | 30000 |
| Morale ([morale](morale.md)) | — | — | — | 0 | −2 | −6 | — |

**The time step is not checked.** The table does not say over what time the
points are added. If per second, resting from "tired" to "fresh" would take
about 700 s; in the [states](states.md) test units rested in 240 s of game time.
So the step is shorter than a second, or there are multipliers.

## How to read it

A unit's `fatigue_state()` is a string such as `threshold_fresh`,
`threshold_tired` (the arena records it as `fat`); more in
[state sensors](state-sensors.md).

## Not checked

Speed on slopes and in forest, the speed of tired units, how fatigue changes
attack and defence.
