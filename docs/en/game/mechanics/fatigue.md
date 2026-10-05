# Fatigue (vigour)

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Fatigue · [Русский](../../../ru/game/mechanics/fatigue.md)

Fatigue levels, what each costs in stats, what tires and what rests. Conventions:
[index](README.md). Ours: [pace and fatigue](../units/pace.md), `tools/nn/sim/fatigue.py`.

## Levels and their penalties — they are in the database

The stat multipliers per level are a database table, `unit_fatigue_effects_tables` (rows:
level, stat, multiplier). Read from our own `db.pack` (v9.0.1) on 02.10.2026; twwstats shows the
same for 9.0.0:

| Level (from points) | Speed | Melee attack | Melee defence | Armour | Charge bonus | AP melee damage | Reload |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh (0) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Active (2800) | 1 | 0.95 | 1 | 1 | 1 | 1 | 1 |
| Winded (6600) | 0.95 | 0.95 | 1 | 1 | 1 | 0.9 | 1 |
| Tired (12600) | 0.9 | 0.85 | 1 | 1 | 0.9 | 0.9 | 1 |
| Very tired (18000) | 0.85 | 0.75 | 1 | 0.9 | 0.75 | 0.9 | 1 |
| Exhausted (27000) | 0.85 | 0.7 | 0.9 | 0.75 | 0.7 | 0.9 | 0.9 |

Stats are `scalar_speed`, `stat_melee_attack`, `stat_melee_defence`, `stat_armour`,
`stat_charge_bonus`, `stat_melee_damage_ap`, `stat_reloading`; missing rows mean ×1.
· our `db.pack`, [twwstats][tws] · high; a modder's list from 2025 gives the
same numbers ([thread][mod]). The fandom table (e.g. Exhausted −35 % reload, −10 % damage) and a
WH1 guide differ — older or cumulative; trust the database. A 6.1.2 bug hid the Exhausted reload
penalty on the card. · CA bug tracker · high. 6.0 shows vigour and terrain effects on the card.
- Ours: multipliers **are modelled** in `fatigue.effects()`; morale is separate in
  `morale.py`. See the [simulator](../../training/simulator.md).
- **Morale:** Very tired −2, Exhausted −6 (Tired 0). · DB · high. Ours: same.

## What tires and what rests

Points per tick (positive tires); the same in WH2 and WH3:

| Activity | Points | Activity | Points |
|---|---:|---|---:|
| charging | +34 | walking | −1 |
| melee | +19 | ready (combat pose, standing) | −7 |
| shooting | +18 | idle | −18 |
| running (infantry) | +4 | idle in heavy rain / snow | −10 |
| running (cavalry, horse artillery) | +6 | limbering/unlimbering a gun | +6 |
| walking (foot artillery) | +12 | climbing ladders / walls | +10000 |

Uphill movement multiplies the movement cost by +50 / +100 / +150 % at gradients above
0.05 / 0.1 / 0.2; downhill gives no benefit. Thresholds: 2800 / 6600 / 12600 / 18000 / 27000,
max 30000. · [twwstats][tws], [fandom Fatigue][fw-fat], [CA elevation blog][elev] · high.

- **Tick length.** One player says the tick is 0.1 s (10 a second): then continuous melee
  (190/s) reaches Tired in ~66 s and Exhausted in ~142 s, and idle (−180/s) recovers from Exhausted
  in ~150 s. · [Steam][tick] · medium-low.
  - Ours: production retains **5 ticks/s** and ready recovery. The measured **10 ticks/s**
    clock, idle recovery and contact-weighted trial are behind `fatigue.calibration.on=false`.
    DB points remain unchanged. In 204 fair recordings (`build/agent-fatigue`):
    unopposed running to the first transition gives 39.44 points/s (512 intervals), single-entity
    melee 186.21 (219), shooting arenas 173.08 (17), single-entity rest −180 (14). These are
    medians: duration between consecutive same-direction transitions, width from the DB
    thresholds. Running starts fresh; standing before running is excluded. Native
    `fatigue_state()` and CCO states 0–5 agree in the old probe; all 3814 initial arena reads
    are Fresh. Exact starting points within that band are unrecorded: zero remains an assumption.
  - Ours, with the trial enabled: idle −18 per tick without unfinished movement, attack or aiming; completed movement
    counts as rest. Ready is −7; walking recovers −1 rather than tiring. Combat and movement
    override idle. This is an order-context approximation: arenas record neither `is_idle`
    nor combat stance. The old probe sometimes retains `is_idle=false` after `halt` while
    recovering at −180/s, so that flag alone does not identify the fatigue activity.
    Charging, walking and ready are not separately fitted: insufficient isolated complete
    intervals. Arena positions lack height; uphill is not modelled.
  - **Limitation:** the global ×5 regression mixed formation activities. The melee flag does
    not mean all soldiers tire at +19: the planner's spearmen stay fresh in all three pairs
    against slaves while continuing to kill. ×10 identifies the clock, not this aggregation;
    replay results and unmet acceptance criteria are in the [simulator](../../training/simulator.md).
- **Perfect Vigour** units never tire (a WH3 bug leaves units that start tired stuck at that
  level). **Strider** ignores terrain/slope penalties. Campaign stances (march, raiding) lower the
  battle's starting and maximum vigour. · fandom, Steam · medium.

[tws]: https://twwstats.com/kv/fatigue
[mod]: https://steamcommunity.com/workshop/filedetails/discussion/3308073985/612032174956240488
[fw-fat]: https://totalwarwarhammer.fandom.com/wiki/Fatigue
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[tick]: https://steamcommunity.com/app/594570/discussions/0/5367692919519734908/
