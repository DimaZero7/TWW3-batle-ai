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
  - Ours: the simulator runs **10 ticks/s** — the calibration is ON (`fatigue.calibration.on=true`,
    the rule below). The legacy model (5 ticks/s, DB points, ready −7 as rest) is kept only for
    `on=false`. In 204 fair recordings (`build/agent-fatigue`):
    unopposed running to the first transition gives 39.44 points/s (512 intervals), single-entity
    melee 186.21 (219), shooting arenas 173.08 (17), single-entity rest −180 (14). These are
    medians: duration between consecutive same-direction transitions, width from the DB
    thresholds. Running starts fresh; standing before running is excluded. Native
    `fatigue_state()` and CCO states 0–5 agree in the old probe; all 3814 initial arena reads
    are Fresh. Exact starting points within that band are unrecorded: zero remains an assumption.
  - Ours (calibration ON): melee tires only a unit with an attack order — a single
    entity at 15 a tick, a formation at 13.7 (not all its men fight); in melee without an attack
    order the unit moves or rests. Charging +34 also only with an attack order, for a single entity
    only in the first 2 s of a contact (the game's lords first turn tired after 60–62 s of melee; with
    +19 and the charge for all 13 s the simulator gave 37–47 s; [lords](../../training/simulator.md#lords)). A move costs by its
    order's run flag, not by speed: run +4, walk −1 (DB; clean walk spans of the game's AI, 2925 s:
    0 crossings up, 6 down). A routing unit +4. Shooting 7.5 (not the DB +18). Idle −18 without
    unfinished movement, attack or aiming and with no standing enemy within 80 m; otherwise ready
    −7 (standing network units in the game recover at 0.48 / 0.62 / 0.79 / 1.00 of the idle rate
    with the nearest enemy at 40 / 40–80 / 80–150 / 150–300 m). Completed movement counts as rest
    (an order-context approximation: arenas record neither `is_idle` nor combat stance, and
    `is_idle` sometimes stays false after `halt` while recovering at −180/s). Dead and departed
    units are frozen. 13.7 / 7.5 are fitted on the units' activities in the 204 recordings so that
    the accumulation reproduces the game's recorded states (bootstrap 13.6–14.5 / 6.5–7.6). The
    simulator then reproduces the game's exhausted shares (own 17.1 versus 17.3 %, enemy 29.1
    versus 28.6 %; the legacy ×5 model
    gave 5.7 and 9.2 % — a third as often as the game). The cost: network battles 96/132 in the
    check instead of 100/132, and the gate trade a little further from the game. Late in a battle
    our units in the simulator stand longer than in the game and some rest back; numbers in the
    [simulator](../../training/simulator.md#fatigue-calibration). Arena positions lack height;
    uphill is not modelled.
  - **Limitation:** the melee flag does not mean all soldiers tire at +19: the planner's spearmen
    stay fresh in the pairs against slaves while continuing to kill (hence 13.7 for a formation,
    not +19).
- **Perfect Vigour** units never tire (a WH3 bug leaves units that start tired stuck at that
  level). **Strider** ignores terrain/slope penalties. Campaign stances (march, raiding) lower the
  battle's starting and maximum vigour. · fandom, Steam · medium.

[tws]: https://twwstats.com/kv/fatigue
[mod]: https://steamcommunity.com/workshop/filedetails/discussion/3308073985/612032174956240488
[fw-fat]: https://totalwarwarhammer.fandom.com/wiki/Fatigue
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[tick]: https://steamcommunity.com/app/594570/discussions/0/5367692919519734908/
