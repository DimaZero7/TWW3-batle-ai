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
- Ours: **not modelled** — [simulator.md](../../training/simulator.md) says "what fatigue does to
  attack, defence and speed is not in the database"; it is. **Gap / conflict with our docs.** With
  units reaching Tired–Very tired in long melees, melee attack ×0.85–0.75 and speed ×0.9–0.85 are
  material.
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
  - Ours: regressing the recorded fatigue states gave **×5 a second** (melee 5.0, shooting 4.8,
    running 6.2; 1315 transitions), so `fatigue.per_second` 5. **Conflict** with "10 ticks a second"
    (×2). Our number is measured; the claim is a single forum post. Our resting test (Tired
    and Very tired back to Fresh within 240 s, [states](../units/states.md)) does not separate them:
    idle at ×5 takes 140–200 s, at ×10 70–100 s. A test timing the transitions of a resting unit
    would settle it.
- **Perfect Vigour** units never tire (a WH3 bug leaves units that start tired stuck at that
  level). **Strider** ignores terrain/slope penalties. Campaign stances (march, raiding) lower the
  battle's starting and maximum vigour. · fandom, Steam · medium.

[tws]: https://twwstats.com/kv/fatigue
[mod]: https://steamcommunity.com/workshop/filedetails/discussion/3308073985/612032174956240488
[fw-fat]: https://totalwarwarhammer.fandom.com/wiki/Fatigue
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[tick]: https://steamcommunity.com/app/594570/discussions/0/5367692919519734908/
