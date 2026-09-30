# Unit roster

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/roster.md) · [Unit cards](catalog/README.md)

The roster is data collected in advance for each unit type: the card as the
player sees it and the measured formation at each width.

One file per type: `data/roster/<unit key>.json`. This data does not have to be
parsed in battle. Units are added gradually, as we check them.

The player sees the enemy army and its unit cards before the battle, so roster
data per enemy **type** is fair. Enemy positions and state in battle stay
limited to what is visible.

## What a file holds

| Part | Source | Content |
|---|---|---|
| `card` | Battle, automatic | The card as the player sees it: `StatList` (armour, attack, defence, damage, charge, speed, morale, arrows, range…), mass, health, class, unit kind, speeds, abilities. Date, game version, run |
| `formation` | Our measurement | For each ordered width: front × depth, reform time from 30 m |
| `ours` | Filled in by hand | `fire`: `arc`, `direct` or `null`; `roles` (empty so far); notes |
| `battle` | Later | Checked in battles: how long it holds, how much it kills |

`update` rewrites only `card` and `formation`; it leaves `ours` and `battle` alone.

## How to add a unit

1. Add its key and number of men to `config/roster/capture.json` (up to 5 units
   a run — as many as there are places on the map; one faction a run).
2. Build and run:

```bash
.venv/Scripts/python -m tools.build roster-capture
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target roster-capture
```

3. Update the roster from the run's log and look at it:

```bash
.venv/Scripts/python -m tools.roster update build/roster-capture/runs/<time>/events.jsonl
```

```bash
.venv/Scripts/python -m tools.roster show
```

4. Fill in `ours` by hand (fire type, notes).

A run of 3 units takes about 10 s of real time at x20 and ends by itself.

## In the roster now (27.09.2026, v9.0.1 build 50381)

| | Empire General | Spearmen | Archers |
|---|---|---|---|
| Key | `wh_main_emp_cha_general_0` | `wh_main_emp_inf_spearmen_0` | `wh2_dlc13_emp_inf_archers_0` |
| Men | 1 | 120 | 90 |
| Unit health | 4068 | 8280 | 6210 |
| Armour | 85 | 30 | 20 |
| Attack / defence | 55 / 45 | 20 / 34 | 14 / 17 |
| Weapon damage | 430 | 25 | 24 |
| Charge | 40 | 4 | 4 |
| Mass | 600 | 100 | 90 |
| Walk / run, m/s | 1.5 / 3.4 | 1.5 / 3.0 | 1.5 / 3.3 |
| Shooting | — | — | 130 m, 20 arrows a man, damage 19 |
| Fire type (ours) | — | — | arc |

Formation (front × depth, m; reform from 30 m, s):

| Width, m | Spearmen | Archers |
|---|---|---|
| 80 | 78.4 × 5.5; 21 | 77.4 × 7.6; 20 |
| 60 | 57.9 × 6.0; 13 | 58.8 × 8.0; 14 |
| 40 | 38.6 × 7.8; 6 | 38.6 × 9.9; 6 |
| 30 | 28.5 × 9.3; 4 | 29.0 × 12.6; 5 |
| 20 | 18.4 × 14.9; 8 | 19.2 × 18.4; 9 |
| 15 | 14.2 × 18.2; 11 | 13.9 × 26.4; 15 |
| 10 | 8.9 × 30.9; 19 | 10.5 × 36.4; 22 |
| 8 | 7.8 × 36.9; 24 | 9.2 × 46.7; 30 |

Archers stand looser than spearmen: at the same width their formation is
deeper, though they have fewer men.

The roster also has Skaven (list `config/roster/capture_skaven.json`, captured
with `python -m tools.build roster-capture --capture config/roster/capture_skaven.json`):
the warlord `wh2_main_skv_cha_warlord_0` (1 man), clanrat spearmen
`wh2_main_skv_inf_clanrat_spearmen_0` (160) and skavenslave spearmen
`wh2_main_skv_inf_skavenslave_spearmen_0` (180). On 30.09.2026 the skavenslave
slingers `wh2_main_skv_inf_skavenslave_slingers_0` (140) were added to the list;
their card was captured in a run with only the warlord (for the
[unit passports](../../training/units.md)). `python -m tools.roster show`
prints every number for any unit.

## What the card does not give, and oddities

- **Card morale = 0** for all three in deployment (the game's database gives
  the archers 50). The cause is not known; we do not use the card's morale yet.
- **No damage split** into normal and armour-piercing, for weapons or arrows:
  the card shows only the sum. The split is taken from the game's database
  ([unit passports](../../training/units.md), since 30.09.2026).
- **Shields:** the spearmen and archers have no shields; the general has one,
  but the card has no line for it. How the card shows a shield's defence
  against arrows was not checked.
- `starting_ammo` (native) = arrows for the whole unit (1800 = 20 × 90), while
  `stat_ammo` on the card is per man.
- The general's `Name` is the name from the scenario ("Roster"), not the type's
  name.

Code: `apps.units.card_adapter`, entry point `entries.roster_capture`, tool `tools/roster.py`.
Raw logs: the local archive `research/evidence/units/roster-20260927`.
