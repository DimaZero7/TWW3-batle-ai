# Unit roster

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/roster.md)

The roster holds pre-collected data per unit **type**, one file per type:
`data/roster/<unit key>.json`. The AI takes force-assessment and formation
data from here. Units are added gradually as we check them. The player sees
the enemy army and its unit cards before the battle, so per-type enemy data is
fair; enemy positions and state in battle stay limited to what is visible.

| Part | Source | Content |
|---|---|---|
| `card` | Battle, automatic | The card as the player sees it (`StatList`), mass, health, class, kind flags, speeds, abilities; date, game version, run |
| `formation` | Our measurement | Per ordered width: front × depth, reform time from 30 m |
| `ours` | By hand | `fire`: `arc`, `direct` or `null`; roles; notes |
| `battle` | Later | Results checked in battles |

`update` rewrites only `card` and `formation`.

Add a unit: list its key and men in `config/roster/capture.json` (up to 5 per
run, one faction per run), then:

```bash
.venv/Scripts/python -m tools.build roster-capture
```

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/launcher/launch.ps1 -Target roster-capture
```

```bash
.venv/Scripts/python -m tools.roster update build/roster-capture/runs/<time>/events.jsonl
```

```bash
.venv/Scripts/python -m tools.roster show
```

Current content (Empire General, Spearmen, Archers), measured values and the
card's gaps (card morale reads 0 in deployment; no base/armour-piercing damage
split; shield protection not seen on the card; native `starting_ammo` is per
unit, card `stat_ammo` per man) are listed in the [Russian page](../../../ru/game/units/roster.md).

Code: `apps.units.card_adapter`, entry `entries.roster_capture`, tool `tools/roster.py`.
