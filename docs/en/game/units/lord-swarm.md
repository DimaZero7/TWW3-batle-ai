# A lord surrounded

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Units](README.md) › A lord surrounded · [Русский](../../../ru/game/units/lord-swarm.md)

How much a lord loses when one to four infantry units attack him from different sides, and what
an armour-piercing unit or the enemy lord adds. The user's question (01.10.2026): damage is dealt
by the soldiers that reach the lord and fight, not by the whole unit, so surrounding him with
four units may gain little. Measured 01.10.2026 in 3 battles of the
[lord swarm probe](../../apps/entries.md#lord_swarm), Normal difficulty, ×20, game v9.0.1.

![Lord HP lost per second by layout: game and simulator](../../../../research/evidence/units/lord-swarm-20261001/lord-swarm.png)

## Short answer

- **The number of units does not matter.** A lord standing in a ring of 1, 2, 3 or 4 spear units
  loses the same: the Empire General 7.8 / 8.3 / 8.7 / 7.4 HP/s against clanrat spearmen, the
  Skaven Warlord 6.2 / 5.8 / 5.1 / 4.8 against Empire spearmen.
- **About 4–5 enemy soldiers stand within 2.5 m of him** (7–10 within 3 m), however many units
  attack; the units share that ring.
- **Direction does not matter.** Spearmen on his back or flank hurt him as much as on his
  front (front only 7.8, front and back 8.3, front and flank 7.4 for the General).
- **Armour-piercing damage counts by its share of the ring.** Stormvermin halberds alone take
  21.5 HP/s from the General (2.8× clanrat spearmen); with one spear unit 16.1, with three 10.7 —
  the mean of the units' own rates (15 and 11). Against the Warlord: halberdiers alone 11.2
  (1.8× spearmen), with one spear unit 9.3, with three 7.0.
- **The enemy lord on top is a different matter.** Lord against lord is noisy (one blow is
  ~230 HP): the General takes 14.3 HP/s from the Warlord, the Warlord 33.3 from the General.
  With infantry added the loss is roughly the lord's plus the infantry's (27–34 HP/s), but the
  spread is ±10 HP/s.

## How it was measured

On the flat MP Crossroads map two lanes 600 m apart: the Empire General attacked by Skaven clanrat
spearmen (160 men, 30 m front) and Stormvermin halberds, the Skaven Warlord attacked by Empire
spearmen (120 men) and halberdiers. Both sides are held by script and fearless, so no lord uses
an ability (General's passive Hold the Line stays). A trial teleports the attackers 20 m from the
lord, facing him, from the front, back, left or right; 3 s later they are ordered to attack him
and he is told to halt (he fights back by himself). The trial ends 40 s after his first contact.
Between trials every unit is healed and its fatigue cut to a tenth; 8 spear units a side rotate
so each fights few trials. The rate is measured from 15 s after contact (the charge has faded) to
the end.

| Battle | Layouts | Repeats |
|---|---|---|
| `20261001-101343` | 8 infantry layouts, both lanes at once | 2 |
| `20261001-102111` | the other lord alone, with 1 or 3 spear units, with a halberd unit | 3 |
| `20261001-102508` | 8 infantry layouts | 3 |

## Results

Steady HP/s of the lord (mean, min–max over the repeats) and the attacking soldiers within 2.5 m
of him (mean of the per-second counts).

| Attackers | General: HP/s | soldiers ≤2.5 m | Warlord: HP/s | soldiers ≤2.5 m |
|---|---:|---:|---:|---:|
| 1 spear unit, front | 7.8 (5.4–9.2) | 4.1 | 6.2 (4.6–8.2) | 2.7 |
| 2: front, back | 8.3 (6.0–11.2) | 4.0 | 5.8 (3.5–7.4) | 4.7 |
| 2: front, left | 7.4 (6.8–8.3) | 4.0 | 6.8 (4.8–9.4) | 4.5 |
| 3: front, left, right | 8.7 (6.9–10.1) | 3.9 | 5.1 (3.4–8.4) | 5.1 |
| 4: all sides | 7.4 (6.8–8.0) | 5.1 | 4.8 (2.9–6.3) | 6.3 |
| halberds, front | 21.5 (11.6–30.6) | 2.6 | 11.2 (5.3–20.1) | 2.9 |
| spear front, halberds back | 16.1 (10.4–24.5) | 3.8 | 9.3 (7.2–15.4) | 4.5 |
| 3 spear, halberds back | 10.7 (8.0–14.0) | 4.5 | 7.0 (4.4–9.6) | 6.3 |
| the other lord | 14.3 (7.7–19.7) | — | 33.3 (25.1–38.1) | — |
| lord + 1 spear unit | 26.9 (13.0–40.3) | 5.1 | 31.8 (21.4–40.4) | 3.2 |
| lord + 3 spear units | 29.8 (19.6–47.9) | 4.5 | 27.6 (23.3–31.9) | 3.8 |
| lord + halberds | 34.0 (28.3–39.6) | 3.8 | 34.5 (28.7–43.3) | 2.5 |

Infantry layouts: 5 repeats; lord layouts: 3. Every attacking unit reports `is_in_melee` for
85–100 % of the time even when only one of its soldiers stands near the lord: the flag says
nothing about how many of its men fight him. The lord kills 0.3–0.5 men a second whatever the
layout.

## Per-soldier data: what the game gives

- **CA's script API** (`battle_unit`, `unitcontroller`, `script_unit` in `data_script.pack`) has
  per-unit reads only: men alive, health, `is_in_melee`, kills, position, bearing.
- **Per soldier** there is only the CCO list `CcoBattleUnit.ManList` (`EntityList`) of
  `CcoBattleEntity`: its fields (names registered in the game's executable) are `Position`,
  alive, `IsMan` / `IsEngine`, reload (`IsReloading`, `ReloadPercent`, `ReloadRemainingTime`),
  the entity record and the unit. No per-soldier health, target, combat state or kills.
- **Per unit** the CCO has `HealthValue` (absolute HP), `NumKills`, `IsInMelee`,
  `DamageInflictedRecently`; `DamageDealt` returns nothing on a battle unit in this build.
- So the probe reads, every 0.2 s of game time, each unit's health, men, kills, melee flag and
  place, and every 1 s every attacking soldier's position, counting those within 1.5–6 m of the
  lord. Who of them strikes is not readable.

## In the simulator

`tools/nn/sim/melee.py`, `config/nn/sim.json` `contact` ([simulator](../../training/simulator.md)):

- at most `lord_max_attackers` = 9 men strike a lord in all, however many units surround him,
  shared by the units in contact; their rates add up (the old rule took the strongest unit and
  0.35 of the others);
- `lord_direction` = 0: the flank and rear do not raise the hit chance on a lord;
- an enemy lord among the attackers keeps his own blow (the cap used to squeeze him out with the
  infantry, so a lord fought by a lord and three units lost half of what he lost to the lord
  alone);
- `lord_rival_others` = 0.35: with the enemy lord on him the infantry counts at 0.35 (from the
  whole battles; the probe's lord trials suggest 0.7 ± 0.5);
- `lord_incidental` = 0.4: a unit told to attack another enemy strikes a lord it only touches at
  0.4 (whole battles: 1.7 HP/s against 4.1 when he is its target).

## Limits

- One lord type a side, spear and halberd infantry only; no cavalry, monsters, other lords.
- Lords are held by script: in a real battle the game's AI uses abilities (Stand Your Ground,
  Deadly Onslaught), the lord moves and gets tired.
- The lord-against-lord rate is asymmetric and noisy (General on the Warlord 33 HP/s, the
  reverse 14) and is not explained by the database's numbers (both ~20 HP/s by the rule).
- Data: `build/lord-swarm/runs/` (not in Git); `python -m tools.nn.lord_swarm` prints the table,
  `--sim` the simulator on the same trials (needs torch: `bash tools/nn/dock.sh tools.nn.lord_swarm --sim`).
