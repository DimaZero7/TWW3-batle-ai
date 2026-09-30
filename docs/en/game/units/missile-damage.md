# Missile damage

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Units](README.md) › Missile damage · [Русский](../../../ru/game/units/missile-damage.md)

How much health an Empire archers' arrow takes, how often they shoot and at
whom. Three sources:

1. the arrow's numbers from the game's database (30.09.2026, [the game's database](../database.md));
2. a range test: archers shoot at standing spearmen (28.09.2026);
3. arena battle recordings: the first 13 fair battles on 30.09.2026
   (`20260930-130637` … `-132200`, Normal difficulty), 7522 s of records; 5 more
   were recorded after the analysis ([data for training](../../training/README.md)).

## The arrow in the game's database

`projectiles_tables`, `wh2_dlc13_emp_bow_arrow` (archers
`wh2_dlc13_emp_inf_archers_0`):

| What | Value |
|---|---|
| Range | 130 m |
| Damage | 17 + 2 armour-piercing = 19 (as on the card) |
| Reload | 10 s — at most 0.1 arrow a man a second |
| Arrows a man | 20 (the card, [roster](roster.md)) |

From `_kv_rules_tables`: a shield holds arrows up to 60° from the front
(`shield_defence_angle_missile`); fire at will may shoot into melee
(`allow_fire_at_will_into_melee` 1). The same table has
`missile_lethality_coefficient_effective_range` 40,
`missile_lethality_coefficient_extreme_range` 120,
`missile_armour_piercing_coefficient` 0.5 and `missile_shield_piercing_coefficient`
0.5 — their meaning was not checked. Under fire a unit loses 5
[morale](morale.md) points.

## Range test (28.09.2026)

Three runs (`python -m tools.build archer-range --range-mode damage
--damage-rotate 0|2|1 --speed 20`): in four lanes 200 m apart Empire archers
(`wh2_dlc13_emp_inf_archers_0`, 90 men, 20 arrows each, card damage 19), 20 m
wide (18 m deep), fire at will at Empire spearmen (`wh_main_emp_inf_spearmen_0`,
120 men, 8280 HP, armour 30) that stand still 30 m wide facing the archers and
never run (`morale_behavior_fearless`). Each run moves the distances between
lanes. Map The Moorlands Route, south (flat field).

Distance: from the archers' middle to the target's front rank.

| Distance | HP per arrow (three runs) | Arrows a second | Half the target's HP gone | Target destroyed |
|---|---|---|---|---|
| 70 m | 12.5 / 11.8 / 11.7 | 8.9–10.3 | 33–42 s | 108–124 s |
| 90 m | 10.4 / 9.7 / 9.9 | 8.3–9.6 | 43–54 s | 140–149 s |
| 105 m | 7.1 / 9.4 / 10.7 | 8.7–9.1 | 45–67 s | 136–187 s |
| 120 m | 9.9 / 10.4 / 8.1 | 9.0–9.5 | 43–57 s | 128–175 s |

HP per arrow: the health the target lost divided by the arrows shot (misses
included), until half its HP was gone.

- **Rate:** a unit shoots about 9 arrows a second — about 0.1 arrow a second a
  man. The archers lost no men (nobody shot back).
- **A thinned target takes less:** HP per arrow with the target's health left at
  75–100 % — 9.0–12.9; 50–75 % — 8.6–11.5; 25–50 % — 6.9–9.8; below 25 % —
  4.7–6.5. Fit over all lanes: HP per arrow ≈ c · h^0.3, h — the target's share
  of health left; c = 13.4 (70 m), 11.6 (90 m), 9.5 (105 m), 10.4 (120 m).
- **Distance:** from 90 to 120 m the damage per arrow is about the same (the
  spread between runs is larger than the difference); at 70 m about 20–30 % more.
- **Spread:** one distance in different lanes and runs — up to ±20 %.

Run logs: `build/archer-range/runs/20260928-131003`, `-131159`, `-131244` (not in
Git).

## In battle: arena recordings (30.09.2026)

```mermaid
flowchart LR
  stop["The unit stopped,<br/>an enemy within 130 m"] -- "4–5 s" --> first["First arrows"]
  first --> rate["Then ~0.073 arrows<br/>a man a second"]
  move["The unit walks"] -. "does not shoot<br/>or reload" .-> stop
```

- **First volley.** After the unit stops with an enemy in range, the first
  arrows leave after 4–5 s (median; 2–7 s): archers do not reload on the move.
  After a pause without moving — about 1 s.
- **Rate.** In battle a unit looses 0.073 arrows a man a second (median;
  0.05–0.097) — less than the 0.1 of the reload and of the range test.
- **Distance.** HP full-health spearmen lose per arrow (neighbouring distances
  are pooled where the damage would rise with distance):

  | Distance | 34 m | 52 m | 93 m | 130 m |
  |---|---:|---:|---:|---:|
  | HP per arrow | ~25 | ~18 | ~11 | ~10 |

  Close range is almost twice as deadly. At 93 m this agrees with the range test
  (9.7–10.4 at 90 m).
- **Target.** Compared with spearmen: archers take 0.71–0.78 of the damage, the
  lord 0.02–0.1 (armour 85).
- **Shield — fitted, not measured.** A shield covers 60° either side of the
  front (`shield_defence_angle_missile` in [the game's database](../database.md)).
  The figure "a shielded target hit from the side or rear takes about 1.54 times
  more" cannot be checked from these recordings: in the arena only the lord has a
  shield ([catalogue](catalog/empire.md)), and he takes only 0.02–0.1 of the
  spearmen's damage. 1.54 = 1 / (1 − 0.35) looks like a fitted parameter. It needs
  a separate test with shields.
- **Fire-at-will target choice.** The nearest enemy in range 55 % of the time,
  the second nearest 19 %. When the nearest is the lord, archers shoot him 65 %
  of the time. When there are enemies in melee, 57 % of shots go at them.

## Not checked

Height, other shooters and targets, shooting at running and walking units
separately.
