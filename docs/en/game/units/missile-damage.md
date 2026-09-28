# Missile damage on infantry

[English](missile-damage.md) | [Русский](../../../ru/game/units/missile-damage.md) · [Units](README.md)

Measured 28.09.2026, three runs (`archer-range --range-mode damage`, distances
moved between lanes): Empire archers (90 men, 20 arrows each, missile damage 19),
20 m wide (18 m deep), fire at will at fearless Empire spearmen (120 men, 8280 HP,
armour 30), 30 m wide, standing and facing them. Distance: from the archers'
middle to the target's front rank.

| Distance | HP per arrow (three runs) | Arrows a second | Half the target's HP gone | Target destroyed |
|---|---|---|---|---|
| 70 m | 12.5 / 11.8 / 11.7 | 8.9–10.3 | 33–42 s | 108–124 s |
| 90 m | 10.4 / 9.7 / 9.9 | 8.3–9.6 | 43–54 s | 140–149 s |
| 105 m | 7.1 / 9.4 / 10.7 | 8.7–9.1 | 45–67 s | 136–187 s |
| 120 m | 9.9 / 10.4 / 8.1 | 9.0–9.5 | 43–57 s | 128–175 s |

- About 0.1 arrow a second per man.
- A thinned target takes less per arrow: HP per arrow ≈ c · h^0.3 (h — the
  target's share of HP left); c = 13.4 (70 m), 11.6 (90 m), 9.5 (105 m), 10.4 (120 m).
- 90–120 m: about the same damage (the spread between runs is larger than the
  difference); 70 m: 20–30 % more. Spread for one distance: up to ±20 %.

Not checked: shields, flank and rear, height, other shooters and targets, moving
or routing targets.
