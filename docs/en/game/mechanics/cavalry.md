# Cavalry and chariots

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Cavalry · [Русский](../../../ru/game/mechanics/cavalry.md)

Charges, cycle charging, mass, bracing against cavalry, anti-large, chariots and collision attacks.
The general charge and bracing rules are in [melee](melee.md). Conventions: [index](README.md).
Ours: the simulator has no cavalry yet ("Not modelled" in `simulator.md`).

## Charge

- **Charge bonus** adds in full to melee attack and weapon damage on impact, split at the weapon's
  AP ratio, and falls linearly to 0 over 13 s (by the linear rule ≈85 % after 2 s, ≈38 % after 8 s;
  one tester reports ~80 % and ~20 %). · [CA damage blog][dmg] · high; [Steam][cyc] · medium.
- **Cycle charging** works because the bonus is front-loaded: hit, pull out after ~6–15 s, re-charge
  from ~35 m. Leave right after impact while enemies are knocked down; repeated move orders are
  needed to break "unit glue". No WH3 patch removed cycle charging; 3.1.0 trimmed some Bretonnian
  charge bonuses and Lance Formation (40 → 30 %). · [Steam][cyc], [Steam][cyc2], fandom 3.1.0 · medium.
- **Charge vs impact.** Since WH2 1.12.1 the charge bonus is not part of collision (impact) damage;
  impact depends on mass and speed only. Collision damage = (power)^0.8 × 0.6, cap 70, 70 % AP,
  4 s per attacker (guides say 3 s — older). · [1.12.1 notes][p1121], [twwstats][tws] · high.
  The WH2 beta formula for power: (speed₁ + speed₂) × mass₁ / mass₂ · medium.
- **Charging into running or routing enemies**: pursuer attacks count as charges. · twwstats · high.
- **Downhill charges** hit much harder (more speed → more collision damage), uphill barely work. ·
  [CA elevation blog][elev] · high (direction), no numbers.
- **Charge morale** +15 for the charger. · [morale](morale.md) · high (value).

## Against braced infantry

- Bracing multiplies the defenders' mass against frontal impacts (×4 at 8 ranks, ×2 more for
  charge reflectors, within 80°); flank and rear charges ignore it. · [melee](melee.md#bracing-and-charge-defence) · high.
- **Charge Defence** cancels the attacker's charge bonus while braced; **Charge Reflection**
  doubles the braced unit's weapon damage against chargers for ~3.9 s; neither reflects impact
  damage, and the charger is not stopped by the attribute itself. · [reflection thread][reflect],
  [no impact reflection][refl2] · high/medium.
- `refusal_chance_vs_charge_reflectors` 0 / `_vs_pike_phalanx` 0: cavalry never refuses to charge. · twwstats · high.
- 8.0 fixed infantry bracing when cavalry was not heading at them (cavalry could slip past). · [8.0][p80] · high.
- **Ogre Charge** keeps half its charge bonus against braced units (`ogre_charge_factor` 0.5), the
  upgraded version loses only a quarter (0.25). · twwstats, fandom · high.

## Anti-large and size classes

- **Bonus vs large** adds its full value to melee attack and to weapon damage (damage part split
  at the weapon's AP ratio). Every entity is either large or small: cavalry, chariots, monstrous
  infantry/cavalry, monsters, war machines are large; small war beasts are small. No unit has both
  bonus vs infantry and vs large. · [Steam][antilarge], [Steam][size] · medium (tooltip confirms).
  - Ours: the simulator adds bonus vs infantry / vs large by the target's class. Agreement.
- **Forests**: large units without Strider fight at ~80 % melee attack and ~80 % speed and charge
  weaker. · [Steam][forest] · medium.
- Horses frightened −20, elephants frightened −30 morale (keys exist; which units cause it is not
  documented); `cavalry_effect_range` 20 m. · DB · high (values), low (use).

## Chariots and collision attacks

- **Chariots keep their charge bonus while moving**; drive them through instead of stopping. · Steam · medium.
- **Collision attacks** (5.1.0): an extra weapon attack on contact while charging/attacking, with a
  per-unit cooldown and max targets; damage from weapon strength (+ charge + bonus vs infantry,
  split among targets); about one a second; once the max targets are used, further contacts deal
  physics impact (cap 70). Multi-entity chariots ~3 targets / 2 s in 5.1; 6.0 set them to 2 / 2. ·
  [5.1.0 blog][p510], [Spellbound's tests][chariot] · high/medium.
- **Masses** (5.3): small chariots 1000, medium 1500, monstrous 2000, single-entity monstrous 3000;
  the mass that counts is one part (usually the mount), not the sum. Rough thresholds: ~1200 pushes
  through light infantry, ~2000 through mid-tier. · [5.3.0][p530], fandom 3.1.0, [chariot thread][chariot] · high/medium.
- 5.1 known issue: projectiles can hit articulated vehicles 2–3 times. · high.

[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[cyc]: https://steamcommunity.com/app/1142710/discussions/0/4357872738329493783
[cyc2]: https://steamcommunity.com/app/1142710/discussions/0/591758371482292288/
[p1121]: https://updatecrazy.com/total-war-warhammer-2-update-1-12-1-patch-notes-sep-8-2021/
[tws]: https://twwstats.com/kv/rules
[reflect]: https://steamcommunity.com/app/1142710/discussions/0/688619343242657770/
[refl2]: https://steamcommunity.com/app/1142710/discussions/0/4339851902698051173
[p80]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98-total-war-warhammer-iii-update-8-0-patch-notes
[antilarge]: https://steamcommunity.com/app/594570/discussions/0/1736588252370891593/
[size]: https://steamcommunity.com/app/594570/discussions/0/2797251375469539017/
[forest]: https://steamcommunity.com/app/594570/discussions/0/1744479063986685346/
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[chariot]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/2263-multi-entity-chariots
