# Artillery and war machines

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Artillery · [Русский](../../../ru/game/mechanics/artillery.md)

Artillery accuracy and arcs, line of sight, damage and morale, crews, armoured vehicles.
General missile rules: [missiles](missiles.md). Chariots: [cavalry](cavalry.md). Conventions:
[index](README.md). Ours: not modelled.

## Accuracy and fire

- **Accuracy** = projectile marksmanship bonus + unit accuracy; beyond the calibration distance
  shots spread much more; the calibration area is the aim circle's radius (artillery: large).
  Accuracy is additive (Ballistics Calibration became +60 accuracy in WH2). · [tw-modding][twm],
  fandom · high.
- **Direct vs indirect.** Direct-fire pieces (cannons, guns; trajectory `low`) need a clear line of
  sight and won't fire through friends; arcing pieces (mortars, catapults; `fixed`) fire over
  troops and walls but less accurately; `dual_low_fixed` falls back to an arc when blocked (the
  Medusa's arc was removed in 4.1.1 because of friendly fire). · [tw-modding][twm], fandom · high.
- **Cannons aim at the central entity** of the target unit; average damage about the same but
  volleys more consistent. Artillery leads moving targets and misses often. · [CA forum][cannon],
  Steam · medium.
- **Range** usually over 200 m; all missiles are more accurate under ~80 m. Calibration examples:
  Deathshrieker distance 240 → 380, area 180 → 10 (3.1.0). · fandom · medium/high.
- **Penetration and explosions.** `projectile_penetration` (e.g. `large_3`: through 3 large
  entities before exploding); explosions are separate records. · [tw-modding][twm] · high.
- **Shields don't block artillery** (`projectile_category` "artillery"; "misc" also ignores shields)
  unless the unit has Ballistic Plating (5.0, abilities only). · [tw-modding][twm], fandom · high.

## Morale and friendly fire

- Being targeted (not hit) by artillery −8, hit and damaged −10; a near miss is within 12 m
  (`artillery_near_miss_distance_squared` 144). Category "artillery" triggers both; 6.2.2 moved the
  Giant Blowpipes to "misc" to stop it. · [twwstats morale][tws-m], fandom · high.
- Arcing fire causes friendly fire easily; no-friendly-fire artillery still applied the "damaged by
  artillery" morale hit to friends (4.2.3 known issue). · [CA bug][artbug] · high.

## AI and artillery

- At Normal the AI doesn't dodge artillery; Very Hard does. 5.2.1 raised AI priority on vulnerable
  artillery. · [battle AI](battle-ai.md) · high.

## Crews and war machines

- Crews can abandon and re-man pieces; another crew can man an empty piece; ammunition is limited.
  Limbered pieces move but can't fire; unlimbered fire but can't move. Monster artillery (Cygor,
  Stegadon, Bone Giant) has no crew. · fandom, [CA script docs][chv] · high.
- **Armoured vehicles** (Steam Tank): Directional Shield blocks missiles 360° at ×1.0 / ×0.5 / ×0.1
  of its chance from front / side / rear (`armoured_vehicle_front/side/rear_block_modifier`): 70 / 35
  / 7 %. 5.0 rework: fires on the move, minimum range 5, reload 10; 5.3: no hull cannon in melee. ·
  [5.0.0 blog][p500], fandom · high.
- No repair mechanic; only healing abilities. · fandom · medium.

[twm]: https://tw-modding.com/wiki/Tutorial:Missiles_and_You
[cannon]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/8008-suggestion-cannon-type-artillery-units-should-not-prefer-the-central-target
[tws-m]: https://twwstats.com/kv/morale
[artbug]: https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/314-artillery-that-does-not-cause-friendly-fire-still-causes-morale-penalties-to-friendly-units
[chv]: https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html
[p500]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0
