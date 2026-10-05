# Flanking and rear attacks

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Flanking · [Русский](../../../ru/game/mechanics/flanking.md)

What a flank or rear attack changes: the defender's melee defence, shields, morale, and how the
game decides the direction. Conventions: [index](README.md).

## Melee defence

- **Multiplier, not subtraction.** When an entity is struck in the flank its melee defence for
  that blow is multiplied by 0.6, in the rear by 0.3 (−40 % / −70 %). The fandom wiki's
  "−30 % / −60 %" is wrong; WH1 used 0.25 for the rear. WH3 · [CA damage blog][dmg] (CA),
  [CA forum thread][t10108], [WH3 kv guide][g3] · high ·
  `melee_defence_direction_penalty_coefficient_flank` 0.6, `_rear` 0.3 (WH2 description:
  "coefficient by which defender's melee defence is multiplied").
  - Ours: the simulator uses ×0.6 / ×0.3, but the defence lost counts at `flank_slope` 2.0 /
    `rear_slope` 0.25 of the rule, fitted to the 28 whole battles where infantry fought on the
    flank lost 1.74× and on the rear 1.31× of the front. **Conflict**: by the database the rear
    must cost more than the flank; that count says the opposite. Counted apart, a lone attacker
    (infantry only, nobody shooting, contacts older than 10 s) takes 1.53× from the flank and
    1.92× from the rear ([measurements](../../training/measurements.md#flank-and-rear-a-lone-attacker)):
    the rear costs more, as the database says. The simulator keeps the old fit (its rule with the
    striker's own front fits these ratios but made whole battles worse: `simulator.md`).
- **Per model, by quadrant.** The penalty is decided for each attacking model against the struck
  entity: a blow from the target entity's left, right or rear quadrant is a flank/rear blow; it
  stops once that model turns to face its attacker. WH3 · [CA damage blog][dmg], [CA forum thread][t10108] ·
  high (per entity), medium (quadrant boundaries).
  - Ours: the simulator decides per unit, by the angle of the enemy's centre off the unit's
    facing: front within 60°, rear beyond 120° (`contact.front_deg`, `rear_deg`) — quadrants
    would put the boundaries at 45° / 135°. A single entity turns at once, which matches the
    measured "no flank/rear rule against a lord" (`contact.lord_direction` 0). **Partial conflict**
    (boundaries); agreement on lords.
- **Formations don't turn in melee.** Not stated by any source as a number; players note
  engaged units stay "glued". Ours: measured 1°/s turning in melee (`contact.melee_turn_deg_s` 2).

## Shields

- **Shields block missiles only, in the front arc.** A shield blocks its percentage of small-arms
  missiles that arrive within 60° either side of the facing (120° in all); it gives no melee
  benefit (any melee defence it implies is already in the card) and nothing from the flank or
  rear. WH1–WH3 · [WH1 kv guide][g1], [CA forum thread][t10108], [WH3 kv guide][g3] · medium ·
  `shield_defence_angle_missile` 60, `shield_defence_angle_melee` 60 (WH2 description: "shield
  defence half-angle vs melee" — the key exists, but no source shows a melee effect in Warhammer).
  - Ours: the simulator applies the shield to missiles within 60° of the front, not in melee. Agreement.
- **Directional Shield** (5.0.0, Steam Tank): 360° block at ×1.0 front, ×0.5 sides, ×0.1 rear
  (`armoured_vehicle_front/side/rear_block_modifier` 1 / 0.5 / 0.1). · [5.0.0 blog][p500],
  [fandom Directional Shield][fw-ds] · high. See [missiles](missiles.md).

## Morale

- **"Attacked in the flank / rear" is a first-contact effect.** The database's WH2 descriptions of
  `was_attacked_in_flank` −6 and `was_attacked_in_rear` −14 say "first contact from flank / rear"
  (`was_attacked_in_front` 0). Guides list −6 / −14 as stacking with "flanks exposed"; one player
  says the flank penalty lasts ~60 s. · [twwstats morale][tws-m], [fandom Flanking][fw-flank] ·
  high (values), medium (that it is tied to the first contact, from the key's description).
  - Ours: `morale.attacked_event=true`: −6 / −14 for one 0.5 s tick at first contact
    from a worse direction (`flank_hit` → `flank_event`). Threat flags `lf/rf/bf` do not
    determine hit direction. Continuous −1 / −2 remain only the fallback mode.
- **Flanks exposed / secure.** −3 if one flank is exposed, −6 if both; +5 if all flanks are secure.
  The "open flanks" range is 120 m (`open_flanks_effect_range`); neighbour range 120 m. ·
  [twwstats morale][tws-m], [fandom Flanking][fw-flank] · high ·
  `ume_concerned_flanks_exposed_single/_multiple`, `ume_encouraged_flanks_secure`.
  - Ours: −3 / −6 use the same `lf/rf/bf` that the network sees; the companion passes
    native `is_left_flank_threatened/is_right_flank_threatened/is_rear_flank_threatened`.
    Production geometry is ON (`threat.calibration.on=true`): non-routing enemy within
    45 m, sectors 60°/150°, enemy facing the unit within 60°, after movement and turning.
    Rear inputs are substantially closer to the game; side/out-of-melee rates still miss
    noise. Game-AI checks fall 23 → 20/26 with mechanics 51/54; adoption prioritizes
    direct input measurements and does not claim complete calibration.
    [Results and rejected trials](../../training/simulator.md#threat-flag-calibration).
- **Surrounded** is not a separate modifier in the tables; it is the sum of flank/rear contacts,
  exposed flanks and "a strong enemy near" (−3…−24). One test: removing the enemies around a unit
  raised it from 14 to 75 points. · [Steam thread][surround] · low.
- **Glorious Charge** (4.2.0) makes the charged target take the flank-attack morale penalty even
  from the front. · [fandom Glorious Charge][fw-gc] · high.

## Charges into the flank

- **Devastating Flanker** doubles the charge bonus against flank/rear. ·
  `devastating_flanker_charge_multiplier` 2 · medium.
- **Bracing works only in front.** Flank and rear impacts ignore bracing mass and charge defence.
  · [1.12.1 notes][p1121] · high.
  - Ours: braced only within `bracing_attack_angle` 80° of the front. Agreement.

## Not found

Exact sector boundaries in degrees (quadrant = 90° sectors is our reading), how long
"attacked in the flank" lasts, and any "encirclement" mechanic beyond the sum above.

[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[t10108]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/10108-flank-attack-damage-increase
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[g1]: https://steamcommunity.com/sharedfiles/filedetails/?id=719168880
[tws-m]: https://twwstats.com/kv/morale
[fw-flank]: https://totalwarwarhammer.fandom.com/wiki/Flanking
[fw-ds]: https://totalwarwarhammer.fandom.com/wiki/Directional_Shield
[fw-gc]: https://totalwarwarhammer.fandom.com/wiki/Glorious_Charge
[p500]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0
[p1121]: https://updatecrazy.com/total-war-warhammer-2-update-1-12-1-patch-notes-sep-8-2021/
[surround]: https://steamcommunity.com/app/594570/discussions/0/2266943550302453936/
