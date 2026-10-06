# Melee

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Melee · [Русский](../../../ru/game/mechanics/melee.md)

How the game resolves a melee blow: hit chance, damage, armour, charge, bracing, mass and
splash. Collected from the web on 02.10.2026; the conventions (confidence, "Ours:") are in the
[index](README.md). Our own measurements: [melee](../units/melee.md), [simulator](../../training/simulator.md).

## Hit chance

- **Formula.** A blow hits when a 1–100 roll is at or below
  35 + melee attack + bonus vs the target's type + the current charge bonus − melee defence,
  clamped to 8–90 %. WH3, CA blog 2023 · [CA damage blog][dmg] · high ·
  `_kv_rules` `melee_hit_chance_base` 35, `_min` 8, `_max` 90.
  - Ours: the simulator keeps 35 / 8 / 90 but counts attack − defence at 0.1 of the rule
    (`melee.hit_slope` 0.1): the four measured pairs need a nearly flat 30–38 % whatever
    attack − defence is (−35…+31). **Conflict** with CA's formula; see the possible causes below.
  - Ours: **lord against lord by the formula** (`contact.lord_hit_slope` 1). In duels of two lords alone on
    the field the blows were counted as health drops (28 battles, 06.10.2026): General on General hits 41 % of
    his blows (formula 40 %: 55 against 45 + 5 Hold the Line), Warlord on Warlord 19 % (formula 30 % fresh,
    ~22 % with fatigue); a blow takes 245 / 223 HP — exactly the passport and armour at its mean 75 %. The
    flat slope gave both ~35 %. So CA's formula holds, and formations flatten it by something of their own
    ([simulator](../../training/simulator.md#lords)).
- **The stats in the formula are the modified ones**: abilities, fatigue, flank/rear
  multipliers and difficulty are applied before the roll. WH2–WH3 · [fandom Melee Attack][fw-ma],
  [CA damage blog][dmg] · medium.
- **`melee_hit_chance_normalisation_coefficient` 1** is "the coefficient that normalises the
  hit-chance calculation"; at 1 it changes nothing. WH2 description, value the same in WH3 ·
  [twwstats kv][tws] · high (value), low (meaning).
- **Outnumbered penalty off.** `melee_outnumbered_defence_penalty` is a multiplier on the
  defence of an entity attacked by several attackers; 1 = no penalty. WH2 description; Rome II
  notes describe the same key · [twwstats kv][tws], [MARS notes][mars] · medium.
- **Formed defence bonus.** `melee_defence_formed_attack_bonus` is added to the defender's
  defence while it is in "formed attack" (fighting in formation). WH2 0, WH3 5 ·
  [twwstats kv][tws] · high (value), medium (meaning).
  - Ours: not modelled (it would only shift every formed unit's defence by 5).

## Damage and armour

- **Split.** A weapon has a base part and an armour-piercing (AP) part. Bonus vs
  infantry/large and the charge bonus are added at the weapon's own AP ratio (e.g. +12 at 77 %
  AP ≈ +3 base, +9 AP). Older guides said the charge part was all non-AP: outdated. WH3 2023 ·
  [CA damage blog][dmg] · high.
  - Ours: agrees — `melee.py` splits charge bonus and bonus vs type by the weapon's share.
- **Armour roll.** Armour removes a uniform random 50–100 % of its value (as %) from the
  *base* part only; a result above 100 % counts as 100 %, so armour 200+ stops all base damage.
  AP damage ignores armour. WH3 · [CA damage blog][dmg], [WH3 kv guide][g3] · high ·
  `armour_roll_lower_cap` 0.5.
  - Ours: agrees (mean 75 % of armour; our units have armour ≤ 90, so the 100 % cap never bites).
- **Order of modifiers**: base → bonus vs type → charge → height → armour (base only) →
  resistances (summed, capped at 90 %) → rounding (base and AP rounded separately). Overkill
  is lost. Base damage has no floor (can be 0). WH3 · [CA damage blog][dmg] · high.
- **Height in melee.** Damage changes with the height difference per entity pair, reaching the
  full ±30 % already at 1 m. WH2–WH3 · [CA elevation blog][elev] · high ·
  `melee_height_damage_modifier_max_coefficient` 0.3, `_max_difference` 1 m. See [terrain](terrain.md).
  - Ours: not modelled (flat map).

## How often a man strikes, and who strikes

- **Attack interval.** `melee_attack_interval` 4 is "the delay after a weapon hit before the
  next attack animation can start" — so a full cycle is 4 s *plus* the animation. A per-weapon
  column `melee_attack_interval` in `melee_weapons_tables` overrides it (vanilla ~3.2–6 s; most
  entities ~3.8 s). Being interrupted does not reset the timer. WH2 description; WH3 modders ·
  [twwstats kv][tws], [modder thread][interval], [WH3 kv guide][g3] · high (description), medium (range).
  - Ours: the simulator uses the passport's `attack_interval_s`; measured lord blows are one per
    4 s (0.25 events/s, [measurements](../../training/measurements.md)). Agreement.
- **Only models that reach strike.** Damage needs an attack animation that physically reaches
  a target; any model with an enemy within weapon reach tries to attack, so a wider front means
  more attackers. WH1–WH3 guides · [WH3 kv guide][g3], [WH1 kv guide][g1] · medium.
  - Ours: agrees in kind — 0.75 of the files in contact strike (`melee.fighting_files`), i.e.
    13–18 % of a 120-man unit; the database has no such number.
- **Matched combat.** A share of blows is played as paired "matched combat" animations
  instead of free strikes: `matched_combat_percentage` WH2 100 → WH3 50; in a formed charge 20 %,
  formed vs formed 40 %, formed vs free 60 %. An entity below 15 % HP cannot start one; while in
  one, damage from outside is clamped so it does not die before it ends (5 %). WH2 descriptions,
  WH3 values · [twwstats kv][tws] · high (values), medium (effect on the kill rate).
  - Ours: not modelled. Long paired animations are one candidate for why fewer blows land than
    the formula expects.
- **Secondary attacker.** `melee_secondary_attack_probability` 1: a second attacker on the same
  target is always allowed to strike. WH2 description · [twwstats kv][tws] · medium.
- **Hit reactions.** After `melee_max_hit_reactions` 2 interrupts in a row an entity ignores
  further hit reactions so it can attack. WH2 description · [twwstats kv][tws] · high.

## Charge

- **Charge bonus.** Added in full to melee attack and to weapon damage on the first blow, then
  falls linearly to 0 over 13 s (half at 6.5 s). WH3, CA 2023; Rome II notes also say "linear" ·
  [CA damage blog][dmg], [MARS notes][mars] · high · `charge_decay_duration` 13 (WH2 name
  `charge_cool_down_time`: "time over which to fade out charge bonus after charge complete").
  WH1-era guides said 15 s (outdated).
  - Ours: agrees (charge bonus to attack and damage, 13 s). On top, the simulator multiplies a
    charging unit's damage by 1 + 1.5 × speed share for those 13 s (`melee.impact`) — a fitted
    stand-in for collision and more men reaching; the game has no such rule (see collisions).
- **Only a real charge counts.** The bonus applies only if the entity reaches its hidden charge
  speed before contact; walking into melee gives none. Charge speed is ~20–40 % above run speed.
  WH3 community · [WH3 kv guide][g3], [charge speed thread][chspeed] · medium.
- **Charge distances** are per entity (`battle_entities`): `charge_distance_commence_run`,
  `_adopt_charge_pose`, `_pick_target`; typical switch to charge speed 20–60 m, pose ~35 m.
  WH3 · [twwstats][tws], [charge distance thread][chdist] · high (fields), medium (values).
- **Blocked chargers don't charge.** Since WH2 1.12.1 an entity whose path is blocked by a
  friend makes no charge attack (back ranks of a counter-charge): ~35 % less counter-charge
  damage. Still the WH3 base · [1.12.1 notes][p1121] · high.
- **Charge morale.** A charging unit gets +15 leadership; `charge_timeout` 60 s is the timeout of
  that morale bonus. WH2 descriptions, same values in WH3 · [twwstats kv][tws] · high (values),
  medium (duration: guides say ~10 s). See [morale](morale.md).
  - Ours: not modelled.
- **Pursuit is a charge.** Every blow of a pursuer on a routing unit counts as a charge
  (`pursuit_charge_bonus_modifier` 1). Patch 5.0 removed the old rule that only 35 % of a
  pursuing unit could attack routers. WH2 description; WH3 patch notes · [twwstats kv][tws],
  [fandom Leadership][fw-lead] · high.
  - Ours: a routing unit is struck as from the rear (defence ×0.3), by every man in contact at 0.43
    of the rule, without the charge bonus: in the recordings a router chased by one pursuer loses 0.64
    of what a standing target loses, and the first 5 s of a pursuit hit no harder than later (17.9
    against 18.2 HP/s).
- **Devastating Flanker** doubles the charge bonus against flank/rear
  (`devastating_flanker_charge_multiplier` 2); **Glorious Charge** (4.2.0, Empire/Kislev) doubles
  its duration and applies the flank morale penalty to the target. WH3 · fandom (snippets) · medium.

## Bracing and charge defence

- **What bracing is.** A standing unit facing a charger multiplies its entities' mass against
  frontal impacts: up to ×4 at 8 ranks deep, falling linearly to ×1 at 1 rank; ×2 more for
  charge reflectors; global clamp ×10. Only within 80° either side of the facing. WH2
  descriptions, WH3 values · [twwstats kv][tws], [1.12.1 notes][p1121] · high ·
  `bracing_calibration_ranks` 8, `_ranks_multiplier` 4, `bracing_charge_reflector_bonus` 2,
  `bracing_max_multiplier_clamp` 10, `bracing_attack_angle` 80 (half-angle). WH3 adds
  `unit_braced_status_threshold` 0.75 (meaning not documented).
- **600 is a threshold, not a bonus.** `bracing_immovable_by_small_entity_mass` 600: once the
  braced mass passes 600, small soft-collision entities cannot push the entity. The forum
  reading "+600 mass when braced" is wrong. WH2 description · [twwstats kv][tws] · high.
- **Charge Defence** (vs large / vs all) cancels the attacker's charge bonus while braced; it does
  not reduce collision damage. WH2 1.12.1 → WH3 · [1.12.1 notes][p1121], fandom · high.
- **Charge Reflection** (WH3): while braced, the unit deals double weapon damage to chargers
  whose charge factor is still ≥ 0.7, i.e. during the first 13 × 0.3 = 3.9 s; it does not stop
  them and reflects no impact damage. Update 1.1 made it require bracing. WH3 · [CA damage blog][dmg]
  (3.9 s), [reflection thread][reflect] · high (3.9 s, ×2), the 0.7 link is our inference ·
  `charge_reflect_damage_multiplier` 2, `charge_reflect_min_charge_factor_threshold` 0.7.
  - Ours: a braced spearman unit (slower than 0.5 m/s, `melee.braced_speed`) meets a frontal
    infantry charge "as a charge" for 13 s, and `sim.json` notes the 0.7 threshold is not
    applied. **Conflict in form**: the game doubles weapon damage for 3.9 s; the simulator was
    fitted to the measured outcome (braced spearmen charged head-on lose 0.80 of what the
    charger loses).
- Update 1.1 also fixed impact maths so the defender uses its own mass (it used the attacker's). WH3 ·
  CA notes via press · high.

## Mass, collisions, knockdowns

- **Collision damage** = (collision power)^0.8 × 0.6, capped at 70; 70 % of it ignores armour;
  the same attacker can't deal it again for 4 s; no charge bonus in it since WH2 1.12.1.
  Collision power grows with both entities' speed and the mass ratio. WH2 descriptions, WH3 values ·
  [twwstats kv][tws], [1.12.1 notes][p1121] · high · `collision_damage_*`.
- **Knock chances.** Base knockback chance per melee hit = 0.1 × attacker/defender mass ratio,
  knockdown 0.05 × ratio, ×2 (scaled by the charge factor) when charging. No knockback below a
  mass ratio of 0.35, no knockdown or step-back below 0.5, no hit reaction below 0.4. WH2
  descriptions, WH3 values · [twwstats kv][tws] · high.
- **Collision thresholds.** An instantaneous speed change of 3 / 4 / 7 / 10.5 m/s triggers
  step-back / knock-back / knock-down / knocked flying (shockwaves: half those). Get-up delay is
  random 1–2.7 s (was 0.7–2 before WH2 1.12.1). · [twwstats kv][tws] · high.
- **Collision attacks** (5.1.0, June 2024): an alternative weapon attack made on contact while
  charging or attacking, with per-unit cooldown and max targets, damage from weapon strength;
  6.0.0 set multi-entity chariots to 2 targets / 2 s. · [5.1.0 blog][p510] · high. See [cavalry](cavalry.md).
- **Masses** are hidden `battle_entities.mass`: infantry ~90–150, Black Orcs/Chosen higher;
  monstrous infantry ~1700–1900; chariots 1000–3000 by class (5.3); dragons 4000–5900 (7.0). ·
  patch notes · high.
  - Ours: passports carry mass (General 600, Warlord 650, spearmen 100, Stormvermin 150); the
    simulator does not use it (no knockdowns, no collisions).

## Splash

- **Damage is divided.** One splash attack's weapon damage is shared equally among the entities
  it hits (each rolls its own hit chance), up to `max_splash_targets` and only on targets no
  larger than its splash target size. WH3 · [splash thread][splash], [WH3 kv guide][g3] · medium.
- **6.0.0 rule:** splash targets = weapon strength / 100, rounded, at least 1 (hotfixed to apply to
  single entities only). Before: infantry 1, monstrous infantry ~4, lords/heroes ~6, monsters
  10–12; 5.1/5.3 cut many monster caps (e.g. Great Unclean One 12 → 5). · CA 6.0.0 notes
  (via [fandom][fw-60]), [5.3.0 notes][p530] · high.
  - Ours: the General's passport splash is 4, matching 430 weapon strength / 100. **Conflict in
    form**: the simulator gives each splash target the full per-hit damage (capped at a man's
    health) instead of dividing it. While damage / targets still exceeds a man's HP (General vs
    spearmen: 430 / 4 ≫ 69) the kills are the same; it matters against tougher men.

## Unit size

- Single entities' health and damage and direct-damage spells scale with the unit-size setting:
  Small 25 %, Medium 50 %, Large 75 %, Ultra 100 %; models per unit scale the same way. WH3 is
  balanced around Ultra. · [GameWatcher, CA's James Martin][gw], [WH3 kv guide][g3] · high/medium.
  See [battle rules](battle-rules.md).
  - Ours: the arena units have Ultra sizes (spearmen 120, clanrats 160, slaves 180).

## Not found

No public source explains `melee_breakoff_secs` 24 / `melee_breakoff_total_immunity_secs` 10
(WH3-only keys; see [movement](movement.md)), `entity_action_attack_formed_combat_distance`
2.5 / `_tether_distance` 2, or how many models can physically reach one target.

## Possible causes of the flat hit chance (for our simulator)

Not established; listed so that tests can target them: (1) a blow cycle longer than 4 s
(the delay starts after the hit, plus the animation); (2) matched combat in 50 % of blows (long
paired animations, a target below 15 % HP not eligible); (3) only entities whose animation
physically connects deal damage, so attackers with high hit chance may still "miss" by reach;
(4) per-entity defence multipliers from the flank inside a melee blob (see [flanking](flanking.md)).

[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[g1]: https://steamcommunity.com/sharedfiles/filedetails/?id=719168880
[tws]: https://twwstats.com/kv/rules
[mars]: https://community.creative-assembly.com/total-war/total-war/forums/118-mars-first-beta/threads/15583-mars-first-beta-patch-notes
[fw-ma]: https://totalwarwarhammer.fandom.com/wiki/Melee_Attack
[fw-lead]: https://totalwarwarhammer.fandom.com/wiki/Leadership
[fw-60]: https://totalwarwarhammer.fandom.com/wiki/Update_6.0
[interval]: https://steamcommunity.com/workshop/filedetails/discussion/3535265856/591777966325959448
[chspeed]: https://steamcommunity.com/app/1142710/discussions/0/3589960830790650436/
[chdist]: https://steamcommunity.com/app/1142710/discussions/0/4630357120384001354/
[p1121]: https://updatecrazy.com/total-war-warhammer-2-update-1-12-1-patch-notes-sep-8-2021/
[reflect]: https://steamcommunity.com/app/1142710/discussions/0/688619343242657770/
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[splash]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
