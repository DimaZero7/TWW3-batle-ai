# Missiles

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Missiles · [Русский](../../../ru/game/mechanics/missiles.md)

How shooting works: accuracy, range, shields, armour, friendly fire, reload, ammunition.
Artillery is in [artillery](artillery.md). Conventions: [index](README.md). Our measurements:
[missile damage](../units/missile-damage.md), [missile range](../units/missile-range.md),
[missile probe](../units/missile-probe.md) (rate, range, arc, hits: the game and the simulator).

## Accuracy and hitting

- **Projectiles are physical.** A shot flies as a body and must touch a model; there is no
  to-hit roll against the target's stats, so melee defence does nothing against missiles and
  big or densely packed targets catch more shots. WH1–WH3 · [Steam thread][phys], [tw-modding][twm] · medium.
- **Accuracy.** The chance of an exact hit is the unit's accuracy (`land_units`) plus the
  projectile's marksmanship bonus (`projectiles_tables`), in %. A shot that misses lands at random
  inside the "calibration area" around the aim point and can still hit someone. · [tw-modding][twm] · medium.
  Accuracy and marksmanship add up and act alike, the spread ~ A / x + B of their sum x, visibly between 0
  and 100 · [Steam][acc] · medium.
- **Calibration distance.** Each projectile has a range at which it has full accuracy (usually
  ~60–80 % of its maximum range); beyond it shots spread more. The "calibration area" is "the size of the
  circle that the projectile is calibrated to and lands in"; whether a radius, an area or a spread limit is
  said nowhere. · [tw-modding][twm] · medium.
  - Ours: the hit comes from the database's spread model (`missile.hit_chance`): a shot aimed at a random man
    of the target lands off by σ = k × calibration area × d / calibration distance × √(1 − (accuracy +
    marksmanship) / 100); a man is hit within his radius or in his "shadow" height / tan(the angle it comes
    down at, from the muzzle velocity); a formation by its files and ranks with the database's spacing (a loose
    or thinned formation catches less), a lone man by a circle. k = 1.1 is the one calibration: it does not
    follow from the definition, it is fitted on 62k recorded shots and confirmed by the probe (29 lanes: error
    0.154 against 0.406 with the old measured rates).
- **No damage fall-off with range** in Warhammer, only spread (Attila had 1/5 damage past maximum
  range). The keys `missile_lethality_coefficient_effective_range` 40 / `_extreme_range` 120 are
  described as "lethality due to range, inside / beyond effective range" — whether WH uses them is
  unknown. · [Steam thread][phys], [twwstats][tws] · low.
- **Aim error on very large targets**: `projectile_aiming_target_size_modifier` 0.9 adds a random
  aim-point error by the target's radius/height. · [twwstats][tws] (WH2 description) · high.
- **Cover.** `missile_target_in_cover_penalty` 0.2 is subtracted from hit chance against a target in
  cover. · [twwstats][tws] · high (description), low (whether WH3 uses "cover").
- **Trajectories.** `low` (fixed speed, flat; guns), `fixed` (fixed angle, variable speed; arcing),
  `dual_low_fixed` (flat, switching to an arc when blocked). Faster projectiles fly flatter and
  hit moving targets better. · [tw-modding][twm] · high (field).
- **Penetration** classes (`projectiles_penetrations`): low = up to 2 small entities, medium = 5
  small, high = 10 medium, very high = 20 large. · [tw-modding][twm], [WH3 kv guide][g3] · medium.
- **Homing projectiles** (5.0.0) each pick a random entity of the target unit. · [5.0.0 blog][p500] · high.
- **Single-entity shooters do ~25 % worse on Ultra** against multi-model units than on Large. · CA forum · low.

## Shields, armour, resistances

- **Shield block first.** A shield blocks its percentage of small-arms missiles (arrows, bullets,
  bolts, javelins, magic bolts) arriving within 60° either side of the facing; then armour and
  resistances apply to what gets through. Shields don't stop artillery, explosions or "misc"
  projectiles. A shield also blocks in melee and on the run; "flank ×0.5, rear ×0.1" belongs only to the
  Directional Shield attribute (Steam Tank), not to ordinary shields. · [WH3 kv guide][g3], [Steam][shield],
  [tw-modding][twm], recordings (`build/shields`) · high · `shield_defence_angle_missile` 60 (half-angle).
  - Ours: agreement (front 60°, then armour and resistance; small arms only: categories arrow, musket, sling,
    javelin, axe). The probe: clanrats with a shield facing the archers let through 0.66 (database 0.65),
    flank and back on the shield does nothing.
- **Block chances** come from the shield record (`missile_block_chance`, e.g. `wh_missile_block_35_metal`).
  5.0.0 raised the Empire General's shield from 35 % to 55 %. · [5.0.0 blog][p500] · high.
  - Ours: our passports read General 55 %, Warlord 35 %, Empire spearmen with shields 35 %. Agreement.
- **Old Attila keys** `missile_saving_bonus_base` 20, `_coefficient` 0.5,
  `missile_armour_piercing_coefficient` 0.5, `_penetrating_coefficient` 0.25,
  `missile_shield_piercing_coefficient` 0.5 are described (WH2) as parts of "missile damage
  resolution" with shield and armour bonuses; a WH3 player's formula from them contradicts the
  shield % the game shows. Treat as legacy. · [twwstats][tws], [Attila guide][attila] · low for WH3.
- **Armour against missiles** works as in melee (base part reduced by a random 50–100 % of
  armour, AP ignores it). Missile resistance and physical resistance (non-magical) apply; all
  resistances add up to a 90 % cap; a hit does at least 1 damage. · [CA damage blog][dmg] · high.
  - Ours: same armour rule as melee; missile and physical resistance together, at most 90 %
    (`missile.physical_resist`). Agreement.
- **Large single-entity monsters** usually have +15 % missile resistance. · Steam · low.

## Friendly fire and line of fire

- **Friendly fire exists for all missiles**, worst for flat-trajectory guns and artillery; high-arc
  bows fire over friends more safely. · [Steam][ff] · medium.
  - Ours: measured share of hits landing on our own units in contact: arrows 0.26, sling 0.56
    (`missile.friendly_fire`). The sling (flatter) causing more agrees with the community.
- **Friends are enlarged in the clear-shot test**: friendly entities count 2.2× wider and 1.15×
  taller (allies 1.7× / 1.0×) so shooters cannot fire just past a screen; a unit's own models are
  ignored (`missile_ignore_own_unit_for_friendly_fire` 1). · [twwstats][tws] (WH2 descriptions),
  [CA bug report][ownff] · high.
  - **All our battles run with the True Sight mod** ([run](../../launch/run.md#required-mod-true-sight)): it sets
    all four multipliers to 1.0 (friends at their own size) and the blocked-line threshold (below) to 1.0; also
    `melee_attack_threshold_modifier_idle_default` / `_idle_ammo_remaining` / `_ordered` 0.14 / 0.85 / 0.22 -> 1.0.
    It changes nothing else (9 rows of `_kv_rules_tables`). The simulator takes the mod's values
    (`config/nn/game_rules.json` `_mods`).
- **Flat shooters hold fire when the line is blocked**; when 75 % of the unit can't see to fire,
  its line of sight counts as blocked (`unit_firing_line_of_sight_considered_obstructed_ratio` 0.75; with
  True Sight 1.0: only when the whole unit cannot see).
  1.3.0 let overlapping friends count as one unit for line of fire; abused by stacking, reverted in
  1.3.1. 8.0: shooters with a blocked line no longer walk up into melee. · [twwstats][tws],
  [PCGamesN][pcg131], [8.0 notes][p80] · high.
- **Fire at will may shoot into melee** (`allow_fire_at_will_into_melee` 1). · DB · high.
  - Ours: agreement; 57 % of arena shots went at enemies in melee when there were some.

## Reload, ammunition, movement

- **Reload** = the projectile's base reload × (1 − reload skill / 100); the animation is a hard
  floor. Each rank cuts ~0.2 s (rank 9 −1.8 s); fatigue cuts it (Exhausted ×0.9 by the
  database, see [fatigue](fatigue.md)). · [tw-modding][twm], fandom · medium.
  - Ours: a man's cycle from the probe: archers 10.0 s (= the database), slave slingers 11.5 (database 9),
    Night Runners 10.5 (database 8), militia 11 (database 9); rank 9 cuts the whole cycle by 2 % a rank, as the
    database says, so it is not an "animation floor". Why the slings and the pistol are ~1.25-1.3x slower than
    the database: no source. Whole-unit volleys; as the target thins the men skip volleys (12-15 s a man over a
    fight): no rule found, not in the simulator.
- **First volley delay** grows with poor training (`fire_volley_max_aim_delay_training_ratio`:
  WH2 0.11, WH3 0). · [twwstats][tws] · medium. Ours: first shot 3.3 s (arrows) / 4.3 s (sling)
  after halting.
- **Out of ammunition: WH3 shooters go into melee by themselves** (earlier games: stood still);
  guard mode doesn't stop it. 7.0: shooters keep their distance and use leftover ammo better. ·
  [Steam][noammo], [7.0 blog][p70] · medium/high. WH3 keys: `melee_attack_threshold_modifier_idle_default`
  0.14, `_idle_ammo_remaining` 0.85, `_ordered` 0.22 (no description; they look like how willing a
  unit is to start melee — far less while it still has ammo; all 1.0 with True Sight). · low.
  - Ours: a missile unit with no projectiles stops shooting; the simulator has no automatic switch
    to melee (only by order or contact). **Gap** — out-of-ammo shooters are on the user's task queue.
- **AI ammo threshold** `battleai_ammo_threshold` 0.75: when the AI counts as low on ammo, used only
  for ammo-replenishing abilities, not to stop shooting. · [twwstats][tws] (WH2 description) · high.
- **Fire while moving** is an attribute (horse archers, chariots, some infantry, flyers); others
  shoot only standing. · fandom, [5.3.0 notes][p530] · high.
  - Ours: shooting from standing only. Agreement for our units.
- **Range is from the shooter's centre to the target's centre**, and then the whole unit fires, the rear ranks
  beyond range too. · our measurements: [range](../units/missile-range.md), [probe](../units/missile-probe.md)
  (the first arrow at 129-131 m between centres for targets 6-36 m deep; with the edges in range and the
  centres beyond it the archers, slingers and Night Runners never fired) · high; the militia's pistols fired at
  97.5 m with range 90 too: open.
  - Ours: the same (`missile.range_centre`); a shooter under an attack order walks until the target's centre is
    in range.
- **The fire arc** is each man's: `battle_entities` fire arc, ±30° (militia ±35°); a man fires if part of the
  target is in his sector. Firing at will a unit does not turn to a target outside it. · database;
  [probe](../units/missile-probe.md): a narrow target at 35° - 0.42 of the men in a volley, a wide one at 35° -
  0.93, a narrow one at 50° - 0.06, no turn · high.
  - Ours: the same (`missile.per_man_arc`, `arc_share`); to the target of an attack order beyond 45° the unit
    turns.
- **Skirmish mode** makes shooters back away from approaching enemies (an option can switch it on
  by default). · Steam, 9.0 notes · medium. The trigger distance is neither in the database nor from the
  community.
  - Ours: not a simulator rule (it is off for the network); `ai_like` copies the game AI's backing off from
    measurements ([training](../../training/training.md)).

## Targeting

- **AI targeting by difficulty (2.3.0):** Easy and Normal shoot the first enemy they meet; Hard
  prefers low-armour targets; Very Hard focus-fires one unit and avoids evasive targets. ·
  [player.one on 2.3.0][p230], [fandom 2.3.0][fw-23] · high.
  - Ours: at Normal the arena's archers shot the nearest enemy 55 % of the time, the second nearest
    19 %. The simulator keeps a target while it is in range and stands (`missile.sticky_target`), else takes the
    nearest; a new target costs 3 s without fire (`missile.retarget_s`, recordings).
- **Morale.** Being shot at: −5 (`ume_concerned_attacked_by_projectile`); the wiki says −8 for
  archers too (outdated; −8 is artillery). · DB · high. See [morale](morale.md).

[twm]: https://tw-modding.com/wiki/Tutorial:Missiles_and_You
[acc]: https://steamcommunity.com/app/594570/discussions/0/3040481180139589116/
[phys]: https://steamcommunity.com/app/364360/discussions/0/1732090362040597429/
[tws]: https://twwstats.com/kv/rules
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[p500]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[p70]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/90-total-war-warhammer-iii-update-7-0-0
[p80]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98-total-war-warhammer-iii-update-8-0-patch-notes
[shield]: https://steamcommunity.com/app/594570/discussions/0/2295094308091532998/
[attila]: https://steamcommunity.com/sharedfiles/filedetails?id=570444181
[ff]: https://steamcommunity.com/app/364360/discussions/0/357287304415816662/
[ownff]: https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/bugs-redirect/1470-variable-missile-ignore-own-unit-for-friendly-fire-is-broken
[pcg131]: https://www.pcgamesn.com/total-war-warhammer-3/ranged-units-hotfix
[noammo]: https://steamcommunity.com/app/1142710/discussions/0/3185740024680941836/
[p230]: https://www.player.one/total-war-warhammer-iii-update-230-brings-major-improvements-ai-152358
[fw-23]: https://totalwarwarhammer.fandom.com/wiki/Update_2.3.0
