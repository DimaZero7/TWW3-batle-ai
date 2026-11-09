# Web notes: missile mechanics (community)

## Q2 calibration
- tw-modding wiki Tutorial:Missiles_and_You https://tw-modding.com/wiki/Tutorial:Missiles_and_You
  - accuracy = projectile marksmanship_bonus + land_units accuracy
  - calibration_distance: "distance beyond which the missile becomes much more inaccurate"
  - calibration_area: "size of the circle that the projectile is calibrated to and lands in" (no radius/diameter, no %)
- Steam WH2 "All the mechanics CA didn't explain" https://steamcommunity.com/app/594570/discussions/0/2999925875999283361/
  - calibration area: "the area a projectile may hit from its intended target"
  - calibration distance: "minimum/maximum range for hitting with maximum accuracy"
- Attila obscure mechanics guide (steam 570444181; mirror steamsolo) https://steamsolo.com/guide/the-obscure-mechanics-of-attila-total-war-attila/
  - aim at random point within 2 m radius of unit centre (kv_rules Projectile_Target_Calibration_area)
  - accuracy skill -20% vs targets in cover; missile_saving_bonus = base chance a hitting missile misses
- TWCenter "Definitive projectile accuracy formula?" (search snippet only, 403) https://www.twcenter.net/threads/definitive-projectile-accuracy-formula.733276/
  - accuracy 10 + marksmanship 10 -> "20% chance per entity shot to exactly hit"; miss lands within calibration area
- kv_rules fields: projectile_calibration_target_area, missile_saving_bonus_base, missile_range_lethality_modifier...

## Q6 fire at will
- Patch 6.1 CA: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/49-total-war-warhammer-iii-patch-notes-6-1
  "Fixed issues with ranged unit's 'Fire at Will' stance by increasing the number of checks units make to start firing. They will now also prioritise firing over maintaining positional unit cohesion."
  Also: "Fixed some issues where chariots and some other entities were being hit by the same projectile multiple times"

## Q1 targeting
- CA bug forum 2841 https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/2841-the-targeting-logic-of-ranged-artillery-is-flawed
  player: "all of the shooters in a unit firing at one single person"; CA_Dimitar: escalated, marked Known Issue

## Q2 more
- Steam WH3 "Any way to see a units accuracy?" https://steamcommunity.com/app/1142710/discussions/0/4357872216491592056/
  Commisar Jon Fuklaw (modder): calibration area "valuated in square meters"; "calibration range" = range at which area reaches listed (max-accurate) value;
  "Attacks made beyond that range become linearly less accurate"; "All the accuracy stat does is reduce the targeting area by a percentage (the accuracy value) of the calibration area."
- Steam WH2 Darkshard changes https://steamcommunity.com/app/594570/discussions/0/2969524751701924312/
  Zane87: "See it as inaccuracy radius in units"; Darkshards cal dist 85->90, area 3.5->3.4
- Steam WH1 "Why range has no accuracy?" https://steamcommunity.com/app/364360/discussions/0/1732090362040597429/
  Gray Riders: marksman bonus 70 + bad calibration -> mediocre accuracy; Wyvern: rank 9 xbow ~+30% vs necromancer, ~+10% vs zombies
- Steam WH2 Gunpowder Mechanics https://steamcommunity.com/app/594570/discussions/0/1750149365327909751/?ctp=2
  Blue: "Each model shoots, each model reloads... Hits are also per model."; bows/muskets cal area ~3.5, mortars 400

## Q5 LoS / friendly fire fields
- "Line of Sight - Melee Reactions - Ranged Targeting adjustments" https://steamcommunity.com/sharedfiles/filedetails/?id=3548112471
  "For a unit to report "Line of sight obstructed" it needs 75% of its entities to have no vision of the target. Increased this to 90%"
  "Entity perceived size radius was 220%. Now 160%." "Entity perceived height was 115%. Now 105%."
  "Ancient Salamanders and the blessed version were not set to track moving targets" (projectiles table flag)
  "Removed center mass targetting from cannons and hellblaster volley gun/organ guns"
- True Sight https://steamcommunity.com/sharedfiles/filedetails/?id=3628832922 : safety coefficients -> 1.0; LoS blocked only when 100% models obstructed; only models in contact switch to melee
- True Sight Lite https://steamcommunity.com/sharedfiles/filedetails/?id=3631418572 : same w/o melee change
- Friendly Fire mod (smods mirror) https://catalogue.smods.ru/archives/183131 : projectile_friendly_fire_man_height_coefficient vanilla 15% clearance -> 0.8
- Even Stronger Guns V2 https://steamcommunity.com/sharedfiles/filedetails/?id=2573488401 : height coeff 1.15 -> 1
- CA forum thread 5842 https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/5842-how-ranged-gun-unit-s-line-of-fire-gets-hard-blocked-and-refuse-to-fire
  player: gun unit won't fire if "centre of the gun unit doesn't have line of sight with the banner of an enemy unit"
- WH3 steam 3825284575368948675: Aleera "Draw a direct line from your unit's shoulders to the enemy shoulders..."

## Q1 more
- CA forum Homing 2.0 thread 1540: after 4.1.0 "all the projectiles target the same model in a unit"
- Cannon Targeting Fix https://steamcommunity.com/sharedfiles/filedetails/?id=3040180685 : vanilla cannons prefer central entities; mod -> random entities throughout unit
- CA bug 1951 artillery prefer central: CA_Liam "It isn't the design that all artillery aim at the centre"
- WH3 cav archer thread https://steamcommunity.com/app/1142710/discussions/0/591776982252081511/ : "every entity in the cav archer unit will target the closest enemy entity"

## More (round 3)
- CA bug 17 Volley of Kurnous https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/bugs-redirect/17-volley-of-kurnous-targetting-issue
  AurumHawke: WH2 arrows "interleaved across the targeted unit"; WH3 "all Hawk Riders select the same target"; "Homing just makes the arrows hit the target (and is mostly irrelevant when fired from above the target)"; CA_Luna: fixed
- Steam WH3 homing 2.0 thread https://steamcommunity.com/app/1142710/discussions/0/4298195085742175381/?ctp=2 : "360-fire + fire-while-moving units target the first entity in range"
- CA bug 1470 missile_ignore_own_unit_for_friendly_fire https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/bugs-redirect/1470-variable-missile-ignore-own-unit-for-friendly-fire-is-broken
  WH2: "only front rank entities fired and rear ones were able to fire if they had high firing arcs"
- WH2 steam 3076503022320167898 Donut Steel: "If only the left half of a unit of handgunners can see past your troops ... only the left half will fire"; "all ranks fire simultaneously, with the shots clipping-through"
- Mod changelog 3535265856 (snippet only): vanilla calibration distance ~70% of max range; peasant archers 50 m in mod, "accuracy diminishes linearly after 50 meters"
- TWC 395330 Kaunitz (ETW hypothesis, cal area 20 at 60 m): 25 m 12.5%, 46 m 6.5%, 70 m 4.3%, 90 m 3.5% (~1/d)
- ETW kv_rules projectile_calibration_target_area = "area of calibration target in square metres" (snippet, TWC)
- Steam WH3 fire at will bug 612031852355559283: unit card "still moving" so FaW doesn't trigger; fix halt/backspace
- CA bug 7133 / 1377: >40 units, direct orders 15-30 s delay (large battles bug)

## Round 4
- Hold Position mod https://steamcommunity.com/sharedfiles/filedetails/?id=2869716515 : vanilla unit "shuffling to face" target outside fire arc; without LoS to ordered target unit walks forward until it finds LoS
- WH1 steam 351659808476547673 BigBadB: "Missile troops won't fire if their target is completely obscured by a friendly unit"
- Search snippet (source unclear, WH): indirect-fire unit obstructed by friendly unit won't fire automatically but will if ordered
- WH2 1750149154149341532 Darklordnj: sweet spot "half to 3/4 of your max range" (opinion)
- WH2 1744482417425061535 Wyvern: reload skill ~1.1% per point (fixed seconds off base)
- NOT FOUND: per-man re-aim after target death; aim/draw time; explicit lead/prediction rule for normal arrows/bullets; True Sight exact field names (Steam 429); RPFM schema descriptions (file >10 MB)
- Web search budget exhausted (200/200) at ~45 min.

## Q7 reload / animation
- WH2 "All the mechanics": reload hardcap = firing animation length (Casket of Souls 7.6 s anim vs 6 s reload)
- reload skill ~1.1% per point (WH2 steam threads)
