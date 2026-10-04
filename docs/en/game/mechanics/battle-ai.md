# The game's battle AI

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Battle AI · [Русский](../../../ru/game/mechanics/battle-ai.md)

What is public about how the built-in battle AI decides, what difficulty changes, the script
planner, and the AI weaknesses players exploit. Our own observations: [the game's own AI](../game-ai.md),
[difficulty](../difficulty.md). Conventions: [index](README.md).

## Difficulty: what changes in battle

- **Battle difficulty is mostly reaction time and targeting.** Since 4.0 (31.08.2023) the AI's
  "intelligence" (battle difficulty) and the AI stat bonus are separate settings. · [NME on 4.0][nme],
  [fandom 4.0][fw-40] · high/medium.
- **Evasion and targeting by difficulty** (2.3.0, table in 4.0):

  | Difficulty | Dodges | Missile targeting | Reaction |
  |---|---|---|---|
  | Easy | nothing | the first enemy it meets | slow |
  | **Normal** | **direct-damage spells only** | the first enemy it meets | average |
  | Hard | + light missiles in settlements | low-armour targets first | quick |
  | Very Hard | + artillery | focus fire, soft targets, avoids evasive ones | instant |

  A unit in melee resolves the melee first. · [player.one on 2.3.0][p230], [fandom 2.3.0][fw-23] · high.
  - Ours: all our battles are at Normal ([difficulty](../difficulty.md)): the AI shoots the first
    enemy it meets — matches the measured "nearest 55 %".
- **Stat bonuses.** WH2 database: AI melee attack, charge, damage ×0.8–0.9 on Easy, ×1.1 Hard,
  ×1.15 Very Hard (defence ×1.2); extra missile reload −10 / +10 / +20; AI morale −4 / 0 / +4 / +10.
  WH3 replaced these with a range (`difficulty_mod_ai_min/max_stats_range` 0.9 / 1.1,
  `_normal_multiplier` 0.5, `_hard_multiplier` 0.75, reload extra ×1 / ×2, morale extra ×0.4 / ×0.8);
  how they combine is not documented. Player morale: Easy +4, Normal 0, Hard −2, Very Hard −4. ·
  [twwstats][tws] · high (values), low (WH3 formula).
  - Ours: at Normal the starting `MoralePercent` of both sides is the same (lord 1.057, spearmen
    1.067), so no visible morale bonus. Agreement with "Normal = 0".
  - Ours: at equal health the AI's units (side 2) in melee hold 0.1–0.2 more `MoralePercent` than
    ours — in AI-against-AI battles too (CA's planner on side 1, 28 battles: at health 0.4–0.6 0.46
    against 0.63), but the simulator, with no bonus for either side, gives the same on the same orders
    (0.49 against 0.68): it is the fight (who wins the melee, routing enemies), not a difficulty
    extra. The localised battle difficulty texts name only reaction time, dodging and targeting; the
    separate `battle_ai_stats_modifier` (the AI's stat multiplier) in `preferences.script.txt` is 1.
    Agreement with "Normal = 0".
- Community figures (Very Hard ≈ +8 leadership for the AI, slider max ≈ +7, Hard/VH +5–10 % melee
  stats) conflict with each other. · Steam threads · low.

## How the AI picks fights

- **Attack-mode ranges (WH3 keys, no description):** free attack 100 m, melee reaction 40 m,
  ranged attack 15 m, local attack 3 m. They look like the radii at which units pick targets or
  react to melee. · [twwstats][tws] · high (values), low (meaning).
  - Ours: the defending game AI holds its place and its melee units charge when an enemy comes
    within ~40–115 m ([game-ai](../game-ai.md)) — the same span as 40 / 100 m. Agreement in kind.
- **Willingness to start melee (WH3 keys):** `melee_attack_threshold_modifier_idle_default` 0.14,
  `_ordered` 0.22, `_idle_ammo_remaining` 0.85, `_artillery` 1.0 — probably shooters with ammo are
  far less willing to melee. · twwstats · low (meaning).
- **Engage or pull back.** CA's battle AI uses a learned model of one-on-one combat outcomes
  (AIIDE paper). · see [game-ai](../game-ai.md#what-ca-published-about-their-ai) · high.
- **Hills.** The AI ignores hills more than 80 m away as defensive features
  (`discard_defensive_feature_distance`). · twwstats (WH2 description) · high.
- **Sally-outs** (sieges): a defender sallies when its strength is ≥ 1.5× the attacker's (each
  tower ~1000 strength) or when sitting under enemy missiles is too costly. · twwstats · high.
- **Mages** switch to melee when their mana is more than 300 s of recharge from full
  (`melee_mage_switch_to_melee_time`). · twwstats · high.
- **Ammo.** `battleai_ammo_threshold` 0.75 only drives ammo-replenishing abilities. · high.

## Known AI behaviours and fixes (patch notes)

- 7.0: shooters keep their distance and spend leftover ammo better. · [7.0][p70] · high.
  - Ours: measured in the 110 network gate battles (03.10.2026, `build/ail`): a free missile unit moves away
    from a closing enemy melee unit in 52–58 % of the seconds within 40 m, 42 % at 40–60 m, 23 % at 60–80 m,
    and one caught in melee moves (tries to leave) 52 % of its first 5 s; in melee 10 % of its standing
    time. Agreement; `ai_like` now does the same ([training](../../training/training.md#the-opponent-ai_like)).
- 7.2: no longer splits its army to chase groups of only flyers; lords no longer flip-flop behind
  their army; no "spaghetti" lines into minor settlements. · [7.2][p72] · high.
- 8.0: the whole army no longer clumps when every unit gets the same order; no idling while the
  player outflanks it; blocked shooters don't walk into melee; lords don't idle at gates. · [8.0][p80] · high.
- 8.1: no longer ignores routers when nothing else is near; no stacking on the map edge while
  re-forming; shooters not constantly re-ordered chasing outflanks. · [8.1][p81] · high.
- 5.2.1: higher priority for vulnerable artillery. · fandom · high.
- 4.1.0: fewer melee units idling under flying enemies. · fandom · high.

## What players exploit

All community, mostly low-medium confidence.

- **Half the army chases one unit** (a fast lord, a flyer): a modder calls it hard-coded since 4.0;
  kiting a lord or flyer back and forth drained AI ammo in WH2. · [Flogis mod][flogis], Steam · medium.
- **Flyers as decoys**: the AI tends to target the closest unit, so a flyer can pull artillery and
  missiles. · Steam guide · low.
  - Ours: at Normal its melee units take the nearest only 41–48 % of the time (3,397 new targets in the 110
    network gate battles, 03.10.2026): from ~90 m, preferring enemies already fighting elsewhere (+20 m),
    avoiding our lord (−26 m), not drawn to missile units; the rest looks random (a logit's spread 27 m).
    Conflict for melee targets (missile fire: the nearest, above).
- **Stacking units** to fire through friends (1.3.0, reverted in 1.3.1). · high.
- **Difficulty ≠ smarter tactics**: many players say higher difficulty is mostly stat bonuses and
  reaction time. · Steam · low.

## The script planner (what we use for our side)

- `script_ai_planner` wraps the built-in planner: unit groups take high-level orders and act
  semi-autonomously. Orders: `move_to_position`, `defend_position(v, radius)`, `defend_force`,
  `rush_position`, `attack_unit`, `rush_unit`, `attack_force`, `rush_force`, `patrol`, `merge_into`,
  `release`. Defaults: patrol radius 100 m, enemy detection 100 m, waypoint arrival 75 m, merge
  120 m, re-orders every 30 s (`set_should_reorder`). · [CA script docs][chv] · high.
- Script units also have `change_behaviour_active` ("defend", "fire_at_will", "skirmish",
  "change_formation_spacing" …) and `fearless_until_casualties`. Released units follow the AI's own
  algorithms and may "look odd". · [CA script docs][chv-u], [old CA wiki][wiki] · high.
- **AI General 3** (a mod) plays the player's army with CA's planner and reaches only the Normal
  AI's level. · [PCGamesN][aig] · medium.

[nme]: https://www.nme.com/news/gaming-news/total-war-warhammer-3-will-make-ai-factions-smarter-and-less-biased-3491045
[fw-40]: https://totalwarwarhammer.fandom.com/wiki/Update_4.0
[fw-23]: https://totalwarwarhammer.fandom.com/wiki/Update_2.3.0
[p230]: https://www.player.one/total-war-warhammer-iii-update-230-brings-major-improvements-ai-152358
[tws]: https://twwstats.com/kv/rules
[p70]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/90-total-war-warhammer-iii-update-7-0-0
[p72]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/95-total-war-warhammer-iii-patch-7-2-release-notes
[p80]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98-total-war-warhammer-iii-update-8-0-patch-notes
[p81]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/101
[flogis]: https://catalogue.smods.ru/archives/238069
[chv]: https://chadvandy.github.io/tw_modding_resources/WH3/battle/script_ai_planner.html
[chv-u]: https://chadvandy.github.io/tw_modding_resources/WH3/battle/script_unit.html
[wiki]: https://wiki.totalwar.com/w/Battle_Script_Documentation.html
[aig]: https://www.pcgamesn.com/total-warhammer-3/mod-ai-general-3
