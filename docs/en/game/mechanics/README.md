# Battle mechanics

[← Back](../README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › Battle mechanics · [Русский](../../../ru/game/mechanics/README.md)

A reference of how Total War: WARHAMMER III land battles work, collected from public sources on
02.10.2026: CA's blogs and patch notes, the game's key-value tables (via twwstats, which mirrors
patch 9.0.0 and, for WH2, CA's own description of each key), the fandom wiki, tw-modding,
Steam guides and threads. Where WH3 information was missing and the engine is the same, WH2
sources are used and marked. Our own measurements in the game are elsewhere:
[units](../units/README.md) (melee, morale, missiles, pace) and the
[simulator](../../training/simulator.md).

## How to read a fact

- Each fact is in our own words, with its number or formula, the game version or patch when
  known, the source (linked) and a **confidence**:
  - **high** — CA (blogs, patch notes, script docs), the game's database, or reproduced tests;
  - **medium** — consistent community knowledge (several sources, or a detailed guide);
  - **low** — a single claim, or our own inference from a key's name.
- A database key is given as `key` value; the table is `_kv_rules` / `_kv_morale` / `_kv_fatigue`
  unless named. Our copies of these values are in `config/nn/game_rules.json`; anything with a key
  can be checked in `db.pack`.
- **"Ours:"** compares a fact with our simulator (`config/nn/sim.json`, `tools/nn/sim/`) or our
  measurements: agreement, conflict, or gap (a game mechanic we do not model).
- "WH2 description" means the description column of the WH2 copy of a key-value table; CA left it
  empty in WH3, the values are the same unless stated.

## Pages

<!-- generated:docs:index -->
- [Abilities, attributes, army mechanics](abilities.md) — Common unit attributes and passives with their numbers, contact effects, barriers, army-wide mechanics (Waaagh!, battle currency) and reinforcements
- [Artillery and war machines](artillery.md) — Artillery accuracy and arcs, line of sight, damage and morale, crews, armoured vehicles
- [The game's battle AI](battle-ai.md) — What is public about how the built-in battle AI decides, what difficulty changes, the script planner, and the AI weaknesses players exploit
- [Battle rules: size, time, victory](battle-rules.md) — Unit size, army caps, the time limit, how a battle is won, the balance of power and difficulty
- [Cavalry and chariots](cavalry.md) — Charges, cycle charging, mass, bracing against cavalry, anti-large, chariots and collision attacks
- [Fatigue (vigour)](fatigue.md) — Fatigue levels, what each costs in stats, what tires and what rests
- [Flanking and rear attacks](flanking.md) — What a flank or rear attack changes: the defender's melee defence, shields, morale, and how the game decides the direction
- [Flying units](flying.md) — Flight, landing, fighting from the air, flyers against missiles and the AI
- [Magic and the Winds of Magic](magic.md) — The battle side of magic: the Winds pool, casting, overcast and miscast, spell types, resistances
- [Melee](melee.md) — How the game resolves a melee blow: hit chance, damage, armour, charge, bracing, mass and splash
- [Missiles](missiles.md) — How shooting works: accuracy, range, shields, armour, friendly fire, reload, ammunition
- [Monsters and large units](monsters.md) — Single-entity and small-entity monsters: mass, knockdowns, splash, fear and terror
- [Morale (leadership)](morale.md) — Every morale modifier the game's tables and the community name, with values, plus the state machine (waver, rout, rally, shatter)
- [Movement and formations](movement.md) — Speeds, charges, formations, collisions, breaking off and pursuit
- [Patch history of battle mechanics](patches.md) — What WH3 patches changed in the battle mechanics of the other pages, oldest first
- [Sieges and settlement battles](sieges.md) — Lower priority for us (we train on field battles)
- [Lords and heroes (single entities)](single-entities.md) — A lord or hero on foot is one entity with a lot of health
- [Terrain, weather, visibility](terrain.md) — Height, slopes, forests, water, line of sight, hiding, weather and time of day
<!-- /generated -->

## Ten facts that matter most for our simulator

1. **The hit-chance formula is confirmed by CA** (2023): 35 + attack + bonus vs type + current
   charge bonus − defence, 8–90 %; no patch changed it. Our flat `hit_slope` 0.1 is therefore a
   stand-in for something else (blow cycle, matched combat, reach) — [melee](melee.md).
2. **Flank/rear: defence ×0.6 / ×0.3, decided per attacking model**, ending when the struck entity
   turns to face the attacker — which explains why a lord shows no flank penalty — [flanking](flanking.md).
3. **"Attacked in the flank / rear" (−6 / −14) is a first-contact effect** by the key's description,
   not a continuous one — likely why we measured only −1 / −2 per second of contact — [morale](morale.md).
4. **Recent casualties are the last 4 s; extended casualties (−4…−60) the last 60 s** — [morale](morale.md).
5. **Fatigue penalties are in the database** (`unit_fatigue_effects_tables`, read from our `db.pack`): melee attack ×0.95…0.7, speed
   ×0.95…0.85, armour, charge, AP damage, reload — [fatigue](fatigue.md).
6. **Charge:** full bonus on the first blow, linear to 0 over 13 s, only if the entity reached its
   hidden charge speed; blocked back ranks don't charge; a charge gives +15 morale — [melee](melee.md#charge).
7. **Army collapse:** when enemy strength ≥ 2.6 × own and own ≤ 0.22 of the start, every unit gets
   −120 and routs — the game ends battles this way — [battle rules](battle-rules.md#victory).
8. **Skaven passives explain two of our Skaven measurements:** Strength in Numbers (+6 leadership,
   +8 defence, −10 % speed above 50 % HP) and Scurry Away! (+10 % speed when wavering) — [abilities](abilities.md).
9. **Splash damage is divided among the targets**, targets = weapon strength / 100 for single
   entities since 6.0 (General: 430 → 4, as in our passport) — [melee](melee.md#splash).
10. **At Normal the AI shoots the first enemy it meets and dodges only spells**; its melee reaction
    radius looks like 40 m and free attack 100 m — matching the 40–115 m charges we saw — [battle AI](battle-ai.md).

## Conflicts with our simulator

The most valuable part: where the game (by CA, the database or the community) and our simulator
or our documents disagree. "Measured" means our number comes from recordings and may well be the
better one; each is a candidate for a test.

| # | Mechanic | The game (source) | Ours | Page |
|---|---|---|---|---|
| 1 | Hit chance vs attack − defence | full slope 1 (CA, DB) | slope 0.1, fitted on 4 pairs | [melee](melee.md#hit-chance) |
| 2 | Flank vs rear | rear costs more: defence ×0.3 vs ×0.6 (CA, DB) | flank costs more: fitted `flank_slope` 2.0, `rear_slope` 0.25 (measured 1.74× vs 1.31×) | [flanking](flanking.md#melee-defence) |
| 3 | Direction sectors | per attacking model, by quadrant of the struck entity (CA; 45°/135° by our reading) | per unit, by the enemy centre, front < 60°, rear > 120° | [flanking](flanking.md#melee-defence) |
| 4 | "Attacked in flank / rear" morale | −6 / −14 at first contact (DB description) | −1 / −2 continuously (regressed) | [flanking](flanking.md#morale) |
| 5 | Recent casualties window | last 4 s (DB description) | 30 s (calibrated) | [morale](morale.md#modifiers-points) |
| 6 | Extended casualties | −4 … −60 for 10–80 % lost in the last 60 s (DB) | not modelled | [morale](morale.md#modifiers-points) |
| 7 | Fatigue rate | tick 0.1 s → ×10 a second (one forum claim, medium-low) | ×10 a second (calibration ON: melee tires only under an attack order — single entity +19, formation 13.7; shooting 7.5, walking 3.4, idle −18; fitted on 204 recordings) | [fatigue](fatigue.md#what-tires-and-what-rests) |
| 8 | Fatigue effects | in the DB table `unit_fatigue_effects_tables` (melee attack to ×0.7, speed to ×0.85, …) | modelled from this table (`fatigue.effects()`) | [fatigue](fatigue.md) |
| 9 | Formation spacing | `entity_formation_spacing` 1.8 m (DB) | 1.5 m (measured) | [movement](movement.md#formations) |
| 10 | Charge reflection / bracing | braced reflectors deal ×2 weapon damage to chargers for 3.9 s; bracing is a mass multiplier (DB, CA) | a braced unit meets a frontal charge "as a charge" for 13 s (fitted to the measured 0.80) | [melee](melee.md#bracing-and-charge-defence) |
| 11 | Charge impact | charge bonus + capped collision damage (≤ 70, 70 % AP) (DB) | extra ×(1 + 1.5 × speed share) damage for 13 s (fitted) | [melee](melee.md#charge) |
| 12 | Splash | damage divided among targets (community, CA forum) | full hit on each target, capped at a man's HP | [melee](melee.md#splash) |
| 13 | Lord's aura | +4 to 70 m, fading to 0 at 105 m (DB) | the same: +4 to 70 m, fading to 0 at 105 m | [morale](morale.md#modifiers-points) |
| 14 | Rally timing | `broken_finish_base_timeout` 180 s + 10 s × rank (DB; meaning unclear) | rally after 44 s median (measured) | [morale](morale.md#waver-rout-rally-shatter) |
| 15 | Strong enemy near | −3 … −24 by combat power within 70 m (DB) | −3 only | [morale](morale.md#modifiers-points) |

**Gaps** — game mechanics our simulator does not have, by likely impact on our battles:
army-losses collapse (−120, ends battles), charge morale (+15), pursuit
blows counting as charges, shooters going into melee when out of ammunition, the entity
`turn_speed` out of melee, the morale shock (25 % lost in 4 s), shattering by casualties.

**Agreements worth knowing** (the simulator's measured numbers match public facts): map edge
1020 m vs playable area 1024; `lord_max_attackers` 9 vs "~9 small entities around one"; no flank
penalty on a lord vs the per-entity rule; the General's shield 55 % (5.0.0); card speed = m/s × 10;
higher hit rate at 70 m vs calibration distance; "flanks exposed" −3 / −6 (measured −3.9 / −8.1);
Skaven +6 points at the start = Strength in Numbers; Skaven routing faster = Scurry Away!; the
General's splash 4; the armour roll; the charge and bonus damage split by the weapon's AP share.

## Sources that were most useful

- CA, [Feature Focus #2: Damage][dmg] and [#1: Elevation][elev] (2023) — formulas.
- twwstats key-value tables: [rules][tws-r], [morale][tws-m], [fatigue][tws-f] (WH3 9.0.0 and WH2 with descriptions).
- CA patch notes 5.0–9.0 (linked on each page), fandom `Update_*` pages.
- Steam guide [KV_Rules and so will… three?][g3] (WH3, 2022) and the WH1 [_KV_rules guide][g1].
- [tw-modding: Missiles and You][twm]; CA script docs for [script_ai_planner][chv].

Reddit, YouTube and twcenter could not be read (blocked); the fandom wiki only through a browser.

[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[tws-r]: https://twwstats.com/kv/rules
[tws-m]: https://twwstats.com/kv/morale
[tws-f]: https://twwstats.com/kv/fatigue
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[g1]: https://steamcommunity.com/sharedfiles/filedetails/?id=719168880
[twm]: https://tw-modding.com/wiki/Tutorial:Missiles_and_You
[chv]: https://chadvandy.github.io/tw_modding_resources/WH3/battle/script_ai_planner.html
