# TWW melee mechanics: what the web says (web research, 06.10.2026)

Confidence labels: **confirmed** = CA patch notes / CA blog / raw DB value; **community** = player guide or
forum claim, sometimes with a test; **speculation** = my reading or an unsupported claim.
The raw WH3 `_kv_rules` values below come from the public dump https://raw.githubusercontent.com/Shazbot/WH3-Dump/main/db/_kv_rules_tables/data__.tsv
(318 rows, key + value, **no description column**). The local file build/melee2/twwstats_kv_all.json also has `description: null`.
The fandom wiki returned 402 errors, honga.net, twcenter and web.archive returned 403 errors, so the ability numbers
come from search-engine snippets of those pages.

---

## Q1. Hit reactions and attack interruption

| Claim | Number | Source | Confidence |
|---|---|---|---|
| kv values | `melee_max_hit_reactions` 2; `interrupt_mass_ratio_clamp_hit_reaction` 0.4, `_knockback` 0.35, `_knockdown` 0.5, `_melee_block` 0.5, `_stepback` 0.5; `melee_on_damage_effect_knockback_chance` 0.10, `_knockdown_chance` 0.05, `_knock_chance_charging_bonus` 2 | WH3 dump (above) | confirmed values; meaning not documented |
| Knock formula: knockback chance = 10 % x (attacker mass / defender mass), knockdown = 5 % x ratio, x2 when charging; **no knockdown if ratio < 50 %, no knockback if ratio < 35 %** | 10 %, 5 %, x2, 0.5, 0.35 | Steam guide "Total war: Warhammer - Some notes" https://steamcommunity.com/sharedfiles/filedetails?id=1114878285 | community (WH1 era), but it **matches the kv values exactly**, so high confidence |
| By analogy: no hit reaction if attacker/defender mass ratio < 0.4 | 0.4 | kv `interrupt_mass_ratio_clamp_hit_reaction` | speculation. If true, a man of mass ~100 (guide: goblins 70, orcs/dwarfs 130, black orcs 190) **cannot cause any hit reaction or knock on a lord whose mass is > 250**. Check the lord's mass in gamedb. Then `hit_reactions_ignore_chance` matters only against heavy attackers (lords, monsters, cavalry) |
| Two ways to interrupt an attack: being hit, or a collision; mass is the core stat; heavier entities interrupt more and are interrupted less | none | Steam guide "Combat Mechanics: _KV_rules and will too!" (updated 2024) https://steamcommunity.com/sharedfiles/filedetails/?id=719168880 | community |
| An interruption "delays the ability of an individual to take his chance at hitting" | none | guide 1114878285 | community |
| Three Kingdoms kv has `attack_interval_multiplier_when_entity_attack_interrupted` 3, `melee_max_hit_reactions_for_heroes` 0; **WH3 kv has neither key** | 3; 0 | 3K dump https://pastebin.com/SR5agYkg ; WH3 dump | confirmed (absence in WH3) |
| WH3 Hotfix 6.2.2: minimum knockback ignore chance for foot characters raised to 40 % (wizards had 20–30 %). The reason given is the characters were "proving a bit more resilient than we'd like when faced with overwhelmingly massive opponents", and the change is "allowing them to both receive (and dish out) attacks a bit more reliably". **So CA says knockdowns both stop the character's own attacks and make it unhittable.** | 40 % | https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-patch-notes-amp-announcements/threads/10367-total-war-warhammer-iii-hotfix-6-2-2 | confirmed |
| 6.2.2 examples: Wight King knock interrupt ignore 40 -> 75 %, hit reaction ignore 65 -> 75 % ("tough to put damage back on with large entities while on foot"); Arachnarok knockback ignore 15 -> 85 %, hit reaction 15 -> 60 %; Cockatrice 50 -> 75 %; Queek knock 40 -> 75 %; Tyrion hit reaction 75 -> 80 %; Orion 70 -> 80 % | as listed | same | confirmed |
| Later: 7.1 Slaanesh hit ignore 50 -> 75 %; 8.1 Fimir Noble 70 %, Tamurkhan 80 %, Aekold 75 % | as listed | https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/93-total-war-warhammer-iii-7-1-patch-notes ; https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/101 (search snippets) | confirmed (snippets) |
| WH2 Empire Undivided: Dwarf lords and heroes +5 hit reaction ignore, +15 knock interrupt ignore, +200 mass | +5 / +15 / +200 | https://www.gamerevolution.com/guides/592843-total-war-warhammer-2-the-empire-undivided-update-patch-notes | confirmed |
| Knock resistance of foot characters ranges from 20 % (WH1/2 casters) to 80 % (Tyrion, Sigvald) | 20–80 % | https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-total-war-warhammer/threads/966-character-knockdown-rates-should-be-reevaluated | community (datamined) |
| A knocked-down entity is "not a valid target"; WH3 4.1: attackers of a knocked-down single entity re-form over it "then wait for it to get up" | none | Steam https://steamcommunity.com/app/1142710/discussions/0/4416424085342483518/ ; 4.1 notes (snippet) https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/1306-total-war-warhammer-iii-patch-4-1 | confirmed (4.1) + community |
| Knockdown get-up delay: "changed from 0.7-2 seconds to 1-2.7 seconds" | 1–2.7 s | WH2 1.12.1 https://updatecrazy.com/total-war-warhammer-2-update-1-12-1-patch-notes-sep-8-2021/ | confirmed (WH2; probably still valid) |
| **Duration of a hit reaction (flinch)** | — | **nothing found** | — |
| **"Stunlock" test with numbers (lord surrounded by infantry)** | — | **nothing found** | — |

## Q2. Bracing

| Claim | Number | Source | Confidence |
|---|---|---|---|
| "Bracing: when infantry is standing still and facing the enemy, they will get a bracing multiplier to their mass. A braced entity can have significantly improved mass, reducing the collision power and damage considerably." | — | WH2 1.12.1 https://updatecrazy.com/total-war-warhammer-2-update-1-12-1-patch-notes-sep-8-2021/ | confirmed |
| "Braced units now get their max bracing bonus at 8 ranks deep (was 6 previously)"; "Max bracing multiplier for mass from ranks was changed from 3.5 to 4"; "Charge Defenders have had their bracing multiplier increased from 1.75 to 2"; charge bonus no longer added to collision power | 8 ranks, x4, x2 | same | confirmed; these map 1:1 to kv `bracing_calibration_ranks` 8, `bracing_calibration_ranks_multiplier` 4, `bracing_charge_reflector_bonus` 2 |
| Brace mass cap x4 (x8 with charge defence) | 4 / 8 | CA bug forum https://community.creative-assembly.com/total-war/total-war-warhammer/bugs/bugs-redirect/12942-unbraced-isolated-infantry-benefits-over-full-brace-mass-bonus | community |
| `bracing_max_multiplier_clamp` 10 = overall cap on the product of all bracing factors | 10 | kv; Attila guide describes the same three knobs | speculation (consistent) |
| `bracing_attack_angle` 80 | 80 | kv | meaning not documented; probably the attacker must come within 80° of the braced entity's facing (speculation). Attila: bracing not applied against flank charges |
| Bracing = 1 until the unit has stood still about 2 s | ~2 s | Attila guide (mirror) https://steamsolo.com/guide/the-obscure-mechanics-of-attila-total-war-attila/ | community (Attila) |
| "A braced unit also receives 600 additional mass" | 600 | https://steamcommunity.com/app/594570/discussions/0/2999925875999283361/ | community; matches kv `bracing_immovable_by_small_entity_mass` 600, whose name suggests "small entities cannot move a braced entity" rather than +600 mass (speculation) |
| `unit_braced_status_threshold` 0.75 | 0.75 | kv | speculation: share of entities that must be braced for the unit to count as braced |
| Back ranks: "Entities will no longer charge attack if their charge path is blocked by a friendly entity" -> "around 35% less attacks" | -35 % | WH2 1.12.1 | confirmed (CA test) |
| 8.0 AI fix: infantry braced even when cavalry was not moving straight at them | — | https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98 | confirmed |
| Collision kv: damage max 70, speed-change thresholds 3 (step back) / 4 (knocked back) / 7 (knocked down) / 10.5 m/s (flying); damage immunity timer 4 s | as listed | kv | confirmed values |
| Infantry vs infantry: bracing only changes mass in collisions (impact damage, knock chance); players say "versus infantry charge in" | — | https://steamcommunity.com/app/594570/discussions/0/3971555000777394206/ | community, **no test found** |

## Q3. Fighting without an attack order (`melee_attack_threshold_*`)

| Claim | Number | Source | Confidence |
|---|---|---|---|
| WH3 kv: `_idle_default` 0.14, `_ordered` 0.22, `_idle_ammo_remaining` 0.85, `_artillery` 1.0. 3K: 0.1 / 0.15 / 0.85 / 1, plus `_guard_mode_active` 1 (that key is not in WH3) | as listed | WH3 dump; https://pastebin.com/SR5agYkg | confirmed |
| Modder reading ("Improved Unit Behavior"): idle_default 0.14 -> 0.01 "makes idle units react faster when attacked"; ordered 0.22 -> 0.75 "reduces reaction speed to melee when units have active orders"; ammo 0.85 -> 0.25; artillery 1 -> 0.5 | as listed | https://steamcommunity.com/sharedfiles/filedetails/?id=3484979921 | community |
| Another mod that lowers these says it makes ranged units and artillery "far more susceptible to be pinned in melee" | — | https://steamcommunity.com/sharedfiles/filedetails/?id=3482692092 | community |
| **Reading:** the threshold decides **when a unit drops its current order and switches to melee** (probably a share of the unit in contact: missile units with ammo switch only at 85 %, artillery never). It is **not** an attack-rate multiplier for idle units | — | from the two rows above | speculation, well supported |
| Units drop their order and default to fighting adjacent enemies when any model is in contact | — | https://steamcommunity.com/app/1142710/discussions/0/4416424085342483518/ | community |
| 5.0.0: removed a limit where "only 35% of the chasing unit was allowed to initiate an attack" (pursuing routers) | 35 % | https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0 | confirmed |
| **Test "idle units deal less damage" with numbers** | — | **nothing found** | — |

## Q4. Melee breakoff

| Claim | Number | Source | Confidence |
|---|---|---|---|
| kv: `melee_breakoff_secs` 24, `melee_breakoff_total_immunity_secs` 10, `break_off_ignore_collision_time` 0, `break_off_speed_and_direction_checks` 0 | as listed | WH3 dump | confirmed values |
| "There's a time-window that was bumped up significantly (to about 25 seconds) for the express purpose of allowing cavalry to escape melee"; "a 25 second window where a unit will retain its order regardless of engaging in melee" | ~25 s | https://steamcommunity.com/app/1142710/discussions/0/4416424085342483518/ ; https://steamcommunity.com/app/1142710/discussions/0/4336483638876545675/ | community, matches kv 24 |
| Reading: a unit ordered out of melee keeps its move order for 24 s; for the first 10 s it is fully immune to being pulled back into melee | 24 / 10 s | — | speculation |
| WH2 1.12.1: get-up delay raised to 1–2.7 s because knocked infantry got up inside cavalry "and disrupt disengaging when cycle charging" | 1–2.7 s | updatecrazy 1.12.1 | confirmed |
| Damage while withdrawing: the leaving entities show their flank or rear (melee defence x0.6 / x0.3); players say "The moment you turn your back your MD goes down to 0" (exaggeration); a lord "wades through units with 0 speed whilst getting clobbered" | x0.6 / x0.3 | https://steamcommunity.com/app/1142710/discussions/0/3805027459309954482/ | community, **no measured numbers** |
| WH3 patch notes about the disengage timer | — | **nothing found** | — |

## Q5. Abilities (fandom / honga pages via search snippets; check against local gamedb)

| Ability | Numbers | Source | Confidence |
|---|---|---|---|
| Foe-Seeker (General, self) | 25 s, recharge 60 s, speed x1.25, charge speed x1.25, fatigue change; WH1 version was 12 s | https://totalwarwarhammer.fandom.com/wiki/Foe-Seeker | community wiki |
| Stand Your Ground! | 18 s, recharge 90 s, melee defence +24, leadership +16, 35 m, allies | https://totalwarwarhammer.fandom.com/wiki/Stand_Your_Ground! | community wiki, less sure (MD +24 looks large) |
| Hold the Line! (passive) | melee defence +5, leadership +4, 35 m, allies | https://totalwarwarhammer.fandom.com/wiki/Hold_the_Line! | community wiki |
| Rally! | leadership +16, 35 m, 14 s, recharge 60 s (WH1: 10 s / 45 s) | https://totalwarwarhammer.fandom.com/wiki/Rally! | community wiki |
| Verminous Valour (Warlord) | 17 s, recharge 60 s, speed +25 %, leadership +8, 25 m | https://totalwarwarhammer.fandom.com/wiki/Verminous_Valour | community wiki |
| Deadly Onslaught (Warlord, self) | 31 s, recharge 90 s, charge bonus x1.6, AP damage x1.25, base damage x1.25 | https://totalwarwarhammer.fandom.com/wiki/Deadly_Onslaught | community wiki (may be WH2 numbers) |
| "Initial recharge" = the cooldown the ability already has at battle start (most abilities 0) | — | modding guide https://steamcommunity.com/sharedfiles/filedetails/?id=1698960734 | community (modder) |

## Q6. Lord against many infantry

| Claim | Number | Source | Confidence |
|---|---|---|---|
| "Small units that are surrounded can be attacked by roughly 9 small units. Horse sized units i think is 12. Larger can get much much higher." | ~9 / ~12 | guide 719168880 | community estimate, no test |
| WH3 6.0.0: "For every 100 weapon strength a unit has, they gain an additional splash attack target, rounded to the nearest whole number (but no less than 1)"; 6.0.1 reverted this for non-single-entity units "while maintaining the single entity splash attack changes" -> **lord max targets = round(weapon strength / 100)**, damage split evenly | WS/100 | https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/53-total-war-warhammer-iii-update-6-0-0 ; https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/8337-total-war-warhammer-iii-hotfix-6-0-1 | confirmed |
| Splash damage "isn't multiplied, it's split"; each target in the splash rolls its own hit chance | — | https://steamcommunity.com/app/1142710/discussions/0/3842178984949679291/ | community |
| Older numbers: lords ~4 targets (WH2), ~6 targets (WH3 guide) | 4–6 | https://steamcommunity.com/app/594570/discussions/0/2280456883445499939/ ; guide 2776861563 | community, outdated by 6.0 |
| Flank/rear: defender melee defence x0.6 / x0.3 when the attacker is in the left/right/rear quadrant **of the target entity** | 0.6 / 0.3 | CA blog "Feature Focus #2: Damage (Part 1)" https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1 | confirmed |
| Per-entity: "40/70% melee defense penalty ... on a per entity basis, so when those rear models turn around ... it stops applying"; single entities take front, flank and rear attacks from the same unit | 40 / 70 % | https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/10108-flank-attack-damage-increase ; fandom "Flanking" (snippet) | community; fandom's "-30 / -60 %" conflicts with the DB and looks outdated |
| `melee_outnumbered_defence_penalty` 1 | 1 | kv | meaning unknown |
| **Test numbers for a lord surrounded by N infantry** | — | **nothing found** | — |

## Q7. First contact of two infantry blocks

| Claim | Number | Source | Confidence |
|---|---|---|---|
| "All units which have an enemy unit in range of their model will attempt an attack animation" -> no rank rule; any man with an enemy in reach attacks | — | guide 719168880 | community |
| "Only outer row unit models deal damage" | — | https://steamcommunity.com/app/594570/discussions/0/3109144584162927066/ | community |
| Back ranks used to charge-attack through friends; since WH2 1.12.1 they do not if a friendly entity blocks the path (about 35 % fewer attacks) | -35 % | updatecrazy 1.12.1 | confirmed |
| "Formation Attack" (Grand Cathay attribute only) keeps the unit in formation in melee with slightly higher melee defence; kv `melee_defence_formed_attack_bonus` 5, `entity_action_attack_formed_combat_distance` 2.5, `_tether_distance` 2.0 -> other infantry fights loose and blobs | +5 MD | https://totalwarwarhammer.fandom.com/wiki/Formation_Attack (snippet) + kv | confirmed values / community |
| Matched combat kv: `matched_combat_percentage` 50, `_formed_vs_formed` 40, `_formed_vs_free` 60, `_formed_charge` 20, `generic_matched_combat_percentage` 60, `min_hitpoints_to_initiate` 15, `_whilst_active` 5 | as listed | kv | values confirmed, meaning not documented |
| A sync kill locks both entities; the lord "still takes damage" during it | — | search snippets of Steam threads (sync kills) | community |
| Rome 2 / Attila: matched combat made units "push into each other and form a big blob" | — | twcenter (snippet) https://www.twcenter.net/threads/matched-combat-no-matched-combat-an-observation.768316/ | community (older engine) |
| **How fast blocks mesh; whether both front ranks strike at once** | — | **nothing found** | — |

## Side notes
- `melee_attack_interval` 4.0 in WH3 kv (the WH1 guide said 3.8 s). `charge_decay_duration` 13, `charge_reflect_damage_multiplier` 2, `melee_hit_chance_base/min/max` 35/8/90.
- Patch 6.1 increased the mass of many cavalry and lords (for example 1300 -> 1500); players say infantry became "stickier" (https://steamcommunity.com/app/1142710/discussions/0/591781130180569696/).
