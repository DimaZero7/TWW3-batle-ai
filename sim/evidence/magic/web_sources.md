# Battle magic in TW:WH3 — web research (12.10.2026)

Access notes: totalwarwarhammer.fandom.com returns HTTP 402 to WebFetch, honga.net 403, breezewiki = captcha.
Fandom text below comes only from search-result excerpts (marked "excerpt"). Best structured source found:
**twdatabase.com** (fan site built from game data, says "patch 9.0.3, October 9, 2026") — current patch is 9.0.3, not 6.x.

## 1. Winds of Magic in battle

| Claim | Source |
|---|---|
| Two pools: Power (current, castable) and Power Reserve (what can still flow in during the battle); recharge = time to move power from reserve into the pool | https://gmanreviews.blog/total-war-warhammer-3-magic-guide/ |
| Reserve cap per army = 100; 105 with the "Magical Reserves" skill; two casters with the skill = 110 | https://steamcommunity.com/app/1142710/discussions/0/3496509339501484197 ; https://steamcommunity.com/app/1142710/discussions/0/3378285310306894046/ |
| Magical Reserves skill = +5 reserve capacity; Gold Scholar (Gold Wizard) +5 too | https://twdatabase.com/skill-planner/gold-wizard |
| Battle reserve comes from the army's campaign reserve (blue bar), which rises/falls each turn by the region's winds (up to ~15/turn; Channelling stance +10/turn) | https://steamcommunity.com/app/1142710/discussions/0/3496509339501484197 ; https://www.pcgamesn.com/total-war-warhammer-3/storm-of-magic-reserve |
| Starting pool drawn randomly at battle start, does not count toward the reserve max | https://www.gamepressure.com/total-war-warhammer-3/winds-of-magic-and-spells/z7f97a |
| Only the army that started the battle supplies winds; reinforcing armies add nothing | https://steamcommunity.com/app/1142710/discussions/0/6400272691576770232/ ; https://steamcommunity.com/app/1142710/discussions/0/3806155895391772037 |
| Chaos Realms: reserve infinite, only limit is the recharge speed | https://www.pcgamesn.com/total-war-warhammer-3/storm-of-magic-reserve |
| WH3: starting reserves much smaller than WH2, recharge "much quicker than it was in W2", constant through the battle (no slowdown) — "If you went into battle with 60 mana, you can eventually access all of it" | https://1d6chan.miraheze.org/wiki/Total_War_Warhammer/Tactics/Magic |
| CONFLICT: Steam/fandom say recharge slows as the reserve empties ("faster the more power reserve you have") — fandom text "power meter 0–30, up to 1 every 3 seconds" is the WH1/WH2 description (excerpt) | https://totalwarwarhammer.fandom.com/wiki/Winds_of_Magic (excerpt) ; https://steamcommunity.com/app/364360/discussions/0/1483233503862181053/ |
| Recharge modifiers are % (Arcane Conduit +40 %, Glean Magic +80 % own / −40 % enemy for 23/46 s and +0.1 reserve/s; Prismatic Amplifier fixed to +160 % in 5.0.0) | https://steamcommunity.com/app/1142710/discussions/0/3415431214836301975/ ; https://steamcommunity.com/app/1142710/discussions/0/3826415019288347326 ; search excerpt for https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0 |
| Patch 4.1.1: "+% reserve when increasing" effects changed to "+x reserve capacity" | search excerpt, https://steamcommunity.com/sharedfiles/filedetails/?id=2882713166 |
| Base recharge rate per second in WH3, current-pool cap in WH3, custom-battle winds setting/defaults | **not found** (check db: these are game-data numbers; twdatabase has no winds page) |

## 2. Costs, cooldowns, overcast, miscast, casting

| Claim | Source |
|---|---|
| Overcast = double-click: more expensive version, longer cooldown, extra effects (more damage/duration/radius/range) | https://steamcommunity.com/app/1142710/discussions/0/3185738755277443230/ ; https://gmanreviews.blog/total-war-warhammer-3-magic-guide/ |
| **Miscast exists in WH3**, only on overcasts: spell still goes off, caster takes damage | same two + https://steamcommunity.com/app/1142710/discussions/0/3806155895391772037 |
| Miscast chance reduced by skills: Earthing −15 %, Star Student −25 %, spell level-3 node −15 %, Trickster's Shard −20 %, Gelt's staff −50 % | https://steamcommunity.com/app/1142710/discussions/0/4416424597686092104/ ; https://twdatabase.com/skill-planner/gold-wizard |
| Miscast explosion 288 base + 300 AP explosive (magic damage, cut by ward/magic resist) — set in the WH2 Hammers & Herdstones update (2021); "50 % miscast chance on overcast" appears in an excerpt, unverified | fandom excerpt, https://totalwarwarhammer.fandom.com/wiki/Winds_of_Magic |
| Base miscast % in WH3 | **not found** (only the excerpt above) |
| Wind-up 1–5 s per spell (see table §3) | https://twdatabase.com/spells/skaven-spells-of-ruin |
| Caster ordered to cast out of range walks forward until in range | https://steamcommunity.com/app/1142710/discussions/0/3826413850807975608/ |
| WH2: "a hit will cancel any pending casts" (melee interrupts wind-up) | https://steamcommunity.com/app/594570/discussions/0/2572002906848940647/ |
| WH3: can a wizard cast while in melee / does a hit interrupt | **not found** for WH3 |

## 3. Spells (twdatabase, patch 9.0.3)

Lore of Metal — https://twdatabase.com/spells/lore-of-metal
- Lore attribute **Metalshifting**: on casting, +12 % AP weapon damage and +12 % base weapon damage (1d6chan says +10 % AP & weapon strength for 13 s — older value: https://1d6chan.miraheze.org/wiki/Total_War_Warhammer/Tactics/Magic).
- **Searing Doom** (bombardment): cost 6 / overcast 10, cooldown 30 s, range 200 m, duration 8 s, wind-up 1 s; 40 projectiles × 36 dmg (12 AP), explosion 30; overcast 80 projectiles. "Strong vs a single unit, weak vs a single combatant". Search excerpt: ranged 36 (33 % AP) + explosive 30 (100 % AP), overcast = bigger radius. Radius: **not found**. Bonus vs armour / burning: **not found** (none listed).
- Casters: Gelt, Archmage (Metal), Gold Wizard, Alchemist, Mage (Metal), Chaos casters (17 units).

Skaven Spells of Ruin — https://twdatabase.com/spells/skaven-spells-of-ruin
- Lore attribute **Musk of Fear**: on casting, enemies −12 melee attack, −4 leadership.
- **Warp Lightning** (bombardment, magical): 6 / 11, cd 30 s, range 200 m, duration 5 s, wind-up 1 s; 5 projectiles × 224 (all 224 AP) + explosion 24; overcast 10 projectiles. Per-bolt radius: **not found**.
- **Howling Warpgale** (hex, area): 5 / 9, cd 30 s, range 100 m, radius 55 m, duration 14 s (28 s overcast), wind-up 1 s; effect "makes flight impossible" — nothing else listed (no damage, no slow, no missile effect for ground units). Players: "can only be used on flying units". https://www.dualshockers.com/warhammer-3-immortal-empires-best-lores-of-magic/ (excerpt)
- **Death Frenzy** (augment): 6 / 10, cd 43 s, range 200 m, 28 s, wind-up 3 s; +24 MA, +30 % AP and base weapon damage.
- **Scorch** (breath, magical, flaming): 7 / 11, cd 44 s, range 150 m (450 overcast), 12 s, wind-up 3 s; 36 dmg/s for 3 s, radius 6 m (overcast 6 s).
- **Cracks Call** (wind, magical): 15 / 20, cd 52 s, range 150 m, 10 s, wind-up 5 s; 35 dmg/s (35 AP), radius 7 m; overcast 70 dmg/s.
- **Flensing Ruin** (direct damage area): 15 / 22, cd 52 s, range 100 m, radius 35 m, 22 s (34 overcast), wind-up 5 s; −30 armour, strong vs characters.
- Best vs ground infantry: Warp Lightning (cheap) and Cracks Call (wind through blobs); Flensing Ruin for single entities/small units; Scorch "can often be ignored". (excerpt, https://www.dualshockers.com/warhammer-3-immortal-empires-best-lores-of-magic/)
- Casters: Ikit Claw, Warlock Master (lords), **Warlock Engineer** (hero).

## 4. Ranks, skills, unlocks

| Claim | Source |
|---|---|
| Max rank 50, 1 skill point per rank, 49 points total (rank 1 has 0 to spend) | https://gamerant.com/total-war-warhammer-3-max-level-guide/ ; excerpt https://totalwarwarhammer.fandom.com/wiki/Experience_and_rank |
| Lords/heroes get skill points on rank up; automatic stat bonuses per rank are for regular units (rank 1: +1 Ld, +1 MA, +1 MD) — characters' stats grow via skills | excerpt https://totalwarwarhammer.fandom.com/wiki/Experience_and_rank |
| Warlock Engineer: Warp Lightning = starting skill (rank 1), 2 levels incl. "Overcast spell: Warp Lightning Upgraded"; Howling Warpgale / Death Frenzy / Scorch need Warp Lightning; Cracks Call / Flensing Ruin need Evasion; every spell has an overcast version → **Warlock Engineer CAN overcast** (unlocked by the 2nd level of each spell skill) | https://twdatabase.com/skill-planner/warlock-engineer |
| Gold Wizard: Searing Doom at rank 1; other Metal spells need Searing Doom or Evasion; Star Student at rank 12 | https://twdatabase.com/skill-planner/gold-wizard |
| Exact rank needed for each later spell node | **not found** (the planner text gives prerequisites, not ranks) |

## 5. Custom battle: hero rank and skills
**Not found** online (no source describing a rank/skill picker in WH3 custom battle). Check in the game UI.

## 6. Battle AI and magic
| Claim | Source |
|---|---|
| AI sees spell target markers like a player and dodges incoming spells unless engaged; pin targets in melee first | search excerpt, https://steamcommunity.com/app/1142710/discussions/0/3820783808420169044/ |
| Spells need AI usage keys in the db to be used by the AI | search excerpt, https://dasang.github.io/blog/warhammer_mod/ |
| AI pounces on unprotected casters that walk forward to cast | https://steamcommunity.com/app/1142710/discussions/0/3826413850807975608/ |
| When/how the AI decides to cast, target choice, whether it saves winds | **not found** |
