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
  - Ours: **by the formula, at weight 1 for every blow** (since 06.10.2026). Why the pairs looked "flat"
    (weight 0.1 suited them): the attack interval only starts after a hit, and a miss costs ~0.5 s — a man
    in formation lands p / (p × interval + 0.5) hits a second, not p / interval (fitted on 121 recorded
    formation-vs-formation battles, 52 pairs, 20,034 s, `build/hitchance`); the other half is smoothed
    overkill (below).
  - Ours: **lord against lord** — one blow per interval, p / interval. In duels of two lords alone on the
    field the blows were counted as health drops (28 battles): General on General hits 41 % of his blows
    (formula 40 %: 55 against 45 + 5 Hold the Line), Warlord on Warlord 19 % (formula 30 % fresh, ~22 % with
    fatigue); a blow takes 245 / 223 HP — exactly the passport and armour at its mean 75 %.
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
  - Ours: agrees — the mean of the roll (75 % of armour up to 100; above 100 — the mean capped at 100 %
    per roll: 2 − 100/A − A/400, 95.8 % at 150; `melee.armour_cut`). The recordings match to single
    digits: a General-on-General blow is 244.9 HP on average in the game, 245.1 by the rule (334 blows);
    infantry blows on a lord are whole numbers within the rule, rounded to the nearest
    (`build/damage/spec.md` D2).
- **Order of modifiers**: base → bonus vs type → charge → height → armour (base only) →
  resistances (summed, capped at 90 %) → rounding (base and AP rounded separately). Overkill
  is lost. Base damage has no floor (can be 0). WH3 · [CA damage blog][dmg] · high.
  - Ours: **smoothed overkill** (`melee.per_hit`): a unit loses hp / E[N] per blow, where E[N] is how many
    blows a fresh man needs — the first blow is exact (it kills if even the best armour roll leaves no
    more than his health), after that the mean: E[N] = 1 + P(first did not kill) × max(1, (hp − mean first
    blow) / mean blow + ½). A single entity (a lord) gets no overkill: his health is one pool. Examples:
    Greatswords on Night Runners 38.4 → 28 HP a blow, the General's share on Stormvermin 58.6 → 38.6, an
    arrow on spearmen 15.2 → 13.7. The exact step (whole blows per man) explains the 121 battles worse
    (error 0.174 against 0.146 without overkill), the smoothed version as well or better (0.144) and is
    right where overkill shows (a lord against infantry).
  - Ours: **melee kills — the wounded pool** (`kills.wound_pool`). Since every model has its own health and
    overkill is lost, the wounded among the living (W = men × a man's health − the unit's health) cannot grow
    without end: each struck man is cut down in E[N] = a man's health / blow, so W stays at most W* = its own
    men in contact × (a man's health − the health-weighted blow of its strikers) (a unit not striking back: the
    enemy men striking it); health lost beyond the pool is whole men. The probe's health series: flagellants →
    clanrats — 4–7 men's worth wounded while 87 die, slaves 10–20; greatswords → slaves — 0 while one blow kills
    (49 + charge ≥ 50 HP), 13–20 once the charge fades. Men killed at 15–30 s 1.00 → 1.10 of the game, at
    60–90 s 0.82 → 1.15 (the old `kills.exponent` 0.5 rule kept the wounded for ever). Missiles: even hits on
    men with their own health (`kills.missile_uniform`, [missiles](missiles.md)), a lord is one pool.
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
- **The first blow comes at once.** Since the interval runs only after a blow, a man who first reaches an
  enemy strikes at once, without waiting. Hence the burst in the first 0.5–1 s after a charge (the melee
  probe, below).
  - Ours (melee core 2): a unit that comes into a fight moving (its melee clock has just started and it was
    moving) or whose charge lands strikes **once at once with every man in contact** (F × struck × hit chance ×
    blow, with the charge's full bonus), then at the usual rate. A unit standing under hold that an enemy
    reaches strikes once at once too — with its men in contact facing the enemy (within `contact.front_deg` 45°
    of its front; `contact.stand_first_strike`); the first strike is not cut by the hold share (0.5): every man
    in reach swings. The probe: the charger loses in the first second 106–194 HP to braced spearmen, 120 to
    swordsmen, 60–131 to units given an attack order at contact, 0 to a target facing away; the struck side's
    first-second error 1.05 → 0.41. No fitted numbers.
- **Only models that reach strike.** Damage needs an attack animation that physically reaches
  a target; any model with an enemy within weapon reach tries to attack, so a wider front means
  more attackers. WH1–WH3 guides · [WH3 kv guide][g3], [WH1 kv guide][g1] · medium.
  - Ours: 0.5 of the files in contact strike (`melee.fighting_files`, fitted on the pairs); a file is the
    contact length / this unit's formation step from the database (h across the front, v across the
    flank: `unit_spacings`, the Empire 1.48 × 1.6 m, clanrats 1.6 × 1.7, slaves 1.6 × 1.8). The database
    has no number for how many strike.
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
  - Ours: agrees — a bonus to attack (weight 1) and to damage (at the weapon's AP share), given in full
    on the first blow, fading to 0 over 13 s on its own clock: leaving contact does not stop or reset it,
    a new charge starts it over. Nothing beyond the bonus (the fitted `melee.impact` ×2.5 charge blow and
    the 20 s "bringing men in" for a unit that did not charge were removed on 06.10.2026). Recordings:
    damage in the first 5 s after contact is 1.31× the 10–20 s pace at a charge bonus ≤ 6 and 2.24× at
    ≥ 8; the weight-1 rule gives 1.2× and 1.6–1.9× on top of an overall ~1.1× decay
    (`build/charge/spec.md` 2.1).
- **Only a real charge counts.** The bonus applies only if the entity reaches its hidden charge
  speed before contact; walking into melee gives none. Charge speed is ~20–40 % above run speed.
  WH3 community · [WH3 kv guide][g3], [charge speed thread][chspeed] · medium.
  - Ours: a charge counts if the unit has an attack order and a run-up ≥ 10 m at ≥ 0.75 of run speed
    (`sim.json charge`: the Empire's walk is 0.5 of run, not a charge; an attack order in the first 2 s of
    contact counts too, if there was a run-up). **The charge sprint** (melee core 2, the database's rule):
    under any attack order - at a run or at a walk - a unit covers the last `charge_distance_commence_run`
    (30 m, lords 35; `battle_entities`) to its target at its charge speed (`charge_speed`), and the run-up
    counts - an attack at a walk is a charge too. A move order gives no sprint. The probe: running 3.0 m/s →
    the last 30 m at 3.65–3.88, the last 10 m 3.9–4.7. The run-up counts only the speed towards the attack
    target (without one, towards the nearest enemy), not running away (`charge.runup_towards`: a Warlord that
    ran ~10 m off came back at 1.5 m/s and struck an ordinary blow 3.5 s after his melee flag).
- **Charge distances** are per entity (`battle_entities`): `charge_distance_commence_run`,
  `_adopt_charge_pose`, `_pick_target`; typical switch to charge speed 20–60 m, pose ~35 m.
  WH3 · [twwstats][tws], [charge distance thread][chdist] · high (fields), medium (values).
- **Blocked chargers don't charge.** Since WH2 1.12.1 an entity whose path is blocked by a
  friend makes no charge attack (back ranks of a counter-charge): ~35 % less counter-charge
  damage. Still the WH3 base · [1.12.1 notes][p1121] · high. CA (WH2): no charge when the path is
  blocked by one's own unit · [pcgamesn][pcgcav] · high.
  - Ours: no sprint and no charge when the attack target is already in melee with another of our units — it
    comes in at a run (`charge.free_target_only`: entering a fight where the target is surrounded by friends is
    a blocked path). The recordings of 221 battles (the centre's speed 2 s before contact / the card's run,
    median): into a free target of its own order the game's AI 1.12 (standing) / 1.02 (moving), the network 1.09
    (p75 1.20–1.26 — the charge speed); into one already fighting 0.79–0.86. Whether entering another unit's
    fight gives the charge bonus — open (no data, a probe).
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
  - Ours: by the database — a unit with reflection that is braced (standing: slower than 0.5 m/s,
    `melee.braced_speed`) deals the charger ×2 damage within 80° of its front while the charger's charge
    factor is ≥ 0.7 (3.9 s); it gets no charge bonus of its own. These blows are not cut by the hold share
    (0.5) (`contact.hold_reflect_full`): the probe — braced spearmen 52 HP/s in 1–5 s after a charge = the rule
    ×2, held swordsmen without reflection 12 HP/s; Steam players: braced models strike whatever is adjacent.
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
  - Ours: a lord's blow is divided by the cap (4) but hits an average of **2.07 men**
    (`contact.lord_splash_struck`): that is what the "lord surrounded" probe's recordings show (506
    blows, 1–4 units around him, 1.84–2.30 by layout); each man hit gets ¼ of the blow, then armour and
    overkill. Dividing by the cap rather than the number hit shows up in the General's blows on
    Stormvermin: 47–57 HP — not lethal (¼ of a blow after armour), whereas dividing by the number hit
    would kill each one. A single entity (a lord) takes the whole blow.

## Unit size

- Single entities' health and damage and direct-damage spells scale with the unit-size setting:
  Small 25 %, Medium 50 %, Large 75 %, Ultra 100 %; models per unit scale the same way. WH3 is
  balanced around Ultra. · [GameWatcher, CA's James Martin][gw], [WH3 kv guide][g3] · high/medium.
  See [battle rules](battle-rules.md).
  - Ours: the arena units have Ultra sizes (spearmen 120, clanrats 160, slaves 180).

## In-game check: the melee probe

One mechanic, one scenario, the game and the simulator alike ([melee probe](../../apps/entries.md#charge_probe)):
in a lane an attacker and a target, everyone held by script and fearless; the simulator plays the same lane
from the same places with the same orders (8 copies). 11 battles, ~21 min of the machine: the `charge` plan
8 battles, the `hit` plan 2, the lord swarm's `damage` 1. Recordings and tables: `build/meleetests` (not in
git). Until the simulator has every base indicator, each mismatch is **OPEN** with candidates, not a bug; no
fitting on these numbers.

**Charge: a volley in the first second.** HP the target loses (swordsmen on clanrats - 4 lanes each, the rest
1-2); game / simulator before / after melee core 2 (the first strike, the charge sprint; the simulator's contact
is the start of the step in which the units met):

| Attacker → target, order | 0–1 s | 1–5 s | 5–15 s | From 15 s, HP/s |
|---|---:|---:|---:|---:|
| swordsmen → clanrats, attack at a run | 220 / 51 / 189 | 218 / 191 / 191 | 391 / 388 / 388 | 29 / 34 / 34 |
| swordsmen → clanrats, attack at a walk | 248 / 35 / 189 | 184 / 138 / 191 | 537 / 345 / 390 | 33 / 34 / 34 |
| swordsmen → clanrats, move order | 78 / 35 / 107 | 102 / 138 / 138 | 180 / 345 / 345 | 20 / 35 / 35 |
| clanrats → braced spearmen, at a run | 148 / 38 / 115 | 232 / 144 / 144 | 326 / 296 / 296 | 25 / 26 / 26 |
| clanrats → braced spearmen, at a walk | 118 / 27 / 115 | 158 / 106 / 144 | 390 / 266 / 297 | 29 / 26 / 26 |
| clanrats → spearmen facing away | 189 / 43 / 172 | 342 / 162 / 162 | 326 / 344 / 344 | 29 / 27 / 27 |
| swordsmen → skavenslaves | 268 / 48 / 201 | 144 / 190 / 190 | 654 / 438 / 439 | 46 / 39 / 39 |
| greatswords → skavenslaves | 300 / 86 / 429 | 400 / 342 / 342 | 1,224 / 736 / 736 | 63 / 41 / 41 |
| spearmen with shields → skavenslaves | 210 / 39 / 129 | 92 / 151 / 151 | 466 / 351 / 351 | 37 / 33 / 33 |
| flagellants → clanrats | 296 / 54 / 277 | 303 / 208 / 208 | 639 / 464 / 465 | 53 / 39 / 39 |
| spearmen → skavenslave spearmen, from 150 m / 20 m | 98, 141 / 31 / 107 | 178, 92 / 119 / 119 | 378, 362 / 275 / 275 | 36, 32 / 26 / 26 |
| General → clanrats, at a run / a walk | 0, 0 / 27 / 139 | 237, 241 / 106 / 109 | 361, 418 / 260 / 262 | 17, 24 / 25 / 25 |
| Warlord → swordsmen | 272 / 31 / 157 | 345 / 124 / 124 | 211 / 296 / 296 | 30 / 29 / 29 |

- Almost all the game's extra damage is in the first 0.5–1 s after contact (×6–17 the steady pace); from 1 s
  the pace is close to the rule. The game's rule: the interval runs only after a blow, so every man who reaches
  an enemy strikes at once. The simulator now does so (**the first strike**): 107–429 HP in the first second
  instead of 27–86; from 1 s nothing changes. An attack at a walk is now a charge with the same burst (189, game
  248).
- OPEN: the greatswords' burst on skavenslaves is above the game (429 against 300), the shielded spearmen's and
  the swordsmen's on skavenslaves below (129 and 201 against 210 and 268); the General's first blow in the game
  comes 3–5 s after contact (0 HP in 0–3 s), in the simulator at once (139); the Warlord's burst is below the
  game (157 against 272).
- The "recharge" (pull back 35–40 m and attack again) turned out to be a chase: the swordsmen never left melee
  (nearest men 0.8–1.8 m apart, the melee flag on for all 25 s), the "second contact" is the lane's end just
  after the `melee_breakoff_secs` 24 s window. The simulator released them after 20 s and they came back with a
  fresh first strike (the target lost 341 HP in 5 s, game 250); with the chase (`contact.chase`) 164; the
  Warlord's lane 143 → 215 (game 204). A new strike by every man after a pause in striking is not modelled (it
  added nothing in these lanes).
- OPEN: a target ordered to attack in the second of contact gives no burst (the attacker loses 16–22 HP in
  1 s). One standing without an order does (clanrats lose 42–172 HP in 1 s on spearmen and swordsmen) — now
  in the simulator too (the first strike of a standing unit, above). A move order into contact gives a smaller
  burst (78) and then ×0.5–0.6 of the rule: now a formation under a move in contact strikes at the 0.5 share
  (`contact.hold_move`; the target in 5–15 s 345 → 173 HP, game 180).
- The attacker has 20–36 men within 2.5 m of an enemy in the first second, 10–16 after 15 s; there are more of
  them at 5–15 s than at 15–30 s at the same pace - the number of men in contact alone does not explain it.

**Charge: speed and order.** With an attack order (at a run and **at a walk**) a unit covers the last ~30 m at
charge speed: on average 3.6–3.9 m/s over 30 m and 3.9–4.7 m/s over 10 m at a run of 3.0 (clanrats 4.3 =
4.8 × 0.9); from 150 m and from 20 m alike. An attack at a walk: 1.56 m/s, then the sprint and the same burst as
at a run - in the game it is a charge. A move order gives no sprint (2.8 m/s). The simulator now follows the
database's rule (**the charge sprint**, `charge_distance_commence_run` 30 m / lords 35 at `charge_speed` under
any attack order): swordsmen over the last 30 / 10 m 3.57 / 3.19 m/s (before 2.86 / 2.61; game 3.65 / 3.89),
contact of an attack at a walk at 50 s (before 63, game 48). OPEN: over the last 10 m the simulator is ~0.7 m/s
slower than the game.

**Braced spearmen and a unit standing without an order.** A unit holding (no order) strikes from 5 s on at
0.49–0.52 of the rule in the game: braced spearmen - clanrats lose 12 HP/s (the rule in full 23), spearmen
facing away 10 (20), swordsmen 17 (35); a unit with an attack order strikes at the rule (clanrats 29 / 27). So
`contact.hold_rate` 0.5 stays for formations (FITTED, no rule for the rate: the keys
`melee_attack_threshold_modifier_*` are the threshold for joining a fight, B8), and a lord without an order
strikes in full (below). In the first second a holding unit strikes hard in the game (42–172 HP, braced ones
with reflection) — now this is the first strike of a standing unit and the reflection without the 0.5 share
(above).

**Hit chance (plan `hit`, from 15 s to the end).** Target HP/s, game / simulator: swordsmen → skavenslaves
46 / 39, greatswords → skavenslaves 63 / 41, spearmen with shields → skavenslaves 37 / 33, flagellants →
clanrats 53 / 39; the skavenslaves back 16 / 15, 10 / 9, 13 / 11; clanrat spearmen → spearmen 20 / 18 and back
21 / 22. Blows a second a man at 10 strikers (unarmoured skavenslaves: HP / blow damage): 0.15–0.25 against the
rule's 0.14–0.20 - the slope in the hit chance is right. Skavenslaves lose more men a second in the game:
0.89 / 0.51, 1.00 / 0.58, 0.58 / 0.42. Kills are now counted by the wounded pool (above, "Damage and armour");
it covers about half of the skavenslaves' gap, the rest is the HP rate (swordsmen on skavenslaves 39 against
46 HP/s, greatswords 41 against 63) — OPEN. Flagellants: both sides strike ×1.4–2 the rule (flagellants →
clanrats 53 against 39 HP/s, back 51 against 31). Probe P2 (`build/probes7`, plan `syg2`, 2 lanes each): flagellants
→ clanrats, the target loses 784 / 806 HP at 0–5 / 5–15 s, the simulator 485 / 464 (×1.6–1.7), from 15 s 51 against
39 HP/s; the flagellants in return 264 / 250, from 15 s 51 against 31; clanrats → held flagellants: the flagellants
lose 520 / 534 against 296 / 329 (×1.6–1.75), the clanrats 286 / 298 against 205 / 200; for comparison swordsmen →
clanrats 493 / 530 against 380 / 389 (×1.3). The gap is largest in the first 15 s and where the hit chance is high
(flagellants have almost no defence) — OPEN, together with Stand Your Ground (below).

**Stand Your Ground** (+24 defence, DB). Spearmen lose 107 HP in the first 15 s with the ability against 440
without (x0.24, one battle). Probe P1 (`syg2`, 2 battles; one General a battle, so one ability lane a battle, cast when
the centres are 25 m + the half depths apart): 0–5 s 34 against 129 HP (x0.26), 5–15 s 136 against 264 (x0.52),
0–15 s x0.43 (lanes 0.33–0.56); after it ends (15–30 s) 336 against 300. The simulator x0.54 / x0.71, over 0–15 s
x0.63. A hit chance of 25 % → 8 % (with the charge) and 19 % → 8 % give x0.32 / x0.42 — the game sits at those, the
simulator at the law p / (p × interval + miss cost). With the flagellants (P2, a high hit chance: the game above the
simulator) this points to a blow rate steeper in the hit chance than the simulator's law — OPEN (the law comes from
121 battles; too early to change it on 4 lanes).

**Damage a blow (plan `damage`, 8 trials).** A greatswords' blow on the Warlord - median 41 HP (quartiles
39–45; 71 % in 36–43, none in 26–31): **the +14 bonus against infantry counts against a foot lord** (as in the
simulator). Swordsmen on the Warlord 14 (rule 9–19), clanrats on the General 14 (9–18), stormvermin on the
General 27 (24–28). A lord's blow: on greatswords 105 HP and 0.60 killed (2.1 struck × a share of 50 HP after
armour - 2.07 holds on armour too), on swordsmen 136 and 1.97, on clanrats 143 and 2.45, on stormvermin 102 and
0.79; 0.18–0.19 lord blows a second (rule ~0.2). The lord loses HP/s, game / sim.: 23 / 22, 10 / 8, 32 / 41,
8 / 10. The unit loses to a lord held by script (no attack order), game / sim. before / after: 18 / 8 / 16,
29 / 13 / 26, 20 / 8 / 16, 24 / 14 / 29 - a lord without an order now strikes in full (melee core 2).

**A lord in a crowd: the gather.** Infantry that ran onto a standing lord strikes him in full at once: in the
first 15 s (1–4 spear or clanrat units) the lord loses 4.9–9.7 HP/s in the game, the simulator gave 2.6–3.0
(a 20 s gather), now 7.1–8.8; the steady pace is unchanged. The gather `lord_gather_s` 20 s stays only for a
formation the lord himself ran into (the pairs: the lord loses 38 / 45 HP in the first 15 s in the game).

**An attack order on a far target from melee is a leave.** Infantry that is fighting and is told to attack an
enemy it does not touch (35 m or more away) nearly stops striking in the game: 0.01–0.03 men killed a second
against 0.27–0.40 for a unit long attacking its target; it is moving 90–98 % of the time but still in melee 87 % of
the time 10 s later and loses men as before. Whether the target lies beyond the enemy or the other way does not
matter (0.04 / 0.03 a second); kills come back at 25–30 s (0.34) - the `melee_breakoff_secs` 24 s window. Missile
units are not concerned (0.11–0.12, moving 15–18 % of the time). Measured on 24 network battles in the game (v2
sampled and greedy, grid_s46; `build/v2gap`). The simulator kept such a unit fighting in full (0.33 at any distance),
and the v2 network used it: in the game 60 % of its attacking melee seconds were on targets 40 m or more away
(grid_s46 16 %). Now it is a leave like a withdraw (`contact.attack_leave`): the target not touched and its edge
`leave_m` 10 m or more away, a unit without a missile weapon touching a standing enemy - it strikes nobody, is held
`pin_melee_s` 2 s, can be chased, the 24 s window drops the order. Fresh orders, game / simulator (v2 greedy, target
centre 35–50 / 50–80 / 80+ m): 0.06 / 0.06, 0.07 / 0.05, 0.12 / 0.10 a second (before 0.29–0.33). The twin of the same
24 battles against `ai_like` (4 copies): v2 sampled wins 16 -> 6 of 32, greedy 16 -> 3, grid_s46 16 -> 16 (game 0 / 0 /
3 of 8); the enemy's HP lost in melee 0.47 -> 0.34 (game 0.23), 0.48 -> 0.31 (0.38), grid 0.48 -> 0.48 (0.52); grid's
ordinary long fight 0.33 -> 0.32 kills a second (game 0.27). No rule online: players only say a unit in contact drops
its order and fights its neighbour and that the exit window is ~25 s; the database has only `melee_breakoff_secs` 24.

**A new order in melee (the `fresh` probe).** Swordsmen and clanrats attack each other from 3 m, a second clanrat
unit stands 4 m aside; 10 s after contact the swordsmen get one more order (2 battles, 2 lanes a variant,
`build/charge-probe/runs/20261008-145559`, `-145641`). Kills by the swordsmen a second (HP the clanrats lose a second)
at 0–5 / 5–10 / 10–20 / 20–30 s after the order:

| Order | Game | Simulator before | Simulator after |
|---|---|---|---|
| none (control) | 0.80 (52) / 0.40 (33) / 0.45 (32) / 0.70 (32) | 0.59 (35) throughout | the same |
| the same attack again | 0.50 (60) / 0.40 (43) / 0.35 (35) / 0.60 (31) | 0.59 (35) | the same |
| attack the second unit | 0.20 (12) / 0.10 (35) / 0.05 (11) / 0.25 (29) | 0.59 (35) | 0 (0) / 0 (10) / 0.09 (17) / 0.50 (30) |
| walk 5 m back | 0.30 (22) / 0 (0) / 0 (0) / 0 (16) | 0.30 (18) | 0 (0) / 0 (0) / 0 (0) / 0.18 (11) |
| halt (the bridge's hold) | 0.70 (34) / 0.10 (39) / 0.35 (18) / 0.30 (23) | 0.30 (18) | the same |

Repeating the same attack costs nothing. Attacking the second unit, the swordsmen turn 75–80 deg, walk 8 m to it in
6–8 s and strike the clanrats they still touch not at all for 25 s after the order, then again - the
`melee_breakoff_secs` 24 s window. A unit told to walk 5 m back turns about, is chased and strikes nothing for
21–25 s. Rules (`contact.fresh_incidental` 0, `retarget_walk`, `leave_away_m` 1, `leave_latch`): an attack order given
in melee strikes only its target for 24 s; a unit in melee whose target is near but not touched walks to it; a point
away from every touched enemy is a leave from 1 m on (the smallest measured 5 m), and the leave lasts while the order
stands, up to the 24 s window. The twin of the 8 it1 battles: melee HP dealt 0.439 -> 0.428 (game 0.447); a new attack
on a near target 0.38 -> 0.23 kills a second (game 0.20), a settled fight 0.39 -> 0.30 (0.31); grid_s46 0.482 -> 0.486
(0.517). Online only: a unit in contact drops its order and fights its neighbour, the exit window is ~25 s.

**Move in melee (the `meleeorders` probe).** 10 s after contact the swordsmen run to a point 5 / 15 / 40 m ahead
through the clanrats or 5 / 15 m aside (`build/charge-probe/runs/20261008-162358`, `-162441`). In all 7 lanes the unit
is "moving" all the time, but the clanrats hold it and it barely strikes: kills a second (the clanrats' HP a second) at
0–5 / 5–10 / 10–20 / 20–30 s - ahead 5 m 0 (5) / 0 (0) / 0 (0) / 0.10 (15), ahead 15 m 0.10 (7) / 0 / 0 / 0.25 (47),
ahead 40 m 0 (17) / 0 / 0 / 0.30 (52), aside 5 m 0 (6) / 0 / 0 / 0.20 (24), aside 15 m 0.20 (21) / 0 / 0 / 0.40 (28);
control 0.60 (61) / 0.70 (42) / 0.55 (33) / 0.40 (33). After the 24 s window it strikes again. So a move order given
in melee is a leave in any direction (`contact.move_melee_leave`); one given before the contact is not (P3: the
planner's push through the enemy from the go fought on; R3: walking into the target 0.70 of attacking). The order's
clock `order_s` tells them apart, no new number. The twin of the probe: ahead 5 / 15 / 40 m was 0.30 (18) in every
window, now 0 / 0 / 0 / 0.18 (11). The twin of the 8 it3 battles: move in melee 0.12 -> 0.03 kills a second (game 0.02),
melee HP dealt 0.486 -> 0.497 (0.468), wins 19 -> 14 of 32 (game 2 of 8); it1 move 0.14 -> 0.04 (0.03); grid_s46 melee
HP 0.486 -> 0.486 (0.517).

## Not found

No public CA source explains `melee_breakoff_secs` 24 / `melee_breakoff_total_immunity_secs` 10
(WH3-only keys; see [movement](movement.md)); by the melee-exit probe (`build/movelords`) 24 s is the window a
leaving unit keeps its order: chased swordsmen and spearmen strike nothing for 24–26 s after a withdraw order
(kills flat, the chasing clanrats lose no health), then fight on (the clanrats lose ~20 HP/s); not chased, a
unit is out of contact ~4 s after the order (its own melee flag goes off only after ~9 s — probably
`melee_breakoff_total_immunity_secs` 10), chased, the chasers run 10–15 m behind it (the simulator:
`contact.breakoff`, `contact.chase`; a Steam player also writes of a ~25 s window); `entity_action_attack_formed_combat_distance`
2.5 / `_tether_distance` 2, or how many models can physically reach one target.

## What is still unclear

- **The opening wave and a weakening unit's blows (the `reform`, `reform2` probes).** 240 s fights from 3 m, the
  soldiers' places every second (`build/charge-probe/runs/20261008-180435`, `-181311`; `build/v2gap/reform_report.py`).
  Men within 2.5 m of an enemy: 43 in the first 10 s, 21 in the next 10, ~13 by 20–40 s, then 10–12 to the end - in
  every lane. Health goes 2.6x faster than the steady pace in the first 10 s while 3.4x as many men touch: each man
  in contact strikes less at first. The blows follow the men in contact; with few men alive ~0.42 of the living touch.
  The striker's losses cut its blows (swordsmen at 30 % deal 58 / 38 / 24 / 6 HP/s while they go 34 -> 10), the
  target's do not (the clanrats strike battered swordsmen as usual; the swordsmen strike clanrats at 30 % as usual
  until about ten are left). The formation's depth matters only at first: the wave grows with the length of the
  fronts' contact (clanrats 15 m wide - 28 in contact, 30 and 50 m - 41–43), then 10 / 11–12 / 13–14 touch at nearly
  any width. The simulator's men fighting are `fighting_files` x the front's width - not proportional to the width in
  the game; no wave (35 HP/s flat against the game's 85 / 46 / 36 / 32). Unexplained: full clanrats' blows fall
  38 -> 15 with 145 -> 46 men alive, the swordsmen's stay ~32 to the end (both sides exhausted by then). An
  estimate of a "wave" rule (an estimate, not taken): men fighting x 1 + 3.0·e^(−t/7.2 s) from the fight's start and
  at most 0.42 of the living - the probe twin's opening as the game (87 / 48 / 37), no fall of a weakening clanrat
  unit's blows; the twin of the 8 it3 battles: wins 14 -> 13 of 32 (game 2 of 8), enemy routs a unit 1.33 -> 1.18
  (game 0.84), melee HP dealt 0.497 -> 0.443 (0.468), our losses 0.68 -> 0.68 (0.80); grid_s46 wins 15 -> 14 (3 of 8).
  Not taken: the multiplier comes from the same health losses it is checked on (close to fitting), and the shift of
  the 8 battles is small - the main gaps (enemy routs, our losses) do not come from the melee pace's shape.

- **Hold in melee.** In whole battles the network's units in melee kill 0.08 a second under hold (twin 0.21), while
  in the probes a halt after attacking strikes 0.69–0.73 of the control (4 lanes; rule `hold_rate` 0.5) - the gap
  points the other way. The whole battles' hold episodes need a look: who touches the unit, from which side, what the
  bridge does after the order.
- **The first 15 s of contact are stronger in the game than the rule.** In the pairs the target loses
  499–764 HP in the first 15 s, while the rule (the charge bonus, no blow) gave the simulator 396–446. The
  melee probe showed where the difference is: nearly all of it in the first second of a charge's contact (a
  volley), after that the pace is the rule's; the simulator now gives the volley by the rule "the first blow at
  once" ([in-game check](#in-game-check-the-melee-probe)), and for a standing unit too; the "recharge" was a
  chase. Why a target ordered at the moment of contact gives no volley - OPEN.
- **A holding unit strikes at half the rule** (0.49–0.52, the probe), a formation under a move in contact the
  same (0.70 of attacking), a lord without an order at the rule. No rule for a holding formation's rate found
  (`contact.hold_rate` 0.5 - FITTED).
- **The pair of CA's planner's spearmen** (its "held" spearmen are under a far move order): at 60–120 s the
  exchange in the game is twice the simulator's on both sides. Probe P3 (`pair`, 240 s): spearmen under a move to a
  point 60 m through the clanrat spearmen attacking them fight — they deal 213 / 317 / 521 HP at 0–5 / 5–15 / 15–30 s
  and 29 HP/s after (answering with an attack order 361 / 319 / 377 and 21 HP/s, held 221 / 161 / 166 and 13); the
  simulator had them leaving (0 / 0 / 98). Now only a point away from the enemy is a leave (`contact.leave_away_only`):
  the simulator 187 / 112 / 169 and 11 HP/s. OPEN: in the game such a unit strikes like an attacker or more (1 lane),
  in the simulator at the move share 0.5.
- **Leaving melee unchased is slower than in the simulator.** Probe `fatleave` (swordsmen, spearmen, greatswords
  withdraw from held clanrats 10 s after contact): the about-face (150°) takes 1.5 s — as in the simulator; but the
  centre gets 1.5 / 4.6 / 9.3 m away in 2 / 4 / 6 s (simulator 3.7 / 9.7 / 15.7), the leaver loses health until
  3.0–5.5 s (simulator 1.5–2.5), the enemy's melee flag goes off at 5.5–6 s (simulator 1.5–2.5), and the leaver still
  strikes for 1–1.5 s after the order (the enemy loses 23–88 HP; simulator 0). A free about-face to a run (no enemy)
  covers 2.6 m in 2 s in the game, so in contact a unit gets out about half as fast. Per 0.5 s (`build/open2/leave_trace.py`,
  with the earlier unchased swordsmen of `build/movelords` 1.6 / 6.2 / 10.2 m): the centre moves 2–6 s after the order
  stop-and-go (stands 0.5–1.5 s, then 3 m/s), from ~6 s at the run; the leaver's own melee flag stays on 7.5–10 s
  (about `melee_breakoff_total_immunity_secs` 10). The simulator reaches the run 1.5 s after the order even free (the
  game's turning probe: 2.0 s). Not found: no CA or community rule for the exit (the internet: players' claims only —
  men finish the animation in progress, a unit "drags along one or two models", a ~5 s delay); the database has no exit
  speed (`break_off_*` 0, `melee_attack_interval` 4, `matched_combat_*`). The simulator holds a leaver without a missile
  weapon 2 s (`contact.pin_melee_s` 2, source: this probe, `build/probes7/leavefat.py`, `build/open2/leave_trace.py`):
  the twin 0 / 3.4 / 9.3 m at 2 / 4 / 6 s (game 1.5 / 4.6 / 9.3), the leaver loses health until 3.5–4.5 s (game
  3.0–5.5), 60 HP in 10 s (game 47); without the hold 3.4 / 9.3 / 15.2 m and until 1.5–2.5 s; with 1.5 s 0 / 4.9 /
  10.8 m, with 2.5 s 0 / 2.0 / 7.8 m; 2 s is the nearest to the trajectory, not fitted to an outcome. On since 08.10:
  network v2 used the cheap exit (4–6 exits a unit-minute in melee). The overall check did not move with it (game-AI
  battles 39 % / −41 → −39 %, network battles 40 % / +24 %): it is not what explains the network battles' gap. In the
  game the centre moves in jerks from 2 to 6 s — the hold is only the simulator's shape of it; the game's leaver
  still strikes 1–1.5 s (the simulator's does not). OPEN: the exit's rule.
- **In whole battles the melee rate falls with the contact's age more steeply than in the simulator**
  (`build/midfight`, 170 network battles, 4 copies, `diag.py` / `agg*.py`). HP/s of a melee unit under one enemy in
  melee by time since the contact began (0–5 / 5–15 / 15–30 / 30–60 / 60–120 / 120+ s): struck by the game's AI —
  game 39 / 35 / 27 / 24 / 24 / 20, simulator 31 / 25 / 23 / 23 / 25 / 22; struck by the network — game 24 / 23 /
  20 / 17 / 14 / 13, simulator 27 / 21 / 20 / 19 / 19 / 17. So in the game the rate after 60 s of contact is ~0.55
  of the first 5 s, in the simulator ~0.7: early the simulator is weaker than the game, late stronger (this is the
  network battles' excess at 60–180 s: the enemy's health in melee x1.08–1.16, out of melee x1.2–1.4, most of it
  routers, below). What does not explain it: (1) reach — in the probe (`build/midfight/near_age.py`, plan `hit` and
  P2) 23–26 men are within 2.5 m of an enemy in the first 5 s, 13 by 15–20 s and 12–14 on to 90 s; (2) men lost —
  the game's formation keeps its front while men die (slaves 180 → 63: width 26.8 → 23.1 m, depth 16.7 → 4.1 m;
  `width.py`), and in the probe's 90 s lanes a unit down to 58 % of its men strikes no slower (flagellants →
  clanrats 48 → 56 HP/s; `lane_rate.py`); (3) fatigue — for the game's AI's strikers at one fatigue level the
  simulator matches (x0.98–1.01 at levels 3–5), for the network's it does not (x1.1–1.46). In the 121 recorded
  pairs (isolated pairs, `age_pairs.py`) the residual to the rule depends most on the striker's share of men (0.69
  / 0.84 / 0.94 / 1.15 at 20–40 / 40–60 / 60–80 / 80–100 %), but the probe rules this out for fresh units — so it
  travels with long fights (fatigue, morale, orders) and is not a rule. No rule found (online: nothing on a
  miss's time or the tempo after a miss; `unit_fatigue_effects` do not touch the attack speed) — OPEN.
- **A lord against one infantry unit hits less often in the game than against a crowd.** At 2.07 men hit
  (measured on 1–4 units around him) a lord in a pair takes 13–31 % more off infantry than in the game.

## Tried and rejected

| What | Result | Why not |
|---|---|---|
| Attack − defence at weight 0.1 in formations (before 06.10.2026) | held the four pairs (51 of 54 within 20 %), but every addition to attack and defence — abilities, fatigue, charge, flanks — worked ten times weaker | not the game's rule; the flat pairs are explained by the cost of a miss and overkill |
| A fixed interval, p / interval, at weight 1 in formations | worse on 121 battles than p / (p × interval + 0.5) (`build/hitchance` fit2) | a miss in the game costs less than the interval |
| The exact overkill step (whole blows per man) | loss-rate error 0.174 against 0.146 without overkill and 0.144 smoothed; worse across all 200 resamples | the step is blurred in formations (candidates: height between models, blows on the wounded) |
| The formation step from `build/mass/spec.md` (the Empire 1.65 × 1.75) | the Empire's roster depth ×0.77–0.92 | these are the numbers from the table's neighbouring row: in the `unit_spacings` row nine numbers come before the key |
| A holding unit strikes in full with its men in contact (no `hold_rate`; the reading of `build/melee2/spec.md` 3) | the probe's holding units strike twice the game: clanrats lose 23 HP/s on braced spearmen (game 12), 20 on spearmen facing away (10), 35 on swordsmen (17) | in the game a holding formation strikes at 0.49–0.52 of the rule; only a lord without an order strikes in full |
| No charge sprint (`charge.rush_m` 0, before 06.10.2026) | an attack at a walk walked in without a charge (contact at 63 s, game 48; burst 35 HP, game 248) | whole battles show no sprint (0.90–0.98 of the run, 3,729 + 2,801 approaches), but their orders are mixed; the probe's attack orders on one target show it |

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
[pcgcav]: https://www.pcgamesn.com/total-war-warhammer-2/cavalry-overhaul-beta
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[splash]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
