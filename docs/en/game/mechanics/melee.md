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
    blow, with the charge's full bonus), then at the usual rate. A standing unit it reaches gives no burst.
    No fitted numbers.
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
    the last 30 m at 3.65–3.88, the last 10 m 3.9–4.7.
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
  - Ours: by the database — a unit with reflection that is braced (standing: slower than 0.5 m/s,
    `melee.braced_speed`) deals the charger ×2 damage within 80° of its front while the charger's charge
    factor is ≥ 0.7 (3.9 s); it gets no charge bonus of its own.
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
- OPEN: a recharge (pull back 35–40 m and attack again) gives no burst in the game: the target loses 250 HP in
  5 s after the second contact (Warlord: 204), the simulator 242 (154) before, 380 (276) now - the rule "the
  first strike in a new fight" is too much here (candidates: a pull-out shorter than `contact.reset_s`,
  pursuit).
- OPEN: a target ordered to attack in the second of contact gives no burst (the attacker loses 16–22 HP in
  1 s), one standing without an order does (clanrats lose 42–172 HP in 1 s on spearmen and swordsmen; the
  simulator 17–19 - a standing unit has no first strike); a move order into contact gives a smaller burst (78)
  and then ×0.5–0.6 of the rule.
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
strikes in full (below). OPEN: in the first second a holding unit strikes hard in the game (42–172 HP, braced
ones with reflection), the simulator 17–19.

**Hit chance (plan `hit`, from 15 s to the end).** Target HP/s, game / simulator: swordsmen → skavenslaves
46 / 39, greatswords → skavenslaves 63 / 41, spearmen with shields → skavenslaves 37 / 33, flagellants →
clanrats 53 / 39; the skavenslaves back 16 / 15, 10 / 9, 13 / 11; clanrat spearmen → spearmen 20 / 18 and back
21 / 22. Blows a second a man at 10 strikers (unarmoured skavenslaves: HP / blow damage): 0.15–0.25 against the
rule's 0.14–0.20 - the slope in the hit chance is right. Skavenslaves lose more men a second in the game:
0.89 / 0.51, 1.00 / 0.58, 0.58 / 0.42 (OPEN: candidates - `kills.exponent`, the smoothed overkill, blows on
fresh or wounded men).

**Stand Your Ground.** Spearmen lose 107 HP in the first 15 s with the ability against 440 without (x0.24,
one battle each); a hit chance of 25 % → 8 % gives x0.32, the rule p / (p × interval + 0.5) x0.62, the
simulator x0.70. OPEN: candidates - the 0.5 s cost of a miss (fitted) is small at a low hit chance, the spread
of one battle.

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

## Not found

No public CA source explains `melee_breakoff_secs` 24 / `melee_breakoff_total_immunity_secs` 10
(WH3-only keys; see [movement](movement.md)); by the melee-exit probe (`build/movelords`) 24 s is the window a
leaving unit keeps its order: chased swordsmen and spearmen strike nothing for 24–26 s after a withdraw order
(kills flat, the chasing clanrats lose no health), then fight on (the clanrats lose ~20 HP/s); not chased, a
unit is out in 8 s (the simulator: `contact.breakoff`; a Steam player also writes of a ~25 s window); `entity_action_attack_formed_combat_distance`
2.5 / `_tether_distance` 2, or how many models can physically reach one target.

## What is still unclear

- **The first 15 s of contact are stronger in the game than the rule.** In the pairs the target loses
  499–764 HP in the first 15 s, while the rule (the charge bonus, no blow) gave the simulator 396–446. The
  melee probe showed where the difference is: nearly all of it in the first second of a charge's contact (a
  volley), after that the pace is the rule's; the simulator now gives the volley by the rule "the first blow at
  once" ([in-game check](#in-game-check-the-melee-probe)). Why a target ordered at the moment of contact and a
  recharge give no volley while a holding unit does - OPEN.
- **A holding unit strikes at half the rule** (0.49–0.52, the probe), a lord without an order at the rule. No
  rule for a holding formation's rate found (`contact.hold_rate` 0.5 - FITTED).
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
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[splash]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
