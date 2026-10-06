# Indicator registry of our units

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Units](README.md) › Indicator registry · [Русский](../../../ru/game/units/indicators.md)

Every indicator of our 17 units that the simulator must count the way the game does: card numbers, the
mechanics that depend on them, attributes, innate effects, abilities, and lord-only rules — with the
game's rule, its source, the simulator's status, and the in-game check.

**Rule:** we match the game only for what our units actually have. **A new unit adds its own unique
indicators here** (section 7) — before the network trains on it.

Sources: `config/nn/units.json`, `effects.json`, `abilities.json`, the game's database v9.0.1; detailed
rows — `build/conform/master.md` (the H, M, L, F, R, V, A, C numbers come from there), per-unit values —
`build/conform/ind_values.txt`; mechanics breakdowns — `build/hitchance`, `build/damage/spec.md`,
`build/charge/spec.md`, `build/mass/spec.md`, `build/accuracy`, `build/shields`, `build/morale_spec`.

**Units.** Empire: **Gen** — General; **Spr** — Spearmen; **SprS** — Spearmen with shields; **Hal** —
Halberdiers; **Swd** — Swordsmen; **Flg** — Flagellants; **GSw** — Greatswords; **Mil** — Free Company
Militia (pistols); **Arc** — Archers. Skaven: **War** — Warlord; **CSpr** — Clanrat spearmen; **CShd** —
Clanrats with shields; **Slv** — Slaves; **SlvS** — Slave spearmen; **Sln** — Slave slingers; **Stm** —
Stormvermin; **NR** — Night Runners.

**Status in the simulator:** MODELLED — done by the game's rule and numbers; FITTED — a fitted number
instead of the game's rule; DIFFERS — there is a rule, but a different one; MISSING — none.
**Research:** the analysis folder where the indicator is broken down.
**In-game test:** done (by which experiment) / planned (the test's number in `master.md`) / none.
**Priority** — how much the indicator changes the outcome of our battles: high / medium / low / none
(does not act in our battles).

**Score** (by each row's first status, 06.10.2026): 40 card numbers (MODELLED 24, FITTED 5, MISSING 9,
DIFFERS 2); 17 mechanics (MODELLED 6, FITTED 8, MISSING 1, other 2); 21 attributes, effects and abilities
(MODELLED 12, MISSING 4, DIFFERS 2, FITTED 3); 16 lord-only rules (FITTED 9, MODELLED 4, MISSING 2, other
1). Our units have no special damage types (magic, fire, poison), ward save, or resistance to magic and
fire (section 4).

## 1. Card numbers

| # | Indicator | Which units (values) | What it does in the game (source) | Simulator: status, where | Research | In-game test | Priority |
|---|---|---|---|---|---|---|---|
| K1 | Men per unit | 1 — Gen, War; 90 — Arc; 120 — Spr, SprS, Hal, Swd, Flg, GSw, Mil, NR; 140 — Sln; 160 — CSpr, CShd, Stm; 180 — Slv, SlvS | the number of models (`main_units.num_men`) | MODELLED (`state.py` men0) | — | done: the battle card of 9 units (`NumEntitiesInitial`) | high |
| K2 | A man's health | 4068 — lords; 76 GSw, 73 Hal/Flg, 69 Spr/SprS/Swd/Arc, 62 Stm, 61 Mil, 60 CSpr/CShd, 56 NR, 50 Slv/SlvS, 48 Sln | each model's own health (`battle_entities.hit_points` + `land_units.bonus_hit_points`) | MODELLED as a sum; kills from health lost — FITTED (mechanic B5) | `build/damage` | done (card `HealthMax`) | high |
| K3 | Melee attack | 55 Gen, 50 War, 32 Swd/Flg/GSw/Stm, 28 Mil, 26 Hal, 24 CShd/NR, 20 Spr/SprS, 18 CSpr, 14 Arc, 12 Slv, 9 SlvS, 6 Sln | hit chance 35 + attack − defence, 8–90 % (`_kv_rules melee_hit_chance_*`; H1) | MODELLED: 35 + attack − defence at weight 1 for every blow (`melee.py` `strikes`; the fitted weight 0.1 removed 06.10.2026) | `build/hitchance` | done: lord duels (the General hits 41 % = the database's 40 %); planned T1 (Stand Your Ground on/off) | high |
| K4 | Melee defence | 55 War, 45 Gen, 42 Hal/SprS, 34 Spr/Stm, 32 Swd, 30 GSw, 25 Mil, 24 CSpr, 22 CShd, 17 Arc, 16 SlvS, 14 NR, 12 Flg, 10 Slv, 7 Sln | the same rule; ×0.6 from the flank, ×0.3 from the rear (`melee_defence_direction_penalty_coefficient_*`) | FITTED (H1; flank/rear — mechanic B8) | `build/hitchance` | as K3 | high |
| K5 | Weapon damage, base | 290 Gen, 280 War, 25 Flg, 21 Swd/Mil/Arc, 20 CShd/NR, 19 Spr/SprS, 18 CSpr, 14 Slv/Sln, 12 SlvS, 10 GSw, 9 Stm, 8 Hal | stopped by armour 50–100 % (`armour_roll_lower_cap` 0.5; CA damage blog) | MODELLED: mean 75 % (`melee.py:45`, M15) | `build/damage` | done: a General-on-General blow is 245 HP = the passport | high |
| K6 | Weapon damage, AP | 140 Gen, 120 War, 25 GSw, 23 Stm, 20 Hal, 8 Flg, 7 Swd/Mil, 6 Spr/SprS/CShd/NR, 5 CSpr, 4 Slv/SlvS/Sln, 3 Arc | ignores armour (CA damage blog) | MODELLED (M15) | `build/damage` | as K5 | high |
| K7 | Bonus vs large | 17 Stm, 16 Hal, 15 Spr/SprS, 9 CSpr, 6 SlvS | + to attack and damage against large models (`melee_weapons.bonus_v_large`) | MODELLED, but never fires: no large models, lords are "small" (M37) | — | none | none (becomes high once we have cavalry and monsters) |
| K8 | Bonus vs infantry | 14 GSw | + to attack and damage against infantry (`bonus_v_infantry`) | MODELLED (M37); whether it hits a lord on foot is not established (ours does) | — | none | low |
| K9 | Charge bonus | 40 Gen, 35 War, 28 Flg, 18 GSw, 14 Swd/Mil, 10 Stm/CShd/NR, 8 Hal, 6 CSpr, 5 Slv, 4 Spr/SprS/Arc/Sln, 3 SlvS | + to attack and damage on the first blow, linear to 0 over 13 s (`charge_decay_duration`); only an attack order with a run-up (`build/charge/spec.md`) | MODELLED: + to attack (weight 1) and to damage (at the AP share), given in full on the first blow, to 0 over 13 s on its own clock; needs an attack order and a run-up ≥ 10 m at ≥ 0.75 of run (`sim.json charge`) | `build/charge` | done: `charge-probe` 29.09, 5,481 contacts in `build/charge` | high |
| K10 | Attack interval | 4.0 lords; 4.2–4.6 most; 5.6 Hal; 5.7 Spr | time between a model's blows (`melee_weapons.melee_attack_interval`) | MODELLED (M16) | — | done: a lord strikes once every 4 s | medium |
| K11 | Splash (a blow hitting several) | lords: up to 4 targets of size up to "medium"; Flg: knockback power 0.2 | a blow hits several models, damage divided (`splash_attack_*`) | MODELLED: a blow is divided by 4 (the cap), hits an average of 2.07 men (`contact.lord_splash_struck`, measured on 506 blows), each gets ¼ of the blow, then armour and overkill; a single entity takes the whole blow | `build/damage` | done: the General's blow on 4 targets | medium |
| K12 | Armour | 95 GSw, 90 War/Stm, 85 Gen, 30 Spr/SprS/Hal/Swd, 25 CSpr/CShd/Mil, 20 Arc, 10 NR, 0 Slv/SlvS/Sln/Flg | stops 50–100 % of base damage, both in melee and from projectiles (`unit_armour_types`) | MODELLED: the mean of the roll (0.75 of armour up to 100, above — 2 − 100/A − A/400), then overkill (`melee.per_hit`) | `build/damage` | done (lord duels) | high |
| K13 | Shield (projectile block) | 55 Gen; 35 War, SprS, Swd, CShd | blocks a projectile from the front within 60° (`shield_defence_angle_missile`); nothing in melee in WH3 (`shield_defence_value` 0) | MODELLED (R19, M50) | `build/shields` | planned (test of 7 cards, T6) | medium |
| K14 | Leadership | 100 Flg, 75 GSw, 70 Gen/Hal/Stm, 60 War/Spr/SprS/Swd, 55 Mil, 50 Arc, 48 NR, 45 CSpr/CShd, 35 Slv/SlvS, 33 Sln | the morale base (`land_units.morale`) | MODELLED (L11) | `build/morale_spec` | done (`MoralePercent` in the recordings) | high |
| K15 | Physical resistance | 20 % NR | less damage from any non-magical source, summed ≤ 90 % (CA damage blog) | MODELLED in melee and from projectiles (M38, R20) | `build/damage` | none | low |
| K16 | Missile resistance | 15 % Gen, War | less damage from projectiles (`damage_mod_missile`) | MODELLED (R20) | `build/damage` | done (`lord-missile`) | medium |
| K17 | Walk speed | 1.5 m/s for everyone | `battle_entities.walk_speed` | MODELLED (V6) | — | done (`pace.md`) | medium |
| K18 | Run speed | 5.4 NR, 4.2 Skaven infantry, 4.0 War, 3.8 Stm, 3.6 Flg/Mil, 3.4 Gen, 3.3 Arc, 3.0 Spr/SprS/Hal/Swd, 2.8 GSw | `battle_entities.run_speed` | MODELLED (V6) | — | done (`pace.md`) | high |
| K19 | Charge speed | 6.0 NR, 4.8 Skaven infantry, 4.7 War, 4.5 Stm, 4.1 Gen, 4.0 Arc/Flg/Mil, 3.8 Spr/SprS/Hal/Swd, 3.5 GSw | the last 25–30 m of an attack (30–35 m for lords) at this speed (`charge_speed`, `charge_distance_*`) | MODELLED (melee core 2): under any attack order, at a run or a walk, the last `charge_distance` of the passport (`battle_entities` `charge_distance_commence_run`: 30 m, lords 35) at `charge_speed` - a charge; a move order gives no sprint. Sim. swordsmen over the last 30 / 10 m 3.57 / 3.19 m/s (game 3.65 / 3.89) - OPEN: the last 10 m slower | `build/charge` | done (melee probe: an attack order at a run and at a walk - the last ~30 m at charge speed, 3.6-3.9 m/s on average against a run of 3.0; a move order - no) | medium |
| K20 | Acceleration and deceleration | 2.0–5.0 / 4.0 m/s² | `battle_entities.acceleration` / `deceleration` | MODELLED (V7) | — | none | low |
| K21 | Mass | 650 War, 600 Gen, 150 Stm, 120 GSw, 100 Spr/SprS/Hal/Swd/Flg/CSpr/CShd, 90 Arc/Mil/Slv/SlvS/Sln/NR | knockback and push by the mass ratio, bracing, collision damage (`battle_entities.mass`; patch 1.12.1) | MISSING (`melee.impact` removed: small models get no collision damage; C7, M18, M39, M40) | `build/mass` | none | medium for lords, low for infantry on infantry |
| K22 | Size, height, model radius | all "small"; height 1.3–2.2 m; radius 0.55–0.7 m | the size class decides the bonus vs large and splash; height and radius are a projectile's target | MISSING: height and radius for hits (part of R2); size MODELLED | `build/accuracy` | none | low |
| K23 | Formation step | 1.48 × 1.6 m Gen/Spr/SprS/Hal/Swd/War; 1.6 × 1.7 CSpr/CShd/Flg/Mil; 1.6 × 1.8 Slv/SlvS; 1.7 × 1.75 GSw; 1.8 × 1.9 Stm; 2.0 × 2.1 Arc; 2.2 × 2.8 Sln/NR | the spacing between men across the front and in depth (`land_units.spacing` → `unit_spacings`) | MODELLED: the template step from the database (`units.json spacing`, `unit_spacings`); ranks = floor(width / h), depth = ceil(men / ranks) × v (`geometry.dims`); those striking = contact length / step (`melee.py`) | no dedicated one (`build/accuracy/spacing.py` looks at density) | planned (test of 3 cards: men's coordinates on deployment) | high |
| K24 | Default ranks | 1 lords, 4 Arc, 5 Sln/NR, 6 Empire infantry, 8 Skaven infantry | `land_units.rank_depth` | MODELLED (V19) | — | none | low |
| K25 | Model turn rate | 240 °/s Mil; 180 Arc, War; 120 Gen and infantry | `battle_entities` (column f16, named per patch 6.1) | FITTED: a formation 80 °/s, a lord 40 °/s (V2, V3) | — | done: a formation turns 90° in 7–8 s (`commands.md`); a lord — planned T9 | medium |
| K26 | Resistance to hit reactions and interrupts | Gen 70 / 40 %, War 75 / 75 %; infantry 0 | a model interrupts its own blow less often (`hit_reactions_ignore_chance`, `knock_interrupts_ignore_chance`) | MISSING (C8) | `build/mass` | none | low–medium |
| K27 | Cost and combat potential | cost: 950 Stm, 850 GSw, 600 Gen/Flg, 550 Hal, 525 War, 450 Mil/NR, 375 Swd, 350 Arc/SprS/CShd, 325 CSpr, 300 Spr, 200 Sln, 150 SlvS, 125 Slv; lords' potential with abilities 950 / 900 | a unit's "strength" for army collapse and "a strong enemy" (`main_units.melee_cp`, `missile_cp` + abilities' `additional_*_cp`) | FITTED: cost × health, a lord ×1.65, an empty quiver 0.32 / 0.57 (C15, L2) | `build/morale_spec` | planned (test of 4 cards: `strategic_value` in the recordings) | medium |
| K28 | Ammunition | 22 Sln/NR, 20 Arc, 18 Mil | shots per man (`land_units.primary_ammo`) | MODELLED (R15) | — | done: the recordings show the stock does not change as men die | medium |
| K29 | Projectile damage | 17 + 2 Arc; 12 + 2 Mil; 10 + 1 NR; 7 + 1 Sln | the same armour roll (`projectiles.damage`, `ap_damage`) | MODELLED (R18) | `build/damage` | done (`archer-range`) | high |
| K30 | Range | 140 NR, 130 Arc, 120 Sln, 90 Mil | the first shot once the shooters' middle is at range from the target's near edge (`projectiles.effective_range`; `missile-range.md`) | DIFFERS: edge to edge, closing to 0.95 of range (R7) | `build/accuracy` | done (`missile-range.md`) | medium |
| K31 | Reload | 10 s Arc, 9 s Sln/Mil, 8 s NR | `projectiles.base_reload_time` + the animation | FITTED: 11.0 / 11.5 s measured; NR 10.2 and Mil 10.8 — an estimate (R3) | — | done: Arc, Sln (`archer-range`); NR, Mil — planned (test of 6 cards) | medium |
| K32 | Accuracy and spread | accuracy 10 for everyone; spread 95 m / 3.7 Arc, 85 / 4.8 Sln, 85 / 4.0 NR, 65 / 2.0 Mil; projectile accuracy 15 NR, 10 the rest | the hit share by range (`land_units.accuracy`, `projectiles.calibration_*`, `marksmanship_bonus`) | FITTED: a share by projectile type + a range table (H2) | `build/accuracy` | done (the range, recordings by target) | high |
| K33 | Trajectory | arcing over friends — Arc, Sln, NR; flat, needs a clear line — Mil | `projectiles.trajectory` | MODELLED (R21, R22) | — | none | low–medium |
| K34 | Penetration | very low Arc/Sln/NR, low Mil | a projectile can pass beyond the first model (`projectiles.penetration`) | MISSING (R29) | `build/accuracy` | none | low |
| K35 | Projectile speed | 90 m/s Mil, 50 Sln/NR, 45 Arc | flight time (~3 s at 130 m) | MISSING (R23; the recordings show no penalty against a moving target) | — | done (recordings: 1.07) | low |
| K36 | Fire arc without turning | ±30° shooters, ±35° Mil | a standing shooter fires within this sector (`battle_entities` f15) | DIFFERS: ±45° standing, ±90° on the move (R16, R17) | — | planned T10 | low–medium |
| K37 | Bracing (`can_brace`) | everyone | standing infantry braces against a charge (a mass multiplier, `bracing_*`) | MISSING: bracing only for units with charge reflection (C19, M30) | `build/mass` | none | low |
| K38 | Skirmisher mode | Arc, Sln, Mil, NR | pulls back from melee by itself | MISSING: the bridge switches it off for the network; not checked for the game's AI | — | none | low |
| K39 | Projectiles per shot, minimum range, explosion | 1 / 0 / none for everyone | — | MODELLED (one projectile) | — | — | none |
| K40 | Stealth (`hiding_scalar`) | 0.8–1.0 | hiding in cover | MISSING: the simulator has no forest or hiding | — | — | none |

## 2. Mechanics that depend on these numbers

| # | Mechanic | From which numbers | What the game does (source) | Simulator: status, where | Research | In-game test | Priority |
|---|---|---|---|---|---|---|---|
| B1 | Hit chance | K3, K4 | 35 + attack − defence, 8–90 % (H1) | MODELLED: weight 1 for everyone; hits a second for a man p / (p × interval + 0.5) (`melee.miss_s`, fitted on 121 battles, `build/hitchance`); lord on lord — p / interval | `build/hitchance` | done (lord duels; melee probe, `hit` plan: the steady pace by the rule within ±25 % with 10 men striking; Stand Your Ground x0.24 against the rule's x0.62 - OPEN) | high |
| B2 | How many men strike | K1, K23 | every model with an enemy within 2.5 m of its weapon strikes (`entity_action_attack_formed_combat_distance`) | FITTED: 0.5 of the files in contact (`melee.fighting_files`); a file is the contact length / the database's formation step | `build/hitchance` | none (no rule by number in the database) | high |
| B3 | Bringing men into a fight without a charge | K1 | no rule; a model strikes as soon as it reaches | MODELLED: no ramp, it strikes at once (`melee.ramp_s` removed); **the first strike** (melee core 2, the rule: the interval runs only after a blow) - a unit coming into a fight moving or charging strikes once at once with every man in contact; the men around a single entity gather over 20 s (`contact.lord_gather_s`, FITTED) only when the lord himself ran into the formation | `build/charge`, `build/meleetests` | done (melee probe: the first second swordsmen → clanrats 220 / sim. 189, was 51; OPEN - a recharge, a holding target, the General) | high |
| B4 | Damage per blow | K5, K6, K12, K15, K16 | AP + base × (1 − the 50–100 % armour roll) | MODELLED: the mean armour roll, overkill smoothed (`melee.per_hit`, `build/damage/spec.md` D3) | `build/damage` | done (lord swarm probe, `damage` plan: a greatswords blow on the War 41 HP - the bonus against infantry counts against a lord on foot) | high |
| B5 | Kills from health lost | K2 | each model has its own health, overkill is lost | FITTED: `kills.exponent` 0.5 (M3); overkill MODELLED (smoothed: the first blow exact, then a mean) | `build/damage` | none | high |
| B6 | Charge | K9, K18, K19 | the charge bonus fading over 13 s, only with an attack order and a run-up; nothing more for infantry and lords on foot (`build/charge`) | MODELLED: the bonus (weight 1), an attack order + a run-up, its own 13 s clock, reflection ×2; `melee.impact` and `melee.ramp_s` removed; the charge sprint (K19) and the first strike (B3) - melee core 2, an attack at a walk is a charge too | `build/charge` | done (`charge-probe`, 5,481 contacts; melee probe: the volley in the first second of contact - now 107-429 HP in the simulator against the game's 78-300) | high |
| B7 | Flank and rear | K4 | defence ×0.6 / ×0.3 for each model, sector by quadrant | MODELLED: defence ×0.6 / ×0.3 at weight 1 (`flank_slope` 2.0 / `rear_slope` 0.25 removed); the sector by the unit's centre DIFFERS (M7) | `build/hitchance` | none (a lone attacker measured in the recordings: 1.53× / 1.92×) | high |
| B8 | Standing without an attack order | K3 | keys `melee_attack_threshold_modifier_idle_default` 0.14 / `_ordered` 0.22 (meaning undocumented) | FITTED: a formation 0.5 (`contact.hold_rate`, M11; the keys 0.14 / 0.22 are the threshold for joining a fight, not a rate, `build/melee2/spec.md` 3); a lord under hold - in full (melee core 2) | `build/hitchance`, `build/meleetests` | melee probe: holding formations strike at 0.49-0.52 of the rule (6 lanes), a lord without an order at the rule (a unit loses 18-29 HP/s to him, sim. was 8-14, now 16-29); OPEN: a unit under a move order in contact x0.6, the holding unit's first second | high |
| B9 | Pursuit | K1, K9 | by the database's rule every blow on a router is a charge; in fact CA bug 5068: pursuers do not hit while running | FITTED: 0.43 (`contact.pursuit_rate`, M12) | `build/charge` | done (recordings: 0.64 of a standing target) | high |
| B10 | Leaving melee | K1 | `melee_breakoff_secs` 24 (meaning undocumented); measured in the game: median 21 s | FITTED: 20 s, damage on a leaving unit ×1.25 / ×0.55 (M13, M14) | — | done (163 recordings) | medium |
| B11 | Mass: knockback, push, bracing | K21, K26, K37 | knockback by the mass ratio, ×2 on a charge; bracing is a mass multiplier | MISSING (M18, M40) | `build/mass` | none | medium (lords) |
| B12 | Shooting: hits | K32, K22, K23 | a physical projectile: a denser and bigger target catches more | FITTED (H2); target density MISSING (R2) — `build/accuracy/summ.txt`: arrows hit slingers 0.41, spearmen 0.62 | `build/accuracy` | done (the range, recordings) | high |
| B13 | Shooting: pace and range | K28, K30, K31, K36 | reload + the animation, volleys, range from the middle, into melee once out of ammunition | reload, volleys, aiming FITTED (R3–R6); range DIFFERS (R7); out of ammunition MISSING (R14); the arc DIFFERS (R16) | `build/accuracy` | done (the range); T5, T10, T13 planned | medium |
| B14 | Friendly fire and spill | K33 | a projectile hits whoever it hits | FITTED: shares for friends and neighbours (R11–R13) | `build/accuracy` | done (recordings) | medium |
| B15 | Morale | K14, K27, K1, K2 | leadership + modifiers: casualties (4 s and 60 s), under fire 15 s, winning/losing the melee, a charge +15, flank/rear, exposed flanks, friends routing nearby, a strong enemy, army collapse −120, rally (L1–L53) | MODELLED: the base, casualties, under fire, exposed flanks, friends routing nearby, collapse (thresholds), rout, shatter; FITTED: winning/losing (thresholds), a strong enemy −3, army strength, rally; DIFFERS: winning/losing only in melee (L4), flank/rear 0.5 s instead of ~5 s (L16); MISSING: a charge's +15 (L9) | `build/morale_spec`, `build/charge` (`charge_morale.txt`: +2.5–3.2 points at contact on a running charge) | done (27.09 probes, `lord-fall`, morale labels in 217 recordings) | high |
| B16 | Fatigue | K18, K31 (level multipliers on attack, defence, armour, charge, speed, reload) | a 0.1 s tick, thresholds and penalties from `unit_fatigue_effects`; melee +19 a tick | the tick, thresholds, penalties MODELLED (F1–F3); melee 13.7 / a lord 15, shooting, rest FITTED (F4, F5, F7, F8); a charge +34 only 2 s after the charge blow MODELLED (F6, 06.10.2026); reload DIFFERS (F12) | — | done (a lord goes "tired" after 60–62 s); T13 planned | medium |
| B17 | Turning and formation movement | K17, K18, K25 | a model's turn rate from the database, re-forming 7–17 s per 90°, not instant on the move | FITTED (V2, V3); on the move DIFFERS (V4); own-unit stacking DIFFERS (V10); routers' path DIFFERS (V13) | — | done (formation), T9, T14, T16 planned | medium |

Experience (ranks) has no effect: everyone is rank 0, a level-1 lord with no skills or items
(`scenario.py:88, 142`; X1–X6).

## 3. Attributes, innate effects, abilities

| # | Indicator | Which units (numbers) | What the game does (source) | Simulator: status, where | Research | In-game test | Priority |
|---|---|---|---|---|---|---|---|
| E1 | Encourage (`encourages`) | Gen, War | an aura +4 morale to 70 m, fading to 0 at 105 m, no effect on himself (`_kv_morale general_aura_*`) | MODELLED (L13, L14, A20) | `build/morale_spec` | done (a lord on himself: 1.129 / 1.083 = without the aura) | medium |
| E2 | Expendable (`expendable`) | Slv, SlvS, Sln | its rout scares only other expendable units (−3) | MODELLED (L28, A19) | `build/morale_spec` | done (recordings: −3.0 / 0.0) | medium |
| E3 | Unbreakable (`unbreakable`) | Flg | never routs | MODELLED (A18, L47) | — | none | medium |
| E4 | Charge Reflection (`charge_reflection`) | Spr, SprS, Hal, CSpr, Stm | while braced, ×2 weapon damage against a frontal charge for 3.9 s (`charge_reflect_*`, `bracing_*`) | MODELLED: a braced unit (slower than 0.5 m/s: `melee.braced_speed`, FITTED) deals the charger ×2 damage within ±80° while its charge is ≥ 0.7 (3.9 s); gets no charge bonus of its own | `build/charge` | none | medium |
| E5 | Charge Defence vs Large (`charge_defense_vs_large`) | the same 5 | cancels a large unit's charge from the front | MISSING — no large units (A23) | — | — | none |
| E6 | Fire Whilst Moving (`mounted_fire_move`) | Mil | shoots while moving | MODELLED; the on-the-move arc ±90° FITTED against the database's ±35° (R17) | — | none | low |
| E7 | Vanguard Deployment (`guerrilla_deploy`) | Mil, NR | deploys outside its own zone | MISSING — the scenario does the deploying (A25) | — | — | none |
| E8 | Hide (forest) (`hide_forest`) | all 17 | hidden in a forest | MISSING — no forest (A24) | — | — | none |
| E9 | Strength in Numbers | CSpr, CShd, Slv, SlvS, Sln, Stm, NR | while health ≥ 50 % of the start: defence +8, leadership +6, speed ×0.9 | MODELLED (A14); the +8 defence weight — through B1 (FITTED) | `build/hitchance` | done (the Skaven start +6 points) | high (through B1) |
| E10 | Scurry Away! | all 8 Skaven units | wavering or routing: speed ×1.1 | MODELLED (A15) | — | done (recordings: routing Skaven 0.977 of the rule) | medium |
| E11 | Frenzy | Flg | attack +10, damage and charge ×1.1, immune to psychology; switches off below ½ of leadership | MODELLED (A16) | `build/hitchance` | none | low |
| E12 | Strength of the Penitent | Flg | losing the melee: defence +14, physical resistance +15 % for 20 s, reload 3 s, initial 3 s | MODELLED: the initial recharge from the database (`unit_special_abilities.initial_recharge` → passport `initial_s`, melee core 2) | — | none | low |
| E13 | "Wounds" (single entity) | Gen, War | health < 25 %: speed ×0.9, damage ×0.8 to the battle's end, after 5 s | FITTED: off (`effects.off`; A3) — the recordings show no speed drop, damage not checked | — | planned T11 (`ActiveEffectList`) | medium |
| E14 | Hold the Line! (passive) | Gen: himself and friends within 35 m | defence +5, leadership +4 | MODELLED (A9); the weight — through B1 | `build/hitchance` | done (the General hits 41 % with +5) | medium |
| E15 | Foe-Seeker | Gen: 25 s, recharge 60 s | speed and charge speed ×1.25; vigour +1 % a second (`special_ability_phases`, field −0.01) | MISSING: vigour (A2); speed and charge speed MODELLED (K19, melee core 2) | — | planned T4 (`FatigueState`) | high (the General) |
| E16 | Stand Your Ground | Gen: himself and friends within 35 m, 18 s, recharge 90 s | defence +24, leadership +16 | FITTED: the +24 defence weight through B1 (−2.4 points of hit chance instead of −24); the numbers MODELLED (A8) | `build/hitchance` | planned T1 (on / off) | high |
| E17 | Rally | War: himself and friends within 35 m, 14 s, recharge 60 s | leadership +16; by its description it helps routers rally | DIFFERS: for routers (A4); the numbers MODELLED (A10) | `build/morale_spec` | planned T3 / T16 | medium |
| E18 | Verminous Valour | War: 17 s, recharge 60 s | speed ×1.25, leadership +8 | MODELLED (A12) | — | none | low |
| E19 | Deadly Onslaught | War: 31 s, recharge 90 s | charge ×1.6, damage ×1.25 | MODELLED (A13); the game's AI never fires it | `build/charge` | none | low |
| E20 | How effects stack | everyone | additions to the base, multipliers multiply; one ability from two sources does not stack | MODELLED (A28, A29); two identical ones — DIFFERS, but our battles have only one lord each (A27) | — | none | none |
| E21 | When the game's AI fires abilities | Gen, War | AI behaviour, not a rule (AI templates in the database) | FITTED on 926 uses (`abilities.triggers`, A5) | — | done (139 gate battles) | medium |

## 4. What our units do not have (do not add until a unit with it arrives)

Magic attacks (`is_magical` false for everyone), fire (`ignition_amount` 0), poison and other
contact-phase effects (no `contact_phase`), a ward save (`damage_mod_all` 0), resistance to magic and fire
(0), fear and terror, regeneration, flight, large and huge models (all "small"), cavalry and chariots,
artillery and explosions, several projectiles per shot, a minimum range, magic and the Winds, experience
and ranks, the lord's skills and items.

## 5. Lords only (Gen, War)

| # | Rule | What the game does | Simulator: status, where | Research | In-game test | Priority |
|---|---|---|---|---|---|---|
| LD1 | How many men strike a lord | ~9 small models around one (WH1 guide) | FITTED: 6.5 for everyone (`contact.lord_max_attackers`, fitted on the pairs at the database's hit chance) | — | done (`lord-swarm`) | high |
| LD2 | A lord strikes a lord | 35 + attack − defence (database) | MODELLED: the database's hit chance, one blow per interval (p / 4 s), the whole blow without overkill | `build/hitchance` | done (lord duels) | high |
| LD3 | A blow on several | up to 4 targets (K11) | MODELLED: divided by 4, hits an average of 2.07 (measured; K11) | `build/damage` | done | medium |
| LD4 | Bringing men to a lord | strikes from the first blow once every 4 s | MODELLED: a lord strikes from the first blow; the men around him gather over 20 s (`contact.lord_gather_s`, FITTED) only when he himself ran into the formation; infantry running onto a standing lord strikes in full at once (melee core 2) | `build/charge`, `build/lord-swarm` | done (lord swarm probe, the first 15 s: game 4.9-9.7 HP/s, sim. was 2.6-3.0, now 7.1-8.8) | high |
| LD5 | The side a lord is struck from | a rule for every model | FITTED: the side does not matter (`contact.lord_direction` 0, M22) | — | done (`lord-swarm`) | medium |
| LD6 | An enemy lord and infantry on a lord; incidental contact | no rule | FITTED (`lord_rival_others` 0.35, `lord_incidental` 0.4; M20, M21) | — | done (recordings, noisy) | medium |
| LD7 | A lord in melee: morale | a single entity "loses" −3 | FITTED (`morale.single_combat`, L15) | `build/morale_spec` | done (recordings) | medium |
| LD8 | A lord's death and rout | −16 for 45 s, then −10; the lord left / is routing | MODELLED (L21–L23) | `build/morale_spec` | done (`lord-fall`) | high |
| LD9 | Shooting at a lord | a projectile hits whichever model it touches | FITTED: 0.43 out of melee, 1.0 in it (R9, R10) | `build/accuracy` | done (`lord-missile`) | medium |
| LD10 | A lord's fatigue in melee | +19 a tick | FITTED: 15 (`fatigue.calibration.single_combat`, F5) | — | done ("tired" after 60–62 s) | medium |
| LD11 | A lord's turn rate | 120 °/s Gen, 180 War (K25) | FITTED: 40 °/s (V3) | — | planned T9 | medium |
| LD12 | Knockback and push by a lord | mass 600–650 against 90–150 (K21) | MISSING (M18) | `build/mass` | none | medium |
| LD13 | Resistance to a blow being interrupted | 70–75 % (K26) | MISSING (C8) | `build/mass` | none | low–medium |
| LD14 | "Wounds" below 25 % | E13 | FITTED (off) | — | planned T11 | medium |
| LD15 | A lord's "strength" in army collapse | combat potential 950 / 900 | FITTED: cost ×1.65 (C15) | `build/morale_spec` | planned (test of 4 cards) | medium |
| LD16 | A lord's own abilities | E14–E19 | see section 3 | — | — | — |

## 6. MISSING or DIFFERS and present in our units — by priority

1. **Target density for arrows** (B12, MISSING) — every target; preliminary `build/accuracy` data: hits
   on loose slingers 0.41 against 0.62 on spearmen.
2. **A charge's +15 morale** (B15 / L9, MISSING) — every unit that attacks with a run-up.
3. **Winning/losing the melee from shooting** (L4, DIFFERS) — Arc, Sln, Mil, NR and their targets.
4. **Foe-Seeker: vigour** (E15, MISSING) — Gen.
5. ~~Charge speed~~ (K19) — done (melee core 2); Foe-Seeker's ×1.25 to charge speed now acts.
6. **"Attacked from the flank / rear" ~5 s instead of 0.5 s** (L16, DIFFERS) — every unit.
7. **Rally for routers** (E17, DIFFERS) — War.
8. **Range from the shooter's middle** (K30, DIFFERS) and **into melee once out of ammunition** (B13 /
   R14, MISSING) — the 4 shooters.
9. **Knockback by a lord's mass** (LD12, MISSING) and **a lord's resistance to interrupts** (LD13,
   MISSING) — Gen, War.
10. **The flank sector by quadrant** (B7 / M7, DIFFERS) — everyone.
11. **Turning on the move, own-unit stacking, routers' path** (B17, DIFFERS) — everyone.
12. **Fire arc ±30 / ±35°** (K36, DIFFERS) — the 4 shooters.
13. **Fatigue: reload of the exhausted** (F12, DIFFERS) — the 4 shooters.
14. **A wavering unit drops its order** (L40, MISSING) — everyone; check against recordings.
15. **Bracing without charge reflection** (K37, MISSING), **a formation's defence bonus** (M31,
    MISSING) — infantry; low.
16. ~~Strength of the Penitent: the initial reload~~ (E12) — done (melee core 2).
17. **Penetration and projectile flight** (K34, K35, MISSING); **the morale shock, shattering by
    casualties, the wavering timer** (L43, L34, L39, MISSING) — low: not seen in the recordings.

Fitted numbers that stayed next to these indicators (changed together with the items above): men in
contact 0.5 (B2), the cost of a miss 0.5 s (B1), gathering around a lord in 20 s and 6.5 men on a lord
(B3, LD1), kills' `kills.exponent` (B5), standing 0.5 (B8), army "strength" from cost (K27). Fitted numbers
that replaced the game's rules on 06.10.2026: the hit-chance weight 0.1, flank / rear 2.0 / 0.25, `impact`
1.5, bringing men in over 20 s, `lord_hit_slope`, a 1.5 m formation step for everyone
([simulator](../../training/simulator.md#tried-and-rejected)).

## 7. Rule: what to add to the registry when a new unit arrives

For a new unit: (1) collect its passport (`py -3.14 -m tools.nn.units --out ...`) and check it against
the database and twwstats (`build/conform/cards_compare.py`); (2) go through its card numbers, attributes,
innate effects and abilities, and add a row for every indicator not yet in the registry, or one that was
zero / "none in our battles" for us but is not zero for it. For example: size "large" and above (which
wakes up the bonus vs large, charge defence vs large, splash by size, mass and collision damage), magic
damage, fire, poison and other contact-phase effects, a ward save, resistance to magic and fire, fear and
terror, flight, regeneration, several projectiles, an explosion, a minimum range, cavalry, new abilities
and attributes. In the row: which units and their values, the game's rule with its source (a database key
or a link), the status in the simulator, the research, the in-game test, the priority. (3) A mechanic that
a new indicator pulls in becomes a row of section 2 only if at least one unit in our pools has that
indicator non-zero; indicators and mechanics come before effects. (4) Review the priority of "none in our
battles" rows that the new unit makes active. (5) For every new MISSING or DIFFERS row with priority medium
or above — a short in-game experiment before training on that unit, and one battle with a card at t = 0 so
the passport matches the game.
