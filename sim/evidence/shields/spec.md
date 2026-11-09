# Shields in TW:WH3 — spec for the simulator

Metric: completeness (every aspect with a rule, a number and a source) and implementability. More = better.

## Bottom line

- A shield in WH3 does one thing: it fully stops its card percentage of **small-arms** projectiles
  (arrows, sling stones, bullets, javelins…) that come from the **front 120° cone** (60° each side of
  the soldier's facing). Side and rear: 0. Armour and resistances work only on what gets through.
- It gives **nothing in melee** by itself (the shield tables' melee defence and armour columns are 0 for all
  24 shield types; the melee benefit is already inside the card's melee defence, e.g. spearmen 34 → 42).
- The "front ×1, side ×0.5, rear ×0.1" rule is **not** for normal shields. It is the *Directional Shield*
  attribute (DB `armoured_vehicle`): Steam Tanks and the Chaos Siege Giant only. None of our units has it.
- **Our recordings confirm the database** once the measurement is done properly: shielded Empire spearmen let
  through 0.64 from the front (DB 0.65), the Warlord 0.66 (0.65), the General 0.47 (0.45). The cut is at
  60° (45–60° fully blocked, 60–75° mostly not). Shields keep blocking **in melee** and **while running**.
- The accuracy analyst's "0.93 instead of 0.65" came from comparing different situations: a unit hit from the
  front is usually **running at the shooter**, and such units take ~1.3× more damage per shot than
  units crossing or running away, shield or not. This is about motion, not shields. Comparing with the
  same unit type without a shield, in the same situation, removes it.
- **Our simulator already does it right** (`tools/nn/sim/missile.py:177-186`). There is no number to change.
  The spec gives one small safety guard (only small arms are blocked), tests, and a short in-game check
  (2 battles), in case the user wants the in-game check of this card stat as the method requires.

## Aspects

Status: MATCH = same as the game; FIT = close, simplified; DIFF = wrong; MISSING = not in the sim.

| # | Aspect | Game rule and numbers | Source | Confidence | Our sim | Status |
|---|---|---|---|---|---|---|
| S1 | Block chance | The card's shield % = chance that a projectile hitting the soldier from the front is stopped completely (0 damage). Tooltip: "will block xx% of all small-arms missile fire hitting it from the front". DB `unit_shield_types.missile_block_chance`: our General `wh_missile_block_55_metal` 55; Warlord `_35_wood` 35; spearmen with shields `wh_main_emp_inf_spearmen_1` and swordsmen `_35_metal` 35; clanrats with shields `wh2_main_skv_inf_clanrats_1` `_35_wood` 35; everyone else `none` 0 (incl. stormvermin, clanrat spearmen, plain spearmen). The whole table: 24 types, 10–90 % | [fandom Shielded](https://totalwarwarhammer.fandom.com/wiki/Shielded); DB (dump `build/shields/db_dump.py`); recordings below | high | `params.py:98` passport `missile_block_chance/100`; `missile.py:186` `hp = landed × hit × (1 − shield)` | MATCH |
| S2 | Directions | Front only, as a cone of ±60° from the facing (`_kv_rules.shield_defence_angle_missile` 60, half-angle). Side and rear 0. Recordings: the pass-through steps from 0.43–0.49 (General) and 0.63–0.67 (Warlord) at 0–60° to 0.78–0.95 at 60–75° and ~1 beyond; no fall-off inside 0–60° for lords | DB; Steam [WH2 arcs](https://steamcommunity.com/app/594570/discussions/0/2280456883446210505/) ("actually 120° in front"); Steam [WH1 shielded missile infantry](https://steamcommunity.com/app/364360/discussions/0/2650805212059950061/) ("Shields cover a 120degree arc"); `build/shields/strat.txt` | high | `missile.py:177` `|rel_j| ≤ 60°` | MATCH |
| S3 | Side ×0.5 / rear ×0.1 | Only the *Directional Shield* attribute (key `armoured_vehicle`, icon `armoured_vehicle.png`, since 5.0): 360° block, front ×1.0, sides ×0.5, rear ×0.1 (`armoured_vehicle_front/side/rear_block_modifier` 1.0 / 0.5 / 0.1). Units: Steam Tank (both), Emperor's Wrath, Chaos Siege Giant | [fandom Directional Shield](https://totalwarwarhammer.fandom.com/wiki/Directional_Shield), [fandom Update 5.0](https://totalwarwarhammer.fandom.com/wiki/Update_5.0); DB `_kv_rules` | high | — (no such unit; attributes of our 17 units checked: none has it) | MISSING, not needed |
| S4 | Which projectiles | Small arms only. "Misc ignores shields. Artillery ignores shields"; "arrow and musket are small arms, blocked by shields". Not blocked: cannonballs, rocks, spells/breath, some named heroes' arrows; *Ballistic Plating* (5.0) is the exception that extends a block to all projectiles. Our projectiles: `wh2_dlc13_emp_bow_arrow` (category arrow), `wh2_main_skv_sling`, `_night_runners` (sling), `wh_dlc04_emp_free_company_pistol_bullet` (musket); all `shot_type small_arm_default`. Recordings: arrows 0.75, slaves' slings 0.66, Night Runners' slings 0.73–0.76, pistols 0.65 pass-through (clanrats/spearmen pairs, `byproj.txt`) — all blocked | [tw-modding Missiles and You](https://tw-modding.com/wiki/Tutorial:Missiles_and_You); fandom Shielded; fandom Update 5.0 (Ballistic Plating); DB `projectiles.category` | high (ours), medium (javelin/axe/grenade) | Every projectile is blocked | MATCH for our roster; guard MISSING (change C1) |
| S5 | When in the hit is it applied | Per projectile that physically touches a soldier (there is no hit roll: a projectile hits when it collides, `docs/ru/game/mechanics/missiles.md`), before armour and resistances. A blocked one does 0 damage. The game uses position and size, not the shield animation | fandom Shielded ("Armour and resistances are only relevant… after the missile block chance has been bypassed"); Steam WH1 thread ("doesn't track where your shield is during animations") | high | Expected value: `landed × per_hit × (1 − shield)`; order block → armour → resistance | MATCH |
| S6 | Whose facing | The hit soldier's own facing (per entity). In a standing or marching formation it is the formation's facing (recordings: running units' facing within 30° of their path 89–95 %) | no direct source (engine detail); recordings | medium | Facing of the unit centre; angle shooter centre → target centre (`geometry.py:66` `rel_j`) | FIT (differences only within ~10° of the 60° edge) |
| S7 | In melee | **Still blocks.** Lords in melee, HP lost per second regressed on arrows from the front and from elsewhere (melee damage as the constant): front / rest = General 0.56 [0.42, 0.77] (out of melee 0.58), Warlord 0.72 [0.47, 0.93]; "does not block" would give 1.0. A WH2 forum post says the opposite ("shields dont work if the unit is engaged in melee"); our data reject it for WH3 | `build/shields/melee.txt` (37.5k + 26k lord-seconds in melee); Steam WH2 arcs thread (contradicted) | medium-high | Shield applies in melee (no contact condition) | MATCH |
| S8 | Running, walking, routing | Blocks at any speed (matched spearmen pair: still 1.19 (57 shots, noise), walk 0.69, run 0.55; clanrats 0.64 / 0.77 / 0.72). Routing units run away, so shots come from behind and the shield does nothing; no special rule found | recordings `strat.txt`; no web source | medium | No speed condition; a moving or routing unit faces its path (`battle.py` "a walking unit faces the way it walks", `movement.py:59`), so a fleeing unit is hit in the rear | MATCH |
| S9 | Shield while the unit shoots | Shielded missile units block from the front while shooting, even though the shield is visually on their back (reported as a bug; still so) | [CA forum 2660](https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/2660-missile-units-with-shields-block-projectiles-even-if-the-shields-aren-t-equipped?page=1) | medium | No such unit in our roster | n/a |
| S10 | Melee effect of the shield | None as a mechanic. `unit_shield_types.shield_defence_value` and `shield_armour_value` (column names from the [RPFM schema](https://github.com/Frodo45127/rpfm-schemas), `schema_wh3.ron` v6) are **0 in all 24 rows**, so `shield_defence_angle_melee` 60 multiplies zero: it does nothing in WH3. The shielded variant simply has a higher card melee defence: Empire spearmen 34 → 42 (+8), same armour 30, same 69 HP; clanrats with shields 22 (sword) vs clanrat spearmen 24 (different weapon) | DB; Steam [WH2 "benefit other than blocking"](https://steamcommunity.com/app/594570/discussions/0/1483235412195054574/) (+8 MD in stats, "+6 armour" is a mod) | high | Card melee defence from the passport; no shield term in `melee.py` | MATCH |
| S11 | Shield and charge | No link. Charge defence comes from the attribute `charge_defense_vs_large` (bracing, `bracing_attack_angle` 80, front only) — not from the shield | DB attributes; Steam WH2 arcs thread ("charge defense… just the front") | medium | Handled in charge/brace code, not by shield | MATCH (charge is out of scope here) |
| S12 | Old Attila keys | `missile_saving_bonus_base` 20, `_coefficient` 0.5, `missile_shield_piercing_coefficient` 0.5: a forum formula "(20 + shield + armour) × 0.5". The data contradict it: Warlord (35 shield, 90 armour) would block 72 %, measured 34 %; General (55, 85) → 80 %, measured 53 %. No projectile has a "shield piercing" column in WH3 (`projectiles` v53, 58 columns) | DB; schema; recordings | high (legacy) | Not used | MATCH |
| S13 | `shield_defence_angle_melee` 60 reused | Our sim takes 60 from it as the front/flank border of melee (`sim.json contact.front_deg` 60). That is a borrowed number for a different rule (flank sectors; conform row M7), not a shield effect | DB; `build/conform/master.md` M7 | — | `melee.py:109`, `geometry.py:78-81` | out of scope (M7) |

## Why the accuracy analyst saw 0.93 (not 0.65)

`build/accuracy/shield.txt` divided "HP lost per shot from the front" by "HP lost per shot from side and rear"
for shielded units. That assumes everything else is equal. It is not:

1. **A unit hit from the front is usually running at the shooter.** Unshielded infantry, running, per shot (HP lost /
   expected without a shield): front 0.606, side 0.525, rear 0.425 (100k / 33k / 40k shots). So front/rest ≈ 1.28
   with no shield at all. 73 % of the shielded units' front shots were on running units (24k of 33k). Expected
   ratio for them is 0.65 × 1.29 ≈ 0.84, not 0.65. Why the motion matters (arrows land along the line of fire, a
   unit coming along that line stays under them; one running away or across leaves the spot) is an accuracy
   question — see "Notes for other analysts".
2. **Different unit types.** Lords: no formation. Infantry: shielded units (melee units that run in) were compared
   with nothing; unshielded control units are mostly missile units that stand.
3. Classification is fine: the facing at landing gives the same numbers as at firing (`moving.txt`), and a running
   unit's facing follows its path (median 1–6°).

Done properly — same body with and without a shield, same speed class, front vs side+rear (`strat.txt`):

| Target (shield) | Pass-through from the front | DB | Shots, front |
|---|---|---|---|
| Empire spearmen with shields vs plain spearmen (35) | **0.64** | 0.65 | 7.7k |
| Clanrats with shields vs clanrat spearmen (35) | 0.72 | 0.65 | 10.4k |
| Warlord (35), control = unshielded infantry by speed × 30 m distance band | 0.66 [0.61, 0.77] | 0.65 | ~33k |
| General (55), same control | 0.47 [0.44, 0.54] | 0.45 | ~57k |
| All shielded infantry (35), mixed types, same control | 0.77 [0.73, 0.85] | 0.65 | ~33k |

The mixed-infantry row is still high, because its control is a different kind of unit; the two matched pairs are
the fair comparison. Clanrats 0.72 is within the type difference (sword vs spear formation).

A side check (high-arc fire over friendly units, which might hit from above): lords keep the block when the
shooter's own units stand in the line of fire (General 0.52, Warlord 0.42); the infantry split is too small to say
(0.00, interval −0.32…0.24, 1.1k shots). No evidence that high arcs ignore shields.

## The change

No number in the simulator changes. Shields: MATCH. Proposed, small:

- **C1 (guard, low priority).** Block only small arms. Passport: add `missile.category` (from
  `projectiles.category`) in `tools/nn/units.py` (the field is already decoded by `dbtables.py`). In
  `missile.py:177-178`: `shield = where(front & small_arms[i], u["shield"][j], 0)` where
  `small_arms = category in {"arrow", "musket", "sling", "javelin", "axe"}` (artillery, misc, grenade: no block).
  Effect on today's roster: none (all four projectiles are small arms). `tools/nn/units.py` and
  `tools/nn/sim` are in `VERSION_FILES`: the sim version changes, so batch it with another sim change.
- **C2 (docs, now).** `docs/ru/game/mechanics/missiles.md` (then en), section "Щиты": add that shields block in
  melee and while running (our recordings, numbers above), that side/rear multipliers belong only to the
  Directional Shield attribute, and the 0.93 explanation; replace "средняя–высокая" with "высокая".
  `build/conform/master.md`: R19 / M51 / C11 — add the recording evidence; mark the Attila formula refuted.
- **C3 (not needed).** Directional Shield (`armoured_vehicle`): add only when a Steam Tank enters the roster:
  multiplier 1.0 front / 0.5 side / 0.1 rear; sector borders are not sourced (assume the melee 60/120).

## Rule-level tests (tests/tools/test_sim.py, class TestMissile)

Existing: `test_a_shield_blocks_from_the_front_only`, `test_shielded_spearmen_take_fewer_hits_from_the_front_only`
(front 0.65, back 1.0). Add:

1. `test_shield_cone_is_60_degrees`: shielded spearmen shot with the target turned so the shooter is at 59° and
   61° from its facing → HP = 0.65× and 1.0× the plain spearmen's at the same angle.
2. `test_side_takes_no_block`: at 90° shielded = plain (no ×0.5 for normal shields).
3. `test_shield_blocks_in_melee`: target in contact with a friendly unit of the shooter (`contact` set), shot from
   the front → its own share is ×0.65 (lock S7).
4. `test_lord_shield_55`: General shot from the front loses 0.45× what he loses from behind.
5. With C1: `test_non_small_arms_ignore_shields` (fake passport with category `artillery` → no block).

## In-game test

Question: from a fixed spot, how much does a 35 % shield stop from front, 45°, 75°, flank and rear, compared with the same unit
without a shield? Our recordings already answer it (spearmen 0.64). The game check removes the last mix-ups
(motion, unit type, distance) and measures how the unit's facing changes hits per shot. Cheap: 2 battles. Run it only
if the user wants the in-game check of this card stat as the method requires; otherwise the
recordings are enough.

**Scenario** (extend `tools/archer_range.py` damage mode into a mode `shield`):

- Shooter: Empire archers `wh2_dlc13_emp_inf_archers_0` (90 men × 20 arrows), 20 m wide, fire at will — one per lane.
- Targets: `wh_main_emp_inf_spearmen_1` (shield 35 %) and `wh_main_emp_inf_spearmen_0` (no shield): the same body
  (120 men, 69 HP, armour 30, rank depth 6), same density, 30 m wide, fearless, held.
- Distance: **90 m centre to centre** in every lane (one distance → same distance factor).
- Target rotation (the angle between its facing and the direction to the archers):
  - Battle 1: L1 shielded 0°, L2 plain 0°, L3 shielded 90°, L4 plain 90°.
  - Battle 2: L1 shielded 180°, L2 plain 180°, L3 shielded 45°, L4 shielded 75°.
- Optional battle 3 (only if battle 1 front pass-through ≥ 0.8): lanes 1–2 as in battle 1 with a friendly
  spearmen unit 30 m in front of the archers (forces the high arc), lanes 3–4 clear.

**Code needed**: `archer_range.run_config("shield")`: per lane `target_key`, `rot_deg`, `d` (centre distance);
`write_scenario` puts the per-lane key into the XML; `src/entries/archer_range.lua` `place_target`: bearing =
`back_bearing + rot`, position by the centre (not the front rank); `tools/build.py` `--range-mode shield`; tests in
`tests/tools/test_build.py` and `tests/entries/test_entries.py`. Run: `python -m tools.build archer-range --range-mode shield`.

**Measure** (per lane, from `range_sample`): arrows fired = ammo drop; HP lost = drop of `target_hp`. Armour 30
makes damage per hit random (50–100 % armour roll), so single hits can't be counted from HP drops; but both
targets have the same armour, so the HP ratio is the hit ratio. Hits per shot depend on how many men are left, so
compare in bands of the target's men left (100–75 %, 75–50 %, 50–25 %) and pool. Expected volume: ~11.6 HP per
arrow on plain spearmen at 90 m (range test 28.09), 15.2 HP per hit → the plain unit lasts ~700 arrows, the
shielded one ~1100 at the front: ±4–5 % per lane, ratio ±6–7 %, enough to tell 0.65 from 1.0 (5+ sigma) and to
place the 60° edge (45° → 0.65, 75° → 1.0).

**Predictions (DB = our sim)**: shielded/plain at 0°: 0.65; 45°: 0.65; 75°: 1.0 (control: mean of plain 0° and
90°); 90°: 1.0; 180°: 1.0. Plain 90° / plain 0° and plain 180° / plain 0°: sim 1.0 (no formation shape in the hit
rate); the game value is a free extra measurement for `build/accuracy` (recordings, still units: 1.04 / ~1.0).
The usual minimum set (rating, pair gold, lord death) does not apply: no network, no armies.

**Matching sim scenario**: a script (or test) with the sim's `face_off`-style setup: archers at the origin, target
at 90 m centre to centre, `u["b"]` of the target = bearing to the archers + rot, for both keys and rot ∈ {0, 45, 59,
61, 75, 90, 180}; `missile.volley(...)` HP per volley; print shielded/plain per rot. Expected: 0.65 for rot ≤ 60,
1.00 beyond, plain equal at all rots.

Time: 2 battles × ~8–10 min (420 s model + loading) ≈ 20 min; the code change ~1 agent hour.

## Notes for other analysts

- **Accuracy (`build/accuracy`)**: per shot, unshielded infantry **running** take front 0.606 / side 0.525 / rear
  0.425 of the expected HP (walking 0.481 / 0.470 / 0.441, still 0.469 / 0.489 / 0.474), stratified by 30 m
  distance bands. So a unit running at the shooter is hit ~1.3× more per shot than one running away; the sim has
  no such term ("moving target 1.07" in `build/prespec` pooled all directions). Worth a look there, not here.
- **Mass/morale/damage**: no shield effect on melee, mass, charge or morale.

## Could not be sourced

- How the engine decides "front" exactly: the soldier's facing vs the projectile's direction, 2D or 3D (a steep
  high-arc arrow might fall outside a 3D cone). Lords keep the block under fire over friends, so 2D or no effect is
  likely; the infantry split is too small.
- Sector borders of the Directional Shield (front/side/rear angles) — not needed now.
- Whether javelins/throwing axes/grenades are blocked in WH3 (tw-modding names only arrow and musket; fandom lists
  javelins as blocked). None in our roster.
- A CA statement about shields in melee; we only have our recordings (S7).

## Files (build/shields)

- `db_dump.py` — DB dump (needs `py -3.14`): 24 shield types, our units' shields and attributes, projectile categories.
- `angle.py` / `angle.txt` — first pass on the accuracy rows by 30° bins.
- `rows.py` → `rows.json` (33 MB, not for git) — shot rows with the angle at landing, movement direction,
  speed, fire-over-friends flag.
- `moving.py` / `moving.txt` — by speed, angle at fire/landing/to the path; facing vs path.
- `strat.py` / `strat.txt` — the controlled estimates, 15° profile, over-friends split, matched pairs.
- `byproj.py` / `byproj.txt` — matched pairs by projectile.
- `melee.py` / `melee.txt` — lords in melee: front vs rest regression.
