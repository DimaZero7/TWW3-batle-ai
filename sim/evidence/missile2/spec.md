# Missile indicators, part 2 — spec

Task: the remaining missile indicators of our four shooters (rate of fire, range, firing arc, penetration,
projectile speed, skirmish mode, ammo / out of ammo): the game's rule with numbers and a source, what our
simulator does, the exact change, rule tests, and short focused in-game tests with sim twins.
Metric: completeness and implementability (more aspects with a sourced number = better).

Confidence: **high** = the DB or our own clean in-game test; **medium** = community text that agrees with
our data; **low** = one community post, or our reading of an unnamed DB field.

Not redone here (links): hit share against target density / thinning — `build/accuracy` (`summ.txt`: arrows
hit slingers 0.41, spearmen 0.62, pistols 0.38-0.39 per shot); "a unit running at the shooter takes ×1.3 per
shot" — `build/shields/spec.md` (section "A unit hit from the front is usually running at the shooter",
`moving.txt`).

## 0. Our shooters in the DB

Read by `py -3.14 build/missile2/db_dump.py` (rows in `db_rows.json`).

| | Archers (Луч) | Militia (Опол) | Slave slingers (Прщ) | Night runners (Бег) |
|---|---|---|---|---|
| land unit | `wh2_dlc13_emp_inf_archers_0` | `wh_dlc04_emp_inf_free_company_militia_0` | `wh2_main_skv_inf_skavenslave_slingers_0` | `wh2_main_skv_inf_night_runners_1` |
| projectile | `wh2_dlc13_emp_bow_arrow` | `wh_dlc04_emp_free_company_pistol_bullet` | `wh2_main_skv_sling` | `wh2_main_skv_sling_night_runners` |
| category / trajectory | arrow / dual_low_fixed | musket / low (direct) | sling / dual_low_fixed | sling / dual_low_fixed |
| range `effective_range`, m | 130 | 90 | 120 | 140 |
| reload `base_reload_time`, s | 10 | 9 | 9 | 8 |
| `land_units.reload` (skill, %) | 0 | 0 | 0 | 0 |
| ammo per man `primary_ammo` | 20 | 18 | 22 | 22 |
| men | 90 | 120 | 140 | 120 |
| muzzle velocity, m/s | 45 | 90 | 50 | 50 |
| penetration | very_low | low | very_low | very_low |
| calibration distance / area | 95 m / 3.7 | 65 m / 2.0 | 85 m / 4.8 | 85 m / 4.0 |
| marksmanship of the projectile (f11) | 10 | 10 | 10 | 15 |
| fire arc `battle_entities` f15 (full angle) | 60° | 70° | 60° | 60° |
| turn rate f16, °/s | 180 | 240 | 120 | 120 |
| `can_skirmish` | yes | yes | yes | yes |
| fire whilst moving | no | yes (attribute) | no | no |
| formation (h × v, ranks) | 2.0 × 2.1, 4 | 1.6 × 1.7, 6 | 2.2 × 2.8, 5 | 2.2 × 2.8, 5 |
| half depth at default ranks | 3.2 m | 4.3 m | 5.6 m | 5.6 m |

Note: `wh2_main_skv_inf_night_runners_0` (not in our roster) throws stars: musket/low, 70 m, 7 s, 10 ammo,
fire arc 360°. Do not mix the two up.

## 1. Rate of fire in battle (K31, R3-R6, K36, Б13)

**The gap.** Standing in range with ammo, the game's shooters fire 0.05-0.07 projectiles a man a second;
the simulator 0.087-0.091. Recordings (216 fair battles, 1 s samples, game AI and network sides;
`rec/an1d.py`, `q1d.txt`, `q1f.txt`) explain **93-108 % of the gap for arrows and slings** with three rules:
a per-man fire arc, ~3 s without fire after a halt, ~3 s without fire after a target change. The reload
itself is not the cause.

**Game rules, with numbers.**

| # | Rule | Numbers | Source | Confidence |
|---|---|---|---|---|
| 1a | A man reloads in `base_reload_time` × (1 − reload skill / 100); skill = `land_units.reload` 0 + rank bonus (rank 0 in our battles) | 10 / 9 / 9 / 8 s (archers / militia / slingers / night runners) | DB; tw-modding ("% reduction") | high |
| 1b | Archers on the scripted range: whole-unit volleys **10.0 s** apart, 78-90 of 90 men per volley — the DB reload, no extra animation (1 s samples) | median 10.0 s, 32 intervals (`range_volley.py`) | our test | high |
| 1c | In battle the reload interval is the same: game-AI archers 10 s (1060 of 1897 clean intervals); slings **11-12 s** | `rec/q1*.txt` | recordings | high (arrows), medium (slings: why 11.5 and not 9 is unknown — a sling animation floor is the guess) |
| 1d | **Per-man fire arc.** A man fires only if the target (its centre) is within the unit's fire arc **seen from where he stands**: ±30° (f15 60°) archers, slingers, night runners; ±35° (f15 70°) militia. A wide line facing a target off to one side fires with its near flank only | standing, facing stable, game-AI archers (25k s): efficiency (rate × DB reload) 0.77 at 0-10° off the facing, 0.75 at 10-20°, 0.52 at 20-30°, 0.27 at 30-40°, 0.14 at 40-60°, 0.05 beyond; by the per-man ±30° share: 0.02 / 0.24 / 0.48 / 0.64 / 0.80 for share 0 / <0.5 / 0.5-0.75 / 0.75-0.99 / 1 (slings 0.00 / 0.23 / 0.47 / 0.64 / 0.70) | DB f15; recordings `q1f.txt` | high (the effect), medium (per man is our reading: it fits, nothing else tested) |
| 1e | In battle volleys are smaller: 0.57-0.62 of the men fire, against 0.9-0.97 on the range — mostly rule 1d | `rec/q1*.txt` | recordings | high |
| 1f | **A unit does not keep turning to its target.** Facing stable (< 10° in 4 s) in 79-81 % of standing in-range seconds; it turns when a new target is far off: ≥ 60° → faced within 20° after ~1 s (median); 99 % of volleys within 45° of the target centre | `q1f.txt`; `sim.json turn.why`, `stand_fire_why` | recordings | high |
| 1g | After a halt: first shot ~3 s later (arrows; slings ~5 s; pistols ~1 s). Archer range: 2-4 s (median 3.0, 4 lanes; the firing flag rises 0-1 s after the halt, arrows leave 2-3 s later) | `halt_split.txt`; `range_volley.py` | recordings, our test | high |
| 1h | **After a target change: ~3 s without fire** (any change of the engine's target, not only a new order) | `q1d.txt` "+0 target changed < 3 s" | recordings | medium-high |
| 1i | A target in melee: efficiency ~0.1 lower (archers 0.86 → 0.75, slings 0.74 → 0.69) | `q1e.txt` | recordings | medium; not needed to close the gap |
| 1j | Settled, target straight ahead, out of melee: arrows 0.85, slings 0.70-0.75 of the DB rate — the range rate | `q1c.txt` | recordings | high |

How much each rule closes (from the range rate 0.90 arrows / 0.78 slings, `q1d.txt`): arc alone 69-75 %;
arc + halt + target change 93-108 % (game AI arrows 108 %, slings 93 %; network arrows 98 %, slings 93 %);
militia only 51-125 % (fires on the move, small n).

Web: nothing on per-man arcs or retarget pauses. The 6.1 notes say fire at will "check[s] more often" and
prioritises firing over cohesion; players report a "fire at will bug" (units not firing in range, worse in
big battles; CA forum). Our recordings: the engine's firing flag is off in 13-24 % of the standing in-range
seconds; the three rules above already cover that.

**Our simulator.**

| Rule | Code, value | Status |
|---|---|---|
| reload | `sim.json missile.reload_s` arrow 11.0, sling 11.5; night runners 11.5 × 8/9 = 10.2, militia 9 × 1.2 = 10.8 (`params.py:86-92`) | arrows FITTED 10 % slow; slings MATCH; NR, militia estimates |
| volleys | `volley_load` 1: whole-unit volleys one reload apart (`battle.py:187-195`) | MATCH |
| arc | unit-level: target centre within `stand_fire_arc_deg` 45° or turn first (`battle.py:160-174`); **every man fires** | DIFFERS (no per-man arc) |
| facing | a firing unit faces its target centre every step at 80 °/s (`battle.py:396`), an aiming one too (`:397-400`) | DIFFERS (the game keeps its facing) |
| halt | `aim_s` 3.3 arrows, 4.3 sling, 3.8 others | MATCH (pistol ~1 s in the game: small n) |
| retarget | aim restarts only on a new order (`aim_reset_on_order`, `battle.py:73-77`) | MISSING for target changes without an order |

**Change** (one pack; sim version changes):
1. `reload_s.arrow` 11.0 → **10.0** (= DB); `reload_ref_s.arrow` stays 10. Slings stay 11.5 (measured; ratio
   to the DB 9 s → night runners 10.2 stays an estimate until T3). Militia stays 10.8 until T3.
2. **Per-man arc share.** New passport field `fire_arc_deg` = `battle_entities` f15 / 2 (30, 35, 30, 30;
   `tools/nn/units.py` → `units.json`, `params.py` → `u["arc"]`). In `missile.py` a function
   `arc_share(u, pw, target)` → [B, N]: the shooter's front `W = front_i` (from `geometry.dims`), the target
   centre at `lat = dist·sin(rel_i)`, `fwd = dist·cos(rel_i)`; a man at lateral `s ∈ [−W/2, W/2]` fires if
   `|lat − s| ≤ fwd·tan(arc)`; share = length of `[lat − fwd·tan(arc), lat + fwd·tan(arc)] ∩ [−W/2, W/2]` / W
   (0 if `fwd ≤ 0`; a lord: 1 if `|rel_i| ≤ arc`). In `battle.py` multiply: `clear = clear × arc_share`
   (the same path as the line-of-fire share; men out of arc start reloading too, like blocked men).
3. **Facing.** Drop "a firing unit faces its target" (`battle.py:396`) and "an aiming one faces it"
   (`:397-400`) except while `turning`: a standing shooter keeps its facing while its target centre is
   within `stand_fire_arc_deg` **45°** (now the *turn trigger*, measured; rename the `why`), and turns to face
   the target (at `turn.formation_deg_s` 80) only beyond it; a turn, once started, goes on until the target
   centre is within 10° (`missile.turn_done_deg`; the game: "faced within 20° after ~1 s"), then the unit
   keeps that facing. Units in melee and walking units: unchanged.
4. **Sticky target + retarget pause.** (a) Without an ordered target, keep last step's target while it is
   still in range, standing and not excluded (the game on Normal "fires at the first enemy it meets", 2.3.0
   notes; our arena: the nearest 55 % of the time, the second nearest 19 %); only then take the nearest.
   Without this the sim's "nearest each step" flips between two close enemies and the pause below would
   stall it. (b) `missile.retarget_s` 3.0: when the target changes from one enemy to another (both ≥ 0),
   `u["aim"]` restarts and the unit fires again only after `retarget_s` (store last step's target in
   `u["aim_tgt"]`). An ordered change already restarts at `aim_s` (keep).
5. Not changed: `volley_load` 1, `aim_s`, `aim_reset_on_order`.
Expected: sim in-range rate 0.087-0.091 → ~0.055-0.06 (the three rules on the game's own geometry give
that). Check on the gap card. Risk: the mechanics check's shooting arenas compare the game AI's 0.091 /
0.087 a man a second (11.0 / 11.5 s); with the arrow reload at 10.0 s and one target straight ahead
(share 1, no target change) the sim's archers give ~0.1 there. Before judging, measure that arena the way
the recordings were measured (full-volley intervals: the game AI's archers in battle fire 10 s apart) — the
arena's 11.0 s is a whole-run average that includes the first-volley wait and pauses.

**Rule tests** (`tests/tools/test_sim.py`, torch):
- archers (front 46 m) standing, facing north, a target centre 60 m away: at 0° every man fires in the
  volley; at 20° 0.76 of them; at 35° 0.37 and the unit does not turn; at 50° none, and the unit turns
  (≈ 0.6 s at 80 °/s), then after `aim_s` fires with share ≈ 1.
- militia (front 32 m), target 60 m at 30°: share 0.70 (±35°); the same with ±30° would be 0.50.
- the shooting-arena layout (target straight ahead, standing): rate 1 / 10.0 s a man (arrows), unchanged
  volley size.
- retarget: two enemy units in range, the nearest one dies → the next volley comes ≥ 3.0 s after the
  switch, even if the men are loaded.
- sticky target: two enemy units at 100 and 101 m whose order swaps every step → the unit keeps the first
  one and fires every reload (no 3 s pauses).
- a unit that fires at a target 30° off keeps its facing (bearing unchanged after 10 s).

**In-game tests.** T3 (reload of all four on the range) and T10 (per-man arc), section 8.

## 2. Range: from where to where (K30, R7)

**Game rule.** A unit opens fire when the distance **between the centres** of the shooter and its target is
within `effective_range` (medium confidence); once it fires, **all its men shoot**, including rear ranks
that are themselves beyond range (high).

| Evidence | Numbers | Confidence |
|---|---|---|
| Our scripted range test, 28.09 (`docs/ru/game/units/missile-range.md`), archers 130 m, blocks 6 / 12 / 27 / 47 m deep, spearmen target 8 m deep | First arrow when the archers' middle is 125-126 m from the target's near rank (≈ 129-130 m to its centre), for every depth (front rank then 122 … 102.5 m away). The engine's `unit_in_range` says yes 8-27 m earlier (front rank 130 m): it is not the firing gate | high |
| Same runs, re-read here (`range_volley.py`) | The 47 m deep block fires 78 of 90 arrows in its first volley (fire at will) and 90 of 90 (attack order) with its middle 126-128 m from the target: its rear rank is 149-151 m away, 20 m beyond range. **No per-man range** | high |
| Same runs, attack order: where the archers stop walking | middle 127.0-127.6 m from the near rank ≈ 131 m centre to centre | high |
| Recordings, first shots of standing shooters at an approaching enemy (`rec/an2.py`, `q2.txt`; 225 arrows, 571 slings) | top 10 % of first shots: centre distance − range = +0.8 m for shallow targets (half depth ~4.3 m), −1 … −2.7 m for deep ones (~6.9 m); the deep targets never beyond 0. A near-edge rule would allow up to +7 m for them | medium (target depth from default ranks; widths not recorded) |
| Militia | first shots beyond range in 30-40 % (61 cases) — unexplained (fires on the move; small n) | low |
| Community (WH2, Steam "Part of your archers not shooting") | "3 wide and 30 deep: the guys in back might not have many chances" — a guess about per-man range; our test says no | low |
| Web on centre vs edge | nothing found | — |

**Our simulator.** `missile.py:57` `choose_target`: in range when `pw["gap"]` (edge to edge) ≤ range;
`battle.py:341-342`: a shooter under ATTACK walks until edge-to-edge ≤ **0.95** × range (no `why`). Status
DIFFERS: our shooters open fire early by the two half depths — archers vs spearmen ~7 m, slingers vs
slingers ~11 m (default ranks), up to ~20 m with a 30 m wide slinger block (28 m deep).

**Change.**
1. `missile.py:57`: `pw["gap"]` → `pw["dist"]` (centre to centre).
2. `battle.py:341-342`: `t_reach = pw["dist"]...`, `in_range = t_reach <= u["range"]` (drop the 0.95: the
   game's archers stopped as soon as they could fire, ~131 m centre to centre).
3. ai_like `missile_stop` 0.9 (centre distance, measured on the game AI, `opponents.py:86`): leave.
4. Docstring of `missile.py` ("from its formation's edge … 118-135 m between centres") and a
   `sim.json missile.range_why` with the numbers above.

**Rule tests** (`tests/tools/test_sim.py`, torch):
- archers (range 130) vs spearmen: centres 129 m apart → in range; 131 m → not (before: in range up to ~137 m).
- a 47 m deep archer block, centres 129 m apart → the whole unit fires (shots = men × loaded).
- ATTACK order from 200 m (centre distance): the shooter stops at ≤ 130 m and fires; it does not walk to 0.95 × range.

**In-game test T5** (reuse `archer-range`, fire_at_will mode, 1 battle, ~3 min): four lanes, the same
archers 30 m wide, targets of different depth — spearmen 60 m wide (~3 m deep), spearmen default (8 m),
slave spearmen 15 m wide (~25 m deep), slave slingers default (11 m, loose). At the first arrow read the
archers' middle → target near rank and → target centre. Same centre distance (±2 m) in all lanes = centre
rule (expected); same near-rank distance = edge rule. Sim twin: the same lanes, first-volley distance per
lane (the target stepping 2 m every 2 s like the script).

## 3. Fire arc and turning (K36, R16, R17)

Covered by section 1 (rules 1d, 1f; changes 2-3). Summary:

| Evidence | Numbers | Confidence |
|---|---|---|
| DB `battle_entities` f15 "Fire Arc Close" (`db_rows.json`) | 60 / 70 / 60 / 60; night runners _0 (stars) 360 | high (value) |
| Mods: "Increased Firing Arc" WH3 (all ranged 140°), "Increased Fire Arc" WH2 (120°), "Firing arc 360", Roundabout (150°, "Attila vanilla") | the column is a full angle (360 = all round) → 60 = ±30° | medium |
| twcenter tutorial (search snippet) | only preset values work: 30, 40, 60, 70, 90, 160, 360 | low-medium |
| Recordings (`q1f.txt`) | efficiency by per-man ±30° share and by target angle (section 1, rule 1d) | high |
| Arena (`stand_fire_why`, 84.5k volleys) | volleys vs the target centre: 70.6 % within 15°, 23.7 % at 15-30°, 4.8 % at 30-45°, 1 % beyond: the unit's flank men keep firing at targets 30-45° off | high |

The old reading "±45° to the centre for the whole unit" (`stand_fire_arc_deg` 45, FITTED) is replaced by
"±30° per man (DB) + turn only beyond 45° (measured)". Moving fire (militia only): `move_fire_arc_deg` 90
stays FITTED (in the sim a walking unit faces its path; in the game it can walk sideways keeping its facing,
so ±35° of the facing looks like ±90° of the path).

## 4. Penetration (K34, R29)

**Game rule (DB `projectile_penetration_junctions`, parsed in this task):**

| class | entities passed | "stopped by" (DB text) |
|---|---|---|
| very_low (arrows, both slings) | **0** | "Stopped by Very Small targets" |
| low (militia pistol) | 2 | "Stopped by Small targets, slows as it passes through entities" |
| medium / high / very_high | 5 / 10 / 20 | medium / large / very large |

tw-modding ("large_3 will penetrate 3 large entities, medium entities or small entities and not go through
very large"): the size is the largest the projectile can pass through. The DB text says "stopped by" that
size. The two readings differ for the pistol (pass through up to 2 small men, or stop at the first small
man); for arrows and slings both say: **stops at the first model**. Confidence: high (arrows, slings),
low (pistol).

**Our simulator.** No penetration (MISSING). The militia hit share is a fitted number per shot
(`hit_rate.musket` 0.5, an estimate; `build/accuracy` measured pistols 0.38-0.39 hits per shot on standing
infantry), so whatever the pistol does already sits inside the measured hit share.

**Change: none.** Arrows and slings do not penetrate; for the pistol the effect (if any) is inside the
measured hit share. Re-open only if a unit with medium+ penetration (handguns, crossbows, artillery) joins.

## 5. Projectile speed, flight time, moving targets (K35, R23)

**Game rule.** Projectiles fly at `muzzle_velocity` (45 / 90 / 50 / 50 m/s); flight ~ distance / speed on a
flat shot, longer on the high arc (arrow at 130 m: ≥ 2.9 s; pistol at 90 m: 1.0 s). Shooters do not lead a
moving target ("archers don't lead their targets", WH2 Steam discussion); faster projectiles "hit moving
targets more easily" (tw-modding). `missile_mirror_lookahead_time` 0.3 belongs to the "missile mirror"
ability (a DB behaviour group of that name), not to aiming. Confidence: medium.

**Our recordings:** no miss penalty on moving targets (moving / standing 1.07, `build/prespec`); per shot a
unit running **at** the shooter takes ×1.3, running away ×0.9 (`build/shields/moving.txt`: front 0.606,
rear 0.425, standing 0.469). The arc and deep blocks hide the lead error.

**Flight time measured here on old data (T15 done without a battle;** `build/archer-range` damage mode,
standing spearmen, 1 s samples; seconds from the volley's ammo-drop second to the target's biggest
HP-drop second): archers' middle → target 70 m: 1.8 s (n 12); 90 m: 2.4 s (9); 100 m: 3.4 s (12); 120 m:
3.6 s (14). A flat shot at 45 m/s would take 1.6 / 2.0 / 2.2 / 2.7 s: the arc adds ~30 %, plus up to 1 s of
sampling. Rule of thumb for arrows: delay ≈ distance / 35 m/s. Slings (50 m/s) and the pistol (90 m/s,
flat) are not measured: expect ~distance / 40 and ~distance / 90.

**Our simulator.** Damage lands in the step of the shot (0.5 s). Status MISSING (delay), MATCHES (no miss
penalty). Side effect: our "under missile attack" morale flag (`battle.py:228` `shot_at`) starts 2-3.5 s
earlier than the game's.

**Change: none now (low value).** If wanted later: keep each step's missile HP in a small ring buffer per
pair and apply it after `round(dist / v_eff / dt)` steps, `v_eff` = 35 m/s arrows, 40 slings, 90 pistol
(`sim.json missile.flight_m_s`); a unit that dies or routs before landing still takes it (the arrows are in
the air). Rule test: archers at 105 m → the target's HP first falls 3.0 s (6 steps) after the volley.
Running-at-the-shooter ×1.3 — see `build/shields`.

## 6. Skirmish mode (K38, R40)

**Game rule.** A unit with `land_units.can_skirmish` (all four of ours) has a "skirmish" behaviour: when an
enemy comes within some distance it falls back by itself, straight away from the enemy (towards the map
edge if chased), stops at a "safe distance" and starts firing again; foot units do not shoot while falling
back. 9.0: units falling back in skirmish mode can be hit by Rampage. The trigger distance and the speed are
**not in the DB** (no kv key; searched all `_kv_*` tables, `kv_scan.py`) and **not published**.

| Evidence | Numbers | Confidence |
|---|---|---|
| fandom "Skirmishing" (search snippet; page blocked) | falls back when an enemy is "within a certain distance", flees straight away / to the map edge, stops at a safe distance and fires again | medium |
| CA 9.0 release notes | "Units falling back due to skirmish mode are now subject to rampage" | high |
| Steam (AI General mod discussion, snippet) | the AI turns skirmish on for all its ranged units | low |
| CA 7.0 notes | "Ranged units will maintain distance and use remaining ammo more effectively" (battle AI) | high |
| Our recordings (`build/ail`, 110 gate battles; `build/evade`, 28 battles; `opponents.py:96`) | the game AI's free shooters move away from an approaching enemy melee unit in 52-58 % of seconds within 40 m, 42 % at 40-60 m, 23 % at 60-80 m; but they keep firing 79 % of seconds at 20-40 m and 63 % at 40-60 m, and move away in only 24 % of approaches | high (behaviour), unknown (skirmish flag or AI orders) |
| Our units (readout `native.behaviour.skirmish`, R40) | skirmish active in 0 of 7926 readouts of our units; the network bridge does not switch it itself (only `orders.prepare_melee` for scripted melee tests does, `src/apps/orders/adapter.lua:20-24`) — off by default for script-controlled units | high (value), medium (why) |

**Q4, measured here (recordings: network runs from 02.10 with the game AI on side 2;
`build/missile2/rec/q34.txt`, `q4b.txt`).** A standing game-AI shooter out of melee, with ammo, an enemy
melee unit (lords included) closing within 80 m — n 1353 approaches:
- it backs off in **51 %** of them (arrows 45 %, slings 56 %, pistols 46 %); the network's shooters (no
  skirmish, bridge) 18 %;
- the backing off starts when the enemy is **59 m** away (p25 48, p75 70), 8 s after the enemy came within
  80 m; it moves at **2.8 m/s** (p25 2.1, p75 3.5: a run), for only **6 s** (p25 3, p75 13), ends ~63 m
  from the enemy; it rarely shoots on the way (3 %), and shoots again within 15 s in 42 % of cases;
- what triggers it is an enemy **that targets this unit and closes**: chance to start backing off per second
  24-45 % with such an enemy within 10-50 m, 17 % at 50-60 m, 10 % at 60-70 m, 6 % at 70-80 m, 3 % at
  80-100 m; an enemy closing on someone else: 4-7 % per second; no enemy within 150 m: 1.3 %.
So the game AI's shooters do "short hops": run ~15-20 m back when an enemy that goes for them comes within
~50 m, stop, shoot. Whether this is the skirmish switch or the AI's orders is not visible in the data.

**Our simulator.** No skirmish behaviour for anyone (MISSING by design: our network's shooters do not
skirmish in the game either — it reads off for them, and they back off in 18 % of approaches, 84 % of those
right after a network order). ai_like copies the game AI's behaviour by orders:
a free shooter steps back (withdraw "away and back") from an enemy melee unit within `skirmish_m` 20 m and
keeps shooting otherwise (`opponents.py:96`, `:401-414`).

**Change: none in the simulator rules; retune ai_like.** Keep it an ai_like behaviour (the game AI's
control, whatever switch drives it), with the measured trigger and hop (`opponents.py:96, 129, 401-414`):
- trigger: an enemy melee unit **whose order target is this shooter** (`order_kind` ATTACK, `order_target`
  = it), closing (speed towards it ≥ 0.5 m/s, as `guard_closing`), within **50 m** — today any enemy melee
  unit within `skirmish_m` 20 m;
- hop: one WITHDRAW of **~17 m** (6 s × 2.8 m/s, run; `withdraw_m` 50 → 17), direction as now ("away from
  it and back", `flee_straight` False — straight away pushed units in front of their line, `build/evade`);
  then stand and shoot; a new hop only if the trigger still holds when the hop ends;
- in melee: unchanged (`escape_s` 15).
Check on the sim twin (ai_like × gate starts): (a) backing off per approach ≈ 0.5, start ≈ 50-60 m, hop
≈ 6 s; (b) the firing shares that set `skirmish_m` 20 (79 % of seconds at 20-40 m, 63 % at 40-60 m,
`build/evade`) still hold. Rule test (`test_opponents`-style, torch): a spearmen unit ordered to attack an
ai_like archer unit, 45 m away and closing → the archers get WITHDRAW to a point 17 m away; the same
spearmen attacking a neighbour → no WITHDRAW.

Free readout: `unit:is_behaviour_active('skirmish')` is already read in `src/apps/units/state_adapter.lua:55-58`;
log it for the game AI's units in the next gate battles (no extra battles) to learn whether the AI's hops are
the skirmish switch.

## 7. Ammo, running out, fire at will (K28, R14, R15, R24)

**Game rule.**
- Ammo per man `primary_ammo` 20 / 18 / 22 / 22; the card shows the unit total (`starting_ammo`, 1800 for
  archers). One projectile per shot (`projectile_number` 1, `shots_per_volley` 1). Our recordings: the
  unit's ammo does not drop when men die without shooting (9287 of 9817 cases, R15) — MATCHES.
- **Out of ammo under an ATTACK order: the unit goes into melee with its target**, even in guard mode
  (Steam "Whose genius idea was it to make ranged units charge into melee once they run out of ammo?";
  snippets: "if they run out of ammo whilst attacking an enemy", "if given an attack command and run out of
  ammo later, and they even do it when guard mode is on"); earlier games stopped and stood. Confidence:
  medium (several posts, no CA text).
- Out of ammo with no order (fire at will, idle): no clear text. DB keys (no descriptions):
  `melee_attack_threshold_modifier_idle_ammo_remaining` 0.85, `_idle_default` 0.14, `_ordered` 0.22,
  `max_range_for_attack_mode_melee_reaction_attack` 40 m — read as "an idle unit with ammo is much less
  willing to start melee than one without; melee reactions reach 40 m". Confidence low.
- 8.0: shooters with a blocked line of fire no longer walk into melee (game AI). 6.1: fire at will checks
  more often and "prioritise firing over maintaining positional unit cohesion". `allow_fire_at_will_into_melee`
  1 (DB). The engine's `missile_range()` turns 0 when ammo is 0 (our test, `missile-range.md`).
- `battleai_ammo_threshold` 0.75 is for ammo-restoring abilities only.

**Q3, measured here (`rec/q34.txt`, `q4b.txt`; the moment ammo hits 0, alive, not routing; next 60 s).**
- Game AI, n 152 (55 archers, 85 slings, 12 pistols): moves within 60 s 97 % (median **1 s** after the last
  shot); reaches melee within 60 s **54 %** (median **30 s**); stands still all 60 s only 3 %; routs 7 %.
  It walks **towards** the nearest enemy: median 1.4 m/s in the first 10 s, 2.0 m/s at 10-30 s (63-76 % of
  units closing); the nearest enemy was ~100 m away at that moment.
- Network side, n 56 (confounded: the bridge gives an empty shooter under ATTACK a melee attack, 61 % got
  one, median 10 s after): melee within 60 s 54 %, stands all 60 s 27 %.
So the game AI's empty shooters go to melee by themselves or by the AI's order, about at once; we cannot
tell which from the data (T13 lane 1 can).

**Our simulator.** `battle.py:340-343`: `shooter = range > 0 & a > 0`; an ATTACK on an enemy with no ammo
is a melee attack (`close_in`), so an empty shooter under ATTACK walks in and fights — MATCHES the game.
With no order it stands (no melee reaction). The bridge gives our empty shooters under ATTACK a melee attack
(`src/apps/bridge/adapter.lua:34`, `nn_empty_melee`). ai_like: a shooter with a = 0 joins the melee line
(`opponents.py:252-255`) and charges like any melee unit.

**Change.**
1. Simulator rules: none for ATTACK (MATCHES). For a unit with no order (fire at will / held) and no ammo:
   wait for T13 lane 1. If it charges by itself in the game, add the melee reaction: an empty, idle shooter
   attacks the nearest enemy within `max_range_for_attack_mode_melee_reaction_attack` 40 m (DB) — small,
   rare case (our network re-orders its units anyway).
2. ai_like: matches already (empty shooters become line units and charge); check the twin: ammo out →
   melee within 60 s ≈ 54 %, median ≈ 30 s.
3. Ammo per man and "dead men's ammo stays": MATCHES (R15), no change.
4. The fire-at-will "not firing" part (engine fire flag off while standing in range with ammo) — see
   section 1.

**Rule tests** (existing behaviour, pin it): an empty archer unit under ATTACK on a unit 60 m away walks to
it and is in melee within ~25 s (walk 1.5 m/s); with HOLD it stays. If change 1 lands: idle, empty, an
enemy at 35 m → it attacks; at 45 m → it stays.

## 8. In-game tests (focused; reuse `archer-range`)

All on the Moorlands Route lanes of `tools/archer_range.py` (4 lanes 200 m apart, script control, 1 s
samples of ammo, firing flag, `unit_in_range`, soldier extents). Tool change needed (small): per-lane unit
keys, widths and target bearing in `tools/archer_range.py` (`LANES` → `[(x, shooter_key, width, target_key,
target_width, bearing_deg), ...]`), `place_target` at an angle off the lane axis, and the shooter's
`bearing` in each Lua sample (one `read`). Each test has a **sim twin**: the same lanes built as a sim
scenario, the same numbers read from the sim (after the changes the twin must match the game; before them
it shows the gap).

| Test | Lanes | Battles | Answers | Expected (game) / pass |
|---|---|---|---|---|
| **T3** reload, all four | damage mode: archers, militia, slave slingers, night runners, each vs a fearless standing spearmen unit straight ahead at ~0.8 of its range, until out of ammo (~3 min) | 1 | the engine's reload per type (archers known: 10.0 s); volley size | full-volley interval: archers 10.0 (control), militia 9?, slingers 9 or ~11.5 (animation floor), night runners 8 or ~11.5. Sim `reload_s` takes the measured values |
| **T10** per-man arc | archers default (46 m front) and a target 60 m away, fire at will, no orders: L1 narrow target (2-file column) at 35°; L2 wide target (spearmen 30 m front, facing them) at 35°; L3 narrow target at 50°; L4 militia (32 m front), narrow target at 30° | 1 | per-man arc; centre or nearest part of the target; when the unit turns | first-volley size / men: L1 ≈ 0.37 and no turn; L2 ≈ 0.37 (centre rule) or ≈ 0.70 (nearest-part rule); L3: turns (~1 s), then ≈ 1.0; L4 ≈ 0.70 (±35°; ±30° would give 0.50) |
| **T5** range point | fire_at_will mode: archers 30 m wide vs spearmen 60 m wide (~3 m deep), spearmen default (8 m), slave spearmen 15 m wide (~25 m deep), slave slingers default (11 m, loose) | 1 | centre or edge rule | centre distance at the first arrow the same (±2 m) in all lanes (≈ 128-132 m: the target steps in 2 m every 2 s) → centre rule |
| **T13** empty shooters + skirmish | archers with ammo cut to one volley (`set_current_ammo_unary`), spearmen 60 m ahead: L1 fire at will, no order; L2 attack order on them; L3 attack order + guard mode; L4 ammo full, **skirmish on**, the spearmen ordered to attack the archers | 1 | what an empty idle unit does by itself; skirmish trigger, hop, speed | L2, L3 walk into melee (community); L1 decides the idle rule (stand, or react within 40 m); L4 vs the recordings: start ~50-60 m, ~2.8 m/s, ~6 s hops |

Total: **4 battles, ~15 min of game time, ~30 min wall** with loading; each answers numbers nothing else
can (the recordings cannot show the engine reload of the slings / militia / night runners, the per-man arc
on a known geometry, or what an empty idle unit does without the AI's orders). Order by value: T3, T10, T5,
T13. Not run: T15 (flight time — answered from old range data, section 5), T12 (ammo of the dead — answered
by recordings, R15). Free: log `is_behaviour_active('skirmish')` of the game AI's units in the next gate
battles (section 6).

## 9. Changes in order of value

| # | Change | Where | Expected effect | Needs |
|---|---|---|---|---|
| 1 | Per-man fire arc share (DB ±30 / ±35°), keep facing within 45°, turn only beyond | `missile.py` (new `arc_share`), `battle.py:160-174, 396-400`, `units.py`/`params.py` (`fire_arc_deg`) | in-battle rate 0.087-0.091 → ~0.06 (game 0.05-0.07); volleys 0.6 of the men | nothing (T10 confirms) |
| 2 | Retarget pause `retarget_s` 3.0 | `battle.py` aim block, `sim.json` | part of the same gap (arrows ~24 %, slings ~8 % of it) | nothing |
| 3 | Arrow reload 11.0 → 10.0 | `sim.json reload_s` | +10 % for archers where share = 1 | nothing |
| 4 | Range centre to centre, no 0.95 | `missile.py:57`, `battle.py:341-342` | shooters open fire 7-20 m later; attack walk stops at range | T5 to confirm (medium) |
| 5 | ai_like skirmish trigger: enemy targeting me, closing, ≤ 50 m; 6 s run back | `opponents.py:96, 401-414` | game AI's hops (51 % of approaches, from ~59 m) | twin check of the `build/evade` firing shares |
| 6 | Night runners / militia reload | `sim.json` | per T3 | T3 |
| 7 | Idle empty shooter melee reaction ≤ 40 m | `battle.py` | rare | T13 L1 |
| — | Penetration, flight delay, moving-target miss | — | none / low | — |

Changes 1-4 are one pack (one sim-version change, one baselines rerun). Docs to update with them:
`docs/ru/game/mechanics/missiles.md` (+en), `docs/ru/game/units/indicators.md` rows K30, K31, K34-K36, K38,
Б13, `docs/ru/training/simulator.md` ("Стрельба" row and the "стрелки в бою стреляют медленнее" gap),
`docs/ru/summary.md` (the 0.05-0.07 vs 0.087-0.091 line).

## 10. What could not be sourced

- Why slings reload in ~11.5 s against the DB 9 s (arrows are exactly the DB 10 s): no web source; T3.
- Reload of the militia pistol and the night runners' sling: never measured (T3).
- Centre vs edge of the target for range: no web source; recordings medium (T5).
- Per-man arc against the target's centre or its nearest soldier: the recordings used the centre (fits);
  not separable without T10 L1 vs L2.
- Skirmish trigger distance / speed: not in the DB (no kv key), not published; only the AI's behaviour
  measured (section 6). Whether the game AI's hops are the skirmish switch or its orders: unknown.
- Out-of-ammo with no order: no text; the DB keys `melee_attack_threshold_modifier_*` have no description.
- Pistol penetration: the DB text ("stopped by small targets") and tw-modding disagree; it sits inside the
  measured hit share anyway.
- `difficulty_mod_ai_reload_extra_multiplier_high` 2.0 / `_low` 1.0: no description; assumed no effect at
  Normal.
- Web fetches of Steam / fandom were blocked (429 / 402); Steam facts come from search snippets.
