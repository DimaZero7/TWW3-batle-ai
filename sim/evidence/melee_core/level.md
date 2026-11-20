# Melee core: THE LEVEL (melee.core.level) - evidence

Binding: `tmp/melee_core/v2/review.md` "THE LEVEL, CONSOLIDATED" over `tmp/melee_core/design_level.md`, critic
`tmp/melee_core/level/critic_review.md`. Copy: `tmp/melee_core/dev_level/sim` (the tree's `sim/` at HEAD 306230d + this
step), launcher `tmp/melee_core/run_level.py`, patch `tmp/melee_core/level/level.patch`. Scripts `tmp/melee_core/lv/`.
Frozen at 07:53 (16.11), before any twin with the level on: `tmp/melee_core/lv/frozen.json`, sha256 in `lv/frozen.sha256`.

## The law (in scope; everything else HEAD)
Scope (`melee.level_scope`): a formation of men on a formation of men, both at most small, neither with a missile
weapon, the target not routing. Hits a fighting man-second of i on j:

    L[i, role] x late[role] / fighting_files x yield x [(1 - q) P(D) + q P(0.6 D)]

- `L` swings a file-second 20-60 s by unit and role (design 1.2 / `lawlib.law`, STATIC `lv_att`, `lv_std` = L x late);
  a unit without a row takes its template family's mean by `units.json` `spacing.template` ('rabble' in the name or
  `spacing.chaotic`): rabble 0.2984, ordered 0.2242, standing 0.1067.
- role: standing = HEAD's held mask (`melee.held_cut`, the old block moved into a function, unchanged); a reflecting /
  ridden-in held unit takes the attacking role. In scope `hold_rate` is unread.
- q = 0.32 attacking / 0.21 standing; with the wrap's share 1 - (1 - q)(1 - q_wrap); on a frontal contact only (a flank /
  rear contact keeps its directional chance). `p_mix` lives only inside `per_s`: HEAD's p (with the wrap's mix) feeds the
  first strike and `back`.
- yield 0.924: a one-number fit given the rule's 0.6 (rows would also take 0.8 of the side bonus with 0.95; totals within
  5 %). No by-target term; a by-target residual of +-0.02 a swing is carried as known.
- worn units (critic change 9): in scope `front_fill_ranks`' cut is lifted down to `worn_floor` 0.44 of the men
  (fill / fill(0.44 men0), at most 1; below it the fill's shape, continuous). Rows: held spearmen 0.120-0.123 a file-second
  at 51-53 of 120 men, held swordsmen 0.130 at 63, attacking spearmen 0.206 at 47 (whole 0.222), attacking clanrats 0.276
  at 72 of 160 (0.286): worn over whole >= 0.9 everywhere (the critic's rule: no fill cut in scope); HEAD cuts x0.84-0.92.

## The late factor, re-read once by the fixed rule
Rule: the mean over cells (unit, role, enemy's role, front) of each lane's own swings a second 60-240 s over 20-60 s, every
one-on-one lane on disk with per-man rows past 120 s; standk01's extraction (a swing = a new mark value; t0 the first HP
drop). Reader `tmp/melee_core/level/lv_late.py` (output `lv_late.txt`, rows `lv_late.json`): 9 plans with memory
(behind, behind2, behind3, own, tight, tight01, intercept, flank2v1, roster incl. 20261116-044349 / -044641); walkout left
out by plan (its order changes in the fight). A lane counts with exactly one of ours and one enemy swinging (>= 5 % of the
busiest), every unit of it with per-man rows (fix after the first pass: the 8 flank2v1 runs of 13.11 had no rows of the
anvil), the window to min(240 s, last poll, a rout - 2 s, a unit under its ordered files). Role by the data: moved >= 0.4
m/s over the 8 s before t0 = attacking. 312 unit-lanes read (554 lanes skipped: 40 a unit without rows, 493 not one on
one, 21 under 120 s).

| role | late factor | cells | the critic's read (10 cells) | the design (1 cell) |
|---|---|---|---|---|
| attacking | **x0.979 +- 0.020** | 21 | 1.01 +- 0.012 | 1.05 |
| standing | **x1.070 +- 0.036** | 13 | 1.07 +- 0.03 | 1.15 |

Cells, attacking (lanes): clan0 v att 30 m 0.995 (19); clan0 v std 24 m 1.014 (26); clan0 v std 30 m 1.059 (72); clan1
v att 1.002 (9); clan1 v std 1.088 (1); cspear0 v att 0.780 (1); cspear0 v std 1.021 (7); cspear1 v att 0.954 (1); cspear1
v std 1.044 (1); gs v att 0.845 (2); halb v std 1.048 (2); slave v std 1.019 (1); spear0 v att 0.979 (27); spear0 v std
1.018 (9); spear1 v att 1.024 (3); sspear v att 0.774 (1); sspear v std 1.014 (1); sv0 v att 0.845 (1); sv0 v std 1.051
(1); sv1 v std 1.002 (1); swords v std 0.979 (2). Standing: clan0 0.971 (7); clan1 0.986 (1); cspear0 1.190 (1); cspear1
1.231 (1); gs 0.911 (1); halb 1.173 (3); slave 1.124 (1); spear0 at 24 m 1.095 (26); spear0 at 30 m 1.129 (72); spear1
1.089 (1); sspear 0.764 (1); sv0 1.168 (1); swords 1.073 (8). The attack-on-attack roster cells run to a unit's last men
(0.77-0.85); the cell mean weighs a one-lane cell as a 72-lane one, as the rule says.

## [P] by arithmetic before any twin with the level on (`lv/p_arith.py`, `lv/p_arith.txt`)
- The flank's premium over the front (an attacking striker): HEAD p_fl / p -> level p_fl / p_mix; mean over the 80
  pairs 1.73 -> 1.37; clan0 on spear0 1.58 -> 1.33, spear0 on sv1 2.80 -> 1.78 (the 80 pairs in p_arith.txt).
- A flank joiner's part over the frontal anvil's, level / HEAD: flank2v1 shallow (swords on clan0's flank, spear0
  attacking its front) x0.98 -> (i') [P] 1.61 / 1.64 (HEAD 1.62 / 1.65, game 1.37 / 1.40): the level lowers the anvil's
  front hits as much as the hammer's flank hits - it does not close (i')'s gap. A1-type held anvils: x0.76-0.94 by pair;
  the stage-0 mix x1.04 (p10 0.67, p90 1.55). strike - keep and TEACH - C: their striking part x1.04 (TEACH - C about
  +0.048 if all of it is the strike) - low confidence, no state traced.
- A1 (hold twin): clanrats 0.2784 x 0.979 = 0.273 a file-second; held spearmen 0.1074 x 1.07 = 0.115, not cut to 53 men.
  A2: row x late x HEAD's wave mean at 20-60 s (x1.034 attacking a stander, x1.053 standing).

## Lines
(filled from `tmp/melee_core/lv/lines_lv.txt` after the chain)

Results at 16:17 (16.11): `tmp/melee_core/lv/lines_lv.txt` (copied below), regressions `out/tab_HR_LV.txt`, OFF proof `out/same_lv0.txt`, cost `out/bench_lv.txt`.

```
line | measure | game | HEAD | [P] / pass rule | level | pass | note
A1 | hold twin, swings a file-second 60-120 / 120-240 s: clanrats attacking | held spearmen | 0.287 / 0.288 | 0.119 / 0.119 | - | 0.273 / 0.273 | 0.115 / 0.115 (worn: no fill cut down to 0.44 of the men) | 0.269 / 0.268 | 0.113 / 0.113 | FAIL | the attackers and the 60-120 s cell within 5 %; held F 10.0 / 10.0 at men 94 / 70
A1b | h24 twin, the attackers' swings a contact file-second 20-240 s | 0.324 (k5) | - | 0.324 +- 7 % | 0.329 | PASS | 
A4 | hold twin, the attacker's fatigue crossings winded / tired / very tired / exhausted, s after contact; stander | 36 / 72 / 113 / 170 (critic 5); fresh | HEAD's own (the level does not touch fatigue) | each within 15 s | 29 / 72 / 112 / 178; stander max level 0 | PASS | 
worn | hold twin, held spearmen's swings a file-second by men left 120-100 / 100-80 / 80-65 / 65-50 / 50-35 (after 20 s) | 0.104 (20-60 s) -> 0.120-0.123 at 51-53 men (k5) | 0.122 / 0.114 / 0.109 / 0.098 / 0.088 | no cut down to 0.44 of the men (53) | 0.118 / 0.113 / 0.113 / 0.113 / nan | - | HEAD = the copy with the level off (hold_rate 0.5, the fill)
A2 | scene twins of the 80 kinds, swings a file-second 20-60 s, 13 units x role (vs row x late x HEAD's wave) | table 1.2 | HEAD's law | within 3 % of the product | swords/att 0.224 (P 0.227, game row 0.224, HEAD 0.276); swords/std 0.130 (P 0.132, game row 0.117, HEAD 0.134); spear0/att 0.217 (P 0.220, game row 0.218, HEAD 0.321); spear0/std 0.119 (P 0.121, game row 0.107, HEAD 0.156); spear1/att 0.247 (P 0.250, game row 0.247, HEAD 0.378); spear1/std 0.119 (P 0.120, game row 0.107, HEAD 0.185); halb/att 0.207 (P 0.210, game row 0.207, HEAD 0.267); halb/std 0.119 (P 0.121, game row 0.107, HEAD 0.130); gs/att 0.219 (P 0.228, game row 0.225, HEAD 0.202); gs/std 0.116 (P 0.120, game row 0.107, HEAD 0.100); clan0/att 0.276 (P 0.282, game row 0.278, HEAD 0.373); clan0/std 0.113 (P 0.116, game row 0.103, HEAD 0.182); clan1/att 0.290 (P 0.296, game row 0.292, HEAD 0.373); clan1/std 0.118 (P 0.120, game row 0.107, HEAD 0.182); cspear0/att 0.292 (P 0.298, game row 0.295, HEAD 0.458); cspear0/std 0.118 (P 0.120, game row 0.107, HEAD 0.224); cspear1/att 0.296 (P 0.302, game row 0.298, HEAD 0.458); cspear1/std 0.118 (P 0.120, game row 0.107, HEAD 0.224); slave/att 0.344 (P 0.351, game row 0.347, HEAD 0.536); slave/std 0.118 (P 0.120, game row 0.107, HEAD 0.269); sspear/att 0.296 (P 0.303, game row 0.299, HEAD 0.556); sspear/std 0.118 (P 0.120, game row 0.107, HEAD 0.280); sv0/att 0.296 (P 0.302, game row 0.298, HEAD 0.286); sv0/std 0.118 (P 0.120, game row 0.107, HEAD 0.140); sv1/att 0.277 (P 0.283, game row 0.279, HEAD 0.249); sv1/std 0.113 (P 0.115, game row 0.102, HEAD 0.121) | FAIL | the over-read on the game stated (S3 / S4): the wave and the late factor
A3 | scene twins, hits a swing 20-60 s on the game's 64 cells | l7's cells | - | mean abs difference by target <= 0.03; chi2 <= 100 | chi2 102.3; by target clan0 0.018, clan1 0.020, cspear0 0.038, cspear1 0.014, gs 0.014, halb 0.018, slave 0.054, spear0 0.024, spear1 0.013, sspear 0.039, sv0 0.034, sv1 0.022, swords 0.028 | FAIL | 
A5 | HP a hit; F (out of scope, whole units); out-of-scope rates; the first strike (whole units), level on vs off on the same inputs | - | - | 0 / 0 / 0 / 1e-5 relative | hit 0; F out 0, whole 0; rate out 0; swing 3.1e-07; in scope F on/off 1.000-1.160 (the worn lift); pairs in / out of scope 83172 / 1556 | PASS | 
D1 | duel: sd of the 160 ln(twin / game) about the two role means | - | 0.303 | <= 0.20 (arith 0.156) | 0.159 | PASS | 
D2 | duel: slope of the 80 cells on ln of HEAD's law; R2 | - | +0.62; 0.72 | within +-0.2; R2 < 0.2 ([P] 0.0..+0.1) | +0.03; 0.01 | PASS | 
D3 | duel: rates beyond +-25 % about the role means | - | 50 | <= 20 (arith 12) | 13 | PASS | 
D4 | duel: role means ln (the attacker's loss | the stander's) - declared in-sample | - | +0.175 | -0.017 | each within +-0.15 ([P] -0.07..-0.14) | -0.033 | -0.164 | FAIL | 
D5 | duel: winners same as the game (of 80); rates in the band (two errors or +-20 %); the four flips (game -> twin) | - | 60; 93; sv1>gs a->t, clan0>halb t->a, cspear0>halb t->none, cspear0>swords t->a | >= 58 blocking; count [P] 114-131 (HEAD 93) | 59; 117; sv1>gs a->t, clan0>halb t->t, cspear0>halb t->t, cspear0>swords t->t | PASS | 
B10 | duel, the 60 s shares, mean ln(twin / game): the attacker's | the stander's | - | +0.330 | +0.055 | neither role farther than HEAD (blocking); [P] +0.13..+0.15 | +0.02..+0.03 | +0.099 | -0.036 | PASS | 
D-grp | duel group: struck spear1, mean ln(twin / game) (n) | - | +0.069 | stated | -0.286 (16) | - | 
D-grp | duel group: standing slave units (their blows: the attacker's loss), mean ln(twin / game) (n) | - | +0.837 | stated | +0.218 (10) | - | 
D-grp | duel group: attacking slave units (their blows: the stander's loss), mean ln(twin / game) (n) | - | +0.002 | stated | -0.240 (10) | - | 
B5 | turn solo_m: E's loss a second to A's rout (1e-3) | 1.02 | 1.28 | within 20 % and nearer than HEAD | 1.23 | FAIL | 
B8 | turn solo_m: A's rout s | 317 | 259 | solo m 252-276; solo s 145-161 (stated) | 302 | - | 
B6 | turn solo_s: E's loss a second to A's rout (1e-3) | 0.33 (pooled 0.33, 0.38) | 0.62 | stated ([P] 0.41) | 0.41 | - | 
B8 | turn solo_s: A's rout s | 171 | 196 | solo m 252-276; solo s 145-161 (stated) | 164 | - | 
duel2 | the 12 attack-on-attack pairs, the losses at 120 s: the level nearer than HEAD (of 24); mean |ln| | - | mean |ln| 0.132 | stated (the blind read: >= 16 holds, < 12 fails) | 15 of 24; 0.082 | - | 
hold240 | behind hold twin, A | E at 240 s | 0.697 | 0.367 | 0.756 | 0.324 | A 0.68-0.72 | E 0.33-0.37 (the worn lift) | 0.640 | 0.339 | FAIL | 
C2 | counterfar, our unit's loss 0-10 s, level over HEAD by kind (the attack lanes) | - | 1 | not above HEAD by more than 12 % (known) | spear_run_hold x1.000, spear_run_c35 x1.001, spear_run_c60 x1.001, spear_run_c90 x1.001, spear_run_c120 x1.001, spear_run_c150 x1.001, spear_walk_hold x1.000, spear_walk_c35 x1.017, spear_walk_c60 x1.018, spear_walk_c90 x1.017, spear_walk_c120 x1.017, spear_walk_c150 x1.018, spear_stand_c45 x0.737, spear_stand_c100 x0.747, spear_stand_c160 x0.747, swords_run_hold x1.022, swords_run_c35 x1.024, swords_run_c60 x1.024, swords_run_c90 x1.024, swords_run_c120 x1.024, swords_run_c150 x1.023, swords_walk_hold x1.022, swords_walk_c35 x1.036, swords_walk_c60 x1.037, swords_walk_c90 x1.036, swords_walk_c120 x1.036, swords_walk_c150 x1.037, swords_stand_c45 x0.763, swords_stand_c100 x0.773, swords_stand_c160 x0.773 | PASS | 
C5 | AI vs AI infantry 8 x 4: the winner changes (of 32); first rout s1 / s2 | 116 / 155 (first rout) | 164 / 186 | winner changes <= 8 (the stop line); first rout earlier | 10; 165 / 206 | FAIL | 
C7 | unit exam (exam_u2 general, 768 battles): unit - stand, trade | - | +0.0327 +- 0.0029 | stated; not below zero by two errors | +0.0147 +- 0.0022 | - | 
```
