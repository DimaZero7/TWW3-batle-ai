# Open shooting mismatches — causes and rule specs

Data: the 10 probe battles (build/missile/probe_runs, 0.5 s samples), their twins (twin_all_after.json), the recorded
battles (build/accuracy/battle_rows.json). Scripts here (host .venv, PYTHONPATH=.): thin.py, burst.py, reaim.py,
thin_hits.py, plane*.py, ksweep.py, battles_check.py, moving.py, lane_dump.py; outputs *.txt.
Web notes: web_notes.md (separate agent).

## S1. Hit chance: spread across the line of fire + men not in exact files (items 2, 4, 6; part of 1b)

Finding (plane2.txt, ksweep.txt, battles_check.txt). Two changes to `missile.hit_chance`, together:
1. **Spread in the plane across the line of fire at the target** (a calibration circle like a range target), not on
   the ground. On the ground it becomes along-range sigma_a = sigma / sin(theta) (theta = the low arc's angle, as now),
   lateral sigma; the projectile aims at the man's middle: the ground offset of the aim c = 0.5 h / tan(theta).
   (Same shadow L = h / tan(theta) as now.)
2. **Ranks not in exact files**: a projectile passes at head height over n = clamp((min(L, D) + 2r) / s_v, 1, ranks)
   ranks and meets a man with cover = 1 - (1 - min(2r / s_h, 1))^n (now: 2r / s_h, exact files).
   lateral = max(cover, erf(r / (sqrt2 sigma))); along as now with sigma_a and c:
   mean_q [Phi((D + L - c - y_q) / sigma_a) - Phi((-r - c - y_q) / sigma_a)].
   A lone man: erf(r / (sqrt2 sigma)) x [Phi((L + r - c) / sigma_a) - Phi((-r - c) / sigma_a)].
3. **calibration_area is an AREA in m^2** (web_notes.md: modder "valuated in square meters", Empire's setting
   "area of calibration target in square metres"; "the accuracy stat cuts the area by that percent"):
   sigma^2 = A x (d / calibration_distance)^2 x (1 - accuracy / 100), i.e. sigma = sqrt(A) x d / D x sqrt(1 - acc/100)
   — no k. The sqrt on (1 - acc) we already use follows from "cuts the area". The reading "A = sigma^2" (the 1-sd box
   sigma_x x sigma_y) is ours; the 1-sd CIRCLE (A = pi sigma^2) is far off (0.273). Sweep c x sqrt(A): optimum
   0.9-1.0 (ksweep_area.txt). Linear readings (sigma = k A d / D) are worse: best k 0.45-0.5, 0.109-0.113.

Numbers below are the linear k 0.5 variant; the area variant (c = 1) is better: probe 0.096 (arrows 0.101, pistols
0.104, slings 0.050, NR 0.121), battles rms 0.260 (biases arrow +0.18, pistol +0.14, sling +0.10, NR -0.02);
pistols 41/60/82/98 m 0.71/0.58/0.49/0.45 (game 0.87/0.63/0.53/0.47); archers 50 m 0.88 (0.90); Warlord 0.13 (0.13),
General 0.08 (0.09). Implement the area variant.

Effect, full-strength standing targets, 27 probe lanes, mean |log(model / game)|:
sim now 0.171 (bias -0.10) -> 0.113 (bias +0.01); arrows 0.164 -> 0.099, pistols 0.205 -> 0.146, slings 0.215 -> 0.128.
Pistols by distance 41/60/82/98 m: game 0.87/0.63/0.53/0.47, sim 0.68/0.67/0.66/0.65, new 0.86/0.74/0.65/0.60
(the slope comes from the flat shot: a height error is lost over the heads instead of a 50 m "shadow").
Close dense: archers at 50 m 0.90 game / 0.67 sim / 0.89 new; spearmen at 90-100 m 0.82-0.84 / 0.72-0.73 / 0.77-0.79.
Lords: Warlord 0.13 / 0.08 / 0.14, General 0.09 / 0.05 / 0.07.
Recorded battles (1151 cells): rms log 0.376 -> 0.304 (slings bias -0.12 -> -0.01; arrows -0.07 -> +0.21, pistols
+0.36 -> +0.46: battles mix moving/loose/thinned targets — see S2).
Outliers: narrow deep target (skavenslave spearmen 15 m wide) 0.86 new vs 0.55 game; loose slingers 0.48 vs 0.37.

Code: tools/nn/sim/missile.py hit_chance (sigma from sqrt(cal_area), sigma_a, c, cover); config/nn/sim.json drop missile.accuracy.k 1.1; a
new key accuracy.plane true (keep the old path for A/B). Sim version changes.
Tests: (a) a lone man: p falls with d for direct fire (pistol 40 m > 1.3 x 100 m); (b) archers on 180 slaves 30 m
wide at 50 m >= 0.85, at 130 m 0.6-0.8; (c) cover -> 2r/s_h for one rank, -> 1 for many ranks under a flat shot;
(d) sigma of militia at 65 m = sqrt(2.0) x 0.894 = 1.26 m.

## S2. Thinned targets are hit less (item 1b)

Game, hits a projectile by men left / band 1 (23 lanes): 0.84 / 0.66 / 0.30 (bands 0.75-0.5, 0.5-0.25, < 0.25).
Mean log(model / game): sim (re-formed: same files, fewer ranks) +0.12 / +0.20 / +0.71; plane S1 re-formed +0.09 /
+0.10 / +0.45; "holes" (the dead leave holes, the footprint stays, density x men left) -0.10 / -0.27 / -0.03.
So the game sits between re-formed and holes, at the end like holes. Cause: most likely the dead are not replaced at
once (holes), refilled only partly — not settled. Probe P1. Interim (if wanted now): p = p_lone + q (p_full - p_lone),
q = men / men at the last formation (re-formed on a move/order) — only after P1.

## S3. Fewer shots as the fight goes on (item 1a)

Game: a volley is ~0.8 of the men at once and the rest within ~1.5 s; against a lone lord (no deaths) this stays so
all fight (burst.txt). Against formations the late share grows cycle by cycle and the lateness grows to 4-8 s, from
the first deaths (target 0.94-0.98). A late man stays late (his reload restarts at his shot): the fire becomes a
stream. Cause (best fit): **a man keeps his target man; when it dies he re-aims before his next shot.**
Per-man model on the game's own recorded deaths (reaim.txt), shots model / game at the end of the lane:
no re-aim (the sim) arrows 1.18-1.39, slings 1.05-1.13, NR 1.03-1.16; re-aim aim_s at his next shot (V2): arrows
1.10-1.26, slings 0.99-1.02, NR 0.96-1.05. Slings/NR closed; arrows ~10 % still high (aim 3.3 s may be short for
arrows, or deaths are concentrated on aimed men: weighting gives 1.08-1.24).
Rule (unit level, no new number): a shooter's men whose target died since their last shot = (target men killed since
the shooter's last volley / target men then); they fire aim_s later. In battle.py keep a per-unit "late share" and
"lateness": shots per step = loaded share x (1 - late share) now + the late share aim_s later; simplest form: the
unit's effective reload R_eff = R + aim_s x (kills per R / target men). Test: 90 archers on 180 slaves at 100 m:
per-man interval over the whole fight 12-13 s (game 12.7-12.9), on a lone lord 10.0 s.
Militia (pistols): fire as a stream from the first volley (main burst 0.36-0.57) and over-predicted 7-29 % even
with re-aim (A ~1 s) — open, P2.

## S4. Running target entering range (item 5)

Game (lane_dump): the firing flag rises when centre distance <= range (archers 126 m, NR 141 m at 7.5 s); the first
projectile 1.5-2.5 s later (archers trickle 15/8/7/13 per 0.5 s, NR 75 at once). The sim fires at once: a standing
shooter's aim clock is long full. Rule: taking a target from none (-1 -> j) is a target change: the existing
retarget pause (missile.retarget_s 3.0, battle.py:214 `switched` now needs aim_tgt >= 0) also when aim_tgt < 0.
Expected first shot: archers 7.25 + 3 = 10.3 s at ~115 m (game 119.8, spread to 111); NR 141 -> ~130 m (game 132).
Test: a target walking in at 3.5 m/s is first shot >= 2.5 s after it enters range; a halted shooter facing a
target already in range: unchanged (aim_s).

## S5. A target running at the shooter (item 3)

The ×1.2-1.3 was a whole-lane figure; the target stopped at ~40 m, so most shots were close. Per volley at the same
distance (moving.txt) the effect is real but small-n: running at the shooter 0.96 / 0.72 / 0.80 game vs 0.76 / 0.59 /
0.66 standing (S1) at 120 / 132 / 95 m (x1.2-1.26), 0.84 vs 0.79 at 56 m; running away 0.65 vs 0.76 at 122 m (x0.86);
crossing at a walk x1.0-1.1. Sign = the descent angle relative to the moving target (shallower when it comes at the
shooter: longer shadow, more ranks crossed), but that gives only +-8 % — not enough. Lead/aim point unknown.
No rule yet: probe P3.

## S6. Pistol range (item 2, range part)

Militia at 97.5 m (range 90): a stream at ~0.4 of the full rate, then no fire at all from 93.8 m once the target has
108 of 180 men (lane_dump militia d88). Battles: pistol shots recorded at 100-150 m centre. A per-man range for direct
fire (each man to the target's men in front of him) fits the trickle; the stop at 108 men does not (the target's
front stayed). Open: P2.

## S7. Line of fire past friends (user note)

The sim (missile.py blocked_share / clear_shot, battle.py:221) does what the rule says with True Sight: friends at
their own size (projectile_friendly_fire_man_radius_coefficient 1.0), only direct fire (trajectory low: pistols),
blocked men do not fire, the unit holds fire / takes another target only when every line is blocked
(obstructed ratio 1.0); own ranks do not block (game: militia first volleys 0.97-1.0 of 7 ranks). Not modelled:
projectile_friendly_fire_ignore_allies_radius_coefficient (1.7 -> 1.0) and the two height coefficients (1.15 -> 1.0;
1.0) — with True Sight they are 1.0, meaning unknown (probably friends near the shooter are ignored within that
radius). So militia behind allies cannot shoot in the sim too. Not measured in the game: P4.

## Probes (tools/nn/missile_probe.py; Normal, True Sight, x20)

- P1 thinned vs fresh (1 battle, 4 lanes, 100 m, archers on skavenslaves 30 m wide): fresh 180 (control), fresh 90,
  fresh 45, fresh 45 then order the target to move 5 m and halt (forces a re-form) after it is shot to 45 — compare
  the first 3 volleys' hits per projectile with the control's band at 90 / 45 men. Holes: fresh 45 >> thinned 45;
  re-formed: equal. Run twice (2 battles, ~10 min).
- P2 pistols (1 battle): militia on skavenslaves at 86 / 90 / 94 / 98 m centre, target 30 m wide 8 ranks and a
  2-rank target; record which share fires per 0.5 s and whether the militia's centre moves. Settles per-man range.
- P3 moving (1 battle): archers on skavenslaves running at them from 160 m and stopping at 110 m (so the shots stay
  at 110-130 m), running away from 100 to 160 m, a standing twin at 120 m; 2 battles for n.
- P4 line of fire (1 battle): militia 40 m behind a friendly spearman unit, offset 0 / 0.5 / 0.75 of their front
  sideways, target at 70 m: share of men firing and HP lost by the spearmen (friendly fire).

## Twin check of S1 (twin_proto.py, the real sim on all probe lanes, CPU container; twin_proto.json)

Mean |log(sim / game)|, standing lanes: full-strength hits a projectile 0.153 now -> 0.118 with S1 (arrows 0.159 ->
0.109, slings 0.145 -> 0.110, pistols 0.167 -> 0.144, NR 0.123 -> 0.151). Whole fight (target thinning) 0.177 ->
0.170: arrows get worse (0.164 -> 0.218) because a thinned, re-formed unit stays easy to hit — S1 must go with S2.
Distance law checked (law.txt): sigma ~ d everywhere fits best (0.096); "max accuracy up to the calibration distance"
(flat below it) 0.145, "area ~ d" 0.123.
