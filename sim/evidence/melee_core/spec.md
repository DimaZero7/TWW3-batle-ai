# Melee core at the window (melee.core share + stack + level ON) - evidence

Switches (`sim/config/sim.json` melee.core): share 1, stack 1, level 1; share_k, stack_k, profile, stand 0. All 0 = the
simulator before the window, bit for bit. Numbers and their `_why`: melee.core_why, core_share_why, core_s2_why,
core_level_why. Raw runs (not in git): `tmp/melee_core/out/` (tags HR = before the window, S2 = share + stack, LV = the
level, M = all three merged), scripts `tmp/melee_core/harness/`, `lv/`, `mg/`.

## The rules
- **share** (`sim/core.py` classes, share): the enemy's blows on a line two of our formations share. Every other enemy
  of a row gets a class (C covered, K corner, M, S, O) against the row's senior, kept from the first contact; the senior
  is relieved by x(tau), the covered junior takes y(tau), by the pair clock tau. Rows: the behind probes (32 lanes) and
  the pin lanes of the three tight recordings (94), `core_share_why`.
- **stack** (`sim/core.py` prepare, stack): our own men on a shared line - v(tau) men through, d(tau) the senior's men
  lost to the junior; w = s / min(L_j, len_a); m = min(1, men_j / (dens_j L_j)) on the junior's ordered line; an
  attacking senior's own rows (x_att, y_att, v_att, d_att: behind3 `att`, 12 lanes); free_blend on. Rows: behind,
  behind2, behind3 (`core_s2_why`).
- **level** (`sim/melee.py` level_scope, held_cut; `params.level_rows`): in scope (men on men, small, no missile, target
  not routing) a file swings at the unit's row a file-second, a swing hits with yield x the side-mixed chance. Rows:
  standk01 + roster per-man memory rows (180 + 40 unit-lanes), late factors re-read once (`level.md`).
- The three compose: share and stack move the men F, the level moves the rate a man (`per_s`); the level's p_mix lives
  only inside `per_s`.

## The short list on the merged copy (tmp/melee_core/out/lines_*_M.txt, tab_HR_M.txt, same_mg_*.txt)
| item | line | HEAD | S2 | level | merged | holds |
|---|---|---|---|---|---|---|
| 1 | all off = HEAD, every family; the suite | - | - | - | bit for bit (exam: wall time only); smoke test passes | yes |
| 1 | share + stack = S2 / level alone = LV | - | - | - | bit for bit, every family | yes |
| 2 | T-m now - turn at the end (game +0.218), line > +0.054 | -0.033 | +0.104 | - | +0.094 (300 s +0.168) | yes |
| 3 | D1 sd of 160 ln(twin / game) <= 0.20 | 0.303 | - | 0.159 | 0.159 | yes |
| 3 | D2 slope +-0.2, R2 < 0.2 | +0.62; 0.72 | - | +0.03; 0.01 | +0.03; 0.01 | yes |
| 3 | D3 rates beyond +-25 % <= 20 | 50 | - | 13 | 13 | yes |
| 3 | winners >= 58 of 80 | 60 | - | 59 | 59 | yes |
| 3 | solo lanes, one-on-one families = level alone | - | - | - | bit for bit | yes |
| 4 | AI vs AI: the game's winner >= 16 of 32 | 17 | 19 | 18 | 12 | NO |
| 4 | unit exam unit - stand above two errors | +0.0327 +- 0.0029 | - | +0.0147 +- 0.0022 | +0.0195 +- 0.0022 | yes |
| 4 | tactic TEACH - C above two errors | +0.0646 | +0.0797 +- 0.0079 | - | +0.0543 +- 0.0069 | yes |
| 4 | scene (ii') b "the pair can win" | +0.0147 +- 0.0083 | +0.0173 +- 0.0082 | - | +0.0158 +- 0.0071 | yes |
| 5 | tactic exam's compiled step, 3 rounds | 32.15 ms | +2.5 % | - | 32.94 ms, +2.5 % (spread 1.6-2.4 %) | stated |

## KNOWN (not polished)
- AI vs AI: the game's winner in 12 of 32, 13 after the review's fixes (each switch alone 18-19): the pair of rules together moves the winner; routs
  a battle 15.06 (game 16.4, HEAD 15.84); first rout 161 / 210 s (game 116 / 155).
- q30 (a 30-man joiner): m stays 1 - the junior's own HEAD line is 10.1 striking men on 30 m (0.5 / 1.48 m), not the
  design's 36.5; v at tau 45-105 s 0.483 (game 0.181); the twin's v is twice the rule's [P] even at m 1.
- An attacking senior's junior: (i') 1.070 / 1.261 / 1.440 (game 1.06 / 1.19 / 1.40) and att - att_h now within the
  lines; stand2 fails with stack_k off (E over hold 1.10 against 1.40); stack_k stays OFF.
- The level: the standing unit's loss 16 % low (D4); the hold lane's A at 240 s 0.640 (game 0.697); solo m E's loss a
  second x1.21 of the game's; the tight lanes' A at 240 s 0.10-0.12 below HEAD and the game (`away_M.txt`).
- b24a, three juniors, two in one column, b30p, H1's trades, flank2v1 shallow rear: as written in `join.md`.

## The review (20.11): seven findings, the fixes
An outside reviewer read the window (`tmp/astra/wide_flank/out/REVIEW_WINDOW.md`, repro scripts `repro/f1-f7.py`; on
the fixed code `tmp/window_fix/repro/`, output `tmp/window_fix/log/repro.log`). Judged by the written rules:
| # | finding | verdict | fix | repro |
|---|---|---|---|---|
| F1 | a pair out of touch when the senior changes keeps code 1: two seniors on its return | defect (R5 ii) | `core.py` classes: its kept class cleared, set at the return | OK |
| F2 | two older juniors on the same metres subtracted twice from a junior's free metres | defect (cons. A.1, the union) | `core.py` prepare `free`: the union by sorted spans | 10 -> 15 m |
| F2 | a C junior's shared metres not bounded by the contact length | as written (`join.md` phase 2 [I]) | none | - |
| F3 | a state built with share off (or by the old simulator) fails a step with it on | out of use | `battle.py`: the ledger before the step's snapshot; an old state with level on still fails loudly | - |
| F4 | the pair clock runs while the enemy routs: back with age and share | defect (cons. A.5) | `battle.py`: the clock counts standing pairs; routing time is a gap (reset_s) | OK |
| F5 | a step dt other than params.dt mis-picks the senior | out of use | `battle.py`: assert dt == params.dt with share | assert |
| F6 | stack without share does nothing | out of use | `battle.py`: assert stack needs share | assert |
| F7 | the worn lift raises the first strike of a worn line (55 of 120 men: +17 %) | defect (the level, consolidated 6) | `melee.py`: F0 / F_rank without the lift (exact while no side cap binds) | OK |
All switches 0 = before the fixes bit for bit. With share + stack + level: flank2v1's 320 rows and the duel matrix
unchanged; AI vs AI the game's winner 12 -> 13 of 32, routs a battle 15.06 -> 15.03 (game 16.4), first rout 161 / 210 s.
`melee.core.shared_front_cap` (off; `core_cap_why`): the target's front shared by two formations, L = W - 0.5 max(W - a,
0) (`tmp/astra/wide_flank/out/CHANGE1.md`).

## Known gross error: the hammer caught on its walk (21.11)
Game probe `side2r` (`game/probe/evidence/side2r_game.json`, 32 battles): a hammer walking round to the flank is attacked
by a second enemy unit 4 s after its order. Going on with the flank against the plain front attack: game +0.022 narrow,
-0.016 wide; the simulator -0.179, -0.239 - the decision "cancel the walk or go on" has the reversed sign. Without the
comer the flank from the second row is understated: +0.039 / +0.061 against +0.096 / +0.089.
Where (two outside passes, `caught_hammer/pass1.md`, `pass2.md`; their numbers, not rerun by us): (1) the formation is
held as one body - the hammer's first damage to the ordered target comes at 47-49 s from the order, the game's first
swing at 36 s; (2) the comer's hits stay "rear" by the unit's sector (23 % of damaging ticks) while in the game the struck
men turn (6 % of swing starts from behind; half-time to front 2.5 s); (3) after the first minute the hammer gives the
target 3-4 fighters of ~15-20 in contact, the game about half its swings. Three rules under switches were written and
REJECTED by the lines set in advance (`caught_hammer/A.patch` free advance: closes 18 %; `B.patch` individual facing:
24 %, but moves flank2v1's rear - front away from the game by 0.02-0.05; `C.patch` charge after interception: 3 %; all
three: 50 %, with the same regression). Not merged; the patches are kept for the next attempt. The same passes show that
the rear on a WIDE target (flank2v1: game +0.181, here +0.037) is not a matter of the rear defence penalty: the game's
rear attackers rarely swing "from behind" after the start (75 of 1326), yet the target loses far more.
Until mended: simulator exams of drills report apart the battles where an enemy comes at a unit AFTER its order to
walk round, and paired play on such states is not trusted; imitation of a teacher's rule is not affected.
