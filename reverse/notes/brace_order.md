# Brace after a new order: what the code does (static analysis, 13.11.2026)

Only the file on disk (exe 15.10), open code `.udata`; no game run, no debugger. Raw: `notes/brace_order/` (`b1.c`,
`b2.c` Ghidra; `asm.txt` capstone of every function below). Continues an external model's partial work
(`tmp/gemini_re/outC/C_brace_after_order_part1.md`, `tmp/gemini_re/ws/out/D_report.md`); what of it holds is in 5.
Marks: **В** read in full (instructions), **С** code read, meaning of a field by context, **Н** guess.
Game numbers: `sim/evidence/reissue/spec.md`, `game/probe/evidence/reissue_game.json` (Empire spearmen, 30 m front).

## 1. Function map

| what | function |
|---|---|
| man's brace bit (+0x68 bit 3) and multiplier, every tick | `143089290` |
| unit: share of braced men → +0x3c18 (also fatigue means +0x3c10/+0x3c14) | `1430472a8` (loop `143047307..1430473c5`) |
| UI/CCO "braced" badge = kv `unit_braced_status_threshold` (+0x6740, 0.75) < +0x3c18 | `143039da4` (`14303a84a..14303a85c`) |
| "man within (radius + 10 m) of the unit's formation rectangle" | `14306bb68` → box `14140e608`, distance `142d5e378` |
| locomotion state +0x26c while moving (speed +0x23c > 0.01) | `14307e1a0` |
| locomotion state from stance +0x318 when stationary | `14307e7ac` (called from `143086f04` at `1430871a5`) |
| stance setter / per-second stance re-validation | `14307db60` / `143076710` (from `143086f04` `1430870be`, once per 1000 ms) |
| unit → man melee order wrapper, delay U[0, 0.5) s | `142f642fc` (+0x84), fired in `142f88ac0` / `142f781c0` |
| unit → man move order wrapper, delay = order +0x4c ticks | `142f647d0` (soldier vfunc 0x368 = `143070a48`) |
| final turn / move-to-slot task: per-man delay rand(0..round(200/tick)+1) ticks | `14326a994` (vtable `14391a5c8`) |
| standing formation upkeep: men off their slot by > f(unit) + 1.5 m sent back | `143286cd0` (1.5 at `143286ec5`) |
| ADOPT_STANCES task (stance layout per slot), 5 s window, per-man delay ≤ 1.5 s | enter `143274740`, update `143275fac` (5.0 at `143276036`), man `14327e688`; READY `143273d38` |
| task factories (pushers): move / final-turn / adopt-stances | `14327eca4` / `14327ed4c` / `14327edec` (vtables `143ae5820` / `143ae5800` / `143ae57e0`) |
| state descriptors {name, update, enter, ?, exit} (names → `1408fa…` / `1408f6…`) | tables `143942c18…143946a70` |

## 2. Rules

### 2.1 The badge — В
`badge = (braced men / men in the unit list +0x184) > 0.75`, recomputed every unit tick (100 ms). With 120 men the
badge needs ≥ 91 braced men; it goes out as soon as 30 men are unbraced. The badge itself feeds no combat rule:
+0x3c18 is read only at `14303a852` and written in `1430472a8` (q.py 3c18).

### 2.2 A man is braced — В (conditions), С (meaning of stances)
Every tick (`143089290`):
```
unit can brace: attribute 3 / 4 / 0x27 or land_units can_brace (+0x1d1)        else unbraced     (1430892ef..143089334)
unit flag +0x3d1c bit 0x10 clear (meaning not found)                          else unbraced     (1430892e1)
man's charge factor f > 0  → return, bit unchanged                                              (143089350..14308935e)
if +0x340 != 0, or vfunc +0x28 of the object at +0x330 returns 0:  check +0x26c   (143089364..14308937d)
   0x1e, 0x1f, 0x29, 0x37            → return, bit unchanged
   0,5,6,0x1a,0x1b,0x21,0x22,0x25,0x2c..0x30,0x35,0x7d..0x7f → go on (standing)
   anything else (walk 8..0xf, run 0x10..0x17/0x20/0x24, idle 1..4, 0x18, 0x19 …) → unbraced (143089336)
man inside or within (his radius +0xa0) + 10 m of the unit's formation RECTANGLE  else unbraced  (1430893ee..14308940f)
braced: bit 3 set, multiplier (charge_module.md 3)
```
- `14307e1a0`: at speed ≤ 0.01 m/s (`14307e1ce`) it leaves +0x26c alone; above it walking/running states are set
  (stance 0: walk 8, run 0x10/0x20), all outside the list. Only stances 0x67..0x69 keep a standing code (0x7d..0x7f)
  while moving (`14307e740..14307e758`). So: **a man is unbraced from the tick he moves faster than 0.01 m/s until he
  stands still in a standing stance**.
- Stationary state = stance map `14307e7ac`: stance 0→0, 5→5, 6→6, 9→0x1a, 0xa→0x1b, 0xc→0x21, 0xd→0x22, 0xe→0x25,
  0x12..0x16→0x2c..0x30, 0x1b→0x35, 0x67..0x69→0x7d..0x7f are braced; stances 1, 2, 3, 4, 7, 8 (→ states 1–4, 0x18,
  0x19) are **standing but unbraced**.
- Stance re-validation once a second (`143076710`): an idle man of stance 0 takes stance 1 when his fatigue step
  +0x708 is 3 / 4 / 5 and (id mod 10) < 2 / 4 / 8 (`143076a02..143076a6e`, needs anim 0x329), stance 2 when his
  +0xb18 measure < 0.3 (`1430769a5`, anim 0x36e), stance 3 when +0x240 == 7 (anim 0x8ae); stance 0xd (alert) is the
  braced twin of 0. С: tired / very tired / exhausted units then have 20 / 40 / 80 % of men unbraced if their animation
  set has 0x329 — at "very tired" and worse the badge cannot come on. Not checked in the game.
- The "+10 m" is from the unit's rectangle (oriented box: unit+0x18 vfuncs 0 / 8 / 0x40, centre formation +0xb38),
  **not from the man's slot** (correction to the external notes). It matters only for men far out of the block.

### 2.3 What an order does to the men — В (delays), С (which task runs)
- A melee order from the unit to a man is queued (`142f642fc`) and fires when now ≥ issue − tick + U[0, 0.5) s
  (`142f6439a`: 7.62951e-06 × rand16; check `142f88b05`). A move order to a man (`142f647d0`) fires after the
  order's +0x4c ticks: in the final-turn task `14326a994` that is rand(0..round(200 / tick) + 1) = 0–2 ticks
  (0–0.2 s), the first man 0.
- `143032c48` (from the order entry `1430374cc`) sets the new target / angle / width and re-assigns slots
  (`1430da51c`); men then walk to new slots (unbraced while walking, 2.2).
- ADOPT_STANCES (`143274740` → `14327e688` per man): a man farther than 0.2·r (0.04·r², 0.63·r for some, 5 m in a
  unit mode) from his slot or turned too far is sent to it — at walk speed within 2 m (2.83 m in stances 0x67..0x69),
  else 0.9 × the other speed (`143067620` × 0.75 × 1.2); stance changes go with a random delay up to 1.5 s
  (`143071360`); the task leaves after 5 s (`143276036`) or when ready (`1430cd2a8`); READY (`143273d38`) sets stances
  from the slot layout (1→0xc, 2→0xe, 3→9, 4→0xa, 5..7→0x67..0x69, 8→5, 9→6 — all braced). С: the descriptor table
  around it is the group-fire machine (BUILD_LAYOUT, FIRE, RELOAD, ORDER_GROUP_FIRE, WAIT_FOR_GROUP_*), its pusher
  `14327edec` is called from `143030f24`, `143030d34`, `143030cb4`, `14302b23c`, `1430391f8`, `143040aac`. **Not shown
  that it runs for a spear unit's re-form**; its 5 s and 1.5 s are not evidence for spearmen.
- Standing upkeep `143286cd0`: slots recomputed once per 1000 ms; a counter compared with round(0.5 × men); men off
  their slot by more than `143046ab0`(unit) + 1.5 m are sent back (vfunc 0x368). С (what `143046ab0` returns).

## 3. Rule for the simulator (code + the probe's numbers)

What the code fixes (В): unbraced = more than 25 % of the men are moving (> 0.01 m/s) or in a non-standing stance;
no timer of its own after the men stop. What the code did not give: the move/re-form task that sets each man's walk,
speed and stop (not read through: tasks are behaviour-tree nodes reached by pointer, see 4) — so the duration is the
probe's, written on the men's path:
```
d_men = median path of the men to their new places
        shift by d:            d_men = d
        facing turned by a:    d_men ≈ (W / 4) · 2 sin(a / 2)       (W the front; 30 m: 1.4 / 2.9 / 5.7 / 10.6 m)
t_settle = 1.75 + d_men / 2.86          s, men's mean speed < 0.25 m/s     (probe fit, 2–20 m)
t_brace  = t_settle + 3.1               s, the badge back                  (probe: +2.95..+3.2 for shifts)
           + 1.5 for a facing change                                        (probe: a11..a45; a90 +0)
same order (no slot moved): 0;  a long run (men keep slots while moving): badge at the stop (−0.6..+0.3 s)
```
| lane | d_men (m) | rule (s) | game (s) |
|---|---|---|---|
| shift 2 | 2.0 | 5.55 | 5.7 ± 0.3 |
| shift 5 | 5.0 | 6.6 | 6.6 ± 0.3 |
| shift 10 | 10.0 | 8.35 | 8.25 ± 0.2 |
| shift 20 | 20.0 | 11.85 | 11.85 ± 0.2 |
| face 11° | 1.75 | 6.95 | 7.5 ± 0.6 |
| face 22.5° | 3.54 | 7.6 | 7.35 ± 0.4 |
| face 45° | 6.9 | 8.75 | 9.15 ± 0.2 |
| face 90° | 9.9 | 9.8 | 9.15 ± 0.8 |
| same | 0 | 0 | 0 |

(d_men for the facings: the probe's median paths.) 11° and 22.5° cost the same because the path term (0.35 s/m ×
1.8 m) is small against the ~6.4 s fixed part; the rule's 0.65 s gap is within the probe's ±0.4–0.6.
A facing order on the run (mface90, 20.1 s) and from the deployment line (f90, 18.3 s): the men re-form slowly
(stop at 17.2 / 15.6 s for 8.3 / 7.8 m), then the same tail +2.9 / +2.7 s. Why their walk takes 15 s — not found.

Against `sim/config/sim.json`:
- `melee.brace_reform` — right in kind: unbraced while the men walk to their places (code: walking men are unbraced;
  the badge needs > 75 % still), no change for the same order (С: no slot moves). `settle_s` 4.0 is a fit of the
  tail; no 4 s timer for a spear unit was found in code (the timers found are 0–0.2 s, 0–0.5 s, 1 s, and the group-fire
  5 s). `reform_m` 30: no distance threshold in code; the march case differs because the men arrive in their slots
  (С) — keep as an approximation.
- `turn.face_after_move_s` 0 — measurement only; nothing in code for or against.
- Missing (С, not in the simulator): a tired unit's idle men go unbraced (fatigue step 3 / 4 / 5 → 20 / 40 / 80 %),
  and a man more than r + 10 m outside the unit's rectangle is unbraced.

## 4. Not found
- The move / re-form task for a melee unit (MOVE_FORMED, MOVE_UNFORMED, FIND_FINAL_TURN_SPINE_*,
  TURN_ENTITIES_TO_FACE, MOVE_TO_SLOT): state tables found (`143946818…1439468b8`, `143945b70…143945d00`), the code
  reaching them is by pointer (no direct references to the descriptors); not read through. Hence no code basis for the
  ~3 s from the men's settle to the badge, the men's re-form walk speed, the arrival threshold of the move task, or
  the 15 s walk of a facing order on the run.
- Whether ADOPT_STANCES runs for spearmen (its pushers' callers not identified as unit order types).
- Meaning of unit +0x3d1c bit 0x10 (read as a flag in `14301bff8`, `143046ab0`; no writer found by q.py).
- Meaning of +0x330/+0x340 in the brace gate; of `143046ab0` in the upkeep threshold.
- UI side of "braced" beyond +0x72a (whether the CCO status list adds conditions).

## 5. The external model's claims
| claim | verdict |
|---|---|
| badge = share of +0x68 bit 3 > 0.75, recomputed in `1430472a8`, `143039da4` | confirmed (instructions) |
| standing state list, 0x1e/0x1f/0x29/0x37 keep the old bit, walking/running clear it | confirmed |
| "+0x340 == 0 or vfunc != 0 → skip the state check" | corrected: the state check is skipped only when +0x340 == 0 **and** vfunc +0x28 ≠ 0 |
| within radius + 10 m of the man's **slot** | corrected: of the unit's formation rectangle (`14306bb68`, `142d5e378`) |
| 0x67..0x69 are "turn/reform" stances | corrected: they are stance-layout stances set by READY; they stay braced even while moving |
| order → `143032c48` → slots → `1430d9cec`/`1430ce8dc`; `1430da7b8`, `1430da6b4` 20 ticks | not re-checked (read only `143032c48`: calls `1430179b0`, `1430da51c`) |
| ADOPT_STANCES 5 s window, delay ≤ 1.5 s, READY sets 0x67..0x69 | confirmed as code; applicability to spearmen not shown (group-fire table) |
| re-check every round(N × 0.5) ticks, threshold r + 1.5 m | partly: slots recomputed each 1000 ms, a counter vs round(0.5 × men), threshold `143046ab0`(unit) + 1.5 m (not "r") |
| final-turn per-man delay 0..0.2 s | confirmed (`14326a994`, order +0x4c used in `142f647d0`) |
