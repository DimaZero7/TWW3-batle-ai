# Brace after a new order: what the code does (static analysis, 13.11.2026)

Only the file on disk (exe 15.10), open code `.udata`; no game run, no debugger. Raw: `notes/brace_order/` (`b1.c`,
`b2.c` Ghidra; `asm.txt` capstone of every function below; `state_records.txt` all state records; `task_vtables.txt`
the idle and adopt tasks' state slots). Continues an external model's work (`tmp/gemini_re/outC/…part1.md`,
`tmp/gemini_re/ws/out/D_report.md`); its claims are judged in 5.
Marks: **В** read in full (instructions), **С** code read, meaning of a field by context, **Н** guess.
Game numbers: `sim/evidence/reissue/spec.md`, `game/probe/evidence/reissue_game.json` (Empire spearmen, 30 m front;
walk 1.5, run 3.0 m/s, decel 4 m/s² — `sim/config/units.json`).

## 1. Function map

| what | function |
|---|---|
| man's brace bit (+0x68 bit 3) and multiplier, every tick | `143089290` |
| unit: share of braced men → +0x3c18 | `1430472a8` (loop `143047307..1430473ff`) |
| UI/CCO "braced" badge = kv `unit_braced_status_threshold` (+0x6740, 0.75) < +0x3c18 | `143039da4` (`14303a84a..14303a85c`) |
| "man within (radius + 10 m) of the unit's formation rectangle" | `14306bb68` → box `14140e608`, distance `142d5e378` |
| locomotion state +0x26c while moving (speed +0x23c > 0.01) / from stance when still | `14307e1a0` / `14307e7ac` |
| stance setter / per-second stance re-validation | `14307db60` / `143076710` |
| unit → man melee order, delay U[0, 0.5) s | `142f642fc` (+0x84), fired in `142f88ac0` |
| unit → man move order, delay = order +0x4c ticks | `142f647d0` (soldier vfunc 0x368 = `143070a48`) |
| move-to-slot task (MOVE_UNFORMED): per-man delay 0–2 ticks | `14326a994` (vtable `14391a5c8`) |
| **formation idle task**: START → IDLE → REFORM / READY / RELOAD / HALT | ctor `14326a4f8` (vtable `143ae4858`), created by the formation tick `1430e5b34` |
| idle REFORM enter: pushes the adopt task with flags +0x64 = +0x65 = 1 | `14327472c` → `14328310c` (`14328316f`) |
| IDLE → REFORM decision | `143283398` |
| **adopt task** (states REFORM → ADOPT_STANCES), not "moving" (vtable slot 0x28 = `1408d3d10`) | ctor `14326b0c8` (vtable `143ae39b0`) |
| adopt REFORM enter / update | `143274740` → `1432831bc` (per man `14327e688`) / `143275fac` |
| adopt ADOPT_STANCES enter (stances from the slot layout) | `143273d38` |
| "all men idle": every man +0x324 == 0 and vfunc 0x3a0 | `1430cd2a8` |
| standing upkeep: men off their slot by > f(unit) + 1.5 m sent back | `143286cd0` (also slot 0x100 of the adopt vtable) |
| facing-order handler of **flying** units (needs unit +0x3e30) | `143035de0` |
| turn hold ≤ 3 s in the direct-intercept task (flag +0x5a) | `14328a1d4` ← `14327bb90` ← `143275028`, `143276428` |
| man turns on the spot before stepping when > 60° (mode +0xb28 == 2) | `1430915a8` (soldier vfunc 0x490, from `143086f04` `1430873f3`) |

State records — В: 0x28-byte records `{update, enter, f3, f4, name()}`, the **name last** (the first record at
`143942a18` begins with two functions right after string data; the last name `143946a70` is followed by a byte table
read by `1408fae7c`). A state id is a global holding the record's address (e.g. `143d20c10` → `143945e70` =
ADOPT_STANCES); a task's vtable slots 0xc8…0xf0 return these ids (transitions). Corrects this note's first version,
which paired each name with the record after it.

## 2. Rules

### 2.1 The badge — В
`badge = (braced men / men in the unit list +0x184) > 0.75`, every unit tick (100 ms). With 120 men: ≥ 91 braced.
+0x3c18 feeds no combat rule (read only at `14303a852`).

### 2.2 A man is braced — В (conditions), С (meaning of stances)
```
unit can brace: attribute 3 / 4 / 0x27 or land_units can_brace (+0x1d1)        else unbraced     (1430892ef..143089334)
unit flag +0x3d1c bit 0x10 clear (meaning not found)                          else unbraced     (1430892e1)
man's charge factor f > 0  → return, bit unchanged                                              (143089350..14308935e)
if +0x340 != 0, or vfunc +0x28 of the object at +0x330 returns 0:  check +0x26c   (143089364..14308937d)
   0x1e, 0x1f, 0x29, 0x37 → bit unchanged;  0,5,6,0x1a,0x1b,0x21,0x22,0x25,0x2c..0x30,0x35,0x7d..0x7f → go on;
   anything else (walk 8..0xf, run 0x10..0x17/0x20/0x24, idle 1..4, 0x18, 0x19 …) → unbraced (143089336)
man inside or within (his radius +0xa0) + 10 m of the unit's formation RECTANGLE  else unbraced  (1430893ee..14308940f)
```
- `14307e1a0` leaves +0x26c alone at speed ≤ 0.01 m/s (`14307e1ce`); faster, a walk/run state is set (outside the
  list) except in stances 0x67..0x69 (states 0x7d..0x7f). **A man is unbraced while he moves faster than 0.01 m/s.**
- Still: stance map `14307e7ac`; stances 1, 2, 3, 4, 7, 8 stand but are unbraced. Once a second (`143076710`) an idle
  man of stance 0 takes stance 1 when his fatigue step +0x708 is 3 / 4 / 5 and (id mod 10) < 2 / 4 / 8 (needs anim
  0x329), stance 2 when +0xb18 < 0.3 (anim 0x36e), stance 3 when +0x240 == 7 (anim 0x8ae). С: a tired / very tired /
  exhausted unit then has 20 / 40 / 80 % of its idle men unbraced if it has that animation. Not checked in the game.
- The "+10 m" is from the unit's oriented rectangle (unit+0x18 vfuncs 0 / 8 / 0x40, centre formation +0xb38), not
  from the man's slot.

### 2.3 After an order: move, then the idle task's re-form — В (code path), С (which unit kinds)
1. The order (`1430374cc` → `143032c48`) sets target / angle / width and re-assigns slots (`1430da51c`); a move task
   walks or runs the men there (per-man start delay 0–0.2 s in `14326a994`). Walking men are unbraced (2.2).
2. When the formation's task stack empties, the formation tick (`1430e5b34`) starts the **idle task** `14326a4f8` —
   every formation has one (С: its only creator). Its slot 0x28 ("moving") returns 0, as does the adopt task's — С:
   this is when the game's `is_moving` falls (the reader `1430343b4` → `1430de4c4` was not read).
3. START (`143276368`) → IDLE (`1432755d8`) → REFORM when `143283398` holds — В: immediately if the idle task has
   not re-formed yet (+0x6c == 0, cleared by its ctor), later only ≥ 5 s after the last re-form (`14328346f`,
   5 × round(1000/tick)) and if the formation is not "recently aligned" (`1430da6b4`) or has laggards (+0xd0c4).
   Gates: no unit attribute 0x53, `14304d01c`(order), `14301d888`(unit), formation +0xcec == 0.
4. Idle REFORM enter `14328310c` pushes the **adopt task** `14326b0c8` with +0x61 = `143279838`(formation),
   +0x64 = 1, +0x65 = 1 (`14328316f`: dword 0x01010000 at struct +0x1a → task +0x62..+0x65).
5. Adopt REFORM enter (`1432831bc`): if +0x61, slots recomputed (`1430d9cec`); for each man, `14327e688` with
   threshold 0.04·r² (`14328321e`; 0.4·r² for a lone man or vfunc 0x250, 25 m² and 10° when unit +0x154 ≠ 0) and
   2° (`143283275`: 364.09 of 65536); delay rand(0 … round(1500/tick)) ticks = **0–1.5 s for every man but the
   first** (`1432832b1`, needs +0x65). A man off his slot by > 0.2·r (0.13 m for r = 0.65) or turned > 2° gets, since
   +0x64 = 1, a deferred move `143071360` at walk × 1.2 within 2 m (2.83 m in stances 0x67..0x69), else
   0.75 × the other speed × 1.2 (`14327e688`); in place he only takes his stance.
6. Adopt REFORM update (`143275fac`), В: all men idle (`1430cd2a8`) → ADOPT_STANCES (`143273d38`: stances from the
   slot layout, all braced ones); **not idle 5 s after entering (`143276036`) → REFORM again** (re-check, new moves,
   new random delays) — not "after 5 s go to READY". If +0x64 and `143033998`(unit) → finish (`143d39df0`).
7. So the tail after `is_moving` falls is the adopt task: men stand (braced) until their random start, walk a short
   way (unbraced), stand again. The badge stays off while > 25 % walk at once.

### 2.4 Walk or run
- The re-form moves of the adopt task: walk × 1.2 within 2 m, else 0.9 × the other speed (2.3 step 5) — В for the
  constants, С for which speed `143067620` is.
- The main move: the order's run flag. The probe's facing orders were sent at a **walk**: `game/probe/unit_probe.py`
  `reissue_battles` gives the `face` steps no `run` (the `move` steps `run: True`), and
  `bridge/lua/entries/charge_probe.lua` passes `step.run == true` to `goto_location_angle_width`. How the run flag
  travels from the command into the move task — not traced (С that it is the order's own flag).
- `143035de0` (the external model's "facing order handler") returns at once unless unit +0x3e30 exists
  (`143035e23..143035e2d`); +0x3e30 is created only for attribute 0x20 `flying` or 0x21 `always_flying`
  (`143020a90` → `14301d098`; attribute numbers `notes/charge/attributes_enum.txt`). Its "run → walk" path is for
  flying units; `14301e0c0` there acts on the flying component, not on a run flag. Not the spearmen's path.

## 3. Rule for the simulator (code structure + the probe's numbers)
Code (В): unbraced while > 25 % of the men move faster than 0.01 m/s; after the move task the adopt task re-moves
every man off by > 0.2·r or > 2° with a random start U[0, 1.5] s, re-issued every 5 s until all are idle.
Timing written on the men's path (fit to the probe; the code gives the shape, not all the numbers):
```
d_men = median path of the men to their new places
        shift by d:            d_men = d
        facing turned by a:    d_men ≈ (W / 4) · 2 sin(a / 2)       (W the front; 30 m: 1.4 / 2.9 / 5.7 / 10.6 m)
t_settle = 1.75 + d_men / 2.86          s (men's mean speed < 0.25 m/s; probe fit, 2–20 m at a run)
t_brace  = t_settle + 3.1  (+ 1.5 for a facing change)
           the +3.1 / +2.25–2.55 after is_moving ≈ the adopt task: q75 of U[0, 1.5] = 1.1 s + a short walk ≈ 1.3 s
same order (no slot moved): 0;  a long run (men keep slots while moving): badge at the stop (−0.6..+0.3 s)
a re-form ordered at a WALK: the last men (wing, path ≈ W/2·√2 for 90°) at 1.5 m/s, then the same tail
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
| f90 (walk, from stand) | wing 21.2 | 21.2/1.5 + ~1.5 start = 15.6 stop, +2.7 → 18.3 | stop 15.6, badge 18.3 ± 0.3 |
| mface90 (walk, mid-run) | wing 21.2 | + braking from 3 m/s (0.75 s) ≈ 16.4 stop, +2.9 → 19.3 | stop 17.2, badge 20.1 ± 0.9 |

(d_men for the facings: the probe's medians.) 11° ≈ 22.5°: the path term (0.35 s/m × 1.8 m) is small against the
~6.4 s fixed part; the 0.65 s gap is inside the probe's noise. Both exceed the 2° adopt threshold (and nearly every
man exceeds 0.2·r), so both get the full adopt pass.

Against `sim/config/sim.json`:
- `melee.brace_reform` — right in kind. `settle_s` 4.0: the code's counterpart is the adopt task (random start
  U[0, 1.5] s + a short walk at ×1.2, re-issued every 5 s), not a fixed timer; 4.0 after the twin's re-form is a fair
  fit. `reform_m` 30: no distance threshold in code (С: after a long formed run the men are already on their slots and
  the adopt pass moves few of them).
- The "facing on the run, 20 s" lane is a **walk** re-form in the probe, not a game rule: a simulator facing order
  should take the run flag it is given (at a walk the wing men walk ~W/2·√2 at 1.5 m/s).
- `turn.face_after_move_s` 0 — measurement only.
- Missing (С): tired units' idle men unbraced (2.2); a man more than r + 10 m outside the rectangle is unbraced.

## 4. Not found
- How the order's run flag reaches the move task; the move task's own states (MOVE_FORMED / MOVE_UNFORMED /
  FIND_FINAL_TURN_SPINE_*) not read through.
- `is_moving` reader (`1430343b4`, `1430de4c4`) not read: that it falls when the idle / adopt tasks take over is С.
- When the direct-intercept task's flag +0x5a is set (written in `14326a574` from a parameter; its callers not read),
  hence when the 3 s turn hold applies.
- What mode +0xb28 == 2 is in `1430915a8` (the 60° turn-on-the-spot branch).
- Meaning of unit +0x3d1c bit 0x10, +0x330/+0x340 in the brace gate, `143046ab0` in the upkeep threshold, unit +0x154.

## 5. The external model's claims
| claim | verdict |
|---|---|
| badge = share of +0x68 bit 3 > 0.75, recomputed in `1430472a8`, `143039da4` | confirmed (В) |
| standing state list; 0x1e/0x1f/0x29/0x37 keep the old bit; walking/running clear it | confirmed (В) |
| "+0x340 == 0 or vfunc ≠ 0 → skip the state check" | refuted: skipped only when +0x340 == 0 **and** vfunc +0x28 ≠ 0 |
| within radius + 10 m of the man's slot | refuted: of the unit's formation rectangle (`14306bb68`) |
| state records: functions before the name (REFORM = `143275fac`/`143274740`, ADOPT_STANCES = `143274a74`/`143273d38`) | confirmed (В: table boundaries, state ids → records); this note's first version had it wrong |
| after the move the formation enters the adopt task (idle REFORM `14327472c` → `14326b0c8`) | confirmed (В for the path; С that every formation's idle task runs it) |
| 0.2·r and 2° thresholds (`14328321e`, `143283275`), 0–1.5 s random start (`1432832b1`) | confirmed (В) |
| "+0x64 = run" | refuted: the idle REFORM push sets +0x64 = +0x65 = 1 always (`14328316f`) |
| walk × 1.2 / 0.75 × run × 1.2 in `14327e688` | confirmed as constants (В); which speed `143067620` returns — С |
| REFORM waits for all at rest or 5 s, then READY / stances | refuted in part: at rest → ADOPT_STANCES; after 5 s → REFORM **again** (`143276056` jumps slot 0xc8 = REFORM) |
| adopt / idle tasks report "not moving" (slot 0x28 = 0) | confirmed for the vtables (В); that `is_moving` reads it — still open |
| facing order on the run: `143035de0` brakes, clears run, re-forms at a walk | refuted for spearmen: `143035de0` needs unit +0x3e30, flying units only (`143035e23`, `143020a90`); the probe's facing orders were sent at a walk |
| 3.0 s turn hold `14328a1d4` with flag +0x5a, ≥ 1° | confirmed as code (В: 1° `14328a21c`, 3000 ms `14328a259`); when +0x5a is set — still open |
| 60° turn on the spot in `1430915a8` | confirmed as code (В: 10922.67 = 60°), for mode +0xb28 == 2 only; its weight in the timings — still open |
| order → `143032c48` → slots → `1430d9cec`/`1430ce8dc`; `1430da7b8`; `1430da6b4` 20 ticks | `143032c48` → `1430da51c` confirmed; the rest not re-checked |
| final-turn per-man delay 0..0.2 s | confirmed (`14326a994`, +0x4c in `142f647d0`) |
| upkeep every round(N × 0.5) ticks, threshold r + 1.5 m | partly: slots recomputed each 1000 ms, a counter vs round(0.5 × men), threshold `143046ab0`(unit) + 1.5 m |
