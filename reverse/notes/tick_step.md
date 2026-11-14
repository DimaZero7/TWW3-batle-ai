# The battle's step: 100 or 200 ms a tick — who picks it and what is counted in ticks (14.11.2026)

Static analysis of Warhammer3.exe of 15.10 (every address is of that exe unless it says 03.11; Ghidra project
`ghidra/gproj_exe1015`, read-only; the exe of 03.11 — the one the game runs — was only searched by byte pattern and
disassembled at the places named). Game data: the step audit of 14.11 (`tmp/tick_audit/`, 89 recorded battles, disk
only) and the recordings of the plans `tight` / `tight01` / `fill` of the same night (`build/charge-probe/runs/`). No
game run for this note. Working files: `tmp/tick_reverse/` (`xrefs_d0e294.txt` every reference to the step,
`classified.txt` what each does with it, `dec*.c`, `mech*.c`, `loop*.c` decompilations, `check2.py`, `check3.py` the
data check of section 5.3).
Marks: **[I]** read in the instructions, **[D]** read in decompiler output only, **[INF]** inferred. "Step" = ms of
battle time a tick; "tick" = the world's counter (world +0x98, `mem/entities.md`).

## 0. The answer in short
1. The step is a variable, the u32 at `143d0e294` (03.11: `143d19284`, rva 0x3d19284); the file holds 100. A battle
   sets it once, in its constructor, before any unit exists: **more than 5000 on one count of the battle's setup →
   200 ms, else more than 80 on a second count → 200 ms, else 100 ms** (a unit scale above 1.0 → 300–500 ms). Nothing
   else in the battle code writes it; it cannot change inside a battle.
2. The two counts are functions virtualized by the copy protection in both exes — not readable. By the recordings the
   first behaves as the total of men of both sides, idle and parked units included (5042 → 200 ms, 4202 → 100 ms).
3. Most of the battle code converts through the step (seconds = ticks × step × 0.001, or ticks = seconds × 1000 /
   step) and does not care. What does not: **fatigue points** (per tick, no step factor), the **morale timers** and the
   **morale recount period**, three **target rules of a fighting man** (every 3 / 8 / 5 ticks), the **attack window**
   (−8 … +5 ticks), and about twenty **tick constants computed once at program start**, while the step is still 100.
   At 200 ms all of these last twice as long in seconds (the recount: 0.4 s instead of 0.5 s).
4. Checked on recordings without the game (5.3): fatigue, the morale timers and the recount period behave as read.
5. A script can read the step: `bm:model_tick_time_ms()`. A command `set_model_tick_time_ms <ms>` exists and is
   registered like `game_startup_mode`; whether the retail game accepts it on the command line is not checked.

## 1. Function map
| what | exe 15.10 | exe 03.11 |
|---|---|---|
| the step, u32 ms a tick (section `.srdata`, writable, file value 100) | `143d0e294` | `143d19284` |
| the chooser | `142dbb2cc` | `142dc2400` |
| its only caller: the battle object's constructor (call at `142d988c1`; copy of the step → battle +0x646f0 at `142d98901`) | `142d98420` | — |
| count 1 (> 5000) / count 2 (> 80): a jump into the protected section `.didata` | `142d4f188` → `14bdcfa08` → `1474a7490`; `142d4f234` → `14bdeb4a6` → `1485e4344` | `142d56524` → `14c7ed69f`; `142d565d0` → `14c80fba0` |
| override byte (set by the command, tested first by the chooser) | `143de7f72` | `143df3518` |
| command `set_model_tick_time_ms <tick time ms>`: handler / registration (Empire.cpp line 536) | `14134890c` / `140149520` | strings at `143749d58`, `143749d18` |
| the campaign environment's two constructors: step = 100, override = 0 | `14239d574` (`14239d920`–`14239d926`), `14239e2f4` (`14239e701`–`14239e707`) | — |
| Lua `model_tick_time_ms` of the battle script interface: table entry / handler | `143d3fc50` / `142ef5928` | `143d49bf0` / `142efccf0` |
| unit scale of the setup (+0x100) → size class 0…3 by the table 0.25 / 0.5 / 0.75 / 1.0 (`1436cc660`) | `1407204b8` | — |
| setup copy-constructor (the battle object begins with a copy of the setup; alliances: vector of 0x80-byte records at +0 / +4 / +8) | `142d239f0` (from `14209125c`) | — |
| the world's tick: the same list of passes every tick, tick + 1 | `142dda860` | — |
| round half up (used as round(1000 / step)) | `1405af474` | — |

Pattern of the chooser for `mem/scan.py` (one hit in each exe): `3d 88 13 00 00 77 ?? 48 8b cb e8 ?? ?? ?? ?? c7 05 ??
?? ?? ?? 64 00 00 00 83 f8 50`; the four bytes after `c7 05` are the step's displacement.

## 2. The rule
### 2.1 Writers — [I]
About 640 code references to the step (`xrefs_d0e294.txt`; 493 functions); four write it: the chooser, the command's
handler, the two campaign constructors. Everything else reads.

### 2.2 The chooser `142dbb2cc` — [I] (the same instruction for instruction in 03.11)
```
override byte set                                   -> leave the step as it is
s = setup +0x100 (unit scale);  s > 1.0:            -> 500 (s >= 9), 450 (>= 8), 400 (>= 7), 350 (>= 6), else 300
else  count1(battle) > 5000   (142dbb351: cmp eax, 0x1388; ja)          -> 200
else  count2(battle) > 80     (142dbb36a: cmp eax, 0x50; jbe keeps 100) -> 200
else                                                                   -> 100
```
- Called once, from the battle object's constructor, after the copy of the setup and before the units are built
  ([I] the call, [D] its place in the constructor). So the step is fixed before tick 1 and before any script callback.
- The scale path is out of reach of the game's own settings: unit size small … ultra = 0.25 … 1.0 ([I] the table).
- Not inputs of the chooser itself: battle type, multiplayer, replay, campaign or custom battle, graphics settings,
  game speed. A campaign's start resets the step to 100 and clears the override (2.1), the battle then chooses again.

### 2.3 The two counts — the thresholds [I], what is counted [INF]
- Both are five-byte jumps into the protected section, in 15.10 and in 03.11; the chooser is their only caller, so
  there is no open twin to read. They take only the battle object, whose head is the copied setup ([D]) — they can
  count nothing but what the setup holds, and no unit exists yet: the armies as handed to the battle, not live men.
- count 1 = the men of both sides ([INF] from the recordings, section 5): 4200 men in 30 units + 2 lords ran at
  100 ms, the same 30 units + 6 idle units parked aside (5042 men) at 200 ms; units of 120–160 men in both; 40
  units of 3117–4168 men at 100 ms. Not known: whether mounts, crews, single entities or reinforcements not yet on the
  field count, and so where exactly 5000 lies for an army with cavalry.
- count 2 is probably units (80 = four full armies) — [INF], never above 40 in our battles.

### 2.4 The command and the getter
- `set_model_tick_time_ms <tick time ms>` (`14134890c`): parses an integer, writes it to the step, sets the override
  byte — [I]. Registered by the same constructor and with the same type and flag (0, 0) as `game_startup_mode`
  (`140146890`), which our launcher passes on the retail command line — [I]; that the game accepts it there — [INF],
  not run. The override lives until a campaign is started; `game_startup_mode battle` starts none.
- Lua: `bm:model_tick_time_ms()` returns the step — [I]. No Lua setter: the strings `model_tick` give only this
  getter, the command and a benchmark statistic `model_tick_time`.
- No preference is read by the chooser. Unit size changes the men total, nothing else.

## 3. What scales
Seconds at 100 ms / at 200 ms. "per tick" = a fixed amount or a count of ticks with no step in it.

| mechanic | counted in | evidence (exe 15.10) | 100 ms | 200 ms |
|---|---|---|---|---|
| the clock "now" of a man's timers; the strike mark (+0xa30) | seconds = tick × step × 0.001 | `14305eca8`, `143251679`, `143251c1e` [I] | grain 0.1 s | grain 0.2 s |
| swing interval (ready when interval < now − mark) | seconds | `14305ec7c` [I] | weapon's interval | the same; a swing starts on a tick, so up to 0.2 s later [INF] |
| "inside my attack window" −8 … +5 | the hit mark = seconds × 1000 / step; the margins are bare ticks | `14305af4a`–`14305af7b` [I] | −0.8 … +0.5 s | −1.6 … +1.0 s |
| candidate targets of a fighting man refreshed | ticks: now − last ≥ 3 | `14305ba38`–`14305ba47` [I] | 0.3 s | 0.6 s |
| new target after the target died, (tick ^ man id) & 7 = 0 | ticks | `14308bce6`–`14308bcf1` [I] | 0.8 s | 1.6 s |
| riders' / monsters' attack start test, tick % 5 = id % 5 | ticks | `143260588`–`1432605bc` [I] | 0.5 s | 1.0 s |
| **fatigue points** of a man | per tick, no step factor | `14308a720`, the add at `14308a9d6` [I]; the function has no reference to the step (the reference list) and takes only the man | run +40, rest −180, fight +190 a second | +20, −90, +95 a second: tiring and recovery take twice as long |
| morale: full recount, the 0.15 step (`14302a500`) | every round(1000 / step) >> 1 ticks, by unit number | `14304b8e5`–`14304b948` [I] | 5 ticks = 0.5 s, 2 a second | 2 ticks = 0.4 s, 2.5 a second |
| morale: timers waver +0x94, rout +0x8c, surprise +0x88, charge +0x98 (reload from `charge_timeout` 60), after rally +0x9c | −1 a tick | `14304c26e`–`14304c2d2` [I] | charge 6 s; waver 2.5 s + 0.5 s a rank; rout 18 s − 2 s a rank | 12 s; 5 s + 1 s a rank; 36 s − 4 s a rank |
| morale: the start values of surprise (`surprise_timeout` 200) and after rally (`post_rally_no_rout_timer` 10) | not traced | — | — | — |
| morale: timed effects of the list (+0x30, float) | − step × 0.001 a tick | `14304c1af` [I] | seconds | the same |
| morale: once-a-second ring of losses | tick % round(1000 / step) = 0 | `142f5ab9f`–`142f5abd7` [I] | 10 ticks = 1 s | 5 ticks = 1 s |
| unit event timers: under fire 15 s, attacked / wounded by artillery 30 s, unit +0x34c8 10 s and others | tick counts computed at program start (section 4), −1 a tick (`142f5aad3` …) | [I] | 15 s, 30 s, 10 s | 30 s, 60 s, 20 s |
| lord killed 450 ticks, lord fled 1200 ticks | immediates | `142dce16d`, `142dce123` [I] | 45 s, 120 s | 90 s, 240 s |
| "shot at" mark of a unit (+0x34d0 / +0x34d4) | 10 ticks, immediate | `14325794b`, `14325795f` [I] | 1 s | 2 s |
| movement of a man | place += velocity × step × 0.001 | `14307967e`, `1430885fc` [I] | 0.1 s of way a tick | 0.2 s of way a tick, the same speed |
| animation clock; events (the hit, the shot's release) | tick count × step × 0.001; event seconds × 1000 / step | `14308913f`, `143257865` [I] / [D] | seconds | the same |
| missile flight | flight time += step × 0.001 | `1430bea5f` [I] | seconds | the same |
| explosion ring | the explosion's float +0x120 += step × 0.001 (its clock, by the look) | `1430ba442`–`1430ba45f` [I]; the growth itself not re-read | seconds | the same [INF] |
| reload countdown | not traced: the ready test is man +0xad0 ≤ 0 (`142dd5550`), its writer not found | — | — | — |
| charge bonus decay (deadline +0xa3c) | seconds: deadline = now + duration | `14305ea1a`, `14308d091` [D] | 13 s | the same |
| no fighting back after a new order: 10 s, 24 s | ticks = round(1000 / step) × seconds, set at the order | `143042422`; `143039854`–`1430398c1` [I] | 100 / 240 ticks | 50 / 120 ticks, the same seconds |
| contact effect duration | round(1000 / step × seconds) at the hit | `1430c9019` [I] | 100 ticks = 10 s | 50 ticks = 10 s |
| collision and touch | once a tick; a push is spread over 3 ticks × step / 3000 | `14308cf40` [D] | 10 checks a second | 5 checks a second, twice the way between them |

Nothing is skipped at the coarse step: the world's tick runs the same passes ([D]), and no battle code compares the
step with a number ([I], search over all references; the one `cmp …, 100` — `142197abc` — is outside the battle code,
not opened). A profiler of the tick (battle +0x64688, `141bad668`) compares the tick's real time with the step.

## 4. Tick constants computed once at program start — [I]
Static initialisers (the table of start-up functions, `.xcode` `1433ca0d0` …) compute round(1000 / step) × N while the
step still holds the file's 100, and store the result; the battle then counts these down one a tick. They do not follow
the chooser: at 200 ms each lasts twice the seconds. (The command of 2.4 cannot reach them either: it runs later.)

| global | ticks (seconds at 100 ms) | computed at | read at (function) |
|---|---|---|---|
| `143f643c4` | 150 (15 s): under fire, `morale_module.md` 2.6 | `1404b0dc0` | `142f50bea`, `142f50c0b` (`142f50ab4`), `142f50d5d`, `142f5ab7b` (`142f5aa58`), `14304d124` |
| `143f64448` | 300 (30 s): artillery, the same place | `1404b0b80` | `142f509b6`, `142f50a90` (`142f5098c`), `142f511d7`, `142f512af` (`142f511b0`), `142f5ab96`, `14304d10a` |
| `143f644c0` | 100 (10 s): unit +0x34c8, set by the damage functions — meaning not opened | `1404b0cc0` | `1430a628d`, `1430a6826`, `1430a822b`, `1430a8d66` (`1430a8ab0`), `1430a9270`, `1430aac6e`, `1430b6c4e` |
| `143f6444c`, `143f64444` | 300 (30 s) | `1404b0ad0`, `1404b0d20` | `142f51125`; `142f50c23`, `142f50c3c`, `14304d153` |
| `143f6443c`, `143f643c0`, `143f643c8` | 150 (15 s) | `1404b0b30`, `1404b0bd0`, `1404b0c20` | `14304d1af`, `1430b615c`, `1430b6240`; `14304d19c`, `14306b50f`; `14304d1c2`, `143260f02` |
| `143f64438`, `143f64440` | 50 (5 s) | `1404b0c70`, `1404b0d70` | `1430c0743`, `1430c078b`; `142f512e9`, `142f5168b`, `14304d189` |
| `143f64958` | 40 (4 s) | `1404b2f40` | `1430408bc` |
| `143f687b4`, `143f687b8`, `143f687b0` | 10, 10, 50 (1 s, 1 s, 5 s): a man's counter +0xdc | `1404b3ca0`, `1404b3b70`, `1404b3bc0` | `14308d00e`, `14308d042`, `14308d05d` (`14308cee8`) |
| `143f68a50`, `143f68a54` | 3000 (300 s), 80 (8 s) | `1404c3f40`, `1404c3fa0` | `1432151e0`, `14320f97c` (the game's AI) |
| `143f68a5c`, `143f68a60`, `143f68a64` | 100, 100, 150 (10 s, 10 s, 15 s) | `1404c40d0`, `1404c4190`, `1404c41c0` | `14324b9a9`; `143265f3c`, `143266250` |
| `143f63928` (20), `143f647d8` (float 0.1) | — | `1404aa6a0`, `1404b2070` | no reader found in the open code |

Only the first three rows were followed to what they time; the rest are listed by their readers.

## 5. Measurements that agree
### 5.1 The audit of 14.11 (`tmp/tick_audit/`, `tmp/night_probes/results.md` 4a)
- 89 recorded battles: 23 at 0.2 s, 66 at 0.1 s. The step is the engine's: the men's strike marks equal tick × 0.2
  there, a running man covers 0.2 s of way a tick, the script's first event `loaded` comes at model_ms 200 (100 in the
  others), `bm:repeat_callback(…, 100)` fires once a tick in both. One step from the first poll to the last in every
  battle; no relation to the game speed (0.6–10×), to the polling or to the unit count alone.
- Men: **5042 in every 0.2 s battle** (36 infantry units of 120–160 men + 2 lords; the memory binding lists 5040 where
  the lords are parked), **at most 4202 in every 0.1 s one** (30 units, 4200 + 2 lords; the 40-unit battles of the
  network and of the human: 3117–4168).
- A mechanic differs: in the first 5 s of a fight 1.5–1.9 times as many men behind the front join against a charging
  enemy at 0.2 s (35–39 against 23); steady fighting and a holding unit's losses agree.

### 5.2 After the audit, the same night
- Plan `fill` (run 20261114-211504): the 30 lane units of the morning's battle `counterfar` 1 (4200 men, ran at
  100 ms, run 20261114-063426) plus six units with no lane, parked at x = ±700 m beside the lords (three of clanrats,
  three of spearmen: 840 men by their sizes in the other runs) → 5042 → `loaded` at model_ms 200. Idle men away from
  the fight flip the step: a count of the setup, as read.
- Plan `tight01` (six runs 20261114-211912 … 213645): the lanes of `tight`, 22 units, 2802 men → `loaded` at 100.

### 5.3 Fatigue and morale by the step — recordings only (`tmp/tick_reverse/check2.py`, `check3.py`)
12 battles `tight` (200 ms) against 6 battles `tight01` (100 ms): the same lanes and units; the script's fatigue word
every 0.4 s at 200 ms and 0.3 s at 100 ms (`probe_sample`), the morale object from memory every 5 ticks (`mem_morale`).
- **Fatigue is per tick.** Clanrats after the same run of about a minute to the contact (61 s in both): at 100 ms 60
  of 60 units are already "active" at their first sample in melee, at 200 ms 0 of 216 (all fresh). Swordsmen attacking
  in the lanes `tight` and `pinE1`, fresh at the contact, seconds from the first sample in melee (median, 10 %–90 %):
  winded 52.2 (45.3–55.5) at 100 ms, n 21, against 104.4 (92.3–119.7) at 200 ms, n 40 — **× 2.00**; tired 82.8
  (78.9–86.1) against 174.8 (168.4–179.0) — **× 2.11**. Between the thresholds of the database (6600 → 12600 points) that is 202 points a second = 20.2 a tick
  at 100 ms and 88 a second = 17.6 a tick at 200 ms; from 2800 to 6600: 111 a second = 11.1 a tick against 65 = 13.0
  a tick. Per tick the same, per second half. "Per second" would give × 1.0.
- **The morale timers are bare ticks.** Charge timer (+0x98): −1.00 a tick in every pair of polls (572 pairs at
  100 ms, 1210 at 200 ms), first value seen 58 (56–60) in both (60 at the start, polls 5 ticks apart), so 5.8 s
  against 11.6 s. Waver timer (+0x94): −1.00 a tick (15 and 22 pairs), first value seen 22 (21–24) in both: 2.2 s
  against 4.4 s left. A waver that ended in a rout was seen only at 200 ms (2 units), so the waver-to-rout time
  itself is not compared.
- **The recount period follows round(1000 / step) >> 1.** Pairs of polls 5 ticks apart with the target unchanged and a
  gap of 3–6 points (every recount then moves the points by exactly 1): at 100 ms the points moved 1 in 2244 of 2244
  pairs (one recount in 5 ticks, 2 a second); at 200 ms 2 or 3 in 2538 of 2852 pairs (89 %), alternating (a recount
  every 2 ticks; 5 ticks hold two or three), mean 2.27 a second; the other 11 % moved 0 or 1 (not looked into).
- Not in these recordings: the unit event timers of section 4 (no missiles, no artillery in the lanes), riders, the
  attack window, the 3- and 8-tick rules as such.

## 6. Not found
- What the two counts include (2.3) — the functions are virtualized in both exes.
- Whether the retail game takes `set_model_tick_time_ms` from the command line (2.4).
- The reload countdown; the start values of the surprise and after-rally timers; how many ticks a frame runs (the
  frame-to-tick loop was not traced — the audit covers its result).
- Why more men join at 200 ms (5.1). By the code the candidates are: a man moves twice the way between two touch
  checks, so bodies overlap deeper before they are pushed apart, and a touch is what gives a man without a fight his
  target (`charge_impact.md` 2.4); the target rules of 3 and 8 ticks. [INF], no reading decides it.
- The step in the functions not opened: about 40 of the 493 that read it were read; the rest are classified by the
  arithmetic right after the read only (`classified.txt`: 293 × 0.001, 125 × 1000 / step, 29 integer products).

## 7. Cheapest checks in the game
1. Log `bm:model_tick_time_ms()` once at the battle's start in every probe and exam battle: the regime of each
   recording for nothing.
2. What count 1 includes: load-and-quit battles around the line — 5000 against 5001 men of infantry; riders below
   5000 whose horses would put the total above; single entities and artillery crews; more than 80 small units for
   count 2.
3. The command: one battle of 5042 men with `set_model_tick_time_ms 100;` on the launcher's line — the getter must
   return 100 and `loaded` come at model_ms 100. If it works, the unit event timers of section 4 are right again too
   (they were computed for 100).
4. The event timers of section 4 at 200 ms: one missile lane in a 5042-man battle with `--mem` — "under fire" must
   stay 150 ticks = 30 s after the last hit.

## 8. For the simulator
The simulator's rules are in seconds and were measured at 100 ms. A battle above the line plays by other numbers:
fatigue rates a second × 0.5 (database values are per tick); morale recount every 0.4 s instead of 0.5 s (the 0.15
step and the 1-point minimum 2.5 times a second); `charge_timeout`, the waver and rout timers, the lord's fall, under
fire, artillery marks × 2 in seconds; a fighting man's target refresh 0.6 s and re-target 1.6 s; riders' start test
1.0 s. Unchanged: swing interval, charge decay, movement speed, missile flight, the no-fight-back window after an
order, contact effects, the ring of losses. So the step belongs to the battle's parameters: men of the setup > 5000
(or the logged getter) → 200 ms.
