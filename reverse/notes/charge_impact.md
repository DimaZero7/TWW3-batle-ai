# The first ten seconds of a charge: who swings, when the blow lands, why the lull (14.11.2026)

Static analysis of Warhammer3.exe of 15.10 (every address below is from that exe; Ghidra project `ghidra/gproj_exe1015`,
read-only) and the per-man memory rows of the game probe `meetfar`, part 1 `counterfar` (6 battles, 90 lanes, runs
20261114-063426 … 064423; game exe of 03.11, layout `mem/sites_exe1103.json`, offsets unchanged). No game run for this
note. Raw: `notes/charge_impact/e1.c … e3.c` (decompilations). Scripts and logs: `tmp/charge_impact/` (`load.py` caches
the rows; `stats1 … stats11.py`, `hitsizes.py`; `twin.py` — the simulator's twin with a patched copy of `sim/melee.py`).
Builds on `charge_module.md` 4–5, `melee_cycle.md`, `hold_rate.md` 2.2–2.3, `melee_stacking.md` 2.1–2.4.
Confidence: **A** read in the code and agrees with the memory rows; **B** read in the code, not checked against data;
**C** inferred. A swing = a new value of a man's strike mark (+0xa30). t0 of a lane = the first tick a man of either
unit loses HP (rows every 0.1 s from 60 m before contact to 5 s after it, every 0.5 s later). ± = standard error.

## 0. The answer in short
Not a weaker impact and not hit reactions. In the game
1. the opening is one blow of every man who has a target — the attacker's first rank; on a standing unit exactly the
   men those attackers aim at — with the database's chance and damage;
2. after it nobody can swing for about 1.5 s: the opening's men wait their weapon's interval, hit or miss, and a man
   behind them gets a fight only when an enemy body touches him, then must turn to within 10° — his blow starts a
   median 0.9 s after the touch and lands a second later;
3. from then on the swings a second stay at 1.15–1.3 of the 20–60 s level, though 2.3–3.3 times as many men stand
   within reach; the HP a second falls over 13 s with the charge bonus only;
4. against a formation that stood at the contact 0.57–0.73 as many men join in the first 5 s as against one that
   charged (the bodies mesh less): a holder loses 590 ± 27 HP in 10 s and 1979 ± 34 in 60 s, a counter-charger 668 ± 13
   and 2162 ± 23 — the simulator gives both the same 802 and 2290;
5. a held unit swings at its steady hold rate from the first second.

## 1. Function map (exe 15.10)
| what | function |
|---|---|
| ready to swing: interval < now − mark, strict | `14305ec7c`; interval `143073320`; time left `14305f948` |
| a man's attack command (slot 0x4b0 of the men's vtables, e.g. `143912950` — the ones whose slot 0x2a0 holds): builds the free-fight action, action state +0x324 = 1 | `14306e89c` → `143053684` |
| may a touch start a fight (slot 0x618) | `14309117c` |
| body touch → both men are told to strike the toucher | `14311b320`, `14311b428` → `14306bc78` → `143056730` |
| approach under the unit's order: the man ahead, the standing target's answer, attack start | `143087d24` → `14308f0e0` (→ `143056be8` → `143074eb4`) |
| free fight: target choice, reach, start | `1430877ec` → `14308d10c` → `14308b938` |
| start an attack: four kinds tried in turn | `143081f18`: `143255edc`, `143256008`, `143256134`, `143255db0` (vtables `143918230`, `…278`, `…2c0`, `…308`) |
| kind "standard" (a man on men of his size): pick the animation / try to start it every tick | `14325f190` / `143264d34` → `14325fca4` → `143251af8` (mark = now) |
| kind for the classes whose slot 0x248 or 0x260 holds (both false for `143912950`; riders, monsters — C): start test every 5th tick | `14325f8d4` / `1432651a0` → `143260464` → `143251608` |
| kind "smaller target" (the target's size class below the man's, `14305e6b0`) | `14325f60c` |
| kind played while the timer still runs (time left ≥ 0.01 s; ends when it falls under the animation's +0x50) — C | `14325f468` / `143264ea4` |
| hit reaction (slot 0x558), knock down (0x548), fly (0x550) | `14306ee40`, `14306f37c`, `14306f594` |
| "inside my attack window": animation frame within −8 … +5 ticks of its hit mark | `14305ae80` |

## 2. What the program does
### 2.1 One timer a man, hit or miss — A
`14305ec7c`: ready when `interval < now − mark`; the mark is written when an attack animation starts (`143251608`,
`143251af8`). Memory: 32,638 pairs of consecutive swings of one man in the 90 lanes, none closer than the weapon's
interval (p5 = 4.3 s swords and clanrat blades, 5.7–5.8 s spear = `melee_attack_interval`); medians 5.3–5.7 s clanrats,
6.1–7.2 s swordsmen, 6.6–8.1 s spearmen, the same after a hit and after a miss (`stats2.py`, `stats9.py`).

### 2.2 The blow is started ahead of the contact so that it lands on arrival — A (the test), B (its constants)
`14325fca4`, every tick while the standard attack has not started:
```
ready (2.1), else nothing
strike point known and the man within 0.2 x scale of it, facing within 10 deg (0x44e38e39 of 65536 a turn)  -> start
else: T and the hit point of the chosen animation's first hit window (record +0x10, 14325503c)
      gap = | target + v x T - man | - the target's radius toward the man (143254fcc)
            v (1432552b0): the target's velocity, less the man's own when 14306e490 or 1432667cc holds; 0 unless
            the attack's flag 2 (+0xc8) or the action's flag 2 (+0x88) is set
      start if gap < the hit point's distance from the man and the facing is within 10 deg
      (60 deg when the action's slot 0x70 holds; 30 deg for entity kinds 1-2; 20 deg with the action's flag 2 and
       the target not in a paired fight)
```
T and the reach are animation data, not in the exe. Memory: the opening swings start 5.5–5.8 m (median) from the
target when one side stands and the other comes at 4.8 m/s, 7.4–8.3 m when both run; the target's HP falls a median
0.9 s (spear) / 1.1 s (sword, clanrat blade) after the mark, p5–p95 0.5–1.5 s; at the mark the man faces his target
within 5–7° (median), p90 12–16° (the row comes up to 0.1 s after the start).

### 2.3 Who swings in the opening — A
- **Attack order**: the unit gives targets to its first rank only (`14327c350`, `melee_stacking` 2.1). Memory: 19.4–19.7
  swings of the Empire's 20 files, 17.4 ± 0.1 of the clanrats' ~19 (16.0 ± 0.4 on a standing unit), all first swings,
  started within ~1 s (the rabble's front is ragged: 3.6–6.5 of the clanrats' start more than a second before t0).
- **Standing, never ordered**: `14308f0e0` of an approaching man calls `143056be8` → `143074eb4(his target, him)` once an
  approach (flag 0x40 at +0xe4); the target answers if he stands (speed ≤ 0.1), his action state is 0, he is fit, and
  the comer is within `bracing_attack_angle` 80° of his facing. Memory: the held men who swing in the opening are the
  men the chargers' opening swings aim at — 14.3 of 14.3 (spearmen), 15.3 of 15.3 and 14.7 of 15.0 (swordsmen); the
  chargers' 15.3–17.0 swings had 14.7–16.3 distinct targets (`stats7.py`). Nobody else of the held unit swings.
  Standing clanrats answer with 10.4 ± 0.4.
- The chance and the blow are the database's: hits by t0 + 1.3 s over swings × chance — clanrats 0.88–1.08 in six
  cells (pooled 0.98 ± 0.05), swordsmen attacking 0.84 ± 0.05 / 0.92 ± 0.10, spearmen attacking 0.75 ± 0.06 / 0.67 ± 0.10,
  swordsmen holding 0.89 ± 0.14, spearmen holding 0.55 ± 0.13 (`stats8.py`). Single hits: 30 HP from charging clanrats
  falling to 21.5 after 13 s, 36–37 → 24 from swordsmen, 25 → 21.5 from spearmen, and **42–44 HP from held spearmen in
  the first 4 s** (26 of 26 hits; 23–29 % of hits at 4–13 s): the charge reflection ×2 holds (`hitsizes.py`).

### 2.4 Everyone else: touch → fight → face → swing — B (the path), durations from memory
A man who is not in the first rank has no enemy of his own. He gets one when
1. an enemy body touches him: `14306bc78` (from the collision handlers `14311b320` / `14311b428`) → `143056730` → slot
   0x4b0 `14306e89c`. `14309117c` refuses a man in action state 1, in a paired fight (+0x5c0 with the task's slot 0x80)
   or of a unit with attribute 0x45; `14306e89c` every state but 0 and 1 (so not a man in a hit reaction or knocked);
2. an enemy swings at him (`143074dcc`, `melee_wrap` 5) or comes at him by order (2.3).
The command builds the free-fight action (`143053684`, state +0x324 = 1); its tick (`14308d10c`, candidates refreshed
every 3 ticks) calls `14308b938`, which starts an attack (`143081f18`) only if the man is not moving by order, no enemy
stands on the line to the target and the gap is under r + r + 3 × scale; the attack then waits in `14325fca4` for the
10° facing (2.2).
Memory, the men whose first swing comes 0.5–3.0 s after t0 (1,300 men of the mutual-charge lanes; `stats4.py`): an
enemy first within 1.35 m at +0.7 … +1.0 s (median; p10 0.3–0.7, p90 1.0–1.2); state 1 a median 0.4–0.5 s after that;
the swing a median 0.4–0.5 s after state 1 (p90 1.1–1.4); **touch → swing: p10 0.2–0.3, median 0.9–1.0, p90 1.7–1.9 s**;
the angle to the target 17–20° (median; p90 41–58°) at state 1 and 5–7° at the swing; while waiting 40 % of the time
still in the charge run (movement state 0x1e), 40 % standing (0x21), 20 % in 0x2a; none was hit before his swing. A
held unit's second-wave men are in state 1 before the touch (commands 2) and swing a median 0.8–1.6 s later, 27–32° off
at state 1. Swings, clock t0 (opening: the count before +0.5 s; 0.5–1: the count in that half second; else a second;
`stats9.py`):

| side | opening | 0.5–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–7 | 7–10 | 10–20 | 20–60 s |
|---|---|---|---|---|---|---|---|---|---|---|
| clanrats charging v attacking swordsmen (30 lanes) | 17.4 | 2.0 | 9.5 | 5.7 | 6.6 | 6.6 | 6.5 | 5.8 | 5.8 | 5.3 |
| … of them first swings of men | 17.4 | 2.0 | 9.5 | 5.5 | 3.8 | 3.6 | 2.6 | 1.3 | 0.8 | 0.5 |
| clanrats charging v attacking spearmen (30) | 17.5 | 0.4 | 7.6 | 6.0 | 6.2 | 7.5 | 6.5 | 6.1 | 6.0 | 5.4 |
| clanrats charging v held swordsmen (6) | 16.3 | 0.7 | 6.8 | 3.3 | 6.3 | 8.2 | 3.9 | 5.7 | 5.2 | 5.2 |
| clanrats charging v held spearmen (6) | 15.7 | 2.0 | 6.3 | 5.5 | 7.2 | 6.2 | 6.1 | 5.3 | 5.6 | 5.1 |
| swordsmen attacking v charging clanrats (30) | 19.4 | 2.5 | 11.6 | 4.9 | 5.1 | 5.2 | 5.8 | 5.4 | 5.0 | 4.6 |
| spearmen attacking v charging clanrats (30) | 19.7 | 1.4 | 9.6 | 6.7 | 5.0 | 3.5 | 6.1 | 5.5 | 4.8 | 4.5 |
| swordsmen holding (6) | 15.2 | 1.4 | 3.0 | 2.7 | 4.0 | 5.5 | 3.2 | 3.0 | 2.4 | 2.4 |
| spearmen holding (6) | 14.3 | 1.4 | 3.2 | 2.3 | 1.7 | 0.8 | 4.4 | 2.8 | 2.2 | 2.2 |

### 2.5 The men within reach are not the men who strike — A (counts), C (why)
Men within 3.15 m of an enemy (the reach of the simulator's `melee.wave`), mutual charge: 49–74 at 0.5–1.5 s, 70–85 at
1.5–3.5 s, 52–72 at 4.5–8 s, 45–58 at 8–10 s, 29–37 at 14–20 s, 20–27 at 20–60 s — 2.3–3.3 times the steady number, as
the wave says. Swings a second (clanrats): 6.5 at 0.5–4.5 s (4.1–8.1 by second and cell), 6.0–6.5 at 4.5–14 s, 5.3–5.8
at 14–60 s — 1.15–1.25 times. Distinct men who have swung: 17.5 (opening), 32 at 2.5 s, 40 at 4.5 s, 51 at 10 s, 63 at
30 s: of 70–85 men within reach at 2–4 s half have not swung by 4.5 s (`stats5.py`). C: they have no fight until
touched (2.4), and their own first rank stands between them and the enemy.
HP a second follows swings × the rule: clanrats on swordsmen 68–74 HP/s for the swings of 0.5–2.5 s, 45–67 for 2.5–10 s,
45 for 10–14 s, 25–32 for 14–60 s; the rule's HP a swing falls 10.7 → 5.6 as the charge runs out, the measured 8.4–10.8
→ 4.8–6.2.

### 2.6 Against a formation that stood, fewer men join — A (counts), C (why)
Attacking unit, by what its enemy did at the contact (`stats10.py`, `stats11.py`):

| attacker | enemy | lanes | opening swings | first swings of men 0.5–5 s | swings 0.5–10 s | swings a second 20–60 s |
|---|---|---|---|---|---|---|
| clanrats | charged | 60 | 17.4 ± 0.1 | 23.2 ± 0.5 | 59.9 ± 0.7 | 5.4 |
| clanrats | stood | 12 | 16.0 ± 0.4 | 16.9 ± 0.9 | 52.8 ± 1.4 | 5.1 |
| spearmen | charged | 30 | 19.7 ± 0.2 | 25.7 ± 0.8 | 54.9 ± 1.1 | 4.5 |
| spearmen | stood | 9 | 17.1 ± 0.4 | 15.7 ± 1.1 | 41.0 ± 1.4 | 4.1 |
| swordsmen | charged | 30 | 19.4 ± 0.5 | 24.1 ± 0.8 | 57.1 ± 1.1 | 4.6 |
| swordsmen | stood | 9 | 17.2 ± 0.5 | 13.8 ± 0.8 | 41.7 ± 1.3 | 4.3 |

Joiners against a standing formation over against a charging one: 0.73 ± 0.04, 0.61 ± 0.05, 0.57 ± 0.04. Clanrats within
2.0 m of an enemy at 1.5–2.5 s: 50.8 ± 0.4 when both ran, 20.8 ± 1.6 on a holder. HP the clanrats deal: 668 ± 13 by
t0 + 10 s and 2162 ± 23 by t0 + 60 s to a counter-charger, 590 ± 27 and 1979 ± 34 to a holder (−78 ± 30, −182 ± 41).
The standing unit's own swings are 0.47–0.56 of an attacker's at 0.5–10 s and 0.45–0.52 at 20–60 s (`contact.hold_rate`
0.5 holds from the first second; standing clanrats 0.39–0.44).

### 2.7 Hit reactions and knocks — A (the rules as read), their part measured
`14306ee40` (per hit that does not knock): no reaction in action states 4, 6, 7, 0xc, 0xf, 0x10, 0x11 (and in 0xe, 0x12
unless the hit's flag +0x14 is set); a roll 0–100 against the man's `hit_reactions_ignore_chance` (slot 0x3c8); with the
flag: no more than `melee_max_hit_reactions` 2 (kv +0x2df8) since his last full swing (+0xa54) and attacker's mass × the
man's 1 / mass ≥ `interrupt_mass_ratio_clamp_hit_reaction` 0.4 (kv +0x26a8); without it: none for a man at a run or in his charge (gait +0x270 = 1, 2) whose +0x18f is 0; none
inside his own attack window (`14305ae80`) nor when `14306e278` holds. A reaction sets state 0x12 and replaces the
man's action (`14324d8f8`): his fight is gone and `14306e89c` takes no command until the state is 0 again.
Memory (`stats6.py`): state 0x12 lasts 2.1–2.5 s (median). A hit that does not kill, on an Empire man from a charging
clanrat: reaction 0.38–0.44, knocked back 0.16–0.20, down 0.13–0.16 (the code's `142e2dc18` at full charge: 0.255, 0.15);
from a standing one 0.58, 0.05, 0.06 (code 0.095, 0.05). On a clanrat inside his own swing or running in his charge:
0.02–0.08 / 0.00–0.02 / 0.02–0.07; otherwise 0.41–0.53 / 0.03–0.19 / 0.03–0.14. Men in a reaction: Empire 0.6–0.7 before
+0.5 s, 5–6 at 0.5–1.5 s, 4–9 at 1.5–5 s; clanrats 0.03–0.14, 0.2–1.0, 1.7–7.5.
- Not the lull: at 0.5–1.5 s at most 6 of the 32–73 Empire men within reach are in a reaction and 0.2–1.0 of the
  clanrats, yet both sides stop swinging alike; none of the second-wave men was hit before his swing.
- The opening: 0.0–0.2 of the clanrats' opening swingers a lane are knocked or staggered before their blow would land,
  2.0–5.7 of the Empire's 14–20 — it takes the Empire side's opening hits to 0.55–0.92 of the rule (2.3) and leaves the
  clanrats' at 0.98.

## 3. The four questions
HP lost a second, clock t0, pooled (`stats7.py`); "opening" = everything before t0 + 1 s:

| lanes | side | opening | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 | 9–10 | 1–10 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| swordsmen attack, clanrats charge (30) | ours | 164 ± 10 | 23 | 98 | 64 | 68 | 60 | 53 | 59 | 36 | 49 | 510 ± 20 |
| | clanrats | 364 ± 18 | 50 | 168 | 61 | 86 | 58 | 82 | 73 | 71 | 69 | 719 ± 26 |
| swordsmen hold (6) | ours | 180 ± 21 | 18 | 72 | 32 | 78 | 70 | 29 | 24 | 25 | 75 | 422 ± 35 |
| | clanrats | 152 ± 23 | 16 | 16 | 24 | 60 | 34 | 37 | 18 | 34 | 22 | 261 ± 26 |
| spearmen attack (30) | ours | 191 ± 13 | 7 | 77 | 59 | 63 | 69 | 56 | 61 | 46 | 43 | 480 ± 20 |
| | clanrats | 150 ± 10 | 16 | 80 | 39 | 27 | 26 | 62 | 31 | 35 | 45 | 362 ± 16 |
| spearmen hold (6) | ours | 130 ± 20 | 65 | 72 | 33 | 73 | 54 | 23 | 46 | 44 | 45 | 456 ± 49 |
| | clanrats | 121 ± 43 | 7 | 36 | 22 | 17 | 10 | 49 | 28 | 9 | 17 | 195 ± 29 |
| twin now (its clock: the contact) | ours, swordsmen / spearmen | 222 / 213 | 67 | 63 | 59 | 68 | 75 | 70 | 65 | 61 | 51 | 579 |

1. **(i) the first second.** The opening is the rule's: the clanrats' 17.4 swings give 173 HP (the 72 lanes where they
   charge) against 17.4 × 0.35 × 29.8 = 181; the twin's first strike is 176–187. The twin's "first second" 212–222 adds
   35 HP of the rate it restarts at 0.5 s; the game's opening lands over ~1.2 s round t0, and `tmp/simfix2/prof.py`
   cuts it with the script's melee flag (0.3 s samples) — its 78–189 are parts of the opening. Against a standing unit
   the opening is 8 % fewer swings (2.6): 180 ± 21 HP on held swordsmen, 130 ± 20 on held spearmen.
2. **(ii) the lull.** 2.1 + 2.4: the opening's men are out for their interval; the others start a median 1.6 s after t0
   (p10 1.0) and their blows land 0.9–1.1 s after the start. Both sides, whatever the order. HP at 1–2 s: 7–23 against
   72–98 at 2–3 s (on held spearmen 65: the spear lands 0.2 s sooner than the blade and sets t0, so the clanrats'
   opening spills into 1–2 s).
3. **(iii) 2–10 s.** With the clock at t0 the game is not lower at 2–5 s (98, 64, 68 against the twin's 63, 59, 68). The
   twin is higher at 1–2 s (67 against 7–23) and at 5–10 s (322 against 257–275 in the attack lanes, 212–223 in the
   hold lanes): it lets all the opening's men back at the bare interval and still multiplies the men striking by the
   wave's 1.9–1.6, while in the game the swings stay flat (2.5), the opening's men return over interval … interval +
   4 s (2.1) and fewer men join against a holder (2.6). Over 1–10 s: 486 ± 12 (72 lanes) against 579. The rates meet
   after ~10 s because the charge bonus is spent, not because a wave of strikers has ebbed.
4. **(iv) held spearmen.** Their answer is 14.3 swings (2.3); its hits are 0.55 ± 0.13 of the rule (17 in 6 lanes for
   31) — 121 ± 43 HP against the rule's 221 and the twin's 152–175; 2.3 of the 14.3 swingers are knocked or staggered
   first. After it they swing at their steady hold rate (2.7, 3.2, 2.3, 1.7, 0.8 a second at 0.5–5 s against 2.2 at
   20–60 s), each blow 42–44 HP while the enemy's charge is above 0.7: 7, 36, 22, 17 HP/s. The twin multiplies the held
   rate by the wave (46, 43, 41 HP/s at 1–4 s). At 4–5 s they hardly swing (0.8 a second): the opening's men are still
   inside their 5.7 s interval.

## 4. Against the simulator (`sim/melee.py` strikes, `sim/config/sim.json`)
| item | game | simulator | verdict |
|---|---|---|---|
| a charger's first strike | its first rank, one blow each, the rule's chance | `first_strike_rank`: the files in contact | right (176–187 against 164–191) |
| a standing unit's first strike | the men the chargers aim at: 14–15 of 20 files | `stand_first_strike`: the steady front | swordsmen 120–135 against 152 ± 23 HP; spearmen 152–175 against 121 ± 43 (their hits fall short of the rule) |
| when the rest of the rate starts | first swings +1.0 s after the first damage, landing +1 s later | `swing_interval`: `miss_s` 0.5 s after the contact | **0.5 s against ~1.7 s** |
| who is back before the interval | men who had not swung (2.4); nobody who missed (2.1) | the share 1 − p "who missed" | close for clanrats and spearmen (1 − p ≈ 0.6); low for swordsmen (0.38: 196 HP at 1–4 s against 279) |
| the opening's men's next blow | no sooner than the interval, median 1.2–1.65 × interval | all at the interval | the twin steps up at 4.3 s (59 → 68 → 75 HP/s), the game does not |
| men striking at 2–10 s | swings 1.15–1.3 × the steady rate | `wave`: × 2.2 → 1.6 | too many, offset by the weak answer of `p / (p × interval + miss_s)` to the charge's attack bonus (× 1.06 where a swing an interval gives × 1.23) |
| the enemy stood at the contact | 0.57–0.73 of the joiners, −12 % swings in 0.5–10 s | the same wave | **the holder and the counter-charger lose the same** |

The twin with one switch changed, HP our unit loses in 0–10 s (game 668 ± 13 attacking, 590 ± 27 holding): now 802;
`melee.wave.amp` 0 → 513; wave 0.35 / 8 s → 571; `melee.cycle.on` 1 → 503; `melee.swing_interval` 0 → 962. The wave and
the flat chance law hold each other up: neither can be changed alone.

## 5. Rule for the simulator (two switches; tried on a patched copy, `tmp/charge_impact/twin.py`)
1. `melee.swing_join {s: 1.7}` (melee.py, the `back` factor of `swing_interval` / `stand_interval`; absent: `miss_s`):
   after a first strike the rest of a unit's rate starts `s` after the contact. `s` = 1.0 (first swings of the men who
   did not swing in the opening, p10 after the first damage, 2.4) + 1.0 (mark → the target's HP, 2.2) − 0.3 (the
   opening's own blows land a mean 0.3 s after the first damage). The share 1 − p and the return at the interval stay.
2. `melee.wave.stand_k` 0.73 (melee.py wave; absent: 1): the wave's amplitude × `stand_k` for the men striking a
   formation that stood at the contact (`stood_in`); the standing formation's own wave stays (its swings keep 0.5 of
   an attacker's early and late). 0.73 ± 0.04 = the clanrats' joiners against a holder over against a counter-charger
   (2.6); spearmen 0.61 ± 0.05, swordsmen 0.57 ± 0.04 against standing clanrats — pooled 0.64.

Twin on the game's lanes (`run_hold`, `run_c60`, `stand_c45`; 3 lanes a cell, 4 copies; `twin_base60.log`, `twin_join60.log`,
`twin_joinstandT073b.log`, `twin_joinstandT064.log`). HP a second our unit loses to charging clanrats, from the contact:

| | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 | 9–10 | 0–10 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| game, swordsmen hold (t0 clock, opening in 0–1) | 180 | 18 | 72 | 32 | 78 | 70 | 29 | 24 | 25 | 75 | 603 ± 39 |
| game, swordsmen attack | 164 | 23 | 98 | 64 | 68 | 60 | 53 | 59 | 36 | 49 | 674 ± 19 |
| game, spearmen hold | 130 | 65 | 72 | 33 | 73 | 54 | 23 | 46 | 44 | 45 | 585 ± 39 |
| game, spearmen attack | 191 | 7 | 77 | 59 | 63 | 69 | 56 | 61 | 46 | 43 | 672 ± 19 |
| twin now, swordsmen (hold = attack) | 222 | 67 | 63 | 59 | 68 | 75 | 70 | 65 | 61 | 51 | 802 |
| twin, 1: attack lanes (swordsmen / spearmen 176 first) | 187 | 0 | 63 | 59 | 68 | 75 | 70 | 65 | 61 | 57 | 704 / 688 |
| twin, 1 + 2 (0.73): hold lanes | 187 | 0 | 54 | 51 | 59 | 66 | 62 | 58 | 54 | 51 | 642 / 626 |
| twin, 1 + 2 (0.64): hold lanes | 187 | 0 | 51 | 48 | 56 | 63 | 59 | 56 | 52 | 49 | 621 / 605 |

Shares of the starting health lost in 60 s, ours / theirs / trade (game: `meetfar_game.json`, attack and stand = the
mean of the cell's lanes):

| cell | game | twin now | 1 | 1 + 2 (0.73) | 1 + 2 (0.64) |
|---|---|---|---|---|---|
| swordsmen hold | 0.239 / 0.156 / −0.083 | 0.276 / 0.151 / −0.125 | 0.265 / 0.147 / −0.118 | 0.253 / 0.147 / −0.107 | 0.249 / 0.147 / −0.103 |
| swordsmen attack | 0.264 / 0.307 / +0.042 | 0.277 / 0.315 / +0.039 | 0.266 / 0.306 / +0.041 | = | = |
| spearmen hold | 0.243 / 0.105 / −0.138 | 0.270 / 0.117 / −0.152 | 0.259 / 0.111 / −0.148 | 0.247 / 0.111 / −0.136 | 0.243 / 0.111 / −0.133 |
| spearmen attack | 0.253 / 0.192 / −0.061 | 0.271 / 0.202 / −0.068 | 0.259 / 0.195 / −0.064 | = | = |
| swordsmen on standing clanrats | 0.099 / 0.272 / +0.174 | 0.123 / 0.331 / +0.208 | 0.120 / 0.322 / +0.202 | 0.120 / 0.314 / +0.193 | 0.120 / 0.311 / +0.191 |
| spearmen on standing clanrats | 0.087 / 0.167 / +0.080 | 0.129 / 0.211 / +0.082 | 0.125 / 0.203 / +0.078 | 0.125 / 0.194 / +0.069 | 0.125 / 0.191 / +0.066 |
| meet − hold, swordsmen / spearmen | 0.125 / 0.077 | 0.164 / 0.084 | 0.159 / 0.084 | 0.148 / 0.072 | 0.144 / 0.069 |

What they do not mend: the share 1 − p (swordsmen's blows on charging clanrats in 0–10 s: game 1079 ± 21, now 1054, with
1: 967 — their second wave of 11.6 swings is cut to 0.38); the opening's men all back at the bare interval (tried: back at
1.3 × interval takes our loss in 0–10 s to 669 / 655 but the swordsmen's blows to 903); our losses to standing clanrats
(0.120–0.125 against 0.087–0.099); a held unit's own wave (iv). A swing an interval a man (`melee.cycle`) is the frame
these belong to; with it alone the twin falls to 503.

## 6. Not found
- The animation's hit time and reach (2.2): animation data; only the measured 0.9–1.1 s and start distances.
- Why state 1 comes a median 0.4–0.5 s after the bodies are within 1.35 m (when `14311b320` / `14311b428` fire).
- Why clanrats inside their swing or charge are almost never staggered or knocked and Empire men are (2.7): the hit's
  flag +0x14, the man's +0x18f, or the swing's push level (`14307fb18`, animation data).
- Why spearmen's opening hits are 0.55–0.75 of the rule beyond the swingers knocked first.
- Whether 2.4–2.6 hold for large bodies: riders and monsters use another attack kind (`14325f8d4`, `143260464`: a start
  test every 5th tick, 20° facing) and knock men down.

## 7. Probes that would decide what is left (not run)
1. **Stand_k by unit and the held answer** (2.6 has 12 hold lanes; the 0.55 ± 0.13): 2 battles × 15 lanes, clanrats and
   stormvermin at a run on held spearmen, swordsmen, halberdiers and the same counter-charging at 60 m (hold : attack
   2 : 1), memory rows every 0.1 s from 60 m to 10 s after contact, every 0.5 s to 60 s, fearless. Numbers: first swings
   of men 0.5–5 s and swings 0.5–10 s, holder over counter-charger (this note: 0.73 and 0.88; "no difference" predicts
   1.0); hits by t0 + 1.3 s over answer swings × chance, and over the swingers not in state 0x12 / 0xe / 0xf before
   mark + 0.9 s ("knocked first" predicts 1.0 for the second; a lower chance of an answer blow predicts both under 1;
   noise predicts 1.0 for the first; 20 hold lanes give ± 0.07).
2. **Swings against men within reach in the wave's own probes**: one battle each of the plans `wave` and `recontact`
   with `--mem`. Numbers: swings a second at 2–10 s over 20–60 s, and men within 3.15 m likewise. This note predicts
   1.15–1.3 and 2.3–3.3; "the wave's men strike" predicts equal ratios.
