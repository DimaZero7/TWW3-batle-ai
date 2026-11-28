# ai_approach: how the game's AI comes at a standing line, 5 v 5 (27-28.11.2026)

Switch `ai_game.approach` (sim/config/sim.json; `on` 0 = off, the old ai_game bit for bit), sim/scripts.py `game_line`,
`_march_round`; read from the config the battle runs with (`State.cfg`: set by `scenario.build`, then by every
`battle.step` / `battle.run` from its own params; a clone gets its own copy).

## The game

64 battles: lane S0-1 price (tmp/drill_s1/price/runs.txt, arms crest / centre) and lane T2 (tmp/lordline/t2/runs.txt,
arms T / Q, our line on the crest); game 9.0, "Normal", ksl_plains_01, 16 starts, 4 infantry + a lord a side, the AI
attacks from the east, our side stands (our infantry 3-24 m off its place at contact). Reader
`tmp/simfix/n27/game_approach.py`, raw rows `tmp/simfix/n27/game_approach.out` (nn_sample every model second, so
every time below is a whole second; AI infantry, men > 1).

| arm | n | first AI unit moves > 5 m, s (min / med / max) | contact s | AI advance at contact m | AI speed m/s | units' own advance spread at 30 / 60 / 90 s, median m |
|---|---|---|---|---|---|---|
| crest | 16 | 2 / 3 / 3 | 101 / 118 / 129 | 338 | 2.9 | 64 / 42 / 65 |
| centre | 16 | 2 / 3 / 3 | 96 / 108.5 / 162 | 322 | 3.0 | 49 / 17 / 33 |
| T | 16 | 2 / 3 / 3 | 97 / 119 / 137 | 341 | 2.9 | 67 / 42 / 62 |
| Q | 16 | 2 / 3 / 3 | 99 / 123 / 136 | 344 | 2.9 | 63 / 33 / 53 |

Each AI infantry unit's own first second > 5 m from its start (the raw rows' `unit_starts`, 256 units): 2 s 7, 3 s 118,
4 s 96, 5 s 34, 6 s 1 (median 4); the first unit at 3 s in 57 battles, at 2 s in 7. The spread (max - min of the units'
own advance along the way, before contact) at 30 s: 33-128 m over all 64; in 8 battles (crest p03, p11, p12; T p01, p04,
p08; Q p03, p04) 187-227 m at 60 s and 231-340 m at 90 s, first contact in those 8 at 100-123 s (median 101; the
other 56: median 116): the front did not wait for the rearmost. In run 20261123-224717 the unit facing the blocked patch
went south round it and on, 100 m behind the others at 100 s.

The old ai_game's 60 s wait (GAME.start_wait_s) is the simulator's value from the owner's 20-v-20 battles M1-M3
(sim/evidence/diag_m1/night.md), not a game onset measured for these battles; its slack 15 m (a unit more than 15 m ahead
of the line's rearmost waits) is from the 20-v-20 pool.

## The cause of "never on the crest" (off)

CPU twin of p01 on the crest line (`tmp/simfix/n27/diag_b.py`, AI side every 10 s): after the 60 s wait the unit whose
30 m march point falls on the blocked patch (u11) gets step_around's point beside the way, on the other side at the next
call, and stands 0-6 m from its start for 300 s (blk_o 0 - no point refused); the march gate (slack 15 m) holds the
other three and the lord. On the centre line the same stall; contact came only by our side walking up (241 s).

## The change (on)

- start_wait_s 2, every unit of the side alike (shooters and the lord too, `Line.wait_all`), the advance only: a move
  towards the enemy or a melee attack waits; withdrawals, side steps and the shooters' fire orders pass; the fast units
  wait as long (fast_go_s is at least start_wait_s). Chosen by the sampled
  start, not the order time: the twin's AI units on the 16 crest starts, sampled per second as the game's
  (`tmp/aiappr/out/w0..w3.json`), first unit min / median / max and each unit's median: hold 0 s 2 / 2 / 2, 2; 1 s
  2 / 2 / 3, 3; **2 s 3 / 3 / 4, 4**; 3 s 4 / 4 / 5, 5 (game 2 / 3 / 3, 4). The simulator's units start 1-2 s after
  their order; the game's spread of unit starts (2-6 s) is wider than the twin's (3-5 s).
- The line's own march point (the infantry's advance at the enemy - not the lord's post, a shooter's post, a flanking
  point or a withdrawal, which keep the old search) on 'cannot stand' ground: the first free point inside the playable
  field straight on (x1.5 / 2 / 2.5 / 3 of the 30 m step; the detour walks round to it), then the old points beside the
  way (inside the field), then short of it (x0.75 / 0.25), none: the order in force goes on when it is a move /
  withdrawal to a free point in the field (as before, when the refused point left it in force), else the unit holds.
- slack_m none (nobody waits for the line's rearmost): the spread above.

## The simulator, before / after

Twin of the calibration (`tmp/aiappr/run.py`: tmp/simjudge/twin4's crest arm, the 16 game starts, ai_game on the enemy
side with its lord free, CPU, 240 s; t_c = the first second with one of our infantry units in melee; min / median / max):

| line | game | off | on, 16 x 8 |
|---|---|---|---|
| crest | 101 / 118 / 129, 16 of 16 | contact in 0 of 128 (600 s; tmp/simjudge/stage16.log) | 104 / 123 / 132, 128 of 128 |
| centre | 96 / 108.5 / 162, 16 of 16 | 223 / 242 / 260 (tmp/simfix/n27/A_ce.log) | 99 / 115.5 / 132, 128 of 128 in 90-170 s |

On, crest: first unit > 5 m at 3 / 3 / 4 s; refused orders 0; blocked march points re-pointed 15,227 unit-calls, of
them 0 to the short points and 0 held. By start the twin is two-valued: 104-117 s on 8 starts (p01, p04, p05, p08,
p09, p12, p13, p16; p12 alone at 115-117), 129-132 s on the other 8 (the starts where a unit walks round the patch); the
median 123 s is the mean of the 64th (117) and 65th (129) of the 128 - no battle has 123 s; the game's spread is
continuous (`tmp/aiappr/out/r5_crest.json`). Centre: 113 of 128 in 101-160 s (15 at
99-100 s, the game's own min 96). The same 16 starts on a flat field (16 x 1): off contact 161 / 173 / 188 s, first move
61 s; on 92 / 106 / 116 s, first move 3 / 3.5 / 4 s, no AI unit more than 36 m behind its infantry's median at contact.

Known: on crest starts p10 and p11 the AI lord sticks at the patch's north end from ~32 s (306-314 m behind his infantry at contact, 16 of 128 battles, he never fights before 240 s): his order is a free post behind his line, the detour's way is refused every step - the detour's, not this switch's (terrain.passable.detour.unstick, a separate patch);
on crest starts p07, p13, p15, p16 the lord's post (20 m behind the line's centroid) lies up to 13 m
ahead of the MEDIAN of his infantry for 6 % of his seconds before contact (one unit far behind pulls the centroid), never
ahead of the centroid; two AI units can merge into one place after one walks round the patch (friendly formations do
not avoid each other in the simulator); the twin's AI start rows are 48 m apart, the game's 38 m (net.stage0.specs
rows, not this switch); the game AI's single far point at 1 s is still modelled as 30 m steps re-given every call.
