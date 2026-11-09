# Open melee items: causes and rule specs (main 4d4099b, sim 6f5d7e38b1c8)

Analysis only, nothing tracked was changed. Scripts and data: `build/open_melee/` (variants.py patches the sim in
memory; killpool*.py and contact_lag.py read the game's probe recordings on the host). Game data: the melee probe
`build/meleetests/runs` (charge 8 battles, hit 2), T-E1 `build/effectsimpl/runs`, the pair recordings
`build/nn-arena/runs/20260930-1821*`. Error measure below: mean |log(sim / game)| over the probe's cells (lower is better).

## Summary of the prototype (all rules R1-R6 together, probe twins, 4 copies)

| Window | base | R1-R6 |
|---|---:|---:|
| struck side (the unit charged), HP lost 0-1 s, 15 cells | 1.05 | 0.41 |
| struck side, 0-5 s, 20 cells | 0.61 (median 0.37) | 0.47 (median 0.17) |
| target 5-15 s | 0.335 | 0.304 |
| target steady (15 s on) | 0.186 | 0.166 |
| after a re-attack, 5 s, 2 lanes | 0.33 | 0.24 |
| mechanics check (sim.check, 54 numbers) | 24 | 24 (no pair changed winner; clanrat charge loss 406 -> 459, game 764) |

## R1 — the struck side strikes at once (items 4, 6)

- Cause: the sim gives the first strike only to a unit that arrives moving or charges. In the game every man whose
  interval has run out strikes as soon as an enemy is in reach (CA Feature Focus #2: the interval runs after a hit),
  so a standing unit that is reached strikes at once too. Measured (cp.measure window from the sample before
  contact): the charger loses in the first second 106-194 HP against braced spearmen, 120 against swordsmen, 60-131
  against units given an attack order at contact; 0 when the target faces away (rear lanes).
- Rule (tools/nn/sim/battle.py, first-strike block): `stand = fresh & (speed <= 0.3) & (order HOLD)`; a pair i->j
  gets the swing when `first_i` or (`stand_i` and |rel_i(j)| <= contact.front_deg 60). The swing (melee.py return)
  is taken before hold_rate (every man in reach swings; hold_rate is about the men who step in later).
- Test: a held unit charged frontally loses-to-it F x p x hit in the contact step; charged from behind: 0.

## R2 — braced reflection is not cut by hold_rate (item 4)

- Cause: in 1-5 s after a charge, braced spearmen (charge_reflection, DB x2 for 3.9 s) strike 52 HP/s = the full
  rule x2 (46); held swordsmen (no reflection) 12 HP/s (hold rate). The sim applies hold_rate 0.5 to the reflecting
  blows too. Community (Steam): braced models strike whatever is adjacent; during the charge every front man is.
- Rule (melee.py, hold_rate line): `held & (reflect <= 1)`.
- Effect R1+R2: chargers' first 5 s against braced spearmen 87 -> 236 HP (game 245 / 314).

## R3 — a unit under a move order still in contact strikes like a held one (item 9)

- Cause: blocked by the enemy, its men do not step into gaps, like a held unit. Probe move lanes (4) against the same
  unit attacking (4): target steady 20.4 vs 29.2 HP/s (x0.70), men within 2.5 m 9.0 vs 12.3 (x0.73).
- Rule (melee.py `held`): HOLD or MOVE (a MOVE that is leaving melee already strikes nobody), formations only.
  Uses the existing contact.hold_rate, no new number.
- Effect: target 5-15 s 345 -> 173 (game 180), steady 34.5 -> 17.3 (game 20.4).
- Caveat: CA's planner units with a far point are replayed as attacking (FIGHT_NEAREST) - unchanged.

## R4 + R5 — the "recharge" was a chased leaver (item 5)

- Cause: in the probe's 4 recharge lanes the swordsmen never left melee: nearest soldiers 0.8-1.8 m apart, engine
  melee flag on for all 25 s (build/open_melee/recharge_gap.py); the "second contact" is the probe's 25 s timeout,
  just after the DB's melee_breakoff_secs 24 dropped their order. The sim releases them at pin_melee_s 20 s, they get
  out for 2-5 s, then re-charge with a fresh first strike: 341 HP in 5 s (game 250).
- R4 (battle.py `hold_for`): a melee unit leaving melee while an enemy in contact has an attack order on it is held
  until melee_breakoff_secs (24, DB), then the existing breakoff drops its order; not chased: pin_melee_s as now.
- R5 (battle.py, after strikes): a unit that starts striking again after not striking for >= its attack interval
  strikes once at once with every man in contact (same CA rule as R1). Needs a per-unit "seconds without striking".
- Effect: 341 -> 164 HP (game 250; R5 added nothing here, probably because its swing lands just before the probe's window - not checked). Warlord lane 143 -> 215 (game 204).

## R6 — run-up only towards the enemy (item 5, lords)

- Cause: the sim adds to `runup` any fast movement out of contact, also running away. In the game the Warlord that
  ran ~10 m off came back at 1.5 m/s and struck 3.5 s after his melee flag, one ordinary blow (204 HP in 5 s).
- Rule (battle.py `fast`): speed component towards the attack target, else towards the nearest enemy.
- Low priority: in the probe the sim no longer bursts there (base 143); it is a correctness fix.

## R7 — wounded pool instead of kills.exponent (item 2)

- Cause: the game keeps per-model HP and overkill is lost (CA). In the probe the HP missing among the living
  (W = men x HP a man - unit HP) saturates: flagellants -> clanrats 4-7 men's worth while 87 men die; slaves 10-20.
  Greatswords -> slaves: W = 0 while the charge bonus lets one blow kill (49 + charge >= 50 HP), then 13-20 once it
  fades (49 < 50) - the damage rule is right to the point. The sim's rule keeps (1 - E[N]^-0.5) of all HP lost as
  wounds forever, so in a long fight it kills too few men.
- Rule (battle.py "damage, men, kills"; new state field `wound`, version change): per unit j, cap
  W* = F_j(own men in contact, the existing count) x hp_man x (1 - 1/E[N]), E[N] = hp_man / hit (the HP-weighted
  hit of its strikers); with D = melee HP lost this step: kills = max(0, min(D, W + D - W*)) / hp_man,
  W += D - kills x hp_man; floor/ceiling as now; lords unchanged; missiles keep the current rule (no data here).
- Evidence on the game's own HP series (killpool_tgF.txt): men lost at 15-30 s: current 1.00 of the game (|log| 0.24),
  pool 1.10 (0.20); at 60-90 s: current 0.82 (0.30), pool 1.15 (0.23).
- Slaves 0.89 vs 0.51 men/s: the pool covers about half; the rest is HP rate (swords -> slaves 39 vs 46 HP/s,
  greatswords 41 vs 63) - open (men in reach: the fitted fighting_files 0.5).

## Item 7 — closed: a measuring artefact

The T-E1 contact was the flagellants' own melee flag. In the win lane the slaves lose HP and get their flag at -3.0 s,
the flagellants' flag comes on at 0.0 when the slaves (facing away) start to fight back; Penitent fires at -3.0, the
true first blow. Over 41 probe lanes the struck side's flag = its first HP loss; the attacker's flag lags 2-4.5 s
when the target does not fight back (rear, move, lords). So the game's is_in_melee = "enemy men attack this unit".
Optional (observation only): sim `m` for the network = an enemy in contact strikes it.

## Item 10 — closed

General / Warlord first blow lands in the contact sample (60 / 118 / 205 / 272 HP), then every 4.0-4.5 s.

## Item 3 — not settled; probe P1

Game: 5-15 s (SYG on, after the first strike) 62 vs 255 HP = x0.24, about 4 vs 15 blows: 95 % range x0.08-0.76;
the rule's x0.62 and the hit-chance-only x0.32 are both inside. The first strike was not cut (94 vs 73 HP): the
ability lands after its cast. P1: plan 'hit' L5 (clanrat spearmen -> Empire spearmen, a General 8 m behind), 2
battles x 4 lanes, SYG cast when the attacker is 25 m away in 2 lanes per battle (swap in battle 2); measure the
target's HP 1-18 s after contact. ~19 blows a lane without SYG: log-ratio SE ~0.24, 0.32 vs 0.62 = 2.7 SE. ~6 min.

## Item 8 — open; probe P2

Both sides deal x1.4-2 the rule in fights with flagellants: hit plan flag -> clanrat 53 vs 39 HP/s, clanrat -> flag
51 vs 31; per man within 2.5 m x2.3 / x2.0 of the swordsmen fights. T-E1: flagellants end at 0.13 / 0.47, sim 0.41 /
0.66; removing Penitent's stats in the sim gives 0.31 / 0.59 (a third of the gap);
their output against stormvermin matches (0.897 / 0.893). Candidates: matched combat formed-vs-free 60 % vs
formed-vs-formed 40 % (DB), hit reactions, the hit-rate law at high hit chance. P2: 1 battle x 4 lanes against
fearless clanrats with soldiers' places: flagellants attack, swordsmen attack, militia attack, flagellants held;
split intake by Penitent on/off samples. ~4 min.

## Item 1 — cause not found; probe P3

The pair's "held" spearmen are CA's planner's unit under a far move order (ox 55-60), moving 63-100 % of the fight,
not held. The replay makes them attack, so in the sim they tire (fatigue 4-5) while in the game they stay at 0-2.
Even so the sim's spearmen win: from 60 to 120 s the game's exchange is ~2x the sim's on both sides (spearmen 40 vs
19 HP/s, clanrat spearmen 34 vs 21); the charge phase too (clanrats lose 620-995 vs 405). R1-R6 do not flip it.
P3: 1 battle x 4 lanes, 240 s, clanrat spearmen attack Empire spearmen from 80 m: target held / far move point
(planner-like) / attack at contact / held; soldiers' places, fatigue, morale; measures per 30 s window: HP/s both
ways, men in reach, attackers beside or behind the target's edges. ~5 min.
