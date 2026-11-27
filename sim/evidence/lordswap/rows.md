# lordswap: a new order to a foot lord in melee (27.11.2026)

Game 9.0, "Normal", `game/probe/charge_probe.py` plan `lordswap`, memreader per man every 0.1 s and the lord's unit field
+0x3254 every 0.5 s (sites file with `brk`; reverse/notes/melee_turn.md 2.2), x10, 6 battles
`build/charge-probe/runs/20261127-174744 … -175529` (out of git). A lane: the lord attacks infantry E1 (it attacks him at a
run from 3 m), a second unit E2 of the same kind 1 m beside E1 attacks him at a run at the go; both touch him. 10 s after
the first contact: X - an attack order on E2 (at a walk), S - the same attack on E1 again, C - nothing. General with
clanrat spearmen, Warlord with Empire swordsmen (14 m fronts); battles 1-4 X/C swapped by lord, 5-6 S. Reader
`tmp/lordwin/swap.py` -> `swap.json`. A strike = his man's strike mark takes a new value; its unit = his target man's.

| lane | first strike at E1 after the order, s | at E2, s | his strikes 0-25 s E1 / E2 | dealt HP 25 s | taken HP | +0x3254 after |
|---|---|---|---|---|---|---|
| X General b1 | 26.5 | 2.8 | 0 / 5 | 540 | 153 | 96 |
| X Warlord b2 | none (to the end, ~40 s) | 4.2 | 0 / 4 | 692 | 322 | 99 |
| X General b3 | none (~40 s) | 2.5 | 0 / 5 | 840 | 157 | 96 |
| X Warlord b4 | 25.9 | 4.7 | 0 / 4 | 623 | 380 | 99 |
| S General b5 | 5.2 | none | 3 / 0 | 540 | 184 | 96 |
| S Warlord b5 | 4.9 | 31.7 | 3 / 0 | 552 | 319 | 99 |
| S General b6 | 1.8 | none | 5 / 0 | 658 | 261 | 96 |
| S Warlord b6 | 4.7 | none | 5 / 0 | 618 | 292 | 96 |
| C Warlord b1 | 3.2 | 16.2 | 3 / 2 | 621 | 255 | 0 |
| C General b2 | 5.9 | 0.3 | 5 / 1 | 660 | 194 | 0 |
| C Warlord b3 | 4.9 | 38.2 | 3 / 0 | 274 | 216 | 0 |
| C General b4 | 3.8 | 35.4 | 4 / 0 | 720 | 140 | 0 |

Means: dealt / taken HP in 25 s - X 674 / 253, S 592 / 264, C 569 / 201; his strike starts 0.16-0.2 a second.

Reading: a lord pays the infantry's break-off window. After a new attack order he starts no strike at the unit that
touched him at the order, the order's own target excepted, for ~25 s (24 s of the window + ~1-2 s to the swing): X never
struck the old target before 25.9 s; S (the same attack again, given by the script) never struck the other touching unit
for 25 s (31.7 s once). The order's target he strikes at once (X: 2.5-4.7 s; S: 1.8-5.2 s). +0x3254 reads 96-99 at the
poll after an X or S order and 0 under C: the script's repeated attack opens the window as a new target does. His damage
does not drop: he strikes the order's target, and without an order he chooses it anyway (C: 3 of 19 strikes at E2 in
25 s). Known: one lord a side a battle (2 lanes), 4 lanes an arm, one infantry kind per lord, the 24 s pair timer (+0xc of
the enemy's record) not read - inferred from the strikes.
