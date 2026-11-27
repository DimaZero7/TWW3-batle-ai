# lordring: a foot lord against infantry, per man (27.11.2026)

Game 9.0, "Normal", `game/probe/charge_probe.py` plan `lordring`, memreader per man every 0.1 s, x10, 6 battles
`build/charge-probe/runs/20261127-133653 … -134118` (out of git). Each battle: General v clanrat spearmen and Warlord v
swordsmen. A: the lord runs into a held unit; B: the unit runs onto the held lord; C: four units round the held lord.
Rows `ring_rows.json` (reader `tmp/lord0b/ring.py`); simulator twins of the same lanes (8 copies) `lanes_<switch>.json`
(`tmp/lord0b/lanes2.py`). HP lost a second from the probe's contact.

| arm (2 lanes x 2) | men in reach 3.85 m | strikers (swings x 4.3 s) | hits flank+rear | lord HP/s 0-5 / 10-15 from touch |
|---|---|---|---|---|
| A | 10.0 | 7.8 | 0.86 | 0.9 / 7.2 |
| B | 13.6 | 8.8 | 0.80 | 6.8 / 4.7 |
| C | 20.9 | 10.2 | 0.80 | 20.5 / 11.5 |

The lord's own blows: 0.16-0.18 swings a second, 1.6-1.9 bodies per swing that hits, each body loses ~60-64 HP (a whole
man), 0.45-0.64 deaths a swing.

Lord HP lost a second 15-30 s, simulator / game:

| lane | game | off | ring sides | sides + 9.5 | strikers 9.5 |
|---|---|---|---|---|---|
| A General | 6.57 | 3.15 | 5.54 | 7.67 | 4.36 |
| A Warlord | 9.47 | 4.89 | 7.40 | 10.82 | 7.15 |
| B General | 8.73 | 6.55 | 11.38 | 15.75 | 9.06 |
| B Warlord | 8.10 | 9.32 | 15.14 | 22.12 | 13.63 |
| C General | 8.37 | 6.55 | 11.38 | 16.63 | 9.57 |
| C Warlord | 14.23 | 9.34 | 15.17 | 22.17 | 13.65 |

Lord kills a second 15-30 s: game 0.27-0.50, simulator 0.41-0.42 in every arm and switch.
