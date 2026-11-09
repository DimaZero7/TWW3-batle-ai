## build/morale-hyp/sr_base

Windows of 10 s; 92 battles, 124124 unit-windows. HP lost sim/game 1.01 (0.98-1.04); rout starts sim/game 0.98 (0.91-1.06), game 2691, sim 2648.5.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.88 | 0.88 | 0.88 | 0.90 | 0.91 | 0.92 | 0.94 | 0.95 | 0.97 | 0.98 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 124124 | 92 | 1631.4 | 1648.7 | 1.01 | 0.98-1.04 | 1.08 | 1.04-1.12 | 1.00 |
| standing, not in melee, no contact in window | 51421 | 92 | 79.4 | 114.0 | 1.44 | 1.32-1.57 | 1.54 | 1.42-1.69 | 2.00 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 196.4 | 193.1 | 0.98 | 0.95-1.02 | 1.14 | 1.08-1.21 | -0.20 |
| contact begins in window, ran in | 9006 | 92 | 204.1 | 197.5 | 0.97 | 0.92-1.02 | 0.95 | 0.90-1.01 | -0.38 |
| contact begins in window, walked in | 327 | 83 | 8.9 | 7.4 | 0.83 | 0.73-0.95 | 0.82 | 0.71-0.95 | -0.09 |
| melee, first 10 s of the fight | 5739 | 92 | 194.2 | 176.8 | 0.91 | 0.88-0.95 | 0.98 | 0.94-1.03 | -1.01 |
| melee, 10-30 s | 8207 | 92 | 268.3 | 245.7 | 0.92 | 0.88-0.95 | 0.98 | 0.94-1.02 | -1.31 |
| melee, 30 s+ | 25884 | 92 | 656.5 | 669.3 | 1.02 | 0.98-1.06 | 1.10 | 1.06-1.14 | 0.74 |
| melee, sim touches 1 enemy | 14716 | 92 | 420.3 | 375.5 | 0.89 | 0.87-0.92 | 0.96 | 0.92-0.99 | -2.59 |
| melee, sim touches 2+ enemies | 19308 | 92 | 580.9 | 625.4 | 1.08 | 1.03-1.12 | 1.14 | 1.09-1.19 | 2.57 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 117.8 | 90.9 | 0.77 | 0.72-0.83 | 0.94 | 0.87-1.00 | -1.56 |
| melee, touched in front | 12723 | 92 | 319.3 | 305.7 | 0.96 | 0.93-0.99 | 1.03 | 0.99-1.07 | -0.79 |
| melee, touched in flank | 15606 | 92 | 479.3 | 492.0 | 1.03 | 0.99-1.07 | 1.09 | 1.05-1.14 | 0.74 |
| melee, touched in rear | 5695 | 92 | 202.6 | 203.2 | 1.00 | 0.95-1.05 | 1.04 | 0.99-1.10 | 0.04 |
| melee, health < 30 % | 4220 | 92 | 87.0 | 100.5 | 1.15 | 1.10-1.22 | 1.29 | 1.20-1.37 | 0.78 |
| melee, health >= 30 % | 35610 | 92 | 1032.0 | 991.3 | 0.96 | 0.93-0.99 | 1.03 | 0.99-1.07 | -2.35 |
| routing, in melee (pursued) | 3976 | 92 | 63.3 | 75.8 | 1.20 | 1.05-1.35 | 1.09 | 0.95-1.26 | 0.72 |
| routing, free | 16974 | 92 | 28.2 | 36.0 | 1.28 | 1.12-1.48 | 1.27 | 1.08-1.51 | 0.46 |
| standing, health < 30 % | 13319 | 92 | 118.5 | 133.2 | 1.12 | 1.07-1.18 | 1.25 | 1.18-1.33 | 0.85 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 9243 | 92 | 110.7 | 125.9 | 1.14 | 1.07-1.20 | 1.16 | 1.09-1.24 | 0.88 |
| missile | 28313 | 92 | 348.2 | 325.6 | 0.94 | 0.90-0.97 | 1.00 | 0.95-1.05 | -1.30 |
| infantry melee | 86568 | 92 | 1172.5 | 1197.1 | 1.02 | 0.98-1.06 | 1.09 | 1.05-1.14 | 1.42 |
| network's side | 60359 | 92 | 913.7 | 918.5 | 1.01 | 0.97-1.04 | 1.07 | 1.03-1.11 | 0.28 |
| the other side | 63765 | 92 | 717.7 | 730.2 | 1.02 | 0.96-1.08 | 1.09 | 1.03-1.16 | 0.72 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 3097 | 92 | 77.8 | 84.4 | 1.08 | 1.02-1.14 | 1.09 | 1.02-1.17 | 0.38 |
| lord, routing | 747 | 69 | 2.5 | 2.8 | 1.11 | 0.83-1.43 | 1.47 | 1.04-2.12 | 0.02 |
| missile, melee | 4468 | 90 | 200.4 | 180.6 | 0.90 | 0.86-0.94 | 0.97 | 0.93-1.01 | -1.15 |
| missile, routing | 3430 | 88 | 20.5 | 26.5 | 1.29 | 1.05-1.60 | 1.23 | 0.94-1.60 | 0.34 |
| infantry melee, melee | 32265 | 92 | 840.8 | 826.8 | 0.98 | 0.95-1.02 | 1.06 | 1.02-1.11 | -0.81 |
| infantry melee, routing | 16773 | 92 | 68.5 | 82.7 | 1.21 | 1.05-1.37 | 1.12 | 0.96-1.30 | 0.82 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 68540 | 92 | 21 | 58.8 | 2.80 | 1.66-5.21 |
| steady (3) | 10339 | 92 | 40 | 80.8 | 2.02 | 1.33-3.01 |
| shaken (4) | 16007 | 92 | 630 | 826.5 | 1.31 | 1.16-1.46 |
| wavering (5) | 8229 | 92 | 1956 | 1659.0 | 0.85 | 0.79-0.91 |
| mp < 0.25 | 7641 | 92 | 1990 | 1663.5 | 0.84 | 0.77-0.91 |
| mp 0.25-0.5 | 12928 | 92 | 581 | 712.2 | 1.23 | 1.09-1.36 |
| mp 0.5-0.75 | 15827 | 92 | 94 | 202.0 | 2.15 | 1.72-2.62 |
| mp >= 0.75 | 66778 | 92 | 26 | 70.8 | 2.72 | 1.60-4.92 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 103174 | 92 | 2691 | 2648.5 | 0.98 | 0.91-1.06 |
| standing, not in melee, no contact in window | 51421 | 92 | 381 | 311.0 | 0.82 | 0.63-0.98 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 238 | 220.8 | 0.93 | 0.84-1.03 |
| contact begins in window, ran in | 9006 | 92 | 166 | 204.5 | 1.23 | 1.04-1.44 |
| contact begins in window, walked in | 327 | 83 | 10 | 6.2 | 0.62 | 0.30-1.25 |
| melee, first 10 s of the fight | 5739 | 92 | 251 | 279.2 | 1.11 | 1.00-1.23 |
| melee, 10-30 s | 8207 | 92 | 403 | 415.0 | 1.03 | 0.94-1.13 |
| melee, 30 s+ | 25884 | 92 | 1280 | 1263.2 | 0.99 | 0.90-1.08 |
| melee, sim touches 1 enemy | 14716 | 92 | 732 | 761.0 | 1.04 | 0.94-1.18 |
| melee, sim touches 2+ enemies | 19308 | 92 | 985 | 1052.5 | 1.07 | 0.97-1.17 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 217 | 144.0 | 0.66 | 0.57-0.76 |
| melee, touched in front | 12723 | 92 | 467 | 437.8 | 0.94 | 0.85-1.04 |
| melee, touched in flank | 15606 | 92 | 811 | 873.2 | 1.08 | 0.98-1.18 |
| melee, touched in rear | 5695 | 92 | 439 | 502.5 | 1.14 | 1.02-1.28 |
| melee, health < 30 % | 4220 | 92 | 843 | 838.0 | 0.99 | 0.92-1.08 |
| melee, health >= 30 % | 35610 | 92 | 1091 | 1119.5 | 1.03 | 0.93-1.12 |
| standing, health < 30 % | 13319 | 92 | 1409 | 1308.0 | 0.93 | 0.85-1.01 |
| lord | 8496 | 92 | 134 | 100.0 | 0.75 | 0.61-0.88 |
| missile | 24883 | 92 | 591 | 582.8 | 0.99 | 0.90-1.08 |
| infantry melee | 69795 | 92 | 1966 | 1965.8 | 1.00 | 0.91-1.10 |
| network's side | 47765 | 92 | 1745 | 1638.2 | 0.94 | 0.86-1.01 |
| the other side | 55409 | 92 | 946 | 1010.2 | 1.07 | 0.92-1.25 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 20950 | 92 | 1541 | 1573.5 | 1.02 | 0.97-1.07 |
| routers in melee | 3976 | 92 | 126 | 198.2 | 1.57 | 1.38-1.85 |
| routers free | 16974 | 92 | 1415 | 1375.2 | 0.97 | 0.93-1.02 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 100483 | -0.016 | -0.011 | +0.004 | +0.001 - +0.007 |
| standing, not in melee, no contact in window | 51040 | +0.012 | +0.002 | -0.010 | -0.013 - -0.007 |
| standing, shot at (an enemy shooting it) | 3747 | -0.080 | -0.085 | -0.005 | -0.012 - +0.003 |
| contact begins in window, ran in | 8840 | -0.046 | -0.053 | -0.007 | -0.015 - +0.001 |
| contact begins in window, walked in | 317 | -0.052 | -0.058 | -0.006 | -0.031 - +0.017 |
| melee, first 10 s of the fight | 5488 | -0.112 | -0.106 | +0.005 | -0.003 - +0.013 |
| melee, 10-30 s | 7804 | -0.056 | -0.043 | +0.013 | +0.006 - +0.019 |
| melee, 30 s+ | 24604 | -0.023 | +0.013 | +0.036 | +0.029 - +0.044 |
| melee, sim touches 1 enemy | 13984 | -0.046 | -0.033 | +0.013 | +0.006 - +0.020 |
| melee, sim touches 2+ enemies | 18323 | -0.046 | -0.022 | +0.024 | +0.017 - +0.031 |
| melee, sim touches nobody (game flag on) | 5589 | -0.024 | +0.047 | +0.071 | +0.061 - +0.082 |
| melee, touched in front | 12256 | -0.040 | -0.012 | +0.028 | +0.021 - +0.035 |
| melee, touched in flank | 14795 | -0.048 | -0.033 | +0.015 | +0.009 - +0.023 |
| melee, touched in rear | 5256 | -0.054 | -0.045 | +0.009 | -0.002 - +0.021 |
| melee, health < 30 % | 3377 | -0.035 | -0.022 | +0.013 | +0.002 - +0.022 |
| melee, health >= 30 % | 34519 | -0.044 | -0.015 | +0.028 | +0.022 - +0.034 |
| standing, health < 30 % | 11910 | -0.012 | -0.008 | +0.003 | -0.004 - +0.012 |
| eager / confident (ms 1-2) | 68519 | -0.021 | -0.029 | -0.008 | -0.010 - -0.005 |
| steady (3) | 10299 | -0.021 | +0.013 | +0.034 | +0.028 - +0.041 |
| shaken (4) | 15377 | -0.011 | +0.017 | +0.028 | +0.018 - +0.038 |
| wavering (5) | 6273 | +0.038 | +0.067 | +0.029 | +0.018 - +0.041 |
| mp < 0.25 | 5651 | +0.035 | +0.066 | +0.031 | +0.017 - +0.047 |
| mp 0.25-0.5 | 12347 | -0.002 | +0.033 | +0.035 | +0.024 - +0.045 |
| mp 0.5-0.75 | 15733 | -0.021 | +0.010 | +0.031 | +0.025 - +0.037 |
| mp >= 0.75 | 66752 | -0.021 | -0.031 | -0.010 | -0.013 - -0.007 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 889 | 86 | 131214 | 52862 | 0.40 | 0.35-0.46 |
| hold, missile unit | 1445 | 84 | 217281 | 124291 | 0.57 | 0.51-0.64 |
| hold, lord | 596 | 87 | 158713 | 70302 | 0.44 | 0.38-0.51 |
| move, formation | 1485 | 88 | 183842 | 57986 | 0.32 | 0.28-0.36 |
| move, missile unit | 899 | 83 | 84489 | 24596 | 0.29 | 0.22-0.38 |
| move, lord | 90 | 45 | 11857 | 3436 | 0.29 | 0.17-0.44 |
| attack, formation | 29891 | 92 | 7200979 | 4843933 | 0.67 | 0.64-0.70 |
| attack, missile unit | 2124 | 89 | 372731 | 229379 | 0.62 | 0.56-0.67 |
| attack, lord | 2411 | 92 | 555622 | 412540 | 0.74 | 0.68-0.82 |
