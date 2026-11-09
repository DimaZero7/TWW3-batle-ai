## build/morale-hyp/sr_all

Windows of 10 s; 92 battles, 124124 unit-windows. HP lost sim/game 1.01 (0.98-1.05); rout starts sim/game 0.88 (0.81-0.96), game 2691, sim 2380.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.88 | 0.88 | 0.89 | 0.90 | 0.91 | 0.92 | 0.94 | 0.96 | 0.97 | 0.98 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 124124 | 92 | 1631.4 | 1651.1 | 1.01 | 0.98-1.05 | 1.08 | 1.04-1.12 | 1.00 |
| standing, not in melee, no contact in window | 51421 | 92 | 79.4 | 114.1 | 1.44 | 1.33-1.57 | 1.54 | 1.42-1.69 | 1.76 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 196.4 | 193.6 | 0.99 | 0.95-1.03 | 1.15 | 1.09-1.22 | -0.14 |
| contact begins in window, ran in | 9006 | 92 | 204.1 | 197.8 | 0.97 | 0.92-1.02 | 0.96 | 0.91-1.01 | -0.32 |
| contact begins in window, walked in | 327 | 83 | 8.9 | 7.4 | 0.83 | 0.73-0.94 | 0.82 | 0.70-0.95 | -0.08 |
| melee, first 10 s of the fight | 5739 | 92 | 194.2 | 177.2 | 0.91 | 0.88-0.95 | 0.98 | 0.94-1.03 | -0.86 |
| melee, 10-30 s | 8207 | 92 | 268.3 | 246.1 | 0.92 | 0.88-0.95 | 0.98 | 0.94-1.03 | -1.12 |
| melee, 30 s+ | 25884 | 92 | 656.5 | 670.1 | 1.02 | 0.98-1.06 | 1.10 | 1.06-1.14 | 0.69 |
| melee, sim touches 1 enemy | 14716 | 92 | 420.3 | 376.4 | 0.90 | 0.87-0.93 | 0.96 | 0.93-1.00 | -2.22 |
| melee, sim touches 2+ enemies | 19308 | 92 | 580.9 | 626.1 | 1.08 | 1.04-1.12 | 1.14 | 1.09-1.19 | 2.29 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 117.8 | 90.8 | 0.77 | 0.72-0.83 | 0.93 | 0.87-1.00 | -1.37 |
| melee, touched in front | 12723 | 92 | 319.3 | 306.5 | 0.96 | 0.93-0.99 | 1.04 | 0.99-1.08 | -0.65 |
| melee, touched in flank | 15606 | 92 | 479.3 | 492.8 | 1.03 | 0.99-1.07 | 1.09 | 1.05-1.14 | 0.69 |
| melee, touched in rear | 5695 | 92 | 202.6 | 203.2 | 1.00 | 0.95-1.05 | 1.04 | 0.99-1.10 | 0.03 |
| melee, health < 30 % | 4220 | 92 | 87.0 | 100.8 | 1.16 | 1.10-1.22 | 1.29 | 1.21-1.38 | 0.70 |
| melee, health >= 30 % | 35610 | 92 | 1032.0 | 992.5 | 0.96 | 0.93-0.99 | 1.03 | 0.99-1.07 | -2.00 |
| routing, in melee (pursued) | 3976 | 92 | 63.3 | 75.9 | 1.20 | 1.05-1.35 | 1.09 | 0.95-1.26 | 0.64 |
| routing, free | 16974 | 92 | 28.2 | 36.0 | 1.28 | 1.11-1.48 | 1.27 | 1.08-1.51 | 0.40 |
| standing, health < 30 % | 13319 | 92 | 118.5 | 133.6 | 1.13 | 1.07-1.19 | 1.26 | 1.18-1.34 | 0.76 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 9243 | 92 | 110.7 | 126.2 | 1.14 | 1.07-1.21 | 1.16 | 1.09-1.24 | 0.79 |
| missile | 28313 | 92 | 348.2 | 326.4 | 0.94 | 0.90-0.97 | 1.01 | 0.96-1.05 | -1.10 |
| infantry melee | 86568 | 92 | 1172.5 | 1198.5 | 1.02 | 0.98-1.06 | 1.10 | 1.05-1.14 | 1.31 |
| network's side | 60359 | 92 | 913.7 | 919.4 | 1.01 | 0.97-1.04 | 1.07 | 1.03-1.11 | 0.29 |
| the other side | 63765 | 92 | 717.7 | 731.8 | 1.02 | 0.97-1.08 | 1.09 | 1.03-1.16 | 0.71 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 3097 | 92 | 77.8 | 84.6 | 1.09 | 1.03-1.15 | 1.10 | 1.03-1.18 | 0.34 |
| lord, routing | 747 | 69 | 2.5 | 2.8 | 1.11 | 0.83-1.43 | 1.48 | 1.04-2.12 | 0.01 |
| missile, melee | 4468 | 90 | 200.4 | 181.0 | 0.90 | 0.86-0.94 | 0.97 | 0.93-1.02 | -0.98 |
| missile, routing | 3430 | 88 | 20.5 | 26.5 | 1.29 | 1.05-1.60 | 1.23 | 0.94-1.59 | 0.30 |
| infantry melee, melee | 32265 | 92 | 840.8 | 827.7 | 0.98 | 0.95-1.02 | 1.06 | 1.02-1.11 | -0.66 |
| infantry melee, routing | 16773 | 92 | 68.5 | 82.7 | 1.21 | 1.05-1.37 | 1.12 | 0.96-1.30 | 0.72 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 68540 | 92 | 21 | 45.8 | 2.18 | 1.27-4.05 |
| steady (3) | 10339 | 92 | 40 | 65.5 | 1.64 | 1.07-2.38 |
| shaken (4) | 16007 | 92 | 630 | 734.0 | 1.17 | 1.02-1.30 |
| wavering (5) | 8229 | 92 | 1956 | 1511.8 | 0.77 | 0.72-0.83 |
| mp < 0.25 | 7641 | 92 | 1990 | 1517.8 | 0.76 | 0.71-0.83 |
| mp 0.25-0.5 | 12928 | 92 | 581 | 631.2 | 1.09 | 0.96-1.21 |
| mp 0.5-0.75 | 15827 | 92 | 94 | 174.5 | 1.86 | 1.48-2.26 |
| mp >= 0.75 | 66778 | 92 | 26 | 56.5 | 2.17 | 1.26-3.96 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 103174 | 92 | 2691 | 2380.0 | 0.88 | 0.81-0.96 |
| standing, not in melee, no contact in window | 51421 | 92 | 381 | 281.5 | 0.74 | 0.58-0.89 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 238 | 212.0 | 0.89 | 0.81-0.98 |
| contact begins in window, ran in | 9006 | 92 | 166 | 180.2 | 1.09 | 0.90-1.29 |
| contact begins in window, walked in | 327 | 83 | 10 | 5.5 | 0.55 | 0.25-1.00 |
| melee, first 10 s of the fight | 5739 | 92 | 251 | 251.8 | 1.00 | 0.90-1.11 |
| melee, 10-30 s | 8207 | 92 | 403 | 385.0 | 0.96 | 0.86-1.05 |
| melee, 30 s+ | 25884 | 92 | 1280 | 1113.8 | 0.87 | 0.79-0.96 |
| melee, sim touches 1 enemy | 14716 | 92 | 732 | 688.2 | 0.94 | 0.85-1.08 |
| melee, sim touches 2+ enemies | 19308 | 92 | 985 | 936.2 | 0.95 | 0.86-1.04 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 217 | 126.0 | 0.58 | 0.49-0.68 |
| melee, touched in front | 12723 | 92 | 467 | 380.5 | 0.81 | 0.73-0.92 |
| melee, touched in flank | 15606 | 92 | 811 | 787.0 | 0.97 | 0.88-1.07 |
| melee, touched in rear | 5695 | 92 | 439 | 457.0 | 1.04 | 0.92-1.17 |
| melee, health < 30 % | 4220 | 92 | 843 | 746.0 | 0.88 | 0.81-0.98 |
| melee, health >= 30 % | 35610 | 92 | 1091 | 1004.5 | 0.92 | 0.82-1.01 |
| standing, health < 30 % | 13319 | 92 | 1409 | 1180.0 | 0.84 | 0.76-0.92 |
| lord | 8496 | 92 | 134 | 96.0 | 0.72 | 0.57-0.88 |
| missile | 24883 | 92 | 591 | 522.8 | 0.88 | 0.80-0.97 |
| infantry melee | 69795 | 92 | 1966 | 1761.2 | 0.90 | 0.81-0.99 |
| network's side | 47765 | 92 | 1745 | 1473.8 | 0.84 | 0.78-0.91 |
| the other side | 55409 | 92 | 946 | 906.2 | 0.96 | 0.81-1.14 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 20950 | 92 | 1541 | 1615.8 | 1.05 | 1.00-1.09 |
| routers in melee | 3976 | 92 | 126 | 201.2 | 1.60 | 1.40-1.87 |
| routers free | 16974 | 92 | 1415 | 1414.5 | 1.00 | 0.95-1.04 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 100483 | -0.016 | -0.005 | +0.011 | +0.008 - +0.014 |
| standing, not in melee, no contact in window | 51040 | +0.012 | +0.004 | -0.009 | -0.012 - -0.006 |
| standing, shot at (an enemy shooting it) | 3747 | -0.080 | -0.080 | +0.001 | -0.007 - +0.009 |
| contact begins in window, ran in | 8840 | -0.046 | -0.046 | +0.000 | -0.008 - +0.008 |
| contact begins in window, walked in | 317 | -0.052 | -0.045 | +0.007 | -0.016 - +0.029 |
| melee, first 10 s of the fight | 5488 | -0.112 | -0.097 | +0.014 | +0.006 - +0.022 |
| melee, 10-30 s | 7804 | -0.056 | -0.032 | +0.024 | +0.018 - +0.031 |
| melee, 30 s+ | 24604 | -0.023 | +0.029 | +0.052 | +0.045 - +0.060 |
| melee, sim touches 1 enemy | 13984 | -0.046 | -0.020 | +0.026 | +0.019 - +0.033 |
| melee, sim touches 2+ enemies | 18323 | -0.046 | -0.006 | +0.040 | +0.033 - +0.047 |
| melee, sim touches nobody (game flag on) | 5589 | -0.024 | +0.058 | +0.082 | +0.072 - +0.092 |
| melee, touched in front | 12256 | -0.040 | +0.003 | +0.043 | +0.036 - +0.050 |
| melee, touched in flank | 14795 | -0.048 | -0.018 | +0.031 | +0.024 - +0.038 |
| melee, touched in rear | 5256 | -0.054 | -0.032 | +0.022 | +0.010 - +0.033 |
| melee, health < 30 % | 3377 | -0.035 | -0.008 | +0.027 | +0.018 - +0.036 |
| melee, health >= 30 % | 34519 | -0.044 | -0.001 | +0.042 | +0.036 - +0.049 |
| standing, health < 30 % | 11910 | -0.012 | -0.000 | +0.011 | +0.003 - +0.020 |
| eager / confident (ms 1-2) | 68519 | -0.021 | -0.024 | -0.003 | -0.006 - -0.001 |
| steady (3) | 10299 | -0.021 | +0.025 | +0.046 | +0.039 - +0.053 |
| shaken (4) | 15377 | -0.011 | +0.030 | +0.040 | +0.029 - +0.052 |
| wavering (5) | 6273 | +0.038 | +0.078 | +0.040 | +0.029 - +0.053 |
| mp < 0.25 | 5651 | +0.035 | +0.078 | +0.043 | +0.027 - +0.061 |
| mp 0.25-0.5 | 12347 | -0.002 | +0.044 | +0.047 | +0.034 - +0.058 |
| mp 0.5-0.75 | 15733 | -0.021 | +0.022 | +0.043 | +0.037 - +0.050 |
| mp >= 0.75 | 66752 | -0.021 | -0.027 | -0.006 | -0.008 - -0.003 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 889 | 86 | 131214 | 53234 | 0.41 | 0.36-0.46 |
| hold, missile unit | 1445 | 84 | 217281 | 125580 | 0.58 | 0.52-0.65 |
| hold, lord | 596 | 87 | 158713 | 70265 | 0.44 | 0.38-0.51 |
| move, formation | 1485 | 88 | 183842 | 58348 | 0.32 | 0.28-0.36 |
| move, missile unit | 899 | 83 | 84489 | 24707 | 0.29 | 0.23-0.38 |
| move, lord | 90 | 45 | 11857 | 3468 | 0.29 | 0.17-0.45 |
| attack, formation | 29891 | 92 | 7200979 | 4852443 | 0.67 | 0.64-0.70 |
| attack, missile unit | 2124 | 89 | 372731 | 231439 | 0.62 | 0.57-0.67 |
| attack, lord | 2411 | 92 | 555622 | 412868 | 0.74 | 0.68-0.82 |
