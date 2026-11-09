## build/engagefix/sr_pos_rel1: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 0.95 (0.87-1.05); rout starts sim/game 1.00 (0.81-1.20), game 683, sim 680.5.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.95 | 0.91 | 0.90 | 0.88 | 0.88 | 0.88 | 0.89 | 0.90 | 0.90 | 0.91 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 358.0 | 0.95 | 0.87-1.05 | 0.97 | 0.88-1.06 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 24.7 | 1.25 | 1.01-1.46 | 1.28 | 1.04-1.54 | -0.26 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 23.8 | 0.95 | 0.77-1.17 | 1.03 | 0.80-1.32 | 0.07 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 41.4 | 0.83 | 0.73-0.95 | 0.77 | 0.68-0.90 | 0.47 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.6 | 0.50 | 0.30-0.78 | 0.61 | 0.32-0.99 | 0.03 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 39.6 | 0.82 | 0.76-0.90 | 0.85 | 0.77-0.94 | 0.45 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 53.1 | 0.88 | 0.80-0.97 | 0.92 | 0.82-1.02 | 0.38 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 152.3 | 0.95 | 0.88-1.01 | 0.98 | 0.92-1.05 | 0.47 |
| melee, sim touches 1 enemy | 4186 | 20 | 111.5 | 98.9 | 0.89 | 0.82-0.97 | 0.91 | 0.83-1.00 | 0.68 |
| melee, sim touches 2+ enemies | 4016 | 20 | 126.7 | 126.3 | 1.00 | 0.94-1.04 | 1.03 | 0.96-1.08 | 0.02 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 31.1 | 19.9 | 0.64 | 0.51-0.81 | 0.74 | 0.62-0.88 | 0.60 |
| melee, touched in front | 3901 | 20 | 99.2 | 89.6 | 0.90 | 0.85-0.96 | 0.94 | 0.87-1.01 | 0.52 |
| melee, touched in flank | 3309 | 20 | 103.9 | 101.0 | 0.97 | 0.91-1.03 | 1.00 | 0.94-1.06 | 0.15 |
| melee, touched in rear | 992 | 20 | 35.1 | 34.6 | 0.98 | 0.89-1.07 | 0.98 | 0.85-1.07 | 0.03 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 24.8 | 1.06 | 0.92-1.20 | 1.17 | 1.02-1.34 | -0.07 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 220.3 | 0.90 | 0.83-0.96 | 0.92 | 0.86-0.99 | 1.37 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 21.1 | 1.49 | 1.19-1.80 | 1.40 | 1.01-1.86 | -0.37 |
| routing, free | 6219 | 20 | 5.5 | 9.5 | 1.72 | 1.25-2.59 | 1.78 | 1.23-2.76 | -0.21 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 34.2 | 1.04 | 0.91-1.20 | 1.16 | 1.02-1.33 | -0.08 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 24.9 | 0.86 | 0.76-0.96 | 0.82 | 0.71-0.94 | 0.22 |
| missile | 3989 | 20 | 31.8 | 30.2 | 0.95 | 0.81-1.12 | 0.97 | 0.81-1.20 | 0.08 |
| infantry melee | 29215 | 20 | 315.8 | 302.9 | 0.96 | 0.87-1.06 | 0.98 | 0.88-1.08 | 0.69 |
| network's side | 16523 | 20 | 184.7 | 162.9 | 0.88 | 0.83-0.94 | 0.88 | 0.83-0.95 | 1.17 |
| the other side | 19331 | 20 | 191.9 | 195.1 | 1.02 | 0.88-1.16 | 1.05 | 0.90-1.19 | -0.17 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 17.2 | 0.85 | 0.76-0.96 | 0.82 | 0.70-0.94 | 0.16 |
| lord, routing | 236 | 15 | 0.9 | 0.7 | 0.77 | 0.45-1.43 | 0.97 | 0.42-3.23 | 0.01 |
| missile, melee | 413 | 18 | 15.9 | 15.4 | 0.97 | 0.87-1.08 | 0.96 | 0.84-1.10 | 0.03 |
| missile, routing | 405 | 16 | 2.2 | 4.6 | 2.07 | 1.09-3.43 | 2.34 | 0.98-4.28 | -0.13 |
| infantry melee, melee | 8824 | 20 | 233.2 | 212.5 | 0.91 | 0.85-0.97 | 0.95 | 0.89-1.02 | 1.12 |
| infantry melee, routing | 6408 | 20 | 16.5 | 25.3 | 1.53 | 1.23-1.83 | 1.43 | 1.09-1.81 | -0.47 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 6.0 | 1.00 | 0.29-4.42 |
| steady (3) | 3212 | 20 | 12 | 33.8 | 2.81 | 1.33-5.61 |
| shaken (4) | 4549 | 20 | 153 | 200.8 | 1.31 | 0.98-1.64 |
| wavering (5) | 2661 | 20 | 503 | 434.0 | 0.86 | 0.71-1.05 |
| mp < 0.25 | 2598 | 20 | 510 | 455.8 | 0.89 | 0.72-1.10 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 173.8 | 1.24 | 1.02-1.51 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 47.0 | 1.74 | 0.93-3.03 |
| mp >= 0.75 | 17870 | 20 | 6 | 4.0 | 0.67 | 0.14-3.25 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 680.5 | 1.00 | 0.81-1.20 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 122.0 | 1.08 | 0.63-1.46 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 50.2 | 1.07 | 0.85-1.57 |
| contact begins in window, ran in | 2307 | 20 | 42 | 55.0 | 1.31 | 0.94-1.87 |
| contact begins in window, walked in | 56 | 15 | 4 | 0.0 | 0.00 | 0.00-0.00 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 81.0 | 0.92 | 0.75-1.10 |
| melee, 10-30 s | 2047 | 20 | 99 | 104.8 | 1.06 | 0.88-1.24 |
| melee, 30 s+ | 6376 | 20 | 297 | 278.5 | 0.94 | 0.74-1.18 |
| melee, sim touches 1 enemy | 4186 | 20 | 193 | 213.5 | 1.11 | 0.86-1.49 |
| melee, sim touches 2+ enemies | 4016 | 20 | 226 | 217.8 | 0.96 | 0.83-1.09 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 65 | 33.0 | 0.51 | 0.38-0.70 |
| melee, touched in front | 3901 | 20 | 145 | 138.0 | 0.95 | 0.78-1.19 |
| melee, touched in flank | 3309 | 20 | 184 | 198.2 | 1.08 | 0.85-1.34 |
| melee, touched in rear | 992 | 20 | 90 | 95.0 | 1.06 | 0.87-1.28 |
| melee, health < 30 % | 1132 | 20 | 225 | 211.8 | 0.94 | 0.74-1.19 |
| melee, health >= 30 % | 8826 | 20 | 259 | 252.5 | 0.97 | 0.81-1.17 |
| standing, health < 30 % | 4614 | 20 | 381 | 372.8 | 0.98 | 0.77-1.21 |
| lord | 2414 | 20 | 34 | 31.8 | 0.93 | 0.57-1.54 |
| missile | 3584 | 20 | 50 | 48.5 | 0.97 | 0.72-1.23 |
| infantry melee | 22807 | 20 | 599 | 600.2 | 1.00 | 0.80-1.24 |
| network's side | 13157 | 20 | 356 | 274.0 | 0.77 | 0.67-0.88 |
| the other side | 15648 | 20 | 327 | 406.5 | 1.24 | 0.93-1.62 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 264.0 | 0.71 | 0.63-0.77 |
| routers in melee | 830 | 20 | 36 | 34.2 | 0.95 | 0.77-1.15 |
| routers free | 6219 | 20 | 335 | 229.8 | 0.69 | 0.60-0.75 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.002 | +0.012 | +0.005 - +0.019 |
| standing, not in melee, no contact in window | 16032 | +0.008 | +0.003 | -0.005 | -0.009 - -0.001 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.070 | -0.003 | -0.040 - +0.038 |
| contact begins in window, ran in | 2265 | -0.034 | -0.002 | +0.033 | +0.021 - +0.043 |
| contact begins in window, walked in | 52 | +0.013 | +0.083 | +0.070 | +0.014 - +0.113 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.061 | +0.051 | +0.032 - +0.066 |
| melee, 10-30 s | 1948 | -0.050 | -0.006 | +0.044 | +0.029 - +0.058 |
| melee, 30 s+ | 6079 | -0.026 | +0.004 | +0.030 | +0.016 - +0.043 |
| melee, sim touches 1 enemy | 3993 | -0.046 | -0.035 | +0.011 | +0.000 - +0.021 |
| melee, sim touches 2+ enemies | 3790 | -0.051 | -0.020 | +0.031 | +0.015 - +0.046 |
| melee, sim touches nobody (game flag on) | 1691 | -0.025 | +0.082 | +0.107 | +0.082 - +0.125 |
| melee, touched in front | 3756 | -0.042 | -0.023 | +0.019 | +0.010 - +0.029 |
| melee, touched in flank | 3125 | -0.053 | -0.035 | +0.017 | +0.004 - +0.031 |
| melee, touched in rear | 902 | -0.057 | -0.021 | +0.037 | +0.015 - +0.058 |
| melee, health < 30 % | 907 | -0.034 | -0.007 | +0.027 | +0.010 - +0.043 |
| melee, health >= 30 % | 8567 | -0.045 | -0.008 | +0.037 | +0.023 - +0.049 |
| standing, health < 30 % | 4233 | -0.009 | -0.007 | +0.003 | -0.007 - +0.017 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.017 | +0.002 | -0.004 - +0.008 |
| steady (3) | 3200 | -0.013 | +0.019 | +0.032 | +0.018 - +0.049 |
| shaken (4) | 4396 | -0.010 | +0.017 | +0.028 | +0.009 - +0.054 |
| wavering (5) | 2158 | +0.031 | +0.060 | +0.029 | +0.010 - +0.051 |
| mp < 0.25 | 2088 | +0.027 | +0.046 | +0.019 | +0.001 - +0.046 |
| mp 0.25-0.5 | 3827 | -0.005 | +0.027 | +0.032 | +0.008 - +0.064 |
| mp 0.5-0.75 | 4343 | -0.015 | +0.021 | +0.036 | +0.023 - +0.046 |
| mp >= 0.75 | 17864 | -0.020 | -0.019 | +0.001 | -0.005 - +0.007 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 11476 | 0.93 | 0.60-1.32 |
| hold, lord | 59 | 18 | 13200 | 7915 | 0.60 | 0.39-0.85 |
| move, formation | 450 | 17 | 31949 | 11390 | 0.36 | 0.28-0.44 |
| move, missile unit | 148 | 14 | 6065 | 650 | 0.11 | 0.04-0.30 |
| move, lord | 41 | 13 | 4562 | 816 | 0.18 | 0.03-0.39 |
| attack, formation | 8219 | 20 | 1923931 | 1432827 | 0.74 | 0.70-0.79 |
| attack, missile unit | 236 | 18 | 39992 | 25271 | 0.63 | 0.51-0.80 |
| attack, lord | 621 | 20 | 133139 | 105696 | 0.79 | 0.63-0.96 |

## build/engagefix/sr_pos_rel1: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.92 (0.90-0.94); rout starts sim/game 0.92 (0.86-0.99), game 2008, sim 1855.2.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.86 | 0.85 | 0.84 | 0.84 | 0.85 | 0.85 | 0.85 | 0.86 | 0.87 | 0.87 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1151.3 | 0.92 | 0.90-0.94 | 0.95 | 0.93-0.98 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 69.9 | 1.17 | 1.10-1.25 | 1.23 | 1.14-1.33 | -0.10 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 154.4 | 0.90 | 0.87-0.93 | 1.02 | 0.98-1.07 | 0.16 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 143.9 | 0.93 | 0.90-0.98 | 0.91 | 0.87-0.96 | 0.10 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.7 | 0.86 | 0.75-1.00 | 0.82 | 0.70-0.98 | 0.01 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 123.5 | 0.84 | 0.82-0.88 | 0.87 | 0.84-0.92 | 0.22 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 169.2 | 0.81 | 0.79-0.84 | 0.84 | 0.81-0.87 | 0.38 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 446.9 | 0.90 | 0.87-0.94 | 0.93 | 0.89-0.97 | 0.47 |
| melee, sim touches 1 enemy | 10530 | 72 | 308.7 | 244.2 | 0.79 | 0.77-0.81 | 0.82 | 0.79-0.85 | 0.62 |
| melee, sim touches 2+ enemies | 15292 | 72 | 454.2 | 447.7 | 0.99 | 0.95-1.03 | 0.99 | 0.96-1.04 | 0.06 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 86.7 | 47.7 | 0.55 | 0.51-0.59 | 0.64 | 0.60-0.69 | 0.38 |
| melee, touched in front | 8822 | 72 | 220.1 | 189.2 | 0.86 | 0.83-0.89 | 0.91 | 0.87-0.95 | 0.30 |
| melee, touched in flank | 12297 | 72 | 375.4 | 352.2 | 0.94 | 0.91-0.97 | 0.95 | 0.92-0.99 | 0.22 |
| melee, touched in rear | 4703 | 72 | 167.5 | 150.5 | 0.90 | 0.85-0.95 | 0.89 | 0.85-0.94 | 0.16 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 65.4 | 1.03 | 0.98-1.09 | 1.10 | 1.03-1.18 | -0.02 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 674.3 | 0.86 | 0.83-0.89 | 0.88 | 0.85-0.91 | 1.08 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 59.3 | 1.21 | 1.06-1.38 | 1.17 | 1.01-1.34 | -0.10 |
| routing, free | 10755 | 72 | 22.6 | 34.6 | 1.53 | 1.36-1.71 | 1.58 | 1.37-1.82 | -0.12 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 85.8 | 1.00 | 0.96-1.04 | 1.06 | 1.01-1.12 | 0.00 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 89.9 | 1.10 | 1.02-1.18 | 1.11 | 1.02-1.19 | -0.08 |
| missile | 24324 | 72 | 316.4 | 274.3 | 0.87 | 0.83-0.90 | 0.90 | 0.85-0.94 | 0.41 |
| infantry melee | 57353 | 72 | 856.7 | 787.1 | 0.92 | 0.89-0.95 | 0.96 | 0.93-0.99 | 0.67 |
| network's side | 43836 | 72 | 729.1 | 673.1 | 0.92 | 0.89-0.96 | 0.95 | 0.92-0.99 | 0.54 |
| the other side | 44434 | 72 | 525.7 | 478.2 | 0.91 | 0.87-0.95 | 0.95 | 0.91-1.00 | 0.46 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 62.2 | 1.08 | 1.00-1.15 | 1.09 | 0.99-1.18 | -0.04 |
| lord, routing | 511 | 54 | 1.6 | 2.3 | 1.46 | 1.12-1.96 | 1.67 | 1.18-2.28 | -0.01 |
| missile, melee | 4055 | 72 | 184.5 | 149.5 | 0.81 | 0.77-0.85 | 0.84 | 0.80-0.88 | 0.34 |
| missile, routing | 3025 | 72 | 18.3 | 25.0 | 1.36 | 1.11-1.67 | 1.32 | 1.05-1.66 | -0.06 |
| infantry melee, melee | 23441 | 72 | 607.5 | 527.9 | 0.87 | 0.84-0.91 | 0.90 | 0.87-0.93 | 0.77 |
| infantry melee, routing | 10365 | 72 | 52.0 | 66.7 | 1.28 | 1.13-1.48 | 1.29 | 1.11-1.52 | -0.14 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 49.0 | 3.27 | 1.89-6.32 |
| steady (3) | 7127 | 72 | 28 | 63.8 | 2.28 | 1.41-3.71 |
| shaken (4) | 11458 | 72 | 477 | 562.5 | 1.18 | 1.06-1.30 |
| wavering (5) | 5568 | 72 | 1453 | 1162.5 | 0.80 | 0.74-0.85 |
| mp < 0.25 | 5043 | 72 | 1480 | 1154.8 | 0.78 | 0.73-0.84 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 505.5 | 1.15 | 1.03-1.27 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 145.5 | 2.17 | 1.67-2.85 |
| mp >= 0.75 | 48908 | 72 | 20 | 49.5 | 2.48 | 1.47-4.63 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 1855.2 | 0.92 | 0.86-0.99 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 239.5 | 0.89 | 0.68-1.11 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 187.8 | 0.98 | 0.86-1.11 |
| contact begins in window, ran in | 6699 | 72 | 124 | 136.5 | 1.10 | 0.94-1.32 |
| contact begins in window, walked in | 271 | 68 | 6 | 7.0 | 1.17 | 0.70-2.25 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 178.2 | 1.09 | 0.99-1.23 |
| melee, 10-30 s | 6160 | 72 | 304 | 279.0 | 0.92 | 0.83-1.02 |
| melee, 30 s+ | 19508 | 72 | 983 | 871.8 | 0.89 | 0.81-0.98 |
| melee, sim touches 1 enemy | 10530 | 72 | 539 | 514.2 | 0.95 | 0.88-1.03 |
| melee, sim touches 2+ enemies | 15292 | 72 | 759 | 739.0 | 0.97 | 0.88-1.07 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 152 | 75.8 | 0.50 | 0.39-0.62 |
| melee, touched in front | 8822 | 72 | 322 | 288.5 | 0.90 | 0.82-0.99 |
| melee, touched in flank | 12297 | 72 | 627 | 621.0 | 0.99 | 0.90-1.08 |
| melee, touched in rear | 4703 | 72 | 349 | 343.8 | 0.98 | 0.87-1.11 |
| melee, health < 30 % | 3088 | 72 | 618 | 554.0 | 0.90 | 0.83-0.96 |
| melee, health >= 30 % | 26784 | 72 | 832 | 775.0 | 0.93 | 0.84-1.03 |
| standing, health < 30 % | 8705 | 72 | 1028 | 897.5 | 0.87 | 0.80-0.94 |
| lord | 6082 | 72 | 100 | 76.8 | 0.77 | 0.59-0.95 |
| missile | 21299 | 72 | 541 | 550.8 | 1.02 | 0.92-1.12 |
| infantry melee | 46988 | 72 | 1367 | 1227.8 | 0.90 | 0.83-0.96 |
| network's side | 34608 | 72 | 1389 | 1321.0 | 0.95 | 0.88-1.03 |
| the other side | 39761 | 72 | 619 | 534.2 | 0.86 | 0.78-0.96 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1133.5 | 0.97 | 0.92-1.02 |
| routers in melee | 3146 | 72 | 90 | 137.0 | 1.52 | 1.31-1.82 |
| routers free | 10755 | 72 | 1080 | 996.5 | 0.92 | 0.88-0.98 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.006 | +0.011 | +0.007 - +0.014 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.001 | -0.015 | -0.020 - -0.011 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.078 | +0.004 | -0.005 - +0.014 |
| contact begins in window, ran in | 6575 | -0.050 | -0.051 | -0.002 | -0.011 - +0.008 |
| contact begins in window, walked in | 265 | -0.065 | -0.073 | -0.008 | -0.036 - +0.018 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.092 | +0.019 | +0.011 - +0.028 |
| melee, 10-30 s | 5856 | -0.058 | -0.027 | +0.031 | +0.023 - +0.040 |
| melee, 30 s+ | 18525 | -0.023 | +0.034 | +0.056 | +0.049 - +0.063 |
| melee, sim touches 1 enemy | 9991 | -0.046 | -0.014 | +0.032 | +0.025 - +0.039 |
| melee, sim touches 2+ enemies | 14533 | -0.045 | -0.010 | +0.035 | +0.028 - +0.042 |
| melee, sim touches nobody (game flag on) | 3898 | -0.024 | +0.098 | +0.122 | +0.108 - +0.135 |
| melee, touched in front | 8500 | -0.039 | +0.009 | +0.048 | +0.041 - +0.055 |
| melee, touched in flank | 11670 | -0.047 | -0.020 | +0.027 | +0.019 - +0.034 |
| melee, touched in rear | 4354 | -0.053 | -0.030 | +0.023 | +0.010 - +0.036 |
| melee, health < 30 % | 2470 | -0.036 | -0.004 | +0.032 | +0.017 - +0.045 |
| melee, health >= 30 % | 25952 | -0.043 | +0.004 | +0.047 | +0.041 - +0.054 |
| standing, health < 30 % | 7677 | -0.013 | -0.008 | +0.005 | -0.007 - +0.016 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.025 | -0.003 | -0.007 - -0.000 |
| steady (3) | 7099 | -0.025 | +0.026 | +0.051 | +0.043 - +0.059 |
| shaken (4) | 10981 | -0.011 | +0.031 | +0.042 | +0.033 - +0.050 |
| wavering (5) | 4115 | +0.042 | +0.070 | +0.028 | +0.009 - +0.043 |
| mp < 0.25 | 3563 | +0.040 | +0.074 | +0.034 | +0.011 - +0.053 |
| mp 0.25-0.5 | 8520 | -0.001 | +0.047 | +0.048 | +0.041 - +0.055 |
| mp 0.5-0.75 | 11390 | -0.024 | +0.021 | +0.045 | +0.037 - +0.052 |
| mp >= 0.75 | 48888 | -0.022 | -0.028 | -0.006 | -0.009 - -0.002 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 42516 | 0.36 | 0.31-0.41 |
| hold, missile unit | 1416 | 72 | 213822 | 118040 | 0.55 | 0.49-0.62 |
| hold, lord | 537 | 69 | 145512 | 68038 | 0.47 | 0.40-0.54 |
| move, formation | 1035 | 71 | 151893 | 48339 | 0.32 | 0.28-0.37 |
| move, missile unit | 751 | 69 | 78425 | 20742 | 0.26 | 0.20-0.35 |
| move, lord | 49 | 32 | 7295 | 2870 | 0.39 | 0.24-0.62 |
| attack, formation | 21672 | 72 | 5277048 | 3223279 | 0.61 | 0.58-0.64 |
| attack, missile unit | 1888 | 71 | 332739 | 205567 | 0.62 | 0.56-0.67 |
| attack, lord | 1790 | 72 | 422482 | 287406 | 0.68 | 0.62-0.74 |
