## ../../../build/contactgap/sr_pos_rect: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 0.96 (0.87-1.06); rout starts sim/game 1.34 (1.14-1.51), game 683, sim 914.8.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.95 | 0.92 | 0.91 | 0.89 | 0.89 | 0.90 | 0.91 | 0.91 | 0.92 | 0.93 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 361.6 | 0.96 | 0.87-1.06 | 0.99 | 0.88-1.09 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 25.3 | 1.28 | 1.06-1.50 | 1.35 | 1.09-1.63 | -0.37 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 22.4 | 0.89 | 0.73-1.08 | 1.02 | 0.80-1.30 | 0.18 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 41.2 | 0.82 | 0.73-0.94 | 0.80 | 0.70-0.92 | 0.60 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.5 | 0.42 | 0.24-0.68 | 0.52 | 0.25-0.88 | 0.05 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 40.1 | 0.83 | 0.76-0.92 | 0.87 | 0.77-0.97 | 0.53 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 53.6 | 0.89 | 0.81-0.98 | 0.93 | 0.83-1.06 | 0.44 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 155.6 | 0.97 | 0.90-1.04 | 1.00 | 0.93-1.08 | 0.36 |
| melee, sim touches 1 enemy | 4186 | 20 | 111.5 | 99.2 | 0.89 | 0.82-0.98 | 0.92 | 0.82-1.02 | 0.82 |
| melee, sim touches 2+ enemies | 4016 | 20 | 126.7 | 129.9 | 1.03 | 0.96-1.08 | 1.06 | 0.99-1.12 | -0.22 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 31.1 | 20.1 | 0.65 | 0.53-0.80 | 0.73 | 0.62-0.88 | 0.73 |
| melee, touched in front | 3901 | 20 | 99.2 | 90.5 | 0.91 | 0.85-0.98 | 0.94 | 0.86-1.02 | 0.58 |
| melee, touched in flank | 3309 | 20 | 103.9 | 103.5 | 1.00 | 0.92-1.07 | 1.04 | 0.96-1.12 | 0.02 |
| melee, touched in rear | 992 | 20 | 35.1 | 35.1 | 1.00 | 0.90-1.09 | 1.02 | 0.89-1.13 | -0.00 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 25.3 | 1.08 | 0.94-1.22 | 1.20 | 1.06-1.36 | -0.13 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 223.9 | 0.91 | 0.84-0.98 | 0.94 | 0.86-1.02 | 1.46 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 21.1 | 1.50 | 1.19-1.82 | 1.42 | 1.05-1.87 | -0.47 |
| routing, free | 6219 | 20 | 5.5 | 9.3 | 1.68 | 1.22-2.53 | 1.72 | 1.16-2.68 | -0.25 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 34.0 | 1.04 | 0.90-1.18 | 1.17 | 1.03-1.31 | -0.09 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 24.8 | 0.86 | 0.76-0.96 | 0.86 | 0.75-0.99 | 0.28 |
| missile | 3989 | 20 | 31.8 | 30.6 | 0.96 | 0.82-1.14 | 1.00 | 0.83-1.22 | 0.08 |
| infantry melee | 29215 | 20 | 315.8 | 306.1 | 0.97 | 0.87-1.07 | 1.00 | 0.89-1.10 | 0.64 |
| network's side | 16523 | 20 | 184.7 | 165.2 | 0.89 | 0.84-0.96 | 0.90 | 0.84-0.98 | 1.30 |
| the other side | 19331 | 20 | 191.9 | 196.4 | 1.02 | 0.89-1.16 | 1.07 | 0.92-1.22 | -0.30 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 16.9 | 0.84 | 0.74-0.93 | 0.83 | 0.71-0.96 | 0.22 |
| lord, routing | 236 | 15 | 0.9 | 0.7 | 0.80 | 0.49-1.48 | 1.51 | 0.51-6.10 | 0.01 |
| missile, melee | 413 | 18 | 15.9 | 15.6 | 0.98 | 0.87-1.11 | 0.96 | 0.83-1.13 | 0.02 |
| missile, routing | 405 | 16 | 2.2 | 4.7 | 2.14 | 1.13-3.46 | 2.45 | 1.10-4.37 | -0.17 |
| infantry melee, melee | 8824 | 20 | 233.2 | 216.8 | 0.93 | 0.86-1.00 | 0.97 | 0.89-1.05 | 1.09 |
| infantry melee, routing | 6408 | 20 | 16.5 | 24.9 | 1.51 | 1.21-1.81 | 1.40 | 1.07-1.77 | -0.56 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 13.5 | 2.25 | 0.78-9.81 |
| steady (3) | 3212 | 20 | 12 | 44.0 | 3.67 | 1.86-7.28 |
| shaken (4) | 4549 | 20 | 153 | 271.2 | 1.77 | 1.33-2.23 |
| wavering (5) | 2661 | 20 | 503 | 580.0 | 1.15 | 1.02-1.26 |
| mp < 0.25 | 2598 | 20 | 510 | 592.8 | 1.16 | 1.01-1.30 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 242.8 | 1.73 | 1.45-2.03 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 68.2 | 2.53 | 1.42-4.32 |
| mp >= 0.75 | 17870 | 20 | 6 | 11.0 | 1.83 | 0.61-7.88 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 914.8 | 1.34 | 1.14-1.51 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 182.0 | 1.61 | 1.16-1.93 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 61.8 | 1.31 | 1.08-1.71 |
| contact begins in window, ran in | 2307 | 20 | 42 | 77.2 | 1.84 | 1.34-2.37 |
| contact begins in window, walked in | 56 | 15 | 4 | 2.5 | 0.62 | 0.00-2.00 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 100.2 | 1.14 | 0.97-1.34 |
| melee, 10-30 s | 2047 | 20 | 99 | 127.8 | 1.29 | 1.03-1.61 |
| melee, 30 s+ | 6376 | 20 | 297 | 377.0 | 1.27 | 1.07-1.47 |
| melee, sim touches 1 enemy | 4186 | 20 | 193 | 285.5 | 1.48 | 1.18-1.83 |
| melee, sim touches 2+ enemies | 4016 | 20 | 226 | 274.2 | 1.21 | 1.08-1.36 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 65 | 45.2 | 0.70 | 0.49-1.03 |
| melee, touched in front | 3901 | 20 | 145 | 179.0 | 1.23 | 1.07-1.40 |
| melee, touched in flank | 3309 | 20 | 184 | 248.0 | 1.35 | 1.11-1.59 |
| melee, touched in rear | 992 | 20 | 90 | 132.8 | 1.48 | 1.20-1.79 |
| melee, health < 30 % | 1132 | 20 | 225 | 274.0 | 1.22 | 1.02-1.39 |
| melee, health >= 30 % | 8826 | 20 | 259 | 331.0 | 1.28 | 1.07-1.50 |
| standing, health < 30 % | 4614 | 20 | 381 | 510.2 | 1.34 | 1.11-1.52 |
| lord | 2414 | 20 | 34 | 45.5 | 1.34 | 0.96-1.89 |
| missile | 3584 | 20 | 50 | 64.2 | 1.28 | 0.91-1.73 |
| infantry melee | 22807 | 20 | 599 | 805.0 | 1.34 | 1.14-1.53 |
| network's side | 13157 | 20 | 356 | 390.0 | 1.10 | 0.95-1.25 |
| the other side | 15648 | 20 | 327 | 524.8 | 1.60 | 1.28-1.93 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 254.8 | 0.69 | 0.62-0.74 |
| routers in melee | 830 | 20 | 36 | 33.0 | 0.92 | 0.74-1.07 |
| routers free | 6219 | 20 | 335 | 221.8 | 0.66 | 0.59-0.72 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.036 | -0.023 | -0.028 - -0.017 |
| standing, not in melee, no contact in window | 16032 | +0.008 | -0.021 | -0.030 | -0.035 - -0.025 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.112 | -0.045 | -0.083 - -0.006 |
| contact begins in window, ran in | 2265 | -0.034 | -0.058 | -0.024 | -0.033 - -0.015 |
| contact begins in window, walked in | 52 | +0.013 | +0.048 | +0.034 | -0.013 - +0.077 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.126 | -0.014 | -0.031 - +0.001 |
| melee, 10-30 s | 1948 | -0.050 | -0.057 | -0.007 | -0.020 - +0.008 |
| melee, 30 s+ | 6079 | -0.026 | -0.036 | -0.010 | -0.020 - +0.001 |
| melee, sim touches 1 enemy | 3993 | -0.046 | -0.077 | -0.031 | -0.044 - -0.020 |
| melee, sim touches 2+ enemies | 3790 | -0.051 | -0.053 | -0.003 | -0.016 - +0.012 |
| melee, sim touches nobody (game flag on) | 1691 | -0.025 | -0.001 | +0.024 | +0.006 - +0.039 |
| melee, touched in front | 3756 | -0.042 | -0.056 | -0.014 | -0.026 - -0.002 |
| melee, touched in flank | 3125 | -0.053 | -0.075 | -0.022 | -0.035 - -0.009 |
| melee, touched in rear | 902 | -0.057 | -0.073 | -0.015 | -0.037 - +0.008 |
| melee, health < 30 % | 907 | -0.034 | -0.071 | -0.037 | -0.054 - -0.020 |
| melee, health >= 30 % | 8567 | -0.045 | -0.052 | -0.007 | -0.017 - +0.004 |
| standing, health < 30 % | 4233 | -0.009 | -0.045 | -0.035 | -0.047 - -0.026 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.048 | -0.029 | -0.034 - -0.024 |
| steady (3) | 3200 | -0.013 | -0.022 | -0.009 | -0.024 - +0.007 |
| shaken (4) | 4396 | -0.010 | -0.024 | -0.014 | -0.025 - +0.006 |
| wavering (5) | 2158 | +0.031 | +0.020 | -0.011 | -0.021 - +0.004 |
| mp < 0.25 | 2088 | +0.027 | +0.015 | -0.012 | -0.023 - +0.008 |
| mp 0.25-0.5 | 3827 | -0.005 | -0.014 | -0.008 | -0.021 - +0.013 |
| mp 0.5-0.75 | 4343 | -0.015 | -0.024 | -0.009 | -0.023 - +0.005 |
| mp >= 0.75 | 17864 | -0.020 | -0.050 | -0.030 | -0.036 - -0.025 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 11639 | 0.94 | 0.60-1.36 |
| hold, lord | 59 | 18 | 13200 | 8291 | 0.63 | 0.39-0.86 |
| move, formation | 450 | 17 | 31949 | 10715 | 0.34 | 0.26-0.42 |
| move, missile unit | 148 | 14 | 6065 | 758 | 0.13 | 0.05-0.30 |
| move, lord | 41 | 13 | 4562 | 835 | 0.18 | 0.04-0.40 |
| attack, formation | 8219 | 20 | 1923931 | 1469431 | 0.76 | 0.71-0.81 |
| attack, missile unit | 236 | 18 | 39992 | 24764 | 0.62 | 0.49-0.80 |
| attack, lord | 621 | 20 | 133139 | 106775 | 0.80 | 0.63-0.99 |

## ../../../build/contactgap/sr_pos_rect: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.92 (0.90-0.95); rout starts sim/game 1.22 (1.15-1.30), game 2008, sim 2457.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.86 | 0.85 | 0.84 | 0.84 | 0.85 | 0.85 | 0.86 | 0.86 | 0.87 | 0.88 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1155.8 | 0.92 | 0.90-0.95 | 0.96 | 0.94-0.99 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 69.5 | 1.17 | 1.10-1.24 | 1.24 | 1.15-1.34 | -0.10 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 152.0 | 0.89 | 0.86-0.91 | 1.02 | 0.97-1.08 | 0.20 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 144.6 | 0.94 | 0.90-0.98 | 0.92 | 0.88-0.96 | 0.10 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.4 | 0.83 | 0.73-0.95 | 0.79 | 0.66-0.95 | 0.01 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 124.9 | 0.85 | 0.83-0.89 | 0.88 | 0.84-0.92 | 0.22 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 170.8 | 0.82 | 0.80-0.85 | 0.86 | 0.82-0.89 | 0.38 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 450.3 | 0.91 | 0.88-0.95 | 0.94 | 0.90-0.99 | 0.46 |
| melee, sim touches 1 enemy | 10530 | 72 | 308.7 | 245.7 | 0.80 | 0.78-0.82 | 0.83 | 0.81-0.86 | 0.64 |
| melee, sim touches 2+ enemies | 15292 | 72 | 454.2 | 451.9 | 0.99 | 0.96-1.04 | 1.01 | 0.97-1.06 | 0.02 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 86.7 | 48.4 | 0.56 | 0.52-0.60 | 0.66 | 0.61-0.71 | 0.39 |
| melee, touched in front | 8822 | 72 | 220.1 | 189.1 | 0.86 | 0.83-0.89 | 0.91 | 0.87-0.95 | 0.31 |
| melee, touched in flank | 12297 | 72 | 375.4 | 355.1 | 0.95 | 0.92-0.98 | 0.96 | 0.93-1.00 | 0.21 |
| melee, touched in rear | 4703 | 72 | 167.5 | 153.3 | 0.92 | 0.87-0.97 | 0.92 | 0.87-0.97 | 0.14 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 65.4 | 1.03 | 0.98-1.09 | 1.12 | 1.05-1.20 | -0.02 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 680.5 | 0.87 | 0.84-0.90 | 0.89 | 0.87-0.93 | 1.07 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 59.1 | 1.20 | 1.06-1.37 | 1.17 | 1.02-1.34 | -0.10 |
| routing, free | 10755 | 72 | 22.6 | 34.3 | 1.51 | 1.35-1.70 | 1.57 | 1.36-1.83 | -0.12 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 84.8 | 0.99 | 0.94-1.04 | 1.07 | 1.01-1.14 | 0.01 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 88.7 | 1.09 | 1.01-1.17 | 1.09 | 1.01-1.18 | -0.07 |
| missile | 24324 | 72 | 316.4 | 274.1 | 0.87 | 0.83-0.90 | 0.90 | 0.86-0.94 | 0.43 |
| infantry melee | 57353 | 72 | 856.7 | 793.0 | 0.93 | 0.90-0.96 | 0.97 | 0.94-1.00 | 0.64 |
| network's side | 43836 | 72 | 729.1 | 677.8 | 0.93 | 0.90-0.96 | 0.96 | 0.92-1.00 | 0.52 |
| the other side | 44434 | 72 | 525.7 | 478.0 | 0.91 | 0.87-0.95 | 0.96 | 0.92-1.01 | 0.48 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 61.7 | 1.07 | 1.00-1.14 | 1.09 | 1.00-1.17 | -0.04 |
| lord, routing | 511 | 54 | 1.6 | 2.1 | 1.35 | 1.01-1.85 | 1.54 | 1.05-2.22 | -0.01 |
| missile, melee | 4055 | 72 | 184.5 | 151.5 | 0.82 | 0.78-0.86 | 0.85 | 0.81-0.89 | 0.33 |
| missile, routing | 3025 | 72 | 18.3 | 24.8 | 1.35 | 1.11-1.66 | 1.35 | 1.07-1.69 | -0.07 |
| infantry melee, melee | 23441 | 72 | 607.5 | 532.7 | 0.88 | 0.85-0.91 | 0.91 | 0.88-0.95 | 0.76 |
| infantry melee, routing | 10365 | 72 | 52.0 | 66.5 | 1.28 | 1.12-1.48 | 1.28 | 1.11-1.51 | -0.15 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 80.0 | 5.33 | 3.13-10.56 |
| steady (3) | 7127 | 72 | 28 | 73.8 | 2.63 | 1.75-4.06 |
| shaken (4) | 11458 | 72 | 477 | 802.0 | 1.68 | 1.51-1.88 |
| wavering (5) | 5568 | 72 | 1453 | 1481.8 | 1.02 | 0.96-1.08 |
| mp < 0.25 | 5043 | 72 | 1480 | 1453.2 | 0.98 | 0.93-1.04 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 692.2 | 1.57 | 1.40-1.76 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 224.8 | 3.35 | 2.70-4.29 |
| mp >= 0.75 | 48908 | 72 | 20 | 86.8 | 4.34 | 2.62-7.57 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 2457.0 | 1.22 | 1.15-1.30 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 349.8 | 1.31 | 1.08-1.54 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 250.0 | 1.31 | 1.18-1.44 |
| contact begins in window, ran in | 6699 | 72 | 124 | 194.0 | 1.56 | 1.35-1.86 |
| contact begins in window, walked in | 271 | 68 | 6 | 9.5 | 1.58 | 1.14-3.17 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 219.0 | 1.34 | 1.21-1.51 |
| melee, 10-30 s | 6160 | 72 | 304 | 360.2 | 1.19 | 1.06-1.30 |
| melee, 30 s+ | 19508 | 72 | 983 | 1131.0 | 1.15 | 1.05-1.26 |
| melee, sim touches 1 enemy | 10530 | 72 | 539 | 690.2 | 1.28 | 1.19-1.38 |
| melee, sim touches 2+ enemies | 15292 | 72 | 759 | 922.0 | 1.21 | 1.09-1.35 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 152 | 98.0 | 0.64 | 0.52-0.79 |
| melee, touched in front | 8822 | 72 | 322 | 355.5 | 1.10 | 1.00-1.23 |
| melee, touched in flank | 12297 | 72 | 627 | 785.8 | 1.25 | 1.15-1.35 |
| melee, touched in rear | 4703 | 72 | 349 | 471.0 | 1.35 | 1.18-1.54 |
| melee, health < 30 % | 3088 | 72 | 618 | 684.8 | 1.11 | 1.04-1.18 |
| melee, health >= 30 % | 26784 | 72 | 832 | 1025.5 | 1.23 | 1.11-1.36 |
| standing, health < 30 % | 8705 | 72 | 1028 | 1156.5 | 1.12 | 1.05-1.20 |
| lord | 6082 | 72 | 100 | 84.2 | 0.84 | 0.63-1.07 |
| missile | 21299 | 72 | 541 | 745.5 | 1.38 | 1.26-1.49 |
| infantry melee | 46988 | 72 | 1367 | 1627.2 | 1.19 | 1.11-1.28 |
| network's side | 34608 | 72 | 1389 | 1692.5 | 1.22 | 1.13-1.31 |
| the other side | 39761 | 72 | 619 | 764.5 | 1.24 | 1.11-1.38 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1088.2 | 0.93 | 0.88-0.98 |
| routers in melee | 3146 | 72 | 90 | 129.2 | 1.44 | 1.21-1.74 |
| routers free | 10755 | 72 | 1080 | 959.0 | 0.89 | 0.84-0.94 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.048 | -0.031 | -0.035 - -0.028 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.037 | -0.051 | -0.057 - -0.046 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.137 | -0.055 | -0.062 - -0.046 |
| contact begins in window, ran in | 6575 | -0.050 | -0.109 | -0.060 | -0.068 - -0.051 |
| contact begins in window, walked in | 265 | -0.065 | -0.101 | -0.036 | -0.061 - -0.015 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.155 | -0.043 | -0.051 - -0.036 |
| melee, 10-30 s | 5856 | -0.058 | -0.079 | -0.021 | -0.028 - -0.014 |
| melee, 30 s+ | 18525 | -0.023 | -0.003 | +0.019 | +0.013 - +0.026 |
| melee, sim touches 1 enemy | 9991 | -0.046 | -0.062 | -0.015 | -0.024 - -0.007 |
| melee, sim touches 2+ enemies | 14533 | -0.045 | -0.038 | +0.007 | +0.000 - +0.013 |
| melee, sim touches nobody (game flag on) | 3898 | -0.024 | +0.006 | +0.030 | +0.019 - +0.041 |
| melee, touched in front | 8500 | -0.039 | -0.027 | +0.012 | +0.003 - +0.019 |
| melee, touched in flank | 11670 | -0.047 | -0.055 | -0.007 | -0.014 - -0.001 |
| melee, touched in rear | 4354 | -0.053 | -0.070 | -0.017 | -0.030 - -0.005 |
| melee, health < 30 % | 2470 | -0.036 | -0.046 | -0.011 | -0.024 - +0.002 |
| melee, health >= 30 % | 25952 | -0.043 | -0.040 | +0.003 | -0.002 - +0.009 |
| standing, health < 30 % | 7677 | -0.013 | -0.061 | -0.048 | -0.063 - -0.035 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.065 | -0.043 | -0.046 - -0.039 |
| steady (3) | 7099 | -0.025 | -0.019 | +0.006 | -0.002 - +0.013 |
| shaken (4) | 10981 | -0.011 | -0.021 | -0.010 | -0.019 - -0.003 |
| wavering (5) | 4115 | +0.042 | +0.032 | -0.010 | -0.028 - +0.005 |
| mp < 0.25 | 3563 | +0.040 | +0.042 | +0.002 | -0.019 - +0.020 |
| mp 0.25-0.5 | 8520 | -0.001 | +0.002 | +0.003 | -0.003 - +0.009 |
| mp 0.5-0.75 | 11390 | -0.024 | -0.031 | -0.007 | -0.015 - +0.000 |
| mp >= 0.75 | 48888 | -0.022 | -0.067 | -0.045 | -0.049 - -0.042 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 42326 | 0.36 | 0.31-0.41 |
| hold, missile unit | 1416 | 72 | 213822 | 119611 | 0.56 | 0.49-0.63 |
| hold, lord | 537 | 69 | 145512 | 67194 | 0.46 | 0.40-0.53 |
| move, formation | 1035 | 71 | 151893 | 46036 | 0.30 | 0.26-0.36 |
| move, missile unit | 751 | 69 | 78425 | 20217 | 0.26 | 0.20-0.34 |
| move, lord | 49 | 32 | 7295 | 2841 | 0.39 | 0.23-0.61 |
| attack, formation | 21672 | 72 | 5277048 | 3284159 | 0.62 | 0.59-0.65 |
| attack, missile unit | 1888 | 71 | 332739 | 206968 | 0.62 | 0.56-0.68 |
| attack, lord | 1790 | 72 | 422482 | 278706 | 0.66 | 0.60-0.72 |
