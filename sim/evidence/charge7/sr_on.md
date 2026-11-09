## C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/build/charge7/sr_on: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 1.04 (0.95-1.13); rout starts sim/game 1.10 (0.90-1.29), game 683, sim 749.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.93 | 0.92 | 0.92 | 0.93 | 0.93 | 0.95 | 0.97 | 0.98 | 1.00 | 1.01 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 391.2 | 1.04 | 0.95-1.13 | 1.10 | 1.01-1.19 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 29.4 | 1.48 | 1.22-1.74 | 1.60 | 1.29-1.92 | 0.65 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 25.0 | 1.00 | 0.81-1.23 | 1.15 | 0.93-1.42 | -0.01 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 46.0 | 0.92 | 0.82-1.03 | 0.92 | 0.81-1.04 | -0.29 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.6 | 0.50 | 0.35-0.70 | 0.62 | 0.41-0.88 | -0.04 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 45.1 | 0.94 | 0.85-1.05 | 1.02 | 0.90-1.16 | -0.20 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 57.4 | 0.95 | 0.87-1.05 | 1.01 | 0.91-1.11 | -0.19 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 168.3 | 1.04 | 0.98-1.11 | 1.13 | 1.07-1.20 | 0.49 |
| melee, sim touches 1 enemy | 4186 | 20 | 111.5 | 107.1 | 0.96 | 0.89-1.05 | 1.04 | 0.96-1.13 | -0.30 |
| melee, sim touches 2+ enemies | 4016 | 20 | 126.7 | 134.3 | 1.06 | 1.01-1.10 | 1.12 | 1.06-1.18 | 0.53 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 31.1 | 29.4 | 0.95 | 0.80-1.11 | 1.10 | 0.94-1.26 | -0.11 |
| melee, touched in front | 3901 | 20 | 99.2 | 96.5 | 0.97 | 0.92-1.04 | 1.06 | 0.98-1.14 | -0.18 |
| melee, touched in flank | 3309 | 20 | 103.9 | 107.6 | 1.04 | 0.98-1.09 | 1.11 | 1.05-1.16 | 0.25 |
| melee, touched in rear | 992 | 20 | 35.1 | 37.4 | 1.06 | 0.97-1.15 | 1.09 | 0.96-1.19 | 0.15 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 28.2 | 1.20 | 1.06-1.34 | 1.36 | 1.20-1.52 | 0.33 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 242.6 | 0.99 | 0.93-1.05 | 1.06 | 1.00-1.12 | -0.22 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 20.5 | 1.45 | 1.16-1.76 | 1.36 | 0.98-1.82 | 0.44 |
| routing, free | 6219 | 20 | 5.5 | 7.8 | 1.42 | 0.95-2.20 | 1.43 | 0.88-2.25 | 0.16 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 38.9 | 1.19 | 1.04-1.35 | 1.38 | 1.21-1.56 | 0.43 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 27.2 | 0.94 | 0.84-1.04 | 0.98 | 0.84-1.13 | -0.13 |
| missile | 3989 | 20 | 31.8 | 32.4 | 1.02 | 0.86-1.20 | 1.08 | 0.88-1.32 | 0.04 |
| infantry melee | 29215 | 20 | 315.8 | 331.6 | 1.05 | 0.96-1.15 | 1.12 | 1.03-1.21 | 1.08 |
| network's side | 16523 | 20 | 184.7 | 179.1 | 0.97 | 0.91-1.03 | 1.01 | 0.95-1.08 | -0.38 |
| the other side | 19331 | 20 | 191.9 | 212.1 | 1.11 | 0.98-1.24 | 1.19 | 1.05-1.33 | 1.38 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 18.7 | 0.93 | 0.84-1.02 | 0.94 | 0.81-1.08 | -0.10 |
| lord, routing | 236 | 15 | 0.9 | 0.5 | 0.52 | 0.26-0.99 | 1.02 | 0.35-3.39 | -0.03 |
| missile, melee | 413 | 18 | 15.9 | 16.5 | 1.04 | 0.93-1.15 | 1.05 | 0.93-1.18 | 0.04 |
| missile, routing | 405 | 16 | 2.2 | 4.7 | 2.14 | 1.10-3.51 | 2.53 | 0.98-4.55 | 0.17 |
| infantry melee, melee | 8824 | 20 | 233.2 | 235.6 | 1.01 | 0.95-1.08 | 1.10 | 1.03-1.17 | 0.17 |
| infantry melee, routing | 6408 | 20 | 16.5 | 23.1 | 1.40 | 1.11-1.70 | 1.26 | 0.92-1.63 | 0.45 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 13.0 | 2.17 | 0.67-10.00 |
| steady (3) | 3212 | 20 | 12 | 33.2 | 2.77 | 1.22-5.63 |
| shaken (4) | 4549 | 20 | 153 | 228.2 | 1.49 | 1.13-1.83 |
| wavering (5) | 2661 | 20 | 503 | 468.5 | 0.93 | 0.79-1.10 |
| mp < 0.25 | 2598 | 20 | 510 | 492.8 | 0.97 | 0.80-1.17 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 195.2 | 1.39 | 1.14-1.68 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 51.2 | 1.90 | 1.01-3.25 |
| mp >= 0.75 | 17870 | 20 | 6 | 9.8 | 1.62 | 0.63-6.75 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 749.0 | 1.10 | 0.90-1.29 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 133.2 | 1.18 | 0.75-1.55 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 51.5 | 1.10 | 0.84-1.64 |
| contact begins in window, ran in | 2307 | 20 | 42 | 60.2 | 1.43 | 1.08-1.91 |
| contact begins in window, walked in | 56 | 15 | 4 | 3.2 | 0.81 | 0.25-2.12 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 92.0 | 1.05 | 0.86-1.25 |
| melee, 10-30 s | 2047 | 20 | 99 | 119.5 | 1.21 | 0.98-1.42 |
| melee, 30 s+ | 6376 | 20 | 297 | 301.8 | 1.02 | 0.82-1.23 |
| melee, sim touches 1 enemy | 4186 | 20 | 193 | 239.2 | 1.24 | 0.96-1.61 |
| melee, sim touches 2+ enemies | 4016 | 20 | 226 | 225.2 | 1.00 | 0.86-1.13 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 65 | 48.8 | 0.75 | 0.57-1.05 |
| melee, touched in front | 3901 | 20 | 145 | 156.8 | 1.08 | 0.86-1.34 |
| melee, touched in flank | 3309 | 20 | 184 | 201.0 | 1.09 | 0.90-1.31 |
| melee, touched in rear | 992 | 20 | 90 | 106.8 | 1.19 | 0.95-1.48 |
| melee, health < 30 % | 1132 | 20 | 225 | 252.2 | 1.12 | 0.91-1.36 |
| melee, health >= 30 % | 8826 | 20 | 259 | 261.0 | 1.01 | 0.84-1.22 |
| standing, health < 30 % | 4614 | 20 | 381 | 425.2 | 1.12 | 0.92-1.34 |
| lord | 2414 | 20 | 34 | 40.0 | 1.18 | 0.80-1.80 |
| missile | 3584 | 20 | 50 | 51.5 | 1.03 | 0.79-1.32 |
| infantry melee | 22807 | 20 | 599 | 657.5 | 1.10 | 0.89-1.32 |
| network's side | 13157 | 20 | 356 | 298.8 | 0.84 | 0.73-0.98 |
| the other side | 15648 | 20 | 327 | 450.2 | 1.38 | 1.08-1.74 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 295.2 | 0.80 | 0.72-0.84 |
| routers in melee | 830 | 20 | 36 | 38.8 | 1.08 | 0.93-1.26 |
| routers free | 6219 | 20 | 335 | 256.5 | 0.77 | 0.68-0.82 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.011 | +0.003 | -0.004 - +0.010 |
| standing, not in melee, no contact in window | 16032 | +0.008 | +0.002 | -0.007 | -0.011 - -0.002 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.084 | -0.017 | -0.055 - +0.024 |
| contact begins in window, ran in | 2265 | -0.034 | -0.023 | +0.011 | +0.000 - +0.022 |
| contact begins in window, walked in | 52 | +0.013 | +0.056 | +0.043 | +0.007 - +0.076 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.090 | +0.022 | +0.005 - +0.036 |
| melee, 10-30 s | 1948 | -0.050 | -0.033 | +0.016 | +0.004 - +0.030 |
| melee, 30 s+ | 6079 | -0.026 | -0.010 | +0.016 | +0.003 - +0.031 |
| melee, sim touches 1 enemy | 3993 | -0.046 | -0.050 | -0.004 | -0.017 - +0.008 |
| melee, sim touches 2+ enemies | 3790 | -0.051 | -0.028 | +0.023 | +0.008 - +0.040 |
| melee, sim touches nobody (game flag on) | 1691 | -0.025 | +0.029 | +0.054 | +0.036 - +0.071 |
| melee, touched in front | 3756 | -0.042 | -0.033 | +0.009 | -0.002 - +0.022 |
| melee, touched in flank | 3125 | -0.053 | -0.048 | +0.005 | -0.007 - +0.018 |
| melee, touched in rear | 902 | -0.057 | -0.037 | +0.021 | -0.002 - +0.046 |
| melee, health < 30 % | 907 | -0.034 | -0.033 | +0.000 | -0.014 - +0.016 |
| melee, health >= 30 % | 8567 | -0.045 | -0.026 | +0.019 | +0.006 - +0.032 |
| standing, health < 30 % | 4233 | -0.009 | -0.015 | -0.006 | -0.015 - +0.007 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.025 | -0.006 | -0.011 - -0.000 |
| steady (3) | 3200 | -0.013 | +0.009 | +0.022 | +0.010 - +0.038 |
| shaken (4) | 4396 | -0.010 | +0.005 | +0.015 | -0.003 - +0.040 |
| wavering (5) | 2158 | +0.031 | +0.052 | +0.021 | +0.003 - +0.042 |
| mp < 0.25 | 2088 | +0.027 | +0.039 | +0.012 | -0.007 - +0.040 |
| mp 0.25-0.5 | 3827 | -0.005 | +0.014 | +0.019 | -0.001 - +0.048 |
| mp 0.5-0.75 | 4343 | -0.015 | +0.011 | +0.025 | +0.015 - +0.037 |
| mp >= 0.75 | 17864 | -0.020 | -0.027 | -0.007 | -0.013 - -0.002 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 12261 | 0.99 | 0.70-1.35 |
| hold, lord | 59 | 18 | 13200 | 7769 | 0.59 | 0.39-0.79 |
| move, formation | 450 | 17 | 31949 | 12882 | 0.40 | 0.32-0.49 |
| move, missile unit | 148 | 14 | 6065 | 966 | 0.16 | 0.07-0.37 |
| move, lord | 41 | 13 | 4562 | 744 | 0.16 | 0.04-0.35 |
| attack, formation | 8219 | 20 | 1923931 | 1569041 | 0.82 | 0.76-0.86 |
| attack, missile unit | 236 | 18 | 39992 | 26728 | 0.67 | 0.54-0.84 |
| attack, lord | 621 | 20 | 133139 | 114600 | 0.86 | 0.68-1.05 |

## C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/build/charge7/sr_on: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.97 (0.95-1.00); rout starts sim/game 1.03 (0.96-1.10), game 2008, sim 2063.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.85 | 0.85 | 0.85 | 0.87 | 0.88 | 0.89 | 0.91 | 0.92 | 0.93 | 0.93 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1220.2 | 0.97 | 0.95-1.00 | 1.03 | 1.00-1.06 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 81.7 | 1.37 | 1.30-1.46 | 1.46 | 1.35-1.58 | -0.64 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 162.3 | 0.95 | 0.92-0.97 | 1.08 | 1.03-1.13 | 0.26 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 149.5 | 0.97 | 0.93-1.01 | 0.95 | 0.91-0.99 | 0.13 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.7 | 0.87 | 0.76-1.01 | 0.85 | 0.73-1.00 | 0.03 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 133.0 | 0.91 | 0.88-0.94 | 0.96 | 0.92-1.00 | 0.38 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 182.9 | 0.88 | 0.85-0.91 | 0.94 | 0.90-0.98 | 0.73 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 478.1 | 0.97 | 0.93-1.01 | 1.03 | 0.99-1.08 | 0.50 |
| melee, sim touches 1 enemy | 10530 | 72 | 308.7 | 261.8 | 0.85 | 0.83-0.87 | 0.90 | 0.87-0.94 | 1.36 |
| melee, sim touches 2+ enemies | 15292 | 72 | 454.2 | 471.5 | 1.04 | 1.00-1.08 | 1.08 | 1.04-1.13 | -0.50 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 86.7 | 60.7 | 0.70 | 0.66-0.75 | 0.85 | 0.79-0.91 | 0.75 |
| melee, touched in front | 8822 | 72 | 220.1 | 199.1 | 0.90 | 0.88-0.94 | 0.97 | 0.94-1.01 | 0.61 |
| melee, touched in flank | 12297 | 72 | 375.4 | 371.8 | 0.99 | 0.96-1.03 | 1.04 | 1.00-1.08 | 0.10 |
| melee, touched in rear | 4703 | 72 | 167.5 | 162.4 | 0.97 | 0.92-1.03 | 1.00 | 0.94-1.07 | 0.15 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 70.8 | 1.11 | 1.06-1.17 | 1.22 | 1.14-1.30 | -0.21 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 723.2 | 0.92 | 0.89-0.95 | 0.98 | 0.94-1.01 | 1.82 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 55.1 | 1.12 | 0.98-1.28 | 1.04 | 0.90-1.21 | -0.17 |
| routing, free | 10755 | 72 | 22.6 | 29.4 | 1.30 | 1.14-1.48 | 1.30 | 1.10-1.53 | -0.20 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 93.4 | 1.09 | 1.04-1.14 | 1.17 | 1.11-1.24 | -0.22 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 93.5 | 1.15 | 1.06-1.23 | 1.14 | 1.05-1.24 | -0.34 |
| missile | 24324 | 72 | 316.4 | 297.3 | 0.94 | 0.90-0.97 | 1.00 | 0.96-1.05 | 0.55 |
| infantry melee | 57353 | 72 | 856.7 | 829.4 | 0.97 | 0.94-1.00 | 1.03 | 1.00-1.07 | 0.79 |
| network's side | 43836 | 72 | 729.1 | 726.2 | 1.00 | 0.96-1.04 | 1.06 | 1.02-1.11 | 0.08 |
| the other side | 44434 | 72 | 525.7 | 494.0 | 0.94 | 0.90-0.98 | 0.99 | 0.95-1.03 | 0.92 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 63.8 | 1.11 | 1.03-1.19 | 1.13 | 1.02-1.23 | -0.18 |
| lord, routing | 511 | 54 | 1.6 | 2.1 | 1.35 | 0.98-1.85 | 1.41 | 0.99-1.90 | -0.02 |
| missile, melee | 4055 | 72 | 184.5 | 165.4 | 0.90 | 0.86-0.94 | 0.97 | 0.92-1.02 | 0.55 |
| missile, routing | 3025 | 72 | 18.3 | 22.4 | 1.22 | 0.99-1.50 | 1.12 | 0.88-1.41 | -0.12 |
| infantry melee, melee | 23441 | 72 | 607.5 | 564.8 | 0.93 | 0.90-0.97 | 0.99 | 0.95-1.03 | 1.24 |
| infantry melee, routing | 10365 | 72 | 52.0 | 60.0 | 1.15 | 1.00-1.35 | 1.12 | 0.95-1.34 | -0.23 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 61.8 | 4.12 | 2.48-8.03 |
| steady (3) | 7127 | 72 | 28 | 77.8 | 2.78 | 1.78-4.44 |
| shaken (4) | 11458 | 72 | 477 | 665.5 | 1.40 | 1.24-1.55 |
| wavering (5) | 5568 | 72 | 1453 | 1239.2 | 0.85 | 0.80-0.91 |
| mp < 0.25 | 5043 | 72 | 1480 | 1218.0 | 0.82 | 0.77-0.88 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 582.2 | 1.32 | 1.17-1.47 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 194.2 | 2.90 | 2.25-3.76 |
| mp >= 0.75 | 48908 | 72 | 20 | 68.5 | 3.42 | 2.12-6.40 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 2063.0 | 1.03 | 0.96-1.10 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 292.2 | 1.09 | 0.88-1.31 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 199.2 | 1.04 | 0.92-1.17 |
| contact begins in window, ran in | 6699 | 72 | 124 | 159.0 | 1.28 | 1.06-1.55 |
| contact begins in window, walked in | 271 | 68 | 6 | 6.2 | 1.04 | 0.45-2.33 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 205.8 | 1.26 | 1.14-1.42 |
| melee, 10-30 s | 6160 | 72 | 304 | 309.0 | 1.02 | 0.91-1.12 |
| melee, 30 s+ | 19508 | 72 | 983 | 939.0 | 0.96 | 0.87-1.05 |
| melee, sim touches 1 enemy | 10530 | 72 | 539 | 555.8 | 1.03 | 0.96-1.10 |
| melee, sim touches 2+ enemies | 15292 | 72 | 759 | 798.2 | 1.05 | 0.95-1.16 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 152 | 99.8 | 0.66 | 0.54-0.78 |
| melee, touched in front | 8822 | 72 | 322 | 292.0 | 0.91 | 0.82-1.00 |
| melee, touched in flank | 12297 | 72 | 627 | 672.2 | 1.07 | 0.98-1.17 |
| melee, touched in rear | 4703 | 72 | 349 | 389.8 | 1.12 | 0.99-1.26 |
| melee, health < 30 % | 3088 | 72 | 618 | 588.0 | 0.95 | 0.89-1.02 |
| melee, health >= 30 % | 26784 | 72 | 832 | 865.8 | 1.04 | 0.94-1.15 |
| standing, health < 30 % | 8705 | 72 | 1028 | 988.0 | 0.96 | 0.88-1.03 |
| lord | 6082 | 72 | 100 | 88.2 | 0.88 | 0.71-1.08 |
| missile | 21299 | 72 | 541 | 598.2 | 1.11 | 1.00-1.22 |
| infantry melee | 46988 | 72 | 1367 | 1376.5 | 1.01 | 0.93-1.09 |
| network's side | 34608 | 72 | 1389 | 1467.5 | 1.06 | 0.97-1.14 |
| the other side | 39761 | 72 | 619 | 595.5 | 0.96 | 0.87-1.07 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1244.2 | 1.06 | 1.01-1.12 |
| routers in melee | 3146 | 72 | 90 | 157.5 | 1.75 | 1.50-2.08 |
| routers free | 10755 | 72 | 1080 | 1086.8 | 1.01 | 0.96-1.06 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.015 | +0.002 | -0.002 - +0.006 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.003 | -0.017 | -0.021 - -0.012 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.089 | -0.007 | -0.015 - +0.002 |
| contact begins in window, ran in | 6575 | -0.050 | -0.064 | -0.014 | -0.023 - -0.005 |
| contact begins in window, walked in | 265 | -0.065 | -0.092 | -0.027 | -0.059 - +0.004 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.116 | -0.004 | -0.013 - +0.005 |
| melee, 10-30 s | 5856 | -0.058 | -0.047 | +0.010 | +0.003 - +0.018 |
| melee, 30 s+ | 18525 | -0.023 | +0.021 | +0.043 | +0.036 - +0.050 |
| melee, sim touches 1 enemy | 9991 | -0.046 | -0.027 | +0.019 | +0.012 - +0.026 |
| melee, sim touches 2+ enemies | 14533 | -0.045 | -0.021 | +0.024 | +0.017 - +0.032 |
| melee, sim touches nobody (game flag on) | 3898 | -0.024 | +0.055 | +0.078 | +0.066 - +0.089 |
| melee, touched in front | 8500 | -0.039 | -0.002 | +0.037 | +0.029 - +0.044 |
| melee, touched in flank | 11670 | -0.047 | -0.030 | +0.017 | +0.009 - +0.025 |
| melee, touched in rear | 4354 | -0.053 | -0.047 | +0.006 | -0.006 - +0.017 |
| melee, health < 30 % | 2470 | -0.036 | -0.025 | +0.010 | -0.004 - +0.023 |
| melee, health >= 30 % | 25952 | -0.043 | -0.012 | +0.032 | +0.025 - +0.038 |
| standing, health < 30 % | 7677 | -0.013 | -0.022 | -0.010 | -0.024 - +0.003 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.030 | -0.008 | -0.012 - -0.005 |
| steady (3) | 7099 | -0.025 | +0.011 | +0.036 | +0.029 - +0.043 |
| shaken (4) | 10981 | -0.011 | +0.012 | +0.023 | +0.013 - +0.032 |
| wavering (5) | 4115 | +0.042 | +0.056 | +0.014 | -0.007 - +0.030 |
| mp < 0.25 | 3563 | +0.040 | +0.061 | +0.020 | -0.004 - +0.040 |
| mp 0.25-0.5 | 8520 | -0.001 | +0.031 | +0.032 | +0.024 - +0.039 |
| mp 0.5-0.75 | 11390 | -0.024 | +0.004 | +0.028 | +0.020 - +0.036 |
| mp >= 0.75 | 48888 | -0.022 | -0.032 | -0.010 | -0.014 - -0.007 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 43711 | 0.37 | 0.32-0.42 |
| hold, missile unit | 1416 | 72 | 213822 | 125114 | 0.59 | 0.52-0.66 |
| hold, lord | 537 | 69 | 145512 | 59254 | 0.41 | 0.35-0.48 |
| move, formation | 1035 | 71 | 151893 | 47855 | 0.32 | 0.27-0.37 |
| move, missile unit | 751 | 69 | 78425 | 24788 | 0.32 | 0.25-0.41 |
| move, lord | 49 | 32 | 7295 | 2771 | 0.38 | 0.22-0.59 |
| attack, formation | 21672 | 72 | 5277048 | 3437997 | 0.65 | 0.62-0.69 |
| attack, missile unit | 1888 | 71 | 332739 | 207740 | 0.62 | 0.56-0.68 |
| attack, lord | 1790 | 72 | 422482 | 301216 | 0.71 | 0.65-0.77 |
