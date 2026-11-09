## C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/build/charge7/sr_off: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 1.03 (0.94-1.12); rout starts sim/game 1.09 (0.88-1.29), game 683, sim 744.8.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 | 0.93 | 0.96 | 0.97 | 0.98 | 0.99 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 386.5 | 1.03 | 0.94-1.12 | 1.09 | 1.00-1.18 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 29.0 | 1.46 | 1.21-1.71 | 1.60 | 1.29-1.93 | 0.92 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 24.8 | 0.99 | 0.80-1.22 | 1.14 | 0.91-1.41 | -0.03 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 45.8 | 0.91 | 0.81-1.03 | 0.91 | 0.81-1.05 | -0.43 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.5 | 0.44 | 0.26-0.66 | 0.55 | 0.34-0.82 | -0.07 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 43.5 | 0.91 | 0.82-1.01 | 0.97 | 0.86-1.10 | -0.46 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 56.8 | 0.94 | 0.87-1.04 | 1.00 | 0.91-1.11 | -0.34 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 166.6 | 1.03 | 0.97-1.09 | 1.12 | 1.05-1.18 | 0.56 |
| melee, sim touches 1 enemy | 4186 | 20 | 111.5 | 105.4 | 0.94 | 0.87-1.03 | 1.00 | 0.92-1.10 | -0.62 |
| melee, sim touches 2+ enemies | 4016 | 20 | 126.7 | 132.5 | 1.05 | 0.99-1.09 | 1.11 | 1.05-1.16 | 0.59 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 31.1 | 29.0 | 0.93 | 0.78-1.10 | 1.09 | 0.93-1.25 | -0.21 |
| melee, touched in front | 3901 | 20 | 99.2 | 95.0 | 0.96 | 0.90-1.02 | 1.02 | 0.95-1.11 | -0.43 |
| melee, touched in flank | 3309 | 20 | 103.9 | 105.8 | 1.02 | 0.97-1.07 | 1.09 | 1.03-1.14 | 0.20 |
| melee, touched in rear | 992 | 20 | 35.1 | 37.0 | 1.05 | 0.96-1.13 | 1.10 | 0.98-1.20 | 0.19 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 27.8 | 1.19 | 1.05-1.32 | 1.34 | 1.19-1.50 | 0.44 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 239.0 | 0.97 | 0.91-1.03 | 1.04 | 0.98-1.10 | -0.69 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 20.5 | 1.45 | 1.16-1.75 | 1.38 | 0.99-1.83 | 0.65 |
| routing, free | 6219 | 20 | 5.5 | 7.8 | 1.41 | 0.95-2.20 | 1.39 | 0.85-2.16 | 0.23 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 38.6 | 1.18 | 1.03-1.34 | 1.37 | 1.21-1.56 | 0.60 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 26.8 | 0.92 | 0.81-1.05 | 0.96 | 0.83-1.10 | -0.23 |
| missile | 3989 | 20 | 31.8 | 32.3 | 1.02 | 0.86-1.18 | 1.06 | 0.86-1.31 | 0.05 |
| infantry melee | 29215 | 20 | 315.8 | 327.4 | 1.04 | 0.95-1.13 | 1.10 | 1.01-1.20 | 1.18 |
| network's side | 16523 | 20 | 184.7 | 176.2 | 0.95 | 0.90-1.01 | 0.99 | 0.93-1.06 | -0.86 |
| the other side | 19331 | 20 | 191.9 | 210.3 | 1.10 | 0.97-1.23 | 1.18 | 1.04-1.32 | 1.86 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 18.1 | 0.90 | 0.80-1.00 | 0.90 | 0.76-1.03 | -0.21 |
| lord, routing | 236 | 15 | 0.9 | 0.5 | 0.52 | 0.30-0.93 | 0.87 | 0.27-2.88 | -0.04 |
| missile, melee | 413 | 18 | 15.9 | 16.1 | 1.02 | 0.92-1.12 | 1.00 | 0.89-1.14 | 0.02 |
| missile, routing | 405 | 16 | 2.2 | 4.8 | 2.16 | 1.11-3.49 | 2.58 | 1.01-4.59 | 0.26 |
| infantry melee, melee | 8824 | 20 | 233.2 | 232.6 | 1.00 | 0.93-1.06 | 1.08 | 1.01-1.15 | -0.07 |
| infantry melee, routing | 6408 | 20 | 16.5 | 23.0 | 1.39 | 1.11-1.68 | 1.26 | 0.93-1.62 | 0.66 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 11.0 | 1.83 | 0.46-8.75 |
| steady (3) | 3212 | 20 | 12 | 33.2 | 2.77 | 1.32-5.46 |
| shaken (4) | 4549 | 20 | 153 | 227.8 | 1.49 | 1.10-1.86 |
| wavering (5) | 2661 | 20 | 503 | 466.8 | 0.93 | 0.78-1.09 |
| mp < 0.25 | 2598 | 20 | 510 | 488.8 | 0.96 | 0.79-1.16 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 194.8 | 1.39 | 1.13-1.67 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 53.2 | 1.97 | 0.98-3.42 |
| mp >= 0.75 | 17870 | 20 | 6 | 8.0 | 1.33 | 0.40-5.75 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 744.8 | 1.09 | 0.88-1.29 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 133.2 | 1.18 | 0.76-1.53 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 51.8 | 1.10 | 0.84-1.63 |
| contact begins in window, ran in | 2307 | 20 | 42 | 61.5 | 1.46 | 1.08-1.94 |
| contact begins in window, walked in | 56 | 15 | 4 | 4.0 | 1.00 | 0.33-2.50 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 89.2 | 1.01 | 0.83-1.21 |
| melee, 10-30 s | 2047 | 20 | 99 | 117.2 | 1.18 | 0.99-1.38 |
| melee, 30 s+ | 6376 | 20 | 297 | 299.5 | 1.01 | 0.78-1.25 |
| melee, sim touches 1 enemy | 4186 | 20 | 193 | 236.2 | 1.22 | 0.94-1.62 |
| melee, sim touches 2+ enemies | 4016 | 20 | 226 | 223.2 | 0.99 | 0.82-1.14 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 65 | 46.5 | 0.72 | 0.57-0.97 |
| melee, touched in front | 3901 | 20 | 145 | 155.2 | 1.07 | 0.85-1.35 |
| melee, touched in flank | 3309 | 20 | 184 | 200.0 | 1.09 | 0.86-1.33 |
| melee, touched in rear | 992 | 20 | 90 | 104.2 | 1.16 | 0.94-1.42 |
| melee, health < 30 % | 1132 | 20 | 225 | 246.0 | 1.09 | 0.87-1.35 |
| melee, health >= 30 % | 8826 | 20 | 259 | 260.0 | 1.00 | 0.82-1.23 |
| standing, health < 30 % | 4614 | 20 | 381 | 420.8 | 1.10 | 0.89-1.34 |
| lord | 2414 | 20 | 34 | 41.5 | 1.22 | 0.89-1.79 |
| missile | 3584 | 20 | 50 | 50.5 | 1.01 | 0.78-1.30 |
| infantry melee | 22807 | 20 | 599 | 652.8 | 1.09 | 0.87-1.32 |
| network's side | 13157 | 20 | 356 | 297.0 | 0.83 | 0.72-0.97 |
| the other side | 15648 | 20 | 327 | 447.8 | 1.37 | 1.05-1.73 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 295.2 | 0.80 | 0.73-0.84 |
| routers in melee | 830 | 20 | 36 | 38.8 | 1.08 | 0.93-1.26 |
| routers free | 6219 | 20 | 335 | 256.5 | 0.77 | 0.68-0.82 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.010 | +0.003 | -0.004 - +0.010 |
| standing, not in melee, no contact in window | 16032 | +0.008 | +0.002 | -0.006 | -0.011 - -0.002 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.082 | -0.015 | -0.053 - +0.027 |
| contact begins in window, ran in | 2265 | -0.034 | -0.022 | +0.012 | +0.000 - +0.024 |
| contact begins in window, walked in | 52 | +0.013 | +0.060 | +0.047 | +0.007 - +0.079 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.088 | +0.024 | +0.007 - +0.038 |
| melee, 10-30 s | 1948 | -0.050 | -0.032 | +0.018 | +0.005 - +0.031 |
| melee, 30 s+ | 6079 | -0.026 | -0.009 | +0.017 | +0.004 - +0.031 |
| melee, sim touches 1 enemy | 3993 | -0.046 | -0.049 | -0.004 | -0.016 - +0.009 |
| melee, sim touches 2+ enemies | 3790 | -0.051 | -0.027 | +0.024 | +0.009 - +0.041 |
| melee, sim touches nobody (game flag on) | 1691 | -0.025 | +0.030 | +0.055 | +0.037 - +0.073 |
| melee, touched in front | 3756 | -0.042 | -0.032 | +0.010 | -0.002 - +0.024 |
| melee, touched in flank | 3125 | -0.053 | -0.047 | +0.006 | -0.006 - +0.019 |
| melee, touched in rear | 902 | -0.057 | -0.036 | +0.022 | -0.001 - +0.048 |
| melee, health < 30 % | 907 | -0.034 | -0.030 | +0.003 | -0.011 - +0.020 |
| melee, health >= 30 % | 8567 | -0.045 | -0.026 | +0.019 | +0.007 - +0.033 |
| standing, health < 30 % | 4233 | -0.009 | -0.014 | -0.005 | -0.014 - +0.009 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.025 | -0.006 | -0.011 - -0.000 |
| steady (3) | 3200 | -0.013 | +0.011 | +0.024 | +0.012 - +0.040 |
| shaken (4) | 4396 | -0.010 | +0.005 | +0.015 | -0.004 - +0.042 |
| wavering (5) | 2158 | +0.031 | +0.053 | +0.022 | +0.002 - +0.043 |
| mp < 0.25 | 2088 | +0.027 | +0.041 | +0.014 | -0.007 - +0.041 |
| mp 0.25-0.5 | 3827 | -0.005 | +0.014 | +0.019 | -0.002 - +0.049 |
| mp 0.5-0.75 | 4343 | -0.015 | +0.011 | +0.026 | +0.016 - +0.037 |
| mp >= 0.75 | 17864 | -0.020 | -0.026 | -0.007 | -0.012 - -0.001 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 12127 | 0.98 | 0.72-1.31 |
| hold, lord | 59 | 18 | 13200 | 7840 | 0.59 | 0.40-0.79 |
| move, formation | 450 | 17 | 31949 | 11927 | 0.37 | 0.28-0.48 |
| move, missile unit | 148 | 14 | 6065 | 913 | 0.15 | 0.06-0.38 |
| move, lord | 41 | 13 | 4562 | 516 | 0.11 | 0.04-0.24 |
| attack, formation | 8219 | 20 | 1923931 | 1558019 | 0.81 | 0.76-0.85 |
| attack, missile unit | 236 | 18 | 39992 | 25954 | 0.65 | 0.53-0.82 |
| attack, lord | 621 | 20 | 133139 | 113577 | 0.85 | 0.68-1.04 |

## C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/build/charge7/sr_off: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.97 (0.95-1.00); rout starts sim/game 1.02 (0.95-1.09), game 2008, sim 2047.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.85 | 0.85 | 0.85 | 0.87 | 0.88 | 0.89 | 0.90 | 0.91 | 0.93 | 0.93 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1215.5 | 0.97 | 0.95-1.00 | 1.03 | 1.00-1.06 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 81.3 | 1.37 | 1.29-1.45 | 1.45 | 1.34-1.58 | -0.55 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 161.7 | 0.94 | 0.92-0.97 | 1.08 | 1.03-1.13 | 0.24 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 147.9 | 0.96 | 0.92-1.00 | 0.94 | 0.90-0.98 | 0.15 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.7 | 0.87 | 0.76-1.00 | 0.87 | 0.74-1.02 | 0.03 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 131.6 | 0.90 | 0.87-0.93 | 0.95 | 0.92-0.99 | 0.37 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 181.9 | 0.87 | 0.85-0.90 | 0.93 | 0.90-0.98 | 0.67 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 477.9 | 0.96 | 0.93-1.01 | 1.03 | 0.98-1.08 | 0.45 |
| melee, sim touches 1 enemy | 10530 | 72 | 308.7 | 259.7 | 0.84 | 0.82-0.86 | 0.90 | 0.87-0.93 | 1.25 |
| melee, sim touches 2+ enemies | 15292 | 72 | 454.2 | 470.8 | 1.04 | 1.00-1.08 | 1.08 | 1.04-1.13 | -0.42 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 86.7 | 60.8 | 0.70 | 0.66-0.75 | 0.85 | 0.79-0.91 | 0.66 |
| melee, touched in front | 8822 | 72 | 220.1 | 198.2 | 0.90 | 0.87-0.93 | 0.97 | 0.94-1.02 | 0.56 |
| melee, touched in flank | 12297 | 72 | 375.4 | 371.6 | 0.99 | 0.96-1.03 | 1.03 | 1.00-1.07 | 0.10 |
| melee, touched in rear | 4703 | 72 | 167.5 | 160.7 | 0.96 | 0.91-1.02 | 0.99 | 0.93-1.06 | 0.17 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 71.2 | 1.12 | 1.07-1.18 | 1.24 | 1.16-1.33 | -0.20 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 720.1 | 0.92 | 0.89-0.95 | 0.97 | 0.94-1.01 | 1.68 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 55.2 | 1.12 | 0.98-1.28 | 1.04 | 0.90-1.21 | -0.15 |
| routing, free | 10755 | 72 | 22.6 | 29.3 | 1.29 | 1.14-1.47 | 1.31 | 1.11-1.54 | -0.17 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 93.5 | 1.09 | 1.04-1.14 | 1.18 | 1.12-1.26 | -0.19 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 91.4 | 1.12 | 1.04-1.20 | 1.13 | 1.04-1.22 | -0.25 |
| missile | 24324 | 72 | 316.4 | 297.5 | 0.94 | 0.90-0.98 | 1.00 | 0.96-1.05 | 0.48 |
| infantry melee | 57353 | 72 | 856.7 | 826.6 | 0.96 | 0.94-1.00 | 1.03 | 0.99-1.06 | 0.77 |
| network's side | 43836 | 72 | 729.1 | 722.8 | 0.99 | 0.96-1.03 | 1.06 | 1.02-1.10 | 0.16 |
| the other side | 44434 | 72 | 525.7 | 492.7 | 0.94 | 0.90-0.98 | 0.99 | 0.94-1.03 | 0.84 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 62.5 | 1.08 | 1.01-1.16 | 1.10 | 1.00-1.20 | -0.12 |
| lord, routing | 511 | 54 | 1.6 | 2.0 | 1.31 | 0.96-1.76 | 1.45 | 1.01-1.88 | -0.01 |
| missile, melee | 4055 | 72 | 184.5 | 165.3 | 0.90 | 0.86-0.94 | 0.97 | 0.92-1.01 | 0.49 |
| missile, routing | 3025 | 72 | 18.3 | 22.5 | 1.23 | 0.99-1.51 | 1.14 | 0.90-1.44 | -0.11 |
| infantry melee, melee | 23441 | 72 | 607.5 | 563.6 | 0.93 | 0.90-0.97 | 0.99 | 0.95-1.03 | 1.12 |
| infantry melee, routing | 10365 | 72 | 52.0 | 59.9 | 1.15 | 1.00-1.34 | 1.12 | 0.95-1.34 | -0.20 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 64.2 | 4.28 | 2.56-8.36 |
| steady (3) | 7127 | 72 | 28 | 72.5 | 2.59 | 1.62-4.12 |
| shaken (4) | 11458 | 72 | 477 | 660.8 | 1.39 | 1.24-1.55 |
| wavering (5) | 5568 | 72 | 1453 | 1230.5 | 0.85 | 0.79-0.90 |
| mp < 0.25 | 5043 | 72 | 1480 | 1210.2 | 0.82 | 0.77-0.87 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 572.5 | 1.30 | 1.15-1.45 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 193.2 | 2.88 | 2.28-3.73 |
| mp >= 0.75 | 48908 | 72 | 20 | 71.0 | 3.55 | 2.17-6.68 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 2047.0 | 1.02 | 0.95-1.09 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 288.2 | 1.08 | 0.87-1.30 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 200.0 | 1.05 | 0.93-1.17 |
| contact begins in window, ran in | 6699 | 72 | 124 | 151.2 | 1.22 | 1.03-1.48 |
| contact begins in window, walked in | 271 | 68 | 6 | 5.5 | 0.92 | 0.38-2.00 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 206.2 | 1.27 | 1.13-1.44 |
| melee, 10-30 s | 6160 | 72 | 304 | 307.5 | 1.01 | 0.91-1.12 |
| melee, 30 s+ | 19508 | 72 | 983 | 933.5 | 0.95 | 0.86-1.05 |
| melee, sim touches 1 enemy | 10530 | 72 | 539 | 559.0 | 1.04 | 0.96-1.12 |
| melee, sim touches 2+ enemies | 15292 | 72 | 759 | 786.2 | 1.04 | 0.93-1.15 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 152 | 102.0 | 0.67 | 0.56-0.80 |
| melee, touched in front | 8822 | 72 | 322 | 293.8 | 0.91 | 0.82-1.01 |
| melee, touched in flank | 12297 | 72 | 627 | 665.0 | 1.06 | 0.97-1.15 |
| melee, touched in rear | 4703 | 72 | 349 | 386.5 | 1.11 | 0.97-1.26 |
| melee, health < 30 % | 3088 | 72 | 618 | 584.0 | 0.94 | 0.89-1.01 |
| melee, health >= 30 % | 26784 | 72 | 832 | 863.2 | 1.04 | 0.94-1.15 |
| standing, health < 30 % | 8705 | 72 | 1028 | 976.0 | 0.95 | 0.88-1.02 |
| lord | 6082 | 72 | 100 | 83.0 | 0.83 | 0.65-1.03 |
| missile | 21299 | 72 | 541 | 590.5 | 1.09 | 0.99-1.20 |
| infantry melee | 46988 | 72 | 1367 | 1373.5 | 1.00 | 0.93-1.08 |
| network's side | 34608 | 72 | 1389 | 1455.5 | 1.05 | 0.96-1.13 |
| the other side | 39761 | 72 | 619 | 591.5 | 0.96 | 0.86-1.07 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1246.2 | 1.07 | 1.01-1.12 |
| routers in melee | 3146 | 72 | 90 | 158.2 | 1.76 | 1.51-2.09 |
| routers free | 10755 | 72 | 1080 | 1088.0 | 1.01 | 0.96-1.06 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.014 | +0.002 | -0.001 - +0.006 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.003 | -0.017 | -0.021 - -0.013 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.088 | -0.006 | -0.014 - +0.004 |
| contact begins in window, ran in | 6575 | -0.050 | -0.062 | -0.012 | -0.021 - -0.003 |
| contact begins in window, walked in | 265 | -0.065 | -0.086 | -0.021 | -0.051 - +0.008 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.114 | -0.002 | -0.011 - +0.007 |
| melee, 10-30 s | 5856 | -0.058 | -0.047 | +0.011 | +0.003 - +0.018 |
| melee, 30 s+ | 18525 | -0.023 | +0.021 | +0.044 | +0.037 - +0.051 |
| melee, sim touches 1 enemy | 9991 | -0.046 | -0.026 | +0.020 | +0.013 - +0.027 |
| melee, sim touches 2+ enemies | 14533 | -0.045 | -0.021 | +0.024 | +0.017 - +0.032 |
| melee, sim touches nobody (game flag on) | 3898 | -0.024 | +0.057 | +0.081 | +0.068 - +0.092 |
| melee, touched in front | 8500 | -0.039 | -0.001 | +0.038 | +0.031 - +0.045 |
| melee, touched in flank | 11670 | -0.047 | -0.030 | +0.017 | +0.009 - +0.025 |
| melee, touched in rear | 4354 | -0.053 | -0.046 | +0.007 | -0.005 - +0.018 |
| melee, health < 30 % | 2470 | -0.036 | -0.023 | +0.012 | -0.002 - +0.025 |
| melee, health >= 30 % | 25952 | -0.043 | -0.011 | +0.032 | +0.026 - +0.039 |
| standing, health < 30 % | 7677 | -0.013 | -0.022 | -0.009 | -0.023 - +0.004 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.030 | -0.008 | -0.011 - -0.005 |
| steady (3) | 7099 | -0.025 | +0.013 | +0.038 | +0.031 - +0.045 |
| shaken (4) | 10981 | -0.011 | +0.012 | +0.023 | +0.013 - +0.032 |
| wavering (5) | 4115 | +0.042 | +0.056 | +0.014 | -0.007 - +0.030 |
| mp < 0.25 | 3563 | +0.040 | +0.061 | +0.020 | -0.003 - +0.040 |
| mp 0.25-0.5 | 8520 | -0.001 | +0.032 | +0.033 | +0.025 - +0.039 |
| mp 0.5-0.75 | 11390 | -0.024 | +0.005 | +0.028 | +0.021 - +0.037 |
| mp >= 0.75 | 48888 | -0.022 | -0.032 | -0.010 | -0.013 - -0.007 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 44123 | 0.37 | 0.32-0.42 |
| hold, missile unit | 1416 | 72 | 213822 | 124268 | 0.58 | 0.52-0.65 |
| hold, lord | 537 | 69 | 145512 | 60531 | 0.42 | 0.36-0.48 |
| move, formation | 1035 | 71 | 151893 | 48886 | 0.32 | 0.28-0.37 |
| move, missile unit | 751 | 69 | 78425 | 24342 | 0.31 | 0.24-0.41 |
| move, lord | 49 | 32 | 7295 | 2497 | 0.34 | 0.20-0.53 |
| attack, formation | 21672 | 72 | 5277048 | 3433956 | 0.65 | 0.62-0.68 |
| attack, missile unit | 1888 | 71 | 332739 | 207755 | 0.62 | 0.56-0.68 |
| attack, lord | 1790 | 72 | 422482 | 298194 | 0.71 | 0.64-0.77 |
