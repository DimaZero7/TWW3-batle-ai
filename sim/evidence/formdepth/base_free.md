## ../../../build/contactgap/sr_free_rect: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 1.02 (0.94-1.12); rout starts sim/game 1.38 (1.16-1.58), game 683, sim 943.8.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.95 | 0.93 | 0.92 | 0.92 | 0.92 | 0.94 | 0.96 | 0.97 | 0.98 | 0.99 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 385.1 | 1.02 | 0.94-1.12 | 1.08 | 0.99-1.18 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 29.4 | 1.48 | 1.25-1.74 | 1.60 | 1.29-1.96 | 1.12 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 24.2 | 0.96 | 0.79-1.17 | 1.15 | 0.92-1.43 | -0.11 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 44.4 | 0.89 | 0.80-0.99 | 0.90 | 0.80-1.02 | -0.67 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.7 | 0.56 | 0.41-0.76 | 0.77 | 0.56-1.01 | -0.06 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 43.6 | 0.91 | 0.82-1.01 | 0.97 | 0.86-1.10 | -0.52 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 57.0 | 0.95 | 0.87-1.04 | 1.01 | 0.91-1.15 | -0.38 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 166.1 | 1.03 | 0.97-1.10 | 1.11 | 1.04-1.18 | 0.59 |
| melee, sim touches 1 enemy | 4186 | 20 | 111.5 | 104.4 | 0.94 | 0.86-1.02 | 0.99 | 0.91-1.09 | -0.84 |
| melee, sim touches 2+ enemies | 4016 | 20 | 126.7 | 134.5 | 1.06 | 1.00-1.11 | 1.13 | 1.06-1.18 | 0.92 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 31.1 | 27.8 | 0.89 | 0.77-1.03 | 1.05 | 0.90-1.20 | -0.38 |
| melee, touched in front | 3901 | 20 | 99.2 | 95.0 | 0.96 | 0.89-1.03 | 1.01 | 0.93-1.09 | -0.50 |
| melee, touched in flank | 3309 | 20 | 103.9 | 107.3 | 1.03 | 0.97-1.09 | 1.11 | 1.03-1.17 | 0.40 |
| melee, touched in rear | 992 | 20 | 35.1 | 36.7 | 1.04 | 0.94-1.13 | 1.10 | 0.96-1.20 | 0.18 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 28.1 | 1.20 | 1.06-1.34 | 1.36 | 1.20-1.53 | 0.54 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 238.7 | 0.97 | 0.91-1.03 | 1.04 | 0.97-1.10 | -0.84 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 20.3 | 1.44 | 1.13-1.76 | 1.37 | 0.97-1.85 | 0.72 |
| routing, free | 6219 | 20 | 5.5 | 7.7 | 1.39 | 0.91-2.23 | 1.39 | 0.80-2.33 | 0.25 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 38.2 | 1.17 | 1.02-1.32 | 1.34 | 1.19-1.51 | 0.64 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 26.7 | 0.92 | 0.82-1.04 | 0.95 | 0.82-1.09 | -0.27 |
| missile | 3989 | 20 | 31.8 | 32.5 | 1.02 | 0.87-1.19 | 1.07 | 0.88-1.30 | 0.08 |
| infantry melee | 29215 | 20 | 315.8 | 326.0 | 1.03 | 0.94-1.13 | 1.10 | 1.00-1.20 | 1.19 |
| network's side | 16523 | 20 | 184.7 | 177.3 | 0.96 | 0.90-1.02 | 1.00 | 0.94-1.06 | -0.86 |
| the other side | 19331 | 20 | 191.9 | 207.8 | 1.08 | 0.95-1.22 | 1.17 | 1.02-1.32 | 1.86 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 18.0 | 0.89 | 0.81-0.98 | 0.90 | 0.79-1.02 | -0.25 |
| lord, routing | 236 | 15 | 0.9 | 0.5 | 0.56 | 0.32-1.02 | 1.21 | 0.38-4.65 | -0.05 |
| missile, melee | 413 | 18 | 15.9 | 16.5 | 1.04 | 0.92-1.16 | 1.03 | 0.89-1.17 | 0.07 |
| missile, routing | 405 | 16 | 2.2 | 4.7 | 2.11 | 1.11-3.48 | 2.41 | 0.96-4.36 | 0.29 |
| infantry melee, melee | 8824 | 20 | 233.2 | 232.2 | 1.00 | 0.93-1.06 | 1.08 | 1.00-1.15 | -0.12 |
| infantry melee, routing | 6408 | 20 | 16.5 | 22.8 | 1.38 | 1.09-1.69 | 1.26 | 0.92-1.64 | 0.74 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 17.0 | 2.83 | 0.93-12.25 |
| steady (3) | 3212 | 20 | 12 | 41.2 | 3.44 | 1.70-6.89 |
| shaken (4) | 4549 | 20 | 153 | 291.0 | 1.90 | 1.41-2.37 |
| wavering (5) | 2661 | 20 | 503 | 588.5 | 1.17 | 1.01-1.29 |
| mp < 0.25 | 2598 | 20 | 510 | 595.2 | 1.17 | 1.01-1.31 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 260.0 | 1.86 | 1.50-2.23 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 72.5 | 2.69 | 1.43-4.55 |
| mp >= 0.75 | 17870 | 20 | 6 | 16.0 | 2.67 | 0.92-11.34 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 943.8 | 1.38 | 1.16-1.58 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 187.0 | 1.65 | 1.19-2.01 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 65.8 | 1.40 | 1.15-1.84 |
| contact begins in window, ran in | 2307 | 20 | 42 | 81.8 | 1.95 | 1.37-2.57 |
| contact begins in window, walked in | 56 | 15 | 4 | 3.0 | 0.75 | 0.20-2.50 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 106.8 | 1.21 | 0.99-1.45 |
| melee, 10-30 s | 2047 | 20 | 99 | 137.2 | 1.39 | 1.11-1.66 |
| melee, 30 s+ | 6376 | 20 | 297 | 376.2 | 1.27 | 1.04-1.49 |
| melee, sim touches 1 enemy | 4186 | 20 | 193 | 283.0 | 1.47 | 1.12-1.85 |
| melee, sim touches 2+ enemies | 4016 | 20 | 226 | 277.0 | 1.23 | 1.07-1.38 |
| melee, sim touches nobody (game flag on) | 1756 | 20 | 65 | 60.2 | 0.93 | 0.69-1.31 |
| melee, touched in front | 3901 | 20 | 145 | 174.8 | 1.21 | 0.98-1.42 |
| melee, touched in flank | 3309 | 20 | 184 | 244.5 | 1.33 | 1.08-1.57 |
| melee, touched in rear | 992 | 20 | 90 | 140.8 | 1.56 | 1.25-1.92 |
| melee, health < 30 % | 1132 | 20 | 225 | 285.2 | 1.27 | 1.05-1.47 |
| melee, health >= 30 % | 8826 | 20 | 259 | 335.0 | 1.29 | 1.07-1.53 |
| standing, health < 30 % | 4614 | 20 | 381 | 529.2 | 1.39 | 1.16-1.57 |
| lord | 2414 | 20 | 34 | 44.8 | 1.32 | 0.90-1.93 |
| missile | 3584 | 20 | 50 | 68.0 | 1.36 | 0.98-1.81 |
| infantry melee | 22807 | 20 | 599 | 831.0 | 1.39 | 1.16-1.61 |
| network's side | 13157 | 20 | 356 | 400.5 | 1.12 | 0.97-1.30 |
| the other side | 15648 | 20 | 327 | 543.2 | 1.66 | 1.32-2.01 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 296.2 | 0.80 | 0.73-0.84 |
| routers in melee | 830 | 20 | 36 | 38.0 | 1.06 | 0.93-1.22 |
| routers free | 6219 | 20 | 335 | 258.2 | 0.77 | 0.70-0.81 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.040 | -0.026 | -0.031 - -0.020 |
| standing, not in melee, no contact in window | 16032 | +0.008 | -0.022 | -0.031 | -0.036 - -0.026 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.129 | -0.062 | -0.098 - -0.022 |
| contact begins in window, ran in | 2265 | -0.034 | -0.069 | -0.034 | -0.044 - -0.026 |
| contact begins in window, walked in | 52 | +0.013 | +0.014 | +0.000 | -0.037 - +0.034 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.135 | -0.023 | -0.037 - -0.009 |
| melee, 10-30 s | 1948 | -0.050 | -0.062 | -0.012 | -0.025 - +0.004 |
| melee, 30 s+ | 6079 | -0.026 | -0.040 | -0.014 | -0.024 - -0.003 |
| melee, sim touches 1 enemy | 3993 | -0.046 | -0.081 | -0.035 | -0.047 - -0.023 |
| melee, sim touches 2+ enemies | 3790 | -0.051 | -0.055 | -0.005 | -0.017 - +0.010 |
| melee, sim touches nobody (game flag on) | 1691 | -0.025 | -0.016 | +0.009 | -0.007 - +0.026 |
| melee, touched in front | 3756 | -0.042 | -0.058 | -0.016 | -0.029 - -0.003 |
| melee, touched in flank | 3125 | -0.053 | -0.080 | -0.027 | -0.037 - -0.017 |
| melee, touched in rear | 902 | -0.057 | -0.074 | -0.016 | -0.036 - +0.006 |
| melee, health < 30 % | 907 | -0.034 | -0.080 | -0.047 | -0.062 - -0.033 |
| melee, health >= 30 % | 8567 | -0.045 | -0.057 | -0.012 | -0.022 - -0.000 |
| standing, health < 30 % | 4233 | -0.009 | -0.049 | -0.040 | -0.054 - -0.030 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.051 | -0.032 | -0.037 - -0.027 |
| steady (3) | 3200 | -0.013 | -0.027 | -0.013 | -0.028 - +0.004 |
| shaken (4) | 4396 | -0.010 | -0.029 | -0.018 | -0.030 - +0.001 |
| wavering (5) | 2158 | +0.031 | +0.017 | -0.014 | -0.024 - +0.002 |
| mp < 0.25 | 2088 | +0.027 | +0.011 | -0.016 | -0.028 - +0.003 |
| mp 0.25-0.5 | 3827 | -0.005 | -0.019 | -0.013 | -0.025 - +0.006 |
| mp 0.5-0.75 | 4343 | -0.015 | -0.028 | -0.013 | -0.026 - +0.001 |
| mp >= 0.75 | 17864 | -0.020 | -0.053 | -0.033 | -0.039 - -0.028 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 12175 | 0.99 | 0.68-1.41 |
| hold, lord | 59 | 18 | 13200 | 7763 | 0.59 | 0.37-0.82 |
| move, formation | 450 | 17 | 31949 | 11311 | 0.35 | 0.27-0.44 |
| move, missile unit | 148 | 14 | 6065 | 1119 | 0.18 | 0.10-0.40 |
| move, lord | 41 | 13 | 4562 | 723 | 0.16 | 0.03-0.36 |
| attack, formation | 8219 | 20 | 1923931 | 1563387 | 0.81 | 0.76-0.85 |
| attack, missile unit | 236 | 18 | 39992 | 26916 | 0.67 | 0.54-0.85 |
| attack, lord | 621 | 20 | 133139 | 111998 | 0.84 | 0.67-1.04 |

## ../../../build/contactgap/sr_free_rect: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.96 (0.94-0.99); rout starts sim/game 1.25 (1.17-1.33), game 2008, sim 2507.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.85 | 0.85 | 0.86 | 0.87 | 0.88 | 0.89 | 0.90 | 0.91 | 0.92 | 0.93 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1210.4 | 0.96 | 0.94-0.99 | 1.02 | 1.00-1.05 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 80.9 | 1.36 | 1.28-1.45 | 1.46 | 1.35-1.58 | -0.48 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 157.7 | 0.92 | 0.90-0.95 | 1.07 | 1.02-1.13 | 0.31 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 148.8 | 0.97 | 0.93-1.01 | 0.94 | 0.90-0.99 | 0.12 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.6 | 0.85 | 0.76-0.97 | 0.81 | 0.70-0.95 | 0.03 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 132.5 | 0.91 | 0.88-0.94 | 0.95 | 0.92-1.00 | 0.31 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 181.5 | 0.87 | 0.85-0.90 | 0.93 | 0.89-0.97 | 0.60 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 475.1 | 0.96 | 0.92-1.00 | 1.02 | 0.98-1.07 | 0.46 |
| melee, sim touches 1 enemy | 10530 | 72 | 308.7 | 258.8 | 0.84 | 0.82-0.86 | 0.89 | 0.86-0.92 | 1.12 |
| melee, sim touches 2+ enemies | 15292 | 72 | 454.2 | 469.7 | 1.03 | 1.00-1.08 | 1.07 | 1.03-1.12 | -0.35 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 86.7 | 60.6 | 0.70 | 0.66-0.74 | 0.85 | 0.80-0.91 | 0.59 |
| melee, touched in front | 8822 | 72 | 220.1 | 197.2 | 0.90 | 0.87-0.93 | 0.96 | 0.92-1.01 | 0.52 |
| melee, touched in flank | 12297 | 72 | 375.4 | 369.5 | 0.98 | 0.95-1.02 | 1.03 | 0.99-1.07 | 0.13 |
| melee, touched in rear | 4703 | 72 | 167.5 | 161.9 | 0.97 | 0.92-1.02 | 0.99 | 0.94-1.05 | 0.13 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 70.0 | 1.10 | 1.05-1.16 | 1.22 | 1.14-1.31 | -0.15 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 719.1 | 0.91 | 0.89-0.95 | 0.97 | 0.93-1.00 | 1.51 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 54.6 | 1.11 | 0.97-1.27 | 1.03 | 0.89-1.19 | -0.12 |
| routing, free | 10755 | 72 | 22.6 | 29.1 | 1.29 | 1.13-1.46 | 1.32 | 1.10-1.56 | -0.15 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 91.0 | 1.06 | 1.01-1.11 | 1.16 | 1.10-1.24 | -0.12 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 92.3 | 1.13 | 1.05-1.21 | 1.14 | 1.05-1.24 | -0.24 |
| missile | 24324 | 72 | 316.4 | 293.2 | 0.93 | 0.89-0.96 | 0.99 | 0.95-1.04 | 0.52 |
| infantry melee | 57353 | 72 | 856.7 | 824.9 | 0.96 | 0.94-0.99 | 1.02 | 0.99-1.06 | 0.72 |
| network's side | 43836 | 72 | 729.1 | 719.5 | 0.99 | 0.95-1.02 | 1.05 | 1.01-1.09 | 0.21 |
| the other side | 44434 | 72 | 525.7 | 490.9 | 0.93 | 0.90-0.97 | 0.99 | 0.95-1.03 | 0.79 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 63.4 | 1.10 | 1.02-1.17 | 1.11 | 1.02-1.20 | -0.13 |
| lord, routing | 511 | 54 | 1.6 | 2.0 | 1.31 | 0.96-1.73 | 1.43 | 1.01-1.92 | -0.01 |
| missile, melee | 4055 | 72 | 184.5 | 163.2 | 0.88 | 0.85-0.93 | 0.95 | 0.91-1.00 | 0.48 |
| missile, routing | 3025 | 72 | 18.3 | 22.2 | 1.22 | 0.98-1.50 | 1.14 | 0.89-1.46 | -0.09 |
| infantry melee, melee | 23441 | 72 | 607.5 | 562.5 | 0.93 | 0.90-0.96 | 0.98 | 0.95-1.02 | 1.01 |
| infantry melee, routing | 10365 | 72 | 52.0 | 59.4 | 1.14 | 0.98-1.34 | 1.11 | 0.94-1.33 | -0.17 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 89.8 | 5.98 | 3.60-11.75 |
| steady (3) | 7127 | 72 | 28 | 81.0 | 2.89 | 1.92-4.56 |
| shaken (4) | 11458 | 72 | 477 | 832.0 | 1.74 | 1.57-1.95 |
| wavering (5) | 5568 | 72 | 1453 | 1483.2 | 1.02 | 0.96-1.08 |
| mp < 0.25 | 5043 | 72 | 1480 | 1449.2 | 0.98 | 0.92-1.03 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 708.5 | 1.61 | 1.44-1.78 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 249.8 | 3.73 | 2.99-4.78 |
| mp >= 0.75 | 48908 | 72 | 20 | 99.5 | 4.97 | 3.06-8.85 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 2507.0 | 1.25 | 1.17-1.33 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 387.5 | 1.45 | 1.22-1.70 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 254.5 | 1.33 | 1.19-1.48 |
| contact begins in window, ran in | 6699 | 72 | 124 | 205.0 | 1.65 | 1.41-1.97 |
| contact begins in window, walked in | 271 | 68 | 6 | 5.8 | 0.96 | 0.33-2.25 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 239.8 | 1.47 | 1.31-1.66 |
| melee, 10-30 s | 6160 | 72 | 304 | 366.5 | 1.21 | 1.09-1.32 |
| melee, 30 s+ | 19508 | 72 | 983 | 1102.5 | 1.12 | 1.03-1.23 |
| melee, sim touches 1 enemy | 10530 | 72 | 539 | 669.0 | 1.24 | 1.16-1.33 |
| melee, sim touches 2+ enemies | 15292 | 72 | 759 | 911.0 | 1.20 | 1.07-1.33 |
| melee, sim touches nobody (game flag on) | 4050 | 72 | 152 | 128.8 | 0.85 | 0.71-1.00 |
| melee, touched in front | 8822 | 72 | 322 | 311.8 | 0.97 | 0.86-1.09 |
| melee, touched in flank | 12297 | 72 | 627 | 773.0 | 1.23 | 1.13-1.34 |
| melee, touched in rear | 4703 | 72 | 349 | 495.2 | 1.42 | 1.24-1.62 |
| melee, health < 30 % | 3088 | 72 | 618 | 686.8 | 1.11 | 1.04-1.19 |
| melee, health >= 30 % | 26784 | 72 | 832 | 1022.0 | 1.23 | 1.11-1.35 |
| standing, health < 30 % | 8705 | 72 | 1028 | 1201.8 | 1.17 | 1.08-1.25 |
| lord | 6082 | 72 | 100 | 83.8 | 0.84 | 0.62-1.07 |
| missile | 21299 | 72 | 541 | 748.0 | 1.38 | 1.27-1.49 |
| infantry melee | 46988 | 72 | 1367 | 1675.2 | 1.23 | 1.14-1.32 |
| network's side | 34608 | 72 | 1389 | 1727.8 | 1.24 | 1.15-1.34 |
| the other side | 39761 | 72 | 619 | 779.2 | 1.26 | 1.14-1.40 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1211.2 | 1.04 | 0.98-1.09 |
| routers in melee | 3146 | 72 | 90 | 150.8 | 1.68 | 1.45-1.98 |
| routers free | 10755 | 72 | 1080 | 1060.5 | 0.98 | 0.93-1.04 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.050 | -0.033 | -0.038 - -0.029 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.039 | -0.053 | -0.059 - -0.048 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.144 | -0.061 | -0.068 - -0.053 |
| contact begins in window, ran in | 6575 | -0.050 | -0.110 | -0.060 | -0.069 - -0.051 |
| contact begins in window, walked in | 265 | -0.065 | -0.120 | -0.054 | -0.088 - -0.026 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.160 | -0.048 | -0.056 - -0.041 |
| melee, 10-30 s | 5856 | -0.058 | -0.081 | -0.024 | -0.031 - -0.016 |
| melee, 30 s+ | 18525 | -0.023 | -0.004 | +0.019 | +0.012 - +0.025 |
| melee, sim touches 1 enemy | 9991 | -0.046 | -0.063 | -0.017 | -0.025 - -0.009 |
| melee, sim touches 2+ enemies | 14533 | -0.045 | -0.039 | +0.006 | -0.001 - +0.012 |
| melee, sim touches nobody (game flag on) | 3898 | -0.024 | +0.002 | +0.026 | +0.015 - +0.036 |
| melee, touched in front | 8500 | -0.039 | -0.028 | +0.011 | +0.002 - +0.018 |
| melee, touched in flank | 11670 | -0.047 | -0.054 | -0.007 | -0.014 - -0.000 |
| melee, touched in rear | 4354 | -0.053 | -0.075 | -0.022 | -0.035 - -0.011 |
| melee, health < 30 % | 2470 | -0.036 | -0.054 | -0.019 | -0.033 - -0.006 |
| melee, health >= 30 % | 25952 | -0.043 | -0.041 | +0.002 | -0.004 - +0.008 |
| standing, health < 30 % | 7677 | -0.013 | -0.069 | -0.057 | -0.072 - -0.042 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.064 | -0.042 | -0.046 - -0.039 |
| steady (3) | 7099 | -0.025 | -0.023 | +0.002 | -0.006 - +0.009 |
| shaken (4) | 10981 | -0.011 | -0.031 | -0.020 | -0.028 - -0.012 |
| wavering (5) | 4115 | +0.042 | +0.022 | -0.021 | -0.040 - -0.005 |
| mp < 0.25 | 3563 | +0.040 | +0.035 | -0.006 | -0.028 - +0.014 |
| mp 0.25-0.5 | 8520 | -0.001 | -0.006 | -0.005 | -0.013 - +0.002 |
| mp 0.5-0.75 | 11390 | -0.024 | -0.038 | -0.014 | -0.022 - -0.007 |
| mp >= 0.75 | 48888 | -0.022 | -0.066 | -0.045 | -0.048 - -0.041 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 43643 | 0.37 | 0.32-0.42 |
| hold, missile unit | 1416 | 72 | 213822 | 125174 | 0.59 | 0.52-0.65 |
| hold, lord | 537 | 69 | 145512 | 62537 | 0.43 | 0.37-0.50 |
| move, formation | 1035 | 71 | 151893 | 47550 | 0.31 | 0.27-0.37 |
| move, missile unit | 751 | 69 | 78425 | 22949 | 0.29 | 0.22-0.39 |
| move, lord | 49 | 32 | 7295 | 2566 | 0.35 | 0.20-0.54 |
| attack, formation | 21672 | 72 | 5277048 | 3438535 | 0.65 | 0.62-0.68 |
| attack, missile unit | 1888 | 71 | 332739 | 209722 | 0.63 | 0.57-0.69 |
| attack, lord | 1790 | 72 | 422482 | 297885 | 0.71 | 0.64-0.77 |
