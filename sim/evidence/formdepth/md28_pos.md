## ../../../build/formdepth/sr_pos_md28: exam

Windows of 10 s; 20 battles, 35854 unit-windows. HP lost sim/game 0.98 (0.90-1.08); rout starts sim/game 1.39 (1.18-1.58), game 683, sim 950.2.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.97 | 0.94 | 0.93 | 0.92 | 0.92 | 0.92 | 0.93 | 0.94 | 0.94 | 0.95 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 35854 | 20 | 376.6 | 370.4 | 0.98 | 0.90-1.08 | 1.00 | 0.91-1.09 | 1.00 |
| standing, not in melee, no contact in window | 16145 | 20 | 19.8 | 26.5 | 1.33 | 1.09-1.57 | 1.35 | 1.11-1.61 | -1.06 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 25.1 | 24.1 | 0.96 | 0.77-1.18 | 1.09 | 0.86-1.40 | 0.17 |
| contact begins in window, ran in | 2307 | 20 | 50.1 | 42.5 | 0.85 | 0.75-0.97 | 0.81 | 0.71-0.93 | 1.22 |
| contact begins in window, walked in | 56 | 15 | 1.2 | 0.5 | 0.38 | 0.20-0.64 | 0.48 | 0.20-0.81 | 0.12 |
| melee, first 10 s of the fight | 1535 | 20 | 48.0 | 40.5 | 0.84 | 0.77-0.92 | 0.86 | 0.78-0.95 | 1.22 |
| melee, 10-30 s | 2047 | 20 | 60.2 | 54.7 | 0.91 | 0.82-1.00 | 0.94 | 0.84-1.04 | 0.89 |
| melee, 30 s+ | 6376 | 20 | 161.1 | 159.8 | 0.99 | 0.92-1.06 | 1.03 | 0.96-1.11 | 0.20 |
| melee, sim touches 1 enemy | 4233 | 20 | 111.1 | 100.4 | 0.90 | 0.82-0.99 | 0.93 | 0.84-1.03 | 1.72 |
| melee, sim touches 2+ enemies | 4241 | 20 | 132.4 | 137.1 | 1.04 | 0.98-1.09 | 1.07 | 1.00-1.13 | -0.75 |
| melee, sim touches nobody (game flag on) | 1484 | 20 | 25.8 | 17.4 | 0.68 | 0.55-0.84 | 0.77 | 0.65-0.92 | 1.34 |
| melee, touched in front | 4093 | 20 | 102.7 | 95.4 | 0.93 | 0.87-1.00 | 0.96 | 0.88-1.04 | 1.17 |
| melee, touched in flank | 3346 | 20 | 104.5 | 104.6 | 1.00 | 0.93-1.06 | 1.04 | 0.96-1.11 | -0.02 |
| melee, touched in rear | 1035 | 20 | 36.3 | 37.5 | 1.03 | 0.92-1.14 | 1.03 | 0.90-1.15 | -0.19 |
| melee, health < 30 % | 1132 | 20 | 23.4 | 28.4 | 1.21 | 1.06-1.38 | 1.35 | 1.17-1.54 | -0.81 |
| melee, health >= 30 % | 8826 | 20 | 245.9 | 226.5 | 0.92 | 0.86-0.98 | 0.95 | 0.88-1.02 | 3.12 |
| routing, in melee (pursued) | 830 | 20 | 14.1 | 21.0 | 1.49 | 1.18-1.81 | 1.39 | 1.03-1.85 | -1.11 |
| routing, free | 6219 | 20 | 5.5 | 9.2 | 1.66 | 1.23-2.48 | 1.68 | 1.16-2.57 | -0.59 |
| standing, health < 30 % | 4614 | 20 | 32.7 | 37.9 | 1.16 | 1.01-1.32 | 1.30 | 1.15-1.47 | -0.83 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 2650 | 20 | 29.0 | 24.7 | 0.85 | 0.77-0.94 | 0.83 | 0.74-0.93 | 0.70 |
| missile | 3989 | 20 | 31.8 | 31.8 | 1.00 | 0.86-1.17 | 1.03 | 0.87-1.24 | -0.01 |
| infantry melee | 29215 | 20 | 315.8 | 313.9 | 0.99 | 0.90-1.10 | 1.01 | 0.91-1.11 | 0.30 |
| network's side | 16523 | 20 | 184.7 | 169.3 | 0.92 | 0.86-0.98 | 0.92 | 0.86-1.00 | 2.48 |
| the other side | 19331 | 20 | 191.9 | 201.1 | 1.05 | 0.92-1.19 | 1.08 | 0.94-1.22 | -1.48 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 721 | 20 | 20.2 | 16.6 | 0.82 | 0.75-0.91 | 0.79 | 0.69-0.90 | 0.57 |
| lord, routing | 236 | 15 | 0.9 | 0.8 | 0.85 | 0.52-1.47 | 1.31 | 0.46-4.70 | 0.02 |
| missile, melee | 413 | 18 | 15.9 | 16.7 | 1.05 | 0.93-1.18 | 1.02 | 0.89-1.19 | -0.13 |
| missile, routing | 405 | 16 | 2.2 | 4.5 | 2.03 | 1.11-3.24 | 2.35 | 1.09-4.08 | -0.37 |
| infantry melee, melee | 8824 | 20 | 233.2 | 221.7 | 0.95 | 0.88-1.02 | 0.99 | 0.92-1.07 | 1.86 |
| infantry melee, routing | 6408 | 20 | 16.5 | 24.9 | 1.51 | 1.21-1.82 | 1.38 | 1.06-1.76 | -1.35 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 18372 | 20 | 6 | 14.0 | 2.33 | 0.72-10.00 |
| steady (3) | 3212 | 20 | 12 | 43.2 | 3.60 | 1.85-7.03 |
| shaken (4) | 4549 | 20 | 153 | 283.5 | 1.85 | 1.39-2.34 |
| wavering (5) | 2661 | 20 | 503 | 603.5 | 1.20 | 1.04-1.33 |
| mp < 0.25 | 2598 | 20 | 510 | 613.2 | 1.20 | 1.03-1.37 |
| mp 0.25-0.5 | 3967 | 20 | 140 | 255.5 | 1.82 | 1.48-2.17 |
| mp 0.5-0.75 | 4370 | 20 | 27 | 71.8 | 2.66 | 1.51-4.40 |
| mp >= 0.75 | 17870 | 20 | 6 | 9.8 | 1.62 | 0.56-7.17 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 28805 | 20 | 683 | 950.2 | 1.39 | 1.18-1.58 |
| standing, not in melee, no contact in window | 16145 | 20 | 113 | 184.8 | 1.63 | 1.17-1.98 |
| standing, shot at (an enemy shooting it) | 509 | 20 | 47 | 61.8 | 1.31 | 1.08-1.77 |
| contact begins in window, ran in | 2307 | 20 | 42 | 85.5 | 2.04 | 1.40-2.66 |
| contact begins in window, walked in | 56 | 15 | 4 | 3.5 | 0.88 | 0.00-4.00 |
| melee, first 10 s of the fight | 1535 | 20 | 88 | 102.8 | 1.17 | 1.01-1.35 |
| melee, 10-30 s | 2047 | 20 | 99 | 135.5 | 1.37 | 1.11-1.67 |
| melee, 30 s+ | 6376 | 20 | 297 | 390.5 | 1.31 | 1.11-1.55 |
| melee, sim touches 1 enemy | 4233 | 20 | 204 | 290.5 | 1.42 | 1.14-1.78 |
| melee, sim touches 2+ enemies | 4241 | 20 | 239 | 306.2 | 1.28 | 1.12-1.43 |
| melee, sim touches nobody (game flag on) | 1484 | 20 | 41 | 32.0 | 0.78 | 0.59-1.13 |
| melee, touched in front | 4093 | 20 | 160 | 193.8 | 1.21 | 1.04-1.40 |
| melee, touched in flank | 3346 | 20 | 183 | 253.5 | 1.39 | 1.13-1.63 |
| melee, touched in rear | 1035 | 20 | 100 | 149.5 | 1.50 | 1.22-1.83 |
| melee, health < 30 % | 1132 | 20 | 225 | 287.8 | 1.28 | 1.08-1.47 |
| melee, health >= 30 % | 8826 | 20 | 259 | 341.0 | 1.32 | 1.09-1.55 |
| standing, health < 30 % | 4614 | 20 | 381 | 530.5 | 1.39 | 1.16-1.60 |
| lord | 2414 | 20 | 34 | 46.8 | 1.38 | 1.01-1.96 |
| missile | 3584 | 20 | 50 | 66.5 | 1.33 | 0.99-1.73 |
| infantry melee | 22807 | 20 | 599 | 837.0 | 1.40 | 1.18-1.62 |
| network's side | 13157 | 20 | 356 | 413.0 | 1.16 | 1.01-1.33 |
| the other side | 15648 | 20 | 327 | 537.2 | 1.64 | 1.30-1.99 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 7049 | 20 | 371 | 255.8 | 0.69 | 0.62-0.75 |
| routers in melee | 830 | 20 | 36 | 33.0 | 0.92 | 0.74-1.12 |
| routers free | 6219 | 20 | 335 | 222.8 | 0.66 | 0.59-0.72 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 28122 | -0.013 | -0.038 | -0.024 | -0.029 - -0.018 |
| standing, not in melee, no contact in window | 16032 | +0.008 | -0.021 | -0.030 | -0.035 - -0.025 |
| standing, shot at (an enemy shooting it) | 462 | -0.067 | -0.119 | -0.053 | -0.090 - -0.013 |
| contact begins in window, ran in | 2265 | -0.034 | -0.062 | -0.028 | -0.037 - -0.019 |
| contact begins in window, walked in | 52 | +0.013 | +0.036 | +0.023 | -0.022 - +0.067 |
| melee, first 10 s of the fight | 1447 | -0.112 | -0.130 | -0.018 | -0.035 - -0.002 |
| melee, 10-30 s | 1948 | -0.050 | -0.058 | -0.008 | -0.022 - +0.008 |
| melee, 30 s+ | 6079 | -0.026 | -0.039 | -0.013 | -0.023 - -0.002 |
| melee, sim touches 1 enemy | 4029 | -0.045 | -0.080 | -0.034 | -0.046 - -0.022 |
| melee, sim touches 2+ enemies | 4002 | -0.050 | -0.055 | -0.005 | -0.018 - +0.009 |
| melee, sim touches nobody (game flag on) | 1443 | -0.025 | +0.002 | +0.027 | +0.010 - +0.043 |
| melee, touched in front | 3933 | -0.041 | -0.057 | -0.016 | -0.028 - -0.004 |
| melee, touched in flank | 3163 | -0.052 | -0.076 | -0.024 | -0.037 - -0.011 |
| melee, touched in rear | 935 | -0.057 | -0.080 | -0.023 | -0.046 - +0.001 |
| melee, health < 30 % | 907 | -0.034 | -0.076 | -0.043 | -0.059 - -0.026 |
| melee, health >= 30 % | 8567 | -0.045 | -0.055 | -0.010 | -0.020 - +0.002 |
| standing, health < 30 % | 4233 | -0.009 | -0.047 | -0.037 | -0.049 - -0.028 |
| eager / confident (ms 1-2) | 18366 | -0.019 | -0.049 | -0.029 | -0.034 - -0.025 |
| steady (3) | 3200 | -0.013 | -0.026 | -0.012 | -0.027 - +0.003 |
| shaken (4) | 4396 | -0.010 | -0.027 | -0.017 | -0.028 - +0.002 |
| wavering (5) | 2158 | +0.031 | +0.019 | -0.012 | -0.023 - +0.004 |
| mp < 0.25 | 2088 | +0.027 | +0.013 | -0.014 | -0.026 - +0.007 |
| mp 0.25-0.5 | 3827 | -0.005 | -0.016 | -0.011 | -0.023 - +0.010 |
| mp 0.5-0.75 | 4343 | -0.015 | -0.028 | -0.013 | -0.027 - +0.001 |
| mp >= 0.75 | 17864 | -0.020 | -0.050 | -0.031 | -0.036 - -0.026 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 155 | 16 | 12355 | 12178 | 0.99 | 0.62-1.37 |
| hold, lord | 59 | 18 | 13200 | 7800 | 0.59 | 0.38-0.82 |
| move, formation | 450 | 17 | 31949 | 10453 | 0.33 | 0.25-0.42 |
| move, missile unit | 148 | 14 | 6065 | 744 | 0.12 | 0.05-0.28 |
| move, lord | 41 | 13 | 4562 | 763 | 0.17 | 0.03-0.35 |
| attack, formation | 8219 | 20 | 1923931 | 1503907 | 0.78 | 0.73-0.83 |
| attack, missile unit | 236 | 18 | 39992 | 26985 | 0.67 | 0.54-0.85 |
| attack, lord | 621 | 20 | 133139 | 106938 | 0.80 | 0.64-0.98 |

## ../../../build/formdepth/sr_pos_md28: other battles

Windows of 10 s; 72 battles, 88270 unit-windows. HP lost sim/game 0.93 (0.91-0.96); rout starts sim/game 1.25 (1.18-1.33), game 2008, sim 2514.5.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.88 | 0.86 | 0.85 | 0.85 | 0.86 | 0.87 | 0.87 | 0.88 | 0.88 | 0.89 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 88270 | 72 | 1254.8 | 1167.0 | 0.93 | 0.91-0.96 | 0.97 | 0.95-1.00 | 1.00 |
| standing, not in melee, no contact in window | 35276 | 72 | 59.5 | 69.1 | 1.16 | 1.09-1.23 | 1.23 | 1.14-1.33 | -0.11 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 171.3 | 152.1 | 0.89 | 0.86-0.91 | 1.03 | 0.98-1.08 | 0.22 |
| contact begins in window, ran in | 6699 | 72 | 154.0 | 145.3 | 0.94 | 0.91-0.98 | 0.93 | 0.89-0.96 | 0.10 |
| contact begins in window, walked in | 271 | 68 | 7.7 | 6.5 | 0.84 | 0.73-0.96 | 0.78 | 0.67-0.93 | 0.01 |
| melee, first 10 s of the fight | 4204 | 72 | 146.2 | 125.8 | 0.86 | 0.83-0.89 | 0.89 | 0.85-0.93 | 0.23 |
| melee, 10-30 s | 6160 | 72 | 208.1 | 171.6 | 0.82 | 0.80-0.85 | 0.85 | 0.82-0.89 | 0.42 |
| melee, 30 s+ | 19508 | 72 | 495.4 | 458.9 | 0.93 | 0.89-0.97 | 0.96 | 0.93-1.01 | 0.42 |
| melee, sim touches 1 enemy | 10422 | 72 | 305.7 | 245.2 | 0.80 | 0.78-0.82 | 0.83 | 0.81-0.86 | 0.69 |
| melee, sim touches 2+ enemies | 15733 | 72 | 463.8 | 465.7 | 1.00 | 0.97-1.05 | 1.03 | 0.99-1.07 | -0.02 |
| melee, sim touches nobody (game flag on) | 3717 | 72 | 80.2 | 45.4 | 0.57 | 0.53-0.61 | 0.66 | 0.61-0.72 | 0.40 |
| melee, touched in front | 8938 | 72 | 222.4 | 193.2 | 0.87 | 0.84-0.91 | 0.91 | 0.88-0.96 | 0.33 |
| melee, touched in flank | 12402 | 72 | 376.8 | 359.7 | 0.95 | 0.92-0.99 | 0.97 | 0.94-1.01 | 0.20 |
| melee, touched in rear | 4815 | 72 | 170.3 | 158.0 | 0.93 | 0.88-0.98 | 0.94 | 0.90-1.00 | 0.14 |
| melee, health < 30 % | 3088 | 72 | 63.6 | 69.7 | 1.10 | 1.04-1.16 | 1.19 | 1.11-1.28 | -0.07 |
| melee, health >= 30 % | 26784 | 72 | 786.1 | 686.6 | 0.87 | 0.85-0.91 | 0.90 | 0.87-0.94 | 1.13 |
| routing, in melee (pursued) | 3146 | 72 | 49.2 | 59.4 | 1.21 | 1.06-1.38 | 1.19 | 1.04-1.37 | -0.12 |
| routing, free | 10755 | 72 | 22.6 | 34.3 | 1.51 | 1.36-1.69 | 1.55 | 1.35-1.81 | -0.13 |
| standing, health < 30 % | 8705 | 72 | 85.9 | 89.6 | 1.04 | 1.00-1.10 | 1.13 | 1.06-1.20 | -0.04 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 6593 | 72 | 81.6 | 89.0 | 1.09 | 1.01-1.17 | 1.10 | 1.02-1.19 | -0.08 |
| missile | 24324 | 72 | 316.4 | 275.1 | 0.87 | 0.84-0.91 | 0.90 | 0.86-0.95 | 0.47 |
| infantry melee | 57353 | 72 | 856.7 | 802.9 | 0.94 | 0.91-0.97 | 0.98 | 0.96-1.02 | 0.61 |
| network's side | 43836 | 72 | 729.1 | 684.9 | 0.94 | 0.90-0.97 | 0.97 | 0.94-1.01 | 0.50 |
| the other side | 44434 | 72 | 525.7 | 482.2 | 0.92 | 0.88-0.96 | 0.97 | 0.92-1.01 | 0.50 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 2376 | 72 | 57.6 | 61.9 | 1.07 | 1.00-1.15 | 1.10 | 1.01-1.19 | -0.05 |
| lord, routing | 511 | 54 | 1.6 | 2.1 | 1.32 | 1.02-1.71 | 1.59 | 1.19-2.12 | -0.01 |
| missile, melee | 4055 | 72 | 184.5 | 152.8 | 0.83 | 0.79-0.87 | 0.86 | 0.81-0.90 | 0.36 |
| missile, routing | 3025 | 72 | 18.3 | 24.9 | 1.36 | 1.11-1.67 | 1.35 | 1.09-1.71 | -0.07 |
| infantry melee, melee | 23441 | 72 | 607.5 | 541.6 | 0.89 | 0.86-0.93 | 0.93 | 0.90-0.97 | 0.75 |
| infantry melee, routing | 10365 | 72 | 52.0 | 66.7 | 1.28 | 1.13-1.48 | 1.29 | 1.12-1.52 | -0.17 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 50168 | 72 | 15 | 86.2 | 5.75 | 3.37-11.66 |
| steady (3) | 7127 | 72 | 28 | 77.2 | 2.76 | 1.85-4.21 |
| shaken (4) | 11458 | 72 | 477 | 828.0 | 1.74 | 1.56-1.94 |
| wavering (5) | 5568 | 72 | 1453 | 1503.8 | 1.03 | 0.98-1.10 |
| mp < 0.25 | 5043 | 72 | 1480 | 1472.5 | 0.99 | 0.94-1.05 |
| mp 0.25-0.5 | 8961 | 72 | 441 | 712.8 | 1.62 | 1.44-1.81 |
| mp 0.5-0.75 | 11457 | 72 | 67 | 239.2 | 3.57 | 2.90-4.56 |
| mp >= 0.75 | 48908 | 72 | 20 | 90.0 | 4.50 | 2.72-8.12 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 74369 | 72 | 2008 | 2514.5 | 1.25 | 1.18-1.33 |
| standing, not in melee, no contact in window | 35276 | 72 | 268 | 354.5 | 1.32 | 1.10-1.55 |
| standing, shot at (an enemy shooting it) | 3476 | 72 | 191 | 248.0 | 1.30 | 1.17-1.44 |
| contact begins in window, ran in | 6699 | 72 | 124 | 201.2 | 1.62 | 1.41-1.92 |
| contact begins in window, walked in | 271 | 68 | 6 | 10.2 | 1.71 | 1.19-3.50 |
| melee, first 10 s of the fight | 4204 | 72 | 163 | 231.8 | 1.42 | 1.28-1.60 |
| melee, 10-30 s | 6160 | 72 | 304 | 367.5 | 1.21 | 1.09-1.32 |
| melee, 30 s+ | 19508 | 72 | 983 | 1157.8 | 1.18 | 1.08-1.28 |
| melee, sim touches 1 enemy | 10422 | 72 | 541 | 691.2 | 1.28 | 1.19-1.37 |
| melee, sim touches 2+ enemies | 15733 | 72 | 788 | 983.2 | 1.25 | 1.13-1.37 |
| melee, sim touches nobody (game flag on) | 3717 | 72 | 121 | 82.5 | 0.68 | 0.52-0.87 |
| melee, touched in front | 8938 | 72 | 329 | 369.8 | 1.12 | 1.02-1.25 |
| melee, touched in flank | 12402 | 72 | 635 | 807.0 | 1.27 | 1.16-1.37 |
| melee, touched in rear | 4815 | 72 | 365 | 497.8 | 1.36 | 1.20-1.55 |
| melee, health < 30 % | 3088 | 72 | 618 | 714.2 | 1.16 | 1.09-1.23 |
| melee, health >= 30 % | 26784 | 72 | 832 | 1042.8 | 1.25 | 1.13-1.38 |
| standing, health < 30 % | 8705 | 72 | 1028 | 1193.8 | 1.16 | 1.08-1.24 |
| lord | 6082 | 72 | 100 | 89.0 | 0.89 | 0.69-1.11 |
| missile | 21299 | 72 | 541 | 754.5 | 1.39 | 1.29-1.51 |
| infantry melee | 46988 | 72 | 1367 | 1671.0 | 1.22 | 1.14-1.31 |
| network's side | 34608 | 72 | 1389 | 1731.0 | 1.25 | 1.16-1.34 |
| the other side | 39761 | 72 | 619 | 783.5 | 1.27 | 1.15-1.40 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 13901 | 72 | 1170 | 1085.8 | 0.93 | 0.88-0.98 |
| routers in melee | 3146 | 72 | 90 | 131.8 | 1.46 | 1.22-1.79 |
| routers free | 10755 | 72 | 1080 | 954.0 | 0.88 | 0.83-0.94 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 72361 | -0.017 | -0.049 | -0.032 | -0.036 - -0.029 |
| standing, not in melee, no contact in window | 35008 | +0.014 | -0.037 | -0.051 | -0.057 - -0.046 |
| standing, shot at (an enemy shooting it) | 3285 | -0.082 | -0.137 | -0.055 | -0.063 - -0.046 |
| contact begins in window, ran in | 6575 | -0.050 | -0.112 | -0.062 | -0.071 - -0.053 |
| contact begins in window, walked in | 265 | -0.065 | -0.105 | -0.040 | -0.069 - -0.017 |
| melee, first 10 s of the fight | 4041 | -0.111 | -0.157 | -0.045 | -0.053 - -0.038 |
| melee, 10-30 s | 5856 | -0.058 | -0.080 | -0.023 | -0.030 - -0.016 |
| melee, 30 s+ | 18525 | -0.023 | -0.006 | +0.016 | +0.010 - +0.023 |
| melee, sim touches 1 enemy | 9881 | -0.046 | -0.064 | -0.018 | -0.026 - -0.009 |
| melee, sim touches 2+ enemies | 14945 | -0.044 | -0.040 | +0.004 | -0.002 - +0.011 |
| melee, sim touches nobody (game flag on) | 3596 | -0.025 | +0.003 | +0.027 | +0.016 - +0.037 |
| melee, touched in front | 8609 | -0.039 | -0.029 | +0.009 | +0.000 - +0.017 |
| melee, touched in flank | 11767 | -0.047 | -0.056 | -0.009 | -0.016 - -0.002 |
| melee, touched in rear | 4450 | -0.053 | -0.071 | -0.019 | -0.032 - -0.007 |
| melee, health < 30 % | 2470 | -0.036 | -0.055 | -0.019 | -0.032 - -0.006 |
| melee, health >= 30 % | 25952 | -0.043 | -0.042 | +0.001 | -0.004 - +0.007 |
| standing, health < 30 % | 7677 | -0.013 | -0.064 | -0.052 | -0.066 - -0.038 |
| eager / confident (ms 1-2) | 50153 | -0.022 | -0.065 | -0.043 | -0.047 - -0.040 |
| steady (3) | 7099 | -0.025 | -0.022 | +0.004 | -0.004 - +0.011 |
| shaken (4) | 10981 | -0.011 | -0.024 | -0.013 | -0.021 - -0.005 |
| wavering (5) | 4115 | +0.042 | +0.029 | -0.013 | -0.031 - +0.003 |
| mp < 0.25 | 3563 | +0.040 | +0.038 | -0.002 | -0.022 - +0.016 |
| mp 0.25-0.5 | 8520 | -0.001 | -0.001 | -0.000 | -0.007 - +0.006 |
| mp 0.5-0.75 | 11390 | -0.024 | -0.033 | -0.009 | -0.017 - -0.002 |
| mp >= 0.75 | 48888 | -0.022 | -0.068 | -0.046 | -0.049 - -0.042 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 734 | 70 | 118859 | 42563 | 0.36 | 0.31-0.41 |
| hold, missile unit | 1416 | 72 | 213822 | 119616 | 0.56 | 0.49-0.62 |
| hold, lord | 537 | 69 | 145512 | 68771 | 0.47 | 0.41-0.54 |
| move, formation | 1035 | 71 | 151893 | 44920 | 0.30 | 0.25-0.35 |
| move, missile unit | 751 | 69 | 78425 | 20460 | 0.26 | 0.20-0.35 |
| move, lord | 49 | 32 | 7295 | 3359 | 0.46 | 0.27-0.71 |
| attack, formation | 21672 | 72 | 5277048 | 3326316 | 0.63 | 0.60-0.66 |
| attack, missile unit | 1888 | 71 | 332739 | 209230 | 0.63 | 0.57-0.69 |
| attack, lord | 1790 | 72 | 422482 | 285796 | 0.68 | 0.61-0.74 |
